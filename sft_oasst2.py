# Copyright 2022 MosaicML Examples authors
# SPDX-License-Identifier: Apache-2.0

"""
Supervised Fine-Tuning (SFT) script for OASST2 dataset using ModernBERT models.

This script performs instruction-following fine-tuning on the OpenAssistant OASST2 dataset,
following the patterns established in the main.py training script.
"""

import os
import sys
import warnings
from pathlib import Path
from typing import Optional, cast
import argparse

# Add folder root to path to allow us to use relative imports regardless of what directory the script is run from
sys.path.append(os.path.dirname(os.path.realpath(__file__)))

import torch
from torch import nn
import torch._dynamo
torch._dynamo.config.suppress_errors = True

import datasets
import transformers
from transformers import AutoTokenizer

from composer import Evaluator, Trainer, algorithms
from composer.callbacks import LRMonitor, MemoryMonitor, OptimizerMonitor, RuntimeEstimator, SpeedMonitor
from composer.core import DataSpec
from composer.core.types import Dataset
from composer.loggers import WandBLogger
from composer.optim import DecoupledAdamW
from composer.optim.scheduler import (
    ConstantWithWarmupScheduler,
    CosineAnnealingWithWarmupScheduler,
    LinearWithWarmupScheduler,
)
from composer.utils import dist, reproducibility
from omegaconf import DictConfig, OmegaConf
from omegaconf import OmegaConf as om
from torch.optim import AdamW
from torch.utils.data import DataLoader

import src.flex_bert as flex_bert_module
import src.hf_bert as hf_bert_module
import src.mosaic_bert as mosaic_bert_module
from src.callbacks.dataloader_speed import DataloaderSpeedMonitor
from src.callbacks.log_grad_norm import LogGradNorm
from src.callbacks.packing_efficiency import PackingEfficency
from src.callbacks.scheduled_gc import ScheduledGarbageCollector
from src.scheduler import CosineInverseSqrtScheduler, OneMinusSqrtScheduler, WarmupStableDecayScheduler


def create_oasst2_sft_dataset(
    tokenizer_name: str,
    split: str = "train",
    max_seq_length: int = 512,
    max_retries: int = 10,
    num_workers: int = 0,
    instruction_template: str = "### Human: {human}\n\n### Assistant: {assistant}",
):
    """
    Create OASST2 dataset for supervised fine-tuning.
    
    Args:
        tokenizer_name: Name of the tokenizer to use
        split: Dataset split to use (train, validation)
        max_seq_length: Maximum sequence length
        max_retries: Maximum retries for dataset loading
        num_workers: Number of workers for preprocessing
        instruction_template: Template for formatting conversations
    """
    
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    
    # Add special tokens if needed
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f"Loading OASST2 dataset on rank {dist.get_global_rank()}")
    download_config = datasets.DownloadConfig(max_retries=max_retries)
    
    # Load OASST2 dataset
    dataset = datasets.load_dataset(
        "OpenAssistant/oasst2",
        split=split,
        download_config=download_config
    )
    
    def process_conversations(examples):
        """
        Process OASST2 data into instruction-following format.
        OASST2 contains conversation trees - we need to extract meaningful assistant/human pairs.
        """
        processed_texts = []
        
        for i in range(len(examples['message_id'])):
            # Only process assistant messages (these are the targets we want to train on)
            if examples['role'][i] == 'assistant':
                # Get the parent message (should be human)
                parent_id = examples['parent_id'][i]
                
                # Find the parent message in the current batch (simplified approach)
                # In a full implementation, you'd want to build the conversation tree properly
                human_text = "Please help me with this request."  # Default fallback
                
                # Look for parent in current batch
                for j in range(len(examples['message_id'])):
                    if examples['message_id'][j] == parent_id and examples['role'][j] == 'prompter':
                        human_text = examples['text'][j]
                        break
                
                assistant_text = examples['text'][i]
                
                # Format as instruction-following conversation
                formatted_text = instruction_template.format(
                    human=human_text.strip(),
                    assistant=assistant_text.strip()
                )
                
                processed_texts.append(formatted_text)
        
        return processed_texts
    
    def tokenize_function(examples):
        """Tokenize the processed conversations."""
        # Process conversations first
        texts = process_conversations(examples)
        
        if not texts:  # Skip if no valid conversations in this batch
            return {"input_ids": [], "attention_mask": [], "labels": []}
        
        # Tokenize
        tokenized = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=max_seq_length,
            return_tensors=None
        )
        
        # For SFT, labels are the same as input_ids (we predict the next token)
        # But we typically mask the human part and only train on assistant responses
        labels = []
        for i, input_ids in enumerate(tokenized["input_ids"]):
            label = input_ids.copy()
            
            # Find where "### Assistant:" starts to only train on assistant responses
            assistant_start = None
            assistant_token_ids = tokenizer.encode("### Assistant:", add_special_tokens=False)
            
            # Simple approach: find the assistant marker
            for j in range(len(input_ids) - len(assistant_token_ids)):
                if input_ids[j:j+len(assistant_token_ids)] == assistant_token_ids:
                    assistant_start = j + len(assistant_token_ids)
                    break
            
            if assistant_start is not None:
                # Mask everything before assistant response
                label[:assistant_start] = [-100] * assistant_start
            
            labels.append(label)
        
        tokenized["labels"] = labels
        return tokenized
    
    print(f"Starting tokenization by preprocessing over {num_workers} threads!")
    
    # Filter to only include valid conversation messages
    dataset = dataset.filter(lambda x: x['role'] in ['assistant', 'prompter'])
    
    # Group by conversation and process
    dataset = dataset.map(
        tokenize_function,
        batched=True,
        num_proc=None if num_workers == 0 else num_workers,
        batch_size=100,
        remove_columns=dataset.column_names,
        load_from_cache_file=True,
    )
    
    # Filter out empty examples
    dataset = dataset.filter(lambda x: len(x['input_ids']) > 0)
    
    return dataset


def build_sft_dataloader(
    cfg,
    tokenizer,
    device_batch_size,
    device_microbatch_size: int | None = None,
):
    """Build dataloader for SFT training."""
    
    dataset = create_oasst2_sft_dataset(
        tokenizer_name=cfg.tokenizer_name,
        split=cfg.split,
        max_seq_length=cfg.max_seq_len,
        num_workers=cfg.get("num_workers", 0),
    )
    
    # Use standard data collator for language modeling
    collator = transformers.DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # We're doing causal LM, not masked LM
        pad_to_multiple_of=8,
    )
    
    dataset = cast(Dataset, dataset)
    data_loader = DataLoader(
        dataset,
        collate_fn=collator,
        batch_size=device_batch_size,
        sampler=dist.get_sampler(dataset, drop_last=cfg.get("drop_last", True), shuffle=cfg.get("shuffle", True)),
        num_workers=cfg.get("num_workers", 0),
        pin_memory=cfg.get("pin_memory", True),
        prefetch_factor=cfg.get("prefetch_factor", 2),
        persistent_workers=cfg.get("persistent_workers", True),
        timeout=cfg.get("timeout", 0),
    )
    
    return data_loader


def build_model(cfg: DictConfig):
    """Build model for SFT training."""
    if cfg.name == "hf_bert":
        return hf_bert_module.create_hf_bert_mlm(
            pretrained_model_name=cfg.pretrained_model_name,
            use_pretrained=cfg.get("use_pretrained", None),
            model_config=cfg.get("model_config", None),
            tokenizer_name=cfg.get("tokenizer_name", None),
            gradient_checkpointing=cfg.get("gradient_checkpointing", None),
        )
    elif cfg.name == "flex_bert":
        return flex_bert_module.create_flex_bert_mlm(
            pretrained_model_name=cfg.pretrained_model_name,
            pretrained_checkpoint=cfg.get("pretrained_checkpoint", None),
            model_config=cfg.get("model_config", None),
            tokenizer_name=cfg.get("tokenizer_name", None),
            gradient_checkpointing=cfg.get("gradient_checkpointing", None),
            recompute_metric_loss=cfg.get("recompute_metric_loss", False),
            disable_train_metrics=cfg.get("disable_train_metrics", False),
        )
    elif cfg.name == "modern_bert":
        return flex_bert_module.create_modern_bert_mlm(
            pretrained_model_name=cfg.pretrained_model_name,
            pretrained_checkpoint=cfg.get("pretrained_checkpoint", None),
            model_config=cfg.get("model_config", None),
            tokenizer_name=cfg.get("tokenizer_name", None),
            gradient_checkpointing=cfg.get("gradient_checkpointing", None),
            recompute_metric_loss=cfg.get("recompute_metric_loss", False),
            disable_train_metrics=cfg.get("disable_train_metrics", False),
            use_dora=cfg.get("use_dora", False),
            mixed_mlm=cfg.get("mixed_mlm", False),
            checkpoint_dict=cfg.get("checkpoint_dict", None),
        )
    else:
        raise ValueError(f"Not sure how to build model with name={cfg.name}")


def build_optimizer(cfg, model):
    """Build optimizer following the main.py pattern."""
    from main import param_groups_weight_decay
    
    if cfg.get("filter_bias_norm_wd", False):
        params = param_groups_weight_decay(model, weight_decay=cfg.weight_decay)
    else:
        params = model.parameters()

    if cfg.name == "decoupled_adamw":
        return DecoupledAdamW(params, lr=cfg.lr, betas=list(cfg.betas), eps=cfg.eps, weight_decay=cfg.weight_decay)
    elif cfg.name == "adamw":
        return AdamW(params, lr=cfg.lr, betas=list(cfg.betas), eps=cfg.eps, weight_decay=cfg.weight_decay)
    else:
        raise ValueError(f"Not sure how to build optimizer: {cfg.name}")


def build_scheduler(cfg):
    """Build scheduler following the main.py pattern."""
    if cfg.name == "constant_with_warmup":
        return ConstantWithWarmupScheduler(t_warmup=cfg.t_warmup)
    elif cfg.name == "cosine_with_warmup":
        return CosineAnnealingWithWarmupScheduler(t_warmup=cfg.t_warmup, alpha_f=cfg.alpha_f)
    elif cfg.name == "linear_decay_with_warmup":
        return LinearWithWarmupScheduler(t_warmup=cfg.t_warmup, alpha_f=cfg.alpha_f)
    else:
        raise ValueError(f"Not sure how to build scheduler: {cfg.name}")


def update_batch_size_info(cfg: DictConfig):
    """Update batch size info following the main.py pattern."""
    from main import update_batch_size_info as update_batch_size_info_main
    return update_batch_size_info_main(cfg)


def build_callback(name, kwargs):
    """Build callbacks following the main.py pattern."""
    from main import build_callback as build_callback_main
    return build_callback_main(name, kwargs)


def build_logger(name, kwargs):
    """Build loggers following the main.py pattern."""
    from main import build_logger as build_logger_main  
    return build_logger_main(name, kwargs)


def main(cfg: DictConfig) -> None:
    """Main SFT training function."""
    print("SFT Training using config: ")
    print(om.to_yaml(cfg))
    reproducibility.seed_all(cfg.seed)

    # Get batch size info
    cfg = update_batch_size_info(cfg)

    # Build Model
    print("Initializing model...")
    model = build_model(cfg.model)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"{n_params=:.4e}")

    # Build dataloaders
    print("Building train loader...")
    train_loader = build_sft_dataloader(
        cfg=cfg.train_loader,
        tokenizer=model.tokenizer,
        device_batch_size=cfg.global_train_batch_size // dist.get_world_size(),
        device_microbatch_size=cfg.device_train_microbatch_size,
    )

    # Optional eval loader
    eval_evaluator = None
    if cfg.get("eval_loader", None) is not None:
        print("Building eval loader...")
        eval_loader = build_sft_dataloader(
            cfg=cfg.eval_loader,
            tokenizer=model.tokenizer,
            device_batch_size=cfg.get("global_eval_batch_size", cfg.global_train_batch_size) // dist.get_world_size(),
        )
        eval_evaluator = Evaluator(
            label="eval",
            dataloader=eval_loader,
            device_eval_microbatch_size=cfg.get("device_eval_microbatch_size", None),
        )

    # Optimizer
    optimizer = build_optimizer(cfg.optimizer, model)

    # Scheduler
    scheduler = build_scheduler(cfg.scheduler)

    # Loggers
    loggers = [build_logger(name, logger_cfg) for name, logger_cfg in cfg.get("loggers", {}).items()]

    # Callbacks
    callbacks = [build_callback(name, callback_cfg) for name, callback_cfg in cfg.get("callbacks", {}).items()]

    if cfg.get("run_name") is None:
        cfg.run_name = os.environ.get("COMPOSER_RUN_NAME", "sft_oasst2")

    # Build the Trainer
    trainer = Trainer(
        run_name=cfg.run_name,
        seed=cfg.seed,
        model=model,
        train_dataloader=train_loader,
        eval_dataloader=eval_evaluator,
        train_subset_num_batches=cfg.get("train_subset_num_batches", -1),
        eval_subset_num_batches=cfg.get("eval_subset_num_batches", -1),
        optimizers=optimizer,
        schedulers=scheduler,
        max_duration=cfg.max_duration,
        eval_interval=cfg.get("eval_interval", "500ba"),
        progress_bar=cfg.get("progress_bar", True),
        log_to_console=cfg.get("log_to_console", True),
        console_log_interval=cfg.get("console_log_interval", "1ba"),
        loggers=loggers,
        callbacks=callbacks,
        precision=cfg.get("precision", "amp_fp16"),
        device=cfg.get("device"),
        device_train_microbatch_size=cfg.get("device_train_microbatch_size", "auto"),
        save_folder=cfg.get("save_folder"),
        save_interval=cfg.get("save_interval", "1000ba"),
        save_num_checkpoints_to_keep=cfg.get("save_num_checkpoints_to_keep", -1),
        save_overwrite=cfg.get("save_overwrite", False),
        load_path=cfg.get("load_path"),
        load_weights_only=True,
        autoresume=False,
    )

    print("Starting SFT training...")
    trainer.fit()

    # Save final model
    if cfg.get("save_final_model", False):
        model_name = cfg.get("final_model_name", "sft_oasst2_model")
        print(f"Saving final model as {model_name}")
        model.model.push_to_hub(model_name)

    print("SFT training completed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SFT training on OASST2 dataset")
    parser.add_argument("config", type=str, help="Path to config YAML file")
    parser.add_argument("--overrides", nargs="*", default=[], help="Config overrides")
    
    args = parser.parse_args()
    
    # Load config
    with open("yamls/defaults.yaml") as f:
        default_cfg = om.load(f)
    with open(args.config) as f:
        yaml_cfg = om.load(f)
    
    cli_cfg = om.from_cli(args.overrides)
    cfg = om.merge(default_cfg, yaml_cfg, cli_cfg)
    cfg = cast(DictConfig, cfg)
    
    main(cfg)