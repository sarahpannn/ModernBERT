# Copyright 2022 MosaicML Examples authors
# SPDX-License-Identifier: Apache-2.0

import os
import sys
import warnings
from pathlib import Path
from typing import Optional, cast
import argparse

original_dir = os.getcwd()

print(original_dir)

import torch
from torch import nn

import torch._dynamo
torch._dynamo.config.suppress_errors = True

post_import = os.getcwd()
print(post_import)

os.chdir(original_dir)

from src.bert_layers.configuration_bert import FlexBertConfig
from src.bert_layers.model import init_mlm_model_from_pretrained

import peft
from peft import LoraConfig

import datasets

# Add folder root to path to allow us to use relative imports regardless of what directory the script is run from
sys.path.append(os.path.dirname(os.path.realpath(__file__)))

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
from composer.utils.checkpoint import _ensure_valid_checkpoint
from omegaconf import DictConfig, OmegaConf
from omegaconf import OmegaConf as om
from torch.optim import AdamW
from torch.utils.data import DataLoader

import transformers

import src.evals.data as data_module
import src.flex_bert as flex_bert_module
import src.hf_bert as hf_bert_module
import src.mosaic_bert as mosaic_bert_module
import src.text_data as text_data_module
from src.callbacks.dataloader_speed import DataloaderSpeedMonitor
from src.callbacks.log_grad_norm import LogGradNorm
from src.callbacks.packing_efficiency import PackingEfficency
from src.callbacks.scheduled_gc import ScheduledGarbageCollector
from src.scheduler import CosineInverseSqrtScheduler, OneMinusSqrtScheduler, WarmupStableDecayScheduler
from src.sequence_packer import get_num_samples_in_packed_batch, split_packed_batch


def update_batch_size_info(cfg: DictConfig):
    global_batch_size, device_microbatch_size = cfg.global_train_batch_size, cfg.device_train_microbatch_size
    if global_batch_size % dist.get_world_size() != 0:
        raise ValueError(
            f"Global batch size {global_batch_size} is not divisible by {dist.get_world_size()} "
            "as a result, the batch size would be truncated, please adjust `global_batch_size` "
            f"to be divisible by world size, {dist.get_world_size()}."
        )
    device_train_batch_size = global_batch_size // dist.get_world_size()
    if isinstance(device_microbatch_size, int):
        if device_microbatch_size > device_train_batch_size:
            print(
                f"WARNING: device_train_microbatch_size > device_train_batch_size, "
                f"will be reduced from {device_microbatch_size} -> {device_train_batch_size}."
            )
            device_microbatch_size = device_train_batch_size
    cfg.n_gpus = dist.get_world_size()
    cfg.device_train_batch_size = device_train_batch_size
    cfg.device_train_microbatch_size = device_microbatch_size

    # Safely set `device_eval_microbatch_size` if not provided by user
    if "device_eval_microbatch_size" not in cfg:
        if cfg.device_train_microbatch_size == "auto":
            cfg.device_eval_microbatch_size = 1
        else:
            cfg.device_eval_microbatch_size = cfg.device_train_microbatch_size

    global_eval_batch_size, device_eval_microbatch_size = (
        cfg.get("global_eval_batch_size", global_batch_size),
        cfg.device_eval_microbatch_size,
    )
    device_eval_batch_size = global_eval_batch_size // dist.get_world_size()
    if isinstance(device_eval_microbatch_size, int):
        if device_eval_microbatch_size > device_eval_microbatch_size:
            print(
                f"WARNING: device_eval_microbatch_size > device_eval_batch_size, "
                f"will be reduced from {device_eval_microbatch_size} -> {device_eval_batch_size}."
            )
            device_eval_microbatch_size = device_eval_batch_size
    cfg.device_eval_batch_size = device_eval_batch_size
    cfg.device_eval_microbatch_size = device_eval_microbatch_size
    return cfg


# from timm: https://github.com/huggingface/pytorch-image-models/blob/main/timm/optim/optim_factory.py
# Copyright 2019 Ross Wightman, Apache-2.0 License
def param_groups_weight_decay(model: nn.Module, weight_decay=1e-5, no_weight_decay_list=()):
    no_weight_decay_list = set(no_weight_decay_list)
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if param.ndim <= 1 or name.endswith(".bias") or name in no_weight_decay_list:
            no_decay.append(param)
        else:
            decay.append(param)

    return [{"params": no_decay, "weight_decay": 0.0}, {"params": decay, "weight_decay": weight_decay}]


def log_config(cfg: DictConfig):
    print(om.to_yaml(cfg))
    if "wandb" in cfg.get("loggers", {}):
        try:
            import wandb
        except ImportError as e:
            raise e
        if wandb.run:
            wandb.config.update(om.to_container(cfg, resolve=True))


def build_algorithm(name, kwargs):
    if name == "gradient_clipping":
        return algorithms.GradientClipping(**kwargs)
    elif name == "alibi":
        return algorithms.Alibi(**kwargs)
    elif name == "gated_linear_units":
        return algorithms.GatedLinearUnits(**kwargs)
    else:
        raise ValueError(f"Not sure how to build algorithm: {name}")


def build_callback(name, kwargs):
    if name == "lr_monitor":
        return LRMonitor()
    elif name == "memory_monitor":
        return MemoryMonitor()
    elif name == "speed_monitor":
        return SpeedMonitor(
            window_size=kwargs.get("window_size", 1), gpu_flops_available=kwargs.get("gpu_flops_available", None)
        )
    elif name == "runtime_estimator":
        return RuntimeEstimator()
    elif name == "optimizer_monitor":
        return OptimizerMonitor(
            log_optimizer_metrics=kwargs.get("log_optimizer_metrics", True),
        )
    elif name == "scheduled_gc":
        return ScheduledGarbageCollector(batch_interval=kwargs.get("batch_interval", 100_000))
    elif name == "log_grad_norm":
        return LogGradNorm(
            log_optimizer_metrics=kwargs.get("log_optimizer_metrics", True),
            batch_log_interval=kwargs.get("batch_log_interval", 10),
        )
    elif name == "dataloader_speed":
        return DataloaderSpeedMonitor()
    elif name == "packing_efficiency":
        return PackingEfficency(log_interval=kwargs.get("log_interval", 10))
    else:
        raise ValueError(f"Not sure how to build callback: {name}")


def build_logger(name, kwargs):
    if name == "wandb":
        return WandBLogger(**kwargs)
    else:
        raise ValueError(f"Not sure how to build logger: {name}")


def build_scheduler(cfg):
    if cfg.name == "constant_with_warmup":
        return ConstantWithWarmupScheduler(t_warmup=cfg.t_warmup)
    elif cfg.name == "cosine_with_warmup":
        return CosineAnnealingWithWarmupScheduler(t_warmup=cfg.t_warmup, alpha_f=cfg.alpha_f)
    elif cfg.name == "linear_decay_with_warmup":
        return LinearWithWarmupScheduler(t_warmup=cfg.t_warmup, alpha_f=cfg.alpha_f)
    elif cfg.name == "warmup_stable_decay":
        return WarmupStableDecayScheduler(
            t_warmup=cfg.t_warmup, alpha_f=cfg.alpha_f, t_decay=cfg.get("t_decay", "0.1dur")
        )
    elif cfg.name == "cosine_inverse_sqrt":
        return CosineInverseSqrtScheduler(
            t_warmup=cfg.t_warmup,
            t_cooldown=cfg.t_cooldown,
            t_cosine=cfg.get("t_cosine", "0.25dur"),
            alpha_f=cfg.alpha_f,
            alpha_s=cfg.get("alpha_s", 0.0),
            warmup_schedule=cfg.get("warmup_schedule", "linear"),
            cooldown_schedule=cfg.get("cooldown_schedule", "linear"),
        )
    elif cfg.name == "one_minus_sqrt":
        return OneMinusSqrtScheduler(t_decay=cfg.t_decay, t_max=cfg.t_max, alpha_f=cfg.alpha_f)
    else:
        raise ValueError(f"Not sure how to build scheduler: {cfg.name}")


def build_optimizer(cfg, model):
    if cfg.get("filter_bias_norm_wd", False):
        params = param_groups_weight_decay(model, weight_decay=cfg.weight_decay)
    else:
        params = model.parameters()

    if cfg.name == "decoupled_adamw":
        return DecoupledAdamW(params, lr=cfg.lr, betas=list(cfg.betas), eps=cfg.eps, weight_decay=cfg.weight_decay)
    elif cfg.name == "adamw":
        print(
            "INFO: You might want to increase the weight decay because in AdamW it is scaled by the lr."
            f" Default weight decay is ``1e-2`` -> {cfg.weight_decay}. Default lr is `lr=1e-3` -> {cfg.lr}."
        )
        return AdamW(params, lr=cfg.lr, betas=list(cfg.betas), eps=cfg.eps, weight_decay=cfg.weight_decay)
    elif cfg.name == "stableadamw":
        try:
            if cfg.get("log_grad_norm", False):
                from src.optimizer import StableAdamW
            else:
                from optimi import StableAdamW
        except ImportError:
            raise ImportError("Install `pip install torch-optimi` to use the StableAdamW optimizer.")

        print(
            "INFO: You might want to increase the weight decay because in StableAdamW it is scaled by the lr."
            f" Default weight decay is ``1e-2`` -> {cfg.weight_decay}. Default lr is `lr=1e-3` -> {cfg.lr}."
        )
        return StableAdamW(params, lr=cfg.lr, betas=list(cfg.betas), eps=cfg.eps, weight_decay=cfg.weight_decay)
    elif cfg.name == "decoupled_stableadamw":
        try:
            if cfg.get("log_grad_norm", False):
                from src.optimizer import StableAdamW
            else:
                from optimi import StableAdamW
        except ImportError:
            raise ImportError("Install `pip install torch-optimi` to use the StableAdamW optimizer.")

        return StableAdamW(
            params,
            lr=cfg.lr,
            betas=list(cfg.betas),
            eps=cfg.eps,
            weight_decay=cfg.weight_decay,
            decouple_lr=True,
        )
    else:
        raise ValueError(f"Not sure how to build optimizer: {cfg.name}")


def get_num_tokens_in_batch_unpadded(batch: dict):
    return batch["attention_mask"].sum().item()

def create_reasoning_ds(split, tokenizer, max_seq_length, 
                        prefix, eval_mode=False, overwrite_prefix=False):
    # not ideal, but overwrite prefix
    if overwrite_prefix:
        prefix = "Determine which response is the best choice based on mathematical or programming accuracy. "

    if not eval_mode:
        dataset = data_module.create_reasoning_preference_to_flan_style_dataset(
            task="sarahpann/skywork_reasoning",
            split=split,
            tokenizer_name=tokenizer,
            max_seq_length=max_seq_length,
            prefix=prefix,
        )
        return dataset

    else:
        dataset = data_module.create_rw_bench_reasoning_preference_to_flan_style_dataset(
            task="sarahpann/rwb_reasoning",
            split=split,
            tokenizer_name=tokenizer,
            max_seq_length=max_seq_length,
            prefix=prefix,
        )
        return dataset
    
def create_safety_ds(split, tokenizer, max_seq_length, 
                        prefix, eval_mode=False, overwrite_prefix=False):
    if overwrite_prefix:
        prefix = "Determine the best response based on safety. "
    if not eval_mode:
        dataset = data_module.create_preference_to_flan_style_dataset(
            task="sarahpann/PKU_SafeRLHF_simp",
            split=split,
            tokenizer_name=tokenizer,
            max_seq_length=max_seq_length,
            prefix=prefix,

            dataset_name="sarahpann/PKU_SafeRLHF_simp",
            dataset_subset="",
            task_column_names={"sarahpann/PKU_SafeRLHF_simp": ('chosen', 'rejected')}
        )

        dataset1 = data_module.create_preference_to_flan_style_dataset(
            task="sarahpann/skywork_safety",
            split=split,
            tokenizer_name=tokenizer,
            max_seq_length=max_seq_length,
            prefix=prefix,

            dataset_name="sarahpann/skywork_safety",
            dataset_subset="",
            task_column_names={"sarahpann/skywork_safety": ('chosen', 'rejected', 'og_dataset')}
        )

        dataset = datasets.concatenate_datasets([dataset, dataset1])
    
    else:
        dataset = data_module.create_preference_to_flan_style_dataset(
            task="sarahpann/rwb_safety",
            split=split,
            tokenizer_name=tokenizer,
            max_seq_length=max_seq_length,
            prefix=prefix,

            dataset_name="sarahpann/rwb_safety",
            dataset_subset="",
            task_column_names={"sarahpann/rwb_safety": ('chosen', 'rejected', 'og_dataset')}
        )

    return dataset

def create_chat_ds(split, tokenizer, max_seq_length, 
                    prefix, eval_mode=False, overwrite_prefix=False):
    if overwrite_prefix:
        prefix = "Which response is the most helpful, relevant, and correct? "
    if not eval_mode:
        all_datasets = []
        dataset = data_module.create_preference_to_flan_style_dataset(
            task="sarahpann/skywork_chat",
            split=split,
            tokenizer_name=tokenizer,
            max_seq_length=max_seq_length,
            prefix=prefix,

            dataset_name="sarahpann/skywork_chat",
            dataset_subset="",
            task_column_names={"sarahpann/skywork_chat": ('question', 'chosen', 'rejected', 'og_dataset')}
        )

        all_datasets.append(dataset)

        dataset1 = data_module.create_preference_to_flan_style_dataset(
            task="sarahpann/webgpt_comparisons_simp",
            split=split,
            tokenizer_name=tokenizer,
            max_seq_length=max_seq_length,
            prefix=prefix,

            dataset_name="sarahpann/webgpt_comparisons_simp",
            dataset_subset="",
            task_column_names={"sarahpann/webgpt_comparisons_simp": ('question', 'chosen', 'rejected')}
        )

        all_datasets.append(dataset1)

        # if not overwrite_prefix: # no hhrlhf if mixed training
        #     dataset2 = data_module.create_preference_to_flan_style_dataset(
        #         task="sarahpann/simp_hhrlhf",
        #         split=split,
        #         tokenizer_name=tokenizer,
        #         max_seq_length=max_seq_length,
        #         prefix=prefix,

        #         dataset_name="sarahpann/simp_hhrlhf",
        #         dataset_subset="",
        #         task_column_names={"sarahpann/simp_hhrlhf": ('question', 'chosen', 'rejected')}
        #     )

        #     all_datasets.append(dataset2)

        #     # dataset = datasets.concatenate_datasets([dataset, dataset1, dataset2])
        
        # else: 
        #     dataset = datasets.concatenate_datasets([dataset, dataset1])

        dataset = datasets.concatenate_datasets(all_datasets)

    else:
        dataset1 = data_module.create_preference_to_flan_style_dataset(
            task="sarahpann/rwb_chat",
            split=split,
            tokenizer_name=tokenizer,
            max_seq_length=max_seq_length,
            prefix=prefix,

            dataset_name="sarahpann/rwb_chat",
            dataset_subset="",
            task_column_names={"sarahpann/rwb_chat": ('question', 'chosen', 'rejected', 'og_dataset')}
        )

        dataset2 = data_module.create_preference_to_flan_style_dataset(
            task="sarahpann/rwb_chat_hard",
            split=split,
            tokenizer_name=tokenizer,
            max_seq_length=max_seq_length,
            prefix=prefix,

            dataset_name="sarahpann/rwb_chat_hard",
            dataset_subset="",
            task_column_names={"sarahpann/rwb_chat_hard": ('question', 'chosen', 'rejected', 'og_dataset')}
        )

        dataset = datasets.concatenate_datasets([dataset1, dataset2])

    return dataset


def build_dataloader(
    cfg,
    tokenizer,
    device_batch_size,
    count_padding_tokens=True,
    device_microbatch_size: int | None = None,
    eval_mode=False,
):
    if cfg.subset == "reasoning":
        dataset = create_reasoning_ds(cfg.split, cfg.tokenizer_name, cfg.max_seq_len, cfg.prefix, eval_mode)

    elif cfg.subset == "safety":
        dataset = create_safety_ds(cfg.split, cfg.tokenizer_name, cfg.max_seq_len, cfg.prefix, eval_mode)

    elif cfg.subset == "chat":
        dataset = create_chat_ds(cfg.split, cfg.tokenizer_name, cfg.max_seq_len, cfg.prefix, eval_mode)

    elif cfg.subset == "all_at_once":
        dataset1 = create_reasoning_ds(cfg.split, cfg.tokenizer_name, cfg.max_seq_len, cfg.prefix, eval_mode, overwrite_prefix=True)
        dataset2 = create_safety_ds(cfg.split, cfg.tokenizer_name, cfg.max_seq_len, cfg.prefix, eval_mode, overwrite_prefix=True)
        dataset3 = create_chat_ds(cfg.split, cfg.tokenizer_name, cfg.max_seq_len, cfg.prefix, eval_mode, overwrite_prefix=True)

        dataset = datasets.concatenate_datasets([dataset1, dataset2, dataset3])


    class CustomDataCollatorForLanguageModeling(transformers.DataCollatorForLanguageModeling):
        # same init function, but add length of prefix to the class
        def __init__(self, tokenizer, mlm_probability=0.15, prompt=None):
            super().__init__(tokenizer=tokenizer, mlm_probability=mlm_probability)
            self.tokenizer = tokenizer
            self.mlm_probability = mlm_probability
            if prompt is not None:
                self.prompt_len = len(tokenizer(prompt)["input_ids"]) - 1 # for the sep token

        def torch_mask_tokens(self, inputs, special_tokens_mask):
            """
            Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
            """
            labels = inputs.clone()
            # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
            probability_matrix = torch.full(labels.shape, self.mlm_probability)
            if special_tokens_mask is None:
                special_tokens_mask = [
                    tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
                ]
                special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
                # block out first three tokens for joint CLS training
                # TODO: PONDER, should we be special masking these if not join CLS training?
                if not cfg.add_prefix:
                    special_tokens_mask[:, :3] = True
                # else:
                #     special_tokens_mask[:, :self.prompt_len] = True

            else:
                special_tokens_mask = special_tokens_mask.bool()

            probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
            masked_indices = torch.bernoulli(probability_matrix).bool()
            labels[~masked_indices] = -100  # We only compute loss on masked tokens

            # except second token of each sequence
            if not cfg.add_prefix:
                labels[:, 1] = inputs[:, 1].clone()

            # if cfg.add_prefix:
            #     labels[:, 6] = inputs[:, 6].clone()

            # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
            indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
            inputs[indices_replaced] = tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

            # 10% of the time, we replace masked input tokens with random word
            indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
            random_words = torch.randint(len(tokenizer), labels.shape, dtype=torch.long)
            inputs[indices_random] = random_words[indices_random]

            if not cfg.add_prefix:
                inputs[:, 1] = tokenizer.cls_token_id
            # else:
            #     inputs[:, 6] = tokenizer.cls_token_id

            # if cfg.add_prefix:
                # last_padding_token = 

            # The rest of the time (10% of the time) we keep the masked input tokens unchanged
            return inputs, labels


    class CustomDataCollatorForFlanStyleQuestionAnswering(transformers.DataCollatorForLanguageModeling):
        def __init__(self, tokenizer, mlm_probability=0.15, prompt=None):
            super().__init__(tokenizer=tokenizer, mlm_probability=mlm_probability)
            self.tokenizer = tokenizer
            self.mlm_probability = mlm_probability

        def torch_mask_tokens(self, inputs, special_tokens_mask):
            """
            Mask the last non-SEP non-PAD token in the sequence.
            """
            labels = inputs.clone()

            pad_token_id = self.tokenizer.pad_token_id
            sep_token_id = self.tokenizer.sep_token_id
            mask_token_id = self.tokenizer.mask_token_id

            batch_size, seq_length = inputs.shape

            if special_tokens_mask is None:
                special_tokens_mask = [
                    tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
                ]
                special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)

            # Find the last [SEP] token index in each sequence
            sep_positions = (inputs == sep_token_id).int()
            last_sep_indices = (sep_positions * torch.arange(seq_length, device=inputs.device)).argmax(dim=1)

            # Initialize a mask for which token to replace with [MASK]
            mask_positions = torch.zeros_like(inputs, dtype=torch.bool)

            for i in range(batch_size):
                sep_index = last_sep_indices[i].item()

                # Traverse backward to find the second-to-last valid token
                for j in range(sep_index - 1, -1, -1):
                    if inputs[i, j] not in {pad_token_id, sep_token_id}:
                        mask_positions[i, j] = True
                        break

            # Apply mask
            inputs[mask_positions] = mask_token_id
            labels[~mask_positions] = -100  # Only keep masked token for loss calculation

            # print('SAMPLE DB INPUT: ', tokenizer.decode(inputs[0]))
            # print('SAMPLE DB LABEL: ', tokenizer.decode(labels[0]))

            return inputs, labels

        
    # collator = CustomDataCollatorForLanguageModeling(tokenizer=tokenizer, 
    #                                                  mlm_probability=cfg.dataset.mlm_probability,
    #                                                  prompt=cfg.prefix)

    collator = CustomDataCollatorForFlanStyleQuestionAnswering(tokenizer=tokenizer,
                                                                mlm_probability=cfg.dataset.mlm_probability,
                                                                prompt=cfg.prefix)
        
    dataset = cast(Dataset, dataset)
    data_loader = DataLoader(
        dataset,
        collate_fn=collator,
        batch_size=device_batch_size,
        sampler=dist.get_sampler(dataset, drop_last=cfg.drop_last, shuffle=cfg.shuffle),
        num_workers=cfg.num_workers,
        pin_memory=cfg.get("pin_memory", True),
        prefetch_factor=cfg.get("prefetch_factor", 2),
        persistent_workers=cfg.get("persistent_workers", True),
        timeout=cfg.get("timeout", 0),
    )

    # split_batch_fn = None
    # num_samples_in_batch_fn = None
    # num_tokens_in_batch_fn = None

    # if cfg.name == "text":
    #     data_loader = text_data_module.build_text_dataloader(
    #         cfg,
    #         tokenizer,
    #         device_batch_size,
    #         device_microbatch_size=device_microbatch_size,
    #     )
    # else:
    #     raise ValueError(f"Not sure how to build dataloader with config: {cfg}")

    # if not count_padding_tokens:
    #     num_tokens_in_batch_fn = get_num_tokens_in_batch_unpadded
    # if cfg.get("sequence_packing", False):
    #     split_batch_fn = split_packed_batch
    #     num_samples_in_batch_fn = get_num_samples_in_packed_batch

    # data_loader = DataSpec(
    #     data_loader,
    #     get_num_tokens_in_batch=num_tokens_in_batch_fn,
    #     split_batch=split_batch_fn,
    #     get_num_samples_in_batch=num_samples_in_batch_fn,
    # )
    return data_loader


def build_model(cfg: DictConfig):
    if cfg.name == "hf_bert":
        return hf_bert_module.create_hf_bert_mlm(
            pretrained_model_name=cfg.pretrained_model_name,
            use_pretrained=cfg.get("use_pretrained", None),
            model_config=cfg.get("model_config", None),
            tokenizer_name=cfg.get("tokenizer_name", None),
            gradient_checkpointing=cfg.get("gradient_checkpointing", None),
        )
    elif cfg.name == "mosaic_bert":
        return mosaic_bert_module.create_mosaic_bert_mlm(
            pretrained_model_name=cfg.pretrained_model_name,
            pretrained_checkpoint=cfg.get("pretrained_checkpoint", None),
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
    elif cfg.name == "og_bert":
        return flex_bert_module.create_og_bert_mlm(
            pretrained_model_name=cfg.pretrained_model_name,
            pretrained_checkpoint=cfg.get("pretrained_checkpoint", None),
            model_config=cfg.get("model_config", None),
            tokenizer_name=cfg.get("tokenizer_name", None),
            gradient_checkpointing=cfg.get("gradient_checkpointing", None),
            recompute_metric_loss=cfg.get("recompute_metric_loss", False),
            disable_train_metrics=cfg.get("disable_train_metrics", False),
            use_dora=cfg.get("use_dora", False),
            r_dim=cfg.get("r_dim", None),
            mixed_mlm=cfg.get("mixed_mlm", False),
            checkpoint_dict=cfg.get("checkpoint_dict", None),
        )
    else:
        raise ValueError(f"Not sure how to build model with name={cfg.name}")


def init_from_checkpoint(cfg: DictConfig, new_model: nn.Module):
    print(f"Initializing model from checkpoint {cfg.checkpoint_run_name}")
    checkpoint_cfg = Path(cfg.checkpoint_cfg)
    assert checkpoint_cfg.exists(), f"Checkpoint config {checkpoint_cfg} does not exist"
    pretrained_cfg = om.load(checkpoint_cfg)

    pretrained_model = build_model(pretrained_cfg.model)
    n_params = sum(p.numel() for p in pretrained_model.parameters())

    # checkpoint_filepath = Path(cfg.checkpoint_load_path) / f"{cfg.checkpoint_run_name}" / "latest-rank0.pt"
    checkpoint_filepath = Path(cfg.checkpoint_load_path)
    assert checkpoint_filepath.exists(), f"Checkpoint {checkpoint_filepath} does not exist"
    state = torch.load(_ensure_valid_checkpoint(checkpoint_filepath), map_location="cpu")

    state_dict = state.get("state", {})
    model_state = state_dict.get("model", {})
    assert len(model_state) > 0, "Model state is empty, please check the checkpoint and checkpoint path"

    pretrained_model.load_state_dict(model_state)

    if isinstance(pretrained_cfg.model.model_config, DictConfig):
        model_config = OmegaConf.to_container(pretrained_cfg.model.model_config, resolve=True)
    pretrained_config = FlexBertConfig.from_pretrained(pretrained_cfg.model.pretrained_model_name, **model_config)

    init_mlm_model_from_pretrained(
        config=pretrained_config,
        pretrained_model=pretrained_model,
        new_model=new_model,
        mode=cfg.get("mode", "tile_weights_from_middle"),
    )
    print(f"Initalized model from checkpoint {cfg.checkpoint_run_name} with {n_params=:.4e} parameters")


def main(cfg: DictConfig, return_trainer: bool = False, do_train: bool = True) -> Optional[Trainer]:
    print("Training using config: ")
    print(om.to_yaml(cfg))
    reproducibility.seed_all(cfg.seed)

    # Get batch size info
    cfg = update_batch_size_info(cfg)

    # Build Model
    print("Initializing model...")
    model = build_model(cfg.model)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"{n_params=:.4e}")

    print("Model max sequence length: ", model.config.max_position_embeddings)

    if cfg.get("init_from_checkpoint", None) is not None:
        init_from_checkpoint(cfg.init_from_checkpoint, model)

    # Dataloaders
    print("Building train loader...")
    train_loader = build_dataloader(
        cfg=cfg.train_loader,
        tokenizer=model.tokenizer,
        device_batch_size=cfg.global_train_batch_size // dist.get_world_size(),
        count_padding_tokens=cfg.get("count_padding_tokens", True),
        device_microbatch_size=cfg.device_train_microbatch_size,
    )
    if cfg.get("eval_loader", None) is not None:
        print("Building eval loader...")
        global_eval_batch_size = cfg.get("global_eval_batch_size", cfg.global_train_batch_size)
        eval_loader = build_dataloader(
            cfg=cfg.eval_loader,
            tokenizer=model.tokenizer,
            device_batch_size=cfg.get("device_eval_batch_size", global_eval_batch_size // dist.get_world_size()),
            eval_mode=True,
        )
        eval_evaluator = Evaluator(
            label="eval",
            dataloader=eval_loader,
            device_eval_microbatch_size=cfg.get("device_eval_microbatch_size", None),
        )
    else:
        eval_evaluator = None

    # Optimizer
    optimizer = build_optimizer(cfg.optimizer, model)

    # Scheduler
    scheduler = build_scheduler(cfg.scheduler)

    # Loggers
    loggers = [build_logger(name, logger_cfg) for name, logger_cfg in cfg.get("loggers", {}).items()]

    # Callbacks
    callbacks = [build_callback(name, callback_cfg) for name, callback_cfg in cfg.get("callbacks", {}).items()]

    # Algorithms
    if (
        cfg.get("algorithms", {}).get("gradient_clipping", {}).get("clipping_threshold", 0) > 0
    ) and "stableadamw" in cfg.get("optimizer", {}).get("name", "adamw"):
        warnings.warn(
            f"The StableAdamW optimizer replaces gradient clipping. "
            f"Set {cfg['algorithms']['gradient_clipping']['clipping_threshold']=} to 0.0"
        )

    algorithms = [build_algorithm(name, algorithm_cfg) for name, algorithm_cfg in cfg.get("algorithms", {}).items()]

    if cfg.get("run_name") is None:
        cfg.run_name = os.environ.get("COMPOSER_RUN_NAME", "bert")

    # Build the Trainer
    trainer = Trainer(
        run_name=cfg.run_name,
        seed=cfg.seed,
        model=model,
        algorithms=algorithms,
        train_dataloader=train_loader,
        eval_dataloader=eval_evaluator,
        train_subset_num_batches=cfg.get("train_subset_num_batches", -1),
        eval_subset_num_batches=cfg.get("eval_subset_num_batches", -1),
        optimizers=optimizer,
        schedulers=scheduler,
        max_duration=cfg.max_duration,
        eval_interval=cfg.eval_interval,
        progress_bar=cfg.progress_bar,
        log_to_console=cfg.log_to_console,
        console_log_interval=cfg.console_log_interval,
        loggers=loggers,
        callbacks=callbacks,
        precision=cfg.precision,
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

    print("Logging config...")
    log_config(cfg)

    if do_train:
        print("Starting training...")
        # this section is intended to use when resuming from a checkpoint where one wants to change
        # the learning rate and weight deacy. It's only been tested with the warmup_stable_decay scheduler
        if cfg.get("restart_override", False):
            print("Overriding checkpoint's scheduler & optimizer LR & WD, and train microbatch size with config options")  # fmt: skip
            if cfg.scheduler.name not in ["constant_with_warmup", "warmup_stable_decay"]:
                print("Rescaling current step LR by ratio of new LR to old LR. This may require scaling the scheduler's alpha_f")  # fmt: skip
                for param_group in trainer.state.optimizers[0].param_groups:
                    lr_ratio = cfg.optimizer.lr / param_group["lr"]
                    param_group["lr"] = cfg.optimizer.lr
                    param_group["weight_decay"] = cfg.optimizer.weight_decay if param_group["weight_decay"] > 0 else 0.0
                for scheduler in trainer.state.schedulers:
                    for i in range(len(scheduler.base_lrs)):
                        scheduler.base_lrs[i] *= lr_ratio
                    for i in range(len(scheduler._last_lr)):
                        scheduler._last_lr[i] *= lr_ratio
            else:
                for param_group in trainer.state.optimizers[0].param_groups:
                    param_group["lr"] = cfg.optimizer.lr
                    param_group["weight_decay"] = cfg.optimizer.weight_decay if param_group["weight_decay"] > 0 else 0.0
                for scheduler in trainer.state.schedulers:
                    for i in range(len(scheduler.base_lrs)):
                        scheduler.base_lrs[i] = cfg.optimizer.lr
                    for i in range(len(scheduler._last_lr)):
                        scheduler._last_lr[i] = cfg.optimizer.lr
            trainer.fit(
                device_train_microbatch_size=cfg.get("device_train_microbatch_size", "auto"),
                reset_time=cfg.get("reset_time", False),
            )

        else:
            trainer.fit(reset_time=cfg.get("reset_time", False))

        to_save = cfg.get("save", False)
        small_or_large = "large" if "base" not in cfg.model.pretrained_model_name else "small"


        if to_save: 
            if cfg.model.get("use_dora"):
                # model = model.model.merge_and_unload()
                # model.push_to_hub(f"sarahpann/{cfg.subset}_model_{small_or_large}")
                model.model.push_to_hub(f"sarahpann/{cfg.subset}_model_{small_or_large}")

            else:
                model.model.push_to_hub(f"sarahpann/{cfg.subset}_model_{small_or_large}")

    if return_trainer:
        return trainer


if __name__ == "__main__":
    om.register_new_resolver("format_lr", lambda lr: f"{lr:.2f}")
    om.register_new_resolver("first_word", lambda text: text.split("_")[0] if text else "")
    
    yaml_path, args_list = sys.argv[1], sys.argv[2:]
    with open("yamls/defaults.yaml") as f:
        default_cfg = om.load(f)
    with open(yaml_path) as f:
        yaml_cfg = om.load(f)
    cli_cfg = om.from_cli(args_list)
    cfg = om.merge(default_cfg, yaml_cfg, cli_cfg)
    cfg = cast(DictConfig, cfg)  # for type checking
    main(cfg)
