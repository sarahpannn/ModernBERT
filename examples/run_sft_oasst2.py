#!/usr/bin/env python3
"""
Example script for running SFT training on OASST2 dataset.

This is a simple example showing how to use the sft_oasst2.py script.
You can modify the configuration parameters below or create your own YAML config file.
"""

import os
import sys

# Add parent directory to path to import the SFT script
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from omegaconf import OmegaConf
from sft_oasst2 import main

def create_example_config():
    """Create a minimal config for testing SFT training."""
    
    config = {
        # Basic settings
        "run_name": "sft_oasst2_example",
        "seed": 42,
        
        # Model configuration - using a smaller model for quick testing
        "model": {
            "name": "hf_bert",  # Using HF BERT for compatibility
            "pretrained_model_name": "bert-base-uncased",
            "use_pretrained": True,
            "tokenizer_name": "bert-base-uncased",
            "gradient_checkpointing": False,
        },
        
        # Training data loader
        "train_loader": {
            "tokenizer_name": "bert-base-uncased",
            "split": "train",
            "max_seq_len": 256,  # Shorter for faster training
            "num_workers": 2,
            "shuffle": True,
            "drop_last": True,
            "pin_memory": True,
        },
        
        # Small batch sizes for testing
        "global_train_batch_size": 8,
        "device_train_microbatch_size": 2,
        
        # Short training duration for testing
        "max_duration": "100ba",  # Just 100 batches for testing
        "eval_interval": "50ba",
        
        # Optimizer
        "optimizer": {
            "name": "adamw",
            "lr": 2e-5,
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "weight_decay": 0.01,
            "filter_bias_norm_wd": True,
        },
        
        # Scheduler
        "scheduler": {
            "name": "constant_with_warmup",
            "t_warmup": "0.1dur",
        },
        
        # Basic settings
        "precision": "fp32",  # Use FP32 for compatibility
        "save_folder": "./test_checkpoints",
        "save_interval": "50ba",
        "save_num_checkpoints_to_keep": 1,
        "progress_bar": True,
        "log_to_console": True,
        "console_log_interval": "10ba",
        
        # Callbacks
        "callbacks": {
            "speed_monitor": {"window_size": 5},
        }
    }
    
    return OmegaConf.create(config)

def run_example():
    """Run the example SFT training."""
    print("Running SFT training example on OASST2 dataset")
    print("=" * 50)
    
    # Create config
    cfg = create_example_config()
    
    print("Configuration:")
    print(OmegaConf.to_yaml(cfg))
    print("=" * 50)
    
    try:
        # Run training
        main(cfg)
        print("\nExample training completed successfully!")
        
    except Exception as e:
        print(f"\nExample training failed with error: {e}")
        print("This might be due to missing dependencies or dataset access issues.")
        print("Make sure you have:")
        print("1. All required dependencies installed (torch, transformers, datasets, composer)")
        print("2. Internet access to download the OASST2 dataset")
        print("3. Sufficient GPU memory if using GPU training")

if __name__ == "__main__":
    run_example()