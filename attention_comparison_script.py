#!/usr/bin/env python3
"""
Script to compare final output layer and attention activations between 
a trained encoder-based model (ModernBERT/FlexBERT) and a larger decoder-based LLM.

This script accumulates attentions over multiple examples for richer demonstration.
"""

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import argparse
from pathlib import Path
import json

# Add project root to path
sys.path.append(os.path.dirname(os.path.realpath(__file__)))

# Import project modules
from src.flex_bert import create_flex_bert_classification
from src.bert_layers.configuration_bert import FlexBertConfig
from transformers import AutoTokenizer, AutoModel, GPT2LMHeadModel, GPT2Tokenizer
import transformers

class AttentionExtractor:
    """Helper class to extract attention weights from models."""
    
    def __init__(self):
        self.attention_weights = []
        self.hooks = []
        
    def attention_hook(self, module, input, output):
        """Hook to capture attention weights."""
        if hasattr(output, 'attentions') and output.attentions is not None:
            self.attention_weights.append(output.attentions.detach().cpu())
        elif isinstance(output, tuple) and len(output) > 1:
            # For some models, attention is in the second element
            if hasattr(output[1], 'detach'):
                self.attention_weights.append(output[1].detach().cpu())
    
    def register_hooks(self, model):
        """Register hooks on attention layers."""
        for name, module in model.named_modules():
            if 'attn' in name.lower() or 'attention' in name.lower():
                hook = module.register_forward_hook(self.attention_hook)
                self.hooks.append(hook)
    
    def clear_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        
    def clear_weights(self):
        """Clear stored attention weights."""
        self.attention_weights = []

def load_encoder_model(model_name: str) -> Tuple[torch.nn.Module, AutoTokenizer]:
    """Load the encoder-based model from HuggingFace."""
    print(f"Loading encoder model from HuggingFace: {model_name}")
    
    try:
        # Load model and tokenizer from HuggingFace
        model = AutoModel.from_pretrained(model_name, output_attentions=True, output_hidden_states=True)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Handle models without pad token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
    except Exception as e:
        print(f"Error loading model {model_name}: {e}")
        print("Falling back to BERT base model...")
        # Fallback to standard BERT
        model = AutoModel.from_pretrained("bert-base-uncased", output_attentions=True, output_hidden_states=True)
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    
    model.eval()
    return model, tokenizer

def load_decoder_model(model_name: str = "gpt2") -> Tuple[torch.nn.Module, AutoTokenizer]:
    """Load a decoder-based LLM for comparison."""
    print(f"Loading decoder model: {model_name}")
    
    if "gpt2" in model_name.lower():
        model = GPT2LMHeadModel.from_pretrained(model_name)
        tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token
    else:
        model = AutoModel.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    model.eval()
    return model, tokenizer

def extract_final_layer_outputs(model: torch.nn.Module, input_ids: torch.Tensor, 
                               attention_mask: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Extract final layer hidden states and attention weights from the model."""
    with torch.no_grad():
        if hasattr(model, 'model'):  # For sequence classification models
            outputs = model.model(input_ids=input_ids, attention_mask=attention_mask, 
                                output_hidden_states=True, output_attentions=True)
        else:  # For base models
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, 
                          output_hidden_states=True, output_attentions=True)
        
        # Extract hidden states
        if hasattr(outputs, 'hidden_states'):
            hidden_states = outputs.hidden_states[-1]  # Last layer
        elif hasattr(outputs, 'last_hidden_state'):
            hidden_states = outputs.last_hidden_state
        else:
            hidden_states = outputs[0]  # First element is usually hidden states
            
        # Extract attention weights
        attentions = None
        if hasattr(outputs, 'attentions') and outputs.attentions is not None:
            # Get the last layer's attention weights
            attentions = outputs.attentions[-1]  # Shape: [batch, num_heads, seq_len, seq_len]
            
        return hidden_states, attentions

def prepare_sample_texts() -> List[str]:
    """Prepare sample texts for comparison."""
    return [
        "The quick brown fox jumps over the lazy dog.",
        "In the beginning was the Word, and the Word was with God.",
        "To be or not to be, that is the question.",
        "It was the best of times, it was the worst of times.",
        "All happy families are alike; each unhappy family is unhappy in its own way.",
        "Call me Ishmael. Some years ago—never mind how long precisely—having little or no money in my purse.",
        "It is a truth universally acknowledged, that a single man in possession of a good fortune, must be in want of a wife.",
        "In a hole in the ground there lived a hobbit."
    ]

def aggregate_attention_weights(attention_weights_list: List[torch.Tensor]) -> torch.Tensor:
    """Aggregate attention weights across multiple examples."""
    if not attention_weights_list:
        return None
    
    # Stack and average across examples
    stacked = torch.stack(attention_weights_list, dim=0)
    aggregated = torch.mean(stacked, dim=0)
    return aggregated

def visualize_attention_comparison(encoder_attn: torch.Tensor, decoder_attn: torch.Tensor, 
                                 encoder_name: str, decoder_name: str, save_path: str):
    """Create visualization comparing attention patterns."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'Attention Pattern Comparison: {encoder_name} vs {decoder_name}', fontsize=16)
    
    # Average across heads and examples for visualization
    if encoder_attn is not None:
        enc_avg = encoder_attn.mean(dim=1)  # Average across heads
        enc_heatmap = enc_avg[0].numpy() if enc_avg.shape[0] > 0 else np.zeros((10, 10))
        
        sns.heatmap(enc_heatmap, ax=axes[0, 0], cmap='Blues', cbar=True)
        axes[0, 0].set_title(f'{encoder_name} - Attention Pattern')
        axes[0, 0].set_xlabel('Key Position')
        axes[0, 0].set_ylabel('Query Position')
        
        # Attention head variance
        if encoder_attn.shape[1] > 1:  # Multiple heads
            head_variance = torch.var(encoder_attn, dim=1).mean(dim=0).numpy()
            sns.heatmap(head_variance, ax=axes[0, 1], cmap='Reds', cbar=True)
            axes[0, 1].set_title(f'{encoder_name} - Head Variance')
    
    if decoder_attn is not None:
        dec_avg = decoder_attn.mean(dim=1)  # Average across heads
        dec_heatmap = dec_avg[0].numpy() if dec_avg.shape[0] > 0 else np.zeros((10, 10))
        
        sns.heatmap(dec_heatmap, ax=axes[1, 0], cmap='Greens', cbar=True)
        axes[1, 0].set_title(f'{decoder_name} - Attention Pattern')
        axes[1, 0].set_xlabel('Key Position')
        axes[1, 0].set_ylabel('Query Position')
        
        # Attention head variance
        if decoder_attn.shape[1] > 1:  # Multiple heads
            head_variance = torch.var(decoder_attn, dim=1).mean(dim=0).numpy()
            sns.heatmap(head_variance, ax=axes[1, 1], cmap='Oranges', cbar=True)
            axes[1, 1].set_title(f'{decoder_name} - Head Variance')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Attention comparison saved to: {save_path}")

def visualize_final_outputs(encoder_outputs: List[torch.Tensor], decoder_outputs: List[torch.Tensor],
                          encoder_name: str, decoder_name: str, save_path: str):
    """Create visualization comparing final layer outputs."""
    # Average across examples and sequence length
    enc_avg = torch.stack(encoder_outputs).mean(dim=0).mean(dim=1)  # [batch, hidden_size]
    dec_avg = torch.stack(decoder_outputs).mean(dim=0).mean(dim=1)  # [batch, hidden_size]
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'Final Layer Output Comparison: {encoder_name} vs {decoder_name}', fontsize=16)
    
    # Distribution of activations
    axes[0, 0].hist(enc_avg.flatten().numpy(), bins=50, alpha=0.7, label=encoder_name, color='blue')
    axes[0, 0].set_title('Final Layer Activation Distribution')
    axes[0, 0].set_xlabel('Activation Value')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].legend()
    
    axes[0, 1].hist(dec_avg.flatten().numpy(), bins=50, alpha=0.7, label=decoder_name, color='green')
    axes[0, 1].set_title('Final Layer Activation Distribution')
    axes[0, 1].set_xlabel('Activation Value')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].legend()
    
    # Activation magnitude comparison
    enc_norms = torch.norm(enc_avg, dim=1).numpy()
    dec_norms = torch.norm(dec_avg, dim=1).numpy()
    
    x = range(len(enc_norms))
    axes[1, 0].bar(x, enc_norms, alpha=0.7, label=encoder_name, color='blue')
    axes[1, 0].set_title('Output Magnitude per Example')
    axes[1, 0].set_xlabel('Example Index')
    axes[1, 0].set_ylabel('L2 Norm')
    axes[1, 0].legend()
    
    x = range(len(dec_norms))
    axes[1, 1].bar(x, dec_norms, alpha=0.7, label=decoder_name, color='green')
    axes[1, 1].set_title('Output Magnitude per Example')
    axes[1, 1].set_xlabel('Example Index')
    axes[1, 1].set_ylabel('L2 Norm')
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Final output comparison saved to: {save_path}")

def main():
    parser = argparse.ArgumentParser(description="Compare encoder vs decoder model activations")
    parser.add_argument("--encoder_model", type=str, 
                       default="answerdotai/ModernBERT-base",
                       help="HuggingFace model name for encoder model")
    parser.add_argument("--decoder_model", type=str, default="gpt2-medium",
                       help="Decoder model name (e.g., gpt2, gpt2-medium, gpt2-large)")
    parser.add_argument("--output_dir", type=str, default="./attention_analysis",
                       help="Directory to save outputs")
    parser.add_argument("--max_length", type=int, default=128,
                       help="Maximum sequence length")
    parser.add_argument("--device", type=str, default="auto",
                       help="Device to use (cuda, cpu, or auto)")
    
    args = parser.parse_args()
    
    # Setup
    device = "cuda" if torch.cuda.is_available() and args.device == "auto" else args.device
    if args.device != "auto":
        device = args.device
    
    print(f"Using device: {device}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Load models
    try:
        encoder_model, encoder_tokenizer = load_encoder_model(args.encoder_model)
        encoder_model = encoder_model.to(device)
        encoder_name = f"Encoder-{args.encoder_model}"
    except Exception as e:
        print(f"Error loading encoder model: {e}")
        return
    
    try:
        decoder_model, decoder_tokenizer = load_decoder_model(args.decoder_model)
        decoder_model = decoder_model.to(device)
        decoder_name = f"Decoder-{args.decoder_model}"
    except Exception as e:
        print(f"Error loading decoder model: {e}")
        return
    
    # Prepare sample texts
    texts = prepare_sample_texts()
    print(f"Analyzing {len(texts)} sample texts...")
    
    # Process texts and collect activations
    encoder_outputs = []
    decoder_outputs = []
    encoder_attentions = []
    decoder_attentions = []
    
    for i, text in enumerate(texts):
        print(f"Processing text {i+1}/{len(texts)}: {text[:50]}...")
        
        # Process with encoder
        enc_inputs = encoder_tokenizer(text, return_tensors="pt", max_length=args.max_length, 
                                     truncation=True, padding=True).to(device)
        enc_output, enc_attention = extract_final_layer_outputs(encoder_model, enc_inputs.input_ids, 
                                                               enc_inputs.attention_mask)
        encoder_outputs.append(enc_output.cpu())
        if enc_attention is not None:
            encoder_attentions.append(enc_attention.cpu())
        
        # Process with decoder
        dec_inputs = decoder_tokenizer(text, return_tensors="pt", max_length=args.max_length, 
                                     truncation=True, padding=True).to(device)
        dec_output, dec_attention = extract_final_layer_outputs(decoder_model, dec_inputs.input_ids, 
                                                               dec_inputs.attention_mask)
        decoder_outputs.append(dec_output.cpu())
        if dec_attention is not None:
            decoder_attentions.append(dec_attention.cpu())
    
    print("\nGenerating visualizations...")
    
    # Aggregate attention weights
    enc_aggregated_attn = aggregate_attention_weights(encoder_attentions) if encoder_attentions else None
    dec_aggregated_attn = aggregate_attention_weights(decoder_attentions) if decoder_attentions else None
    
    # Create visualizations
    if enc_aggregated_attn is not None or dec_aggregated_attn is not None:
        attention_save_path = output_dir / "attention_comparison.png"
        visualize_attention_comparison(enc_aggregated_attn, dec_aggregated_attn, 
                                     encoder_name, decoder_name, str(attention_save_path))
    
    # Final output comparison
    if encoder_outputs and decoder_outputs:
        output_save_path = output_dir / "final_output_comparison.png"
        visualize_final_outputs(encoder_outputs, decoder_outputs, 
                               encoder_name, decoder_name, str(output_save_path))
    
    # Save summary statistics
    summary = {
        "encoder_model": encoder_name,
        "decoder_model": decoder_name,
        "num_texts_analyzed": len(texts),
        "encoder_output_shape": list(encoder_outputs[0].shape) if encoder_outputs else None,
        "decoder_output_shape": list(decoder_outputs[0].shape) if decoder_outputs else None,
        "encoder_attention_collected": len(encoder_attentions) > 0,
        "decoder_attention_collected": len(decoder_attentions) > 0,
    }
    
    if encoder_outputs:
        enc_stacked = torch.stack(encoder_outputs)
        summary["encoder_stats"] = {
            "mean_activation": float(enc_stacked.mean()),
            "std_activation": float(enc_stacked.std()),
            "max_activation": float(enc_stacked.max()),
            "min_activation": float(enc_stacked.min()),
        }
    
    if decoder_outputs:
        dec_stacked = torch.stack(decoder_outputs)
        summary["decoder_stats"] = {
            "mean_activation": float(dec_stacked.mean()),
            "std_activation": float(dec_stacked.std()),
            "max_activation": float(dec_stacked.max()),
            "min_activation": float(dec_stacked.min()),
        }
    
    summary_path = output_dir / "analysis_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nAnalysis complete! Results saved to: {output_dir}")
    print(f"Summary statistics saved to: {summary_path}")
    
if __name__ == "__main__":
    main()