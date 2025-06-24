#!/usr/bin/env python3
"""
Simplified script to analyze final layer activations of a single model 
on the RewardBench dataset for poster visualization.

Memory-efficient version that processes one model at a time.
"""

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple, Optional, Dict
import argparse
from pathlib import Path
import json
from datasets import load_dataset

# Add project root to path
sys.path.append(os.path.dirname(os.path.realpath(__file__)))

from transformers import AutoTokenizer, AutoModel
import transformers

def load_model(model_name: str, device: str) -> Tuple[torch.nn.Module, AutoTokenizer]:
    """Load model and tokenizer from HuggingFace."""
    print(f"Loading model: {model_name}")
    
    try:
        model = AutoModel.from_pretrained(model_name, output_attentions=True, output_hidden_states=True)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Handle models without pad token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
    except Exception as e:
        print(f"Error loading model {model_name}: {e}")
        print("Falling back to BERT base model...")
        model = AutoModel.from_pretrained("bert-base-uncased", output_attentions=True, output_hidden_states=True)
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    
    model = model.to(device)
    model.eval()
    return model, tokenizer

def load_reward_bench_samples(subset: str = "reasoning", num_samples: int = 50) -> List[Dict]:
    """Load samples from RewardBench dataset."""
    print(f"Loading RewardBench {subset} samples...")
    
    try:
        # Load the dataset
        # dataset = load_dataset("allenai/reward-bench", split="filtered")
        dataset = load_dataset(f"sarahpann/rwb_{subset}", split="train")
        
        # Filter by subset
        # filtered_data = [item for item in dataset if item.get("subset") == subset]
        
        # Take first num_samples
        # samples = filtered_data[:num_samples]

        samples = dataset.select(range(num_samples))
        
        print(f"Loaded {len(samples)} samples from {subset} subset")
        return samples
        
    except Exception as e:
        print(f"Error loading RewardBench: {e}")
        print("Using fallback sample texts...")
        # Fallback to sample texts
        return [
            {"prompt": "What is the capital of France?", "chosen": "The capital of France is Paris.", "rejected": "I don't know."},
            {"prompt": "Explain quantum computing.", "chosen": "Quantum computing uses quantum mechanical phenomena like superposition and entanglement to process information.", "rejected": "Quantum computing is just regular computing but faster."},
            {"prompt": "How do you solve 2+2?", "chosen": "2 + 2 = 4. You can solve this by counting or basic arithmetic.", "rejected": "2 + 2 = 5."},
        ] * (num_samples // 3 + 1)

def extract_activations(model: torch.nn.Module, input_ids: torch.Tensor, 
                       attention_mask: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Extract final layer activations and attention weights."""
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, 
                       output_hidden_states=True, output_attentions=True)
        
        # Extract final layer hidden states
        final_hidden = outputs.hidden_states[-1]  # [batch, seq_len, hidden_size]
        
        # Extract final layer attention weights
        final_attention = None
        if outputs.attentions is not None and len(outputs.attentions) > 0:
            final_attention = outputs.attentions[-1]  # [batch, num_heads, seq_len, seq_len]
            
        return final_hidden, final_attention

def aggregate_sequence_representations(hidden_states: torch.Tensor, 
                                     attention_mask: torch.Tensor, 
                                     method: str = "mean") -> torch.Tensor:
    """Aggregate sequence representations using different pooling methods."""
    if method == "mean":
        # Mean pooling with attention mask
        mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
        sum_embeddings = torch.sum(hidden_states * mask_expanded, 1)
        sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask
    elif method == "cls":
        # Use [CLS] token (first token)
        return hidden_states[:, 0, :]
    elif method == "max":
        # Max pooling
        return torch.max(hidden_states, dim=1)[0]
    else:
        raise ValueError(f"Unknown pooling method: {method}")

def visualize_activations(activations: List[torch.Tensor], labels: List[str], 
                         model_name: str, save_path: str):
    """Create visualization of model activations."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'Model Activations Analysis: {model_name}', fontsize=16)
    
    # Combine all activations
    all_activations = torch.cat(activations, dim=0)  # [num_samples, hidden_size]
    
    # 1. Activation distribution
    axes[0, 0].hist(all_activations.flatten().numpy(), bins=50, alpha=0.7, color='blue')
    axes[0, 0].set_title('Final Layer Activation Distribution')
    axes[0, 0].set_xlabel('Activation Value')
    axes[0, 0].set_ylabel('Frequency')
    
    # 2. Activation magnitude per sample
    norms = torch.norm(all_activations, dim=1).numpy()
    axes[0, 1].bar(range(len(norms)), norms, alpha=0.7, color='green')
    axes[0, 1].set_title('Activation Magnitude per Sample')
    axes[0, 1].set_xlabel('Sample Index')
    axes[0, 1].set_ylabel('L2 Norm')
    
    # 3. Dimensionality analysis (PCA-like visualization)
    # Show variance across hidden dimensions
    dim_variance = torch.var(all_activations, dim=0).numpy()
    axes[1, 0].plot(dim_variance, alpha=0.7, color='red')
    axes[1, 0].set_title('Variance Across Hidden Dimensions')
    axes[1, 0].set_xlabel('Hidden Dimension')
    axes[1, 0].set_ylabel('Variance')
    
    # 4. Sample comparison (if we have different types)
    if len(set(labels)) > 1:
        # Group by label and show mean activations
        unique_labels = list(set(labels))
        means = []
        for label in unique_labels:
            label_indices = [i for i, l in enumerate(labels) if l == label]
            label_activations = all_activations[label_indices]
            means.append(torch.mean(label_activations, dim=0).numpy())
        
        # Plot mean activation patterns
        for i, (label, mean_act) in enumerate(zip(unique_labels, means)):
            axes[1, 1].plot(mean_act[:100], alpha=0.7, label=label)  # Show first 100 dims
        
        axes[1, 1].set_title('Mean Activation Patterns by Type')
        axes[1, 1].set_xlabel('Hidden Dimension (first 100)')
        axes[1, 1].set_ylabel('Mean Activation')
        axes[1, 1].legend()
    else:
        # Show activation heatmap for first few samples
        sample_activations = all_activations[:min(10, len(all_activations)), :100]
        sns.heatmap(sample_activations.numpy(), ax=axes[1, 1], cmap='viridis', cbar=True)
        axes[1, 1].set_title('Activation Patterns (First 10 samples, 100 dims)')
        axes[1, 1].set_xlabel('Hidden Dimension')
        axes[1, 1].set_ylabel('Sample Index')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Activation analysis saved to: {save_path}")

def visualize_attention_patterns(attention_weights: List[torch.Tensor], 
                               model_name: str, save_path: str):
    """Visualize aggregated attention patterns."""
    if not attention_weights:
        print("No attention weights available for visualization")
        return
    
    # Average attention across all samples and heads
    all_attention = torch.stack(attention_weights, dim=0)  # [num_samples, batch, heads, seq, seq]
    
    # Take mean across samples, batch, and heads
    avg_attention = torch.mean(all_attention, dim=(0, 1, 2))  # [seq_len, seq_len]
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle(f'Attention Pattern Analysis: {model_name}', fontsize=16)
    
    # 1. Average attention heatmap
    sns.heatmap(avg_attention.numpy(), ax=axes[0], cmap='Blues', cbar=True)
    axes[0].set_title('Average Attention Pattern')
    axes[0].set_xlabel('Key Position')
    axes[0].set_ylabel('Query Position')
    
    # 2. Attention head diversity (variance across heads)
    head_variance = torch.var(all_attention, dim=2).mean(dim=(0, 1))  # [seq_len, seq_len]
    sns.heatmap(head_variance.numpy(), ax=axes[1], cmap='Reds', cbar=True)
    axes[1].set_title('Attention Head Diversity (Variance)')
    axes[1].set_xlabel('Key Position')
    axes[1].set_ylabel('Query Position')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Attention pattern analysis saved to: {save_path}")

def main():
    parser = argparse.ArgumentParser(description="Analyze model activations on RewardBench")
    parser.add_argument("--model", type=str, default="answerdotai/ModernBERT-base",
                       help="HuggingFace model name")
    parser.add_argument("--subset", type=str, default="reasoning",
                       choices=["reasoning", "safety", "chat", "chat_hard"],
                       help="RewardBench subset to analyze")
    parser.add_argument("--num_samples", type=int, default=50,
                       help="Number of samples to analyze")
    parser.add_argument("--max_length", type=int, default=512,
                       help="Maximum sequence length")
    parser.add_argument("--pooling", type=str, default="mean",
                       choices=["mean", "cls", "max"],
                       help="Sequence pooling method")
    parser.add_argument("--output_dir", type=str, default="./model_analysis",
                       help="Directory to save outputs")
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
    
    # Load model
    model, tokenizer = load_model(args.model, device)
    
    # Load dataset
    samples = load_reward_bench_samples(args.subset, args.num_samples)
    
    print(f"\nProcessing {len(samples)} samples...")
    
    # Process samples and collect activations
    all_activations = []
    all_attention = []
    sample_labels = []
    
    for i, sample in enumerate(samples):
        if i % 10 == 0:
            print(f"Processing sample {i+1}/{len(samples)}")
        
        # Process chosen response
        prompt = sample.get("prompt", "")
        chosen = sample.get("chosen", "")
        text = f"{prompt} {chosen}"
        
        # Tokenize
        inputs = tokenizer(text, return_tensors="pt", max_length=args.max_length, 
                          truncation=True, padding=True).to(device)
        
        # Extract activations
        hidden_states, attention = extract_activations(model, inputs.input_ids, inputs.attention_mask)
        
        # Pool sequence representation
        pooled = aggregate_sequence_representations(hidden_states, inputs.attention_mask, args.pooling)
        
        all_activations.append(pooled.cpu())
        sample_labels.append("chosen")
        
        if attention is not None:
            all_attention.append(attention.cpu())
        
        # Also process rejected response if available
        if "rejected" in sample:
            rejected = sample["rejected"]
            text_rejected = f"{prompt} {rejected}"
            
            inputs_rejected = tokenizer(text_rejected, return_tensors="pt", max_length=args.max_length, 
                                      truncation=True, padding=True).to(device)
            
            hidden_states_rej, attention_rej = extract_activations(model, inputs_rejected.input_ids, 
                                                                 inputs_rejected.attention_mask)
            
            pooled_rej = aggregate_sequence_representations(hidden_states_rej, inputs_rejected.attention_mask, args.pooling)
            
            all_activations.append(pooled_rej.cpu())
            sample_labels.append("rejected")
            
            if attention_rej is not None:
                all_attention.append(attention_rej.cpu())
    
    print("\nGenerating visualizations...")
    
    # Create visualizations
    activation_save_path = output_dir / f"{args.model.replace('/', '_')}_activations.png"
    visualize_activations(all_activations, sample_labels, args.model, str(activation_save_path))
    
    if all_attention:
        attention_save_path = output_dir / f"{args.model.replace('/', '_')}_attention.png"
        visualize_attention_patterns(all_attention, args.model, str(attention_save_path))
    
    # Save analysis summary
    activations_tensor = torch.cat(all_activations, dim=0)
    
    summary = {
        "model": args.model,
        "subset": args.subset,
        "num_samples_processed": len(samples),
        "pooling_method": args.pooling,
        "activation_shape": list(activations_tensor.shape),
        "activation_stats": {
            "mean": float(activations_tensor.mean()),
            "std": float(activations_tensor.std()),
            "max": float(activations_tensor.max()),
            "min": float(activations_tensor.min()),
        },
        "attention_collected": len(all_attention) > 0,
        "sample_types": list(set(sample_labels)),
    }
    
    # Add comparison between chosen and rejected if available
    if "chosen" in sample_labels and "rejected" in sample_labels:
        chosen_indices = [i for i, label in enumerate(sample_labels) if label == "chosen"]
        rejected_indices = [i for i, label in enumerate(sample_labels) if label == "rejected"]
        
        chosen_activations = activations_tensor[chosen_indices]
        rejected_activations = activations_tensor[rejected_indices]
        
        summary["chosen_vs_rejected"] = {
            "chosen_mean": float(chosen_activations.mean()),
            "rejected_mean": float(rejected_activations.mean()),
            "mean_difference": float(chosen_activations.mean() - rejected_activations.mean()),
            "cosine_similarity": float(torch.cosine_similarity(
                chosen_activations.mean(dim=0, keepdim=True),
                rejected_activations.mean(dim=0, keepdim=True)
            )),
        }
    
    summary_path = output_dir / f"{args.model.replace('/', '_')}_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nAnalysis complete! Results saved to: {output_dir}")
    print(f"Summary: {summary_path}")
    
    # Print key insights
    print("\n=== Key Insights ===")
    print(f"Model: {args.model}")
    print(f"Activation dimensionality: {activations_tensor.shape[1]}")
    print(f"Mean activation: {summary['activation_stats']['mean']:.4f}")
    print(f"Activation std: {summary['activation_stats']['std']:.4f}")
    
    if "chosen_vs_rejected" in summary:
        print(f"Chosen vs Rejected mean difference: {summary['chosen_vs_rejected']['mean_difference']:.4f}")
        print(f"Chosen vs Rejected cosine similarity: {summary['chosen_vs_rejected']['cosine_similarity']:.4f}")

if __name__ == "__main__":
    main()