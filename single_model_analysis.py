#!/usr/bin/env python3
"""
Enhanced script to analyze transformer activations with comprehensive visualization.

Features:
- Feed-forward layer activation analysis across all transformer blocks
- Detailed attention pattern visualization (bidirectional vs decoder)
- Final token attention analysis for both model types
- Poster-ready comparison visualizations
- Memory-efficient processing for large models
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

# Set matplotlib backend for better compatibility
plt.rcParams['figure.max_open_warning'] = 0
plt.style.use('default')

from transformers import AutoTokenizer, AutoModel
import transformers

def detect_model_type(model_name: str) -> str:
    """Detect if model is bidirectional or decoder-based."""
    model_name_lower = model_name.lower()
    
    # Decoder-based models
    if any(keyword in model_name_lower for keyword in ['gpt', 'llama', 'mistral', 'phi', 'gemma']):
        return "decoder"
    
    # Bidirectional models
    if any(keyword in model_name_lower for keyword in ['bert', 'roberta', 'electra', 'deberta']):
        return "bidirectional"
    
    # Default assumption - could be enhanced with more sophisticated detection
    return "bidirectional"

def load_model(model_name: str, device: str) -> Tuple[torch.nn.Module, AutoTokenizer, str]:
    """Load model and tokenizer from HuggingFace."""
    print(f"Loading model: {model_name}")
    
    model_type = detect_model_type(model_name)
    print(f"Detected model type: {model_type}")
    
    try:
        if model_type == "decoder":
            # For decoder models, try to load as AutoModelForCausalLM first
            try:
                from transformers import AutoModelForCausalLM
                model = AutoModelForCausalLM.from_pretrained(model_name, output_attentions=True, output_hidden_states=True)
            except:
                model = AutoModel.from_pretrained(model_name, output_attentions=True, output_hidden_states=True)
        else:
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
        model_type = "bidirectional"
    
    model = model.to(device)
    model.eval()
    return model, tokenizer, model_type

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
                       attention_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
    """Extract comprehensive activations including FFN layers and attention weights."""
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, 
                       output_hidden_states=True, output_attentions=True)
        
        activations = {
            'final_hidden': outputs.hidden_states[-1],  # [batch, seq_len, hidden_size]
            'all_hidden_states': outputs.hidden_states,  # List of [batch, seq_len, hidden_size]
            'all_attentions': outputs.attentions if outputs.attentions else None,
        }
        
        # Extract feed-forward layer activations from each transformer block
        ffn_activations = []
        if hasattr(model, 'encoder') and hasattr(model.encoder, 'layer'):
            # BERT-style model
            for i, layer in enumerate(model.encoder.layer):
                if hasattr(layer, 'intermediate') and hasattr(layer.intermediate, 'dense'):
                    # Hook into intermediate layer to get FFN activations
                    def make_hook(layer_idx):
                        def hook(module, input, output):
                            ffn_activations.append((layer_idx, output.clone()))
                        return hook
                    
                    handle = layer.intermediate.dense.register_forward_hook(make_hook(i))
                    # Run forward pass to trigger hooks
                    _ = model(input_ids=input_ids, attention_mask=attention_mask)
                    handle.remove()
        
        activations['ffn_activations'] = ffn_activations
        
        return activations

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

def visualize_ffn_activations(ffn_activations: List[Tuple[int, torch.Tensor]], 
                             model_name: str, save_path: str):
    """Visualize feed-forward network activations across transformer blocks."""
    if not ffn_activations:
        print("No FFN activations available for visualization")
        return
    
    # Group activations by layer
    layer_activations = {}
    for layer_idx, activation in ffn_activations:
        if layer_idx not in layer_activations:
            layer_activations[layer_idx] = []
        # Pool across sequence length and batch
        pooled = torch.mean(activation, dim=(0, 1))  # [hidden_size]
        layer_activations[layer_idx].append(pooled)
    
    # Average across samples for each layer
    layer_means = {}
    layer_stds = {}
    for layer_idx, activations in layer_activations.items():
        stacked = torch.stack(activations, dim=0)  # [num_samples, hidden_size]
        layer_means[layer_idx] = torch.mean(stacked, dim=0)
        layer_stds[layer_idx] = torch.std(stacked, dim=0)
    
    num_layers = len(layer_means)
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'Feed-Forward Network Activations: {model_name}', fontsize=16)
    
    # 1. FFN activation magnitude by layer
    layer_norms = [torch.norm(layer_means[i]).item() for i in sorted(layer_means.keys())]
    axes[0, 0].bar(range(len(layer_norms)), layer_norms, alpha=0.7, color='blue')
    axes[0, 0].set_title('FFN Activation Magnitude by Layer')
    axes[0, 0].set_xlabel('Transformer Layer')
    axes[0, 0].set_ylabel('L2 Norm of Activations')
    
    # 2. FFN activation patterns across layers (heatmap)
    if num_layers > 1:
        activation_matrix = torch.stack([layer_means[i][:100] for i in sorted(layer_means.keys())], dim=0)
        sns.heatmap(activation_matrix.numpy(), ax=axes[0, 1], cmap='viridis', cbar=True)
        axes[0, 1].set_title('FFN Activation Patterns (First 100 dims)')
        axes[0, 1].set_xlabel('Hidden Dimension')
        axes[0, 1].set_ylabel('Transformer Layer')
    else:
        axes[0, 1].text(0.5, 0.5, 'Single layer only', ha='center', va='center', 
                        transform=axes[0, 1].transAxes)
    
    # 3. Activation diversity across layers
    if num_layers > 1:
        diversity_scores = []
        for i in range(min(100, layer_means[0].shape[0])):
            dim_values = [layer_means[layer][i].item() for layer in sorted(layer_means.keys())]
            diversity_scores.append(np.std(dim_values))
        
        axes[1, 0].plot(diversity_scores, alpha=0.7, color='red')
        axes[1, 0].set_title('Activation Diversity Across Layers')
        axes[1, 0].set_xlabel('Hidden Dimension')
        axes[1, 0].set_ylabel('Standard Deviation Across Layers')
    else:
        axes[1, 0].text(0.5, 0.5, 'Need multiple layers\nfor diversity analysis', 
                        ha='center', va='center', transform=axes[1, 0].transAxes)
    
    # 4. Layer-wise activation statistics
    mean_activations = [torch.mean(layer_means[i]).item() for i in sorted(layer_means.keys())]
    std_activations = [torch.mean(layer_stds[i]).item() for i in sorted(layer_means.keys())]
    
    x_pos = range(len(mean_activations))
    axes[1, 1].bar(x_pos, mean_activations, alpha=0.7, color='green', label='Mean')
    axes[1, 1].bar([x + 0.4 for x in x_pos], std_activations, alpha=0.7, color='orange', 
                   width=0.4, label='Std Dev')
    axes[1, 1].set_title('FFN Activation Statistics by Layer')
    axes[1, 1].set_xlabel('Transformer Layer')
    axes[1, 1].set_ylabel('Activation Value')
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"FFN activation analysis saved to: {save_path}")

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
                               model_name: str, save_path: str, model_type: str = "bidirectional"):
    """Visualize attention patterns with focus on final token attention."""
    if not attention_weights:
        print("No attention weights available for visualization")
        return
    
    # Average attention across all samples
    all_attention = torch.stack(attention_weights, dim=0)  # [num_samples, batch, heads, seq, seq]
    avg_attention_all_heads = torch.mean(all_attention, dim=(0, 1))  # [heads, seq_len, seq_len]
    avg_attention = torch.mean(avg_attention_all_heads, dim=0)  # [seq_len, seq_len]
    
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle(f'Attention Analysis: {model_name} ({model_type})', fontsize=16)
    
    # 1. Overall attention heatmap
    sns.heatmap(avg_attention.numpy(), ax=axes[0, 0], cmap='Blues', cbar=True)
    axes[0, 0].set_title('Average Attention Pattern')
    axes[0, 0].set_xlabel('Key Position')
    axes[0, 0].set_ylabel('Query Position')
    
    # 2. Final token attention (for decoder models) or last meaningful token
    seq_len = avg_attention.shape[0]
    if model_type == "decoder":
        # For decoder models, show how final token attends to previous tokens
        final_token_attention = avg_attention[-1, :].numpy()
        axes[0, 1].bar(range(len(final_token_attention)), final_token_attention, alpha=0.7, color='orange')
        axes[0, 1].set_title('Final Token Attention to Context')
        axes[0, 1].set_xlabel('Context Position')
        axes[0, 1].set_ylabel('Attention Weight')
    else:
        # For bidirectional models, show attention from each position to all others
        attention_entropy = -torch.sum(avg_attention * torch.log(avg_attention + 1e-10), dim=1).numpy()
        axes[0, 1].plot(attention_entropy, marker='o', alpha=0.7, color='green')
        axes[0, 1].set_title('Attention Entropy by Position')
        axes[0, 1].set_xlabel('Token Position')
        axes[0, 1].set_ylabel('Attention Entropy')
    
    # 3. Attention head diversity
    head_variance = torch.var(avg_attention_all_heads, dim=0)  # [seq_len, seq_len]
    sns.heatmap(head_variance.numpy(), ax=axes[0, 2], cmap='Reds', cbar=True)
    axes[0, 2].set_title('Attention Head Diversity')
    axes[0, 2].set_xlabel('Key Position')
    axes[0, 2].set_ylabel('Query Position')
    
    # 4. Attention by layer (if multiple layers available)
    if len(attention_weights) > 1:
        layer_attention = torch.stack([att.mean(dim=(0, 1, 2)) for att in attention_weights[:6]], dim=0)
        for i, layer_att in enumerate(layer_attention):
            axes[1, 0].plot(layer_att.numpy(), alpha=0.7, label=f'Sample {i+1}')
        axes[1, 0].set_title('Attention Patterns Across Samples')
        axes[1, 0].set_xlabel('Token Position')
        axes[1, 0].set_ylabel('Average Attention')
        axes[1, 0].legend()
    else:
        axes[1, 0].text(0.5, 0.5, 'Insufficient data\nfor layer comparison', 
                        ha='center', va='center', transform=axes[1, 0].transAxes)
    
    # 5. Position-wise attention distribution
    position_attention = torch.mean(avg_attention, dim=0).numpy()
    axes[1, 1].bar(range(len(position_attention)), position_attention, alpha=0.7, color='purple')
    axes[1, 1].set_title('Average Attention Received by Position')
    axes[1, 1].set_xlabel('Token Position')
    axes[1, 1].set_ylabel('Average Attention Received')
    
    # 6. Attention span analysis
    attention_spans = []
    for i in range(seq_len):
        # Calculate effective attention span for each query position
        att_weights = avg_attention[i, :].numpy()
        # Find positions that receive >5% of total attention
        significant_positions = np.where(att_weights > 0.05)[0]
        if len(significant_positions) > 0:
            span = significant_positions.max() - significant_positions.min() + 1
            attention_spans.append(span)
        else:
            attention_spans.append(1)
    
    axes[1, 2].plot(attention_spans, marker='s', alpha=0.7, color='brown')
    axes[1, 2].set_title('Attention Span by Query Position')
    axes[1, 2].set_xlabel('Query Position')
    axes[1, 2].set_ylabel('Effective Attention Span')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Enhanced attention analysis saved to: {save_path}")

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
    parser.add_argument("--compare_model", type=str, default=None,
                       help="Optional second model for comparison (e.g., GPT-2 for decoder comparison)")
    
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
    model, tokenizer, model_type = load_model(args.model, device)
    print(f"Model type: {model_type}")
    
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
        
        # Extract comprehensive activations
        activations = extract_activations(model, inputs.input_ids, inputs.attention_mask)
        
        # Pool sequence representation from final hidden states
        pooled = aggregate_sequence_representations(activations['final_hidden'], inputs.attention_mask, args.pooling)
        
        all_activations.append(pooled.cpu())
        sample_labels.append("chosen")
        
        if activations['all_attentions'] is not None:
            all_attention.append(activations['all_attentions'][-1].cpu())  # Final layer attention
        
        # Also process rejected response if available
        if "rejected" in sample:
            rejected = sample["rejected"]
            text_rejected = f"{prompt} {rejected}"
            
            inputs_rejected = tokenizer(text_rejected, return_tensors="pt", max_length=args.max_length, 
                                      truncation=True, padding=True).to(device)
            
            activations_rej = extract_activations(model, inputs_rejected.input_ids, 
                                                 inputs_rejected.attention_mask)
            
            pooled_rej = aggregate_sequence_representations(activations_rej['final_hidden'], inputs_rejected.attention_mask, args.pooling)
            
            all_activations.append(pooled_rej.cpu())
            sample_labels.append("rejected")
            
            if activations_rej['all_attentions'] is not None:
                all_attention.append(activations_rej['all_attentions'][-1].cpu())
    
    print("\nGenerating visualizations...")
    
    # Create visualizations
    activation_save_path = output_dir / f"{args.model.replace('/', '_')}_activations.png"
    visualize_activations(all_activations, sample_labels, args.model, str(activation_save_path))
    
    if all_attention:
        attention_save_path = output_dir / f"{args.model.replace('/', '_')}_attention.png"
        visualize_attention_patterns(all_attention, args.model, str(attention_save_path), model_type)
    
    # Save analysis summary
    activations_tensor = torch.cat(all_activations, dim=0)
    
    summary = {
        "model": args.model,
        "model_type": model_type,
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
    
    # Optional: Add comparison with different model type
    if args.compare_model:
        print(f"\nRunning comparison with {args.compare_model}...")
        compare_model, compare_tokenizer, compare_model_type = load_model(args.compare_model, device)
        
        if compare_model_type != model_type:
            print(f"Comparing {model_type} vs {compare_model_type} models")
            
            # Process a subset of samples with comparison model
            compare_activations = []
            compare_attention = []
            
            for i, sample in enumerate(samples[:10]):  # Just first 10 for comparison
                prompt = sample.get("prompt", "")
                chosen = sample.get("chosen", "")
                text = f"{prompt} {chosen}"
                
                inputs = compare_tokenizer(text, return_tensors="pt", max_length=args.max_length, 
                                         truncation=True, padding=True).to(device)
                
                activations = extract_activations(compare_model, inputs.input_ids, inputs.attention_mask)
                pooled = aggregate_sequence_representations(activations['final_hidden'], inputs.attention_mask, args.pooling)
                
                compare_activations.append(pooled.cpu())
                if activations['all_attentions'] is not None:
                    compare_attention.append(activations['all_attentions'][-1].cpu())
            
            # Create comparison visualizations
            if compare_attention:
                compare_attention_path = output_dir / f"{args.compare_model.replace('/', '_')}_attention_comparison.png"
                visualize_attention_patterns(compare_attention, args.compare_model, 
                                           str(compare_attention_path), compare_model_type)
                
                # Create poster-ready comparison if we have both types
                if model_type != compare_model_type:
                    poster_path = output_dir / "poster_ready_comparison.png"
                    if model_type == "bidirectional":
                        create_poster_comparison_visualization(all_attention, compare_attention,
                                                             args.model, args.compare_model, str(poster_path))
                    else:
                        create_poster_comparison_visualization(compare_attention, all_attention,
                                                             args.compare_model, args.model, str(poster_path))
    
    print(f"\nAnalysis complete! Results saved to: {output_dir}")
    print(f"Summary: {summary_path}")
    
    # Print key insights
    print("\n=== Key Insights ===")
    print(f"Model: {args.model} ({model_type})")
    print(f"Activation dimensionality: {activations_tensor.shape[1]}")
    print(f"Mean activation: {summary['activation_stats']['mean']:.4f}")
    print(f"Activation std: {summary['activation_stats']['std']:.4f}")
    
    if "chosen_vs_rejected" in summary:
        print(f"Chosen vs Rejected mean difference: {summary['chosen_vs_rejected']['mean_difference']:.4f}")
        print(f"Chosen vs Rejected cosine similarity: {summary['chosen_vs_rejected']['cosine_similarity']:.4f}")
    
    print(f"\n=== Usage Instructions ===\n")
    print(f"To run comparison with decoder model (e.g., GPT-2):")
    print(f"python {sys.argv[0]} --model {args.model} --compare_model gpt2")
    print(f"\nTo run with different subsets:")
    print(f"python {sys.argv[0]} --model {args.model} --subset safety")

if __name__ == "__main__":
    main()