#!/usr/bin/env python3
"""
Example scripts to run the single model analysis on different models and datasets.
"""

import subprocess
import sys

def run_analysis(model_name: str, subset: str = "reasoning", num_samples: int = 30):
    """Run analysis for a specific model and dataset subset."""
    
    cmd = [
        sys.executable, "single_model_analysis.py",
        "--model", model_name,
        "--subset", subset,
        "--num_samples", str(num_samples),
        "--max_length", "8192",  # Reduced for memory efficiency
        "--pooling", "mean",
        "--output_dir", f"./analysis_{model_name.replace('/', '_')}_{subset}",
        "--device", "auto"
    ]
    
    print(f"\n{'='*60}")
    print(f"Running analysis for {model_name} on {subset} subset")
    print(f"{'='*60}")
    print("Command:", " ".join(cmd))
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("✅ Analysis completed successfully!")
        print("Key output:", result.stdout.split("=== Key Insights ===")[-1] if "=== Key Insights ===" in result.stdout else "")
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running analysis: {e}")
        print("STDERR:", e.stderr[-500:])  # Last 500 chars of error

def main():
    """Run analysis on multiple models for comparison."""
    
    # Models to compare
    models = [
        "answerdotai/ModernBERT-base",  # Your small bidirectional model
        "gpt2",                        # Small decoder model
        # "bert-base-uncased",           # Standard BERT for comparison
    ]
    
    # Dataset subsets to analyze
    subsets = ["reasoning", "chat"]
    
    print("🔍 Starting model activation analysis...")
    print(f"Will analyze {len(models)} models on {len(subsets)} subsets")
    
    for model in models:
        for subset in subsets:
            try:
                run_analysis(model, subset, num_samples=20)  # Small number for memory efficiency
            except Exception as e:
                print(f"Failed to analyze {model} on {subset}: {e}")
                continue
    
    print("\n🎉 All analyses complete!")
    print("\nGenerated directories:")
    print("- analysis_answerdotai_ModernBERT-base_reasoning/")
    print("- analysis_answerdotai_ModernBERT-base_chat/") 
    print("- analysis_gpt2_reasoning/")
    print("- analysis_gpt2_chat/")
    # print("- analysis_bert-base-uncased_reasoning/")
    # print("- analysis_bert-base-uncased_chat/")
    
    print("\nFor your poster, compare:")
    print("1. ModernBERT vs GPT-2 activation patterns")
    print("2. Chosen vs rejected response representations")
    print("3. Attention patterns differences")
    print("4. Activation magnitude and distribution differences")

if __name__ == "__main__":
    main()