#!/usr/bin/env python3
"""
Example usage of the attention comparison script.
"""

import subprocess
import sys
import os

def run_comparison():
    """Run the attention comparison with example parameters."""
    
    # Run the comparison script with HuggingFace models
    cmd = [
        sys.executable, "attention_comparison_script.py",
        "--encoder_model", "answerdotai/ModernBERT-base",  # Or try "bert-base-uncased"
        "--decoder_model", "gpt2-medium",  # You can also try "gpt2-large" or "gpt2"
        "--output_dir", "./attention_analysis_results",
        "--max_length", "128",
        "--device", "auto"
    ]
    
    print("Running attention comparison...")
    print("Command:", " ".join(cmd))
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("STDOUT:", result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        print("Comparison completed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"Error running comparison: {e}")
        print("STDOUT:", e.stdout)
        print("STDERR:", e.stderr)

if __name__ == "__main__":
    run_comparison()