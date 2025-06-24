#!/usr/bin/env python3
"""
Latency Analysis Script for ModernBERT Models

This script performs detailed latency analysis on different BERT model architectures,
providing comprehensive timing statistics and comparison metrics.
"""

import time
import json
import statistics
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict

import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer
import typer
from rich.console import Console
from rich.table import Table
from rich.progress import track
from omegaconf import OmegaConf

from src.flex_bert import create_flex_bert_classification, create_flex_bert_mlm
from src.mosaic_bert import create_mosaic_bert_classification, create_mosaic_bert_mlm
from src.hf_bert import create_hf_bert_classification, create_hf_bert_mlm

try:
    import pynvml
    pynvml.nvmlInit()
    GPU_AVAILABLE = True
except:
    GPU_AVAILABLE = False

console = Console()

@dataclass
class LatencyMetrics:
    """Container for latency measurement results"""
    model_name: str
    model_type: str
    batch_size: int
    sequence_length: int
    num_samples: int
    
    # Timing metrics (in milliseconds)
    mean_latency: float
    median_latency: float
    std_latency: float
    min_latency: float
    max_latency: float
    p95_latency: float
    p99_latency: float
    
    # Throughput metrics
    tokens_per_second: float
    samples_per_second: float
    
    # Memory metrics (in MB)
    peak_memory_mb: float
    allocated_memory_mb: float
    
    # GPU metrics (if available)
    avg_gpu_power_w: Optional[float] = None
    max_gpu_power_w: Optional[float] = None

class SimpleTextDataset(Dataset):
    """Simple dataset for latency testing"""
    
    def __init__(self, texts: List[str], tokenizer, max_length: int = 512):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze()
        }

def generate_test_texts(num_samples: int, seq_length: int) -> List[str]:
    """Generate test texts of approximately the given sequence length"""
    # Rough estimate: ~4 chars per token for English text
    chars_per_text = seq_length * 4
    
    base_text = "This is a sample sentence for testing model inference latency. " * 20
    texts = []
    
    for _ in range(num_samples):
        # Vary text length slightly around target
        target_chars = chars_per_text + np.random.randint(-50, 50)
        if target_chars > len(base_text):
            # Repeat and truncate
            repeats = (target_chars // len(base_text)) + 1
            text = (base_text * repeats)[:target_chars]
        else:
            text = base_text[:target_chars]
        texts.append(text)
    
    return texts

def load_model(model_config: Dict[str, Any], device: torch.device):
    """Load model based on configuration"""
    model_type = model_config.get('model_type', 'flex_bert')
    task_type = model_config.get('task_type', 'classification')
    
    console.print(f"Loading {model_type} model for {task_type}...")
    
    if model_type == 'flex_bert':
        if task_type == 'mlm':
            model = create_flex_bert_mlm(
                pretrained_model_name=model_config.get('pretrained_model_name', 'bert-base-uncased'),
                model_config=model_config.get('model_config', {}),
                pretrained_checkpoint=model_config.get('pretrained_checkpoint')
            )
        else:
            model = create_flex_bert_classification(
                pretrained_model_name=model_config.get('pretrained_model_name', 'bert-base-uncased'),
                model_config=model_config.get('model_config', {}),
                pretrained_checkpoint=model_config.get('pretrained_checkpoint'),
                num_labels=model_config.get('num_labels', 2)
            )
    elif model_type == 'mosaic_bert':
        if task_type == 'mlm':
            model = create_mosaic_bert_mlm(
                pretrained_model_name=model_config.get('pretrained_model_name', 'bert-base-uncased'),
                model_config=model_config.get('model_config', {}),
                pretrained_checkpoint=model_config.get('pretrained_checkpoint')
            )
        else:
            model = create_mosaic_bert_classification(
                pretrained_model_name=model_config.get('pretrained_model_name', 'bert-base-uncased'),
                model_config=model_config.get('model_config', {}),
                pretrained_checkpoint=model_config.get('pretrained_checkpoint'),
                num_labels=model_config.get('num_labels', 2)
            )
    elif model_type == 'hf_bert':
        if task_type == 'mlm':
            model = create_hf_bert_mlm(
                pretrained_model_name=model_config.get('pretrained_model_name', 'bert-base-uncased'),
                model_config=model_config.get('model_config', {}),
                pretrained_checkpoint=model_config.get('pretrained_checkpoint')
            )
        else:
            model = create_hf_bert_classification(
                pretrained_model_name=model_config.get('pretrained_model_name', 'bert-base-uncased'),
                model_config=model_config.get('model_config', {}),
                pretrained_checkpoint=model_config.get('pretrained_checkpoint'),
                num_labels=model_config.get('num_labels', 2)
            )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    model = model.to(device)
    model.eval()
    return model

def measure_gpu_power() -> Tuple[float, float]:
    """Measure current GPU power consumption"""
    if not GPU_AVAILABLE:
        return 0.0, 0.0
    
    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        power_mw = pynvml.nvmlDeviceGetPowerUsage(handle)
        max_power_mw = pynvml.nvmlDeviceGetMaxPowerManagement(handle)
        return power_mw / 1000.0, max_power_mw / 1000.0  # Convert to watts
    except:
        return 0.0, 0.0

def benchmark_model_latency(
    model,
    dataloader: DataLoader,
    device: torch.device,
    model_name: str,
    model_type: str,
    warmup_batches: int = 10,
    measure_batches: int = 100
) -> LatencyMetrics:
    """Benchmark model inference latency"""
    
    console.print(f"Benchmarking {model_name} ({model_type})...")
    
    # Get batch and sequence info
    first_batch = next(iter(dataloader))
    batch_size = first_batch['input_ids'].shape[0]
    sequence_length = first_batch['input_ids'].shape[1]
    
    # Warmup
    console.print("Warming up...")
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= warmup_batches:
                break
            
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            _ = model(input_ids=input_ids, attention_mask=attention_mask)
    
    # Clear cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    
    # Measure latency
    latencies = []
    power_measurements = []
    
    console.print("Measuring latency...")
    with torch.no_grad():
        for i, batch in enumerate(track(dataloader, description="Measuring...")):
            if i >= measure_batches:
                break
            
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            # Synchronize before timing
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            # Measure power before
            power_before, _ = measure_gpu_power()
            
            # Time inference
            start_time = time.perf_counter()
            _ = model(input_ids=input_ids, attention_mask=attention_mask)
            
            # Synchronize after timing
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            end_time = time.perf_counter()
            
            # Measure power after
            power_after, max_power = measure_gpu_power()
            
            latency_ms = (end_time - start_time) * 1000
            latencies.append(latency_ms)
            power_measurements.append((power_before + power_after) / 2)
    
    # Calculate memory usage
    if torch.cuda.is_available():
        peak_memory_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
        allocated_memory_mb = torch.cuda.memory_allocated() / 1024 / 1024
    else:
        peak_memory_mb = 0.0
        allocated_memory_mb = 0.0
    
    # Calculate statistics
    mean_latency = statistics.mean(latencies)
    median_latency = statistics.median(latencies)
    std_latency = statistics.stdev(latencies) if len(latencies) > 1 else 0.0
    min_latency = min(latencies)
    max_latency = max(latencies)
    p95_latency = np.percentile(latencies, 95)
    p99_latency = np.percentile(latencies, 99)
    
    # Calculate throughput
    tokens_per_batch = batch_size * sequence_length
    tokens_per_second = tokens_per_batch / (mean_latency / 1000)
    samples_per_second = batch_size / (mean_latency / 1000)
    
    # Power statistics
    avg_gpu_power = statistics.mean(power_measurements) if power_measurements else None
    max_gpu_power = max(power_measurements) if power_measurements else None
    
    return LatencyMetrics(
        model_name=model_name,
        model_type=model_type,
        batch_size=batch_size,
        sequence_length=sequence_length,
        num_samples=len(latencies),
        mean_latency=mean_latency,
        median_latency=median_latency,
        std_latency=std_latency,
        min_latency=min_latency,
        max_latency=max_latency,
        p95_latency=p95_latency,
        p99_latency=p99_latency,
        tokens_per_second=tokens_per_second,
        samples_per_second=samples_per_second,
        peak_memory_mb=peak_memory_mb,
        allocated_memory_mb=allocated_memory_mb,
        avg_gpu_power_w=avg_gpu_power,
        max_gpu_power_w=max_gpu_power
    )

def print_results_table(results: List[LatencyMetrics]):
    """Print formatted results table"""
    table = Table(title="Latency Analysis Results")
    
    table.add_column("Model", style="cyan")
    table.add_column("Type", style="magenta")
    table.add_column("Batch Size", justify="right")
    table.add_column("Seq Len", justify="right")
    table.add_column("Mean (ms)", justify="right", style="green")
    table.add_column("P95 (ms)", justify="right", style="yellow")
    table.add_column("P99 (ms)", justify="right", style="red")
    table.add_column("Tokens/s", justify="right", style="blue")
    table.add_column("Memory (MB)", justify="right")
    
    for result in results:
        table.add_row(
            result.model_name,
            result.model_type,
            str(result.batch_size),
            str(result.sequence_length),
            f"{result.mean_latency:.2f}",
            f"{result.p95_latency:.2f}",
            f"{result.p99_latency:.2f}",
            f"{result.tokens_per_second:.0f}",
            f"{result.peak_memory_mb:.1f}"
        )
    
    console.print(table)

def save_results(results: List[LatencyMetrics], output_file: str):
    """Save results to JSON and CSV files"""
    # Save to JSON
    json_data = [asdict(result) for result in results]
    json_file = output_file.replace('.csv', '.json')
    
    with open(json_file, 'w') as f:
        json.dump(json_data, f, indent=2)
    
    # Save to CSV
    df = pd.DataFrame(json_data)
    df.to_csv(output_file, index=False)
    
    console.print(f"Results saved to {output_file} and {json_file}")

def main(
    config_file: str = typer.Option("latency_config.yaml", help="Configuration file path"),
    output_file: str = typer.Option("latency_results.csv", help="Output file path"),
    batch_size: int = typer.Option(8, help="Batch size for inference"),
    sequence_length: int = typer.Option(512, help="Sequence length"),
    num_samples: int = typer.Option(1000, help="Number of test samples"),
    warmup_batches: int = typer.Option(10, help="Number of warmup batches"),
    measure_batches: int = typer.Option(100, help="Number of batches to measure"),
    device: str = typer.Option("auto", help="Device to use (cuda/cpu/auto)")
):
    """Run latency analysis on BERT models"""
    
    # Set device
    if device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)
    
    console.print(f"Using device: {device}")
    
    # Load configuration
    if Path(config_file).exists():
        config = OmegaConf.load(config_file)
        models_to_test = config.models
    else:
        console.print(f"Config file {config_file} not found. Using default models.")
        models_to_test = [
            {
                'name': 'flex-bert-base',
                'model_type': 'flex_bert',
                'task_type': 'classification',
                'pretrained_model_name': 'bert-base-uncased',
                'model_config': {}
            }
        ]
    
    # Generate test data
    console.print(f"Generating {num_samples} test samples with sequence length {sequence_length}...")
    test_texts = generate_test_texts(num_samples, sequence_length)
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    dataset = SimpleTextDataset(test_texts, tokenizer, sequence_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    # Run benchmarks
    results = []
    for model_config in models_to_test:
        try:
            model = load_model(model_config, device)
            
            metrics = benchmark_model_latency(
                model=model,
                dataloader=dataloader,
                device=device,
                model_name=model_config['name'],
                model_type=model_config['model_type'],
                warmup_batches=warmup_batches,
                measure_batches=measure_batches
            )
            
            results.append(metrics)
            
            # Clean up
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        except Exception as e:
            console.print(f"Error benchmarking {model_config['name']}: {str(e)}", style="red")
            continue
    
    # Display and save results
    if results:
        print_results_table(results)
        save_results(results, output_file)
    else:
        console.print("No successful benchmark results.", style="red")

if __name__ == "__main__":
    typer.run(main)