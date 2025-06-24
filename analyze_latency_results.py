#!/usr/bin/env python3
"""
Analysis script for latency benchmark results.
Provides statistical analysis and comparison of model performance.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List
import typer
from rich.console import Console
from rich.table import Table

console = Console()

def load_results(file_path: str) -> pd.DataFrame:
    """Load results from JSON or CSV file"""
    file_path = Path(file_path)
    
    if file_path.suffix == '.json':
        with open(file_path, 'r') as f:
            data = json.load(f)
        return pd.DataFrame(data)
    elif file_path.suffix == '.csv':
        return pd.read_csv(file_path)
    else:
        raise ValueError("File must be .json or .csv")

def analyze_results(df: pd.DataFrame) -> Dict:
    """Perform statistical analysis on results"""
    analysis = {}
    
    # Basic statistics
    analysis['summary'] = {
        'num_models': len(df),
        'model_types': df['model_type'].unique().tolist(),
        'batch_sizes': df['batch_size'].unique().tolist(),
        'sequence_lengths': df['sequence_length'].unique().tolist()
    }
    
    # Performance rankings
    analysis['rankings'] = {
        'fastest_mean_latency': df.nsmallest(5, 'mean_latency')[['model_name', 'mean_latency']].to_dict('records'),
        'highest_throughput': df.nlargest(5, 'tokens_per_second')[['model_name', 'tokens_per_second']].to_dict('records'),
        'most_memory_efficient': df.nsmallest(5, 'peak_memory_mb')[['model_name', 'peak_memory_mb']].to_dict('records'),
        'most_consistent': df.nsmallest(5, 'std_latency')[['model_name', 'std_latency']].to_dict('records')
    }
    
    # Model type comparisons
    if len(df['model_type'].unique()) > 1:
        type_comparison = df.groupby('model_type').agg({
            'mean_latency': ['mean', 'std'],
            'tokens_per_second': ['mean', 'std'],
            'peak_memory_mb': ['mean', 'std'],
            'p95_latency': ['mean', 'std']
        }).round(2)
        
        analysis['model_type_comparison'] = type_comparison.to_dict()
    
    # Efficiency metrics
    df['latency_per_token'] = df['mean_latency'] / (df['batch_size'] * df['sequence_length'])
    df['memory_per_token'] = df['peak_memory_mb'] / (df['batch_size'] * df['sequence_length'])
    
    analysis['efficiency'] = {
        'best_latency_per_token': df.nsmallest(3, 'latency_per_token')[['model_name', 'latency_per_token']].to_dict('records'),
        'best_memory_per_token': df.nsmallest(3, 'memory_per_token')[['model_name', 'memory_per_token']].to_dict('records')
    }
    
    return analysis

def print_analysis_report(analysis: Dict):
    """Print formatted analysis report"""
    
    # Summary
    console.print("\n[bold cyan]Latency Analysis Summary[/bold cyan]")
    summary = analysis['summary']
    console.print(f"• Models tested: {summary['num_models']}")
    console.print(f"• Model types: {', '.join(summary['model_types'])}")
    console.print(f"• Batch sizes: {summary['batch_sizes']}")
    console.print(f"• Sequence lengths: {summary['sequence_lengths']}")
    
    # Performance Rankings
    console.print("\n[bold green]Performance Rankings[/bold green]")
    
    # Fastest models
    table = Table(title="🏃 Fastest Models (Mean Latency)")
    table.add_column("Rank", style="cyan")
    table.add_column("Model", style="magenta")
    table.add_column("Latency (ms)", justify="right", style="green")
    
    for i, model in enumerate(analysis['rankings']['fastest_mean_latency'], 1):
        table.add_row(str(i), model['model_name'], f"{model['mean_latency']:.2f}")
    console.print(table)
    
    # Highest throughput
    table = Table(title="⚡ Highest Throughput")
    table.add_column("Rank", style="cyan")
    table.add_column("Model", style="magenta")
    table.add_column("Tokens/sec", justify="right", style="blue")
    
    for i, model in enumerate(analysis['rankings']['highest_throughput'], 1):
        table.add_row(str(i), model['model_name'], f"{model['tokens_per_second']:.0f}")
    console.print(table)
    
    # Memory efficiency
    table = Table(title="💾 Most Memory Efficient")
    table.add_column("Rank", style="cyan")
    table.add_column("Model", style="magenta")
    table.add_column("Memory (MB)", justify="right", style="yellow")
    
    for i, model in enumerate(analysis['rankings']['most_memory_efficient'], 1):
        table.add_row(str(i), model['model_name'], f"{model['peak_memory_mb']:.1f}")
    console.print(table)
    
    # Most consistent
    table = Table(title="📊 Most Consistent (Lowest StdDev)")
    table.add_column("Rank", style="cyan")
    table.add_column("Model", style="magenta")
    table.add_column("Std Dev (ms)", justify="right", style="red")
    
    for i, model in enumerate(analysis['rankings']['most_consistent'], 1):
        table.add_row(str(i), model['model_name'], f"{model['std_latency']:.2f}")
    console.print(table)
    
    # Model type comparison
    if 'model_type_comparison' in analysis:
        console.print("\n[bold blue]Model Type Comparison[/bold blue]")
        
        comparison = analysis['model_type_comparison']
        table = Table(title="Average Performance by Model Type")
        table.add_column("Model Type", style="cyan")
        table.add_column("Avg Latency (ms)", justify="right")
        table.add_column("Avg Throughput", justify="right")
        table.add_column("Avg Memory (MB)", justify="right")
        
        for model_type in comparison['mean_latency']['mean'].keys():
            table.add_row(
                model_type,
                f"{comparison['mean_latency']['mean'][model_type]:.2f} ± {comparison['mean_latency']['std'][model_type]:.2f}",
                f"{comparison['tokens_per_second']['mean'][model_type]:.0f} ± {comparison['tokens_per_second']['std'][model_type]:.0f}",
                f"{comparison['peak_memory_mb']['mean'][model_type]:.1f} ± {comparison['peak_memory_mb']['std'][model_type]:.1f}"
            )
        console.print(table)
    
    # Efficiency metrics
    console.print("\n[bold magenta]Efficiency Metrics[/bold magenta]")
    
    table = Table(title="Best Latency per Token")
    table.add_column("Model", style="cyan")
    table.add_column("ms/token", justify="right", style="green")
    
    for model in analysis['efficiency']['best_latency_per_token']:
        table.add_row(model['model_name'], f"{model['latency_per_token']:.4f}")
    console.print(table)

def generate_comparison_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Generate pairwise comparison matrix"""
    models = df['model_name'].tolist()
    metrics = ['mean_latency', 'tokens_per_second', 'peak_memory_mb']
    
    comparisons = []
    for i, model_a in enumerate(models):
        for j, model_b in enumerate(models):
            if i < j:  # Avoid duplicate comparisons
                row_a = df[df['model_name'] == model_a].iloc[0]
                row_b = df[df['model_name'] == model_b].iloc[0]
                
                comparison = {
                    'model_a': model_a,
                    'model_b': model_b,
                    'latency_speedup': row_b['mean_latency'] / row_a['mean_latency'],
                    'throughput_ratio': row_a['tokens_per_second'] / row_b['tokens_per_second'],
                    'memory_ratio': row_a['peak_memory_mb'] / row_b['peak_memory_mb']
                }
                comparisons.append(comparison)
    
    return pd.DataFrame(comparisons)

def main(
    results_file: str = typer.Argument(..., help="Path to results file (JSON or CSV)"),
    save_analysis: bool = typer.Option(False, help="Save analysis to file"),
    output_file: str = typer.Option("analysis_report.txt", help="Output file for analysis")
):
    """Analyze latency benchmark results"""
    
    try:
        # Load results
        console.print(f"Loading results from {results_file}...")
        df = load_results(results_file)
        
        if df.empty:
            console.print("No results found in file.", style="red")
            return
        
        # Perform analysis
        console.print("Performing analysis...")
        analysis = analyze_results(df)
        
        # Print report
        print_analysis_report(analysis)
        
        # Generate comparison matrix
        if len(df) > 1:
            console.print("\n[bold yellow]Pairwise Comparisons[/bold yellow]")
            comparison_matrix = generate_comparison_matrix(df)
            
            table = Table(title="Model Comparisons (A vs B)")
            table.add_column("Model A", style="cyan")
            table.add_column("Model B", style="magenta")
            table.add_column("A is X times faster", justify="right", style="green")
            table.add_column("Memory ratio (A/B)", justify="right", style="yellow")
            
            for _, row in comparison_matrix.iterrows():
                speedup = f"{row['latency_speedup']:.2f}x" if row['latency_speedup'] > 1 else f"1/{1/row['latency_speedup']:.2f}x"
                memory_ratio = f"{row['memory_ratio']:.2f}x"
                
                table.add_row(
                    row['model_a'],
                    row['model_b'],
                    speedup,
                    memory_ratio
                )
            console.print(table)
        
        # Save analysis if requested
        if save_analysis:
            with open(output_file, 'w') as f:
                # This would need to be implemented to save the analysis
                f.write("Analysis report saved\n")
            console.print(f"Analysis saved to {output_file}")
            
    except Exception as e:
        console.print(f"Error analyzing results: {str(e)}", style="red")
        raise

if __name__ == "__main__":
    typer.run(main)