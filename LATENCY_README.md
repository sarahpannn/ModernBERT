# Latency Analysis Tools

Scripts for comprehensive latency analysis and comparison of ModernBERT models.

## Overview

This toolkit provides:
- **Detailed latency measurements** with statistical analysis (mean, median, P95, P99)
- **Throughput metrics** (tokens/second, samples/second)
- **Memory usage tracking** (peak and allocated memory)
- **GPU power monitoring** (if available)
- **Multi-model comparison** with side-by-side analysis
- **Rich formatted output** with tables and rankings

## Files

- `latency_analysis.py` - Main benchmarking script
- `latency_config.yaml` - Configuration file for multiple models
- `analyze_latency_results.py` - Results analysis and comparison script

## Quick Start

### 1. Basic Usage

Run latency analysis on a single FlexBERT model:

```bash
python latency_analysis.py
```

### 2. Compare Multiple Models

Edit `latency_config.yaml` to specify models to compare, then run:

```bash
python latency_analysis.py --config-file latency_config.yaml --output-file my_results.csv
```

### 3. Analyze Results

```bash
python analyze_latency_results.py my_results.csv
```

## Configuration

### Model Configuration

In `latency_config.yaml`, specify models to test:

```yaml
models:
  - name: "flex-bert-base"
    model_type: "flex_bert"          # flex_bert, mosaic_bert, hf_bert
    task_type: "classification"      # classification, mlm
    pretrained_model_name: "bert-base-uncased"
    num_labels: 2
    model_config: {}
    
  - name: "custom-model"
    model_type: "flex_bert"
    task_type: "classification"
    pretrained_checkpoint: "path/to/checkpoint.pt"
    model_config:
      hidden_size: 768
      num_attention_heads: 12
```

### Command Line Options

```bash
python latency_analysis.py \
  --config-file latency_config.yaml \
  --output-file results.csv \
  --batch-size 8 \
  --sequence-length 512 \
  --num-samples 1000 \
  --warmup-batches 10 \
  --measure-batches 100 \
  --device cuda
```

## Metrics Collected

### Latency Metrics
- **Mean latency** - Average inference time per batch
- **Median latency** - 50th percentile latency
- **P95/P99 latency** - 95th/99th percentile latencies
- **Standard deviation** - Latency consistency measure
- **Min/Max latency** - Range measurements

### Throughput Metrics
- **Tokens per second** - Processing throughput
- **Samples per second** - Batch processing rate

### Resource Metrics
- **Peak memory** - Maximum GPU memory usage
- **Allocated memory** - Current memory allocation
- **GPU power consumption** - Average and peak power usage (if available)

## Analysis Features

The analysis script provides:

### Performance Rankings
- 🏃 **Fastest models** by mean latency
- ⚡ **Highest throughput** models
- 💾 **Most memory efficient** models
- 📊 **Most consistent** models (lowest std dev)

### Efficiency Metrics
- **Latency per token** - Normalized latency comparison
- **Memory per token** - Memory efficiency comparison

### Model Type Comparisons
- Average performance statistics by model architecture
- Standard deviation across model types

### Pairwise Comparisons
- Direct model-to-model speedup calculations
- Memory usage ratios
- Performance trade-off analysis

## Example Output

```
                           Latency Analysis Results
┏━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━━━━┓
┃ Model           ┃ Type        ┃ Batch Size ┃ Seq Len ┃ Mean (ms) ┃ P95 (ms)  ┃ P99 (ms)  ┃ Tokens/s   ┃ Memory (MB)  ┃
┡━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━━━━┩
│ flex-bert-base  │ flex_bert   │          8 │     512 │     45.23 │     52.10 │     58.34 │       73580 │        1234.5 │
│ mosaic-bert     │ mosaic_bert │          8 │     512 │     48.91 │     55.78 │     62.15 │       68390 │        1156.7 │
│ hf-bert-base    │ hf_bert     │          8 │     512 │     52.67 │     59.22 │     65.88 │       61820 │        1445.2 │
└─────────────────┴─────────────┴────────────┴─────────┴───────────┴───────────┴───────────┴────────────┴──────────────┘
```

## Tips for Accurate Measurements

1. **Warmup is crucial** - Always include warmup batches to ensure accurate timing
2. **Multiple runs** - Run with different batch sizes and sequence lengths
3. **GPU memory** - Clear CUDA cache between model comparisons
4. **Consistent environment** - Use the same hardware and software setup
5. **Statistical significance** - Use enough measurement batches for stable statistics

## Troubleshooting

### Common Issues

1. **CUDA out of memory**: Reduce batch size or sequence length
2. **Missing dependencies**: Install required packages from requirements.txt
3. **Model loading errors**: Check model configuration and checkpoint paths
4. **GPU power monitoring unavailable**: Install `pynvml` for power measurements

### Performance Tips

- Use `torch.compile()` for faster inference (PyTorch 2.0+)
- Enable mixed precision with `torch.autocast()`
- Consider using `torch.jit.script()` for production deployments
- Profile with different sequence lengths to find optimal operating points

## Integration with Existing Scripts

This latency analysis complements the existing `benchmark.py` script:

- `benchmark.py` - Training and general benchmarking
- `latency_analysis.py` - Detailed inference latency analysis
- Both scripts share similar infrastructure and can be used together for comprehensive performance evaluation