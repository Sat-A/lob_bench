# Test Mode Guide for run_bench.py

## Overview

A new **test mode** has been added to `run_bench.py` that allows you to test individual metrics in isolation. This is useful for:
- Debugging specific metrics before running full benchmarks
- Validating that new metrics work correctly
- Generating plots for individual metrics
- Quick validation of metric implementations

## Basic Usage

### Test the time_lagged_evals metric:
```bash
python run_bench.py --test time_lagged_evals --stock GOOG --model_name s5_main
```

### Test any other metric:
```bash
python run_bench.py --test spread --stock GOOG --model_name s5_main
python run_bench.py --test orderbook_imbalance --stock INTC --model_name s5v2_uncond
python run_bench.py --test log_inter_arrival_time --stock GOOG --model_name s5_main
```

### Test with custom parameters:
```bash
python run_bench.py --test time_lagged_evals \
    --stock GOOG \
    --model_name s5_main \
    --time_period 2023_Jan \
    --data_dir ./data/data/evalsequences \
    --save_dir ./results
```

## What the Test Mode Does

When you run `--test METRIC_NAME`, the `test_metric()` function:

1. **Validates** the metric exists in `DEFAULT_SCORING_CONFIG`
   - Lists all available metrics if the requested one is not found

2. **Loads** only the specified metric's data
   - Efficiently loads real and generated sequences
   - Materializes them into memory

3. **Computes** the metric using the full scoring pipeline
   - Runs `scoring.run_benchmark()` with just that metric
   - Shows all errors immediately with full traceback

4. **Displays** comprehensive results:
   - Point estimate and 95% confidence intervals
   - Bootstrap sample count
   - Per-type statistics (real vs generated)
   - Min/max/mean comparisons

5. **Generates** visualization plots
   - Histograms comparing real vs generated distributions
   - Saved to `./results/plots/test_<metric>_<stock>_<timestamp>.png`

6. **Saves** results
   - Full score data and dataframes
   - Saved to `./results/scores/test_<metric>_<stock>_<timestamp>.pkl`

## Example Output

```
======================================================================
TESTING METRIC: time_lagged_evals
======================================================================

[*] Loading data from: ./data/data/evalsequences/s5_main/GOOG/2023_Jan
[*] Materializing sequences...
[*] Running metric: time_lagged_evals

======================================================================
RESULTS FOR time_lagged_evals
======================================================================

Metric scores computed:

  wasserstein:
    Point estimate: 2.456321
    CI [95%]: [2.341523, 2.567890]
    Bootstrap samples: 101

Score DataFrame:
  Shape: (400, 4)
  Columns: ['score', 'group', 'type', 'index']
  Real samples: 200
  Generated samples: 200

  Real scores - Min: 0.234567, Max: 3.456789, Mean: 1.234567
  Gen scores  - Min: 0.456789, Max: 5.123456, Mean: 2.123456

[*] Generating plots...
  ✓ Plot saved to: ./results/plots/test_time_lagged_evals_GOOG_20260131_153042.png

[*] Saving test results...
  ✓ Results saved to: ./results/scores/test_time_lagged_evals_GOOG_20260131_153042.pkl

======================================================================
TEST COMPLETE: time_lagged_evals
======================================================================
```

## Implementation Details

### New Function: `test_metric()`

Located in `run_bench.py` before the `if __name__ == "__main__":` block.

**Signature:**
```python
def test_metric(
    metric_name: str,
    stock: str = "GOOG",
    model_name: str = "s5_main",
    time_period: str = "2023_Jan",
    data_dir: str = "./data/data/evalsequences",
    save_dir: str = "./results",
) -> None:
```

**Parameters:**
- `metric_name`: Name of the metric to test (required)
- `stock`: Stock symbol (default: "GOOG")
- `model_name`: Model name (default: "s5_main")
- `time_period`: Time period (default: "2023_Jan")
- `data_dir`: Path to data directory
- `save_dir`: Path to save results

### New Argument: `--test`

Added to the argument parser in the `if __name__ == "__main__":` block.

```python
parser.add_argument("--test", type=str, default=None,
                    help="Test a specific metric (e.g., --test time_lagged_evals)")
```

## Interpreting Results for time_lagged_evals

### Wasserstein Distance (Point Estimate)

The Wasserstein distance compares the distribution of drift scores between real and generated sequences:

- **Low value (e.g., 0.5-1.0)**: Real and generated have similar drift patterns ✓
  - Model maintains consistency
  
- **Medium value (e.g., 1.5-3.0)**: Moderate divergence in drift patterns
  - Some degradation but within acceptable range
  
- **High value (e.g., > 3.0)**: Generated shows significantly different drift ✗
  - Model degradation detected
  - Long-horizon predictions may be unreliable

### Real vs Generated Comparison

Compare the score statistics:

- **Real scores mean ≈ Generated scores mean**: Model maintains similar drift ✓
- **Generated scores mean >> Real scores mean**: Model shows more variability ✗
- **Real scores std >> Generated scores std**: Model is more stable but may be too smooth
- **Generated scores std >> Real scores std**: Model has high variability (drift detected)

### Confidence Intervals

- **Narrow CI**: Consistent results across bootstrap samples (stable metric)
- **Wide CI**: High variability in metric (may indicate noisy data or unstable model)

## Available Metrics for Testing

All metrics in `DEFAULT_SCORING_CONFIG` can be tested:

1. spread
2. orderbook_imbalance
3. log_inter_arrival_time
4. log_time_to_cancel
5. ask_volume_touch
6. bid_volume_touch
7. ask_volume
8. bid_volume
9. limit_ask_order_depth
10. limit_bid_order_depth
11. ask_cancellation_depth
12. bid_cancellation_depth
13. limit_ask_order_levels
14. limit_bid_order_levels
15. ask_cancellation_levels
16. bid_cancellation_levels
17. vol_per_min
18. ofi
19. ofi_up
20. ofi_stay
21. ofi_down
22. time_lagged_evals (NEW)

## Quick Reference

| Task | Command |
|------|---------|
| Test time_lagged_evals | `python run_bench.py --test time_lagged_evals --stock GOOG --model_name s5_main` |
| List all metrics | View the output when metric not found, or check `DEFAULT_SCORING_CONFIG` |
| Test with INTC | `python run_bench.py --test time_lagged_evals --stock INTC --model_name s5_main` |
| Custom time period | Add `--time_period 2023_Jan` to command |
| View saved plots | Check `./results/plots/test_*.png` |
| Load results programmatically | `pickle.load(open('./results/scores/test_*.pkl'))` |

## Troubleshooting

### Metric not found
If you see "not found in DEFAULT_SCORING_CONFIG", check the spelling and list available metrics by running the test with an invalid name.

### Data not found
Ensure the data path exists: `./data/data/evalsequences/{model_name}/{stock}/{time_period}/`

### Plot generation failed
Check that matplotlib is installed and the `./results/plots/` directory is writable.

### Out of memory
For large datasets, test with just a subset or check available RAM.

## Integration with Full Benchmark

Test mode is **independent** from the full benchmark runs:
- Does not interfere with `--uncond_only`, `--cond_only`, `--context_only`, or `--div_only`
- Exits immediately after test (no full benchmark run)
- Can be used before running full benchmarks to validate metrics

## Next Steps

1. **Run a test**: `python run_bench.py --test time_lagged_evals --stock GOOG --model_name s5_main`
2. **Review results**: Check printed output and generated plots
3. **Validate metric**: Verify that drift patterns make sense
4. **Run full benchmark** (if satisfied): `python run_bench.py --uncond_only --stock GOOG --model_name s5_main`
