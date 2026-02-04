# Time-Lagged Evaluation: Root Cause Analysis and Best Practices

## Root Cause Analysis

### The Core Issue
Your time-lagged evaluation function was trying to analyze data with a time lag of 500 steps in situations where:
1. The sequences were shorter than 500 steps after accounting for the conditional prefix
2. Some datasets had no conditional sequence at all (empty `m_cond`/`b_cond`)

### Why It Failed

**Scenario**: Lag = 500, Conditional Sequence = 500 steps, Data = only a few hundred steps

1. You merge: `conditional (500) + data (300)` = 800 total steps
2. You remove conditional prefix (first 500): leaves 300 steps
3. You try to apply lag of 500: but only 300 steps remain
4. Result: No valid data window to evaluate → crashes

**Empty Conditional Case**: When conditional data doesn't exist
1. The code tried to concatenate `empty_conditional + data`
2. Pandas warned about concatenating with empty dataframes (FutureWarning)
3. Still produced results but with unexpected behavior

## Why the Original Error Message Was Unhelpful

```
ValueError: No valid time-lagged scores produced. 
Check: lag=500, sequence lengths, and whether data is too short. 
Got 0 real and 0 generated.
```

While technically correct, the error message:
- Didn't distinguish between "no conditional sequences" vs "data is too short"
- Crashed the entire pipeline instead of skipping just that configuration
- Gave no context about which score configuration failed

## The Solution: Defensive Programming

The fix implements three layers of defense:

### Layer 1: Handle Empty Conditionals (Merge Stage)
```python
if len(m_cond) == 0 or len(b_cond) == 0:
    # Use data without attempting merge
    m_merged = m_data.copy()
    cond_len = 0
```
This prevents unnecessary concatenation warnings and sets `cond_len=0` so the algorithm works correctly.

### Layer 2: Skip Invalid Sequences (Validation Stage)
```python
if lag >= len(m_after_cond) or lag >= len(b_after_cond):
    continue  # Skip this sequence, it's too short
```
This efficiently filters out sequences that can't produce valid lagged pairs.

### Layer 3: Graceful Degradation (Result Handling)
```python
if not scores_lagged_real or not scores_lagged_gen:
    return None, None  # Signal insufficient data
```
Instead of crashing, return `None` to signal that this configuration couldn't be evaluated.

### Layer 4: User-Friendly Feedback (Caller)
```python
if score_result[0] is None:
    print(f"  Skipping score '{score_name}' due to insufficient data...")
    continue
```
Skip the configuration and inform the user why it was skipped.

## When to Adjust Parameters

If you find scores being skipped, consider:

### Reduce the lag
```python
lag = 250  # Instead of 500
```
Smaller lags work with shorter data windows.

### Ensure sufficient conditional data
Make sure your `m_cond`/`b_cond` are being properly populated. If they're empty, either:
- Change how you load/split the data
- Use a different conditioning strategy
- Set `lag = 0` for no time-lagging

### Check data length
Ensure raw data is longer than `conditional_length + lag + minimum_eval_length`:
```python
# For lag=500, cond=500, need at least eval_length points
min_total = 500 + 500 + 100  # 1100 minimum
```

### Use multi-lag evaluation
Test multiple lags to find what works with your data:
```python
for lag in [50, 100, 200, 500]:
    # Run evaluation with different lags
```

## Best Practices for Time-Lagged Analysis

1. **Always validate data length**: Check that sequences are long enough before processing
2. **Log intermediate results**: Track how many sequences fail at each stage
3. **Make parameters adaptive**: Adjust lag based on available data length
4. **Provide context in errors**: Include which score/dataset failed and why
5. **Fail gracefully**: Skip individual configurations rather than crashing
6. **Document assumptions**: Note minimum required sequence lengths

## Expected Output After Fix

When running with the fix, you should see:

**Good case** (data is sufficient):
```
Calculating scores and metrics for: score_name
[No warnings, scores computed successfully]
```

**Insufficient data case**:
```
Calculating scores and metrics for: score_name
  Warning: Insufficient data for time-lagged analysis with lag=500. 
  Sequences may be too short relative to lag. Skipping this configuration.
  Skipping score 'score_name' due to insufficient data for time-lagged analysis.
```

This way:
- ✅ FutureWarnings are eliminated
- ✅ Script doesn't crash
- ✅ You get clear feedback on what was skipped and why
- ✅ Other score configurations continue to be evaluated
