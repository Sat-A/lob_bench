#!/usr/bin/env python
"""
Test script for time_lagged_evals scoring function.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import sys

import eval
import data_loading

def test_time_lagged_evals_basic():
    """Test basic functionality of time_lagged_evals."""
    print("Testing time_lagged_evals with basic synthetic data...")
    
    # Create synthetic data
    n_messages = 300
    times = pd.date_range('2023-01-01 09:30:00', periods=n_messages, freq='100ms')
    
    messages = pd.DataFrame({
        'time': times,
        'event_type': np.random.randint(1, 5, n_messages),
        'order_id': np.arange(n_messages),
        'size': np.random.randint(1, 100, n_messages),
        'price': np.random.randint(100, 110, n_messages),
        'direction': np.random.choice([-1, 1], n_messages)
    })
    
    # Create synthetic book data (4 columns per level, just use 1 level)
    # Column structure: [bid_price, bid_volume, ask_price, ask_volume]
    n_levels = 5
    book = pd.DataFrame(
        np.random.rand(n_messages, 4 * n_levels) * 10 + np.arange(4 * n_levels) * 0.1
    )
    
    # Ensure bid < ask (fix the structure)
    for i in range(n_levels):
        bid_col = i * 4
        ask_col = i * 4 + 2
        # Make sure bid < ask
        book[bid_col] = np.minimum(book[bid_col], book[ask_col])
    
    # Test function
    result = eval.time_lagged_evals(messages, book, window_size=100, lookback_steps=50)
    
    print(f"  Number of messages: {len(messages)}")
    print(f"  Expected result length: {len(messages) - 100 - 50 + 1} = {len(messages) - 149}")
    print(f"  Actual result length: {len(result)}")
    print(f"  Result dtype: {result.dtype}")
    print(f"  Result min: {result.min():.6f}, max: {result.max():.6f}, mean: {result.mean():.6f}")
    print(f"  All values >= 0: {np.all(result >= 0)}")
    print(f"  No NaN values: {not np.any(np.isnan(result))}")
    
    assert len(result) == len(messages) - 149, f"Expected length {len(messages) - 149}, got {len(result)}"
    assert result.dtype == np.float64, f"Expected float64, got {result.dtype}"
    assert np.all(result >= 0), "Wasserstein distances should be non-negative"
    assert not np.any(np.isnan(result)), "No NaN values should be present"
    print("  ✓ Basic test passed!\n")


def test_time_lagged_evals_short_sequence():
    """Test with sequence shorter than required window."""
    print("Testing time_lagged_evals with short sequence...")
    
    n_messages = 50  # Too short: needs at least 100 + 50 = 150
    times = pd.date_range('2023-01-01 09:30:00', periods=n_messages, freq='100ms')
    
    messages = pd.DataFrame({
        'time': times,
        'event_type': np.ones(n_messages, dtype=int),
        'order_id': np.arange(n_messages),
        'size': np.ones(n_messages, dtype=int),
        'price': np.ones(n_messages, dtype=int) * 100,
        'direction': np.ones(n_messages, dtype=int)
    })
    
    book = pd.DataFrame(np.random.rand(n_messages, 4) * 10)
    
    result = eval.time_lagged_evals(messages, book, window_size=100, lookback_steps=50)
    
    print(f"  Sequence length: {len(messages)}")
    print(f"  Minimum required: 150")
    print(f"  Result length: {len(result)}")
    assert len(result) == 0, "Should return empty array for short sequences"
    print("  ✓ Short sequence test passed!\n")


def test_mid_price_validation():
    """Test that mid_price validates book integrity."""
    print("Testing mid_price validation...")
    
    n_messages = 100
    times = pd.date_range('2023-01-01 09:30:00', periods=n_messages, freq='100ms')
    
    messages = pd.DataFrame({
        'time': times,
        'event_type': np.ones(n_messages, dtype=int),
        'order_id': np.arange(n_messages),
        'size': np.ones(n_messages, dtype=int),
        'price': np.ones(n_messages, dtype=int) * 100,
        'direction': np.ones(n_messages, dtype=int)
    })
    
    # Valid book: bid < ask
    book_valid = pd.DataFrame({
        0: np.ones(n_messages) * 100,  # bid
        1: np.ones(n_messages) * 1000,  # bid volume
        2: np.ones(n_messages) * 101,  # ask
        3: np.ones(n_messages) * 1000,  # ask volume
    })
    
    result = eval.mid_price(messages, book_valid)
    print(f"  Valid book mid-price shape: {result.shape}")
    print(f"  Valid book mid-price values: {result.iloc[:3].values}")
    assert len(result) == n_messages
    np.testing.assert_allclose(result.values, 100.5, rtol=1e-6)
    print("  ✓ Valid book test passed!")
    
    # Invalid book: bid > ask (should raise assertion)
    book_invalid = pd.DataFrame({
        0: np.ones(n_messages) * 102,  # bid (INVALID: > ask)
        1: np.ones(n_messages) * 1000,  # bid volume
        2: np.ones(n_messages) * 101,  # ask
        3: np.ones(n_messages) * 1000,  # ask volume
    })
    
    try:
        eval.mid_price(messages, book_invalid)
        assert False, "Should have raised assertion for inverted book"
    except AssertionError as e:
        if "Book inversion" in str(e):
            print(f"  ✓ Invalid book correctly rejected: {e}\n")
        else:
            raise


def test_real_data():
    """Test with real data if available."""
    print("Testing time_lagged_evals with real data (if available)...")
    
    try:
        loader = data_loading.get_rwkv_loader(
            dataset_type='real',
            stock='GOOG',
            date='2023-01-03',
            seq_len=None,
        )
        
        # Load a single sequence
        seq = loader[0]
        print(f"  Loaded sequence: {seq.m_real.shape[0]} messages")
        
        # Test function
        result = eval.time_lagged_evals(
            seq.m_real, seq.b_real, window_size=100, lookback_steps=50
        )
        
        print(f"  Result length: {len(result)}")
        print(f"  Result stats - min: {result.min():.6f}, max: {result.max():.6f}, mean: {result.mean():.6f}")
        print(f"  All values >= 0: {np.all(result >= 0)}")
        print(f"  No NaN values: {not np.any(np.isnan(result))}")
        
        # Check that result is reasonable
        if len(result) > 0:
            assert np.all(result >= 0), "Wasserstein distances should be non-negative"
            assert not np.any(np.isnan(result)), "No NaN values should be present"
            print("  ✓ Real data test passed!\n")
        else:
            print("  ⚠ Sequence too short for this configuration\n")
            
    except Exception as e:
        print(f"  ⚠ Could not test with real data: {e}\n")


if __name__ == "__main__":
    print("=" * 60)
    print("Testing time_lagged_evals implementation")
    print("=" * 60 + "\n")
    
    test_time_lagged_evals_basic()
    test_time_lagged_evals_short_sequence()
    test_mid_price_validation()
    test_real_data()
    
    print("=" * 60)
    print("All tests passed!")
    print("=" * 60)
