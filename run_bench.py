from typing import Any,Union
import numpy as np
import pandas as pd
import pathlib
from glob import glob
import pickle
import gzip
import argparse
from datetime import datetime

import data_loading
import scoring
import eval
import metrics


import time

###################### UNCONDITIONAL SCORING ########################
DEFAULT_METRICS = {
    'l1': metrics.l1_by_group,
    'wasserstein': metrics.wasserstein,
}


DEFAULT_SCORING_CONFIG = {
    "spread": {
        "fn": lambda m, b: eval.spread(m, b).values,
        "discrete": True,
    },
    "orderbook_imbalance": {
        "fn": lambda m, b: eval.orderbook_imbalance(m, b).values,
    },

    #  TIMES (log scale)
    "log_inter_arrival_time": {
        "fn": lambda m, b: np.log(
            eval.inter_arrival_time(m)
            .replace({0: 1e-9}).values.astype(float)
        ),
    },
    "log_time_to_cancel": {
        "fn": lambda m, b: np.log(
            eval.time_to_cancel(m)
            .dt.total_seconds()
            .replace({0: 1e-9})
            .values.astype(float)
        ),
    },

    # VOLUMES:
    "ask_volume_touch": {
        "fn": lambda m, b: eval.l1_volume(m, b).ask_vol.values,
    },
    "bid_volume_touch": {
        "fn": lambda m, b: eval.l1_volume(m, b).bid_vol.values,
    },
    "ask_volume": {
        "fn": lambda m, b: eval.total_volume(m, b, 10)
        .ask_vol_10.values,
    },
    "bid_volume": {
        "fn": lambda m, b: eval.total_volume(m, b, 10)
        .bid_vol_10.values,
    },

    # DEPTHS:
    "limit_ask_order_depth": {
        "fn": lambda m, b: eval.limit_order_depth(m, b)[0].values,
    },
    "limit_bid_order_depth": {
        "fn": lambda m, b: eval.limit_order_depth(m, b)[1].values,
    },
    "ask_cancellation_depth": {
        "fn": lambda m, b: eval.cancellation_depth(m, b)[0].values,
    },
    "bid_cancellation_depth": {
        "fn": lambda m, b: eval.cancellation_depth(m, b)[1].values,
    },

    # LEVELS:
    "limit_ask_order_levels": {
        "fn": lambda m, b: eval.limit_order_levels(m, b)[0].values,
        "discrete": True,
    },
    "limit_bid_order_levels": {
        "fn": lambda m, b: eval.limit_order_levels(m, b)[1].values,
        "discrete": True,
    },
    "ask_cancellation_levels": {
        "fn": lambda m, b: eval.cancel_order_levels(m, b)[0].values,
        "discrete": True,
    },
    "bid_cancellation_levels": {
        "fn": lambda m, b: eval.cancel_order_levels(m, b)[1].values,
        "discrete": True,
    },

    # TRADES
    "vol_per_min": {
        "fn": lambda m, b: eval.volume_per_minute(m, b).values,
    },
    "ofi": {
        "fn": lambda m, b: eval.orderflow_imbalance(m, b).values,
    },
    "ofi_up": {
        "fn": lambda m, b: eval.orderflow_imbalance_cond_tick(m, b, 1).values,
    },
    "ofi_stay": {
        "fn": lambda m, b: eval.orderflow_imbalance_cond_tick(m, b, 0).values,
    },
    "ofi_down": {
        "fn": lambda m, b: eval.orderflow_imbalance_cond_tick(m, b, -1).values,
    },
    "time_lagged_evals": {
        "fn": lambda m, b: eval.time_lagged_evals(m, b, window_size=100, lookback_steps=50),
        "discrete": False,
        "metric_fns": {
            "wasserstein": metrics.wasserstein,
        }
    },
}


######################## CONDITIONAL SCORING ########################
DEFAULT_SCORING_CONFIG_COND = {
    "ask_volume | spread": {
        "eval": {
            "fn": lambda m, b: eval.l1_volume(m, b).ask_vol.values,
        },
        "cond": {
            "fn": lambda m, b: eval.spread(m, b).values,
            "discrete": True,
        }
    },
    "spread | time": {
        "eval": {
            "fn": lambda m, b: eval.spread(m, b).values,
        },
        "cond": {
            "fn": lambda m, b: eval.time_of_day(m).values,
            # group by hour of the day (start of sequence)
            "thresholds": np.linspace(0, 24*60*60, 24),
        }
    },
    "spread | volatility": {
        "eval": {
            "fn": lambda m, b: eval.spread(m, b).values,
            "discrete": True,
        },
        "cond": {
            "fn": lambda m, b: [eval.volatility(m,b,'0.1s')] * len(m),
        }
    }
}

DEFAULT_SCORING_CONFIG_CONDEXT = {
    "ask_volume | spread": {
        "eval": {
            "fn": lambda m, b: eval.l1_volume(m, b).ask_vol.values,
        },
        "cond": {
            "fn": lambda m, b: eval.spread(m, b).values,
            "discrete": True,
        }
    },
}


DEFAULT_SCORING_CONFIG_CONTEXT = {
    "ask_volume | spread": {
        "fn": lambda m, b: eval.l1_volume(m, b).ask_vol.values,
        "context_fn": lambda m, b: eval.spread(m, b).values,
        "context_config": {
            "discrete": True,
        }
    },
    # "bid_volume_touch | spread": {
    #     "fn": lambda m, b: eval.l1_volume(m, b).bid_vol.values,
    #     "context_fn": lambda m, b: eval.spread(m, b).values,
    #     "context_config": {
    #         "discrete": True,
    #     }
    # },
}


def save_results(scores, scores_dfs, save_path, protocol=-1):
    # make sure the folder exists
    folder_path = save_path.rsplit("/", 1)[0]
    pathlib.Path(folder_path).mkdir(parents=True, exist_ok=True)
    # save tuple as pickle
    with gzip.open(save_path, 'wb') as f:
        tup = (scores, scores_dfs)
        pickle.dump(tup, f, protocol)


def load_results(save_path):
    with gzip.open(save_path, 'rb') as f:
        tup = pickle.load(f)
    return tup


def run_benchmark(
    args: argparse.Namespace,
    scoring_config: dict[str, Any] = None,
    scoring_config_cond: dict[str, Any] = None,
    scoring_config_context: dict[str, Any] = None,
    metric_config: dict[str, Any] = None,
) -> None:
    
    print("*** \tThe assumed file structure for files in this package is:\n"
                "***\t \t {DATA_DIR}/{MODEL}/{STOCK}/data_*\n"
                "***\twhereby DATA_DIR is passed as an argument when launching the script\n"
                "***\t{MODEL}s and {STOCK}s may either be passed as a str or a list of str\n"
                "***\tThe script will iterate over all combinations of models and stocks\n")

    if scoring_config is None:
        scoring_config = DEFAULT_SCORING_CONFIG
    if scoring_config_cond is None:
        scoring_config_cond = DEFAULT_SCORING_CONFIG_COND
    if scoring_config_context is None:
        scoring_config_context = DEFAULT_SCORING_CONFIG_CONTEXT
    if metric_config is None:
        metric_config = DEFAULT_METRICS

    if isinstance(args.stock, str):
        args.stock = [args.stock]
    if isinstance(args.model_name, str):
        args.model_name = [args.model_name]
    if isinstance(args.time_period, str):
        args.time_period = [args.time_period]
    if isinstance(args.model_version, str):
        args.model_version = [args.model_version]

    if args.model_version is None:
        args.model_version = [None]

    for stock in args.stock:
        for model_name in args.model_name:
            for time_period in args.time_period:
                for mv in args.model_version:
                    if mv is not None:
                        stock_model_path = f"{args.data_dir}/{model_name}/{stock}/{time_period}/{mv}"
                    else:
                        stock_model_path = f"{args.data_dir}/{model_name}/{stock}/{time_period}"

                    print(f"[*] Loading generated data from {stock_model_path}")
                    loader = data_loading.Simple_Loader(
                        stock_model_path  + "/data_real", 
                        stock_model_path  + "/data_gen", 
                        stock_model_path  + "/data_cond", 
                    )

                    # materialize all sequences, so we keep them in memory
                    # for multiple accesses
                    for s in loader:
                        s.materialize()

                    time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
                    
                    # Unconditional Scoring
                    if (args.run_all or args.uncond_only) and not (args.cond_only or args.context_only or args.div_only):
                        print("[*] Running unconditional scoring")
                        scores, score_dfs, plot_fns = scoring.run_benchmark(
                            loader,
                            scoring_config,
                            default_metric=metric_config
                        )
                        print("[*] Saving results...")
                        save_results(
                            scores,
                            score_dfs,
                            args.save_dir+"/scores"
                            + f"/scores_uncond_{stock}_{model_name}_{str(mv)}_{time_str}.pkl"
                        )
                        print("... done")

                    # Conditional Scoring
                    if (args.run_all or args.cond_only) and not (args.uncond_only or args.context_only or args.div_only):
                        print("[*] Running conditional scoring")
                        scores_cond, score_dfs_cond, plot_fns_cond = scoring.run_benchmark(
                            loader,
                            scoring_config_cond,
                            default_metric=metric_config
                        )
                        print("[*] Saving results...")
                        save_results(
                            scores_cond,
                            score_dfs_cond,
                            args.save_dir+"/scores"
                            + f"/scores_cond_{stock}_{model_name}_{time_str}.pkl"
                        )
                        print("... done")

                    # Contextual Scoring
                    if (args.run_all or args.context_only) and not (args.uncond_only or args.cond_only or args.div_only):
                        print("[*] Running contextual scoring:")
                        scores_context, score_dfs_context, plot_fns_context = scoring.run_benchmark(
                            loader,
                            scoring_config_context,
                            default_metric=metric_config,
                            contextual=True
                        )
                        print("[*] Saving contextual results...")
                        save_results(
                            scores_context,
                            score_dfs_context,
                            args.save_dir+"/scores"
                            + f"/scores_context_{stock}_{model_name}_{time_str}.pkl"
                        )
                        print("... done")

                    # Divergence Scoring
                    if (args.run_all or args.div_only) and not (args.uncond_only or args.cond_only or args.context_only):
                        print("[*] Running divergence scoring")
                        scores_, score_dfs_, plot_fns_ = scoring.run_benchmark(
                            loader,
                            scoring_config,
                            default_metric=metric_config,
                            divergence_horizon=args.divergence_horizon,
                            divergence=True
                        )
                        print("[*] Saving results...")
                        save_results(
                            scores_,
                            score_dfs_,
                            args.save_dir+"/scores"
                            + f"/scores_div_{stock}_{model_name}_"
                            + f"{args.divergence_horizon}_{time_str}.pkl"
                        )
                        print("... done")

                        if args.div_error_bounds:
                            print("[*] Calculating divergence lower bounds...")
                            baseline_errors_by_score = scoring.calc_baseline_errors_by_score(
                                score_dfs_,
                                metric_config
                            )
                            print("[*] Saving baseline errors...")
                            save_results(
                                baseline_errors_by_score,
                                None,
                                args.save_dir+"/scores"
                                + f"/scores_div_{stock}_REAL_"
                                + f"{args.divergence_horizon}_{time_str}.pkl"
                            )
                            print("... done")

                    print("[*] Done")


def test_metric(
    metric_name: str,
    stock: str = "GOOG",
    model_name: str = "s5_main",
    time_period: str = "2023_Jan",
    data_dir: str = "./data/data/evalsequences",
    save_dir: str = "./results",
) -> None:
    """
    Test a specific metric with real and generated data.
    Loads data, computes the metric, and generates plots.
    
    Args:
        metric_name: Name of the metric to test (must be in DEFAULT_SCORING_CONFIG)
        stock: Stock symbol
        model_name: Model name
        time_period: Time period
        data_dir: Data directory path
        save_dir: Directory to save results
    """
    print("=" * 70)
    print(f"TESTING METRIC: {metric_name}")
    print("=" * 70)
    
    # Validate metric exists
    if metric_name not in DEFAULT_SCORING_CONFIG:
        print(f"\n[!] ERROR: '{metric_name}' not found in DEFAULT_SCORING_CONFIG")
        print(f"\nAvailable metrics:")
        for i, name in enumerate(DEFAULT_SCORING_CONFIG.keys(), 1):
            print(f"  {i}. {name}")
        return
    
    # Build path to data
    stock_model_path = f"{data_dir}/{model_name}/{stock}/{time_period}"
    
    print(f"\n[*] Loading data from: {stock_model_path}")
    try:
        loader = data_loading.Simple_Loader(
            stock_model_path + "/data_real",
            stock_model_path + "/data_gen",
            stock_model_path + "/data_cond",
        )
    except Exception as e:
        print(f"[!] ERROR loading data: {e}")
        return
    
    # Materialize sequences
    print("[*] Materializing sequences...")
    for s in loader:
        s.materialize()
    
    # Create config with only the test metric
    test_config = {metric_name: DEFAULT_SCORING_CONFIG[metric_name]}
    
    print(f"[*] Running metric: {metric_name}")
    try:
        scores, score_dfs, plot_fns = scoring.run_benchmark(
            loader,
            test_config,
            default_metric=DEFAULT_METRICS
        )
    except Exception as e:
        print(f"[!] ERROR computing metric: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Display results
    print(f"\n{'=' * 70}")
    print(f"RESULTS FOR {metric_name}")
    print(f"{'=' * 70}")
    
    if metric_name in scores:
        metric_results = scores[metric_name]
        print(f"\nMetric scores computed:")
        for metric_fn_name, (point_est, ci, bootstraps) in metric_results.items():
            print(f"\n  {metric_fn_name}:")
            print(f"    Point estimate: {point_est:.6f}")
            print(f"    CI [95%]: [{ci[0]:.6f}, {ci[1]:.6f}]")
            print(f"    Bootstrap samples: {len(bootstraps)}")
    
    # Display score dataframe info
    if metric_name in score_dfs:
        score_df = score_dfs[metric_name]
        print(f"\nScore DataFrame:")
        print(f"  Shape: {score_df.shape}")
        print(f"  Columns: {list(score_df.columns)}")
        print(f"  Real samples: {(score_df['type'] == 'real').sum()}")
        print(f"  Generated samples: {(score_df['type'] == 'generated').sum()}")
        real_scores = score_df[score_df['type']=='real']['score']
        gen_scores = score_df[score_df['type']=='generated']['score']
        if len(real_scores) > 0 and len(gen_scores) > 0:
            print(f"\n  Real scores - Min: {real_scores.min():.6f}, "
                  f"Max: {real_scores.max():.6f}, "
                  f"Mean: {real_scores.mean():.6f}")
            print(f"  Gen scores  - Min: {gen_scores.min():.6f}, "
                  f"Max: {gen_scores.max():.6f}, "
                  f"Mean: {gen_scores.mean():.6f}")
    
    # Save results
    print(f"\n[*] Saving test results...")
    time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = f"{save_dir}/scores/test_{metric_name}_{stock}_{time_str}.pkl"
    pathlib.Path(f"{save_dir}/scores").mkdir(parents=True, exist_ok=True)
    save_results(scores, score_dfs, results_path)
    print(f"  ✓ Results saved to: {results_path}")
    
    print(f"\n{'=' * 70}")
    print(f"TEST COMPLETE: {metric_name}")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stock", nargs='+', default="GOOG")
    parser.add_argument("--time_period", nargs='+', default="2023_Jan")
    parser.add_argument("--model_version", nargs='+', default=None)
    parser.add_argument("--data_dir", default="./data/data/evalsequences", type=str)
    parser.add_argument("--save_dir", type=str,default="./results")
    parser.add_argument("--model_name", nargs='+', default="s5_main")
    parser.add_argument("--uncond_only", action="store_true")
    parser.add_argument("--cond_only", action="store_true")
    parser.add_argument("--context_only", action="store_true")
    parser.add_argument("--div_only", action="store_true")
    parser.add_argument("--all", action="store_true", dest="run_all")
    parser.add_argument("--div_error_bounds", action="store_true")
    parser.add_argument("--divergence_horizon", type=int, default=100)
    parser.add_argument("--test", type=str, default=None,
                        help="Test a specific metric (e.g., --test time_lagged_evals)")
    args = parser.parse_args()

    # Handle test mode
    if args.test:
        print("\n[*] Running in TEST MODE")
        test_metric(
            args.test,
            stock=args.stock[0] if isinstance(args.stock, list) else args.stock,
            model_name=args.model_name[0] if isinstance(args.model_name, list) else args.model_name,
            time_period=args.time_period[0] if isinstance(args.time_period, list) else args.time_period,
            data_dir=args.data_dir,
            save_dir=args.save_dir,
        )
        exit(0)

    # Validate that --all is not combined with specific scoring flags
    if args.run_all:
        assert not (args.uncond_only or args.cond_only or args.context_only or args.div_only), \
            "Cannot use --all flag with --uncond_only, --cond_only, --context_only, or --div_only"
    
    # Validate that at least one scoring type is specified
    scoring_flags = [args.uncond_only, args.cond_only, args.context_only, args.div_only, args.run_all]
    if not any(scoring_flags):
        print("\n[!] No scoring type specified. Please choose one of the following:")
        print("\n    Scoring Options:")
        print("    --uncond_only         : Run only unconditional scoring")
        print("    --cond_only           : Run only conditional scoring")
        print("    --context_only        : Run only contextual scoring")
        print("    --div_only            : Run only divergence scoring")
        print("    --all                 : Run all scoring types")
        print("\n    Testing Options:")
        print("    --test METRIC         : Test a specific metric (e.g., --test time_lagged_evals)")
        print("\n    Example: python run_bench.py --context_only --stock GOOG --model_name s5_main")
        print("             python run_bench.py --all --stock GOOG INTC --model_name s5_main s5v2_uncond")
        print("             python run_bench.py --test time_lagged_evals --stock GOOG --model_name s5_main\n")
        exit(1)
    
    # Prevent conflicting single-type flags
    if sum(scoring_flags) > 1 and not args.run_all:
        assert False, \
            "Cannot specify multiple scoring flags (--uncond_only, --cond_only, --context_only, --div_only) together. Use --all to run all types."
    
    assert not (args.div_error_bounds and not (args.div_only or args.run_all)), \
        "Cannot calculate divergence error bounds without running divergence scoring (use --div_only or --all)"
    t0=time.time()
    run_benchmark(args)
    t1=time.time()
    print("Finished Run, time (s) is:", t1-t0)
