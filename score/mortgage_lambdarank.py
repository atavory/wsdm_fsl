"""Nested XGBoost LambdaMART baselines for cached mortgage samples."""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mortgage_nested import (
    FANNIE_CATEGORICAL,
    FANNIE_NUMERIC,
    FREDDIE_CATEGORICAL,
    FREDDIE_NUMERIC,
    metrics,
    prospective_metrics,
    transform_split,
    validation_threshold,
)


def relevance_bins(values: np.ndarray, positive_bins: int = 31) -> np.ndarray:
    """Map a nonnegative training gain to integer LambdaRank relevance."""
    values = np.asarray(values, dtype=np.float64)
    relevance = np.zeros(len(values), dtype=np.int32)
    positive = values > 0
    if not np.any(positive):
        return relevance
    # Rank rather than quantile-edge binning handles tied and heavy-tailed gains.
    ranks = pd.Series(values[positive]).rank(method="average", pct=True).to_numpy()
    relevance[positive] = np.minimum(
        np.ceil(ranks * positive_bins).astype(np.int32), positive_bins
    )
    return relevance


def query_order(n: int, seed: int, query_size: int):
    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    groups = [query_size] * (n // query_size)
    remainder = n % query_size
    if remainder:
        groups.append(remainder)
    qid = np.repeat(np.arange(len(groups), dtype=np.int32), groups)
    return order, qid


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("fannie", "freddie"), required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--query-size", type=int, default=512)
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    frame = pd.read_pickle(args.cache)
    numeric_map = FANNIE_NUMERIC if args.dataset == "fannie" else FREDDIE_NUMERIC
    categorical_map = (
        FANNIE_CATEGORICAL if args.dataset == "fannie" else FREDDIE_CATEGORICAL
    )
    numeric = list(numeric_map) + ["origination_year", "origination_month"]
    categorical = list(categorical_map)
    label = frame["label"].to_numpy(dtype=np.float32)
    loss = frame["loss"].to_numpy(dtype=np.float32)
    exposure = frame["exposure"].to_numpy(dtype=np.float32)
    value = exposure * (1.0 - label)
    indices = np.arange(len(frame))
    rows = []

    for seed in range(args.seeds):
        outer_train, test_idx = train_test_split(
            indices, test_size=0.20, random_state=seed, stratify=label
        )
        train_idx, val_idx = train_test_split(
            outer_train, test_size=0.20, random_state=10_000 + seed,
            stratify=label[outer_train],
        )
        x_train, x_val, x_test = transform_split(
            frame, numeric, categorical, train_idx, val_idx, test_idx
        )
        order, qid = query_order(len(train_idx), 70_000 + seed, args.query_size)

        gains = {
            "LambdaMART(B)": loss[train_idx],
            "LambdaMART(B/A)": loss[train_idx] / np.maximum(value[train_idx], 1e-8),
        }
        for method, gain in gains.items():
            relevance = relevance_bins(gain)
            started = time.perf_counter()
            model = xgb.XGBRanker(
                objective="rank:ndcg",
                n_estimators=args.n_estimators,
                learning_rate=0.05,
                max_depth=6,
                min_child_weight=10,
                subsample=1.0,
                colsample_bytree=1.0,
                reg_lambda=1.0,
                lambdarank_pair_method="topk",
                lambdarank_num_pair_per_sample=32,
                random_state=80_000 + seed,
                n_jobs=-1,
                tree_method="hist",
                verbosity=0,
            )
            model.fit(x_train[order], relevance[order], qid=qid, verbose=False)
            val_scores = model.predict(x_val)
            test_scores = model.predict(x_test)
            threshold = validation_threshold(val_scores, value[val_idx])
            elapsed = time.perf_counter() - started
            model_bytes = len(model.get_booster().save_raw())
            row = {
                "dataset": args.dataset,
                "seed": seed,
                "method": method,
                "n": len(frame),
                "train_n": len(train_idx),
                "validation_n": len(val_idx),
                "test_n": len(test_idx),
                "query_size": args.query_size,
                "n_estimators": args.n_estimators,
                "seconds": elapsed,
                "model_bytes": model_bytes,
                **metrics(test_scores, loss[test_idx], value[test_idx]),
                **prospective_metrics(
                    test_scores, loss[test_idx], value[test_idx], threshold
                ),
            }
            rows.append(row)
            print(row, flush=True)

    result = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    result.to_csv(args.output, index=False)
    print(f"saved {args.output}")
