"""Nested evaluation on Australian private motor insurance policies."""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


NUMERIC = ["Exposure", "VehValue"]
CATEGORICAL = ["VehAge", "VehBody", "Gender", "DrivAge"]


def read_ausprivauto(path: str) -> pd.DataFrame:
    if Path(path).suffix.lower() == ".csv":
        return pd.read_csv(path)
    try:
        import pyreadr
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "RData input requires pyreadr. Use a CSV export of ausprivauto0405 "
            "when running through Buck."
        ) from error
    return pyreadr.read_r(path)["ausprivauto0405"]


def load_data(path: str):
    frame = read_ausprivauto(path)
    loss = pd.to_numeric(frame["ClaimAmount"], errors="coerce").fillna(0)
    exposure = pd.to_numeric(frame["Exposure"], errors="coerce").fillna(0)
    label = (loss > 0).astype(np.float32).to_numpy()
    # Paper convention: A is value on a good allowed case; B is loss on a bad
    # allowed case. The two settled branches are mutually exclusive.
    value = exposure.to_numpy(dtype=np.float32) * (1.0 - label)
    valid = (exposure > 0).to_numpy()
    return (
        frame.loc[valid, NUMERIC + CATEGORICAL].reset_index(drop=True),
        label[valid],
        loss.loc[valid].to_numpy(dtype=np.float32),
        value[valid],
    )


def transform_split(frame, train_idx, val_idx, test_idx):
    numeric = frame[NUMERIC].apply(pd.to_numeric, errors="coerce")
    numeric = numeric.fillna(numeric.iloc[train_idx].median()).fillna(0)
    categorical = frame[CATEGORICAL].fillna("MISSING").astype(str)
    clean = pd.concat([numeric, categorical], axis=1)
    transform = ColumnTransformer(
        [
            ("numeric", StandardScaler(), NUMERIC),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL,
            ),
        ]
    )
    return (
        transform.fit_transform(clean.iloc[train_idx]).astype(np.float32),
        transform.transform(clean.iloc[val_idx]).astype(np.float32),
        transform.transform(clean.iloc[test_idx]).astype(np.float32),
    )


def validation_threshold(scores, cost, budget=0.10):
    order = np.argsort(-scores, kind="stable")
    cumulative = np.cumsum(cost[order], dtype=np.float64)
    target = budget * cumulative[-1]
    position = min(np.searchsorted(cumulative, target, side="left"), len(order) - 1)
    return float(scores[order[position]])


def prospective_metrics(scores, loss, cost, threshold):
    selected = scores >= threshold
    return {
        "prospective_capture": float(loss[selected].sum() / loss.sum()),
        "prospective_realized_budget": float(cost[selected].sum() / cost.sum()),
        "validation_score_threshold": threshold,
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--pairs-per-batch", type=int, default=2048)
    parser.add_argument(
        "--output", default="results/csvs/icaif_australian_mtpl_nested.csv"
    )
    return parser.parse_args()


if __name__ == "__main__":
    from wsdm_baselines import metrics, run_method

    args = parse_args()
    frame, label, loss, exposure = load_data(args.input)
    methods = [
        "BCE",
        "W-BCE(raw)",
        "W-BCE(log)",
        "Ratio regression",
        "RankNet(A)",
        "RankNet(A/B)",
        "Plug-in (Poisson)",
    ]
    display_method = {"RankNet(A)": "RankNet(B)", "RankNet(A/B)": "RankNet(B/A)"}
    rows = []
    indices = np.arange(len(frame))
    for seed in range(args.seeds):
        outer_train, test_idx = train_test_split(
            indices, test_size=0.20, random_state=seed, stratify=label
        )
        train_idx, val_idx = train_test_split(
            outer_train,
            test_size=0.20,
            random_state=10_000 + seed,
            stratify=label[outer_train],
        )
        x_train, x_val, x_test = transform_split(frame, train_idx, val_idx, test_idx)
        x_eval = np.concatenate([x_val, x_test], axis=0)
        for method in methods:
            started = time.perf_counter()
            combined_scores = run_method(
                method,
                x_train,
                x_eval,
                loss[train_idx],
                exposure[train_idx],
                label[train_idx],
                80_000 + seed,
                args.epochs,
                args.batch_size,
                args.pairs_per_batch,
            )
            val_scores = combined_scores[: len(val_idx)]
            test_scores = combined_scores[len(val_idx) :]
            threshold = validation_threshold(val_scores, exposure[val_idx])
            row = {
                "dataset": "ausprivauto0405",
                "seed": seed,
                "method": display_method.get(method, method),
                "n": len(frame),
                "train_n": len(train_idx),
                "validation_n": len(val_idx),
                "test_n": len(test_idx),
                "seconds": time.perf_counter() - started,
                **metrics(test_scores, loss[test_idx], exposure[test_idx]),
                **prospective_metrics(
                    test_scores, loss[test_idx], exposure[test_idx], threshold
                ),
            }
            rows.append(row)
            print(row, flush=True)
    result = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    result.to_csv(args.output, index=False)
    print(f"saved {args.output}")
