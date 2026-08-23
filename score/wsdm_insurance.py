"""Evaluate ratio-of-expectations ranking on freMTPL2 insurance data."""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wsdm_baselines import metrics, run_method


def load_data(n: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    frequency = fetch_openml(
        name="freMTPL2freq", as_frame=True, parser="auto", data_home="/tmp/sklearn_data"
    )
    severity = fetch_openml(
        name="freMTPL2sev", as_frame=True, parser="auto", data_home="/tmp/sklearn_data"
    )
    frame = frequency.data.copy()
    frame["IDpol"] = frame["IDpol"].astype(int)
    severity_frame = severity.data.copy()
    severity_frame["IDpol"] = severity_frame["IDpol"].astype(int)
    claims = severity_frame.groupby("IDpol")["ClaimAmount"].sum()
    leakage = frame["IDpol"].map(claims).fillna(0.0).to_numpy(dtype=np.float32)
    exposure = frame["Exposure"].to_numpy(dtype=np.float32)
    label = (leakage > 0).astype(np.float32)

    # ClaimNb is an outcome and IDpol is an identifier, so neither is a feature.
    feature_names = [
        "Exposure",
        "Area",
        "VehPower",
        "VehAge",
        "DrivAge",
        "BonusMalus",
        "VehBrand",
        "VehGas",
        "Density",
        "Region",
    ]
    features = frame[feature_names]
    categorical = ["Area", "VehBrand", "VehGas", "Region"]
    numerical = [name for name in feature_names if name not in categorical]
    transform = ColumnTransformer(
        [
            ("numeric", StandardScaler(), numerical),
            ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical),
        ]
    )
    x = transform.fit_transform(features).astype(np.float32)
    if n < len(x):
        rng = np.random.default_rng(seed)
        positive = np.flatnonzero(label == 1)
        negative = np.flatnonzero(label == 0)
        positive_n = max(1, round(n * len(positive) / len(label)))
        indices = np.concatenate(
            [
                rng.choice(positive, positive_n, replace=False),
                rng.choice(negative, n - positive_n, replace=False),
            ]
        )
        rng.shuffle(indices)
        x, label, leakage, exposure = (
            x[indices],
            label[indices],
            leakage[indices],
            exposure[indices],
        )
    return x, label, leakage, exposure


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=100_000)
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--pairs-per-batch", type=int, default=2048)
    parser.add_argument(
        "--methods",
        nargs="+",
        default=[
            "BCE",
            "BCE(XGB)",
            "W-BCE(raw)",
            "W-BCE(log)",
            "W-BCE-XGB(raw)",
            "W-BCE-XGB(log)",
            "Ratio regression",
            "RankNet(A)",
            "RankNet(A/B)",
            "Dual Poisson",
        ],
    )
    parser.add_argument("--output", default="results/csvs/wsdm_insurance.csv")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    x, label, leakage, exposure = load_data(args.n, 42)
    rows = []
    indices = np.arange(len(label))
    for seed in range(args.seeds):
        train_idx, test_idx = train_test_split(
            indices, test_size=0.2, random_state=seed, stratify=label
        )
        for method in args.methods:
            started = time.perf_counter()
            scores = run_method(
                method,
                x[train_idx],
                x[test_idx],
                leakage[train_idx],
                exposure[train_idx],
                label[train_idx],
                10_000 + seed,
                args.epochs,
                args.batch_size,
                args.pairs_per_batch,
            )
            row = {
                "dataset": "freMTPL2",
                "seed": seed,
                "objective": 1,
                "method": method,
                "seconds": time.perf_counter() - started,
                **metrics(scores, leakage[test_idx], exposure[test_idx]),
            }
            rows.append(row)
            print(row, flush=True)
    result = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    result.to_csv(args.output, index=False)
    print(f"saved {args.output}")
