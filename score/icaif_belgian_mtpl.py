"""Cross-country claim-per-exposure replication on CASdatasets beMTPL97."""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import pyreadr
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wsdm_baselines import metrics, run_method


NUMERIC = ["expo", "ageph", "bm", "power", "agec", "long", "lat"]
CATEGORICAL = ["coverage", "sex", "fuel", "use", "fleet"]


def load_data(path: str, n: int, seed: int):
    frame = pyreadr.read_r(path)["beMTPL97"]
    amount = pd.to_numeric(frame["amount"], errors="coerce").fillna(0)
    exposure = pd.to_numeric(frame["expo"], errors="coerce").fillna(0)
    label = (amount > 0).astype(np.float32).to_numpy()
    valid = (exposure > 0).to_numpy()
    frame = frame.loc[valid].reset_index(drop=True)
    amount = amount.loc[valid].to_numpy(dtype=np.float32)
    exposure = exposure.loc[valid].to_numpy(dtype=np.float32)
    label = label[valid]
    if n < len(frame):
        idx, _ = train_test_split(
            np.arange(len(frame)), train_size=n, random_state=seed, stratify=label
        )
        frame = frame.iloc[idx].reset_index(drop=True)
        amount, exposure, label = amount[idx], exposure[idx], label[idx]
    return frame[NUMERIC + CATEGORICAL], label, amount, exposure


def transform_split(frame, train_idx, test_idx):
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
        transform.transform(clean.iloc[test_idx]).astype(np.float32),
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--n", type=int, default=100_000)
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--pairs-per-batch", type=int, default=2048)
    parser.add_argument("--output", default="results/csvs/icaif_belgian_mtpl.csv")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    frame, label, loss, exposure = load_data(args.input, args.n, 42)
    print(
        f"policies={len(frame)} claim-rate={label.mean():.4f} "
        f"claims={loss.sum():.2f} exposure={exposure.sum():.2f}", flush=True
    )
    methods = [
        "BCE", "W-BCE(raw)", "W-BCE(log)", "Ratio regression",
        "RankNet(A)", "RankNet(A/B)", "Plug-in (Poisson)",
    ]
    rows = []
    indices = np.arange(len(frame))
    for seed in range(args.seeds):
        train_idx, test_idx = train_test_split(
            indices, test_size=0.2, random_state=seed, stratify=label
        )
        x_train, x_test = transform_split(frame, train_idx, test_idx)
        for method in methods:
            started = time.perf_counter()
            scores = run_method(
                method, x_train, x_test, loss[train_idx], exposure[train_idx],
                label[train_idx], 60_000 + seed, args.epochs, args.batch_size,
                args.pairs_per_batch,
            )
            row = {
                "dataset": "beMTPL97", "seed": seed, "method": method,
                "n": len(frame), "loss_rate": float(label.mean()),
                "seconds": time.perf_counter() - started,
                **metrics(scores, loss[test_idx], exposure[test_idx]),
            }
            rows.append(row)
            print(row, flush=True)
    result = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    result.to_csv(args.output, index=False)
    print(f"saved {args.output}")
