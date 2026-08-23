"""Evaluate ratio-of-expectations ranking on resolved Lending Club loans.

The input is the public Kaggle snapshot ``wordsforthewise/lending-club``.
Only origination-time fields are features. Realized outcomes are:

Paper convention after masking:
  B = bad-loan principal loss
  A = good-loan interest + late fees - collection recovery fees

Both are clipped at zero. Current and otherwise unresolved loans are excluded.
"""

from __future__ import annotations

import argparse
import gzip
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wsdm_baselines import metrics, run_method


NUMERIC = [
    "loan_amnt", "funded_amnt", "int_rate", "installment", "annual_inc",
    "dti", "delinq_2yrs", "fico_range_low", "fico_range_high",
    "inq_last_6mths", "open_acc", "pub_rec", "revol_bal", "revol_util",
    "total_acc",
]
CATEGORICAL = [
    "term", "grade", "sub_grade", "emp_length", "home_ownership",
    "verification_status", "purpose", "initial_list_status",
    "application_type",
]
OUTCOMES = [
    "loan_status", "total_rec_prncp", "total_rec_int",
    "total_rec_late_fee", "recoveries", "collection_recovery_fee",
]
USECOLS = NUMERIC + CATEGORICAL + OUTCOMES
RESOLVED = {"Fully Paid", "Charged Off", "Default"}


def load_resolved(path: str, n: int, seed: int) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    frames = []
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8", errors="replace") as handle:
        for chunk in pd.read_csv(handle, usecols=USECOLS, chunksize=200_000, low_memory=False):
            chunk = chunk[chunk["loan_status"].isin(RESOLVED)]
            if len(chunk):
                frames.append(chunk)
    frame = pd.concat(frames, ignore_index=True)

    principal_loss = (
        frame["funded_amnt"] - frame["total_rec_prncp"] - frame["recoveries"]
    ).clip(lower=0).fillna(0)
    realized_return = (
        frame["total_rec_int"] + frame["total_rec_late_fee"]
        - frame["collection_recovery_fee"]
    ).clip(lower=0).fillna(0)
    label = (principal_loss > 0).astype(np.float32)
    realized_return = realized_return * (1.0 - label)

    valid = (principal_loss + realized_return > 0) & frame["funded_amnt"].notna()
    frame = frame.loc[valid].reset_index(drop=True)
    principal_loss = principal_loss.loc[valid].to_numpy(dtype=np.float32)
    realized_return = realized_return.loc[valid].to_numpy(dtype=np.float32)
    label = label.loc[valid].to_numpy(dtype=np.float32)

    if n < len(frame):
        indices, _ = train_test_split(
            np.arange(len(frame)), train_size=n, random_state=seed, stratify=label
        )
        frame = frame.iloc[indices].reset_index(drop=True)
        principal_loss = principal_loss[indices]
        realized_return = realized_return[indices]
        label = label[indices]
    return frame[NUMERIC + CATEGORICAL], label, principal_loss, realized_return


def transform_split(frame: pd.DataFrame, train_idx: np.ndarray, test_idx: np.ndarray):
    numeric = frame[NUMERIC].apply(pd.to_numeric, errors="coerce")
    numeric = numeric.fillna(numeric.iloc[train_idx].median()).fillna(0)
    categorical = frame[CATEGORICAL].fillna("MISSING").astype(str)
    clean = pd.concat([numeric, categorical], axis=1)
    transform = ColumnTransformer(
        [
            ("numeric", StandardScaler(), NUMERIC),
            ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL),
        ]
    )
    x_train = transform.fit_transform(clean.iloc[train_idx]).astype(np.float32)
    x_test = transform.transform(clean.iloc[test_idx]).astype(np.float32)
    return x_train, x_test


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--n", type=int, default=100_000)
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--pairs-per-batch", type=int, default=2048)
    parser.add_argument("--output", default="results/csvs/icaif_lending_club.csv")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    frame, label, loss, realized_return = load_resolved(args.input, args.n, 42)
    print(
        f"resolved sample={len(frame)} loss-positive={label.mean():.4f} "
        f"loss_sum={loss.sum():.2f} return_sum={realized_return.sum():.2f}",
        flush=True,
    )
    methods = [
        "BCE", "W-BCE(raw)", "W-BCE(log)", "Ratio regression",
        "RankNet(A)", "RankNet(A/B)", "Dual Poisson",
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
                method, x_train, x_test, loss[train_idx], realized_return[train_idx],
                label[train_idx], 20_000 + seed, args.epochs, args.batch_size,
                args.pairs_per_batch,
            )
            row = {
                "dataset": "LendingClub-resolved",
                "seed": seed,
                "method": method,
                "n": len(frame),
                "loss_rate": float(label.mean()),
                "seconds": time.perf_counter() - started,
                **metrics(scores, loss[test_idx], realized_return[test_idx]),
            }
            rows.append(row)
            print(row, flush=True)
    result = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    result.to_csv(args.output, index=False)
    print(f"saved {args.output}")
