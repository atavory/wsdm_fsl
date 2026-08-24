"""Resolved Prosper loans with paper A=good value and B=bad loss."""

from __future__ import annotations

import argparse
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
    "Term", "BorrowerAPR", "BorrowerRate", "EmploymentStatusDuration",
    "CreditScoreRangeLower", "CreditScoreRangeUpper", "CurrentCreditLines",
    "OpenCreditLines", "TotalCreditLinespast7years",
    "OpenRevolvingAccounts", "OpenRevolvingMonthlyPayment",
    "InquiriesLast6Months", "TotalInquiries", "CurrentDelinquencies",
    "AmountDelinquent", "DelinquenciesLast7Years",
    "PublicRecordsLast10Years", "PublicRecordsLast12Months",
    "RevolvingCreditBalance", "BankcardUtilization",
    "AvailableBankcardCredit", "TotalTrades",
    "TradesNeverDelinquent (percentage)", "TradesOpenedLast6Months",
    "DebtToIncomeRatio", "StatedMonthlyIncome", "TotalProsperLoans",
    "ProsperPrincipalBorrowed", "LoanOriginalAmount", "MonthlyLoanPayment",
]
CATEGORICAL = [
    "ListingCategory (numeric)", "BorrowerState", "Occupation",
    "EmploymentStatus", "IsBorrowerHomeowner", "IncomeRange",
    "IncomeVerifiable",
]
OUTCOMES = ["LoanStatus", "LP_NetPrincipalLoss", "LP_InterestandFees"]
RESOLVED = {"Completed", "Chargedoff", "Defaulted"}


def load_data(path: str):
    frame = pd.read_csv(path, usecols=NUMERIC + CATEGORICAL + OUTCOMES, low_memory=False)
    frame = frame[frame["LoanStatus"].isin(RESOLVED)].reset_index(drop=True)
    loss = frame["LP_NetPrincipalLoss"].clip(lower=0).fillna(0).to_numpy(dtype=np.float32)
    realized_return = frame["LP_InterestandFees"].clip(lower=0).fillna(0).to_numpy(dtype=np.float32)
    label = (loss > 0).astype(np.float32)
    realized_return = realized_return * (1.0 - label)
    valid = loss + realized_return > 0
    return (
        frame.loc[valid, NUMERIC + CATEGORICAL].reset_index(drop=True),
        label[valid],
        loss[valid],
        realized_return[valid],
    )


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
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--pairs-per-batch", type=int, default=2048)
    parser.add_argument("--output", default="results/csvs/prosper.csv")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    frame, label, loss, realized_return = load_data(args.input)
    print(
        f"resolved={len(frame)} loss-positive={label.mean():.4f} "
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
                method, x_train, x_test, loss[train_idx],
                realized_return[train_idx], label[train_idx], 30_000 + seed,
                args.epochs, args.batch_size, args.pairs_per_batch,
            )
            row = {
                "dataset": "Prosper-resolved",
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
