"""Nested mortgage-risk experiments on Fannie HARP and Freddie CRT files.

The estimand is expected realized credit loss per dollar of original principal.
Only origination-time attributes are used as features. Both loaders retain only
resolved loans and take a deterministic reservoir sample without conditioning
on the outcome, preserving the observed loss prevalence.
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import sys
import time
import zipfile

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wsdm_baselines import metrics, run_method


METHODS = [
    "BCE", "BCE(XGB)", "W-BCE(raw)", "W-BCE(log)",
    "W-BCE-XGB(raw)", "W-BCE-XGB(log)", "Ratio regression",
    "RankNet(A)", "RankNet(A/B)", "Plug-in (Poisson)",
]

FANNIE_NUMERIC = {
    "original_rate": 8, "original_upb": 10, "original_term": 13,
    "ltv": 20, "cltv": 21, "borrowers": 22, "dti": 23,
    "fico": 24, "co_fico": 25, "units": 29, "mi_pct": 34,
}
FANNIE_CATEGORICAL = {
    "purpose": 27, "property_type": 28, "occupancy": 30, "state": 31,
    "amortization": 35, "prepay_penalty": 36, "interest_only": 37,
}
FREDDIE_NUMERIC = {
    "original_term": 11, "original_rate": 12, "original_upb": 13,
    "units": 18, "borrowers": 20, "fico": 23, "ltv": 24,
    "cltv": 25, "dti": 26, "mi_pct": 27,
}
FREDDIE_CATEGORICAL = {
    "amortization": 4, "state": 6, "purpose": 15, "channel": 16,
    "property_type": 17, "occupancy": 19, "first_home": 21,
    "prepay_penalty": 22, "special_program": 29,
}


def field(row: list[str], position: int) -> str:
    """Return a one-indexed disclosure field."""
    index = position - 1
    return row[index].strip() if index < len(row) else ""


def number(row: list[str], position: int) -> float:
    value = field(row, position)
    try:
        return float(value) if value else 0.0
    except ValueError:
        return 0.0


def date_parts(value: str) -> tuple[float, float]:
    if len(value) != 6 or not value.isdigit():
        return np.nan, np.nan
    # Fannie dates are MMYYYY; Freddie dates are CCYYMM and are normalized by
    # the caller before reaching this helper.
    return float(value[:4]), float(value[4:])


def reservoir_add(
    reservoir: list[dict], item: dict, seen: int, limit: int, rng: random.Random
) -> None:
    if len(reservoir) < limit:
        reservoir.append(item)
        return
    replacement = rng.randrange(seen)
    if replacement < limit:
        reservoir[replacement] = item


def fannie_record(row: list[str]) -> dict | None:
    # Field 1 (Reference Pool ID) is blank in SF performance files, so the
    # leading empty token is part of the documented schema.
    zero_balance = field(row, 44)
    if not zero_balance:
        return None
    original_upb = number(row, 10)
    if original_upb <= 0:
        return None

    # UPB at removal is not a loss for an ordinary payoff. Reconstruct net
    # credit loss only for Fannie's credit-event resolution codes. Codes 01
    # (prepaid/matured), 06 (repurchased), and 96 (non-credit removal) have
    # zero realized credit loss under this construction.
    credit_event_codes = {"02", "03", "09", "15", "16", "97", "98"}
    expenses = sum(number(row, p) for p in (54, 55, 56, 57, 58))
    proceeds = sum(number(row, p) for p in (59, 60, 61, 62))
    loss = 0.0
    if zero_balance in credit_event_codes:
        loss = max(number(row, 46) + number(row, 64) + expenses - proceeds, 0.0)
    record = {name: field(row, p) for name, p in FANNIE_CATEGORICAL.items()}
    record.update({name: number(row, p) for name, p in FANNIE_NUMERIC.items()})
    origination = field(row, 14)  # MMYYYY
    record["origination_year"] = float(origination[2:]) if len(origination) == 6 else np.nan
    record["origination_month"] = float(origination[:2]) if len(origination) == 6 else np.nan
    record.update(
        zero_balance_code=zero_balance, loss=loss, exposure=original_upb,
        label=float(loss > 0),
    )
    return record


def extract_fannie(path: str, limit: int, seed: int) -> pd.DataFrame:
    rng = random.Random(seed)
    reservoir: list[dict] = []
    seen = 0
    with zipfile.ZipFile(path) as archive, archive.open("HARPLPPub.csv") as raw:
        # Rows for a loan are contiguous. Outcome components occur at the end,
        # so retain the final row carrying a zero-balance resolution.
        previous_id = None
        resolved = None
        for line_number, raw_line in enumerate(raw, 1):
            row = raw_line.decode("utf-8", "replace").rstrip("\r\n").split("|")
            loan_id = field(row, 2)
            if loan_id != previous_id:
                if resolved is not None:
                    seen += 1
                    reservoir_add(reservoir, resolved, seen, limit, rng)
                previous_id = loan_id
                resolved = None
            candidate = fannie_record(row)
            if candidate is not None:
                resolved = candidate
            if line_number % 10_000_000 == 0:
                print(f"Fannie extraction rows={line_number:,} resolved={seen:,}", flush=True)
        if resolved is not None:
            seen += 1
            reservoir_add(reservoir, resolved, seen, limit, rng)
    print(f"Fannie resolved population={seen:,} reservoir={len(reservoir):,}", flush=True)
    return pd.DataFrame(reservoir)


def freddie_record(row: list[str]) -> dict | None:
    if not field(row, 43):
        return None
    original_upb = number(row, 13)
    if original_upb <= 0:
        return None
    loss = max(number(row, 92), 0.0)
    record = {name: field(row, p) for name, p in FREDDIE_CATEGORICAL.items()}
    record.update({name: number(row, p) for name, p in FREDDIE_NUMERIC.items()})
    first_payment = field(row, 9)  # CCYYMM
    year, month = date_parts(first_payment)
    record.update(
        origination_year=year, origination_month=month,
        loss=loss, exposure=original_upb, label=float(loss > 0),
    )
    return record


def extract_freddie(
    path: str, limit: int, seed: int, product: str = "stacr"
) -> pd.DataFrame:
    rng = random.Random(seed)
    reservoir: list[dict] = []
    seen = 0
    rows = 0
    with zipfile.ZipFile(path) as archive:
        names = [
            name for name in sorted(archive.namelist())
            if os.path.basename(name).lower().startswith(product.lower() + "_")
        ]
        if not names:
            raise RuntimeError(f"no Freddie {product!r} file found in archive")
        for name in names:
            with archive.open(name) as raw:
                reader = csv.reader(
                    (line.decode("utf-8", "replace") for line in raw), delimiter="|"
                )
                for row in reader:
                    rows += 1
                    record = freddie_record(row)
                    if record is not None:
                        seen += 1
                        reservoir_add(reservoir, record, seen, limit, rng)
                    if rows % 5_000_000 == 0:
                        print(f"Freddie extraction rows={rows:,} resolved={seen:,}", flush=True)
    print(f"Freddie resolved population={seen:,} reservoir={len(reservoir):,}", flush=True)
    return pd.DataFrame(reservoir)


def load_or_extract(args: argparse.Namespace) -> tuple[pd.DataFrame, list[str], list[str]]:
    numeric_map = FANNIE_NUMERIC if args.dataset == "fannie" else FREDDIE_NUMERIC
    categorical_map = FANNIE_CATEGORICAL if args.dataset == "fannie" else FREDDIE_CATEGORICAL
    numeric = list(numeric_map) + ["origination_year", "origination_month"]
    categorical = list(categorical_map)
    if args.cache and os.path.exists(args.cache):
        frame = pd.read_pickle(args.cache)
        print(f"loaded cache {args.cache}: {len(frame):,} loans", flush=True)
    else:
        if args.dataset == "fannie":
            frame = extract_fannie(args.input, args.n, args.sample_seed)
        else:
            frame = extract_freddie(
                args.input, args.n, args.sample_seed, args.freddie_product
            )
        if args.cache:
            os.makedirs(os.path.dirname(os.path.abspath(args.cache)), exist_ok=True)
            frame.to_pickle(args.cache)
            print(f"saved cache {args.cache}", flush=True)
    return frame, numeric, categorical


def transform_split(frame, numeric, categorical, train_idx, val_idx, test_idx):
    numeric_frame = frame[numeric].apply(pd.to_numeric, errors="coerce")
    medians = numeric_frame.iloc[train_idx].median()
    numeric_frame = numeric_frame.fillna(medians).fillna(0)
    categorical_frame = frame[categorical].fillna("MISSING").astype(str)
    clean = pd.concat([numeric_frame, categorical_frame], axis=1)
    transform = ColumnTransformer(
        [
            ("numeric", StandardScaler(), numeric),
            ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical),
        ]
    )
    return (
        transform.fit_transform(clean.iloc[train_idx]).astype(np.float32),
        transform.transform(clean.iloc[val_idx]).astype(np.float32),
        transform.transform(clean.iloc[test_idx]).astype(np.float32),
    )


def validation_threshold(scores, exposure, budget=0.10):
    order = np.argsort(-scores, kind="stable")
    cumulative = np.cumsum(exposure[order], dtype=np.float64)
    target = budget * cumulative[-1]
    position = min(np.searchsorted(cumulative, target, side="left"), len(order) - 1)
    return float(scores[order[position]])


def prospective_metrics(scores, loss, exposure, threshold):
    selected = scores >= threshold
    return {
        "prospective_capture": float(loss[selected].sum() / max(loss.sum(), 1e-12)),
        "prospective_realized_budget": float(exposure[selected].sum() / exposure.sum()),
        "validation_score_threshold": threshold,
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("fannie", "freddie"), required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--cache")
    parser.add_argument("--n", type=int, default=300_000)
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument(
        "--freddie-product", choices=("stacr", "acis", "deepmi"),
        default="stacr",
        help="evaluate one CRT product to avoid cross-product pool overlap",
    )
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--pairs-per-batch", type=int, default=2048)
    parser.add_argument("--methods", nargs="+", default=METHODS)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    frame, numeric, categorical = load_or_extract(args)
    label = frame["label"].to_numpy(dtype=np.float32)
    loss = frame["loss"].to_numpy(dtype=np.float32)
    exposure = frame["exposure"].to_numpy(dtype=np.float32)
    # Paper convention: A is value if good, B is loss if bad. Origination
    # exposure is the public-data proxy for incremental value and is zeroed on
    # the bad branch so A*B=0.
    value = exposure * (1.0 - label)
    print(
        f"{args.dataset}: n={len(frame):,} loss-positive={label.mean():.6f} "
        f"loss={loss.sum():.2f} good-value={value.sum():.2f}", flush=True,
    )
    if label.sum() < 100:
        raise RuntimeError("fewer than 100 loss-positive loans; experiment is underpowered")

    rows = []
    indices = np.arange(len(frame))
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
        x_eval = np.concatenate([x_val, x_test], axis=0)
        # A shared scale leaves paper B/A and the exact W-BCE target unchanged while
        # preventing overflow in the Poisson baseline on dollar-valued targets.
        scale = float(np.mean(loss[train_idx] + value[train_idx]))
        train_loss = loss[train_idx] / scale
        train_value = value[train_idx] / scale
        for method in args.methods:
            started = time.perf_counter()
            scores = run_method(
                method, x_train, x_eval, train_loss, train_value,
                label[train_idx], 90_000 + seed, args.epochs, args.batch_size,
                args.pairs_per_batch,
            )
            val_scores, test_scores = scores[: len(val_idx)], scores[len(val_idx) :]
            threshold = validation_threshold(val_scores, value[val_idx])
            row = {
                "dataset": args.dataset, "seed": seed,
                "method": {"RankNet(A)": "RankNet(B)", "RankNet(A/B)": "RankNet(B/A)"}.get(method, method),
                "n": len(frame), "train_n": len(train_idx),
                "validation_n": len(val_idx), "test_n": len(test_idx),
                "loss_rate": float(label.mean()),
                "seconds": time.perf_counter() - started,
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
