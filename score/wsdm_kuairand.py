"""Nested yield-ranking evaluation on randomized KuaiRand exposures."""

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


LOG_COLUMNS = [
    "user_id",
    "video_id",
    "date",
    "hourmin",
    "is_hate",
    "play_time_ms",
    "duration_ms",
    "tab",
]
USER_COLUMNS = [
    "user_id",
    "user_active_degree",
    "is_lowactive_period",
    "is_live_streamer",
    "is_video_author",
    "follow_user_num",
    "follow_user_num_range",
    "fans_user_num",
    "fans_user_num_range",
    "friend_user_num",
    "friend_user_num_range",
    "register_days",
    "register_days_range",
]
VIDEO_COLUMNS = [
    "video_id",
    "video_type",
    "upload_dt",
    "upload_type",
    "video_duration",
    "server_width",
    "server_height",
    "music_type",
]
NUMERIC = [
    "hour",
    "tab",
    "duration_seconds",
    "video_age_days",
    "is_lowactive_period",
    "is_live_streamer",
    "is_video_author",
    "follow_user_num",
    "fans_user_num",
    "friend_user_num",
    "register_days",
    "video_duration_seconds",
    "server_width",
    "server_height",
]
CATEGORICAL = [
    "day",
    "user_active_degree",
    "follow_user_num_range",
    "fans_user_num_range",
    "friend_user_num_range",
    "register_days_range",
    "video_type",
    "upload_type",
    "music_type",
]


def load_data(path: str, max_rows: int, sample_seed: int):
    logs = pd.read_csv(
        os.path.join(path, "log_random_4_22_to_5_08_pure.csv"),
        usecols=LOG_COLUMNS,
    )
    if max_rows and len(logs) > max_rows:
        logs = logs.sample(max_rows, random_state=sample_seed).sort_index()
    users = pd.read_csv(
        os.path.join(path, "user_features_pure.csv"), usecols=USER_COLUMNS
    )
    videos = pd.read_csv(
        os.path.join(path, "video_features_basic_pure.csv"), usecols=VIDEO_COLUMNS
    )
    frame = logs.merge(users, on="user_id", how="left", validate="many_to_one")
    frame = frame.merge(videos, on="video_id", how="left", validate="many_to_one")

    interaction_date = pd.to_datetime(
        frame["date"].astype("Int64").astype(str), format="%Y%m%d", errors="coerce"
    )
    upload_date = pd.to_datetime(frame["upload_dt"], errors="coerce")
    frame["hour"] = pd.to_numeric(frame["hourmin"], errors="coerce") // 100
    frame["day"] = interaction_date.dt.dayofweek.astype("Int64").astype(str)
    frame["duration_seconds"] = (
        pd.to_numeric(frame["duration_ms"], errors="coerce") / 1000.0
    )
    frame["video_duration_seconds"] = (
        pd.to_numeric(frame["video_duration"], errors="coerce") / 1000.0
    )
    frame["video_age_days"] = (interaction_date - upload_date).dt.days

    harm = pd.to_numeric(frame["is_hate"], errors="coerce").fillna(0)
    value = (
        pd.to_numeric(frame["play_time_ms"], errors="coerce").fillna(0) / 1000.0
    )
    valid = (harm >= 0) & (value >= 0)
    frame = frame.loc[valid].reset_index(drop=True)
    harm = harm.loc[valid].to_numpy(dtype=np.float32)
    value = value.loc[valid].to_numpy(dtype=np.float32)
    group = frame["user_active_degree"].fillna("UNKNOWN").astype(str).to_numpy()
    return frame[NUMERIC + CATEGORICAL], harm, value, group


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


def validation_threshold(scores, value, budget=0.10):
    order = np.argsort(-scores, kind="stable")
    cumulative = np.cumsum(value[order], dtype=np.float64)
    target = budget * cumulative[-1]
    position = min(np.searchsorted(cumulative, target, side="left"), len(order) - 1)
    return float(scores[order[position]])


def prospective_metrics(scores, harm, value, threshold):
    selected = scores >= threshold
    return {
        "prospective_harm_capture": float(harm[selected].sum() / harm.sum()),
        "prospective_realized_value_budget": float(
            value[selected].sum() / value.sum()
        ),
        "validation_score_threshold": threshold,
    }


def group_metrics(scores, harm, value, group, threshold):
    selected = scores >= threshold
    result = {}
    for name in np.unique(group):
        mask = group == name
        key = name.lower().replace(" ", "_")
        if mask.sum() < 100 or harm[mask].sum() <= 0 or value[mask].sum() <= 0:
            continue
        result[f"group_{key}_n"] = int(mask.sum())
        result[f"group_{key}_gini"] = metrics(
            scores[mask], harm[mask], value[mask]
        )["gini"]
        result[f"group_{key}_value_fraction"] = float(
            value[mask & selected].sum() / value[mask].sum()
        )
        result[f"group_{key}_harm_capture"] = float(
            harm[mask & selected].sum() / harm[mask].sum()
        )
    return result


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--pairs-per-batch", type=int, default=2048)
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--sample-seed", type=int, default=42)
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
            "Plug-in (Poisson)",
        ],
    )
    parser.add_argument("--output", default="results/csvs/wsdm_kuairand.csv")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    frame, harm, value, group = load_data(
        args.input, args.max_rows, args.sample_seed
    )
    print(
        {
            "n": len(frame),
            "harm_count": int(harm.sum()),
            "harm_rate": float(harm.mean()),
            "mean_watch_seconds": float(value.mean()),
        },
        flush=True,
    )
    display_method = {"RankNet(A)": "RankNet(B)", "RankNet(A/B)": "RankNet(B/A)"}
    rows = []
    indices = np.arange(len(frame))
    for seed in range(args.seeds):
        outer_train, test_idx = train_test_split(
            indices, test_size=0.20, random_state=seed, stratify=harm
        )
        train_idx, val_idx = train_test_split(
            outer_train,
            test_size=0.20,
            random_state=10_000 + seed,
            stratify=harm[outer_train],
        )
        x_train, x_val, x_test = transform_split(frame, train_idx, val_idx, test_idx)
        x_eval = np.concatenate([x_val, x_test], axis=0)
        for method in args.methods:
            started = time.perf_counter()
            combined_scores = run_method(
                method,
                x_train,
                x_eval,
                harm[train_idx],
                value[train_idx],
                harm[train_idx],
                90_000 + seed,
                args.epochs,
                args.batch_size,
                args.pairs_per_batch,
            )
            val_scores = combined_scores[: len(val_idx)]
            test_scores = combined_scores[len(val_idx) :]
            threshold = validation_threshold(val_scores, value[val_idx])
            row = {
                "dataset": "KuaiRand-Pure-random",
                "seed": seed,
                "method": display_method.get(method, method),
                "n": len(frame),
                "train_n": len(train_idx),
                "validation_n": len(val_idx),
                "test_n": len(test_idx),
                "test_harm_count": int(harm[test_idx].sum()),
                "seconds": time.perf_counter() - started,
                **metrics(test_scores, harm[test_idx], value[test_idx]),
                **prospective_metrics(
                    test_scores, harm[test_idx], value[test_idx], threshold
                ),
                **group_metrics(
                    test_scores,
                    harm[test_idx],
                    value[test_idx],
                    group[test_idx],
                    threshold,
                ),
            }
            rows.append(row)
            print(row, flush=True)
    result = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    result.to_csv(args.output, index=False)
    print(f"saved {args.output}")
