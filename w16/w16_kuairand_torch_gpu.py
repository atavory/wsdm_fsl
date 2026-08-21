"""GPU diagnostic W16 per-candidate extractor for KuaiRand-Pure."""

from __future__ import annotations

import argparse
import csv
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from simple_preprocess import stratified_train_test_split
from simple_preprocess import transform_split as transform_frame_split
from provenance import print_provenance


LOG_COLUMNS = ["user_id", "video_id", "date", "hourmin", "is_hate", "play_time_ms", "duration_ms", "tab"]
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


class MLP(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def load_data(path: str, max_rows: int, sample_seed: int):
    logs = pd.read_csv(os.path.join(path, "log_random_4_22_to_5_08_pure.csv"), usecols=LOG_COLUMNS)
    if max_rows and len(logs) > max_rows:
        logs = logs.sample(max_rows, random_state=sample_seed).sort_index()
    users = pd.read_csv(os.path.join(path, "user_features_pure.csv"), usecols=USER_COLUMNS)
    videos = pd.read_csv(os.path.join(path, "video_features_basic_pure.csv"), usecols=VIDEO_COLUMNS)
    frame = logs.merge(users, on="user_id", how="left", validate="many_to_one")
    frame = frame.merge(videos, on="video_id", how="left", validate="many_to_one")

    interaction_date = pd.to_datetime(frame["date"].astype("Int64").astype(str), format="%Y%m%d", errors="coerce")
    upload_date = pd.to_datetime(frame["upload_dt"], errors="coerce")
    frame["hour"] = pd.to_numeric(frame["hourmin"], errors="coerce") // 100
    frame["day"] = interaction_date.dt.dayofweek.astype("Int64").astype(str)
    frame["duration_seconds"] = pd.to_numeric(frame["duration_ms"], errors="coerce") / 1000.0
    frame["video_duration_seconds"] = pd.to_numeric(frame["video_duration"], errors="coerce") / 1000.0
    frame["video_age_days"] = (interaction_date - upload_date).dt.days

    loss = pd.to_numeric(frame["is_hate"], errors="coerce").fillna(0).to_numpy(dtype=np.float32)
    value = (pd.to_numeric(frame["play_time_ms"], errors="coerce").fillna(0) / 1000.0).to_numpy(dtype=np.float32)
    valid = (loss >= 0) & (value >= 0)
    frame = frame.loc[valid].reset_index(drop=True)
    loss = loss[valid]
    value = value[valid]
    group = frame["user_active_degree"].fillna("UNKNOWN").astype(str).to_numpy()
    return frame[NUMERIC + CATEGORICAL], loss, value, group


def transform_split(frame: pd.DataFrame, train, val, test):
    return transform_frame_split(frame, NUMERIC, CATEGORICAL, train, val, test)


def batches(n: int, batch_size: int, generator: torch.Generator):
    order = torch.randperm(n, generator=generator)
    for start in range(0, n, batch_size):
        yield order[start : start + batch_size]


def train_wbce(x, value, loss, seed, epochs, batch_size, device):
    torch.manual_seed(seed)
    model = MLP(x.shape[1]).to(device)
    total = value + loss
    target = loss / np.maximum(total, 1e-8)
    weight = total / max(float(total.mean()), 1e-8)
    xt = torch.from_numpy(x).to(device)
    yt = torch.from_numpy(target.astype(np.float32)).to(device)
    wt = torch.from_numpy(weight.astype(np.float32)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    gen = torch.Generator(device="cpu").manual_seed(seed)
    for _ in range(epochs):
        for idx in batches(len(x), batch_size, gen):
            idx = idx.to(device)
            loss_value = F.binary_cross_entropy_with_logits(model(xt[idx]), yt[idx], weight=wt[idx])
            opt.zero_grad()
            loss_value.backward()
            opt.step()
    return model


def train_log_outcome(x, outcome, seed, epochs, batch_size, device):
    torch.manual_seed(seed)
    model = MLP(x.shape[1]).to(device)
    xt = torch.from_numpy(x).to(device)
    yt = torch.from_numpy(np.log1p(np.maximum(outcome, 0.0)).astype(np.float32)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    gen = torch.Generator(device="cpu").manual_seed(seed)
    for _ in range(epochs):
        for idx in batches(len(x), batch_size, gen):
            idx = idx.to(device)
            loss_value = F.mse_loss(model(xt[idx]), yt[idx])
            opt.zero_grad()
            loss_value.backward()
            opt.step()
    return model


def predict(model, x, device, batch_size):
    model.eval()
    parts = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            xt = torch.from_numpy(x[start : start + batch_size]).to(device)
            parts.append(model(xt).detach().cpu().numpy())
    return np.concatenate(parts)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--max-rows", type=int, default=300_000)
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print_provenance(["numpy", "pandas", "torch"])
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print({"torch": torch.__version__, "device": str(device), "cuda": torch.cuda.is_available()}, flush=True)
    frame, loss, value, group = load_data(args.input_dir, args.max_rows, 42)
    print({"n": len(frame), "harm_count": int(loss.sum()), "value_sum": float(value.sum())}, flush=True)
    indices = np.arange(len(frame))
    rows = []
    for seed in range(args.seeds):
        train_val, test = stratified_train_test_split(indices, test_size=0.20, random_state=seed, stratify=loss)
        train, val = stratified_train_test_split(train_val, test_size=0.20, random_state=10_000 + seed, stratify=loss[train_val])
        x_train, x_val, x_test = transform_split(frame, train, val, test)
        score_model = train_wbce(x_train, value[train], loss[train], 810_000 + seed, args.epochs, args.batch_size, device)
        value_model = train_log_outcome(x_train, value[train], 820_000 + seed, args.epochs, args.batch_size, device)
        loss_model = train_log_outcome(x_train, loss[train], 830_000 + seed, args.epochs, args.batch_size, device)
        for split_name, split_idx, x_part in (("validation", val, x_val), ("test", test, x_test)):
            ratio_score = predict(score_model, x_part, device, args.batch_size)
            a_hat = np.maximum(np.expm1(predict(value_model, x_part, device, args.batch_size)), 0.0)
            b_hat = np.maximum(np.expm1(predict(loss_model, x_part, device, args.batch_size)), 0.0)
            for i, idx in enumerate(split_idx):
                rows.append(
                    {
                        "dataset": "KuaiRand-Pure-random-sample",
                        "seed": seed,
                        "split": split_name,
                        "group": group[idx],
                        "value": float(value[idx]),
                        "loss": float(loss[idx]),
                        "ratio_score": float(ratio_score[i]),
                        "a_hat": float(a_hat[i]),
                        "b_hat": float(b_hat[i]),
                    }
                )
        print(f"seed {seed} rows={len(rows)}", flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    columns = ["dataset", "seed", "split", "group", "value", "loss", "ratio_score", "a_hat", "b_hat"]
    with open(args.output, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    print(f"saved {args.output} rows={len(rows)}", flush=True)


if __name__ == "__main__":
    main()
