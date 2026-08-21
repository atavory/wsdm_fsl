"""GPU diagnostic W16 per-candidate extractor for CASdatasets inputs."""

from __future__ import annotations

import argparse
import csv
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
import pyreadr
import torch
import torch.nn as nn
import torch.nn.functional as F

from simple_preprocess import stratified_sample_indices
from simple_preprocess import stratified_train_test_split
from simple_preprocess import transform_split as transform_frame_split
from provenance import print_provenance


@dataclass
class Dataset:
    name: str
    frame: pd.DataFrame
    label: np.ndarray
    value: np.ndarray
    loss: np.ndarray
    group: np.ndarray
    numeric: list[str]
    categorical: list[str]


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


def load_australian(path: str) -> Dataset:
    frame = pyreadr.read_r(path)["ausprivauto0405"]
    loss_s = pd.to_numeric(frame["ClaimAmount"], errors="coerce").fillna(0)
    exposure_s = pd.to_numeric(frame["Exposure"], errors="coerce").fillna(0)
    label = (loss_s > 0).astype(np.float32).to_numpy()
    valid = (exposure_s > 0).to_numpy()
    numeric = ["Exposure", "VehValue"]
    categorical = ["VehAge", "VehBody", "DrivAge"]
    clean = frame.loc[valid, numeric + categorical + ["Gender"]].reset_index(drop=True)
    loss = loss_s.loc[valid].to_numpy(dtype=np.float32)
    exposure = exposure_s.loc[valid].to_numpy(dtype=np.float32)
    label = label[valid]
    value = exposure * (1.0 - label)
    return Dataset(
        "ausprivauto0405",
        clean,
        label,
        value,
        loss,
        clean["Gender"].fillna("MISSING").astype(str).to_numpy(),
        numeric,
        categorical,
    )


def load_belgian(path: str, sample_n: int, sample_seed: int) -> Dataset:
    frame = pyreadr.read_r(path)["beMTPL97"]
    loss_s = pd.to_numeric(frame["amount"], errors="coerce").fillna(0)
    exposure_s = pd.to_numeric(frame["expo"], errors="coerce").fillna(0)
    label = (loss_s > 0).astype(np.float32).to_numpy()
    valid = (exposure_s > 0).to_numpy()
    frame = frame.loc[valid].reset_index(drop=True)
    loss = loss_s.loc[valid].to_numpy(dtype=np.float32)
    exposure = exposure_s.loc[valid].to_numpy(dtype=np.float32)
    label = label[valid]
    if sample_n and sample_n < len(frame):
        idx, _ = stratified_sample_indices(
            np.arange(len(frame)),
            train_size=sample_n,
            random_state=sample_seed,
            stratify=label,
        )
        frame = frame.iloc[idx].reset_index(drop=True)
        loss = loss[idx]
        exposure = exposure[idx]
        label = label[idx]
    numeric = ["expo", "ageph", "bm", "power", "agec", "long", "lat"]
    categorical = ["coverage", "fuel", "use", "fleet"]
    value = exposure * (1.0 - label)
    return Dataset(
        "beMTPL97",
        frame[numeric + categorical + ["sex"]].reset_index(drop=True),
        label,
        value,
        loss,
        frame["sex"].fillna("MISSING").astype(str).to_numpy(),
        numeric,
        categorical,
    )


def transform_split(data: Dataset, train: np.ndarray, val: np.ndarray, test: np.ndarray):
    return transform_frame_split(data.frame, data.numeric, data.categorical, train, val, test)


def batches(n: int, batch_size: int, generator: torch.Generator):
    order = torch.randperm(n, generator=generator)
    for start in range(0, n, batch_size):
        yield order[start : start + batch_size]


def train_wbce(
    x: np.ndarray,
    value: np.ndarray,
    loss: np.ndarray,
    seed: int,
    epochs: int,
    batch_size: int,
    device: torch.device,
) -> MLP:
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
            loss_value = F.binary_cross_entropy_with_logits(
                model(xt[idx]),
                yt[idx],
                weight=wt[idx],
            )
            opt.zero_grad()
            loss_value.backward()
            opt.step()
    return model


def train_log_outcome(
    x: np.ndarray,
    outcome: np.ndarray,
    seed: int,
    epochs: int,
    batch_size: int,
    device: torch.device,
) -> MLP:
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


def predict(model: MLP, x: np.ndarray, device: torch.device, batch_size: int) -> np.ndarray:
    model.eval()
    parts = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            xt = torch.from_numpy(x[start : start + batch_size]).to(device)
            parts.append(model(xt).detach().cpu().numpy())
    return np.concatenate(parts)


def run_dataset(
    data: Dataset,
    seeds: int,
    epochs: int,
    batch_size: int,
    device: torch.device,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    indices = np.arange(len(data.frame))
    for seed in range(seeds):
        train_val, test = stratified_train_test_split(indices, test_size=0.20, random_state=seed, stratify=data.label)
        train, val = stratified_train_test_split(
            train_val,
            test_size=0.20,
            random_state=10_000 + seed,
            stratify=data.label[train_val],
        )
        x_train, x_val, x_test = transform_split(data, train, val, test)
        score_model = train_wbce(x_train, data.value[train], data.loss[train], 710_000 + seed, epochs, batch_size, device)
        value_model = train_log_outcome(x_train, data.value[train], 720_000 + seed, epochs, batch_size, device)
        loss_model = train_log_outcome(x_train, data.loss[train], 730_000 + seed, epochs, batch_size, device)
        for split_name, split_idx, x_part in (("validation", val, x_val), ("test", test, x_test)):
            ratio_score = predict(score_model, x_part, device, batch_size)
            a_hat = np.maximum(np.expm1(predict(value_model, x_part, device, batch_size)), 0.0)
            b_hat = np.maximum(np.expm1(predict(loss_model, x_part, device, batch_size)), 0.0)
            for i, idx in enumerate(split_idx):
                rows.append(
                    {
                        "dataset": data.name,
                        "seed": seed,
                        "split": split_name,
                        "group": data.group[idx],
                        "value": float(data.value[idx]),
                        "loss": float(data.loss[idx]),
                        "ratio_score": float(ratio_score[i]),
                        "a_hat": float(a_hat[i]),
                        "b_hat": float(b_hat[i]),
                    }
                )
        print(f"{data.name} seed {seed} rows={len(rows)}", flush=True)
    return rows


def write_rows(path: str, rows: list[dict[str, object]]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    columns = ["dataset", "seed", "split", "group", "value", "loss", "ratio_score", "a_hat", "b_hat"]
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--australian-rda", required=True)
    parser.add_argument("--belgian-rda", required=True)
    parser.add_argument("--datasets", default="australian,belgian")
    parser.add_argument("--belgian-n", type=int, default=100_000)
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print_provenance(["numpy", "pandas", "pyreadr", "torch"])
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print({"torch": torch.__version__, "device": str(device), "cuda": torch.cuda.is_available()}, flush=True)
    selected = {name.strip() for name in args.datasets.split(",") if name.strip()}
    rows: list[dict[str, object]] = []
    if "australian" in selected:
        rows.extend(run_dataset(load_australian(args.australian_rda), args.seeds, args.epochs, args.batch_size, device))
    if "belgian" in selected:
        rows.extend(run_dataset(load_belgian(args.belgian_rda, args.belgian_n, 42), args.seeds, args.epochs, args.batch_size, device))
    write_rows(args.output, rows)
    print(f"saved {args.output} rows={len(rows)}", flush=True)


if __name__ == "__main__":
    main()
