"""Run reproducible WSDM baseline experiments.

This runner intentionally leaves the legacy KDD experiment scripts untouched.
It separates the exact W-BCE objective from practical damping and compares it
with pointwise, pairwise, ratio-regression, and dual-regression alternatives.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Dataset:
    x: np.ndarray
    y: np.ndarray
    leakage: np.ndarray
    revenue: np.ndarray
    phi1: np.ndarray
    phi2: np.ndarray


def generate_synthetic(n: int, seed: int, scenario: str) -> Dataset:
    """Generate outcomes with known conditional-mean ratios."""
    rng = np.random.default_rng(seed)
    d = 20
    x = rng.normal(size=(n, d)).astype(np.float32)
    wy = rng.normal(scale=0.3, size=d)
    wl = rng.normal(scale=0.5, size=10)
    wr = rng.normal(scale=0.5, size=10)

    py = 1.0 / (1.0 + np.exp(-(x @ wy)))
    mu_l = x[:, :10] @ wl
    mu_r = x[:, 10:] @ wr
    sigma_l = np.full(n, 0.5)
    if scenario == "independent":
        sigma_r = np.full(n, 0.5)
    elif scenario == "heteroskedastic":
        # Revenue uncertainty is predictable from x. Increasing sigma_r raises
        # E[R|x], lowering E[L|x]/E[R|x], while it raises E[L/R|x].
        # This creates a controlled disagreement between the two targets.
        sigma_r = 0.15 + 1.35 / (1.0 + np.exp(-2.0 * x[:, 0]))
        # Give L and R the same conditional log-location. Their ratio of
        # conditional means is therefore driven only by predictable variance.
        # MSE on log(L/R) has a constant population target in this setting.
        common = x[:, 1:10] @ wl[1:]
        mu_l = common
        mu_r = common
    else:
        raise ValueError(f"unknown scenario: {scenario}")
    leakage = np.exp(mu_l + rng.normal(size=n) * sigma_l)
    revenue = np.exp(mu_r + rng.normal(size=n) * sigma_r)
    y = rng.binomial(1, py).astype(np.float32)

    mean_l = np.exp(mu_l + 0.5 * sigma_l**2)
    mean_r = np.exp(mu_r + 0.5 * sigma_r**2)
    phi1 = mean_l / mean_r
    phi2 = (py * mean_l) / ((1.0 - py) * mean_r)
    return Dataset(
        x=x,
        y=y,
        leakage=leakage.astype(np.float32),
        revenue=revenue.astype(np.float32),
        phi1=phi1.astype(np.float32),
        phi2=phi2.astype(np.float32),
    )


class MLP(nn.Module):
    def __init__(self, d: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


DEFAULT_METHODS = [
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
]


def make_model(d: int, seed: int) -> MLP:
    torch.manual_seed(seed)
    return MLP(d)


def batches(n: int, batch_size: int, generator: torch.Generator):
    order = torch.randperm(n, generator=generator)
    for start in range(0, n, batch_size):
        yield order[start : start + batch_size]


def fit_bce(
    x: np.ndarray,
    target: np.ndarray,
    weight: np.ndarray,
    seed: int,
    epochs: int,
    batch_size: int,
) -> MLP:
    model = make_model(x.shape[1], seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    xt = torch.from_numpy(x)
    target_t = torch.from_numpy(target.astype(np.float32))
    weight_t = torch.from_numpy(weight.astype(np.float32))
    generator = torch.Generator().manual_seed(seed)
    for _ in range(epochs):
        for idx in batches(len(xt), batch_size, generator):
            loss = F.binary_cross_entropy_with_logits(
                model(xt[idx]), target_t[idx], weight=weight_t[idx]
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return model


def fit_xgb_bce(
    x: np.ndarray,
    target: np.ndarray,
    weight: np.ndarray,
    seed: int,
):
    try:
        import xgboost as xgb
    except ImportError as error:
        raise RuntimeError(
            "XGBoost methods require the xgboost Python package"
        ) from error

    model = xgb.XGBRegressor(
        objective="binary:logistic",
        eval_metric="logloss",
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        min_child_weight=10,
        subsample=1.0,
        colsample_bytree=1.0,
        reg_lambda=1.0,
        random_state=seed,
        n_jobs=-1,
        tree_method="hist",
        verbosity=0,
    )
    model.fit(
        x,
        target.astype(np.float32),
        sample_weight=weight.astype(np.float32),
        verbose=False,
    )
    return model


def fit_regression(
    x: np.ndarray,
    target: np.ndarray,
    seed: int,
    epochs: int,
    batch_size: int,
    poisson: bool = False,
) -> MLP:
    model = make_model(x.shape[1], seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    xt = torch.from_numpy(x)
    target_t = torch.from_numpy(target.astype(np.float32))
    generator = torch.Generator().manual_seed(seed)
    for _ in range(epochs):
        for idx in batches(len(xt), batch_size, generator):
            prediction = model(xt[idx])
            if poisson:
                loss = F.poisson_nll_loss(prediction, target_t[idx], log_input=True)
            else:
                loss = F.mse_loss(prediction, target_t[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return model


def fit_ranknet(
    x: np.ndarray,
    relevance: np.ndarray,
    seed: int,
    epochs: int,
    batch_size: int,
    pairs_per_batch: int,
) -> MLP:
    """Pairwise logistic RankNet with random within-batch pairs."""
    model = make_model(x.shape[1], seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    xt = torch.from_numpy(x)
    relevance_t = torch.from_numpy(relevance.astype(np.float32))
    generator = torch.Generator().manual_seed(seed)
    for _ in range(epochs):
        for idx in batches(len(xt), batch_size, generator):
            if len(idx) < 2:
                continue
            scores = model(xt[idx])
            rel = relevance_t[idx]
            left = torch.randint(len(idx), (pairs_per_batch,), generator=generator)
            right = torch.randint(len(idx), (pairs_per_batch,), generator=generator)
            direction = torch.sign(rel[left] - rel[right])
            keep = direction != 0
            if not torch.any(keep):
                continue
            margin = direction[keep] * (scores[left[keep]] - scores[right[keep]])
            loss = -F.logsigmoid(margin).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return model


def predict(model, x: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict") and not isinstance(model, nn.Module):
        return model.predict(x)
    model.eval()
    with torch.no_grad():
        return model(torch.from_numpy(x)).numpy()


def financial_arrays(data: Dataset, objective: int) -> tuple[np.ndarray, np.ndarray]:
    if objective == 1:
        return data.leakage, data.revenue
    return data.y * data.leakage, (1.0 - data.y) * data.revenue


def metrics(scores: np.ndarray, a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    order = np.argsort(-scores, kind="stable")
    a_sorted = a[order].astype(np.float64)
    b_sorted = b[order].astype(np.float64)
    total_a = a_sorted.sum()
    total_b = b_sorted.sum()
    cum_a = np.concatenate(([0.0], np.cumsum(a_sorted) / total_a))
    cum_b = np.concatenate(([0.0], np.cumsum(b_sorted) / total_b))
    result = {"gini": float(2.0 * np.trapezoid(cum_a, cum_b) - 1.0)}
    for budget in (0.01, 0.05, 0.10):
        result[f"capture_at_{int(100 * budget)}pct_value"] = float(
            np.interp(budget, cum_b, cum_a)
        )
    return result


def normalize(train: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = train.mean(axis=0)
    std = train.std(axis=0) + 1e-8
    return ((train - mean) / std).astype(np.float32), ((test - mean) / std).astype(np.float32)


def run_method(
    name: str,
    x_train: np.ndarray,
    x_test: np.ndarray,
    a_train: np.ndarray,
    b_train: np.ndarray,
    y_train: np.ndarray,
    seed: int,
    epochs: int,
    batch_size: int,
    pairs_per_batch: int,
) -> np.ndarray:
    total = a_train + b_train
    if name == "BCE":
        model = fit_bce(x_train, y_train, np.ones_like(y_train), seed, epochs, batch_size)
        return predict(model, x_test)
    if name == "BCE(XGB)":
        model = fit_xgb_bce(x_train, y_train, np.ones_like(y_train), seed)
        return predict(model, x_test)
    if name in {"W-BCE(raw)", "W-BCE(log)"}:
        target = a_train / np.maximum(total, 1e-8)
        weight = total if name == "W-BCE(raw)" else np.log1p(total)
        weight = weight / max(weight.mean(), 1e-8)
        model = fit_bce(x_train, target, weight, seed, epochs, batch_size)
        return predict(model, x_test)
    if name in {"W-BCE-XGB(raw)", "W-BCE-XGB(log)"}:
        target = a_train / np.maximum(total, 1e-8)
        weight = total if name == "W-BCE-XGB(raw)" else np.log1p(total)
        weight = weight / max(weight.mean(), 1e-8)
        model = fit_xgb_bce(x_train, target, weight, seed)
        return predict(model, x_test)
    if name == "Ratio regression":
        target = np.log((a_train + 1e-3) / (b_train + 1e-3))
        model = fit_regression(x_train, target, seed, epochs, batch_size)
        return predict(model, x_test)
    if name in {"RankNet(A)", "RankNet(A/B)"}:
        relevance = a_train if name == "RankNet(A)" else np.log((a_train + 1e-3) / (b_train + 1e-3))
        model = fit_ranknet(
            x_train, relevance, seed, epochs, batch_size, pairs_per_batch
        )
        return predict(model, x_test)
    if name in {"Dual Poisson", "Plug-in (Poisson)"}:
        model_a = fit_regression(x_train, a_train, seed, epochs, batch_size, poisson=True)
        model_b = fit_regression(x_train, b_train, seed + 10_000, epochs, batch_size, poisson=True)
        return predict(model_a, x_test) - predict(model_b, x_test)
    raise ValueError(name)


def run(args: argparse.Namespace) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for seed in range(args.seeds):
        data = generate_synthetic(args.n, seed, args.scenario)
        split = int(0.8 * args.n)
        x_train, x_test = normalize(data.x[:split], data.x[split:])
        y_train = data.y[:split]
        test_data = Dataset(
            x=x_test,
            y=data.y[split:],
            leakage=data.leakage[split:],
            revenue=data.revenue[split:],
            phi1=data.phi1[split:],
            phi2=data.phi2[split:],
        )
        for objective in (1, 2):
            a_all, b_all = financial_arrays(data, objective)
            a_train, b_train = a_all[:split], b_all[:split]
            a_test, b_test = financial_arrays(test_data, objective)
            oracle = test_data.phi1 if objective == 1 else test_data.phi2
            for name in args.methods + ["Oracle"]:
                started = time.perf_counter()
                if name == "Oracle":
                    scores = oracle
                else:
                    scores = run_method(
                        name,
                        x_train,
                        x_test,
                        a_train,
                        b_train,
                        y_train,
                        args.model_seed + seed,
                        args.epochs,
                        args.batch_size,
                        args.pairs_per_batch,
                    )
                elapsed = time.perf_counter() - started
                row = {
                    "scenario": args.scenario,
                    "seed": seed,
                    "objective": objective,
                    "method": name,
                    "seconds": elapsed,
                    **metrics(scores, a_test, b_test),
                }
                rows.append(row)
                print(json.dumps(row), flush=True)
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=100_000)
    parser.add_argument(
        "--scenario",
        choices=("independent", "heteroskedastic"),
        default="independent",
    )
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--pairs-per-batch", type=int, default=2048)
    parser.add_argument("--model-seed", type=int, default=10_000)
    parser.add_argument("--methods", nargs="+", default=DEFAULT_METHODS)
    parser.add_argument(
        "--output",
        default=os.path.join("results", "csvs", "wsdm_baselines.csv"),
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    frame = run(arguments)
    os.makedirs(os.path.dirname(arguments.output), exist_ok=True)
    frame.to_csv(arguments.output, index=False)
    print(f"saved {arguments.output}")
