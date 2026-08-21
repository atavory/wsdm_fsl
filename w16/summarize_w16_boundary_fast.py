"""Vectorized W16 W-BCE fairness-boundary summarizer.

This emits the same schema as summarize_w16_boundary.py, but uses pandas and
NumPy to avoid repeated pure-Python sorting during threshold searches.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
from collections.abc import Iterable

import numpy as np
import pandas as pd

from provenance import print_provenance


REQUIRED_COLUMNS = {
    "dataset",
    "seed",
    "split",
    "group",
    "value",
    "loss",
    "ratio_score",
    "a_hat",
    "b_hat",
}


def sanitized(value: object) -> str:
    return str(value).strip().lower().replace(" ", "_").replace("-", "_")


def group_indices(values: np.ndarray) -> dict[str, np.ndarray]:
    return {
        str(group): np.flatnonzero(values == group)
        for group in sorted({str(value) for value in values})
    }


def descending_order(scores: np.ndarray) -> np.ndarray:
    return np.lexsort((np.arange(len(scores)), -scores))


def threshold_for_count(scores: np.ndarray, fraction: float) -> float:
    if not 0.0 <= fraction <= 1.0:
        raise ValueError(f"count fraction must be in [0, 1], got {fraction}")
    if len(scores) == 0 or fraction <= 0.0:
        return math.inf
    ordered = np.sort(scores)[::-1]
    index = min(math.ceil(fraction * len(ordered)) - 1, len(ordered) - 1)
    return float(ordered[max(index, 0)])


def threshold_for_mass(scores: np.ndarray, mass: np.ndarray, fraction: float) -> float:
    if len(scores) == 0 or fraction <= 0.0:
        return math.inf
    total_mass = float(mass.sum())
    if total_mass <= 0.0:
        return math.inf
    order = descending_order(scores)
    cumulative = np.cumsum(mass[order])
    index = min(int(np.searchsorted(cumulative, fraction * total_mass, side="left")), len(order) - 1)
    return float(scores[order[index]])


def selected_value_fraction_for_scores(
    scores: np.ndarray,
    values: np.ndarray,
    groups: np.ndarray,
    grouped: dict[str, np.ndarray],
    rate: float,
) -> float:
    total_value = float(values.sum())
    if total_value <= 0.0:
        return 0.0
    selected = 0.0
    for indices in grouped.values():
        threshold = threshold_for_count(scores[indices], rate)
        selected += float(values[indices][scores[indices] >= threshold].sum())
    return selected / total_value


def common_count_rate_at_value_budget(
    scores: np.ndarray,
    values: np.ndarray,
    groups: np.ndarray,
    grouped: dict[str, np.ndarray],
    target: float,
    iterations: int,
) -> float:
    low, high = 0.0, 1.0
    for _ in range(iterations):
        mid = (low + high) / 2.0
        spent = selected_value_fraction_for_scores(scores, values, groups, grouped, mid)
        if spent < target:
            low = mid
        else:
            high = mid
    return (low + high) / 2.0


def lambda_grid(a_hat: np.ndarray, b_hat: np.ndarray, points: int) -> list[float]:
    if points < 2:
        raise ValueError("--lambda-grid-points must be at least 2")
    positive_value = a_hat[a_hat > 0]
    positive_loss = b_hat[b_hat > 0]
    if len(positive_value) == 0 or len(positive_loss) == 0:
        return [0.0, 1.0]
    scale = float(np.median(positive_loss) / max(float(np.median(positive_value)), 1e-12))
    scale = max(scale, 1e-9)
    low = math.log(scale / 1_000.0)
    high = math.log(scale * 1_000.0)
    return [0.0] + [
        math.exp(low + (high - low) * index / (points - 1))
        for index in range(points)
    ]


def choose_affine_lambda(
    a_hat: np.ndarray,
    b_hat: np.ndarray,
    value: np.ndarray,
    groups: np.ndarray,
    grouped: dict[str, np.ndarray],
    rate: float,
    target: float,
    grid_points: int,
) -> tuple[float, float]:
    best_lambda = 0.0
    best_error = math.inf
    for lambda_value in lambda_grid(a_hat, b_hat, grid_points):
        scores = b_hat - lambda_value * a_hat
        spent = selected_value_fraction_for_scores(scores, value, groups, grouped, rate)
        error = abs(spent - target)
        if error < best_error:
            best_lambda = lambda_value
            best_error = error
    return best_lambda, best_error


def apply_thresholds(
    scores: np.ndarray,
    groups: np.ndarray,
    thresholds: dict[str, float],
) -> np.ndarray:
    selected = np.zeros(len(scores), dtype=bool)
    for group, threshold in thresholds.items():
        mask = groups == group
        selected[mask] = scores[mask] >= threshold
    return selected


def evaluate_policy(
    selected: np.ndarray,
    value: np.ndarray,
    loss: np.ndarray,
    groups: np.ndarray,
    target: float,
    policy: str,
    extra: dict[str, float | str],
) -> dict[str, float | str]:
    selected_value = float(value[selected].sum())
    selected_loss = float(loss[selected].sum())
    row: dict[str, float | str] = {
        "row_type": "policy",
        "target": target,
        "policy": policy,
        "loss_capture": selected_loss / max(float(loss.sum()), 1e-12),
        "value_forfeiture": selected_value / max(float(value.sum()), 1e-12),
        "decline_rate": float(selected.mean()) if len(selected) else 0.0,
        **extra,
    }
    value_rates: list[float] = []
    loss_rates: list[float] = []
    count_rates: list[float] = []
    for group, indices in group_indices(groups).items():
        key = sanitized(group)
        group_selected = selected[indices]
        group_value = value[indices]
        group_loss = loss[indices]
        value_rate = float(group_value[group_selected].sum()) / max(float(group_value.sum()), 1e-12)
        loss_rate = float(group_loss[group_selected].sum()) / max(float(group_loss.sum()), 1e-12)
        count_rate = float(group_selected.mean()) if len(group_selected) else 0.0
        row[f"value_rate_{key}"] = value_rate
        row[f"loss_rate_{key}"] = loss_rate
        row[f"count_rate_{key}"] = count_rate
        value_rates.append(value_rate)
        loss_rates.append(loss_rate)
        count_rates.append(count_rate)
    row["value_gap"] = max(value_rates) - min(value_rates)
    row["loss_gap"] = max(loss_rates) - min(loss_rates)
    row["count_gap"] = max(count_rates) - min(count_rates)
    return row


def gap_rows(
    target: float,
    lambda_value: float,
    affine_selected: np.ndarray,
    wbce_selected: np.ndarray,
    value: np.ndarray,
    loss: np.ndarray,
    a_hat: np.ndarray,
    b_hat: np.ndarray,
    groups: np.ndarray,
) -> Iterable[dict[str, float | str]]:
    difference = affine_selected.astype(float) - wbce_selected.astype(float)
    predicted_utility = b_hat - lambda_value * a_hat
    realized_utility = loss - lambda_value * value
    for group, indices in group_indices(groups).items():
        yield {
            "row_type": "representability_gap",
            "target": target,
            "policy": "count_affine_vs_count_wbce",
            "group": group,
            "lambda": lambda_value,
            "predicted_gap": float((predicted_utility[indices] * difference[indices]).sum()) / max(len(indices), 1),
            "realized_gap": float((realized_utility[indices] * difference[indices]).sum()) / max(len(indices), 1),
            "affine_decline_rate": float(affine_selected[indices].mean()) if len(indices) else 0.0,
            "wbce_decline_rate": float(wbce_selected[indices].mean()) if len(indices) else 0.0,
        }


def summarize_block(
    dataset: str,
    seed: object,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    targets: list[float],
    lambda_grid_points: int,
    count_search_iterations: int,
) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    v_group = validation["group"].to_numpy(dtype=str)
    t_group = test["group"].to_numpy(dtype=str)
    v_grouped = group_indices(v_group)
    v_value = validation["value"].to_numpy(float)
    v_score = validation["ratio_score"].to_numpy(float)
    v_a_hat = validation["a_hat"].to_numpy(float)
    v_b_hat = validation["b_hat"].to_numpy(float)
    t_value = test["value"].to_numpy(float)
    t_loss = test["loss"].to_numpy(float)
    t_score = test["ratio_score"].to_numpy(float)
    t_a_hat = test["a_hat"].to_numpy(float)
    t_b_hat = test["b_hat"].to_numpy(float)

    for target in targets:
        value_thresholds = {
            group: threshold_for_mass(v_score[indices], v_value[indices], target)
            for group, indices in v_grouped.items()
        }
        value_selected = apply_thresholds(t_score, t_group, value_thresholds)
        rows.append(
            {
                "dataset": dataset,
                "seed": seed,
                **evaluate_policy(
                    value_selected,
                    t_value,
                    t_loss,
                    t_group,
                    target,
                    "value_parity_wbce",
                    {"lambda": "", "validation_budget_error": 0.0},
                ),
            }
        )

        count_rate = common_count_rate_at_value_budget(
            v_score,
            v_value,
            v_group,
            v_grouped,
            target,
            count_search_iterations,
        )
        count_thresholds = {
            group: threshold_for_count(v_score[indices], count_rate)
            for group, indices in v_grouped.items()
        }
        count_selected = apply_thresholds(t_score, t_group, count_thresholds)
        rows.append(
            {
                "dataset": dataset,
                "seed": seed,
                **evaluate_policy(
                    count_selected,
                    t_value,
                    t_loss,
                    t_group,
                    target,
                    "count_parity_wbce",
                    {
                        "count_rate": count_rate,
                        "lambda": "",
                        "validation_budget_error": 0.0,
                    },
                ),
            }
        )

        lambda_value, budget_error = choose_affine_lambda(
            v_a_hat,
            v_b_hat,
            v_value,
            v_group,
            v_grouped,
            count_rate,
            target,
            lambda_grid_points,
        )
        v_affine = v_b_hat - lambda_value * v_a_hat
        t_affine = t_b_hat - lambda_value * t_a_hat
        affine_thresholds = {
            group: threshold_for_count(v_affine[indices], count_rate)
            for group, indices in v_grouped.items()
        }
        affine_selected = apply_thresholds(t_affine, t_group, affine_thresholds)
        rows.append(
            {
                "dataset": dataset,
                "seed": seed,
                **evaluate_policy(
                    affine_selected,
                    t_value,
                    t_loss,
                    t_group,
                    target,
                    "count_parity_affine",
                    {
                        "count_rate": count_rate,
                        "lambda": lambda_value,
                        "validation_budget_error": budget_error,
                    },
                ),
            }
        )
        for row in gap_rows(
            target,
            lambda_value,
            affine_selected,
            count_selected,
            t_value,
            t_loss,
            t_a_hat,
            t_b_hat,
            t_group,
        ):
            rows.append({"dataset": dataset, "seed": seed, **row})
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--targets",
        default="0.01,0.025,0.05,0.10,0.15,0.20,0.30",
    )
    parser.add_argument("--lambda-grid-points", type=int, default=401)
    parser.add_argument("--count-search-iterations", type=int, default=35)
    return parser.parse_args()


def write_rows(path: str, rows: list[dict[str, float | str]]) -> None:
    preferred = [
        "dataset",
        "seed",
        "row_type",
        "target",
        "policy",
        "group",
        "lambda",
        "count_rate",
        "validation_budget_error",
        "loss_capture",
        "value_forfeiture",
        "decline_rate",
        "value_gap",
        "loss_gap",
        "count_gap",
        "predicted_gap",
        "realized_gap",
        "affine_decline_rate",
        "wbce_decline_rate",
    ]
    fieldnames = list(preferred)
    extras = sorted({key for row in rows for key in row} - set(fieldnames))
    fieldnames.extend(extras)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    print_provenance(["numpy", "pandas"])
    data = pd.read_csv(args.input)
    missing = REQUIRED_COLUMNS - set(data.columns)
    if missing:
        raise ValueError(f"missing required columns: {sorted(missing)}")
    data["split"] = data["split"].astype(str).str.lower()
    data["group"] = data["group"].astype(str)
    targets = [float(value) for value in args.targets.split(",") if value.strip()]
    rows: list[dict[str, float | str]] = []
    for (dataset, seed), block in data.groupby(["dataset", "seed"], sort=True):
        validation = block[block["split"].isin(["validation", "val"])]
        test = block[block["split"].isin(["test", "outer_test"])]
        if len(validation) == 0 or len(test) == 0:
            raise ValueError(f"{dataset} seed {seed} lacks validation or test rows")
        print(
            {
                "dataset": dataset,
                "seed": seed,
                "validation": len(validation),
                "test": len(test),
            },
            flush=True,
        )
        rows.extend(
            summarize_block(
                str(dataset),
                seed,
                validation,
                test,
                targets,
                args.lambda_grid_points,
                args.count_search_iterations,
            )
        )
    write_rows(args.output, rows)


if __name__ == "__main__":
    main()
