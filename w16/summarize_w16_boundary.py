"""Summarize the W16 W-BCE fairness-boundary comparison.

Input rows are per candidate, produced by a runnable experiment runner.  The
runner must provide validation and test rows with outcomes and model scores:

dataset, seed, split, group, value, loss, ratio_score, a_hat, b_hat

This script selects all policy thresholds on validation rows and evaluates the
frozen policies on test rows.  It is a deterministic CSV transform; it does not
train models or fetch data, and it intentionally uses only the Python standard
library so it can run in minimal submission environments.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import statistics
from collections import defaultdict
from collections.abc import Iterable

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


def number(row: dict[str, str], column: str) -> float:
    return float(row[column])


def sanitized(value: object) -> str:
    return str(value).strip().lower().replace(" ", "_").replace("-", "_")


def groups(rows: list[dict[str, str]]) -> list[str]:
    return sorted({row["group"] for row in rows})


def threshold_for_count(scores: list[float], fraction: float) -> float:
    if not 0.0 <= fraction <= 1.0:
        raise ValueError(f"count fraction must be in [0, 1], got {fraction}")
    if not scores or fraction <= 0.0:
        return math.inf
    ordered = sorted(scores, reverse=True)
    index = min(math.ceil(fraction * len(ordered)) - 1, len(ordered) - 1)
    return ordered[max(index, 0)]


def threshold_for_mass(
    rows: list[dict[str, str]],
    score_column: str,
    mass_column: str,
    fraction: float,
) -> float:
    if not rows or fraction <= 0.0:
        return math.inf
    total_mass = sum(number(row, mass_column) for row in rows)
    if total_mass <= 0.0:
        return math.inf
    ordered = sorted(
        enumerate(rows),
        key=lambda item: (-number(item[1], score_column), item[0]),
    )
    target = fraction * total_mass
    cumulative = 0.0
    for _, row in ordered:
        cumulative += number(row, mass_column)
        if cumulative >= target:
            return number(row, score_column)
    return number(ordered[-1][1], score_column)


def by_group(rows: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    result: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        result[row["group"]].append(row)
    return result


def equal_value_thresholds(
    validation: list[dict[str, str]],
    score_column: str,
    target: float,
) -> dict[str, float]:
    return {
        group: threshold_for_mass(part, score_column, "value", target)
        for group, part in by_group(validation).items()
    }


def count_thresholds(
    rows: list[dict[str, str]],
    score_column: str,
    rate: float,
) -> dict[str, float]:
    return {
        group: threshold_for_count([number(row, score_column) for row in part], rate)
        for group, part in by_group(rows).items()
    }


def apply_group_thresholds(
    rows: list[dict[str, str]],
    score_column: str,
    thresholds: dict[str, float],
) -> list[bool]:
    return [
        number(row, score_column) >= thresholds.get(row["group"], math.inf)
        for row in rows
    ]


def selected_value_fraction(
    rows: list[dict[str, str]],
    score_column: str,
    rate: float,
) -> float:
    total_value = sum(number(row, "value") for row in rows)
    if total_value <= 0.0:
        return 0.0
    selected_value = 0.0
    for part in by_group(rows).values():
        threshold = threshold_for_count([number(row, score_column) for row in part], rate)
        selected_value += sum(
            number(row, "value")
            for row in part
            if number(row, score_column) >= threshold
        )
    return selected_value / total_value


def common_count_rate_at_value_budget(
    validation: list[dict[str, str]],
    score_column: str,
    target: float,
) -> float:
    low, high = 0.0, 1.0
    for _ in range(35):
        mid = (low + high) / 2.0
        spent = selected_value_fraction(validation, score_column, mid)
        if spent < target:
            low = mid
        else:
            high = mid
    return (low + high) / 2.0


def lambda_grid(rows: list[dict[str, str]], points: int) -> list[float]:
    if points < 2:
        raise ValueError("--lambda-grid-points must be at least 2")
    positive_value = [number(row, "a_hat") for row in rows if number(row, "a_hat") > 0]
    positive_loss = [number(row, "b_hat") for row in rows if number(row, "b_hat") > 0]
    if not positive_value or not positive_loss:
        return [0.0, 1.0]
    scale = statistics.median(positive_loss) / max(statistics.median(positive_value), 1e-12)
    scale = max(scale, 1e-9)
    values = [0.0]
    low = math.log(scale / 1_000.0)
    high = math.log(scale * 1_000.0)
    for index in range(points):
        values.append(math.exp(low + (high - low) * index / (points - 1)))
    return values


def add_affine_score(rows: list[dict[str, str]], lambda_value: float) -> None:
    for row in rows:
        row["_affine_score"] = str(
            number(row, "b_hat") - lambda_value * number(row, "a_hat")
        )


def selected_value_fraction_for_scores(
    rows: list[dict[str, str]],
    scores: list[float],
    rate: float,
) -> float:
    total_value = sum(number(row, "value") for row in rows)
    if total_value <= 0.0:
        return 0.0
    grouped: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for row, score in zip(rows, scores):
        grouped[row["group"]].append((score, number(row, "value")))
    selected_value = 0.0
    for part in grouped.values():
        threshold = threshold_for_count([score for score, _ in part], rate)
        selected_value += sum(value for score, value in part if score >= threshold)
    return selected_value / total_value


def choose_affine_lambda(
    validation: list[dict[str, str]],
    rate: float,
    target: float,
    grid_points: int,
) -> tuple[float, float]:
    best_lambda = 0.0
    best_error = math.inf
    for lambda_value in lambda_grid(validation, grid_points):
        scores = [
            number(row, "b_hat") - lambda_value * number(row, "a_hat")
            for row in validation
        ]
        spent = selected_value_fraction_for_scores(validation, scores, rate)
        error = abs(spent - target)
        if error < best_error:
            best_lambda = lambda_value
            best_error = error
    return best_lambda, best_error


def evaluate_policy(
    selected: list[bool],
    test: list[dict[str, str]],
    target: float,
    policy: str,
    extra: dict[str, float | str],
) -> dict[str, float | str]:
    value_total = sum(number(row, "value") for row in test)
    loss_total = sum(number(row, "loss") for row in test)
    selected_value = sum(number(row, "value") for keep, row in zip(selected, test) if keep)
    selected_loss = sum(number(row, "loss") for keep, row in zip(selected, test) if keep)
    row: dict[str, float | str] = {
        "row_type": "policy",
        "target": target,
        "policy": policy,
        "loss_capture": selected_loss / max(loss_total, 1e-12),
        "value_forfeiture": selected_value / max(value_total, 1e-12),
        "decline_rate": sum(selected) / max(len(selected), 1),
        **extra,
    }
    value_rates: list[float] = []
    loss_rates: list[float] = []
    count_rates: list[float] = []
    for group in groups(test):
        key = sanitized(group)
        mask = [candidate["group"] == group for candidate in test]
        selected_group = [keep and in_group for keep, in_group in zip(selected, mask)]
        group_rows = [candidate for candidate, in_group in zip(test, mask) if in_group]
        group_value = sum(number(candidate, "value") for candidate in group_rows)
        group_loss = sum(number(candidate, "loss") for candidate in group_rows)
        group_selected_value = sum(
            number(candidate, "value")
            for candidate, keep in zip(test, selected_group)
            if keep
        )
        group_selected_loss = sum(
            number(candidate, "loss")
            for candidate, keep in zip(test, selected_group)
            if keep
        )
        value_rate = group_selected_value / max(group_value, 1e-12)
        loss_rate = group_selected_loss / max(group_loss, 1e-12)
        count_rate = sum(selected_group) / max(len(group_rows), 1)
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
    test: list[dict[str, str]],
    target: float,
    lambda_value: float,
    affine_selected: list[bool],
    wbce_selected: list[bool],
) -> Iterable[dict[str, float | str]]:
    for group in groups(test):
        values = []
        realized_values = []
        affine_count = 0
        wbce_count = 0
        total = 0
        for row, affine_keep, wbce_keep in zip(test, affine_selected, wbce_selected):
            if row["group"] != group:
                continue
            total += 1
            difference = float(affine_keep) - float(wbce_keep)
            predicted_utility = number(row, "b_hat") - lambda_value * number(row, "a_hat")
            realized_utility = number(row, "loss") - lambda_value * number(row, "value")
            values.append(predicted_utility * difference)
            realized_values.append(realized_utility * difference)
            affine_count += int(affine_keep)
            wbce_count += int(wbce_keep)
        yield {
            "row_type": "representability_gap",
            "target": target,
            "policy": "count_affine_vs_count_wbce",
            "group": group,
            "lambda": lambda_value,
            "predicted_gap": sum(values) / max(total, 1),
            "realized_gap": sum(realized_values) / max(total, 1),
            "affine_decline_rate": affine_count / max(total, 1),
            "wbce_decline_rate": wbce_count / max(total, 1),
        }


def summarize_block(
    dataset: str,
    seed: str,
    validation: list[dict[str, str]],
    test: list[dict[str, str]],
    targets: list[float],
    lambda_grid_points: int,
) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    for target in targets:
        value_thresholds = equal_value_thresholds(validation, "ratio_score", target)
        value_selected = apply_group_thresholds(test, "ratio_score", value_thresholds)
        rows.append(
            {
                "dataset": dataset,
                "seed": seed,
                **evaluate_policy(
                    value_selected,
                    test,
                    target,
                    "value_parity_wbce",
                    {"lambda": "", "validation_budget_error": 0.0},
                ),
            }
        )

        count_rate = common_count_rate_at_value_budget(validation, "ratio_score", target)
        count_threshold_map = count_thresholds(validation, "ratio_score", count_rate)
        count_selected = apply_group_thresholds(test, "ratio_score", count_threshold_map)
        rows.append(
            {
                "dataset": dataset,
                "seed": seed,
                **evaluate_policy(
                    count_selected,
                    test,
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
            validation,
            count_rate,
            target,
            lambda_grid_points,
        )
        validation_affine = [dict(row) for row in validation]
        test_affine = [dict(row) for row in test]
        add_affine_score(validation_affine, lambda_value)
        add_affine_score(test_affine, lambda_value)
        affine_threshold_map = count_thresholds(
            validation_affine,
            "_affine_score",
            count_rate,
        )
        affine_selected = apply_group_thresholds(
            test_affine,
            "_affine_score",
            affine_threshold_map,
        )
        rows.append(
            {
                "dataset": dataset,
                "seed": seed,
                **evaluate_policy(
                    affine_selected,
                    test,
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
            test,
            target,
            lambda_value,
            affine_selected,
            count_selected,
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
        help="Comma-separated aggregate value-budget fractions.",
    )
    parser.add_argument(
        "--lambda-grid-points",
        type=int,
        default=401,
        help="Number of positive affine-lambda grid points.",
    )
    return parser.parse_args()


def read_rows(path: str) -> list[dict[str, str]]:
    with open(path, newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("input CSV has no header")
        missing = REQUIRED_COLUMNS - set(reader.fieldnames)
        if missing:
            raise ValueError(f"missing required columns: {sorted(missing)}")
        return [dict(row, split=row["split"].lower(), group=str(row["group"])) for row in reader]


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
    print_provenance([])
    data = read_rows(args.input)
    targets = [float(value) for value in args.targets.split(",") if value.strip()]
    blocks: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in data:
        blocks[(row["dataset"], row["seed"])].append(row)

    rows: list[dict[str, float | str]] = []
    for (dataset, seed), block in sorted(blocks.items()):
        validation = [
            row for row in block if row["split"] in {"validation", "val"}
        ]
        test = [
            row for row in block if row["split"] in {"test", "outer_test"}
        ]
        if not validation or not test:
            raise ValueError(f"{dataset} seed {seed} lacks validation or test rows")
        rows.extend(
            summarize_block(
                dataset,
                seed,
                validation,
                test,
                targets,
                args.lambda_grid_points,
            )
        )
    write_rows(args.output, rows)


if __name__ == "__main__":
    main()
