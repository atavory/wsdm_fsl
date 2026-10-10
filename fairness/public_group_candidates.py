"""Promote public group-bearing datasets to enriched W16 candidate logs.

This wrapper reuses the frozen cross-domain loader/model utilities but emits
the W-BCE score and validation-frozen fixed-policy decisions required by the
group-complaint evaluator.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np

from public_w16_candidates_registry import (  # noqa: E402
    CRITERIA,
    REGISTRY,
    _adam_l1p,
    account_nested_split,
    fit_wbce,
    predict_l1p,
    stratified_split,
    threshold_for_mass,
    transform,
)


def load_euhealth_gender(path: str, n: int, seed: int):
    import rdata
    import pandas as pd

    obj = rdata.read_rda(path)
    frame = pd.DataFrame(obj["euhealthinsurance"])
    amount_columns = [column for column in frame.columns if str(column).startswith("amt_")]
    total_amount = (
        frame[amount_columns]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0.0)
        .clip(lower=0.0)
        .sum(axis=1)
        .to_numpy(dtype=np.float64)
    )
    policy_years = pd.to_numeric(frame["policy_years"], errors="coerce").fillna(0.0)
    gender = frame["gender"].astype(str)
    valid = (policy_years > 0.0) & gender.isin(["F", "M"])
    frame = frame.loc[valid].reset_index(drop=True)
    total_amount = total_amount[valid.to_numpy()]
    policy_years = policy_years.loc[valid].to_numpy(dtype=np.float64)
    group = gender.loc[valid].to_numpy()
    label = (total_amount > 0.0).astype(np.float64)
    value = policy_years * (1.0 - label)
    loss = total_amount
    features = frame[["relation", "policy_years", "age_at_inception"]].copy()
    if n and len(features) > n:
        rng = np.random.default_rng(seed)
        keep = rng.choice(len(features), n, replace=False)
        features = features.iloc[keep].reset_index(drop=True)
        label, value, loss, group = label[keep], value[keep], loss[keep], group[keep]
    return (
        features,
        ["policy_years", "age_at_inception"],
        ["relation"],
        label,
        value,
        loss,
        group,
        "euhealth_gender",
        None,
    )


def load_uci_credit(path: str, n: int, seed: int, attribute: str):
    import pandas as pd

    raw = pd.read_csv(path)
    rename = {
        "X1": "limit_bal",
        "X2": "sex",
        "X3": "education",
        "X4": "marriage",
        "X5": "age",
        "Y": "default_next_month",
    }
    frame = raw.rename(columns=rename).copy()
    if "default_next_month" not in frame:
        raise RuntimeError("missing UCI credit target column")
    label = pd.to_numeric(frame["default_next_month"], errors="coerce")
    limit = pd.to_numeric(frame["limit_bal"], errors="coerce")
    sex = pd.to_numeric(frame["sex"], errors="coerce")
    age = pd.to_numeric(frame["age"], errors="coerce")
    valid = label.isin([0, 1]) & (limit > 0) & sex.isin([1, 2]) & age.notna()
    frame = frame.loc[valid].reset_index(drop=True)
    label = label.loc[valid].to_numpy(dtype=np.float64)
    limit = limit.loc[valid].to_numpy(dtype=np.float64)
    sex = sex.loc[valid].astype(int)
    age = age.loc[valid].to_numpy(dtype=np.float64)
    if attribute == "sex":
        group = sex.map({1: "Male", 2: "Female"}).to_numpy()
        dataset_name = "uci_credit_sex"
    elif attribute == "age":
        group = np.where(age < 40.0, "under_40", "40_plus")
        dataset_name = "uci_credit_age"
    else:
        raise ValueError(f"unknown UCI credit attribute: {attribute}")
    value = limit * (1.0 - label)
    loss = limit * label
    feature_columns = [
        "limit_bal",
        "X6",
        "X7",
        "X8",
        "X9",
        "X10",
        "X11",
        "X12",
        "X13",
        "X14",
        "X15",
        "X16",
        "X17",
        "X18",
        "X19",
        "X20",
        "X21",
        "X22",
        "X23",
    ]
    features = frame[feature_columns].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    if n and len(features) > n:
        rng = np.random.default_rng(seed)
        keep = rng.choice(len(features), n, replace=False)
        features = features.iloc[keep].reset_index(drop=True)
        label, value, loss, group = label[keep], value[keep], loss[keep], group[keep]
    return (
        features,
        feature_columns,
        [],
        label,
        value,
        loss,
        group,
        dataset_name,
        None,
    )


def load_german_credit(path: str, n: int, seed: int, attribute: str):
    import pandas as pd

    columns = [
        "checking_status",
        "duration",
        "credit_history",
        "purpose",
        "credit_amount",
        "savings",
        "employment",
        "installment_rate",
        "personal_status_sex",
        "other_debtors",
        "residence_since",
        "property",
        "age",
        "other_installment",
        "housing",
        "existing_credits",
        "job",
        "liable_people",
        "telephone",
        "foreign_worker",
        "credit_class",
    ]
    frame = pd.read_csv(path, sep=r"\s+", names=columns, engine="python")
    credit_class = pd.to_numeric(frame["credit_class"], errors="coerce")
    amount = pd.to_numeric(frame["credit_amount"], errors="coerce")
    age = pd.to_numeric(frame["age"], errors="coerce")
    personal = frame["personal_status_sex"].astype(str)
    valid = credit_class.isin([1, 2]) & (amount > 0.0) & age.notna() & personal.notna()
    frame = frame.loc[valid].reset_index(drop=True)
    credit_class = credit_class.loc[valid].to_numpy(dtype=np.float64)
    amount = amount.loc[valid].to_numpy(dtype=np.float64)
    age = age.loc[valid].to_numpy(dtype=np.float64)
    personal = personal.loc[valid]
    label = (credit_class == 2.0).astype(np.float64)
    if attribute == "sex":
        group = personal.map(
            {
                "A91": "Male",
                "A92": "Female",
                "A93": "Male",
                "A94": "Male",
                "A95": "Female",
            }
        ).to_numpy()
        dataset_name = "german_credit_sex"
        categorical = [
            column
            for column in [
                "checking_status",
                "credit_history",
                "purpose",
                "savings",
                "employment",
                "other_debtors",
                "property",
                "other_installment",
                "housing",
                "job",
                "telephone",
                "foreign_worker",
            ]
        ]
        numeric = [
            "duration",
            "credit_amount",
            "installment_rate",
            "residence_since",
            "age",
            "existing_credits",
            "liable_people",
        ]
    elif attribute == "age":
        group = np.where(age < 35.0, "under_35", "35_plus")
        dataset_name = "german_credit_age"
        categorical = [
            "checking_status",
            "credit_history",
            "purpose",
            "savings",
            "employment",
            "personal_status_sex",
            "other_debtors",
            "property",
            "other_installment",
            "housing",
            "job",
            "telephone",
            "foreign_worker",
        ]
        numeric = [
            "duration",
            "credit_amount",
            "installment_rate",
            "residence_since",
            "existing_credits",
            "liable_people",
        ]
    else:
        raise ValueError(f"unknown German credit attribute: {attribute}")
    if any(pd.isna(value) for value in group):
        raise RuntimeError("German credit sex group mapping produced missing values")
    value = amount * (1.0 - label)
    loss = amount * label
    features = frame[numeric + categorical].copy()
    if n and len(features) > n:
        rng = np.random.default_rng(seed)
        keep = rng.choice(len(features), n, replace=False)
        features = features.iloc[keep].reset_index(drop=True)
        label, value, loss, group = label[keep], value[keep], loss[keep], group[keep]
    return (
        features,
        numeric,
        categorical,
        label,
        value,
        loss,
        group,
        dataset_name,
        None,
    )


LOCAL_REGISTRY = {
    **REGISTRY,
    "euhealth_gender": load_euhealth_gender,
    "german_credit_age": lambda path, n, seed: load_german_credit(path, n, seed, "age"),
    "german_credit_sex": lambda path, n, seed: load_german_credit(path, n, seed, "sex"),
    "uci_credit_age": lambda path, n, seed: load_uci_credit(path, n, seed, "age"),
    "uci_credit_sex": lambda path, n, seed: load_uci_credit(path, n, seed, "sex"),
}


FIELDS: tuple[str, ...] = (
    "dataset",
    "family",
    "seed",
    "target",
    "sample_split",
    "group",
    "value",
    "loss",
    "score",
    "ratio_score",
    "estimated_value",
    "estimated_loss",
    "selected_global_value",
    "selected_equal_group_value",
    "selected_equal_group_decline",
    "selected_equal_group_protection",
    "selected_criterion_wbce_ratio",
    "selected_criterion_harm",
    "selected_criterion_net_harm",
    "account",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=sorted(LOCAL_REGISTRY))
    parser.add_argument("--input", required=True)
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--n", type=int, default=50_000)
    parser.add_argument("--target", type=float, default=0.10)
    parser.add_argument("--net-lambda", type=float, default=1.0)
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def format_float(value: float) -> str:
    if math.isnan(value):
        return ""
    if math.isinf(value):
        return "inf" if value > 0 else "-inf"
    return f"{value:.10g}"


def threshold_map(groups: np.ndarray, threshold: float) -> dict[str, float]:
    return {str(group): threshold for group in np.unique(groups)}


def select_by_thresholds(
    scores: np.ndarray,
    groups: np.ndarray,
    thresholds: dict[str, float],
) -> np.ndarray:
    return np.asarray(
        [
            score >= thresholds.get(str(group), float("inf")) and score > -float("inf")
            for score, group in zip(scores, groups, strict=True)
        ],
        dtype=bool,
    )


def equal_decline_thresholds_at_value_budget(
    scores: np.ndarray,
    groups: np.ndarray,
    value: np.ndarray,
    value_fraction: float,
) -> dict[str, float]:
    curves: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for group in np.unique(groups):
        mask = groups == group
        order = np.argsort(-scores[mask], kind="stable")
        ordered_scores = scores[mask][order]
        tie_boundaries = np.r_[
            np.flatnonzero(ordered_scores[:-1] != ordered_scores[1:]),
            len(ordered_scores) - 1,
        ]
        curves[str(group)] = (
            ordered_scores,
            np.cumsum(value[mask][order], dtype=np.float64),
            tie_boundaries,
        )
    low, high = 0.0, 1.0
    thresholds: dict[str, float] = {}
    for _ in range(30):
        decline_fraction = (low + high) / 2.0
        spent_value = 0.0
        thresholds = {}
        for group, (ordered_scores, cumulative_value, tie_boundaries) in curves.items():
            index = min(
                int(np.ceil(decline_fraction * len(ordered_scores))) - 1,
                len(ordered_scores) - 1,
            )
            index = max(index, 0)
            thresholds[group] = float(ordered_scores[index])
            tie_end = tie_boundaries[np.searchsorted(tie_boundaries, index)]
            spent_value += float(cumulative_value[tie_end])
        spent = spent_value / max(float(value.sum()), 1e-12)
        if spent < value_fraction:
            low = decline_fraction
        else:
            high = decline_fraction
    return thresholds


def equal_protection_thresholds_at_value_budget(
    scores: np.ndarray,
    groups: np.ndarray,
    value: np.ndarray,
    loss: np.ndarray,
    value_fraction: float,
) -> dict[str, float]:
    curves: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
    for group in np.unique(groups):
        mask = groups == group
        order = np.argsort(-scores[mask], kind="stable")
        ordered_scores = scores[mask][order]
        tie_boundaries = np.r_[
            np.flatnonzero(ordered_scores[:-1] != ordered_scores[1:]),
            len(ordered_scores) - 1,
        ]
        curves[str(group)] = (
            ordered_scores,
            np.cumsum(loss[mask][order], dtype=np.float64),
            np.cumsum(value[mask][order], dtype=np.float64),
            tie_boundaries,
        )
    low, high = 0.0, 1.0
    thresholds: dict[str, float] = {}
    for _ in range(30):
        protection_fraction = (low + high) / 2.0
        spent_value = 0.0
        thresholds = {}
        for group, (
            ordered_scores,
            cumulative_loss,
            cumulative_value,
            tie_boundaries,
        ) in curves.items():
            if cumulative_loss[-1] <= 0:
                index = len(ordered_scores) - 1
            else:
                index = min(
                    int(
                        np.searchsorted(
                            cumulative_loss,
                            protection_fraction * cumulative_loss[-1],
                            side="left",
                        )
                    ),
                    len(ordered_scores) - 1,
                )
            thresholds[group] = float(ordered_scores[index])
            tie_end = tie_boundaries[np.searchsorted(tie_boundaries, index)]
            spent_value += float(cumulative_value[tie_end])
        spent = spent_value / max(float(value.sum()), 1e-12)
        if spent < value_fraction:
            low = protection_fraction
        else:
            high = protection_fraction
    return thresholds


def policy_thresholds(
    scores: np.ndarray,
    groups: np.ndarray,
    value: np.ndarray,
    loss: np.ndarray,
    target: float,
) -> dict[str, dict[str, float]]:
    global_threshold = threshold_for_mass(scores, value, target)
    value_thresholds = {
        str(group): threshold_for_mass(
            scores[groups == group],
            value[groups == group],
            target,
        )
        for group in np.unique(groups)
    }
    return {
        "global_value": threshold_map(groups, global_threshold),
        "equal_group_value": value_thresholds,
        "equal_group_decline": equal_decline_thresholds_at_value_budget(
            scores,
            groups,
            value,
            target,
        ),
        "equal_group_protection": equal_protection_thresholds_at_value_budget(
            scores,
            groups,
            value,
            loss,
            target,
        ),
    }


def criterion_thresholds(
    scores: np.ndarray,
    estimated_value: np.ndarray,
    estimated_loss: np.ndarray,
    value: np.ndarray,
    target: float,
    net_lambda: float,
) -> dict[str, float]:
    criteria = {
        "wbce_ratio": scores,
        "harm": estimated_loss,
        "net_harm": estimated_loss - net_lambda * estimated_value,
    }
    return {
        criterion: threshold_for_mass(criterion_scores, value, target)
        for criterion, criterion_scores in criteria.items()
    }


def criterion_scores(
    scores: np.ndarray,
    estimated_value: np.ndarray,
    estimated_loss: np.ndarray,
    net_lambda: float,
) -> dict[str, np.ndarray]:
    return {
        "wbce_ratio": scores,
        "harm": estimated_loss,
        "net_harm": estimated_loss - net_lambda * estimated_value,
    }


def main() -> None:
    args = parse_args()
    frame, numeric, categorical, label, value, loss, group, dataset_name, account = (
        LOCAL_REGISTRY[args.dataset](args.input, args.n, 42)
    )
    print(
        f"{dataset_name}: n={len(frame)} pos_rate={label.mean():.4f} "
        f"groups={dict(zip(*np.unique(group, return_counts=True), strict=True))}",
        flush=True,
    )
    rows: list[dict[str, str]] = []
    indices = np.arange(len(frame))
    for seed in range(args.seeds):
        if account is None:
            outer, test_idx = stratified_split(indices, label, 0.20, seed)
            train_idx, val_idx = stratified_split(outer, label, 0.20, 10_000 + seed)
        else:
            train_idx, val_idx, test_idx = account_nested_split(account, label, seed)

        x_train, x_val, x_test = transform(
            frame,
            numeric,
            categorical,
            train_idx,
            val_idx,
            test_idx,
        )
        wbce_coef, wbce_intercept = fit_wbce(
            x_train,
            loss[train_idx],
            value[train_idx],
            500_000 + seed,
            args.steps,
        )
        value_coef, value_intercept = _adam_l1p(
            x_train,
            value[train_idx],
            args.steps,
            seed + 1,
        )
        loss_coef, loss_intercept = _adam_l1p(
            x_train,
            loss[train_idx],
            args.steps,
            seed + 2,
        )

        val_scores = x_val @ wbce_coef + wbce_intercept
        val_estimated_value = np.maximum(predict_l1p(x_val, value_coef, value_intercept), 1e-9)
        val_estimated_loss = predict_l1p(x_val, loss_coef, loss_intercept)
        policies = policy_thresholds(
            val_scores,
            group[val_idx],
            value[val_idx],
            loss[val_idx],
            args.target,
        )
        criteria = criterion_thresholds(
            val_scores,
            val_estimated_value,
            val_estimated_loss,
            value[val_idx],
            args.target,
            args.net_lambda,
        )

        for split, split_idx, design in (
            ("validation", val_idx, x_val),
            ("test", test_idx, x_test),
        ):
            split_scores = design @ wbce_coef + wbce_intercept
            split_estimated_value = np.maximum(
                predict_l1p(design, value_coef, value_intercept),
                1e-9,
            )
            split_estimated_loss = predict_l1p(design, loss_coef, loss_intercept)
            split_group = group[split_idx]
            policy_selected = {
                policy: select_by_thresholds(split_scores, split_group, thresholds)
                for policy, thresholds in policies.items()
            }
            crit_scores = criterion_scores(
                split_scores,
                split_estimated_value,
                split_estimated_loss,
                args.net_lambda,
            )
            criterion_selected = {
                criterion: crit_scores[criterion] >= threshold
                for criterion, threshold in criteria.items()
            }
            for local_idx, row_idx in enumerate(split_idx):
                rows.append(
                    {
                        "dataset": dataset_name,
                        "family": "natural",
                        "seed": str(seed),
                        "target": format_float(args.target),
                        "sample_split": split,
                        "group": str(split_group[local_idx]),
                        "value": format_float(float(value[row_idx])),
                        "loss": format_float(float(loss[row_idx])),
                        "score": format_float(float(split_scores[local_idx])),
                        "ratio_score": format_float(float(split_scores[local_idx])),
                        "estimated_value": format_float(
                            float(split_estimated_value[local_idx])
                        ),
                        "estimated_loss": format_float(
                            float(split_estimated_loss[local_idx])
                        ),
                        "selected_global_value": str(
                            int(policy_selected["global_value"][local_idx])
                        ),
                        "selected_equal_group_value": str(
                            int(policy_selected["equal_group_value"][local_idx])
                        ),
                        "selected_equal_group_decline": str(
                            int(policy_selected["equal_group_decline"][local_idx])
                        ),
                        "selected_equal_group_protection": str(
                            int(policy_selected["equal_group_protection"][local_idx])
                        ),
                        "selected_criterion_wbce_ratio": str(
                            int(criterion_selected["wbce_ratio"][local_idx])
                        ),
                        "selected_criterion_harm": str(
                            int(criterion_selected["harm"][local_idx])
                        ),
                        "selected_criterion_net_harm": str(
                            int(criterion_selected["net_harm"][local_idx])
                        ),
                        "account": (
                            str(account[row_idx])
                            if account is not None
                            else ""
                        ),
                    }
                )
        print(f"  seed {seed} done", flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"saved {args.output} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
