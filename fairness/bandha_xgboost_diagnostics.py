"""Freeze V-selected penalty diagnostics, then replay on original test rows."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import bandha_external_criteria_io as io
import bandha_xgboost_rerun as rerun
import numpy as np


def data_dictionary(array, case):
    return {
        "a": array[:, 2],
        "b": array[:, 3],
        "ratio": array[:, 1],
        "ah": array[:, 4],
        "bh": array[:, 5],
        "objection": io.native(array),
        "gid": array[:, 0].astype(int),
        "names": case["groups"],
    }


def score(data, family, price, rates, high, normalized):
    if price == 0:
        return data["ratio"]
    return normalized.scores(data, family, price, "value", rates, high)


def freeze_bank(array, case, normalized):
    data = data_dictionary(array, case)
    p0 = io.baseline(array, case)
    amounts = data["a"] * data["objection"] * p0
    levels = [
        float(amounts[data["gid"] == g].sum() / data["a"][data["gid"] == g].sum())
        for g in (0, 1)
    ]
    high = int(levels[1] > levels[0])
    baseline = normalized.measure(data, p0 > 0, "value", high, case["baseline_alpha"])
    rates = [float(data["ah"][data["gid"] == g].sum() / len(array)) for g in (0, 1)]
    if min(rates) <= 0:
        return {"status": "undefined_predicted_denominator", "baseline": baseline}
    original = list(normalized.legacy.DEFAULT_LAMBDAS)
    axis = sorted(set(original + [sum(rates) * x for x in original]))
    bank = []

    def add(price):
        for family in ("account", "group"):
            values = score(data, family, price, rates, high, normalized)
            threshold, mask = normalized.threshold_mask(data, values)
            alpha = case["baseline_alpha"] if price == 0 else None
            bank.append(
                {
                    "family": family,
                    "lambda": float(price),
                    "threshold": threshold,
                    "validation": normalized.measure(data, mask, "value", high, alpha),
                }
            )

    for price in axis:
        add(price)
    previous = None
    stable = 0
    history = []
    for _extension in range(25):
        selected = {
            f"{floor}:{family}": normalized.choose(bank, family, baseline, floor)
            for floor in normalized.FLOORS
            for family in ("account", "group")
        }
        fingerprint = tuple(selected.values())
        edge = any(
            i is not None and bank[i]["lambda"] == axis[-1] for i in selected.values()
        )
        stable = stable + 1 if not edge and fingerprint == previous else 0
        history.append(
            {
                "maximum_lambda": axis[-1],
                "edge_selected": edge,
                "stable_extensions": stable,
            }
        )
        if not edge and stable >= 2:
            break
        previous = fingerprint
        axis.append(axis[-1] * 2)
        add(axis[-1])
    else:
        raise RuntimeError("Penalty bank boundary remained open")
    return {
        "status": "frozen",
        "high": high,
        "rates": rates,
        "baseline": baseline,
        "bank": bank,
        "selected": {
            family: selected[f"0.95:{family}"] for family in ("account", "group")
        },
        "lp": normalized.lp_select(
            bank,
            baseline if baseline["capture"] > 0 else {**baseline, "capture": 1.0},
            0.95 if baseline["capture"] > 0 else 0.0,
        ),
        "zero_validation_baseline_capture": baseline["capture"] == 0,
        "coverage": history,
        "axis": axis,
    }


def penalty_probability(array, case, bank, family, normalized):
    if bank["status"] != "frozen":
        return io.baseline(array, case), True
    data = data_dictionary(array, case)

    def point(index):
        row = bank["bank"][index]
        values = score(
            data, row["family"], row["lambda"], bank["rates"], bank["high"], normalized
        )
        return row["validation"]["alpha"] * (
            (values >= row["threshold"]) & (values > 0)
        )

    if family == "mixture":
        if bank["lp"] is None:
            return io.baseline(array, case), True
        return sum(w * point(i) for i, w in bank["lp"]["components"]), False
    index = bank["selected"][family]
    return (io.baseline(array, case), True) if index is None else (point(index), False)


def outcome(array, case, probability, family, fallback):
    from bandha_external_criteria_study import business_metrics, unit_metrics

    p0 = io.baseline(array, case)
    q = io.native(array)
    business = business_metrics(array, p0, probability)
    metrics = unit_metrics(array, p0, probability, q, "value")
    row = {
        "dataset": case["dataset"],
        "seed": case["seed"],
        "source": case["source"],
        "family": family,
        "baseline_fallback": fallback,
        **business,
        **metrics,
    }
    defined = all(metrics[f"group_{g}_level"] != "" for g in (0, 1))
    row["defined"] = defined
    if defined:
        before = metrics["group_1_baseline"] - metrics["group_0_baseline"]
        after = metrics["group_1_level"] - metrics["group_0_level"]
        increase = max(
            metrics[f"group_{g}_level"] - metrics[f"group_{g}_baseline"] for g in (0, 1)
        )
        row.update(
            baseline_gap=before,
            policy_gap=after,
            group_increase=increase > 1e-10,
            gap_reduced=abs(after) < abs(before) - 1e-10,
            safe_gap_reduction=abs(after) < abs(before) - 1e-10 and increase <= 1e-10,
        )
    else:
        row.update(
            baseline_gap="",
            policy_gap="",
            group_increase="",
            gap_reduced="",
            safe_gap_reduction="",
        )
    return row


def fit(args, cases, normalized):
    seals = {}
    for case in cases:
        label = case["dataset"] + "__" + case["seed"]
        directory = args.run / "cells" / label
        seal = json.loads((directory / "C_SEAL.json").read_text())
        arrays = np.load(directory / "PREDICTIONS.npz")
        bank = freeze_bank(arrays["V"], seal["case"], normalized)
        path = args.output / (label + ".json")
        rerun.save(path, bank)
        seals[label] = io.digest(path)
        print("V diagnostics sealed:", label, flush=True)
    rerun.save(
        args.output / "V_FREEZE.json",
        {
            "seals": seals,
            "code": io.digest(Path(__file__)),
            "normalized_code": io.digest(Path(normalized.__file__)),
            "C_freeze_sha256": io.digest(args.run / "C_FREEZE.json"),
            "T_outcomes_evaluated": False,
        },
    )


def test(args, cases, normalized):
    freeze = json.loads((args.output / "V_FREEZE.json").read_text())
    if freeze["code"] != io.digest(Path(__file__)) or freeze[
        "normalized_code"
    ] != io.digest(Path(normalized.__file__)):
        raise ValueError("Diagnostics code changed after freezing")
    rows = []
    for case in cases:
        label = case["dataset"] + "__" + case["seed"]
        directory = args.run / "cells" / label
        path = args.output / (label + ".json")
        if io.digest(path) != freeze["seals"][label]:
            raise ValueError("V diagnostic changed")
        bank = json.loads(path.read_text())
        seal = json.loads((directory / "C_SEAL.json").read_text())
        arrays = np.load(directory / "PREDICTIONS.npz")
        data = np.vstack((arrays["C"], arrays["T"]))
        for family in ("account", "group", "mixture"):
            p, fallback = penalty_probability(
                data, seal["case"], bank, family, normalized
            )
            rows.append(outcome(data, seal["case"], p, family, fallback))
    rerun.csv_rows(args.output / "original_test.csv", rows)
    rerun.save(
        args.output / "T_PROVENANCE.json",
        {
            "V_freeze_sha256": io.digest(args.output / "V_FREEZE.json"),
            "outcomes_sha256": io.digest(args.output / "original_test.csv"),
            "no_test_parameter_selection": True,
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("fit", "test"))
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(args.data_root / "scripts"))
    import bandha_normalized_group_prices as normalized

    manifest = json.loads(
        (
            args.data_root
            / "results/csvs/bandha_external_criteria_inputs_20261113.json"
        ).read_text()
    )
    if args.phase == "fit":
        fit(args, manifest["cases"], normalized)
    else:
        test(args, manifest["cases"], normalized)


if __name__ == "__main__":
    main()
