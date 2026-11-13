"""Export boosted V diagnostics and freeze/replay ratio-restoration sensitivity."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import bandha_external_criteria_fit as fit_module
import bandha_external_criteria_io as io
import bandha_xgboost_rerun as rerun
import numpy as np
from bandha_external_criteria_study import business_metrics, unit_metrics


def identities(args):
    return {
        "source": io.digest(Path(__file__)),
        "fit_module": io.digest(Path(fit_module.__file__)),
        "run_freeze": io.digest(args.run / "C_FREEZE.json"),
    }


def fit(args, cases):
    decisions = {}
    vrows = []
    rounds = []
    for case in cases:
        label = case["dataset"] + "__" + case["seed"]
        directory = args.run / "cells" / label
        record = json.loads((directory / "C_SEAL.json").read_text())
        boosted = record["case"]
        arrays = np.load(directory / "PREDICTIONS.npz")
        v, c = arrays["V"], arrays["C"]
        threshold = fit_module.plugin_threshold(v)
        expanded = fit_module.fit_common(v, c, boosted, "native_wbce", False, threshold)
        decisions[label] = {
            "case": boosted,
            "policy": expanded,
            "predictions_sha256": record["predictions_sha256"],
        }
        vrows.extend(
            [
                r
                for r in rerun.evaluate(v, boosted, record["decision"], "V")
                if r["policy"] == "common" and r["unit"] == "value"
            ]
        )
        for name in ("wbce", "value", "loss"):
            rounds.append(
                {
                    "dataset": case["dataset"],
                    "seed": case["seed"],
                    "model": name,
                    "rounds": record["models"][name]["rounds"],
                    "best_inner_metric": record["models"][name]["best_inner_metric"],
                }
            )
    rerun.csv_rows(args.output / "V_metrics.csv", vrows)
    rerun.csv_rows(args.output / "model_rounds.csv", rounds)
    rerun.save(
        args.output / "C_FREEZE.json",
        {
            "identities": identities(args),
            "decisions": decisions,
            "T_outcomes_evaluated": False,
        },
    )
    print("Ratio-restoration decisions sealed:", len(decisions), flush=True)


def outcome(array, case, p, family):
    p0 = io.baseline(array, case)
    q = io.criterion(array, "native_wbce", 0.0)
    row = {
        "dataset": case["dataset"],
        "seed": case["seed"],
        "source": case["source"],
        "family": family,
    }
    row.update(business_metrics(array, p0, p))
    row.update(unit_metrics(array, p0, p, q, "value"))
    increases = [
        row[f"group_{g}_level"] - row[f"group_{g}_baseline"]
        for g in (0, 1)
        if row[f"group_{g}_level"] != ""
    ]
    row["group_increase"] = bool(max(increases, default=0.0) > 1e-10)
    return row


def test(args):
    seal = json.loads((args.output / "C_FREEZE.json").read_text())
    if seal["identities"] != identities(args):
        raise ValueError("Sensitivity code or source freeze changed")
    rows = []
    for label, decision in seal["decisions"].items():
        directory = args.run / "cells" / label
        if io.digest(directory / "PREDICTIONS.npz") != decision["predictions_sha256"]:
            raise ValueError("Predictions changed")
        arrays = np.load(directory / "PREDICTIONS.npz")
        record = json.loads((directory / "C_SEAL.json").read_text())
        current = rerun.probability(
            arrays["T"], decision["case"], record["decision"], "common"
        )
        expanded = io.probability(arrays["T"], decision["case"], decision["policy"])
        rows.append(outcome(arrays["T"], decision["case"], current, "original_repair"))
        rows.append(outcome(arrays["T"], decision["case"], expanded, "rebuilt_union"))
    rerun.csv_rows(args.output / "T.csv", rows)
    rerun.save(
        args.output / "T_PROVENANCE.json",
        {
            "C_freeze_sha256": io.digest(args.output / "C_FREEZE.json"),
            "T_sha256": io.digest(args.output / "T.csv"),
            "no_T_parameter_selection": True,
        },
    )
    print("Ratio-restoration T rows:", len(rows), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("fit", "test"))
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(args.data_root / "scripts"))
    manifest = json.loads(
        (
            args.data_root
            / "results/csvs/bandha_external_criteria_inputs_20261113.json"
        ).read_text()
    )
    if args.phase == "fit":
        fit(args, manifest["cases"])
    else:
        test(args)


if __name__ == "__main__":
    main()
