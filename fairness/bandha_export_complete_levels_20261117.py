"""Freeze the completed two-objective experiment and refinement moments."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path):
    with path.open() as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def freeze_csvs(args):
    initial = json.loads(
        (args.run / "bandha_VALUE_CONTINUATION_SOURCE.json").read_text()
    )
    for name, expected in initial["capture_hashes"].items():
        if digest(args.run / name) != expected:
            raise ValueError("Capture result changed: " + name)
    value_provenance = json.loads(
        (args.run / "bandha_VALUE_T_PROVENANCE.json").read_text()
    )
    for key, name in (
        ("C_freeze_sha256", "bandha_VALUE_C_FREEZE.json"),
        ("T_sha256", "bandha_VALUE_T.csv"),
        ("C_sha256", "bandha_VALUE_C.csv"),
    ):
        if digest(args.run / name) != value_provenance[key]:
            raise ValueError("Value result differs from its provenance: " + name)
    for name in (
        "bandha_C_FREEZE.json",
        "bandha_T_PROVENANCE.json",
        "bandha_VALUE_C_FREEZE.json",
        "bandha_VALUE_T_PROVENANCE.json",
        "bandha_VALUE_CONTINUATION_SOURCE.json",
        "bandha_VALUE_PROGRESS.json",
    ):
        shutil.copyfile(args.run / name, args.output / name)
    tables = {}
    for part in ("C", "T"):
        rows = []
        for source_name in (f"bandha_{part}.csv", f"bandha_VALUE_{part}.csv"):
            selected = read_csv(args.run / source_name)
            if len(selected) != 1440:
                raise ValueError("Expected 240 cases, two variants and three units")
            rows.extend(selected)
        rows.sort(
            key=lambda row: (
                row["dataset"],
                int(row["seed"]),
                row["family"],
                row["unit"],
            )
        )
        write_csv(args.output / f"bandha_{part}.csv", rows)
        tables[part] = {
            (r["dataset"] + "__" + r["seed"], r["family"]): r
            for r in rows
            if r["unit"] == "value"
        }
    return tables, initial


def validated_seal(args, cell, objective, freeze, baseline_hash):
    directory = args.run / ("bandha_" + cell) / objective
    path = directory / "bandha_C_seal.json"
    if digest(path) != freeze["seals"][cell][objective]:
        raise ValueError("C seal changed: " + cell + "/" + objective)
    seal = json.loads(path.read_text())
    if seal["original_predictions_sha256"] != baseline_hash:
        raise ValueError("Original predictions changed: " + cell)
    if digest(directory / "bandha_predictions.npz") != seal["predictions_sha256"]:
        raise ValueError("Fitted predictions changed: " + cell)
    return directory, seal


def fit_row(cell, objective, seal, directory):
    resumed = directory / "bandha_continuation_checks.json"
    checks = json.loads(resumed.read_text()) if resumed.exists() else {}
    return {
        "cell": cell,
        "objective": objective,
        "source": seal["case"]["source"],
        "gap_converged": seal["gap_converged"],
        "best_gap": seal["best_gap"],
        "oracle_calls": seal["oracle_calls"],
        "n_classifiers": seal["n_classifiers"],
        "C_status": seal["Ccal_certificate"]["status"],
        "C_active_max_violation": seal["Ccal_certificate"]["active_max_violation"],
        "feature_reconstruction_max_error": max(
            seal["feature_reconstruction"].values()
        ),
        "refitted_in_continuation": resumed.exists(),
        "partial_C_max_absolute_difference": max(
            checks.get("partial_C_max_absolute_difference", {}).values(), default=0
        ),
    }


def baseline(array, case):
    return case["baseline_alpha"] * (
        (array[:, 1] >= case["baseline_threshold"]) & (array[:, 1] > 0)
    )


def check_value_row(array, before, after, row):
    q = ~((array[:, 6] > 0.5) & (array[:, 7] > 0.5))
    values = {
        "baseline_capture": float(np.sum(before * array[:, 3])),
        "capture": float(np.sum(after * array[:, 3])),
        "objection_mass": float(np.sum(after * q * array[:, 2])),
        "added_objection_mass": float(
            np.sum(np.maximum(after - before, 0) * q * array[:, 2])
        ),
        "blocked_value": float(np.sum(after * array[:, 2])),
    }
    for key, found in values.items():
        expected = float(row[key])
        if abs(found - expected) > 1e-8 * max(1, abs(expected)):
            raise ValueError("Frozen aggregate differs from row probabilities: " + key)


def refinement_rows(array, before, after, labels, info):
    q = ~((array[:, 6] > 0.5) & (array[:, 7] > 0.5))
    result = []
    for group in np.unique(labels):
        selected = labels == group
        weight = array[selected, 2]
        denominator = float(weight.sum())
        if denominator <= 0:
            continue
        result.append(
            {
                **info,
                "group": int(group),
                "n_rows": int(selected.sum()),
                "total_value": denominator,
                "before_objection_value": float(
                    np.sum(before[selected] * q[selected] * weight)
                ),
                "after_objection_value": float(
                    np.sum(after[selected] * q[selected] * weight)
                ),
            }
        )
    return result


def export_case(args, cell, freezes, tables):
    original_path = args.baseline_run / "cells" / cell / "PREDICTIONS.npz"
    baseline_hash = digest(original_path)
    diagnostics, moments = [], []
    with np.load(original_path) as saved:
        cuts = np.unique(np.quantile(saved["V"][:, 1], [0.2, 0.4, 0.6, 0.8]))
        for objective in ("capture", "value"):
            directory, seal = validated_seal(
                args, cell, objective, freezes[objective], baseline_hash
            )
            diagnostics.append(fit_row(cell, objective, seal, directory))
            with np.load(directory / "bandha_predictions.npz") as predictions:
                for part in ("C", "T"):
                    array = saved[part]
                    before = baseline(array, seal["case"])
                    group = array[:, 0].astype(int)
                    partitions = {
                        "baseline_support": 2 * group + (before > 0).astype(int),
                        "V_score_quintile": group * (len(cuts) + 1)
                        + np.searchsorted(cuts, array[:, 1], side="right"),
                    }
                    for key, suffix in (("raw", "raw"), ("cal", "Ccal")):
                        family = f"levels_{objective}_{suffix}"
                        after = predictions[part + "_" + key]
                        check_value_row(
                            array, before, after, tables[part][cell, family]
                        )
                        info = {
                            "cell": cell,
                            "source": seal["case"]["source"],
                            "part": part,
                            "family": family,
                        }
                        for name, labels in partitions.items():
                            moments.extend(
                                refinement_rows(
                                    array,
                                    before,
                                    after,
                                    labels,
                                    {**info, "partition": name},
                                )
                            )
    return diagnostics, moments


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "run", "baseline-run", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    tables, initial = freeze_csvs(args)
    freezes = {
        name: json.loads((args.run / file).read_text())
        for name, file in (
            ("capture", "bandha_C_FREEZE.json"),
            ("value", "bandha_VALUE_C_FREEZE.json"),
        )
    }
    if (
        set(freezes["capture"]["seals"]) != set(freezes["value"]["seals"])
        or len(freezes["value"]["seals"]) != 240
    ):
        raise ValueError("The completed arms do not cover the same 240 cases")
    diagnostics, moments = [], []
    for index, cell in enumerate(sorted(freezes["value"]["seals"]), 1):
        fits, groups = export_case(args, cell, freezes, tables)
        diagnostics.extend(fits)
        moments.extend(groups)
        if index % 60 == 0:
            print(f"Verified and exported {index}/240 cells", flush=True)
    write_csv(args.output / "bandha_fit_diagnostics.csv", diagnostics)
    write_csv(args.output / "bandha_refinement_moments.csv", moments)
    metadata = {
        "cases_per_objective": 240,
        "objectives": ["capture", "value"],
        "value_C_seals_reused": initial["initial_value_C_sealed"],
        "value_fits_completed_in_continuation": len(initial["missing_cases"]),
        "capture_results_examined_before_value_continuation": True,
        "exporter_sha256": digest(Path(__file__)),
        "files": {
            p.name: digest(p)
            for p in sorted(args.output.iterdir())
            if p.is_file() and p.name != "bandha_EXPORT.json"
        },
    }
    (args.output / "bandha_EXPORT.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    print("Frozen completed comparison:", args.output, flush=True)


if __name__ == "__main__":
    main()
