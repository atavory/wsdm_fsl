"""Fit on C, seal decisions, then assess fixed-criterion policies on T."""

from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path

import numpy as np

from bandha_external_criteria_fit import fit_case
from bandha_external_criteria_io import (
    baseline,
    criterion,
    digest,
    load_parts,
    probability,
    read_manifest,
    save_json,
)

LOGGER = logging.getLogger(__name__)


def code_hashes():
    root = Path(__file__).parent
    names = (
        "bandha_external_criteria_io.py",
        "bandha_external_criteria_fit.py",
        "bandha_external_criteria_study.py",
    )
    return {name: digest(root / name) for name in names}


def csv_output(path, rows):
    if not rows:
        raise ValueError("No result rows")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def unit_metrics(data, p0, p, q, unit):
    amount = (
        np.ones(len(data)) if unit == "count" else data[:, 2 if unit == "value" else 3]
    )
    base_mass, mass = float(np.sum(p0 * q * amount)), float(np.sum(p * q * amount))
    result = {
        "baseline_objection_mass": base_mass,
        "objection_mass": mass,
        "removal": 1 - mass / base_mass if base_mass > 0 else "",
        "added_objection_mass": float(np.sum(np.maximum(p - p0, 0) * q * amount)),
    }
    for group in (0, 1):
        mask = data[:, 0] == group
        denom = float(np.sum(amount[mask]))
        before, after = np.sum((p0 * q * amount)[mask]), np.sum((p * q * amount)[mask])
        result[f"group_{group}_denominator"] = denom
        result[f"group_{group}_baseline"] = float(before / denom) if denom > 0 else ""
        result[f"group_{group}_level"] = float(after / denom) if denom > 0 else ""
    return result


def business_metrics(data, p0, p):
    base_capture = float(np.sum(p0 * data[:, 3]))
    capture, value = float(np.sum(p * data[:, 3])), float(np.sum(p * data[:, 2]))
    total_value = float(np.sum(data[:, 2]))
    return {
        "n_rows": len(data),
        "baseline_capture": base_capture,
        "capture": capture,
        "capture_retained": capture / base_capture if base_capture > 0 else "",
        "total_value": total_value,
        "blocked_value": value,
        "blocked_value_fraction": value / total_value if total_value > 0 else "",
        "capture_target_met": capture
        >= 0.95 * base_capture - 1e-9 * max(1, base_capture),
        "value_cap_met": value <= 0.1 * total_value + 1e-9 * max(1, total_value),
    }


def results_for_case(data, case, fitted, part):
    p0, rows = baseline(data, case), []
    for family, policy in fitted["policies"].items():
        p = probability(data, case, policy)
        if np.min(p) < -1e-9 or np.max(p) > 1 + 1e-9:
            raise ValueError("Invalid probability")
        business = business_metrics(data, p0, p)
        for name in ("native", "native_wbce", "native_plugin", "native_joint"):
            q = criterion(data, name, fitted["plugin_threshold"])
            for unit in ("count", "value", "loss"):
                row = {
                    "dataset": case["dataset"],
                    "seed": case["seed"],
                    "source": case["source"],
                    "part": part,
                    "family": family,
                    "criterion": name,
                    "unit": unit,
                    "s": policy["s"],
                    "fit_status": policy["fit_status"],
                    "maximum_probability_increase": float(np.max(p - p0)),
                    "factor_error": float(np.max(np.abs(q * (p - policy["s"] * p0)))),
                }
                row.update(business)
                row.update(unit_metrics(data, p0, p, q, unit))
                rows.append(row)
    return rows


def capacity_for_case(data, case, fitted, part):
    p0, rows = baseline(data, case), []
    total_a, total_b = np.sum(data[:, 2:4], axis=0)
    base_b = np.sum(p0 * data[:, 3])
    for name in ("native", "native_wbce", "native_plugin", "native_joint"):
        q = criterion(data, name, fitted["plugin_threshold"])
        h = (~q) * (1 - p0)
        q_capture = float(np.sum(p0 * q * data[:, 3]))
        extra_b = float(np.sum(h * data[:, 3]))
        lower = (
            max(0, 1 - ((1 - 0.95) * base_b + extra_b) / q_capture)
            if q_capture > 0
            else 0
        )
        rows.append(
            {
                "dataset": case["dataset"],
                "seed": case["seed"],
                "source": case["source"],
                "part": part,
                "criterion": name,
                "n_rows": len(data),
                "nonobjector_count": int(np.sum(~q)),
                "headroom_count": float(np.sum(h)),
                "headroom_value_fraction": float(np.sum(h * data[:, 2]) / total_a),
                "headroom_loss_fraction": (
                    float(extra_b / total_b) if total_b > 0 else ""
                ),
                "nonobjector_anchor_value_fraction": float(
                    np.sum(p0 * (~q) * data[:, 2]) / total_a
                ),
                "objector_capture_share": q_capture / base_b if base_b > 0 else "",
                "necessary_s_for_capture": float(lower),
                "wbce_baseline_support_disagreements": int(
                    np.sum((p0 > 0) != (data[:, 8] > 0.5))
                ),
            }
        )
    return rows


def match_replay(data, case, fitted, evidence):
    cell = evidence[(case["dataset"], case["seed"])]
    published = probability(data, case, fitted["policies"]["published"])
    p0 = baseline(data, case)
    business = business_metrics(data, p0, published)
    expected = cell["policies"]["common_contraction"]
    comparisons = [
        (business["capture"], expected["captured_loss"]),
        (business["blocked_value"], expected["blocked_value"]),
        (business["baseline_capture"], cell["baseline_captured_loss"]),
    ]
    q = criterion(data, "native", fitted["plugin_threshold"])
    for unit in ("count", "value", "loss"):
        comparisons.append(
            (
                unit_metrics(data, p0, published, q, unit)["objection_mass"],
                expected["objection_masses"][unit],
            )
        )
    error = max(abs(a - b) / max(1.0, abs(b)) for a, b in comparisons)
    if len(data) != cell["n_T_rows"] or error > 1e-9:
        raise ValueError(
            f"Published-policy replay failed for {case['dataset']}/{case['seed']}: {error}"
        )
    return error


def fit_command(args, manifest):
    parts = load_parts(args.root, manifest, {"V", "C"})
    decisions, rows, capacities = [], [], []
    for case in manifest["cases"]:
        key = case["dataset"], case["seed"]
        v, c = parts[(key, "V")], parts[(key, "C")]
        fitted = fit_case(v, c, case)
        decisions.append({"dataset": key[0], "seed": key[1], "fitted": fitted})
        rows.extend(results_for_case(c, case, fitted, "C"))
        capacities.extend(capacity_for_case(c, case, fitted, "C"))
    csv_output(args.output / "C.csv", rows)
    csv_output(args.output / "C_capacity.csv", capacities)
    save_json(
        args.output / "C_freeze.json",
        {
            "manifest_sha256": digest(args.manifest),
            "protocol_sha256": digest(args.protocol),
            "code_sha256": code_hashes(),
            "decisions": decisions,
            "T_outcomes_parsed": False,
            "exploratory": True,
            "C_csv_sha256": digest(args.output / "C.csv"),
        },
    )
    LOGGER.info("Sealed %s C decisions before T outcomes", len(decisions))


def test_command(args, manifest):
    freeze = json.loads((args.output / "C_freeze.json").read_text())
    if (
        freeze["manifest_sha256"] != digest(args.manifest)
        or freeze["code_sha256"] != code_hashes()
    ):
        raise ValueError("C-frozen code or inputs changed")
    if freeze["protocol_sha256"] != digest(args.protocol) or freeze[
        "C_csv_sha256"
    ] != digest(args.output / "C.csv"):
        raise ValueError("C-frozen protocol or results changed")
    decisions = {(r["dataset"], r["seed"]): r["fitted"] for r in freeze["decisions"]}
    evidence = json.loads(args.evidence.read_text())
    cells = {(r["dataset"], str(r["seed"])): r for r in evidence["cell_evidence"]}
    parts = load_parts(args.root, manifest, {"T"})
    rows, capacities, errors = [], [], []
    for case in manifest["cases"]:
        key = case["dataset"], case["seed"]
        data, fitted = parts[(key, "T")], decisions[key]
        errors.append(match_replay(data, case, fitted, cells))
        rows.extend(results_for_case(data, case, fitted, "T"))
        capacities.extend(capacity_for_case(data, case, fitted, "T"))
    csv_output(args.output / "T.csv", rows)
    csv_output(args.output / "T_capacity.csv", capacities)
    save_json(
        args.output / "T_provenance.json",
        {
            "C_freeze_sha256": digest(args.output / "C_freeze.json"),
            "code_sha256": code_hashes(),
            "replay_max_relative_error": max(errors),
            "replay_cells": len(errors),
            "T_csv_sha256": digest(args.output / "T.csv"),
            "evidence_sha256": digest(args.evidence),
            "no_T_parameter_selection": True,
        },
    )
    LOGGER.info(
        "T complete: %s primary replays, max relative error %g",
        len(errors),
        max(errors),
    )


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("fit", "test"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--evidence", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = read_manifest(args.manifest, args.root)
    if args.phase == "fit":
        fit_command(args, manifest)
    else:
        if args.evidence is None:
            parser.error("test requires --evidence")
        test_command(args, manifest)


if __name__ == "__main__":
    main()
