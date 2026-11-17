"""Replay the frozen XGBoost primary across nine cap/floor combinations."""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import hashlib
import importlib
import importlib.metadata
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np


BETAS = (0.05, 0.10, 0.20)
FLOORS = (0.90, 0.95, 0.99)
TOLERANCE = 1e-8


def digest(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def save(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    temporary.replace(path)


def csv_output(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def runtime(args):
    location = str(args.data_root / "scripts")
    if location not in sys.path:
        sys.path.insert(0, location)
    names = {
        "safe": "bandha_objection_safe_selector",
        "empirical": "bandha_empirical_cut_selector",
        "direct": "bandha_direct_constrained_selector",
        "outcomes": "bandha_fairness_group_outcomes",
        "metrics": "bandha_external_criteria_study",
        "frozen": "bandha_external_criteria_io",
        "public": "public_group_candidates",
        "legacy": "fairness_candidates",
    }
    return SimpleNamespace(
        **{key: importlib.import_module(name) for key, name in names.items()}
    )


def identity(args):
    rt = runtime(args)
    paths = {Path(__file__), *(Path(module.__file__) for module in vars(rt).values())}
    paths.add(Path(rt.metrics.__file__).with_name("bandha_external_criteria_fit.py"))
    return {
        "code": {str(path): digest(path) for path in sorted(paths)},
        "protocol_sha256": digest(
            args.data_root / "bandha_contraction_sweep_protocol_20261117.md"
        ),
        "original_C_freeze_sha256": digest(args.run / "C_FREEZE.json"),
        "manifest_sha256": digest(
            args.data_root
            / "results/csvs/bandha_external_criteria_inputs_20261113.json"
        ),
        "betas": list(BETAS),
        "floors": list(FLOORS),
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("numpy", "scipy", "pandas")
        },
        "model_fits": 0,
        "criteria": ["harm", "net_harm"],
        "net_lambda": 1,
        "fit_split": "C",
        "replacement_cut_split": "V",
        "evaluation_split": "T",
        "scope": "Exploratory replay of previously examined frozen predictions.",
    }


def label(case):
    return case["dataset"] + "__" + case["seed"]


def config(beta, rho):
    return f"beta_{beta:.2f}_rho_{rho:.2f}"


def case_source(args, case):
    cell = label(case)
    directory = args.run / "cells" / cell
    freeze = json.loads((args.run / "C_FREEZE.json").read_text())
    if digest(directory / "C_SEAL.json") != freeze["seals"][cell]:
        raise ValueError("Original C seal changed: " + cell)
    record = json.loads((directory / "C_SEAL.json").read_text())
    if digest(directory / "PREDICTIONS.npz") != record["predictions_sha256"]:
        raise ValueError("Original predictions changed: " + cell)
    if digest(directory / "C.csv") != record["C_csv_sha256"]:
        raise ValueError("Original C outcomes changed: " + cell)
    return directory, record


def parameters(rt, original, validation, beta):
    threshold = rt.public.threshold_for_mass(validation[:, 1], validation[:, 2], beta)
    spending = float(validation[validation[:, 1] >= threshold, 2].sum())
    alpha = (
        min(1.0, beta * float(validation[:, 2].sum()) / spending)
        if spending > 0
        else 1.0
    )
    thresholds = {
        "harm": rt.public.threshold_for_mass(validation[:, 5], validation[:, 2], beta),
        "net_harm": rt.public.threshold_for_mass(
            validation[:, 5] - validation[:, 4], validation[:, 2], beta
        ),
        "wbce_ratio_exp": threshold,
    }
    total_value = float(validation[:, 2].sum())
    thresholds["V_baseline_blocked_value_fraction"] = alpha * spending / total_value
    thresholds["V_harm_blocked_value_fraction"] = (
        float(validation[validation[:, 5] >= thresholds["harm"], 2].sum()) / total_value
    )
    thresholds["V_net_harm_blocked_value_fraction"] = (
        float(
            validation[
                validation[:, 5] - validation[:, 4] >= thresholds["net_harm"], 2
            ].sum()
        )
        / total_value
    )
    case = {
        key: original[key] for key in ("dataset", "seed", "source", "input", "groups")
    }
    case.update(baseline_threshold=threshold, baseline_alpha=alpha)
    if beta == 0.10:
        found = (threshold, alpha, thresholds["harm"], thresholds["net_harm"])
        old = (
            original["baseline_threshold"],
            original["baseline_alpha"],
            original["boosted_criterion_thresholds"]["harm"],
            original["boosted_criterion_thresholds"]["net_harm"],
        )
        if any(abs(a - b) > 1e-12 * max(1, abs(b)) for a, b in zip(found, old)):
            raise ValueError("Primary V thresholds do not reproduce")
    return case, thresholds


def rebuild(array, thresholds, beta):
    result = array.copy()
    result[:, 6] = array[:, 5] >= thresholds["harm"]
    result[:, 7] = array[:, 5] - array[:, 4] >= thresholds["net_harm"]
    result[:, 8] = array[:, 1] >= thresholds["wbce_ratio_exp"]
    if beta == 0.10 and not np.array_equal(array[:, 6:8], result[:, 6:8]):
        raise ValueError("Primary harm/net-harm indicators do not reproduce")
    return result


def batch(rt, array):
    return rt.empirical.ValidationBatch(
        array[:, 0].astype(int),
        array[:, 1],
        rt.frozen.native(array),
        array[:, 2],
        array[:, 3],
    )


def calibration_cell(rt, case, validation, tuning):
    scope = rt.direct.Scope(
        (case["dataset"], case["seed"], "value"),
        case["source"],
        case["input"],
        tuple(case["groups"]),
        0,
        case["baseline_threshold"],
        case["baseline_alpha"],
    )
    vc = rt.safe.build_cell(scope, batch(rt, validation))
    cb = batch(rt, tuning)
    cc = rt.safe.initial_cell(scope, cb)
    cuts = list(dict.fromkeys(point.threshold for point in vc.q0_raw_cuts))
    cc.q0_raw_cuts = []
    for threshold in cuts:
        chosen = (
            np.zeros(cb.size, dtype=bool)
            if threshold is None
            else (~cb.objectable) & (cb.ratio >= threshold)
        )
        cc.q0_raw_cuts.append(
            rt.empirical.Cut(
                threshold,
                float(np.sum(cb.value[chosen] * (1 - cc.pi0[chosen]))),
                float(np.sum(cb.loss[chosen] * (1 - cc.pi0[chosen]))),
                int(chosen.sum()),
            )
        )
    cc.q0_hull = rt.empirical.upper_hull(cc.q0_raw_cuts, cc.totals)
    return cc, cuts


def policy_probability(rt, array, case, rule, policy, rho):
    before = rt.frozen.baseline(array, case)
    if policy == "thinning":
        return rho * before, rho
    return rt.safe.batch_probability(batch(rt, array), rule), rule["contraction_s"]


def evaluate(rt, array, case, rule, beta, rho, part):
    before, q = rt.frozen.baseline(array, case), rt.frozen.native(array)
    rows = []
    total_loss = float(array[:, 3].sum())
    for policy in ("common", "thinning"):
        after, factor = policy_probability(rt, array, case, rule, policy, rho)
        error = max(
            float(np.max(-after, initial=0)),
            float(np.max(after - 1, initial=0)),
            float(np.max((after - before)[q], initial=0)),
        )
        factor_error = float(np.max(np.abs(q * (after - factor * before)), initial=0))
        if max(error, factor_error) > TOLERANCE:
            raise ValueError("Probability or pointwise contraction check failed")
        business = rt.metrics.business_metrics(array, before, after)
        business["capture_target_met"] = business["capture"] >= rho * business[
            "baseline_capture"
        ] - 1e-9 * max(1, business["baseline_capture"])
        business["value_cap_met"] = business["blocked_value"] <= beta * business[
            "total_value"
        ] + 1e-9 * max(1, business["total_value"])
        for unit in ("count", "value", "loss"):
            rows.append(
                {
                    "dataset": case["dataset"],
                    "seed": case["seed"],
                    "source": case["source"],
                    "part": part,
                    "config": config(beta, rho),
                    "design_cap": beta,
                    "capture_floor": rho,
                    "policy": policy,
                    "unit": unit,
                    "common_factor": factor,
                    "pointwise_safety_error": error,
                    "factor_error": factor_error,
                    "C_resource_infeasible": rule["resource_infeasible"]
                    if policy == "common"
                    else "",
                    "C_capture_goal_infeasible": rule.get("capture_goal_infeasible")
                    if policy == "common"
                    else "",
                    "C_selection_feasible": rule["selection_feasible"]
                    if policy == "common"
                    else "",
                    **business,
                    **rt.metrics.unit_metrics(array, before, after, q, unit),
                    "total_loss": total_loss,
                    "baseline_blocked_value_fraction": float(
                        np.sum(before * array[:, 2])
                    )
                    / business["total_value"]
                    if business["total_value"] > 0
                    else "",
                    "baseline_capture_fraction_loss": business["baseline_capture"]
                    / total_loss
                    if total_loss > 0
                    else "",
                    "capture_fraction_loss": business["capture"] / total_loss
                    if total_loss > 0
                    else "",
                }
            )
    return rows


def check_reference(rows, originals):
    maximum = 0.0
    for row in rows:
        old = originals[row["policy"], row["unit"]]
        for key in ("capture_target_met", "value_cap_met"):
            if str(row[key]).lower() != str(old[key]).lower():
                raise ValueError("Primary attainment flag changed: " + key)
        for key in (
            "capture",
            "baseline_capture",
            "capture_retained",
            "blocked_value",
            "blocked_value_fraction",
            "baseline_objection_mass",
            "objection_mass",
            "removal",
            "group_0_level",
            "group_1_level",
        ):
            if row[key] == "" or old[key] == "":
                if row[key] != old[key]:
                    raise ValueError("Primary undefined support changed: " + key)
                continue
            difference = abs(float(row[key]) - float(old[key])) / max(
                1, abs(float(old[key]))
            )
            maximum = max(maximum, difference)
            if difference > 1e-9:
                raise ValueError("Primary numeric result changed: " + key)
    return maximum


def fit_worker(task):
    case, args, source = task
    rt = runtime(args)
    original_dir, original = case_source(args, case)
    directory = args.output / "cells" / label(case)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "bandha_C_SEAL.json"
    if path.exists():
        cached = json.loads(path.read_text())
        if cached["identity"] != source:
            raise ValueError("Cached sweep identity changed")
        if digest(directory / "bandha_C.csv") != cached["C_csv_sha256"]:
            raise ValueError("Cached sweep C outcomes changed")
        return label(case), True
    with (original_dir / "C.csv").open() as handle:
        primary = {
            (r["policy"], r["unit"]): r
            for r in csv.DictReader(handle)
            if r["policy"] in ("common", "thinning")
        }
    points, rows = [], []
    with np.load(original_dir / "PREDICTIONS.npz") as saved:
        for beta in BETAS:
            recalibrated, thresholds = parameters(
                rt, original["case"], saved["V"], beta
            )
            v, c = (
                rebuild(saved["V"], thresholds, beta),
                rebuild(saved["C"], thresholds, beta),
            )
            cell, cuts = calibration_cell(rt, recalibrated, v, c)
            for rho in FLOORS:
                rule = rt.safe.fit_max_removal(cell, rho, beta)
                output = evaluate(rt, c, recalibrated, rule, beta, rho, "C")
                reference_error = (
                    check_reference(output, primary)
                    if beta == 0.10 and rho == 0.95
                    else None
                )
                rows.extend(output)
                points.append(
                    {
                        "beta": beta,
                        "rho": rho,
                        "case": recalibrated,
                        "criterion_thresholds": thresholds,
                        "rule": rule,
                        "V_replacement_cuts_sha256": hashlib.sha256(
                            json.dumps(cuts).encode()
                        ).hexdigest(),
                        "V_replacement_cut_count": len(cuts),
                        "primary_C_max_relative_error": reference_error,
                    }
                )
    csv_output(directory / "bandha_C.csv", rows)
    save(
        path,
        {
            "identity": source,
            "original_C_seal_sha256": digest(original_dir / "C_SEAL.json"),
            "original_predictions_sha256": original["predictions_sha256"],
            "points": points,
            "C_csv_sha256": digest(directory / "bandha_C.csv"),
            "T_outcomes_evaluated": False,
        },
    )
    return label(case), False


def fit(args, cases):
    source = identity(args)
    save(args.output / "bandha_SOURCE_SEAL.json", source)
    completed = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(fit_worker, (case, args, source)) for case in cases]
        for future in concurrent.futures.as_completed(futures):
            cell, cached = future.result()
            completed.append(cell)
            print(
                f"C sealed {len(completed)}/{len(cases)}: {cell}, cached={cached}",
                flush=True,
            )
            save(
                args.output / "bandha_PROGRESS.json",
                {"completed": completed, "cases": len(cases), "points_per_case": 9},
            )
    if source != identity(args):
        raise ValueError("Sweep source changed while fitting")
    seals = {
        label(case): digest(args.output / "cells" / label(case) / "bandha_C_SEAL.json")
        for case in cases
    }
    rows = []
    for case in cases:
        directory = args.output / "cells" / label(case)
        sealed = json.loads((directory / "bandha_C_SEAL.json").read_text())
        if digest(directory / "bandha_C.csv") != sealed["C_csv_sha256"]:
            raise ValueError("Sweep C outcomes changed before global freezing")
        with (directory / "bandha_C.csv").open() as handle:
            rows.extend(csv.DictReader(handle))
    csv_output(args.output / "bandha_C.csv", rows)
    save(
        args.output / "bandha_C_FREEZE.json",
        {
            "identity": source,
            "seals": seals,
            "cases": len(cases),
            "common_policies": 9 * len(cases),
            "C_csv_sha256": digest(args.output / "bandha_C.csv"),
            "T_outcomes_evaluated": False,
        },
    )
    print(
        "C FREEZE COMPLETE:",
        len(cases),
        "cases;",
        9 * len(cases),
        "common policies",
        flush=True,
    )


def reference_table(args):
    path = args.data_root / "results/csvs/bandha_xgboost_20261113/main/T.csv"
    with path.open() as handle:
        return {
            (r["dataset"] + "__" + r["seed"], r["policy"], r["unit"]): r
            for r in csv.DictReader(handle)
            if r["policy"] in ("common", "thinning")
        }


def test_case(args, case, freeze, reference):
    rt = runtime(args)
    directory, original = case_source(args, case)
    seal_path = args.output / "cells" / label(case) / "bandha_C_SEAL.json"
    if digest(seal_path) != freeze["seals"][label(case)]:
        raise ValueError("Sweep C seal changed")
    sealed = json.loads(seal_path.read_text())
    if original["predictions_sha256"] != sealed["original_predictions_sha256"]:
        raise ValueError("Sweep input predictions changed")
    rows, errors = [], []
    with np.load(directory / "PREDICTIONS.npz") as saved:
        for point in sealed["points"]:
            array = rebuild(saved["T"], point["criterion_thresholds"], point["beta"])
            output = evaluate(
                rt,
                array,
                point["case"],
                point["rule"],
                point["beta"],
                point["rho"],
                "T",
            )
            if point["beta"] == 0.10 and point["rho"] == 0.95:
                old = {
                    (policy, unit): reference[label(case), policy, unit]
                    for policy in ("common", "thinning")
                    for unit in ("count", "value", "loss")
                }
                errors.append(check_reference(output, old))
            rows.extend(output)
    return rows, max(errors)


def test(args, cases):
    freeze_path = args.output / "bandha_C_FREEZE.json"
    freeze = json.loads(freeze_path.read_text())
    if freeze["identity"] != identity(args) or len(freeze["seals"]) != len(cases):
        raise ValueError("Sweep source or case count changed after C freezing")
    if digest(args.output / "bandha_C.csv") != freeze["C_csv_sha256"]:
        raise ValueError("Sweep C outcomes changed after global freezing")
    reference = reference_table(args)
    rows, errors = [], []
    for number, case in enumerate(cases, 1):
        output, error = test_case(args, case, freeze, reference)
        rows.extend(output)
        errors.append(error)
        if number % 30 == 0:
            print(f"T evaluated {number}/{len(cases)} cases", flush=True)
    csv_output(args.output / "bandha_T.csv", rows)
    save(
        args.output / "bandha_T_PROVENANCE.json",
        {
            "C_freeze_sha256": digest(freeze_path),
            "T_sha256": digest(args.output / "bandha_T.csv"),
            "cases": len(cases),
            "common_policies": 9 * len(cases),
            "primary_T_max_relative_error": max(errors),
            "model_fits": 0,
            "no_T_parameter_selection": True,
            "exploratory": True,
        },
    )
    print(
        "T COMPLETE:",
        9 * len(cases),
        "common policies and matching thinning controls",
        flush=True,
    )


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("fit", "test"))
    for name in ("data-root", "run", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--canary", action="store_true")
    return parser.parse_args()


def main():
    args = arguments()
    args.output.mkdir(parents=True, exist_ok=True)
    cases = json.loads(
        (
            args.data_root
            / "results/csvs/bandha_external_criteria_inputs_20261113.json"
        ).read_text()
    )["cases"]
    if len(cases) != 240 or len({label(case) for case in cases}) != 240:
        raise ValueError("Expected the exact 240 primary cases")
    if args.canary:
        cases = [
            case
            for case in cases
            if (case["dataset"], case["seed"])
            in {
                ("uci_credit_age", "0"),
                ("german_credit_sex", "0"),
                ("freMPL1[Gender]", "0"),
            }
        ]
    (fit if args.phase == "fit" else test)(args, cases)


if __name__ == "__main__":
    main()
