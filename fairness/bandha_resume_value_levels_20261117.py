"""Complete the original value arm while preserving the capture-only result."""

from __future__ import annotations

import argparse
import ast
import concurrent.futures
import datetime as dt
import importlib
import json
import os
import sys
import time
from pathlib import Path


def load_runtime(args):
    directory = str(args.root / "scripts")
    if directory not in sys.path:
        sys.path.insert(0, directory)
    return importlib.import_module("bandha_agarwal_level_run_20261117")


def save(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    os.replace(temporary, path)


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def label(case):
    return case["dataset"] + "__" + case["seed"]


def source_identity(args):
    runtime = load_runtime(args)
    original = json.loads(
        (args.output / "bandha_SOURCE_SEAL_before_export_fix.json").read_text()
    )
    current = runtime.identity(args)
    driver = str(Path(runtime.__file__))
    protocol = str(args.root / "bandha_level_reductions_protocol_20261117.md")
    for key in original:
        if key != "code" and original[key] != current[key]:
            raise ValueError("Original fitting configuration changed: " + key)
    if set(original["code"]) != set(current["code"]):
        raise ValueError("Original fitting source inventory changed")
    for path, expected in original["code"].items():
        if path not in (driver, protocol) and current["code"][path] != expected:
            raise ValueError("Original fitting dependency changed: " + path)
    compatibility = json.loads(
        (args.output / "bandha_EXPORT_COMPATIBILITY.json").read_text()
    )
    if current["code"][driver] != compatibility["fixed_driver_sha256"]:
        raise ValueError("Exporter differs from the verified fix")
    prior_driver = args.output / "bandha_driver_before_export_fix.py"
    if runtime.original.sha(prior_driver) != original["code"][driver]:
        raise ValueError("Original driver snapshot changed")
    trees = [ast.parse(p.read_text()) for p in (prior_driver, Path(runtime.__file__))]
    unchanged = {}
    for name in (
        "fit_classifiers",
        "objective_for",
        "predict_bank",
        "check_reconstructed_features",
        "seal_arm",
    ):
        functions = [
            next(
                n
                for n in tree.body
                if isinstance(n, ast.FunctionDef) and n.name == name
            )
            for tree in trees
        ]
        if name == "seal_arm":
            # The exporter call changed only by delegating serialization to a helper.
            continue
        unchanged[name] = ast.dump(functions[0], include_attributes=False) == ast.dump(
            functions[1], include_attributes=False
        )
        if not unchanged[name]:
            raise ValueError("Fitting function changed: " + name)
    return {
        "original_fit_identity": original,
        "current_fit_identity": current,
        "unchanged_fitting_functions": unchanged,
        "continuation_runner_sha256": runtime.original.sha(Path(__file__)),
        "continuation_protocol_sha256": runtime.original.sha(
            args.root / "bandha_value_levels_continuation_protocol_20261117.md"
        ),
        "timing": "User authorized completion after examining capture-arm T results.",
    }


def verify_existing(args, cases, identity):
    runtime = load_runtime(args)
    original = runtime.original
    seals, pending = {}, []
    permitted = (identity["original_fit_identity"], identity["current_fit_identity"])
    for case in cases:
        cell = label(case)
        original_predictions = original.sha(
            args.run / "cells" / cell / "PREDICTIONS.npz"
        )
        seals[cell] = {}
        for objective in ("capture", "value"):
            directory = args.output / ("bandha_" + cell) / objective
            path = directory / "bandha_C_seal.json"
            if not path.is_file():
                if objective == "capture":
                    raise ValueError("Completed capture seal missing: " + cell)
                pending.append(case)
                continue
            record = json.loads(path.read_text())
            if record["fit_identity"] not in permitted:
                raise ValueError("Unrecognized fit identity: " + cell + "/" + objective)
            if record["original_predictions_sha256"] != original_predictions:
                raise ValueError("Original predictions changed: " + cell)
            if (
                original.sha(directory / "bandha_predictions.npz")
                != record["predictions_sha256"]
            ):
                raise ValueError(
                    "Sealed predictions changed: " + cell + "/" + objective
                )
            for entry in record.get("predictor_manifest", []):
                if (
                    entry["kind"] == "xgboost"
                    and not (directory / entry["file"]).is_file()
                ):
                    raise ValueError("Exported classifier missing: " + cell)
            seals[cell][objective] = original.sha(path)
    return seals, pending


def worker(task):
    case, args = task
    runtime = load_runtime(args)
    directory = args.run / "cells" / label(case)
    record = json.loads((directory / "C_SEAL.json").read_text())
    with runtime.np.load(directory / "PREDICTIONS.npz") as saved:
        xtrain, xc, xt, labels, train = runtime.training.training_data(
            case, args, record, saved
        )
        checks = runtime.check_reconstructed_features(case, args, record, xc, saved)
        result = runtime.seal_arm(
            case, args, record, saved, train, xtrain, xc, xt, labels, "value", checks
        )
    archive = args.output / "bandha_partial_before_value_resume_20261117" / label(case)
    previous = archive / "bandha_predictions.npz"
    difference = {}
    if previous.exists():
        with (
            runtime.np.load(previous) as before,
            runtime.np.load(
                args.output / ("bandha_" + label(case)) / "value/bandha_predictions.npz"
            ) as after,
        ):
            for key in ("C_raw", "C_cal"):
                if before[key].shape != after[key].shape:
                    raise ValueError("Partial C prediction shape changed")
                difference[key] = float(
                    runtime.np.max(runtime.np.abs(before[key] - after[key]), initial=0)
                )
                if difference[key] > 1e-7:
                    raise ValueError(
                        "Deterministic refit changed partial C predictions: "
                        + label(case)
                    )
    result["partial_C_max_absolute_difference"] = difference
    save(
        args.output
        / ("bandha_" + label(case))
        / "value/bandha_continuation_checks.json",
        result,
    )
    return label(case), result


def capture_hashes(args):
    runtime = load_runtime(args)
    return {
        n: runtime.original.sha(args.output / n)
        for n in (
            "bandha_C_FREEZE.json",
            "bandha_C.csv",
            "bandha_T.csv",
            "bandha_T_PROVENANCE.json",
        )
    }


def run_pending_fits(args, pending):
    completed, errors = [], []
    start = time.monotonic()
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
        jobs = {pool.submit(worker, (case, args)): case for case in pending}
        for future in concurrent.futures.as_completed(jobs):
            case = jobs[future]
            try:
                cell, result = future.result()
                completed.append(cell)
                print(
                    f"Value C sealed {240 - len(pending) + len(completed)}/240: {cell} {result}",
                    flush=True,
                )
            except (
                ValueError,
                RuntimeError,
                OSError,
                TypeError,
                AttributeError,
            ) as exc:
                errors.append({"cell": label(case), "error": repr(exc)})
                print("FIT ERROR:", label(case), repr(exc), flush=True)
            save(
                args.output / "bandha_VALUE_PROGRESS.json",
                {
                    "initial_C_sealed": 240 - len(pending),
                    "completed": completed,
                    "pending_at_start": len(pending),
                    "failures": errors,
                    "elapsed_seconds": time.monotonic() - start,
                    "updated_utc": now(),
                },
            )
    if errors:
        raise RuntimeError("Value fits failed: " + json.dumps(errors))


def fit(args, cases):
    identity = source_identity(args)
    existing, pending = verify_existing(args, cases, identity)
    old_hashes = capture_hashes(args)
    source_path = args.output / "bandha_VALUE_CONTINUATION_SOURCE.json"
    if source_path.exists():
        snapshot = json.loads(source_path.read_text())
        if snapshot["identity"] != identity or snapshot["capture_hashes"] != old_hashes:
            raise ValueError("Continuation source or capture evidence changed")
    else:
        snapshot = {
            "created_utc": now(),
            "identity": identity,
            "capture_hashes": old_hashes,
            "initial_value_C_sealed": 240 - len(pending),
            "missing_cases": [label(c) for c in pending],
        }
        save(source_path, snapshot)
    archive = args.output / "bandha_partial_before_value_resume_20261117"
    archive.mkdir(exist_ok=True)
    for case in pending:
        directory = args.output / ("bandha_" + label(case)) / "value"
        target = archive / label(case)
        if directory.exists() and not target.exists():
            os.replace(directory, target)
    print(
        f"Resume: {240 - len(pending)}/240 value C seals reused; {len(pending)} fits pending.",
        flush=True,
    )
    run_pending_fits(args, pending)
    if source_identity(args) != identity or capture_hashes(args) != old_hashes:
        raise ValueError("Source or capture evidence changed during continuation")
    sealed, remaining = verify_existing(args, cases, identity)
    if remaining:
        raise ValueError("Value arm incomplete after fitting")
    for cell, arms in existing.items():
        for objective, expected in arms.items():
            if sealed[cell][objective] != expected:
                raise ValueError("Previously complete C seal changed")
    freeze = {
        "identity": identity,
        "created_utc": now(),
        "objectives": ["value"],
        "seals": {cell: {"value": arms["value"]} for cell, arms in sealed.items()},
        "continuation_source_sha256": load_runtime(args).original.sha(source_path),
        "initial_value_C_sealed": snapshot["initial_value_C_sealed"],
        "completed_value_fits": 240 - snapshot["initial_value_C_sealed"],
        "capture_results_examined_before_continuation": True,
        "value_T_metrics_evaluated": False,
    }
    save(args.output / "bandha_VALUE_C_FREEZE.json", freeze)
    print(
        "VALUE C FREEZE COMPLETE: all 240 policies sealed; capture outputs unchanged.",
        flush=True,
    )


def test(args, cases):
    runtime = load_runtime(args)
    freeze_path = args.output / "bandha_VALUE_C_FREEZE.json"
    freeze = json.loads(freeze_path.read_text())
    if freeze["identity"] != source_identity(args):
        raise ValueError("Continuation code changed after C freeze")
    snapshot = json.loads(
        (args.output / "bandha_VALUE_CONTINUATION_SOURCE.json").read_text()
    )
    if capture_hashes(args) != snapshot["capture_hashes"]:
        raise ValueError("Completed capture result changed")
    rows = {"C": [], "T": []}
    for number, case in enumerate(cases, 1):
        with runtime.np.load(
            args.run / "cells" / label(case) / "PREDICTIONS.npz"
        ) as saved:
            result = runtime.evaluate_arm(args, case, "value", freeze, saved)
        for part in rows:
            rows[part].extend(result[part])
        if number % 30 == 0:
            print(f"Value T evaluated {number}/240", flush=True)
    for part, records in rows.items():
        runtime.original.csv_output(args.output / f"bandha_VALUE_{part}.csv", records)
    save(
        args.output / "bandha_VALUE_T_PROVENANCE.json",
        {
            "created_utc": now(),
            "C_freeze_sha256": runtime.original.sha(freeze_path),
            "C_sha256": runtime.original.sha(args.output / "bandha_VALUE_C.csv"),
            "T_sha256": runtime.original.sha(args.output / "bandha_VALUE_T.csv"),
            "evaluation_code_sha256": runtime.original.sha(Path(__file__)),
            "cases": 240,
            "objectives_evaluated": ["value"],
            "capture_results_examined_before_continuation": True,
            "no_T_parameter_selection": True,
            "initial_value_C_sealed": freeze["initial_value_C_sealed"],
            "completed_value_fits": freeze["completed_value_fits"],
        },
    )
    if capture_hashes(args) != snapshot["capture_hashes"]:
        raise ValueError("Capture evidence changed during value evaluation")
    print(
        "VALUE T COMPLETE: all 240 cases; earlier capture result preserved.", flush=True
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("fit", "test"))
    for name in ("root", "public-root", "run", "raw", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    cases = json.loads(
        (
            args.root / "results/csvs/bandha_external_criteria_inputs_20261113.json"
        ).read_text()
    )["cases"]
    if len(cases) != 240 or len({label(c) for c in cases}) != 240:
        raise ValueError("Expected the original 240 unique cases")
    (fit if args.phase == "fit" else test)(args, cases)


if __name__ == "__main__":
    main()
