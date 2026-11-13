"""Rerun the exact 240-cell fairness panel with boosted baseline and criteria."""

from __future__ import annotations

import argparse
import base64
import concurrent.futures
import csv
import hashlib
import importlib.metadata
import json
import sys
import time
from pathlib import Path

import bandha_external_criteria_io as frozen
import bandha_xgboost_models as models
import fairness_candidates as legacy
import numpy as np
import public_group_candidates as public


def save(path, data):
    path.write_text(json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n")


def array_hash(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def prepare_raw(raw):
    for name in ("ausprivauto0405", "beMTPL97"):
        path = raw / (name + ".csv")
        if not path.exists():
            public.rda_frame(str(raw / (name + ".rda"))).to_csv(path, index=False)


def source_for(dataset):
    if dataset.startswith("australian_mtpl_"):
        return "ausprivauto0405.csv"
    if dataset.startswith("belgian_mtpl_"):
        return "beMTPL97.csv"
    if dataset.startswith("bondora_"):
        return "LoanData_Bondora.csv"
    if dataset.startswith("cms_"):
        return "cms_train_provider_patient.csv"
    if dataset.startswith("uci_credit_"):
        return "default_credit_card_clients.csv"
    if dataset.startswith("german_credit_"):
        return "german.data"
    return (
        dataset.split("[")[0].replace("euhealth_gender", "euhealthinsurance") + ".rda"
    )


def load_legacy_case(dataset, path, seed):
    name = dataset.rsplit("_group_", 1)[0]
    loader = {
        "australian_mtpl": legacy.load_australian_data,
        "belgian_mtpl": legacy.load_belgian_data,
        "bondora": legacy.load_bondora_data,
    }[name]
    data = loader(str(path))
    ids = np.arange(len(data.frame))
    outer, test = legacy.stratified_split(ids, data.label, 0.2, seed)
    train, validation = legacy.stratified_split(outer, data.label, 0.2, 10000 + seed)
    designs = legacy.transform_split(
        data.frame,
        train,
        validation,
        test,
        dataset.endswith("group_aware"),
        data.numeric_columns,
        data.categorical_columns,
        data.group_column,
    )
    return (
        designs,
        (train, validation, test),
        data.label,
        data.value.astype(float),
        data.loss.astype(float),
        data.groups,
        None,
    )


def public_loader(dataset):
    if dataset == "beMTPL97":
        return public.LOCAL_REGISTRY["belgian"], 50000
    if dataset.startswith(("freMPL", "norauto", "swmotorcycle")):
        return public.LOCAL_REGISTRY["auto"], 50000
    n = (
        30000
        if dataset.startswith("uci_")
        else 1000
        if dataset.startswith("german_")
        else 200000
    )
    return public.LOCAL_REGISTRY[dataset], n


def load_public_case(dataset, path, seed):
    loader, n = public_loader(dataset)
    frame, numeric, categorical, label, value, loss, group, name, accounts = loader(
        str(path), n, 42
    )
    if name != dataset:
        raise ValueError(f"Loader cohort changed: {name} != {dataset}")
    for column in categorical:
        frame[column] = frame[column].astype(object)
    ids = np.arange(len(frame))
    if accounts is None:
        outer, test = public.stratified_split(ids, label, 0.2, seed)
        train, validation = public.stratified_split(outer, label, 0.2, 10000 + seed)
    else:
        train, validation, test = public.account_nested_split(accounts, label, seed)
    designs = public.transform(frame, numeric, categorical, train, validation, test)
    return designs, (train, validation, test), label, value, loss, group, accounts


def load_case(case, raw):
    dataset = case["dataset"]
    path = raw / source_for(dataset)
    seed = int(case["seed"])
    historical = dataset.startswith(("australian_mtpl_", "belgian_mtpl_", "bondora_"))
    loader = load_legacy_case if historical else load_public_case
    designs, ids, label, value, loss, group, accounts = loader(dataset, path, seed)
    labels = np.asarray(group).astype(str)
    encoded = np.asarray([case["groups"].index(g) for g in labels], dtype=int)
    return designs, ids, np.asarray(label), value, loss, encoded, accounts, path


def partition(case):
    bits = np.frombuffer(
        base64.b64decode(case["partition_bits_base64"]), dtype=np.uint8
    )
    return np.unpackbits(bits, bitorder="little")[: case["n_test_rows"]].astype(bool)


def check_membership(case, ids, value, loss, group, originals):
    train, validation, test = ids
    if len(test) != case["n_test_rows"]:
        raise ValueError(f"Test count changed for {case['dataset']}/{case['seed']}")
    c = partition(case)
    errors = {}
    for name, rows in (("V", validation), ("C", test[c]), ("T", test[~c])):
        reference = originals[name]
        if len(rows) != len(reference) or not np.array_equal(
            group[rows], reference[:, 0]
        ):
            raise ValueError(f"Ordered group membership changed in {name}")
        for column, actual in ((2, value[rows]), (3, loss[rows])):
            target = reference[:, column]
            error = np.max(
                np.abs(actual - target) / np.maximum(1, np.abs(target)), initial=0
            )
            # Legacy candidate archives serialize amounts to seven digits.
            if error > 6e-7:
                raise ValueError(f"Ordered raw outcomes changed in {name}: {error}")
            errors[f"{name}_{column}"] = float(error)
    return {
        "ordered_groups_equal": True,
        "outcome_max_relative_errors": errors,
        "train_ids_sha256": array_hash(train),
        "V_ids_sha256": array_hash(validation),
        "test_ids_sha256": array_hash(test),
        "C_ids_sha256": array_hash(test[c]),
        "T_ids_sha256": array_hash(test[~c]),
    }


def boosted_arrays(case, outputs, ids, value, loss, group):
    vscore, va, vb = outputs[0]
    tscore, ta, tb = outputs[1]
    v, test = ids[1:]
    threshold = public.threshold_for_mass(vscore, value[v], 0.1)
    selected = vscore >= threshold
    spending = np.sum(value[v][selected])
    alpha = min(1.0, 0.1 * np.sum(value[v]) / spending) if spending > 0 else 1.0
    criteria = public.criterion_thresholds(vscore, va, vb, value[v], 0.1, 1.0)

    def array(rows, score, a, b):
        flags = public.criterion_scores(score, a, b, 1.0)
        return np.column_stack(
            (
                group[rows],
                np.exp(np.clip(score, -50, 50)),
                value[rows],
                loss[rows],
                a,
                b,
                flags["harm"] >= criteria["harm"],
                flags["net_harm"] >= criteria["net_harm"],
                flags["wbce_ratio"] >= criteria["wbce_ratio"],
            )
        )

    new = {
        **case,
        "baseline_threshold": float(np.exp(np.clip(threshold, -50, 50))),
        "baseline_alpha": float(alpha),
        "boosted_criterion_thresholds": criteria,
    }
    return new, array(v, vscore, va, vb), array(test, tscore, ta, tb)


def batch(array):
    import bandha_empirical_cut_selector as empirical

    return empirical.ValidationBatch(
        array[:, 0].astype(int),
        array[:, 1],
        frozen.native(array),
        array[:, 2],
        array[:, 3],
    )


def scope(case):
    import bandha_direct_constrained_selector as direct

    return direct.Scope(
        (case["dataset"], case["seed"], "value"),
        case["source"],
        case["input"],
        tuple(case["groups"]),
        0,
        case["baseline_threshold"],
        case["baseline_alpha"],
    )


def fit_policies(case, v, c):
    import bandha_empirical_cut_selector as empirical
    import bandha_objection_safe_selector as safe
    import ultra_value_retention_selector_20261105 as selective

    s = scope(case)
    vb, cb = batch(v), batch(c)
    vc = safe.build_cell(s, vb)
    v_rule = safe.fit_max_removal(vc, 0.95, 0.1)
    cc = safe.initial_cell(s, cb)
    # C moments at exactly the frozen V score-cut locations, including ties.
    cuts = [point.threshold for point in vc.q0_raw_cuts]
    p0 = cc.pi0
    cc.q0_raw_cuts = []
    mask = ~cb.objectable
    for threshold in dict.fromkeys(cuts):
        chosen = (
            np.zeros(cb.size, dtype=bool)
            if threshold is None
            else mask & (cb.ratio >= threshold)
        )
        cc.q0_raw_cuts.append(
            empirical.Cut(
                threshold,
                float(np.sum(cb.value[chosen] * (1 - p0[chosen]))),
                float(np.sum(cb.loss[chosen] * (1 - p0[chosen]))),
                int(chosen.sum()),
            )
        )
    cc.q0_hull = empirical.upper_hull(cc.q0_raw_cuts, cc.totals)
    common = safe.fit_max_removal(cc, 0.95, 0.1)
    sv = selective.build_cell(s, vb)
    selective_v = selective.fit_selector(sv, 0.95, 0.1)
    thresholds = [[point.threshold for point in points] for points in sv.raw_cuts]
    sc = selective.build_cell(s, cb, thresholds)
    selected = selective.fit_selector(sc, 0.95, 0.1)
    chosen = selective.batch_probability(cb, selected)
    q = cb.objectable
    base = frozen.baseline(c, case)
    anchor = np.where(q, chosen, base)
    extra = chosen - anchor
    anchor_cost, added_cost = np.sum(anchor * c[:, 2]), np.sum(extra * c[:, 2])
    reserve = (
        min(1.0, max(0.0, (0.095 * np.sum(c[:, 2]) - anchor_cost) / added_cost))
        if added_cost > 0
        else (1.0 if anchor_cost <= 0.095 * np.sum(c[:, 2]) + 1e-10 else 0.0)
    )
    return {
        "common": common,
        "common_V": v_rule,
        "selective": selected,
        "selective_V": selective_v,
        "reserve_multiplier": float(reserve),
        "C_parameters_frozen": True,
    }


def probability(array, case, decision, name):
    import bandha_objection_safe_selector as safe
    import ultra_value_retention_selector_20261105 as selective

    p0 = frozen.baseline(array, case)
    if name == "baseline":
        return p0
    if name == "thinning":
        return 0.95 * p0
    if name == "common":
        return safe.batch_probability(batch(array), decision["common"])
    result = selective.batch_probability(batch(array), decision["selective"])
    if name == "reserve":
        q = frozen.native(array)
        result = np.where(
            q, result, p0 + decision["reserve_multiplier"] * (result - p0)
        )
    return result


def evaluate(array, case, decision, part):
    from bandha_external_criteria_study import business_metrics, unit_metrics

    p0, q = frozen.baseline(array, case), frozen.native(array)
    rows = []
    for name in ("baseline", "thinning", "common", "selective", "reserve"):
        p = probability(array, case, decision, name)
        bound = max(
            float(np.max(-p, initial=0)),
            float(np.max(p - 1, initial=0)),
            float(np.max((p - p0)[q], initial=0)),
        )
        if bound > 1e-8:
            raise ValueError(f"Probability/objector safety failure: {bound}")
        factor = (
            0.95
            if name == "thinning"
            else decision["common"]["contraction_s"]
            if name == "common"
            else 1.0
        )
        error = float(np.max(np.abs(q * (p - factor * p0)), initial=0))
        if name in ("common", "thinning") and error > 1e-8:
            raise ValueError(f"Common-factor identity failure: {error}")
        business = business_metrics(array, p0, p)
        for unit in ("count", "value", "loss"):
            row = {
                "dataset": case["dataset"],
                "seed": case["seed"],
                "source": case["source"],
                "part": part,
                "policy": name,
                "common_factor": factor if name in ("common", "thinning") else "",
                "pointwise_safety_error": bound,
                "factor_error": error if name in ("common", "thinning") else "",
            }
            row.update(business)
            row.update(unit_metrics(array, p0, p, q, unit))
            row["unit"] = unit
            rows.append(row)
    return rows


def worker(task):
    case, raw, output, data_root, originals_path, threads = task
    sys.path.insert(0, str(data_root / "scripts"))
    label = case["dataset"] + "__" + case["seed"]
    directory = output / "cells" / label
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / "C_SEAL.json").exists():
        return label, "cached"
    started = time.monotonic()
    designs, ids, y, a, b, group, accounts, raw_path = load_case(case, raw)
    originals_file = np.load(originals_path)
    originals = {key: originals_file[key] for key in ("V", "C", "T")}
    receipt = check_membership(case, ids, a, b, group, originals)
    eval_a, eval_b = a.copy(), b.copy()
    cmask = partition(case)
    for part, rows in (("V", ids[1]), ("C", ids[2][cmask]), ("T", ids[2][~cmask])):
        eval_a[rows] = originals[part][:, 2]
        eval_b[rows] = originals[part][:, 3]
    train = ids[0]
    fitted, model_record = models.fit_models(
        designs[0],
        a[train],
        b[train],
        y[train],
        None if accounts is None else accounts[train],
        int(case["seed"]),
        threads,
        directory,
    )
    predictions = [
        models.predict(fitted, model_record, design, threads) for design in designs[1:]
    ]
    boosted, v, test = boosted_arrays(case, predictions, ids, eval_a, eval_b, group)
    c, t = test[cmask], test[~cmask]
    decision = fit_policies(boosted, v, c)
    np.savez_compressed(
        directory / "PREDICTIONS.npz",
        V=v,
        C=c,
        T=t,
        train_ids=train,
        V_ids=ids[1],
        test_ids=ids[2],
        C_ids=ids[2][cmask],
        T_ids=ids[2][~cmask],
    )
    # Readout of C occurs before the seal; numeric T readout is a separate command.
    csv_rows(directory / "C.csv", evaluate(c, boosted, decision, "C"))
    save(
        directory / "C_SEAL.json",
        {
            "case": boosted,
            "decision": decision,
            "models": model_record,
            "membership": receipt,
            "raw_sha256": frozen.digest(raw_path),
            "predictions_sha256": frozen.digest(directory / "PREDICTIONS.npz"),
            "C_csv_sha256": frozen.digest(directory / "C.csv"),
            "T_outcomes_evaluated": False,
            "seconds": time.monotonic() - started,
        },
    )
    return label, round(time.monotonic() - started, 2)


def csv_rows(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def code_hashes(data_root):
    paths = [Path(__file__), Path(models.__file__)]
    paths += [Path(public.__file__), Path(legacy.__file__), Path(frozen.__file__)]
    paths += [
        data_root / "scripts" / name
        for name in (
            "bandha_objection_safe_selector.py",
            "bandha_empirical_cut_selector.py",
            "ultra_value_retention_selector_20261105.py",
            "bandha_direct_constrained_selector.py",
        )
    ]
    return {str(p): frozen.digest(p) for p in paths}


def fit_phase(args, manifest, cases, identity):
    prepare_raw(args.raw)
    missing = sorted(
        {
            source_for(c["dataset"])
            for c in cases
            if not (args.raw / source_for(c["dataset"])).exists()
        }
    )
    if missing:
        raise FileNotFoundError(f"Missing raw inputs: {missing}")
    save(args.output / "SOURCE_SEAL.json", identity)
    originals = args.output / "originals"
    originals.mkdir(exist_ok=True)
    grouped = {}
    for c in cases:
        grouped.setdefault(c["input"], []).append(c)
    for relative, group in grouped.items():
        parts = frozen.scan_file(
            args.data_root, relative, group, manifest, {"V", "C", "T"}
        )
        for c in group:
            label = c["dataset"] + "__" + c["seed"]
            np.savez_compressed(
                originals / (label + ".npz"),
                **{p: parts[((c["dataset"], c["seed"]), p)] for p in ("V", "C", "T")},
            )
        print("Original membership loaded:", relative, flush=True)
    tasks = [
        (
            c,
            args.raw,
            args.output,
            args.data_root,
            originals / (c["dataset"] + "__" + c["seed"] + ".npz"),
            args.threads,
        )
        for c in cases
    ]
    completed = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
        future_map = {pool.submit(worker, task): task[0] for task in tasks}
        for future in concurrent.futures.as_completed(future_map):
            label, seconds = future.result()
            completed.append(label)
            print(
                f"C sealed {len(completed)}/{len(cases)}: {label} ({seconds}s)",
                flush=True,
            )
            save(
                args.output / "PROGRESS.json",
                {"completed": completed, "total": len(cases)},
            )
    seals = {
        c["dataset"] + "__" + c["seed"]: frozen.digest(
            args.output / "cells" / (c["dataset"] + "__" + c["seed"]) / "C_SEAL.json"
        )
        for c in cases
    }
    save(
        args.output / "C_FREEZE.json",
        {**identity, "seals": seals, "T_outcomes_evaluated": False},
    )


def test_phase(args, cases, identity):
    seal = json.loads((args.output / "C_FREEZE.json").read_text())
    for key in ("manifest_sha256", "code", "protocol_sha256", "cells"):
        if seal[key] != identity[key]:
            raise ValueError(f"Frozen input/code changed: {key}")
    rows, crows, membership = [], [], []
    for case in cases:
        label = case["dataset"] + "__" + case["seed"]
        directory = args.output / "cells" / label
        if frozen.digest(directory / "C_SEAL.json") != seal["seals"][label]:
            raise ValueError("Cell seal changed")
        record = json.loads((directory / "C_SEAL.json").read_text())
        if frozen.digest(directory / "PREDICTIONS.npz") != record["predictions_sha256"]:
            raise ValueError("Predictions changed")
        arrays = np.load(directory / "PREDICTIONS.npz")
        rows.extend(evaluate(arrays["T"], record["case"], record["decision"], "T"))
        crows.extend(evaluate(arrays["C"], record["case"], record["decision"], "C"))
        membership.append(
            {
                "dataset": case["dataset"],
                "seed": case["seed"],
                **record["membership"],
            }
        )
    csv_rows(args.output / "T.csv", rows)
    csv_rows(args.output / "C.csv", crows)
    save(
        args.output / "T_PROVENANCE.json",
        {
            "C_freeze_sha256": frozen.digest(args.output / "C_FREEZE.json"),
            "T_sha256": frozen.digest(args.output / "T.csv"),
            "membership": membership,
            "cells": len(cases),
            "no_T_parameter_selection": True,
        },
    )
    print(f"T complete: {len(cases)} cells, {len(rows)} outcome rows", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("fit", "test"))
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--raw", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--threads", type=int, default=3)
    parser.add_argument("--canary", action="store_true")
    args = parser.parse_args()
    sys.path.insert(0, str(args.data_root / "scripts"))
    args.output.mkdir(parents=True, exist_ok=True)
    manifest_path = (
        args.data_root / "results/csvs/bandha_external_criteria_inputs_20261113.json"
    )
    manifest = frozen.read_manifest(manifest_path, args.data_root)
    cases = manifest["cases"]
    if args.canary:
        cases = [
            c for c in cases if c["dataset"] == "german_credit_sex" and c["seed"] == "0"
        ]
    identity = {
        "manifest_sha256": frozen.digest(manifest_path),
        "code": code_hashes(args.data_root),
        "protocol_sha256": frozen.digest(
            args.data_root / "bandha_xgboost_protocol_20261113.md"
        ),
        "packages": {
            p: importlib.metadata.version(p)
            for p in ("numpy", "pandas", "scipy", "xgboost")
        },
        "parameters": models.PARAMETERS,
        "max_rounds": models.MAX_ROUNDS,
        "patience": models.PATIENCE,
        "cells": len(cases),
        "exploratory": True,
    }
    if args.phase == "fit":
        fit_phase(args, manifest, cases, identity)
    else:
        test_phase(args, cases, identity)


if __name__ == "__main__":
    main()
