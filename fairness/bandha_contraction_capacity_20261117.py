"""Export per-cell headroom coefficients for the completed contraction sweep."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import bandha_contraction_sweep_20261117 as replay
import numpy as np


def capacity(array, before, q):
    baseline = float(np.sum(before * array[:, 3]))
    objected = float(np.sum(before * q * array[:, 3]))
    headroom = float(np.sum((1 - q) * (1 - before) * array[:, 3]))
    return {
        "B0": baseline,
        "Bq": objected,
        "HB": headroom,
        "sigma": objected / baseline if baseline > 0 else "",
        "h": headroom / baseline if baseline > 0 else "",
    }


def case_rows(args, case, freeze):
    rt = replay.runtime(args)
    directory, original = replay.case_source(args, case)
    path = args.sweep / "cells" / replay.label(case) / "bandha_C_SEAL.json"
    if replay.digest(path) != freeze["seals"][replay.label(case)]:
        raise ValueError("Sweep C seal changed during capacity export")
    sealed = json.loads(path.read_text())
    if original["predictions_sha256"] != sealed["original_predictions_sha256"]:
        raise ValueError("Original predictions changed during capacity export")
    rows = []
    with np.load(directory / "PREDICTIONS.npz") as saved:
        for beta in replay.BETAS:
            points = [point for point in sealed["points"] if point["beta"] == beta]
            anchor = points[0]
            if any(
                point["case"] != anchor["case"]
                or point["criterion_thresholds"] != anchor["criterion_thresholds"]
                for point in points
            ):
                raise ValueError("Baseline or criteria change with rho")
            for part in ("C", "T"):
                array = replay.rebuild(
                    saved[part], anchor["criterion_thresholds"], beta
                )
                before = rt.frozen.baseline(array, anchor["case"])
                rows.append(
                    {
                        "cell": replay.label(case),
                        "source": case["source"],
                        "beta": beta,
                        "part": part,
                        **capacity(array, before, rt.frozen.native(array)),
                    }
                )
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data-root", "run", "sweep", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    freeze_path = args.sweep / "bandha_C_FREEZE.json"
    freeze = json.loads(freeze_path.read_text())
    if freeze["identity"] != replay.identity(args):
        raise ValueError("Sweep runtime changed before capacity export")
    cases = json.loads(
        (
            args.data_root
            / "results/csvs/bandha_external_criteria_inputs_20261113.json"
        ).read_text()
    )["cases"]
    rows = []
    for case in cases:
        rows.extend(case_rows(args, case, freeze))
    if len(rows) != 1440:
        raise ValueError("Capacity export must retain 240 cases, three caps and C/T")
    args.output.mkdir(parents=True, exist_ok=True)
    output = args.output / "bandha_capacity.csv"
    replay.csv_output(output, rows)
    replay.save(
        args.output / "bandha_capacity_provenance.json",
        {
            "C_freeze_sha256": replay.digest(freeze_path),
            "capacity_csv_sha256": replay.digest(output),
            "generator_sha256": replay.digest(Path(__file__)),
            "cases": 240,
            "model_fits": 0,
            "scope": "Per-cell pooled capture bound; apply requested-rho bounds only where the floor is attained.",
        },
    )
    print("Capacity coefficients frozen:", output, flush=True)


if __name__ == "__main__":
    main()
