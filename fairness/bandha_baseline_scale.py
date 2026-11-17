#!/usr/bin/env python3
"""Baseline scale: what the manuscript's relative removals are relative to.

Reads the frozen final panel straight out of
``artifacts/bandha_xgboost_evidence_20261113.zip`` and uses only the Python
standard library, so it runs from a clone with no dependencies installed.

Why this exists. Every headline in the Experiments section is a *relative*
removal: a fraction of the objection mass the baseline already carries.
Without the denominator a reader cannot tell whether the contested set is a
rounding error or most of the problem. This script reports the denominator.

Two quantities, both at the baseline policy and both per cell:

  * ``sigma`` = B_q / B_0, the share of the loss the baseline captures that
    at least one criterion contests. This is the sigma of the feasibility
    bound, so the number here is the one that bound is about.
  * the group objection levels D_g^U(pi_0) themselves, in each aggregation,
    which is the level the common factor scales.

Medians, not means: the panel is 240 cells over ten source families with one
family supplying 90, so a mean is a statement about the French portfolio.
The per-family medians are printed alongside for the same reason.

Falsification tests abort the run:
  * the panel must contain the 240 recorded cells and ten source families
  * every cell must carry all three aggregations at the baseline policy
  * sigma must be read from the loss aggregation, where numerator and
    denominator are both captured loss; the ratio is meaningless elsewhere
    and the script refuses to report it for the other two

Usage:
    python3 fairness/bandha_baseline_scale.py
    python3 fairness/bandha_baseline_scale.py --csv path/to/T.csv
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import statistics as st
import zipfile
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
EVIDENCE_ZIP = REPO / "artifacts/bandha_xgboost_evidence_20261113.zip"
PANEL_MEMBER = "results/csvs/bandha_xgboost_20261113/main/T.csv"

EXPECTED_FAMILIES = 10
EXPECTED_CELLS = 240
AGGREGATIONS = ("count", "value", "loss")


def panel(path):
    if path is not None:
        with path.open() as handle:
            return list(csv.DictReader(handle))
    with zipfile.ZipFile(EVIDENCE_ZIP) as archive:
        text = archive.read(PANEL_MEMBER).decode()
    return list(csv.DictReader(io.StringIO(text)))


def baseline_rows(rows):
    kept = defaultdict(dict)
    for row in rows:
        if row["policy"] != "baseline":
            continue
        kept[(row["dataset"], row["seed"])][row["unit"]] = row
    if len(kept) != EXPECTED_CELLS:
        raise ValueError(f"Panel holds {len(kept)} cells, expected {EXPECTED_CELLS}")
    for key, units in kept.items():
        if set(units) != set(AGGREGATIONS):
            raise ValueError(f"Cell {key} is missing an aggregation: {sorted(units)}")
    return kept


def summarize(cells):
    sigmas, by_family = [], defaultdict(list)
    levels = {unit: {"0": [], "1": []} for unit in AGGREGATIONS}
    for units in cells.values():
        loss = units["loss"]
        capture = float(loss["baseline_capture"])
        if capture > 0:
            sigma = float(loss["baseline_objection_mass"]) / capture
            sigmas.append(sigma)
            by_family[loss["source"]].append(sigma)
        for unit in AGGREGATIONS:
            for group in ("0", "1"):
                levels[unit][group].append(float(units[unit][f"group_{group}_baseline"]))
    if len(by_family) != EXPECTED_FAMILIES:
        raise ValueError(f"Found {len(by_family)} families, expected {EXPECTED_FAMILIES}")
    return {
        "sigma_contested_share_of_captured_loss": {
            "support": len(sigmas),
            "median": st.median(sigmas),
            "median_of_family_medians": st.median(
                [st.median(v) for v in by_family.values()]
            ),
            "per_family_median": {k: st.median(v) for k, v in sorted(by_family.items())},
            "read_from": "loss aggregation only",
        },
        "baseline_group_objection_level": {
            unit: {
                "group_0_median": st.median(levels[unit]["0"]),
                "group_1_median": st.median(levels[unit]["1"]),
                "absolute_gap_median": st.median(
                    [abs(a - b) for a, b in zip(levels[unit]["0"], levels[unit]["1"])]
                ),
            }
            for unit in AGGREGATIONS
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=None, help="panel instead of the zip")
    args = parser.parse_args()
    print(json.dumps(summarize(baseline_rows(panel(args.csv))), indent=2))


if __name__ == "__main__":
    main()
