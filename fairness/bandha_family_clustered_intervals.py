#!/usr/bin/env python3
"""Source-family clustered intervals for the manuscript's Table 1 contrasts.

Reads the frozen final-T panel straight out of
``artifacts/bandha_xgboost_evidence_20261113.zip`` and uses only the Python
standard library, so it runs from a clone with no dependencies installed.

Why cluster. The 240 cells are not 240 independent observations: they are 24
evaluation settings under ten seeds, the seeds resample rows within a setting,
five pairs of settings run on identical splits in all ten seeds, and one
French source family supplies 90 of the cells. The independent unit is the
source family, of which the panel records ten. Averaging within each family
first and forming the interval across the ten family means is therefore the
honest comparison, and it is the one the manuscript quotes.

What this implies about adding runs. More seeds shrink the within-family term,
which is not the binding one here. Quoting a cell-level interval over more
cells, as though the cells were independent, would understate uncertainty
rather than measure it. The script prints the naive paired-cell standard error
alongside the clustered one so the ratio can be read directly; it does not
assume which is larger. Clustering widens the interval where between-family
heterogeneity dominates, as it does for removal, and narrows it where
within-family noise dominates, as it does for retained capture.

Falsification tests abort the run:
  * the panel must contain exactly ten source families
  * each contrast's families must pair exactly across the two policies
  * the panel must contain the 240 recorded cells
  * the two contrasts the manuscript quotes must reproduce to the precision
    at which it quotes them

Usage:
    python3 fairness/bandha_family_clustered_intervals.py
    python3 fairness/bandha_family_clustered_intervals.py --csv path/to/T.csv
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import statistics as st
import zipfile
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
EVIDENCE_ZIP = REPO / "artifacts/bandha_xgboost_evidence_20261113.zip"
PANEL_MEMBER = "results/csvs/bandha_xgboost_20261113/main/T.csv"

EXPECTED_FAMILIES = 10
EXPECTED_CELLS = 240

# two-sided 97.5th percentile of Student t, by degrees of freedom
T_975 = {9: 2.262157, 8: 2.306004, 7: 2.364624, 6: 2.446912}

CONTRASTS = (("common", "thinning"), ("selective", "thinning"))
METRICS = (("removal", "value"), ("capture_retained", "value"))

# the two figures the manuscript states, in percentage points, with the
# tolerance implied by the decimal place it prints them to
QUOTED = {
    "common_minus_thinning_removal_value": (44.6, 0.05),
    "common_minus_thinning_capture_retained_value": (-5.6, 0.05),
}


def load_rows(csv_path: Path | None) -> list[dict[str, str]]:
    if csv_path is not None:
        with csv_path.open(newline="") as handle:
            return list(csv.DictReader(handle))
    with zipfile.ZipFile(EVIDENCE_ZIP) as archive:
        raw = archive.read(PANEL_MEMBER).decode("utf-8")
    return list(csv.DictReader(io.StringIO(raw)))


def as_float(value: str) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def by_family(
    rows: list[dict[str, str]], policy: str, column: str, unit: str
) -> dict[str, list[float]]:
    out: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        if row["policy"] != policy or row["unit"] != unit:
            continue
        value = as_float(row[column])
        if value is not None:
            out[row["source"]].append(value)
    return out


def by_cell(
    rows: list[dict[str, str]], policy: str, column: str, unit: str
) -> dict[tuple[str, str], float]:
    out: dict[tuple[str, str], float] = {}
    for row in rows:
        if row["policy"] != policy or row["unit"] != unit:
            continue
        value = as_float(row[column])
        if value is not None:
            out[(row["dataset"], row["seed"])] = value
    return out


def contrast(
    rows: list[dict[str, str]], a: str, b: str, column: str, unit: str
) -> tuple[dict[str, object], list[str]]:
    failures: list[str] = []
    fam_a = by_family(rows, a, column, unit)
    fam_b = by_family(rows, b, column, unit)
    families = sorted(set(fam_a) & set(fam_b))
    if set(families) != set(fam_a) or set(families) != set(fam_b):
        failures.append(f"{a}-{b} {column}: families do not pair exactly")

    diffs = [st.mean(fam_a[f]) - st.mean(fam_b[f]) for f in families]
    n = len(diffs)
    mean = st.mean(diffs)
    clustered_se = st.stdev(diffs) / math.sqrt(n)
    crit = T_975[n - 1]

    # the comparison a reader must not be given: the same contrast paired cell
    # by cell and treated as 240 independent observations. This is the standard
    # error more seeds would shrink, which is why it is printed next to the one
    # that governs the claim.
    cell_a = by_cell(rows, a, column, unit)
    cell_b = by_cell(rows, b, column, unit)
    keys = sorted(set(cell_a) & set(cell_b))
    cell_diffs = [cell_a[k] - cell_b[k] for k in keys]
    naive_se = st.stdev(cell_diffs) / math.sqrt(len(cell_diffs))

    return (
        {
            "families": n,
            "mean_difference_points": 100.0 * mean,
            "ci95_low_points": 100.0 * (mean - crit * clustered_se),
            "ci95_high_points": 100.0 * (mean + crit * clustered_se),
            "t": mean / clustered_se,
            "significant_at_05": abs(mean / clustered_se) > crit,
            "clustered_se_points": 100.0 * clustered_se,
            "naive_paired_cell_se_points": 100.0 * naive_se,
            "naive_paired_cells": len(cell_diffs),
            "clustered_se_over_naive_se": clustered_se / naive_se,
        },
        failures,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="read an already extracted final-T panel instead of the archive",
    )
    args = parser.parse_args()

    rows = load_rows(args.csv)
    failures: list[str] = []

    anchor = by_family(rows, "common", "capture_retained", "value")
    cells_per_family = {k: len(v) for k, v in sorted(anchor.items())}
    total_cells = sum(cells_per_family.values())
    if len(anchor) != EXPECTED_FAMILIES:
        failures.append(f"{len(anchor)} source families, expected {EXPECTED_FAMILIES}")
    if total_cells != EXPECTED_CELLS:
        failures.append(f"{total_cells} cells in the panel, expected {EXPECTED_CELLS}")

    result: dict[str, object] = {
        "kind": "source-family clustered intervals on the frozen final-T panel",
        "independent_unit": "source family",
        "panel": PANEL_MEMBER,
        "cells_per_family": cells_per_family,
        "contrasts": {},
    }
    contrasts: dict[str, dict[str, object]] = {}
    for column, unit in METRICS:
        for a, b in CONTRASTS:
            stats, problems = contrast(rows, a, b, column, unit)
            contrasts[f"{a}_minus_{b}_{column}_{unit}"] = stats
            failures.extend(problems)
    result["contrasts"] = contrasts

    for key, (expected, tol) in QUOTED.items():
        got = contrasts[key]["mean_difference_points"]
        if abs(got - expected) > tol:
            failures.append(
                f"{key}: panel gives {got:+.4f} points, manuscript quotes {expected:+.1f}"
            )

    if failures:
        raise SystemExit("FALSIFICATION TESTS FAILED:\n  " + "\n  ".join(failures))

    result["falsification_tests"] = {
        "ten_independent_source_families": "pass",
        "families_pair_exactly_across_policies": "pass",
        "panel_has_240_cells": "pass",
        "manuscript_quoted_contrasts_reproduce": "pass",
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
