#!/usr/bin/env python3
"""Source-family clustered intervals for the contraction / reductions contrast.

Reads the frozen receipt straight out of
``artifacts/bandha_agarwal_numerical_receipt_20261116.zip`` and uses only the
Python standard library, so it runs from a clone with no dependencies
installed.

Why cluster, and why not bootstrap. The manuscript already quotes one
clustered interval, for common contraction against thinning, and
``bandha_family_clustered_intervals.py`` produces it by averaging within each
source family and forming a Student-t interval across the ten family means.
Any interval placed beside that one has to be built the same way; a bootstrap
here would be a different estimator reported in the same table, which is
worse than no interval at all. Ten clusters is few, so the interval is wide
by construction, and that is the honest width.

What is contrasted. The comparison is restricted to the cells where both
policies satisfy the 10% value cap, which is the only subset on which the two
are answering the same question. Within it the paired per-cell difference is
averaged by family, so a family contributing nine settings does not carry
nine times the weight of one contributing a single setting.

Falsification tests abort the run:
  * the receipt's recorded sha256 digests must match
  * the matched subset must be the 92 cells the receipt publishes, with the
    90 defined removal ratios
  * both per-policy means must reproduce the receipt's published values
  * the panel must contain exactly ten source families

Usage:
    python3 fairness/bandha_agarwal_clustered_intervals.py
    python3 fairness/bandha_agarwal_clustered_intervals.py --root path/to/dir
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import statistics as st
import zipfile
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RECEIPT_ZIP = REPO / "artifacts/bandha_agarwal_numerical_receipt_20261116.zip"

EXPECTED_FAMILIES = 10
EXPECTED_MATCHED = 92
EXPECTED_SUPPORT = 90
PUBLISHED_MEANS = {"common": 47.612319286013495, "calibrated_EG": -19.616416386357987}

# two-sided 97.5th percentile of Student t, by degrees of freedom
T_975 = {9: 2.262157, 8: 2.306004, 7: 2.364624, 6: 2.446912}


def members(root):
    """Return a name -> text reader over either the zip or an extracted tree."""
    if root is None:
        archive = zipfile.ZipFile(RECEIPT_ZIP)
        digests = json.loads(archive.read("bandha_manifest.json"))["sha256"]
        for name, sha in digests.items():
            if hashlib.sha256(archive.read(name)).hexdigest() != sha:
                raise ValueError("File checksum mismatch: " + name)
        return lambda name: io.StringIO(archive.read(name).decode())
    digests = json.loads((root / "bandha_manifest.json").read_text())["sha256"]
    for name, sha in digests.items():
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != sha:
            raise ValueError("File checksum mismatch: " + name)
    return lambda name: (root / name).open()


def number(row, name):
    raw = row[name]
    return None if raw == "" else float(raw)


def derive(row):
    total, value = number(row, "total_value"), number(row, "blocked_value")
    safe = all(
        number(row, f"group_{g}_level") <= number(row, f"group_{g}_baseline") + 1e-10
        for g in (0, 1)
    )
    return {
        "key": (row["dataset"], row["seed"]),
        "source": row["source"],
        "safe": safe,
        "value_met": value <= 0.10 * total + 1e-9 * max(1, total),
        "removal": number(row, "removal"),
    }


def interval(by_family):
    """Student-t interval across family means, the manuscript's protocol."""
    means = [st.mean(values) for _, values in sorted(by_family.items())]
    if len(means) != EXPECTED_FAMILIES:
        raise ValueError(f"Expected {EXPECTED_FAMILIES} families, got {len(means)}")
    centre = st.mean(means)
    error = st.stdev(means) / math.sqrt(len(means))
    half = T_975[len(means) - 1] * error
    return centre, centre - half, centre + half, error


def naive_error(values):
    return st.stdev(values) / math.sqrt(len(values))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=None, help="extracted receipt")
    parser.add_argument("--json", action="store_true", help="emit JSON only")
    args = parser.parse_args()
    member = members(args.root)

    with member("bandha_controls_T.csv") as handle:
        common = [derive(r) for r in csv.DictReader(handle) if r["unit"] == "value" and r["policy"] == "common"]
    with member("original_train/bandha_T.csv") as handle:
        calibrated = [
            derive(r)
            for r in csv.DictReader(handle)
            if r["unit"] == "value" and r["family"] == "agarwal_eg_Ccal"
        ]

    left = {r["key"]: r for r in common}
    right = {r["key"]: r for r in calibrated}
    matched = {k for k in left if k in right and left[k]["value_met"] and right[k]["value_met"]}
    if len(matched) != EXPECTED_MATCHED:
        raise ValueError(f"Matched subset is {len(matched)}, expected {EXPECTED_MATCHED}")

    support = sorted(k for k in matched if left[k]["removal"] is not None and right[k]["removal"] is not None)
    if len(support) != EXPECTED_SUPPORT:
        raise ValueError(f"Removal support is {len(support)}, expected {EXPECTED_SUPPORT}")
    for name, side in (("common", left), ("calibrated_EG", right)):
        got = 100 * st.mean([side[k]["removal"] for k in support])
        if abs(got - PUBLISHED_MEANS[name]) > 1e-9:
            raise ValueError(f"{name} mean {got} does not reproduce the receipt")

    removal_by_family = defaultdict(list)
    for k in support:
        removal_by_family[left[k]["source"]].append(100 * (left[k]["removal"] - right[k]["removal"]))
    safe_by_family = defaultdict(list)
    for k in sorted(matched):
        safe_by_family[left[k]["source"]].append(100 * (int(left[k]["safe"]) - int(right[k]["safe"])))

    removal = interval(removal_by_family)
    safe = interval(safe_by_family)
    naive = naive_error([v for values in removal_by_family.values() for v in values])

    result = {
        "matched_cells": len(matched),
        "removal_support": len(support),
        "families": len(removal_by_family),
        "paired_value_removal_difference": {
            "points": removal[0], "ci": [removal[1], removal[2]],
            "clustered_standard_error": removal[3],
            "naive_cell_standard_error": naive,
            "understatement_ratio": removal[3] / naive,
        },
        "paired_group_nonincrease_difference": {
            "points": safe[0], "ci": [safe[1], safe[2]],
            "clustered_standard_error": safe[3],
        },
        "unpaired_counts": {
            "common_group_nonincrease": sum(left[k]["safe"] for k in matched),
            "calibrated_EG_group_nonincrease": sum(right[k]["safe"] for k in matched),
        },
    }
    if args.json:
        print(json.dumps(result, indent=2))
        return
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
