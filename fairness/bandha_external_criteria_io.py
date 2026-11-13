"""Portable loading of the immutable primary panel, with T outcomes gated."""

from __future__ import annotations

import base64
import csv
import gzip
import hashlib
import json
import logging
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

LOGGER = logging.getLogger(__name__)


def digest(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def save_json(path, content):
    path.write_text(
        json.dumps(content, sort_keys=True, indent=2, allow_nan=False) + "\n"
    )


def read_manifest(path, root):
    manifest = json.loads(path.read_text())
    for item in manifest["inputs"]:
        if digest(root / item["path"]) != item["sha256"]:
            raise ValueError(f"Input checksum mismatch: {item['path']}")
    if len(manifest["cases"]) != 240:
        raise ValueError("Expected 240 primary cases")
    return manifest


def metadata(row, manifest):
    if "d" in row:
        dataset = manifest["compact_dataset"].get(row["d"])
        return dataset, row["s"], float(row["t"]), "V" if row["p"] == "0" else "test"
    split = row.get("sample_split", row.get("split"))
    return (
        row["dataset"],
        row["seed"],
        float(row["target"]),
        "V" if split == "validation" else split,
    )


def numeric_row(row, case, manifest):
    compact = "d" in row
    group = manifest["compact_group"][row["g"]] if compact else row["group"]
    a_hat = max(float(row["ah"] if compact else row["estimated_value"]), 0)
    logit = float(row["q"] if compact else row["score"])
    ratio = math.exp(max(-50, min(50, logit)))
    ratio = ratio * a_hat / a_hat if a_hat > 0 else ratio
    fields = (
        ("A", "B", "bh", "h", "nh", "w")
        if compact
        else (
            "value",
            "loss",
            "estimated_loss",
            "selected_criterion_harm",
            "selected_criterion_net_harm",
            "selected_criterion_wbce_ratio",
        )
    )
    a, b, b_hat, h, nh, w = [float(row[k]) for k in fields]
    values = (case["groups"].index(group), ratio, a, b, a_hat, max(b_hat, 0), h, nh, w)
    if not all(math.isfinite(x) for x in values) or min(a, b) < 0:
        raise ValueError("Nonfinite or negative outcome")
    if any(x not in (0, 1) for x in (h, nh, w)):
        raise ValueError("Expected frozen binary criterion indicators")
    return values


def assigned_part(case, ordinal):
    bits = base64.b64decode(case["partition_bits_base64"])
    return "C" if bits[ordinal // 8] & (1 << (ordinal % 8)) else "T"


def scan_file(root, relative, cases, manifest, wanted):
    lookup = {(c["dataset"], c["seed"]): c for c in cases}
    counts, arrays = defaultdict(int), defaultdict(list)
    decoded = {
        key: base64.b64decode(c["partition_bits_base64"]) for key, c in lookup.items()
    }
    with gzip.open(root / relative, "rt", newline="") as handle:
        for row in csv.DictReader(handle):
            dataset, seed, target, split = metadata(row, manifest)
            key = dataset, seed
            if key not in lookup or target != 0.1 or split not in ("V", "test"):
                continue
            part = split
            if split == "test":
                ordinal = counts[key]
                counts[key] += 1
                part = "C" if decoded[key][ordinal // 8] & (1 << (ordinal % 8)) else "T"
            if part in wanted:
                arrays[(key, part)].append(numeric_row(row, lookup[key], manifest))
    for key, case in lookup.items():
        if counts[key] != case["n_test_rows"]:
            raise ValueError(f"Test membership length changed: {key}")
    return {key: np.asarray(value, dtype=np.float64) for key, value in arrays.items()}


def load_parts(root, manifest, wanted):
    files = defaultdict(list)
    for case in manifest["cases"]:
        files[case["input"]].append(case)
    result = {}
    for relative, cases in files.items():
        result.update(scan_file(root, relative, cases, manifest, wanted))
        LOGGER.info("Loaded %s for %s", sorted(wanted), relative)
    return result


def native(data):
    return ~((data[:, 6] > 0.5) & (data[:, 7] > 0.5))


def baseline(data, case):
    return case["baseline_alpha"] * (
        (data[:, 1] >= case["baseline_threshold"]) & (data[:, 1] > 0)
    )


def plug_in_score(data):
    return data[:, 5] / np.maximum(data[:, 4], 1e-9)


def criterion(data, name, plugin_threshold):
    q = native(data)
    if name == "native":
        return q
    if name == "native_wbce":
        return q | (data[:, 8] < 0.5)
    if name == "native_plugin":
        return q | (plug_in_score(data) < plugin_threshold)
    if name == "native_joint":
        return q | (data[:, 8] < 0.5) | (plug_in_score(data) < plugin_threshold)
    raise ValueError(f"Unknown criterion: {name}")


def cut(ratio, threshold):
    if threshold is None:
        return np.zeros(len(ratio))
    if threshold == "full":
        return np.ones(len(ratio))
    return (ratio >= threshold).astype(float)


def mixture(ratio, thresholds, weights):
    result = np.zeros(len(ratio))
    for threshold, weight in zip(thresholds, weights, strict=True):
        result += weight * cut(ratio, threshold)
    return result


def original_probability(data, case):
    p0, q = baseline(data, case), native(data)
    params = case["original_common"]
    rule = params["q0_rule"]
    weights = [1 - rule["right_weight"], rule["right_weight"]]
    fill = mixture(data[:, 1], rule["thresholds"], weights)
    s = 0 if params["resource_infeasible"] else params["contraction_s"]
    return np.where(q, s * p0, p0 + (1 - p0) * fill)


def probability(data, case, policy):
    p0 = baseline(data, case)
    if policy["mode"] == "published":
        return original_probability(data, case)
    if policy["mode"] == "uniform":
        return policy["s"] * p0
    q = criterion(data, policy["criterion"], policy["plugin_threshold"])
    if policy["mode"] == "no_addition":
        return np.where(q, policy["s"] * p0, p0)
    fill = mixture(data[:, 1], policy["thresholds"], policy["weights"])
    nonq = fill if policy["mode"] == "flexible" else p0 + (1 - p0) * fill
    return np.where(q, policy["s"] * p0, nonq)
