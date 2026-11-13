"""Package frozen aggregate evidence and exact supporting runtime sources."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import zipfile
from pathlib import Path


def sha(content):
    return hashlib.sha256(content).hexdigest()


def write_archive(path, files):
    metadata = {
        name: {"sha256": sha(content), "bytes": len(content)}
        for name, content in files.items()
    }
    files = {
        **files,
        "MANIFEST.json": (
            json.dumps(metadata, indent=2, sort_keys=True) + "\n"
        ).encode(),
    }
    with zipfile.ZipFile(
        path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6
    ) as archive:
        for name, content in sorted(files.items()):
            info = zipfile.ZipInfo(name, date_time=(2026, 11, 13, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, content)
    return {
        "path": str(path),
        "sha256": sha(path.read_bytes()),
        "bytes": path.stat().st_size,
    }


def support_sources(root):
    pending = [
        "bandha_objection_safe_selector",
        "bandha_empirical_cut_selector",
        "bandha_direct_constrained_selector",
        "ultra_value_retention_selector_20261105",
        "bandha_normalized_group_prices",
    ]
    files = {}
    while pending:
        name = pending.pop()
        relative = "scripts/" + name + ".py"
        path = root / relative
        if relative in files or not path.is_file():
            continue
        content = path.read_bytes()
        files[relative] = content
        tree = ast.parse(content)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                pending.extend(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                pending.append(node.module.split(".")[0])
    files["README.md"] = (
        b"# XGBoost runtime support\n\nExact, unchanged supporting sources used by the boosted fairness run. Extract into a fresh copy of the frozen data project, giving it a scripts/ directory. Original module names and source attribution are retained. MANIFEST.json binds every file; these are runtime dependencies, not a new experiment.\n"
    )
    return files


def evidence_files(root):
    prefix = "results/csvs/bandha_xgboost_20261113/"
    names = [
        prefix + "main/T.csv",
        prefix + "main/C.csv",
        prefix + "diagnostics/original_test.csv",
        prefix + "diagnostics/V_FREEZE.json",
        prefix + "sensitivity/T.csv",
        prefix + "sensitivity/V_metrics.csv",
        prefix + "sensitivity/model_rounds.csv",
        "scripts/bandha_xgboost_report.py",
        "paper_support/wsdm_short/ultra_wsdm_frozen_evidence_20261106.json",
        "results/bandha_xgboost_20261113/summary.json",
        "results/bandha_xgboost_20261113/results_table.tex",
    ]
    files = {name: (root / name).read_bytes() for name in names}
    files["README.md"] = (
        b"# Frozen XGBoost paper evidence\n\nAll 240 cells; boosted baseline and criteria. Aggregate outcomes contain no raw feature rows. Python 3.12+ stdlib suffices.\n\nRun from this extracted directory:\n\n```sh\npython scripts/bandha_xgboost_report.py --root . --run results/csvs/bandha_xgboost_20261113/main --diagnostics results/csvs/bandha_xgboost_20261113/diagnostics --sensitivity results/csvs/bandha_xgboost_20261113/sensitivity --v-metrics results/csvs/bandha_xgboost_20261113/sensitivity/V_metrics.csv --output reproduced\n```\n\nThe reproduced table must match the frozen results_table.tex byte-for-byte. Summary scientific fields must agree; provenance paths differ on another machine. The training runtime support is a separate archive.\n"
    )
    return files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    outputs = [
        write_archive(
            args.output / "bandha_xgboost_runtime_20261113.zip",
            support_sources(args.data_root),
        ),
        write_archive(
            args.output / "bandha_xgboost_evidence_20261113.zip",
            evidence_files(args.data_root),
        ),
    ]
    (args.output / "bandha_xgboost_packages_20261113.json").write_text(
        json.dumps(outputs, indent=2) + "\n"
    )
    print(json.dumps(outputs, indent=2))


if __name__ == "__main__":
    main()
