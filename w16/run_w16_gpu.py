"""Run W16 CUDA candidate generation and deterministic summaries."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from provenance import print_provenance


def run(command: list[str]) -> None:
    print(" ".join(command), flush=True)
    subprocess.run(command, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--australian-rda", required=True)
    parser.add_argument("--belgian-rda", required=True)
    parser.add_argument("--kuairand-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--summary-python", default=sys.executable)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--lambda-grid-points", type=int, default=41)
    parser.add_argument("--kuairand-max-rows", type=int, default=300_000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print_provenance([])
    root = Path(__file__).resolve().parent
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    cas_candidates = out / "w16_cas_torch_gpu_candidates.csv"
    cas_summary = out / "w16_cas_torch_gpu_summary.csv"
    kuairand_candidates = out / "w16_kuairand_torch_gpu_candidates.csv"
    kuairand_summary = out / "w16_kuairand_torch_gpu_summary.csv"

    run(
        [
            args.python,
            str(root / "w16_cas_torch_gpu.py"),
            "--australian-rda",
            args.australian_rda,
            "--belgian-rda",
            args.belgian_rda,
            "--datasets",
            "australian,belgian",
            "--belgian-n",
            "100000",
            "--seeds",
            str(args.seeds),
            "--epochs",
            str(args.epochs),
            "--batch-size",
            "4096",
            "--device",
            args.device,
            "--output",
            str(cas_candidates),
        ]
    )
    run(
        [
            args.summary_python,
            str(root / "summarize_w16_boundary.py"),
            "--input",
            str(cas_candidates),
            "--output",
            str(cas_summary),
            "--lambda-grid-points",
            str(args.lambda_grid_points),
        ]
    )
    run(
        [
            args.python,
            str(root / "w16_kuairand_torch_gpu.py"),
            "--input-dir",
            args.kuairand_dir,
            "--max-rows",
            str(args.kuairand_max_rows),
            "--seeds",
            str(args.seeds),
            "--epochs",
            str(args.epochs),
            "--batch-size",
            "8192",
            "--device",
            args.device,
            "--output",
            str(kuairand_candidates),
        ]
    )
    run(
        [
            args.summary_python,
            str(root / "summarize_w16_boundary.py"),
            "--input",
            str(kuairand_candidates),
            "--output",
            str(kuairand_summary),
            "--lambda-grid-points",
            str(args.lambda_grid_points),
        ]
    )


if __name__ == "__main__":
    main()
