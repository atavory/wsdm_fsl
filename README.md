# WSDM FSL Experiment Runners

This repository contains public Python runners and deterministic aggregators
for the value-weighted blocking and fairness experiments.

The package includes:

- `w16/w16_cas_torch_gpu.py`: Australian and Belgian CASdatasets inputs.
- `w16/w16_kuairand_torch_gpu.py`: KuaiRand-Pure randomized exposures.
- `w16/summarize_w16_boundary.py`: deterministic per-candidate CSV
  aggregation.
- `w16/summarize_w16_boundary_fast.py`: vectorized deterministic aggregation
  with the same output schema, validated against the stdlib summarizer on the
  smoke run.
- `w16/run_w16_gpu.py`: one-command public Python orchestration for both
  candidate generators and summaries.

Raw datasets and generated CSVs are not committed. Supply local copies of the
public datasets when running the scripts.

## Inputs

- Australian MTPL: `ausprivauto0405.rda` from CASdatasets.
- Belgian MTPL: `beMTPL97.rda` from CASdatasets.
- KuaiRand-Pure: the extracted `data` directory containing
  `log_random_4_22_to_5_08_pure.csv`, `user_features_pure.csv`, and
  `video_features_basic_pure.csv`.

## Environment

Install the pinned public dependencies:

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
```

CUDA is used when available; otherwise the runners fall back to CPU execution.

## Run

Use the combined runner for the standard W16 candidate and summary outputs:

```bash
python w16/run_w16_gpu.py \
  --australian-rda <path/to/ausprivauto0405.rda> \
  --belgian-rda <path/to/beMTPL97.rda> \
  --kuairand-dir <path/to/KuaiRand-Pure/data> \
  --output-dir <run output dir> \
  --device cuda:0 \
  --seeds <N> \
  --epochs <N> \
  --lambda-grid-points 41
```

The runner writes candidate-level CSVs and deterministic summary CSVs into the
output directory. Each runner prints command-line and package-version
provenance to stdout.

For a reproducible run record the Git commit, exact command, pinned
requirements, raw input source URLs, raw input hashes, generated CSV paths, and
CSV checksums.
