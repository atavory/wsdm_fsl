# WSDM FSL Public Runners

This repository contains public Python runners for the value-weighted blocking
experiments.  It intentionally excludes raw datasets, frozen result CSVs,
aggregation scripts, and plotting scripts.  Those artifacts live in the
submission evidence bundle.

The package includes:

- `score/`: public score-learning runners for the ten yield-Gini datasets.
- `fairness/export_cas_rda_to_csv.py`: converts CASdatasets `.rda` data frames to
  CSV without requiring an R runtime.
- `fairness/fairness_candidates.py`: generates candidate-level fairness logs for Australian
  MTPL, Belgian MTPL, and KuaiRand-Pure.

Supply local copies of the public datasets when running the scripts.

## Inputs

- Australian MTPL: `ausprivauto0405.rda` from CASdatasets.
- Belgian MTPL: `beMTPL97.rda` from CASdatasets.
- KuaiRand-Pure: the extracted `data` directory containing
  `log_random_4_22_to_5_08_pure.csv`, `user_features_pure.csv`, and
  `video_features_basic_pure.csv`.

## Environment

Install the public dependencies:

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
```

## Run

Score-learning runners live under `score/`.  Each script exposes its input and
output arguments through `--help`; generated CSVs should stay in a local output
directory and are not committed here.

Export CASdatasets inputs when needed:

```bash
python fairness/export_cas_rda_to_csv.py \
  --input <path/to/ausprivauto0405.rda> \
  --output <path/to/ausprivauto0405.csv>
```

Generate candidate-level logs:

```bash
python fairness/fairness_candidates.py \
  --dataset australian_mtpl \
  --input <path/to/ausprivauto0405.csv> \
  --dataset-scope all \
  --seeds 10 \
  --epochs 20 \
  --targets 0.1 \
  --output <output-dir>/australian_candidates.csv
```

See `fairness/RUN_COMMANDS.md` for Belgian MTPL and KuaiRand examples.  Save the Git
commit, exact command, pinned requirements, raw input source URLs, raw input
hashes, generated CSV paths, and CSV checksums with any run.
