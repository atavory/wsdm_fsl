# W16 Public Runner Commands

The public runner writes candidate-level CSVs.  The paper's frozen CSVs,
deterministic aggregators, and plotting scripts live in the submission evidence
bundle.

## CASdatasets Inputs

Convert each CASdatasets `.rda` file to CSV:

```bash
python w16/export_cas_rda_to_csv.py \
  --input <path/to/ausprivauto0405.rda> \
  --output <path/to/ausprivauto0405.csv>

python w16/export_cas_rda_to_csv.py \
  --input <path/to/beMTPL97.rda> \
  --output <path/to/beMTPL97.csv>
```

Generate Australian MTPL candidate logs, including the constructed stress
cohorts:

```bash
python w16/w16_candidates.py \
  --dataset australian_mtpl \
  --input <path/to/ausprivauto0405.csv> \
  --dataset-scope all \
  --seeds 10 \
  --epochs 20 \
  --batch-size 256 \
  --targets 0.1 \
  --net-lambda 1.0 \
  --floor-quantile 0.05 \
  --output <output-dir>/wsdm_fairness_boundary_candidates.csv
```

Generate Belgian MTPL candidate logs:

```bash
python w16/w16_candidates.py \
  --dataset belgian_mtpl \
  --input <path/to/beMTPL97.csv> \
  --dataset-scope natural \
  --seeds 10 \
  --epochs 20 \
  --batch-size 256 \
  --targets 0.1 \
  --net-lambda 1.0 \
  --floor-quantile 0.05 \
  --output <output-dir>/wsdm_fairness_boundary_candidates_belgian.csv
```

## KuaiRand-Pure Input

Use the extracted KuaiRand-Pure `data` directory containing
`log_random_4_22_to_5_08_pure.csv`, `user_features_pure.csv`, and
`video_features_basic_pure.csv`.

```bash
python w16/w16_candidates.py \
  --dataset kuairand \
  --input <path/to/KuaiRand-Pure/data> \
  --max-rows 300000 \
  --sample-seed 42 \
  --seeds 10 \
  --epochs 20 \
  --batch-size 512 \
  --targets 0.1 \
  --dataset-scope natural \
  --net-lambda 0.0001642368368238962 \
  --floor-quantile 0.05 \
  --output <output-dir>/wsdm_fairness_boundary_candidates_kuairand_300k.csv
```

For parallel runs, split seeds with `--seed-start` and `--seeds`, then combine
the shard CSVs by keeping one header.
