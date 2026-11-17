# Completed direct-level reductions

Both original objectives completed all 240 cases and T evaluation. The
capture arm completed first. The user later requested completion of the
value arm, after capture results had been examined. We reused 180 value C
seals and refit the remaining 60 under the same objective, splits, group
bounds, resources, and learner settings. The 60 reruns reproduce the saved
partial C predictions exactly. This continuation supplies exploratory
evidence, with its timing preserved.

## Numerical reproduction

Extract `artifacts/bandha_complete_levels_receipt_20261117.zip` and run:

```sh
python3 scripts/bandha_complete_level_results_20261117.py \
  --root . \
  --frozen results/csvs/bandha_level_reductions_complete_20261117 \
  --output verification
```

The program uses Python stdlib and frozen CSVs. It regenerates both
objectives, original-group and refined-partition comparisons, equal-source
summaries, and matched value-cap comparisons, with every resource failure.

## Experiment source

- `fairness/bandha_resume_value_levels_20261117.py`: continuation and separate
  value-arm sealing and evaluation.
- `fairness/bandha_export_complete_levels_20261117.py`: prediction and seal
  checks, frozen moments, and both-objective CSV export.
- `artifacts/bandha_complete_levels_runtime_20261117.zip`: exact source
  versions used for the completed fits and export, including the data-side
  fitting helpers and the constant-classifier exporter fix.

The immutable runtime snapshot preserves the source hashes in the C freezes.
Later formatting of the public scripts does not change those historical
sources. Reproducing predictions requires the original raw-input and fitted
model caches identified in the source metadata. The numerical receipt
reproduces the comparisons independently of those caches.
