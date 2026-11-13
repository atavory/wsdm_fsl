# XGBoost Fairness Rerun — 2026-11-13

The current short-paper primary uses XGBoost W-BCE and two boosted
raw-mean models. Same 24 cohorts × ten seeds; same immutable V/C/T rows.
All C decisions freeze before final T evaluation. This is exploratory reuse
of previously examined data.

## Results

| Policy | Value-objection removal (%) | Retained capture (%) | Joint cells / 240 |
|---|---:|---:|---:|
| Thinning | 5.00 | 95.00 | 183 |
| Common | 47.59 | 94.62 | 57 |
| Selective | 64.83 | 93.34 | 48 |
| Selective + reserve | 64.83 | 91.33 | 49 |

All 240 repair cells have both-group objection nonincrease. Common factor
error is zero. Removal means have count/value/loss support 240/238/239.
At margin .005 in both V and T, baseline direction reverses in 20 cells
across 14 cohorts. Signed penalties increase a group in 202 original-test
cells, including 137 of 144 gap reductions.

Common results remain similar to the linear run (46.46% removal, 93.17%
capture, 56 joint cells). Selective's old mean-capture advantage does not
transfer. Each setup uses its own baseline and objection definition.
Frequent business failures remain a material limitation.

## Reproduce the table without training

Extract `artifacts/bandha_xgboost_evidence_20261113.zip` into a fresh directory
and follow its README. It includes the exact frozen aggregate inputs, the
linear comparison, and `scripts/bandha_xgboost_report.py`. Python 3.12+ stdlib
suffices. Verify every file against MANIFEST.json; the generated TeX table
must match the frozen table byte-for-byte.

## Retrain and recalibrate

Supply the raw datasets and a copy of the frozen data project containing
`results/csvs/bandha_external_criteria_inputs_20261113.json` and its hashed
candidate inputs. Extract `artifacts/bandha_xgboost_runtime_20261113.zip`
into that fresh data-project copy. The archive provides the exact original
supporting module sources, preserving their names and attribution.

Install `fairness/bandha_xgboost_requirements.txt`. On macOS, XGBoost also
requires an available OpenMP runtime. Configure these task-specific paths:

```sh
FSL_DATA=/path/to/data-copy
FSL_RAW=/path/to/raw-inputs
FSL_RUN=/path/to/boosted-run
FSL_DIAGNOSTICS=/path/to/boosted-diagnostics
FSL_SENSITIVITY=/path/to/boosted-sensitivity
python fairness/bandha_xgboost_rerun.py fit --data-root "$FSL_DATA" --raw "$FSL_RAW" --output "$FSL_RUN" --workers 2 --threads 3
python fairness/bandha_xgboost_rerun.py test --data-root "$FSL_DATA" --raw "$FSL_RAW" --output "$FSL_RUN"
python fairness/bandha_xgboost_diagnostics.py fit --data-root "$FSL_DATA" --run "$FSL_RUN" --output "$FSL_DIAGNOSTICS"
python fairness/bandha_xgboost_diagnostics.py test --data-root "$FSL_DATA" --run "$FSL_RUN" --output "$FSL_DIAGNOSTICS"
python fairness/bandha_xgboost_sensitivity.py fit --data-root "$FSL_DATA" --run "$FSL_RUN" --output "$FSL_SENSITIVITY"
python fairness/bandha_xgboost_sensitivity.py test --data-root "$FSL_DATA" --run "$FSL_RUN" --output "$FSL_SENSITIVITY"
```

The baseline trains `binary:logistic` on target B/(A+B), weighted by A+B;
this is the W-BCE data loss and is not an NDCG objective. Auxiliary targets
are affine-standardized raw amounts with inversion and nonnegative clipping.
Tree depth is six, eta .05, subsample/column subsample .8, lambda one, and
minimum child weight five. A training-only inner holdout selects up to 600
rounds with 40-round early stopping, followed by a full-training refit.
Provider boundaries remain disjoint in CMS. Harm/net harm stay primary;
ratio restoration is separate sensitivity.

## Evidence boundaries

The archive provides aggregate reproduction, not raw candidate features or
the complete fitted booster/prediction snapshot. That snapshot is retained
locally in the data workspace and hash-bound in snapshots.json.
Original-test penalty diagnostics and final-T repair comparisons use
different populations. No pooled means establish per-cohort business
feasibility, estimator dominance, causal harm reduction, or norm legitimacy.
