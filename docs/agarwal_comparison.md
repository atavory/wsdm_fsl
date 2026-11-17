# Constrained-learning reductions comparison

The short paper compares the baseline-bounded contraction policy against a
constrained-learning reduction. This document records that comparison and the
commands that reproduce its numbers.

**Receipt:** `artifacts/bandha_agarwal_numerical_receipt_20261116.zip`.

## What was run

Fairlearn 0.15.0 `ExponentiatedGradient` with XGBoost cost-sensitive best
responses, not a hand-selected penalty bank. All 240 setting-seed
configurations were fitted on the original training split. Baseline and
criterion predictions, V-frozen thresholds, and C/T memberships are the same
objects the primary run uses, so the comparison is on identical rows.

The adaptation minimizes missed captured loss under three moment constraints:
both signs of a `.01` bound on the value-unit objection gap, a pooled `.10`
cap on blocked value, and a `.95` floor on same-split baseline capture.
Learner settings are depth 6, 100 rounds, `eta .05`, `min_child_weight 5`,
row and column subsampling `.8`, `lambda 1`, initial probability `.5`; EG uses
`eps/nu .002` and at most 50 iterations.

Acquired classifiers are separately mixed on C against the baseline and an
all-pass rule, with all C weights frozen before any T outcome is read. Raw EG
and this calibrated bank are different methods and are recorded separately.
The earlier C-fitted variant is retained in the receipt as a diagnostic rather
than dropped, because its C-to-T divergence is what prompted the
original-training protocol.

## Results

Calibrated EG meets its own T gap bound in 125 of 240 cells and the value cap
in 117. Sixty-six cells meet both, and **34 of those 66 raise a group's
value-objection level**. The paper does not quote this figure. Its "66 cells
meeting both targets and every known bound", of which 37 show an increase
under baseline-support refinement, is a different quantity from a different
run: the direct-levels value arm recorded in
`bandha_complete_levels_20261117.md`. The two 66s coincide by accident.

On the 92 cells where both the contraction policy and calibrated EG satisfy
the value cap, with 90 defined removal ratios:

| | mean value removal | group non-increase | joint attainment |
|---|---:|---:|---:|
| Common contraction | 47.61% | 92 / 92 | 37 |
| Calibrated EG | −19.62% | 36 / 92 | 18 |

Negative removal means objection mass increased.

## Scope

These results distinguish controlling a gap from protecting each group's
level. They do **not** establish inferiority of the reductions framework. The
objectives differ: the comparator maximizes capture under an absolute gap
bound, while contraction minimizes its objection retention factor under
capture and value targets. The framework can impose group-level non-increase
directly, and that variant has not been run. Post-processing and
exposure-parity methods are likewise not drop-in substitutes for this
cost/objection problem without a declared adaptation.

## Verify

The verifier uses only the Python standard library and reads the frozen CSVs
in the receipt. Extract into an empty directory and run it against that
directory:

```bash
mkdir -p agarwal_receipt
unzip artifacts/bandha_agarwal_numerical_receipt_20261116.zip -d agarwal_receipt
python3 agarwal_receipt/bandha_verify_agarwal_receipt_20261116.py --root agarwal_receipt
```

It recomputes the 34/66 result and the 92-cell comparison above, and checks
the included source copies against the recorded training seal hashes. Its
verified scope is numerical reproduction. Full retraining additionally
requires the public baseline runtime, the raw datasets, and the generated
baseline model and prediction cache.

## Contents

- `bandha_verify_agarwal_receipt_20261116.py` — standalone stdlib verifier.
- `bandha_controls_T.csv` — contraction-policy T outcomes on the shared rows.
- `original_train/` — primary comparator C and T outcomes.
- `C_fitted_diagnostic/` — the retained earlier variant.
- `training_code/` — protocol and the fitting and moment definitions.
- `bandha_manifest.json`, `bandha_recorded_environment.json` — hashes and
  recorded environment.
