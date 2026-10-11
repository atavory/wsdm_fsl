# Code-To-Algorithm Audit, 2026-10-10

Scope: paper commit `79514dd5e3d0247cc56d6752ef2e2e1f6059cca6`,
data-evidence commit `e039b9119162d2d60cc69968bd0a460df42fdb4e`, and the
public mirror runners under `fairness/`.

## Findings

The implementation and theorem use group-denominator normalization for the
group-aware complaint price. The appendix sentence saying the group family is
normalized by "validation baseline complaint rates" was wrong.

The code computes, for value-unit comparisons,
`group_base_rates[g] = sum(estimated_value_i for G_i = g) / n_validation`.
This is the empirical version of the theorem's `P_g U_g^A` denominator. The
group penalty then subtracts or adds
`lambda * estimated_value_i / group_base_rates[g]` on complaint-bearing rows.
It is not normalized by the baseline complaint rate or by the mass of only
complaint-bearing rows.

Resolution: the paper appendix now says "validation group-denominator masses"
in Overleaf paper commit `cd69ad8afaa25131ff70981af7a960b895e77ae4`.
The current evidence repository records the audit, regression test, raw-proxy
rerun, and selected-policy mass closure in commit
`e039b9119162d2d60cc69968bd0a460df42fdb4e`.

The second audit finding was an implementation mismatch. The public-group and
W16-style candidate generators used auxiliary `log1p(A)` and `log1p(B)`
regressions, then exponentiated predictions to provide value/loss proxies for
complaint construction. That does not estimate the raw conditional means used
by the population theory. The repaired public runners now fit raw-scale linear
mean regressions with nonnegative prediction clipping.

A later 2026-10-11 data-side audit found a second aggregation mismatch: the
promoted public-auto group-policy bank used W-BCE for the fixed
baseline/reference but a separate loss-regression plug-in ratio for the direct
price sweep. The evidence bundle was regenerated with `--base-score wbce`, so
the promoted public-auto direct group/account policies now use the W-BCE-implied
loss side, `estimated_value * exp(clip(wbce_logit, -50, 50))`. This public
mirror does not ship the private aggregation scripts or generated CSVs; it
records the verified evidence-bundle state.

## Implementation Map

Score construction:

- Public fairness reruns use linear weighted-logistic W-BCE fit on training rows
  plus separate raw-scale linear regressions for positive value and loss
  proxies.
- Frozen W-BCE logits are replayed as loss/value odds: `exp(score)` is
  multiplied by `estimated_value` to recover the fitted loss side before
  complaint prices are applied.
- Rows with nonpositive adjusted loss are not selected by direct price policies.
  The repaired cached scorer preserves W-BCE order for zero estimated-value rows
  with positive W-BCE odds instead of dropping them to zero.

Threshold and budget boundary:

- Validation scores are sorted descending with row index as a deterministic
  tie-breaker for threshold discovery.
- The frozen deployed mask is `score >= threshold and score > 0`.
- If this mask exceeds the configured validation blocked-value budget, the
  evaluated randomized policy uses
  `alpha = min(1, beta / validation_value_budget)`.
- The same alpha, selected price, and threshold are frozen for test. Reported
  metrics are expectations under that thinning.
- This finite-grid protocol is candidate-policy evaluation. It is intentionally
  narrower than the population theorem, which allows boundary randomization to
  satisfy active constraints exactly.

Policy-bank selection:

- Account and group families share the same lambda grid within each generator.
- Account pricing subtracts a common complaint price from all
  complaint-bearing rows.
- Group pricing uses the validation baseline to choose high- and low-complaint
  groups, then applies opposite signed prices normalized by validation
  group-denominator masses.
- For each family, capture floor, split cell, and unit, selection uses
  validation only: minimize absolute validation signed complaint gap, then break
  ties by higher validation capture, smaller price, and deterministic policy
  order.

Comparator LP:

- The comparator optimizes the full frozen group-blind account-price bank with a
  standard-library simplex solver.
- Mixture weights are chosen on validation only, minimizing absolute signed
  complaint gap subject to the same capture floor and expected blocked-value
  cap.
- Among gap-optimal mixtures, the comparator maximizes validation capture and
  then minimizes validation spend. Component thinning and mixture weights are
  frozen for test.

## Status

The denominator wording error is fixed in the paper. The auxiliary log-proxy
mismatch required a rerun and has been repaired in the evidence bundle. The
promoted public-auto scorer mismatch was also repaired by uniform W-BCE
regeneration. The selected-policy mass accounting gate is closed in the
evidence bundle: 2,880/2,880 value-unit selected policies now have exact
repaired mass rows.

The current paper headline after the uniform W-BCE regeneration is median lift
`+0.5447`, source signs `6/0/1`, and two-sided cluster p-value `0.002938` at the
primary `.95` floor. The previous raw-scale but mixed-scorer headline was
`+0.5010` with signs `7/0/0`; the uniform regeneration increases the median
lift and restores the full 180 paired cells.

## Accepted method and verification boundary

Auxiliary regressions minimize squared error on raw nonnegative outcomes.
Training-only centering/scaling are inverted, then negative predictions are
projected to zero; value proxies used in ratios have a 1e-9 numerical floor.
This does not assert exact conditional-mean recovery by a finite linear model.
The separate log-damped score-learning robustness comparison is unaffected.

The raw-proxy repair changed eligibility to 149 paired cells, versus 180 before
that repair, so +0.6131 to +0.5010 was not a prediction-only change on an
identical sample. The later uniform-W-BCE regeneration restores 180 paired
seven-source cells. The exact seven-source sign-flip p is 0.015625 one-sided and
0.03125 two-sided. Independent data-side headline verification checked 5,028
selection rows, LP verification checked 920 cells with no failures, zero-price
identity checked 360 public-auto lambda-zero pairs with no mismatches, and the
submission artifact tests pass 16 tests.

Selected-policy objection-mass records are now complete: 2,880/2,880
value-unit selected policies have exact repaired mass rows and zero validation
budget violations. Data/evidence commits
`e5eb6240f4f1e5767f04d6845675a1befa625787` and
`663525698be943068c6ee90393173347f21ecbce` record the regeneration and
provenance update. Final Overleaf rendering and the full review harness remain
pending. Previous all-PASS or old-PDF reports do not certify the current paper.
