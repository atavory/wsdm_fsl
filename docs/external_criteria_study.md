# Fixed-criterion expansion and backfill capacity

The three `fairness/bandha_external_criteria_*.py` modules evaluate the exact
primary 24-cohort panel with frozen fractional baseline probabilities and
immutable V/C/T membership. They need NumPy and SciPy, plus the frozen
candidate logs identified in the exported input manifest. They fit no new
predictive models.

The data project's `scripts/bandha_external_criteria_inputs.py` exports the
portable manifest from the original selector, calibration, and membership
freezes. The manifest includes source checksums, group decoding, original
C-fitted policy parameters, and outcome-independent partition bitsets.

Run the phases separately, using the data project as `DATA`:

```sh
python3 fairness/bandha_external_criteria_study.py fit \
  --root "$DATA" \
  --manifest "$DATA/results/csvs/bandha_external_criteria_inputs_20261113.json" \
  --protocol "$DATA/bandha_external_criteria_protocol_20261113.md" \
  --output "$DATA/results/csvs/bandha_external_criteria_20261113"
python3 fairness/bandha_external_criteria_study.py test \
  --root "$DATA" \
  --manifest "$DATA/results/csvs/bandha_external_criteria_inputs_20261113.json" \
  --protocol "$DATA/bandha_external_criteria_protocol_20261113.md" \
  --output "$DATA/results/csvs/bandha_external_criteria_20261113" \
  --evidence "$DATA/paper_support/wsdm_short/ultra_wsdm_frozen_evidence_20261106.json"
```

Fitting skips T before parsing its outcome amounts, writes C CSVs, and seals
all decisions and code hashes. Testing checks those hashes and replays the
published primary common controller against the original sealed evidence.
It then applies frozen new policies without changing parameters.

The comparisons expand the native harm/net-harm union using either the
already supplied W-BCE rule or a plug-in ratio rule thresholded on V. These
are criterion sensitivities, not evidence of stakeholder endorsement. The
flexible controller permits lower nonobjector probabilities than the
baseline while maintaining the common factor on objectors. The no-addition
controller guarantees pointwise probability nonincrease for any fixed
external objection indicator. Capture and value targets remain separate
empirical constraints.

The initial nine-controller study uses runner commit `32b931d`. The
joint-criterion/fixed-reserve follow-up uses `1ce0186`, the separate
`bandha_external_criteria_followup_protocol_20261113.md`, and output directory
`results/csvs/bandha_external_criteria_followup_20261113`. It adds the joint
union, native/joint common controllers fitted at fixed C targets of 0.095
value and 0.98 capture, and a no-addition controller with those reserves.
The original T assessment targets remain 0.10 value and 0.95 capture.

This study follows earlier examination of T and is exploratory. Report
denominators, cohort counts, fit failures, and T failures explicitly.
