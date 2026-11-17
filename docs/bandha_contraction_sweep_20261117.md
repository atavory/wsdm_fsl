# Contraction cap and capture-floor replay

We replay the frozen XGBoost primary over beta in `(0.05, 0.10, 0.20)`
and rho in `(0.90, 0.95, 0.99)`: all nine combinations and all 240 cases.
We train no model. We compare common contraction with uniform rho thinning.

V fixes each cap's baseline cut, fractional multiplier, harm/net-harm cuts
and replacement cuts. C fits the common factor and replacement weights for
each cap/floor pair. We seal every C policy before T evaluation. Every
failure remains in the outputs. This replay uses previously examined data.

## Commands

Supply the original frozen data checkout and XGBoost prediction cache:

```sh
python fairness/bandha_contraction_sweep_20261117.py fit \
  --data-root /path/to/data-checkout \
  --run /path/to/bandha_fsl_xgboost_20261113 \
  --output /path/to/contraction-sweep --workers 4

python fairness/bandha_contraction_sweep_20261117.py test \
  --data-root /path/to/data-checkout \
  --run /path/to/bandha_fsl_xgboost_20261113 \
  --output /path/to/contraction-sweep
```

The 10%/.95 point must reproduce the original C and T outcomes, including
attainment flags and undefined ratios. Prediction and C-seal hashes must
match the original cache. The source identity includes the original
selector modules from the data checkout. Use the project's pinned NumPy,
SciPy and pandas environment; no raw data or XGBoost execution is needed.

## Interpretation

Each cap changes the baseline and the criterion thresholds. Comparisons
therefore use each cap's own baseline and fixed contested-candidate set.
The full grid includes baseline capture as a fraction of total loss,
realized baseline value spend, removal, retained capture, target attainment,
and C feasibility. Source-family summaries keep seeds within their source
family, as in the original primary.
