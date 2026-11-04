# Results

This folder collects the outputs produced by the scripts in [`src/`](../src/).
All generated CSV tables and figures are written here (tracked outputs are
committed; generated files match `results/*.csv` / `results/*.png` in
`.gitignore`).

## What each script produces

All model scripts print to the console:

- Per-seed held-out metrics for both targets (width, depth): **R², RMSE, MAE, MAPE**
- GWO convergence history per seed
- Pooled residual statistics over all seeds (mean and standard deviation, in μm)
- Final **mean ± std** summary over all five seeds

### Model scripts

| Script | Console output | Files written here |
|---|---|---|
| `src/gpr.py` | metrics + **conformal calibration (PICP / NLPD)** + **SHAP** + **PDP/ICE** | `gpr_results_<mode>_<test>.csv` |
| `src/svr.py`, `src/mlp.py`, `src/random_forest.py`, `src/polynomial_ridge.py` | metrics + convergence + residuals | — |

### Validation & analysis scripts

| Script | Console output | Files written here |
|---|---|---|
| `src/vif_analysis.py` | VIF + condition number per feature set S1–S5 | `vif_table.csv`, `correlation_analysis.png` |
| `src/gwo_sensitivity.py` | NLL at iterations 10/25/50 + runtime per GWO population | `gwo_sensitivity.csv`, `GWO_Sensitivity_Analysis.png` |
| `src/quality_gating.py` | three-target gating verdicts (dense / sparse / out-of-domain), verdict accuracy, sigma inflation | `gating_target1_dense.csv`, `gating_target2_sparse.csv`, `gating_target3_outofdomain.csv`, `gating_three_target_summary.csv` |

## Figures

Diagnostic figures (parity plots, residual diagnostics, QQ plots, GWO convergence
curves, SHAP summaries, PDP/ICE curves) are opened in interactive matplotlib
windows when a script runs; the analysis scripts additionally save their key
figures here. Adapt further plotting calls to `plt.savefig(...)` to keep more.

## Reproducing the paper tables

A step-by-step guide (environment setup → data → per-figure/per-table commands) is in
[`docs/reproduction.md`](../docs/reproduction.md).
