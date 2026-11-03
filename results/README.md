# Results

This folder collects the outputs produced by the scripts in [`src/`](../src/).

## What each script produces

All scripts print to the console:

- Per-seed held-out metrics for both targets (width, depth): **R², RMSE, MAE, MAPE**
- GWO convergence history per seed
- Pooled residual statistics over all seeds (mean and standard deviation, in μm)
- Final **mean ± std** summary over all five seeds

`src/gpr.py` additionally performs:

- **Conformal calibration** — empirical coverage (PICP), interval widths, and Negative Log
  Predictive Density (NLPD) at the 95% nominal level
- **SHAP value analysis** — feature attribution ranking for both targets
- **PDP / ICE curves** — partial dependence and individual conditional expectation plots

and writes the per-run result table to the working directory as
`gpr_results_<mode>_<test>.csv` (e.g. `gpr_results_s1_100pct_Data.csv`).

## Figures

Diagnostic figures (parity plots, residual diagnostics, QQ plots, GWO convergence
curves, SHAP summaries, PDP/ICE curves) are opened in interactive matplotlib windows
when a script runs. Move the figures you want to keep into this folder, or adapt the
plotting calls to `plt.savefig(...)`.

## Reproducing the paper tables

A step-by-step guide (environment setup → data → per-figure/per-table commands) is in
[`docs/reproduction.md`](../docs/reproduction.md).
