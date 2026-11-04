# Reproduction guide

This guide reproduces the experiments reported in
[*Robust conformal Gaussian process regression for CO2 laser engraving: Distribution-free uncertainty quantification and automated quality gating*](https://doi.org/10.1016/j.eswa.2026.134356).

## 1. Environment

```bash
# conda (recommended)
conda env create -f environment.yml
conda activate co2-laser-pmma

# or pip
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## 2. Data

Place the experimental datasets in `data/` (schema in [`data/README.md`](../data/README.md)):

```
data/data.csv     # used by all five models
```

## 3. Headline experiment (GPR, raw process parameters)

```bash
python src/gpr.py
```

This runs the full pipeline on selection mode `s1` (Speed, Defocus length, Power):
GWO hyperparameter search → 5-fold CV → multi-seed training (5 seeds) → held-out
evaluation → conformal calibration (PICP / NLPD) → SHAP + PDP/ICE interpretability.
This reproduces the headline result: **R² = 0.999 (depth) / 0.998 (width)** with
conformal prediction intervals achieving **PICP ≥ 0.94** at the 95% nominal level.

## 4. Baseline models

```bash
python src/svr.py
python src/mlp.py
python src/random_forest.py
python src/polynomial_ridge.py
```

Each baseline uses the same protocol (identical seeds, 80/20 split, 5-fold CV,
GWO-tuned hyperparameters), which reproduces the five-architecture comparison
reported in the paper.

## 5. Multicollinearity diagnostics (VIF + condition number)

```bash
python src/vif_analysis.py
```

Computes per-feature VIF and the condition number of the feature matrix for every
candidate feature set S1–S5, reproducing the input-selection table of the paper
(including the exclusion of S5 on grounds of severe collinearity), plus the
feature-feature and feature-output correlation heatmaps. Writes
`results/vif_table.csv` and `results/correlation_analysis.png`.

## 6. GWO budget sensitivity analysis

```bash
python src/gwo_sensitivity.py
```

Compares GWO convergence across population sizes {5, 8, 12, 15, 20} over 50
iterations (identically seeded initial positions), justifying the budget adopted
in the paper (12 wolves, 25 iterations). Writes `results/gwo_sensitivity.csv`
and `results/GWO_Sensitivity_Analysis.png`.

## 7. Quality-gating validation

```bash
python src/quality_gating.py
```

Trains the conformal GPR-S1 surrogate on a single-seed (589) 80/20 split and
evaluates the three-tier decision policy (ACCEPT / FLAG / REJECT) on the 25
held-out points against three targets: dense-region, sparse-region (both drawn
from real measured geometries via the neighborhood-density criterion), and
out-of-domain extrapolation queries. Reproduces the gating tables of the paper:
verdict accuracy 96.0% (dense) / 100% (sparse), uniform rejection out-of-domain
with 15–39× calibrated-sigma inflation. Writes `results/gating_target1_dense.csv`,
`results/gating_target2_sparse.csv`, `results/gating_target3_outofdomain.csv`,
and `results/gating_three_target_summary.csv`.

## 8. Reproducing individual studies

The experiment knobs are at the **bottom of each script**:

| Paper study | Script | Knob |
|---|---|---|
| Feature-set comparison (raw vs composite vs hybrid) | any | `selection_mode = 's1' \| 's2' \| 's3' \| 's4'` |
| Seed robustness | any | `seeds = [405, 166, 327, 347, 589]` |
| Data-efficiency study | `src/gpr.py` | `downsample_fraction` (e.g. `0.50`, `0.25` for the paper's two scarcity levels) |
| Noise-robustness study | `src/gpr.py` | `noise_test=True, noise_level=...` (e.g. `0.05`, `0.10` of per-feature std) |
| Final-paper SHAP figures | `src/gpr.py` | `n_samples=500` (comment at the bottom of the script) |

## 9. Robustness analysis (sensor noise + data scarcity)

Reproduces Section "Robustness of Conformal Calibration Under Data Scarcity
and Sensor Noise" of the paper (Table `tab:robustness_analysis`), which stresses
the conformal GPR surrogate under two exchangeability-violating conditions:
additive Gaussian input noise (5% and 10% of per-feature std, mimicking sensor
degradation) and training-set downsampling (50% and 25% of the 100-point
partition). Both stress conditions are evaluated for **GPR-S1** and **GPR-S3**.

The four adversity scenarios are exposed as keyword arguments at the bottom of
`src/gpr.py` (in `run_multi_seed_experiment_gpr_separate`); `noise_test=True`
injects the same Gaussian perturbation into both the test inputs and the 5-fold
calibration set ("noise-aware fold injection"), so the calibrated intervals
remain valid under covariate shift.

| Scenario | Knob setting | Paper result (GPR-S1, width / depth) |
|---|---|---|
| Full fidelity (baseline) | `downsample_fraction=1.0, noise_test=False` | R² 0.998 / 0.999, PICP 0.98 / 0.95 |
| +5% sensor noise | `noise_test=True, noise_level=0.05` | R² 0.993 / 0.994, PICP 0.94 / 0.94 |
| +10% sensor noise | `noise_test=True, noise_level=0.10` | R² 0.979 / 0.978, PICP 0.96 / 0.96 |
| 50% training data | `downsample_fraction=0.50` | R² 0.994 / 0.958, PICP 0.96 / 0.88 |
| 25% training data | `downsample_fraction=0.25` | R² 0.965 / 0.884, PICP 0.976 / 0.98 |

To reproduce the GPR-S3 row of the table, switch `selection_mode = 's3'` at the
bottom of `src/gpr.py` and re-run each scenario. The operational implication
reported in the paper — that the quality gate self-disables under both adversity
conditions by issuing universal REJECT — emerges automatically once MPIW grows
beyond the tolerance window set in `src/quality_gating.py`.

## 10. GWO budget

For fast iteration during review, `src/gpr.py` ships with `GWO_MAX_ITER = 5`.
The paper's reported numbers use the full budget of 12 wolves × 25 iterations
(justified by `src/gwo_sensitivity.py`); set `GWO_MAX_ITER = 25` at the top of
`src/gpr.py` to reproduce the headline R² / PICP exactly.

## 11. Outputs

Console metrics, pooled residual statistics, CSV tables, and diagnostic figures —
all generated files are written to `results/` (see [`results/README.md`](../results/README.md)).

## Notes

- Runs are seeded (`set_global_determinism`), so results are reproducible up to
  floating-point and hardware nondeterminism (GPU/CPU differences may cause small
  variations in the torch-based GPR pipeline).
- All evaluations use an 80/20 train/test hold-out split; hyperparameter search and
  calibration use 5-fold CV **on the training partition only**, so the test split never
  leaks into model selection or calibration.
