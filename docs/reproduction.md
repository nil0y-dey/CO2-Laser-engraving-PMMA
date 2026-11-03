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

## 5. Reproducing individual studies

The experiment knobs are at the **bottom of each script**:

| Paper study | Script | Knob |
|---|---|---|
| Feature-set comparison (raw vs composite vs hybrid) | any | `selection_mode = 's1' \| 's2' \| 's3' \| 's4'` |
| Seed robustness | any | `seeds = [405, 166, 327, 347, 589]` |
| Data-efficiency study | `src/gpr.py` | `downsample_fraction` (e.g. `0.6`, `0.8`, `1.0`) |
| Noise-robustness study | `src/gpr.py` | `noise_test=True, noise_level=...` |
| Final-paper SHAP figures | `src/gpr.py` | `n_samples=500` (comment at the bottom of the script) |

## 6. Outputs

Console metrics, pooled residual statistics, `gpr_results_*.csv` files, and interactive
diagnostic figures — see [`results/README.md`](../results/README.md).

## Notes

- Runs are seeded (`set_global_determinism`), so results are reproducible up to
  floating-point and hardware nondeterminism (GPU/CPU differences may cause small
  variations in the torch-based GPR pipeline).
- All evaluations use an 80/20 train/test hold-out split; hyperparameter search and
  calibration use 5-fold CV **on the training partition only**, so the test split never
  leaks into model selection or calibration.
