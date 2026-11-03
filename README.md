# Conformal-Calibrated Gaussian Process Surrogate for CO2 Laser Engraving of PMMA

[![Paper DOI](https://img.shields.io/badge/DOI-10.1016%2Fj.eswa.2026.134356-blue)](https://doi.org/10.1016/j.eswa.2026.134356)
[![Journal](https://img.shields.io/badge/Expert_Systems_with_Applications-Vol._333-blueviolet)](https://www.sciencedirect.com/science/article/abs/pii/S0957417426032537)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)

Research code for the paper:

> **Robust conformal Gaussian process regression for CO2 laser engraving: Distribution-free uncertainty quantification and automated quality gating**
> Niloy Dey, Mohammad Muhshin Aziz Khan, Ahmed Sayem
> *Expert Systems with Applications*, Vol. 333, Part G, 2027. [DOI: 10.1016/j.eswa.2026.134356](https://doi.org/10.1016/j.eswa.2026.134356)

---

## Overview

Selecting CO2 laser process parameters for a target groove geometry normally requires physical trial-and-error iterations that consume material and machine time. This work replaces those trials with a **calibrated surrogate model**: it predicts the engraved groove geometry (depth and width) in PMMA together with **prediction intervals**, and feeds both into an **automated quality gate** that *accepts*, *flags*, or *rejects* candidate parameter sets — based on the predicted geometry **and** the reliability of that prediction.

Five machine-learning architectures were compared on a 125-run full-factorial design space:

| Model | Role in this repository |
|---|---|
| **Gaussian Process Regression (GPR)** | Final surrogate — GWO-tuned, conformal-calibrated, with SHAP/PDP/ICE interpretability analysis (`src/gpr.py`) |
| Support Vector Regression (SVR) | GWO-tuned baseline (`src/svr.py`) |
| Multilayer Perceptron (MLP) | GWO-tuned baseline (`src/mlp.py`) |
| Random Forest (RF) | GWO-tuned baseline (`src/random_forest.py`) |
| Polynomial Ridge Regression (PRR) | GWO-tuned baseline (`src/polynomial_ridge.py`) |

All architectures share the same evaluation protocol (identical seeds, splits and 5-fold CV), so the comparison reflects model capacity rather than tuning disparity. Hyperparameters of every architecture are optimized with the **Grey Wolf Optimizer (GWO)**.

## Key results

GPR on raw process parameters, evaluated on held-out test data (80/20 split, multi-seed experiment):

| Target | R² (held-out) | 95% prediction-interval coverage (PICP) | Nominal level |
|---|---|---|---|
| Groove **depth** | **0.999** | ≥ 0.94 | 0.95 |
| Groove **width** | **0.998** | ≥ 0.94 | 0.95 |

Conformal calibration was verified with **Prediction Interval Coverage Probability (PICP)** and **Negative Log Predictive Density (NLPD)** at the 95% nominal level, confirming that the reported intervals are reliable rather than merely narrow. The full five-model comparison, the multicollinearity diagnostics, and the reproducible negative result on pure composite feature sets (traced to an exact algebraic rank deficiency) are reported in the paper.

## Pipeline

Each script implements the same end-to-end protocol:

```
125-run full-factorial dataset
        │
        ├── feature-set selection (modes s1–s4, see below)
        ├── 80/20 train/test hold-out split
        ├── 5-fold cross-validation on the training partition
        ├── Grey Wolf Optimizer hyperparameter search (12 wolves, 25 iterations)
        ├── multi-seed experiment (seeds: 405, 166, 327, 347, 589)
        ├── held-out evaluation (R², RMSE, MAE, MAPE)
        └── (GPR only) conformal calibration → PICP / NLPD → SHAP + PDP / ICE interpretability
```

## Repository structure

```
CO2-Laser-engraving-PMMA/
├── README.md                 # this file
├── LICENSE                   # MIT
├── CITATION.cff              # citation metadata (GitHub renders this automatically)
├── requirements.txt          # pip dependencies
├── environment.yml           # conda environment (recommended for PyTorch/GPyTorch)
├── data/                     # place the experimental datasets here (see data/README.md)
│   └── data_template.csv     # column schema of the dataset
├── src/
│   ├── gpr.py                # GPR surrogate: GWO tuning + conformal calibration + SHAP/PDP/ICE
│   ├── svr.py                # SVR baseline (GWO-tuned)
│   ├── mlp.py                # MLP baseline (GWO-tuned)
│   ├── random_forest.py      # Random Forest baseline (GWO-tuned)
│   └── polynomial_ridge.py   # Polynomial Ridge baseline (GWO-tuned)
├── results/                  # experiment outputs (see results/README.md)
└── docs/
    └── reproduction.md       # step-by-step reproduction guide
```

## Getting started

### Installation

Option A — conda (recommended, handles PyTorch cleanly):

```bash
conda env create -f environment.yml
conda activate co2-laser-pmma
```

Option B — pip:

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

> PyTorch users with a CUDA GPU may want to install a CUDA-enabled build from
> [pytorch.org](https://pytorch.org/get-started/locally/) first; the code also runs on CPU.

### Data preparation

The scripts read the experimental dataset from `data/`. The dataset is a 125-run
full-factorial CO2 laser engraving experiment with the column schema documented in
[`data/README.md`](data/README.md) (template: [`data/data_template.csv`](data/data_template.csv)).

Place your dataset files as:

```
data/data.csv     # used by gpr.py, svr.py, mlp.py, random_forest.py
data/data2.csv    # variant dataset used by polynomial_ridge.py (same schema)
```

### Running the models

From the repository root:

```bash
python src/gpr.py                # GPR + conformal calibration + SHAP / PDP / ICE
python src/svr.py                # SVR baseline
python src/mlp.py                # MLP baseline
python src/random_forest.py      # Random Forest baseline
python src/polynomial_ridge.py   # Polynomial Ridge baseline (reads data/data2.csv)
```

Each script runs the full multi-seed GWO experiment and prints per-seed metrics,
pooled residual statistics, and a mean ± std summary over all seeds. GPR additionally
performs conformal calibration (PICP / NLPD) and the SHAP + PDP/ICE interpretability
analysis. Generated CSV outputs are written to the working directory (see
[`results/README.md`](results/README.md)).

### Feature selection modes

The active feature set is controlled by the `selection_mode` variable at the bottom of
each script:

| Mode | Features | Description |
|---|---|---|
| `s1` | Speed, Defocus length, Power | Raw process parameters (headline configuration of the paper) |
| `s2` | Spot Diameter, Linear ED, Area ED | Composite energy-density features |
| `s3` | Speed, Defocus length, Area ED | Hybrid raw + composite |
| `s4` | Defocus length, Linear ED, Area ED | Alternative hybrid subset |

Additional experiment knobs available at the bottom of each script: the seed list
(`seeds`), and in `gpr.py` the downsampling and noise-robustness options
(`downsample_fraction`, `noise_test`, `noise_level`) used for the data-efficiency and
noise-sensitivity studies in the paper.

## Citation

If you use this code, please cite the paper:

```bibtex
@article{Dey2027ConformalGPR,
  title   = {Robust conformal {G}aussian process regression for {CO2} laser engraving:
             Distribution-free uncertainty quantification and automated quality gating},
  author  = {Dey, Niloy and Khan, Mohammad Muhshin Aziz and Sayem, Ahmed},
  journal = {Expert Systems with Applications},
  volume  = {333},
  year    = {2027},
  doi     = {10.1016/j.eswa.2026.134356}
}
```

GitHub also renders the machine-readable citation in [`CITATION.cff`](CITATION.cff)
("Cite this repository" button).

## License

This project is released under the [MIT License](LICENSE).

## Acknowledgements

This work was carried out at the Department of Industrial and Production Engineering (IPE),
Shahjalal University of Science and Technology.
