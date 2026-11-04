# Dataset

The experiments underlying the paper are a **125-run full-factorial CO2 laser engraving
study on PMMA**: each run records the laser process parameters and the resulting engraved
groove geometry (width and depth).

## Files

| File | Used by | Description |
|---|---|---|
| `data.csv` | all five models (`src/gpr.py`, `src/svr.py`, `src/mlp.py`, `src/random_forest.py`, `src/polynomial_ridge.py`) | 125-run full-factorial dataset |
| `data_template.csv` | — | Column schema template |

## Column schema

**Column order matters** — the scripts select features by column position
(e.g. `df.iloc[:, :3]` for mode `s1`), so the header must match this layout exactly:

| # | Column | Role | Units (as used in the study) |
|---|--------|------|------------------------------|
| 0 | `Speed` | Input (raw process parameter) | mm/s |
| 1 | `Defocus length` | Input (raw process parameter) | mm |
| 2 | `Power` | Input (raw process parameter) | W |
| 3 | `Spot Diameter` | Input (derived) | mm |
| 4 | `Linear ED` | Input (composite energy density) | J/mm |
| 5 | `Area ED` | Input (composite energy density) | J/mm² |
| 6 | `Width` | **Target** | μm |
| 7 | `Depth` | **Target** | μm |

The input feature sets compared in the paper (selection modes `s1`–`s4`) are subsets of
these six input columns; see the "Feature selection modes" section of the main
[README](../README.md).

## Availability

The experimental dataset is **private** and is not distributed with this repository,
because it originates from in-house laser machining trials. The dataset is available
from the corresponding author on reasonable request. This repository provides the
complete column schema (above) together with `data_template.csv`, so that any
equivalent dataset can be plugged into the pipeline.
