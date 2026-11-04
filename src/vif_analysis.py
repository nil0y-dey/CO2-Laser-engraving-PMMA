"""
Multicollinearity diagnostics for the candidate input feature sets.

Reproduces the input-selection diagnostics reported in the paper: per-feature
Variance Inflation Factors (VIF) and the condition number of the feature matrix
for every candidate feature set S1-S5, plus feature-feature and
feature-output correlation heatmaps.

Definitions (paper, Input Selection):
    VIF_i = 1 / (1 - R_i^2)
        where R_i^2 is obtained by regressing feature x_i on the remaining
        features of the set. VIF < 5 negligible, 5-10 moderate, > 10 severe.
    kappa(X) = sqrt(lambda_max / lambda_min) of X^T X
        computed as the 2-norm condition number of X (np.linalg.cond).
        kappa < 10 negligible, 10-30 moderate, > 30 potentially severe.

Candidate feature sets (column indices refer to data/data.csv, see
data/README.md; VIF is invariant to column order within a set):
    S1: Power, Speed, Defocus length          (raw process parameters)
    S2: Spot Diameter, Linear ED, Area ED     (physics-derived composites)
    S3: Speed, Defocus length, Area ED        (hybrid)
    S4: Linear ED, Area ED, Defocus length    (hybrid, no spot diameter)
    S5: all six features                      (excluded from modelling: severe
                                               collinearity, ill-conditioned)

Usage:
    python src/vif_analysis.py

Outputs (written to results/):
    vif_table.csv               VIF + condition number per feature set
    correlation_analysis.png    feature-feature and feature-output heatmaps
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from statsmodels.stats.outliers_influence import variance_inflation_factor

REPO_ROOT   = Path(__file__).resolve().parents[1]
DATA_PATH   = REPO_ROOT / 'data' / 'data.csv'
RESULTS_DIR = REPO_ROOT / 'results'

# Candidate feature sets (paper, Input Selection). Indices into data.csv.
FEATURE_SETS = {
    'S1': {'cols': [2, 0, 1],            'label': 'Power, Speed, Defocus length'},
    'S2': {'cols': [3, 4, 5],            'label': 'Spot Diameter, Linear ED, Area ED'},
    'S3': {'cols': [0, 1, 5],            'label': 'Speed, Defocus length, Area ED'},
    'S4': {'cols': [4, 5, 1],            'label': 'Linear ED, Area ED, Defocus length'},
    'S5': {'cols': [0, 1, 2, 3, 4, 5],   'label': 'All six features'},
}


def vif_severity(v):
    if v < 5:
        return 'OK (<5)'
    if v < 10:
        return 'Moderate (5-10)'
    return 'Severe (>10)'


def kappa_severity(k):
    if k < 10:
        return 'Negligible (<10)'
    if k <= 30:
        return 'Moderate (10-30)'
    return 'Potentially severe (>30)'


def compute_set_diagnostics(X):
    """Per-feature VIF and condition number for a feature matrix X.

    Parameters
    ----------
    X : pd.DataFrame
        Feature subset (columns in their original units, no intercept).

    Returns
    -------
    vif_values : list of float
    cond_number : float
    """
    values = X.values.astype(float)
    vif_values = [variance_inflation_factor(values, i)
                  for i in range(X.shape[1])]
    cond_number = float(np.linalg.cond(values))
    return vif_values, cond_number


def main():
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded dataset: {df.shape[0]} runs x {df.shape[1]} columns")
    print(f"Columns: {df.columns.tolist()}\n")

    y = df.iloc[:, -2:]

    # ---- VIF + condition number for every candidate feature set -----------
    rows = []
    print("=" * 60)
    print("MULTICOLLINEARITY DIAGNOSTICS PER FEATURE SET (VIF + kappa)")
    print("=" * 60)
    for set_name, spec in FEATURE_SETS.items():
        X = df.iloc[:, spec['cols']]
        vif_values, cond_number = compute_set_diagnostics(X)
        print(f"\n[{set_name}] {spec['label']}")
        print(f"  Condition number kappa = {cond_number:.2f}  "
              f"({kappa_severity(cond_number)})")
        for feat, v in zip(X.columns, vif_values):
            print(f"  VIF  {feat:<20s} {v:8.2f}  ({vif_severity(v)})")
            rows.append({'Set': set_name,
                         'Feature': feat,
                         'VIF': round(v, 2),
                         'Condition Number': round(cond_number, 2)})
    print("\n" + "=" * 60)
    print("S5 is excluded from model training: severe multicollinearity "
          "(VIF up to ~39) and an ill-conditioned design matrix.")
    print("=" * 60)

    RESULTS_DIR.mkdir(exist_ok=True)
    vif_table = pd.DataFrame(rows)
    vif_table.to_csv(RESULTS_DIR / 'vif_table.csv', index=False)
    print(f"\nSaved: results/vif_table.csv")

    # ---- Correlation heatmaps (all six inputs vs both outputs) ------------
    X_all = df.iloc[:, :6]
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    sns.heatmap(X_all.corr(), annot=True, cmap='coolwarm', fmt='.2f',
                ax=axes[0], linewidths=0.5, vmin=-1, vmax=1)
    axes[0].set_title('Feature-Feature Correlation Matrix',
                      fontsize=13, fontweight='bold')

    corr_fo = pd.concat([X_all, y], axis=1).corr().iloc[:6, 6:]
    sns.heatmap(corr_fo, annot=True, cmap='coolwarm', fmt='.2f',
                ax=axes[1], linewidths=0.5, vmin=-1, vmax=1)
    axes[1].set_title('Feature-Output Correlation Matrix',
                      fontsize=13, fontweight='bold')

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'correlation_analysis.png', dpi=300)
    plt.show()
    print("Saved: results/correlation_analysis.png")


if __name__ == '__main__':
    main()
