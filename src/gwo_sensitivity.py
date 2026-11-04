"""
GWO population-size and iteration-budget sensitivity analysis for the GPR-S1
surrogate.

Justifies the Grey Wolf Optimizer budget used throughout the paper
(12 wolves, 25 iterations): best-fitness (mean 5-fold CV negative log
likelihood of the GPR-S1 width model) convergence is compared across
population sizes {5, 8, 12, 15, 20} over 50 iterations. The initial wolf
positions are seeded identically across populations (fixed seed), so the
comparison isolates the effect of population size. The convergence curves
show diminishing returns beyond the chosen budget, marked on the figure.

Usage:
    python src/gwo_sensitivity.py

Outputs (written to results/):
    gwo_sensitivity.csv             NLL at iterations 10 / 25 / 50 + runtime per population
    GWO_Sensitivity_Analysis.png    convergence curves with chosen budget marked
"""

import time
import numpy as np
import pandas as pd
import torch
import gpytorch
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import KFold, train_test_split

REPO_ROOT   = Path(__file__).resolve().parents[1]
DATA_PATH   = REPO_ROOT / 'data' / 'data.csv'
RESULTS_DIR = REPO_ROOT / 'results'

NOISE_LOW  = 1e-4     # likelihood noise lower bound (10 ** -4)
NOISE_HIGH = 0.794    # likelihood noise upper bound (10 ** -0.1)

POPULATIONS = [5, 8, 12, 15, 20]
MAX_ITER    = 50
CHOSEN_ITER = 25      # budget adopted in the paper
SEED        = 405


class SingleTaskGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module  = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(ard_num_dims=train_x.size(-1))
        )

    def forward(self, x):
        return gpytorch.distributions.MultivariateNormal(
            self.mean_module(x), self.covar_module(x)
        )


def fitness_function_sensitivity(params, X_data, y_data_1d, seed=SEED):
    """Mean 5-fold CV NLL of a GPR initialized at the given GWO position.

    params = [ls_1, ls_2, ls_3, log10(noise)]; lengthscale dimensions match
    the S1 feature set (speed, defocus, power).
    """
    kf = KFold(n_splits=5, shuffle=True, random_state=seed)
    all_fold_nll = []

    ls_params = np.array(params[:3], dtype=np.float32).reshape(1, -1)
    actual_noise = float(np.clip(10 ** params[3], NOISE_LOW, NOISE_HIGH))

    for train_idx, _ in kf.split(X_data):
        X_tr = torch.tensor(X_data[train_idx], dtype=torch.float32)
        y_tr = torch.tensor(y_data_1d[train_idx], dtype=torch.float32)

        m_x, s_x = X_tr.mean(0), X_tr.std(0) + 1e-6
        m_y, s_y = y_tr.mean(), y_tr.std() + 1e-6
        X_tr_norm = (X_tr - m_x) / s_x
        y_tr_norm = (y_tr - m_y) / s_y

        likelihood = gpytorch.likelihoods.GaussianLikelihood(
            noise_constraint=gpytorch.constraints.GreaterThan(NOISE_LOW))
        model = SingleTaskGPModel(X_tr_norm, y_tr_norm, likelihood)
        model.initialize(**{
            'covar_module.base_kernel.lengthscale': torch.as_tensor(ls_params),
            'likelihood.noise': torch.tensor(actual_noise)
        })

        model.train(); likelihood.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

        try:
            for _ in range(70):
                optimizer.zero_grad()
                loss = -mll(model(X_tr_norm), y_tr_norm)
                loss.backward()
                optimizer.step()
            all_fold_nll.append(loss.item())
        except Exception:
            all_fold_nll.append(1e6)

    return np.mean(all_fold_nll)


def main():
    # ---- Data: S1 raw process parameters, width target ---------------------
    df_main = pd.read_csv(DATA_PATH)
    X_f = df_main.iloc[:, :3].values
    y_f = df_main.iloc[:, -2:].values
    X_train_val, _, y_train_val, _ = train_test_split(
        X_f, y_f, test_size=0.20, random_state=SEED)
    y_train_1d = y_train_val[:, 0]   # Width

    # ---- GWO sensitivity loop across population sizes ----------------------
    convergence_history = {}
    summary_matrix = []

    print("Starting GWO Population & Iteration Sensitivity Analysis...")
    print(f"Feature set: S1 (raw process parameters) | Target: Width | "
          f"Seed: {SEED}\n")

    for pop in POPULATIONS:
        print(f"Testing Population = {pop} Wolves...")
        start_time = time.time()

        # GWO initialization
        dim = 4
        lb = np.array([.1, .1, .1, -4.0])
        ub = np.array([10, 10, 10, -0.1])

        # Fixed seed for the initial positions, so the test is fair across
        # population sizes
        np.random.seed(SEED)
        Positions = np.random.uniform(lb, ub, (pop, dim))
        alpha_pos = np.zeros(dim); alpha_score = float("inf")
        beta_pos  = np.zeros(dim); beta_score  = float("inf")
        delta_pos = np.zeros(dim); delta_score = float("inf")

        curve = []

        # GWO main loop
        for l in range(MAX_ITER):
            for i in range(pop):
                fitness = fitness_function_sensitivity(
                    Positions[i, :], X_train_val, y_train_1d)
                if fitness < alpha_score:
                    delta_score, delta_pos = beta_score, beta_pos.copy()
                    beta_score, beta_pos = alpha_score, alpha_pos.copy()
                    alpha_score, alpha_pos = fitness, Positions[i].copy()
                elif fitness < beta_score:
                    delta_score, delta_pos = beta_score, beta_pos.copy()
                    beta_score, beta_pos = fitness, Positions[i].copy()
                elif fitness < delta_score:
                    delta_score, delta_pos = fitness, Positions[i].copy()

            a = 2 - l * (2 / MAX_ITER)
            for i in range(pop):
                r1, r2 = np.random.rand(dim), np.random.rand(dim)
                A1 = 2 * a * r1 - a; C1 = 2 * r2
                D_alpha = abs(C1 * alpha_pos - Positions[i])
                X1 = alpha_pos - A1 * D_alpha

                r1, r2 = np.random.rand(dim), np.random.rand(dim)
                A2 = 2 * a * r1 - a; C2 = 2 * r2
                D_beta = abs(C2 * beta_pos - Positions[i])
                X2 = beta_pos - A2 * D_beta

                r1, r2 = np.random.rand(dim), np.random.rand(dim)
                A3 = 2 * a * r1 - a; C3 = 2 * r2
                D_delta = abs(C3 * delta_pos - Positions[i])
                X3 = delta_pos - A3 * D_delta

                Positions[i] = np.clip((X1 + X2 + X3) / 3, lb, ub)

            curve.append(alpha_score)

        elapsed_time = time.time() - start_time

        convergence_history[f'{pop} Wolves'] = curve
        summary_matrix.append({
            'Population': f'{pop} Wolves',
            'NLL @ Iter 10': curve[9],
            'NLL @ Iter 25': curve[24],
            'NLL @ Iter 50': curve[49],
            'Time (sec)': elapsed_time
        })
        print(f"  Completed. NLL @ 25: {curve[24]:.4f} | "
              f"Time: {elapsed_time:.2f}s")

    # ---- Summary matrix -----------------------------------------------------
    df_matrix = pd.DataFrame(summary_matrix)
    print("\n" + "=" * 60)
    print("GWO SENSITIVITY ANALYSIS MATRIX")
    print("=" * 60)
    print(df_matrix.to_string(index=False))
    print("=" * 60)

    RESULTS_DIR.mkdir(exist_ok=True)
    df_matrix.to_csv(RESULTS_DIR / 'gwo_sensitivity.csv', index=False)
    print("Saved: results/gwo_sensitivity.csv")

    # ---- Convergence plot ---------------------------------------------------
    plt.figure(figsize=(10, 6))
    for label, curve in convergence_history.items():
        plt.plot(range(1, MAX_ITER + 1), curve,
                 marker='o', markersize=4, label=label)

    plt.axvline(x=CHOSEN_ITER, color='red', linestyle='--', alpha=0.7,
                label=f'Chosen Iterations ({CHOSEN_ITER})')

    plt.title('GWO Population Size & Iteration Sensitivity Analysis')
    plt.xlabel('Iterations')
    plt.ylabel('Best NLL (Fitness)')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'GWO_Sensitivity_Analysis.png', dpi=300)
    plt.show()
    print("Saved: results/GWO_Sensitivity_Analysis.png")


if __name__ == '__main__':
    main()
