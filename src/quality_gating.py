"""
Automated quality-gating validation on held-out experimental data (paper,
Section: Validation of the Decision Policy).

Trains the calibrated GPR-S1 surrogate (raw process parameters: speed,
defocus, power) on a single-seed (589) 80/20 split, derives the conformal
scale factor via 5-fold cross-conformal calibration, then evaluates the
three-tier decision policy (ACCEPT / FLAG / REJECT) against three target
conditions:

    Target 1 - Dense Region : target = held-out point whose prediction has the
        most neighbouring predictions within tolerance (paper, Eq. density).
        A realistic, densely-supported, physically achievable geometry.
    Target 2 - Sparse Region: target = held-out point with the fewest
        neighbours within tolerance (mirror of Target 1). Brackets the
        in-domain gating behaviour from densely to sparsely supported
        conditions.
    Target 3 - Out-of-Domain : the surrogate is queried at parameter
        combinations 50-150% beyond the validated bounds (single factor at a
        time). No physical ground truth exists; only verdict distribution and
        calibrated-uncertainty inflation are reported.

Tolerances are set equal to the calibrated MPIW of GPR-S1 per output.
Width and depth are physically coupled (defocus enlarges the footprint while
suppressing depth), so targets are drawn from real, measured (w, d) pairs --
never from independently-chosen marginal percentiles, which can specify
physically unachievable joint targets.

Note: the surrogate-assisted NSGA-II search policy is deliberately NOT
exercised here; this script isolates the gating logic using parameter sets
drawn from the experimental design (paper, Section: Validation of the
Decision Policy).

Usage:
    python src/quality_gating.py

Outputs (written to results/):
    gating_target1_dense.csv
    gating_target2_sparse.csv
    gating_target3_outofdomain.csv
    gating_three_target_summary.csv
"""

import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import gpytorch
from sklearn.model_selection import KFold, train_test_split

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
REPO_ROOT   = Path(__file__).resolve().parents[1]
DATA_PATH   = REPO_ROOT / 'data' / 'data.csv'
RESULTS_DIR = REPO_ROOT / 'results'

VALIDATION_SEED = 589        # single-seed validation split (paper: seed 589)
NUM_WOLVES      = 12         # GWO population size (paper configuration)
GWO_MAX_ITER    = 5          # GWO budget for this validation run. The
                             # multi-seed experiments (src/gpr.py) use 25
                             # iterations as reported in the paper; raise this
                             # to 25 to match the headline budget.
TOLERANCE_MULTIPLIER = 2.0   # delta = multiplier * (MPIW / 2) = MPIW


def set_global_determinism(seed=42):
    # Kept for parity with the experiment notebook; the pipeline below seeds
    # numpy/torch directly per task.
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


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


def run_single_task_pipeline(X_train_val, X_test, y_train_val_1d, y_test_1d,
                             seed, task_name,
                             NOISE_LOG_LB, NOISE_LOG_UB,
                             noise_test=False, noise_level=0.0):
    """GWO-tuned single-output GPR with cross-conformal calibration.

    Returns held-out predictions, calibrated sigmas, the conformal scale
    factor, MPIW, and the fitted model with its normalization statistics.
    """
    NOISE_LOW = 10 ** NOISE_LOG_LB
    NOISE_HIGH = 10 ** NOISE_LOG_UB

    def fitness_function(params, X_data, y_data_1d):
        kf = KFold(n_splits=5, shuffle=True, random_state=seed)
        all_fold_nll = []
        ls_params = np.array(params[:3], dtype=np.float32).reshape(1, -1)
        actual_noise = float(np.clip(10 ** params[3], NOISE_LOW, NOISE_HIGH))

        for train_idx, val_idx in kf.split(X_data):
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
                'likelihood.noise': torch.tensor(actual_noise)})
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

    # ---- Grey Wolf Optimizer (hyperparameter search) ----------------------
    dim = 4
    lb = np.array([.1, .1, .1, NOISE_LOG_LB])
    ub = np.array([10, 10, 10, NOISE_LOG_UB])
    Positions = np.random.uniform(lb, ub, (NUM_WOLVES, dim))
    alpha_pos = np.zeros(dim); alpha_score = float("inf")
    beta_pos  = np.zeros(dim); beta_score  = float("inf")
    delta_pos = np.zeros(dim); delta_score = float("inf")

    for l in range(GWO_MAX_ITER):
        for i in range(NUM_WOLVES):
            fitness = fitness_function(Positions[i, :], X_train_val,
                                       y_train_val_1d)
            if fitness < alpha_score:
                delta_score, delta_pos = beta_score, beta_pos.copy()
                beta_score, beta_pos = alpha_score, alpha_pos.copy()
                alpha_score, alpha_pos = fitness, Positions[i].copy()
            elif fitness < beta_score:
                delta_score, delta_pos = beta_score, beta_pos.copy()
                beta_score, beta_pos = fitness, Positions[i].copy()
            elif fitness < delta_score:
                delta_score, delta_pos = fitness, Positions[i].copy()
        a = 2 - l * (2 / GWO_MAX_ITER)
        for i in range(NUM_WOLVES):
            r1, r2 = np.random.rand(dim), np.random.rand(dim)
            A1 = 2*a*r1-a; C1 = 2*r2
            D_alpha = abs(C1*alpha_pos - Positions[i]); X1 = alpha_pos - A1*D_alpha
            r1, r2 = np.random.rand(dim), np.random.rand(dim)
            A2 = 2*a*r1-a; C2 = 2*r2
            D_beta = abs(C2*beta_pos - Positions[i]); X2 = beta_pos - A2*D_beta
            r1, r2 = np.random.rand(dim), np.random.rand(dim)
            A3 = 2*a*r1-a; C3 = 2*r2
            D_delta = abs(C3*delta_pos - Positions[i]); X3 = delta_pos - A3*D_delta
            Positions[i] = np.clip((X1 + X2 + X3) / 3, lb, ub)

    # ---- Final model fit on the full training partition -------------------
    X_train_t = torch.tensor(X_train_val, dtype=torch.float32)
    y_train_t = torch.tensor(y_train_val_1d, dtype=torch.float32)
    m_x, s_x = X_train_t.mean(0), X_train_t.std(0) + 1e-7
    m_y, s_y = y_train_t.mean(), y_train_t.std() + 1e-7
    X_train_norm = (X_train_t - m_x) / s_x
    y_train_norm = (y_train_t - m_y) / s_y

    likelihood = gpytorch.likelihoods.GaussianLikelihood(
        noise_constraint=gpytorch.constraints.GreaterThan(NOISE_LOW))
    model = SingleTaskGPModel(X_train_norm, y_train_norm, likelihood)
    best_noise = float(np.clip(10 ** alpha_pos[3], NOISE_LOW, NOISE_HIGH))
    model.initialize(**{'covar_module.base_kernel.lengthscale':
                         torch.tensor(np.array([alpha_pos[:3]], dtype=np.float32)),
                        'likelihood.noise': torch.tensor(best_noise)})
    model.train(); likelihood.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    for _ in range(180):
        optimizer.zero_grad()
        loss = -mll(model(X_train_norm), y_train_norm)
        loss.backward()
        optimizer.step()

    # ---- Cross-conformal calibration --------------------------------------
    kf_cal = KFold(n_splits=5, shuffle=True, random_state=seed)
    cv_val_resid_norm, cv_val_sigma_norm = [], []
    best_ls = torch.tensor(np.array([alpha_pos[:3]], dtype=np.float32))
    best_noise = float(np.clip(10 ** alpha_pos[3], NOISE_LOW, NOISE_HIGH))

    for train_idx, val_idx in kf_cal.split(X_train_val):
        X_tr  = torch.tensor(X_train_val[train_idx], dtype=torch.float32)
        y_tr  = torch.tensor(y_train_val_1d[train_idx], dtype=torch.float32)
        X_val = torch.tensor(X_train_val[val_idx], dtype=torch.float32)
        y_val = torch.tensor(y_train_val_1d[val_idx], dtype=torch.float32)
        if noise_test:
            stds_val = np.std(X_val.numpy(), axis=0)
            noise_val = np.random.normal(0, noise_level * stds_val, X_val.shape)
            X_val = X_val + torch.tensor(noise_val, dtype=torch.float32)
        m_xc, s_xc = X_tr.mean(0), X_tr.std(0) + 1e-7
        m_yc, s_yc = y_tr.mean(), y_tr.std() + 1e-7
        X_tr_c  = (X_tr  - m_xc) / s_xc
        y_tr_c  = (y_tr  - m_yc) / s_yc
        X_val_c = (X_val - m_xc) / s_xc
        y_val_c = (y_val - m_yc) / s_yc
        lik_c = gpytorch.likelihoods.GaussianLikelihood(
            noise_constraint=gpytorch.constraints.GreaterThan(NOISE_LOW))
        mod_c = SingleTaskGPModel(X_tr_c, y_tr_c, lik_c)
        mod_c.initialize(**{'covar_module.base_kernel.lengthscale': best_ls,
                            'likelihood.noise': torch.tensor(best_noise)})
        mod_c.train(); lik_c.train()
        opt_c = torch.optim.Adam(mod_c.parameters(), lr=0.1)
        mll_c = gpytorch.mlls.ExactMarginalLogLikelihood(lik_c, mod_c)
        for _ in range(130):
            opt_c.zero_grad()
            loss_c = -mll_c(mod_c(X_tr_c), y_tr_c)
            loss_c.backward()
            opt_c.step()
        mod_c.eval(); lik_c.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            val_pred = lik_c(mod_c(X_val_c))
            val_mu = val_pred.mean.numpy()
            val_sigma = torch.sqrt(val_pred.variance).numpy()
        val_resid = y_val_c.numpy() - val_mu
        cv_val_resid_norm.extend(val_resid.tolist())
        cv_val_sigma_norm.extend(val_sigma.tolist())

    cv_resid = np.array(cv_val_resid_norm)
    cv_sigma = np.array(cv_val_sigma_norm)
    cv_std_resid = np.abs(cv_resid / (cv_sigma + 1e-8))
    conformal_quantile = float(np.quantile(cv_std_resid, 0.95))
    scale_factor = (conformal_quantile / 1.96) * 1.05

    # ---- Held-out evaluation with calibrated sigmas ------------------------
    model.eval(); likelihood.eval()
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    X_test_norm = (X_test_t - m_x) / s_x
    y_test_t = torch.tensor(y_test_1d, dtype=torch.float32)
    y_test_norm = (y_test_t - m_y) / s_y

    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        observed_pred = likelihood(model(X_test_norm))
        y_pred_norm = observed_pred.mean
        std_pred_norm = torch.sqrt(observed_pred.variance)

    std_phys_raw = (std_pred_norm * s_y).numpy().copy()
    y_pred = (y_pred_norm * s_y + m_y).numpy()
    sigma_phys = std_phys_raw * scale_factor
    mpiw = float(np.mean(2 * 1.96 * sigma_phys))

    return {
        '_true_phys': y_test_1d, '_pred_phys': y_pred, '_sigma_phys': sigma_phys,
        '_scale_factor': scale_factor, 'MPIW': mpiw, '_model': model,
        '_likelihood': likelihood,
        '_m_x': m_x, '_s_x': s_x, '_m_y': m_y, '_s_y': s_y
    }


def run_gwo_gpr_separate_pipeline(X_raw, y_raw, seed):
    """Train one conformal-calibrated GPR per output column.

    X_raw columns, in true file order, are [speed, defocus, power].
    y_raw columns, in true file order, are [width, depth].
    """
    np.random.seed(seed); torch.manual_seed(seed)
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X_raw, y_raw, test_size=0.20, random_state=seed)
    NOISE_BOUNDS = {'Width': (-4, -0.1), 'Depth': (-4, -0.1)}
    out_names = ['Width', 'Depth']   # matches y_raw column order [width, depth]
    results = {}
    for i, name in enumerate(out_names):
        np.random.seed(seed); torch.manual_seed(seed)
        lb, ub = NOISE_BOUNDS[name]
        res = run_single_task_pipeline(
            X_train_val, X_test, y_train_val[:, i], y_test[:, i],
            seed, name, lb, ub)
        results[name] = res
    return X_test, y_test, results


# ---------------------------------------------------------------------------
# Quality-gating evaluation
# ---------------------------------------------------------------------------

def evaluate_gate(target_w, target_d, delta_w, delta_d, pred_w, pred_d,
                  ci_low_w, ci_high_w, ci_low_d, ci_high_d,
                  X_eval, y_eval=None):
    """Three-tier decision policy over a set of evaluation points.

    Rules (paper, Decision Policy and Gating Rules):
        ACCEPT : calibrated 95% CI of BOTH outputs inside the tolerance window
        FLAG   : predicted means of BOTH outputs inside the window (CI too wide)
        REJECT : predicted mean of either output outside the window

    y_eval=None marks points with no physical ground truth (out-of-domain).
    Verdict accuracy compares (ACCEPT or FLAG) against the actual measurement
    falling inside the tolerance window.
    """
    rows = []
    counts = {"ACCEPT": 0, "FLAG": 0, "REJECT": 0}
    n_correct = 0
    has_ground_truth = y_eval is not None

    for i in range(len(X_eval)):
        v, dz, p = X_eval[i]

        if (ci_low_w[i] >= target_w - delta_w and ci_high_w[i] <= target_w + delta_w) and \
           (ci_low_d[i] >= target_d - delta_d and ci_high_d[i] <= target_d + delta_d):
            verdict = "ACCEPT"
        elif (abs(pred_w[i] - target_w) <= delta_w and abs(pred_d[i] - target_d) <= delta_d):
            verdict = "FLAG"
        else:
            verdict = "REJECT"
        counts[verdict] += 1

        row = [f"P={p:.0f}W,v={v:.0f}mm/s,dz={dz:.0f}mm",
               f"{pred_w[i]:.1f} [{ci_low_w[i]:.1f},{ci_high_w[i]:.1f}]",
               f"{pred_d[i]:.1f} [{ci_low_d[i]:.1f},{ci_high_d[i]:.1f}]"]

        if has_ground_truth:
            act_w, act_d = y_eval[i]
            actual_ok = (abs(act_w - target_w) <= delta_w) and \
                        (abs(act_d - target_d) <= delta_d)
            gate_pass = verdict in ("ACCEPT", "FLAG")
            correct = (gate_pass == actual_ok)
            n_correct += int(correct)
            row += [f"{act_w:.1f},{act_d:.1f}", verdict,
                    "Pass" if actual_ok else "Fail", "Yes" if correct else "No"]
        else:
            row += ["N/A (no ground truth)", verdict, "N/A", "N/A"]

        rows.append(row)

    cols = ['Params', 'Pred Width [95% CI]', 'Pred Depth [95% CI]',
            'Actual (W,D)', 'Verdict', 'Actual Outcome', 'Correct?']
    df = pd.DataFrame(rows, columns=cols)
    accuracy = (n_correct / len(X_eval)) if has_ground_truth else None
    return df, counts, accuracy


def compute_neighbor_counts(pred_w, pred_d, delta_w, delta_d):
    """Neighbourhood density N_i of every held-out prediction (paper, Eq. density).

    N_i counts other test predictions within tolerance of prediction i.
    """
    n = len(pred_w)
    neighbor_counts = np.zeros(n, dtype=int)
    for i in range(n):
        close_w = np.abs(pred_w - pred_w[i]) <= delta_w
        close_d = np.abs(pred_d - pred_d[i]) <= delta_d
        neighbor_counts[i] = np.sum(close_w & close_d) - 1
    return neighbor_counts


def predict_oob(gpr_result_dict, X_new):
    """Query a fitted, calibrated single-output GPR at new parameter points."""
    model = gpr_result_dict['_model']
    likelihood = gpr_result_dict['_likelihood']
    m_x, s_x = gpr_result_dict['_m_x'], gpr_result_dict['_s_x']
    m_y, s_y = gpr_result_dict['_m_y'], gpr_result_dict['_s_y']
    scale_factor = gpr_result_dict['_scale_factor']

    X_new_t = torch.tensor(X_new, dtype=torch.float32)
    X_new_norm = (X_new_t - m_x) / s_x

    model.eval(); likelihood.eval()
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        pred = likelihood(model(X_new_norm))
        mu_norm = pred.mean
        std_norm = torch.sqrt(pred.variance)

    mu_phys = (mu_norm * s_y + m_y).numpy()
    sigma_phys = (std_norm * s_y).numpy() * scale_factor
    return mu_phys, sigma_phys


def main():
    # ---- Load data and confirm column order before doing anything else -----
    df_main = pd.read_csv(DATA_PATH)
    print("Columns (in file order):", df_main.columns.tolist())

    # Expected column order:
    #   [speed, defocus length, power, spot diameter, LED, AED, width, depth]
    SPEED_COL, DEFOCUS_COL, POWER_COL = 0, 1, 2
    WIDTH_COL, DEPTH_COL = -2, -1

    X_f = df_main.iloc[:, [SPEED_COL, DEFOCUS_COL, POWER_COL]].values
    y_f = df_main.iloc[:, [WIDTH_COL, DEPTH_COL]].values

    print(f"\nRunning GPR-S1 pipeline (seed={VALIDATION_SEED})...")
    X_test, y_test, gpr_results = run_gwo_gpr_separate_pipeline(
        X_f, y_f, VALIDATION_SEED)
    print("GPR training and calibration complete.\n")

    print(f"Calibrated MPIW -- Width: {gpr_results['Width']['MPIW']:.2f} um | "
          f"Depth: {gpr_results['Depth']['MPIW']:.2f} um")
    print(f"Held-out test set size: {len(X_test)}")
    print(f"Speed range in test set:   [{X_test[:,0].min():.1f}, {X_test[:,0].max():.1f}] mm/s")
    print(f"Defocus range in test set: [{X_test[:,1].min():.1f}, {X_test[:,1].max():.1f}] mm")
    print(f"Power range in test set:   [{X_test[:,2].min():.1f}, {X_test[:,2].max():.1f}] W")
    # Sanity check: speed should fall in [10,50], defocus in [-4,12],
    # power in [48,80] per the paper's validated domain. If these ranges look
    # wrong, the column mapping above does not match the data file.

    RESULTS_DIR.mkdir(exist_ok=True)

    # ---- Calibrated predictions and tolerances -----------------------------
    pred_w  = gpr_results['Width']['_pred_phys']
    sigma_w = gpr_results['Width']['_sigma_phys']
    pred_d  = gpr_results['Depth']['_pred_phys']
    sigma_d = gpr_results['Depth']['_sigma_phys']

    # Tolerance = calibrated MPIW per output (multiplier 2.0 x half-width)
    delta_w = TOLERANCE_MULTIPLIER * (gpr_results['Width']['MPIW'] / 2)
    delta_d = TOLERANCE_MULTIPLIER * (gpr_results['Depth']['MPIW'] / 2)

    ci_low_w  = pred_w  - 1.96 * sigma_w
    ci_high_w = pred_w  + 1.96 * sigma_w
    ci_low_d  = pred_d  - 1.96 * sigma_d
    ci_high_d = pred_d  + 1.96 * sigma_d

    neighbor_counts = compute_neighbor_counts(pred_w, pred_d, delta_w, delta_d)

    # ---- TARGET 1: DENSE REGION --------------------------------------------
    best_idx = int(np.argmax(neighbor_counts))
    target_w_1 = float(pred_w[best_idx])
    target_d_1 = float(pred_d[best_idx])

    print("=" * 100)
    print("TARGET 1: DENSE REGION")
    print("=" * 100)
    print(f"Anchor: test point #{best_idx} "
          f"({neighbor_counts[best_idx]} neighbors within tolerance)")
    print(f"Target: Width={target_w_1:.1f}um, Depth={target_d_1:.1f}um")
    print(f"Tolerance: +/-{delta_w:.1f}um width ({100*delta_w/target_w_1:.1f}%), "
          f"+/-{delta_d:.1f}um depth ({100*delta_d/target_d_1:.1f}%)\n")

    df_t1, counts_t1, acc_t1 = evaluate_gate(
        target_w_1, target_d_1, delta_w, delta_d, pred_w, pred_d,
        ci_low_w, ci_high_w, ci_low_d, ci_high_d, X_test, y_test)

    print(df_t1.to_string(index=False))
    print(f"\nVerdict distribution: ACCEPT={counts_t1['ACCEPT']} | "
          f"FLAG={counts_t1['FLAG']} | REJECT={counts_t1['REJECT']}")
    print(f"Verdict accuracy: {acc_t1*len(X_test):.0f}/{len(X_test)} "
          f"({100*acc_t1:.1f}%)\n")
    df_t1.to_csv(RESULTS_DIR / 'gating_target1_dense.csv', index=False)

    # ---- TARGET 2: SPARSE REGION -------------------------------------------
    # Mirror image of Target 1: an achievable (actually measured) point in a
    # sparsely-supported region of the design space -- the test point with the
    # FEWEST neighbors within tolerance. (Independently-chosen marginal
    # percentiles are avoided: width and depth are physically coupled, so they
    # can specify a joint target with no achievable parameter combination.)
    sparse_idx = int(np.argmin(neighbor_counts))
    target_w_2 = float(pred_w[sparse_idx])
    target_d_2 = float(pred_d[sparse_idx])

    print("=" * 100)
    print("TARGET 2: SPARSE REGION")
    print("=" * 100)
    print(f"Anchor: test point #{sparse_idx} "
          f"({neighbor_counts[sparse_idx]} neighbors within tolerance, "
          f"vs {neighbor_counts[best_idx]} for Target 1)")
    print(f"Target: Width={target_w_2:.1f}um, Depth={target_d_2:.1f}um")
    print(f"Tolerance: +/-{delta_w:.1f}um width, +/-{delta_d:.1f}um depth\n")

    df_t2, counts_t2, acc_t2 = evaluate_gate(
        target_w_2, target_d_2, delta_w, delta_d, pred_w, pred_d,
        ci_low_w, ci_high_w, ci_low_d, ci_high_d, X_test, y_test)

    print(df_t2.to_string(index=False))
    print(f"\nVerdict distribution: ACCEPT={counts_t2['ACCEPT']} | "
          f"FLAG={counts_t2['FLAG']} | REJECT={counts_t2['REJECT']}")
    print(f"Verdict accuracy: {acc_t2*len(X_test):.0f}/{len(X_test)} "
          f"({100*acc_t2:.1f}%)\n")
    df_t2.to_csv(RESULTS_DIR / 'gating_target2_sparse.csv', index=False)

    # ---- TARGET 3: OUT-OF-DOMAIN (extrapolation) ----------------------------
    # Single factor at a time, 50-150% beyond the validated domain
    # (P in [48,80] W, v in [10,50] mm/s, dz in [-4,+12] mm).
    out_of_domain_params = np.array([
        [60,   8, 120],   # power 50% above max (120 vs 80)
        [60,   8,  20],   # power 58% below min (20 vs 48)
        [ 5,   8,  60],   # speed 50% below min (5 vs 10)
        [80,   8,  60],   # speed 60% above max (80 vs 50)
        [30,  25,  60],   # defocus 108% above max (25 vs 12)
        [30, -10,  60],   # defocus 150% below min (-10 vs -4)
    ], dtype=np.float32)
    # columns are [speed, defocus, power], matching X_f order above

    print("=" * 100)
    print("TARGET 3: OUT-OF-DOMAIN (extrapolation)")
    print("=" * 100)
    print("Querying GPR at parameter combinations outside the validated domain:")
    for row in out_of_domain_params:
        print(f"  v={row[0]:.0f} mm/s, dz={row[1]:.0f} mm, P={row[2]:.0f} W")

    pred_w_oob, sigma_w_oob = predict_oob(gpr_results['Width'],
                                          out_of_domain_params)
    pred_d_oob, sigma_d_oob = predict_oob(gpr_results['Depth'],
                                          out_of_domain_params)

    ci_low_w_oob  = pred_w_oob - 1.96 * sigma_w_oob
    ci_high_w_oob = pred_w_oob + 1.96 * sigma_w_oob
    ci_low_d_oob  = pred_d_oob - 1.96 * sigma_d_oob
    ci_high_d_oob = pred_d_oob + 1.96 * sigma_d_oob

    # Tested against the Target 1 (dense region) specification: can an
    # extrapolated prediction ever falsely ACCEPT against a reasonable
    # in-domain spec?
    df_t3, counts_t3, _ = evaluate_gate(
        target_w_1, target_d_1, delta_w, delta_d, pred_w_oob, pred_d_oob,
        ci_low_w_oob, ci_high_w_oob, ci_low_d_oob, ci_high_d_oob,
        out_of_domain_params, y_eval=None)

    print("\n" + df_t3.to_string(index=False))
    print(f"\nVerdict distribution: ACCEPT={counts_t3['ACCEPT']} | "
          f"FLAG={counts_t3['FLAG']} | REJECT={counts_t3['REJECT']}")

    mean_sigma_w_indomain = float(np.mean(sigma_w))
    mean_sigma_d_indomain = float(np.mean(sigma_d))
    print(f"\nIn-domain mean calibrated sigma  -- Width: "
          f"{mean_sigma_w_indomain:.2f}um | Depth: {mean_sigma_d_indomain:.2f}um")
    print(f"Out-of-domain mean calibrated sigma -- Width: {np.mean(sigma_w_oob):.2f}um | "
          f"Depth: {np.mean(sigma_d_oob):.2f}um")
    print(f"Sigma inflation factor -- Width: "
          f"{np.mean(sigma_w_oob)/mean_sigma_w_indomain:.2f}x | Depth: "
          f"{np.mean(sigma_d_oob)/mean_sigma_d_indomain:.2f}x")
    print("(No ground truth exists for out-of-domain points. The relevant check "
          "is whether sigma is meaningfully inflated and whether REJECT "
          "dominates the verdict distribution.)")
    df_t3.to_csv(RESULTS_DIR / 'gating_target3_outofdomain.csv', index=False)

    # ---- SUMMARY ACROSS ALL THREE TARGET CONDITIONS -------------------------
    print("\n" + "=" * 100)
    print("SUMMARY: VERDICT DISTRIBUTION ACROSS ALL THREE TARGET CONDITIONS")
    print("=" * 100)
    summary = pd.DataFrame([
        {"Target": "1: Dense Region",
         "ACCEPT": counts_t1['ACCEPT'], "FLAG": counts_t1['FLAG'],
         "REJECT": counts_t1['REJECT'],
         "Verdict Accuracy": f"{100*acc_t1:.1f}%"},
        {"Target": "2: Sparse Region",
         "ACCEPT": counts_t2['ACCEPT'], "FLAG": counts_t2['FLAG'],
         "REJECT": counts_t2['REJECT'],
         "Verdict Accuracy": f"{100*acc_t2:.1f}%"},
        {"Target": "3: Out-of-Domain",
         "ACCEPT": counts_t3['ACCEPT'], "FLAG": counts_t3['FLAG'],
         "REJECT": counts_t3['REJECT'],
         "Verdict Accuracy": "N/A (no ground truth)"},
    ])
    print(summary.to_string(index=False))
    summary.to_csv(RESULTS_DIR / 'gating_three_target_summary.csv', index=False)
    print(f"\nSaved to results/: gating_target1_dense.csv, "
          f"gating_target2_sparse.csv, gating_target3_outofdomain.csv, "
          f"gating_three_target_summary.csv")


if __name__ == '__main__':
    main()
