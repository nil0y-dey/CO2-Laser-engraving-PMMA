
import numpy as np
import pandas as pd
import torch
import gpytorch
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import random
import os
import shap
import matplotlib.gridspec as gridspec




#GLOBAL DETERMINISM
def set_global_determinism(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


# GPR MODEL ARCHITECTURE
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


# PIPELINE
def run_single_task_pipeline(X_train_val, X_test,
                              y_train_val_1d, y_test_1d,
                              seed, task_name,
                              NOISE_LOG_LB, NOISE_LOG_UB,
                              input_names=None,
                              noise_test=False, noise_level=0.0):
    NOISE_LOW  = 10 ** NOISE_LOG_LB
    NOISE_HIGH = 10 ** NOISE_LOG_UB

    device = torch.device("cpu")

    # FITNESS FUNCTION
    def fitness_function(params, X_data, y_data_1d):
        kf = KFold(n_splits=5, shuffle=True, random_state=seed)
        all_fold_nll = []

        ls_params    = np.array(params[:3], dtype=np.float32).reshape(1, -1)
        actual_noise = float(np.clip(10 ** params[3], NOISE_LOW, NOISE_HIGH))

        for train_idx, val_idx in kf.split(X_data):
            X_tr  = torch.tensor(X_data[train_idx], dtype=torch.float32)
            y_tr  = torch.tensor(y_data_1d[train_idx], dtype=torch.float32)

            m_x, s_x = X_tr.mean(0), X_tr.std(0) + 1e-6
            m_y, s_y = y_tr.mean(), y_tr.std() + 1e-6

            X_tr_norm = (X_tr - m_x) / s_x
            y_tr_norm = (y_tr - m_y) / s_y

            likelihood = gpytorch.likelihoods.GaussianLikelihood(
                noise_constraint=gpytorch.constraints.GreaterThan(NOISE_LOW))
            model = SingleTaskGPModel(X_tr_norm, y_tr_norm, likelihood)

            model.initialize(**{
                'covar_module.base_kernel.lengthscale': torch.as_tensor(ls_params),
                'likelihood.noise':                     torch.tensor(actual_noise)
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

    # GWO
    num_wolves, max_iter, dim = 12, 25, 4
    lb = np.array([.1, .1, .1, NOISE_LOG_LB])
    ub = np.array([10, 10, 10, NOISE_LOG_UB])

    Positions   = np.random.uniform(lb, ub, (num_wolves, dim))
    alpha_pos = np.zeros(dim); alpha_score = float("inf")
    beta_pos  = np.zeros(dim); beta_score  = float("inf")
    delta_pos = np.zeros(dim); delta_score = float("inf")

    print(f"\n  [{task_name}] Running GWO...")
    convergence_curve = []

    for l in range(max_iter):
        for i in range(num_wolves):
            fitness = fitness_function(Positions[i, :], X_train_val, y_train_val_1d)
            if fitness < alpha_score:
                delta_score = beta_score;  delta_pos = beta_pos.copy()
                beta_score  = alpha_score; beta_pos  = alpha_pos.copy()
                alpha_score = fitness;     alpha_pos = Positions[i].copy()
            elif fitness < beta_score:
                delta_score = beta_score;  delta_pos = beta_pos.copy()
                beta_score  = fitness;     beta_pos  = Positions[i].copy()
            elif fitness < delta_score:
                delta_score = fitness;     delta_pos = Positions[i].copy()

        a = 2 - l * (2 / max_iter)
        for i in range(num_wolves):
            r1, r2 = np.random.rand(dim), np.random.rand(dim)
            A1 = 2 * a * r1 - a;  C1 = 2 * r2
            D_alpha = abs(C1 * alpha_pos - Positions[i])
            X1 = alpha_pos - A1 * D_alpha

            r1, r2 = np.random.rand(dim), np.random.rand(dim)
            A2 = 2 * a * r1 - a;  C2 = 2 * r2
            D_beta = abs(C2 * beta_pos - Positions[i])
            X2 = beta_pos - A2 * D_beta

            r1, r2 = np.random.rand(dim), np.random.rand(dim)
            A3 = 2 * a * r1 - a;  C3 = 2 * r2
            D_delta = abs(C3 * delta_pos - Positions[i])
            X3 = delta_pos - A3 * D_delta

            Positions[i] = np.clip((X1 + X2 + X3) / 3, lb, ub)

        convergence_curve.append(alpha_score)
        print(f"    Iter {l+1}/{max_iter} | Best NLL: {alpha_score:.6f}")

    if input_names is None:
        input_names = [f'F{i}' for i in range(3)]
    ls_str = ' | '.join([f'{n}={alpha_pos[i]:.4f}' for i, n in enumerate(input_names)])
    print(f"  [{task_name}] Optimized LS: {ls_str} | Noise={10**alpha_pos[3]:.6f}")

    #FINAL MODEL TRAINING
    X_train_t = torch.tensor(X_train_val, dtype=torch.float32)
    y_train_t = torch.tensor(y_train_val_1d, dtype=torch.float32)

    m_x, s_x = X_train_t.mean(0), X_train_t.std(0) + 1e-7
    m_y, s_y = y_train_t.mean(), y_train_t.std() + 1e-7

    X_train_norm = (X_train_t - m_x) / s_x
    y_train_norm = (y_train_t - m_y) / s_y

    likelihood = gpytorch.likelihoods.GaussianLikelihood(
        noise_constraint=gpytorch.constraints.GreaterThan(NOISE_LOW)
    )
    model = SingleTaskGPModel(X_train_norm, y_train_norm, likelihood)
    best_noise = float(np.clip(10 ** alpha_pos[3], NOISE_LOW, NOISE_HIGH))
    model.initialize(**{
        'covar_module.base_kernel.lengthscale': torch.tensor(
            np.array([alpha_pos[:3]], dtype=np.float32)),
        'likelihood.noise': torch.tensor(best_noise)
    })

    model.train(); likelihood.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    print(f"  [{task_name}] Training Final Model...")
    for _ in range(180):
        optimizer.zero_grad()
        loss = -mll(model(X_train_norm), y_train_norm)
        loss.backward()
        optimizer.step()

    print(f"  [{task_name}] Final Noise: {likelihood.noise.item():.6f}")

    # CONFORMAL PREDICTION
    kf_cal = KFold(n_splits=5, shuffle=True, random_state=seed)
    cv_val_resid_norm = []
    cv_val_sigma_norm = []

    best_ls    = torch.tensor(np.array([alpha_pos[:3]], dtype=np.float32))
    best_noise = float(np.clip(10 ** alpha_pos[3], NOISE_LOW, NOISE_HIGH))


    for train_idx, val_idx in kf_cal.split(X_train_val):
        X_tr  = torch.tensor(X_train_val[train_idx], dtype=torch.float32)
        y_tr  = torch.tensor(y_train_val_1d[train_idx], dtype=torch.float32)
        X_val = torch.tensor(X_train_val[val_idx],   dtype=torch.float32)
        y_val = torch.tensor(y_train_val_1d[val_idx], dtype=torch.float32)

        #NOISE-AWARE CALIBRATION
        if noise_test:
            np.random.seed(seed) # Reproducible noise
            stds_val = np.std(X_val.numpy(), axis=0)
            noise_val = np.random.normal(0, noise_level * stds_val, X_val.shape)
            X_val = X_val + torch.tensor(noise_val, dtype=torch.float32)
        # --------------------------------

        m_xc, s_xc = X_tr.mean(0), X_tr.std(0) + 1e-7
        m_yc, s_yc = y_tr.mean(), y_tr.std() + 1e-7

        X_tr_c  = (X_tr  - m_xc) / s_xc
        y_tr_c  = (y_tr  - m_yc) / s_yc
        X_val_c = (X_val - m_xc) / s_xc
        y_val_c = (y_val - m_yc) / s_yc

        lik_c = gpytorch.likelihoods.GaussianLikelihood(
            noise_constraint=gpytorch.constraints.GreaterThan(NOISE_LOW))
        mod_c = SingleTaskGPModel(X_tr_c, y_tr_c, lik_c)
        mod_c.initialize(**{
            'covar_module.base_kernel.lengthscale': best_ls,
            'likelihood.noise':                     torch.tensor(best_noise)
        })

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
            val_pred  = lik_c(mod_c(X_val_c))
            val_mu    = val_pred.mean.numpy()
            val_sigma = torch.sqrt(val_pred.variance).numpy()

        val_resid = y_val_c.numpy() - val_mu
        cv_val_resid_norm.extend(val_resid.tolist())
        cv_val_sigma_norm.extend(val_sigma.tolist())

    cv_resid     = np.array(cv_val_resid_norm)
    cv_sigma     = np.array(cv_val_sigma_norm)

    
    # CONFORMAL calculation 
    
    cv_std_resid = np.abs(cv_resid / (cv_sigma + 1e-8))
    conformal_quantile = float(np.quantile(cv_std_resid, 0.95))
    scale_factor = (conformal_quantile / 1.96) * 1.05  # Divide by 1.96 so it applies correctly to the 1.96*sigma bounds
    

    print(f"  [{task_name}] CV-based Conformal Scale Factor: {scale_factor:.4f}")

    #PREDICT ON TEST
    model.eval(); likelihood.eval()

    X_test_t    = torch.tensor(X_test, dtype=torch.float32)
    X_test_norm = (X_test_t - m_x) / s_x

    y_test_t    = torch.tensor(y_test_1d, dtype=torch.float32)
    y_test_norm = (y_test_t - m_y) / s_y

    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        observed_pred = likelihood(model(X_test_norm))
        y_pred_norm   = observed_pred.mean
        std_pred_norm = torch.sqrt(observed_pred.variance)

    sigma_norm_raw = std_pred_norm.numpy().copy()
    std_phys_raw   = (std_pred_norm * s_y).numpy().copy()

    raw_picp_95 = np.mean(
        (y_test_norm.numpy() >= y_pred_norm.numpy() - 1.96 * sigma_norm_raw) &
        (y_test_norm.numpy() <= y_pred_norm.numpy() + 1.96 * sigma_norm_raw)
    )
    raw_dist = torch.distributions.Normal(y_pred_norm, std_pred_norm + 1e-8)
    raw_nlpd = -raw_dist.log_prob(y_test_norm).mean().item()

    print(f"  [RAW] PICP: {raw_picp_95:.4f} | NLPD: {raw_nlpd:.4f}")

    y_pred     = (y_pred_norm * s_y + m_y).numpy()
    sigma_norm = sigma_norm_raw * scale_factor
    sigma_phys = std_phys_raw  * scale_factor

    true_phys  = y_test_1d
    pred_phys  = y_pred
    resid_phys = true_phys - pred_phys

    true_norm  = y_test_norm.numpy()
    pred_norm  = y_pred_norm.numpy()
    resid_norm = true_norm - pred_norm
    std_norm_r = resid_norm / (sigma_norm + 1e-8)

    r2   = r2_score(true_phys, pred_phys)
    rmse = np.sqrt(mean_squared_error(true_phys, pred_phys))
    mae  = mean_absolute_error(true_phys, pred_phys)
    mape = np.mean(np.abs(resid_phys / true_phys)) * 100
    kurt = stats.kurtosis(resid_phys, fisher=True)

    std_norm_resid = np.std(std_norm_r)
    picp_95 = np.mean(
        (true_norm >= pred_norm - 1.96 * sigma_norm) &
        (true_norm <= pred_norm + 1.96 * sigma_norm)
    )
    std_pred_norm_cal = torch.tensor(sigma_norm, dtype=torch.float32)
    dist = torch.distributions.Normal(y_pred_norm, std_pred_norm_cal + 1e-8)
    nlpd = -dist.log_prob(y_test_norm).mean().item()

    raw_mpiw = np.mean(2 * 1.96 * std_phys_raw)
    mpiw = np.mean(2 * 1.96 * sigma_phys)

    print(f"\n  --- {task_name} Final Metrics ---")
    print(f"  R²: {r2:.4f} | RMSE: {rmse:.4f} μm | MAE: {mae:.4f} μm | MAPE: {mape:.2f}%")
    print(f"  MPIW: {mpiw:.4f} μm | Kurtosis: {kurt:.4f} | "
          f"Std Norm Resid: {std_norm_resid:.4f} | PICP: {picp_95:.4f} | NLPD: {nlpd:.4f}")

    return {
        'R2': r2, 'RMSE': rmse, 'MAE': mae, 'MAPE (%)': mape,
        'MPIW': mpiw,
        'Kurtosis': kurt, 'Std Norm Resid': std_norm_resid,
        'PICP': picp_95, 'NLPD': nlpd,
        'Scale Factor': scale_factor,
        'Raw_PICP': raw_picp_95,
        'Raw_NLPD': raw_nlpd,
        '_true_phys':      true_phys,
        '_pred_phys':      pred_phys,
        '_sigma_phys':     sigma_phys,
        '_resid_phys':     resid_phys,
        '_true_norm':      true_norm,
        '_pred_norm':      pred_norm,
        '_sigma_norm':     sigma_norm,
        '_resid_norm':     resid_norm,
        '_std_norm_r':     std_norm_r,
        '_convergence':    convergence_curve,
        '_resid_phys_1d':  resid_phys,
        '_sigma_norm_raw': sigma_norm_raw,
        '_sigma_phys_raw': std_phys_raw,
        'Raw_MPIW': raw_mpiw,
        '_model':       model,
        '_likelihood':  likelihood,
        '_m_x':         m_x,
        '_s_x':         s_x,
        '_m_y':         m_y,
        '_s_y':         s_y,
        '_scale_factor': scale_factor,
    }



# SINGLE-SEED run
def run_gwo_gpr_separate_pipeline(X_raw, y_raw, seed,
                                   downsample_fraction=1.0,
                                   noise_test=False,
                                   noise_level=0.0):

    np.random.seed(seed)
    torch.manual_seed(seed)

   
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X_raw, y_raw, test_size=0.20, random_state=seed
    )

    # 1. DOWNSAMPLING
    if downsample_fraction < 1.0:
        X_train_val, _, y_train_val, _ = train_test_split(
            X_train_val, y_train_val, train_size=downsample_fraction, random_state=seed
        )
        print(f"\n[Seed {seed}] Downsampled Train Size: {len(X_train_val)} (Test Size: {len(X_test)})")

    # 2. NOISE INJECTION (Train on clean, test on noisy)
    if noise_test:
        # Calculate std of X_test to scale noise appropriately
        stds = np.std(X_test, axis=0)
        # Create Gaussian noise
        noise = np.random.normal(0, noise_level * stds, X_test.shape)
        X_test = X_test + noise
        print(f"\n[Seed {seed}] Added {noise_level*100:.1f}% Gaussian Noise to Test Set.")
    # ==============================================================

    NOISE_BOUNDS = {
        'Width': (-4, -0.1),
        'Depth': (-4, -0.1),
    }

    out_names = ['Width', 'Depth']
    results   = {}

    print(f"[Seed {seed}] Running Separate Single-Task GPR...")

    for i, name in enumerate(out_names):
        np.random.seed(seed)
        torch.manual_seed(seed)

        lb, ub = NOISE_BOUNDS[name]

        res = run_single_task_pipeline(
            X_train_val, X_test,
            y_train_val[:, i], y_test[:, i],
            seed, name, lb, ub,
            noise_test=noise_test,
            noise_level=noise_level
        )
        results[name] = res

    # --- DIAGNOSTIC PLOTS ---
    colors      = ['#1f77b4', '#ff7f0e']
    conf_levels = np.linspace(0.1, 0.95, 15)

    for i, name in enumerate(out_names):
        true_phys  = results[name]['_true_phys']
        pred_phys  = results[name]['_pred_phys']
        sigma_phys = results[name]['_sigma_phys']
        resid_phys = results[name]['_resid_phys']
        true_norm  = results[name]['_true_norm']
        pred_norm  = results[name]['_pred_norm']
        sigma_norm = results[name]['_sigma_norm']
        resid_norm = results[name]['_resid_norm']
        std_norm_r = results[name]['_std_norm_r']
        raw_sigma_n = results[name]['_sigma_norm_raw']
        raw_sigma_p = results[name]['_sigma_phys_raw']

        fig, ax = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle(
            f'GPR (Single-Task) Diagnostic Suite: {name} (Seed {seed})',
            fontsize=14, fontweight='bold'
        )

        lims = [min(true_phys.min(), pred_phys.min()),
                max(true_phys.max(), pred_phys.max())]
        ax[0, 0].errorbar(
            true_phys, pred_phys,
            yerr=1.96 * raw_sigma_p,
            fmt='none', ecolor='gray', alpha=0.3, capsize=3, label='Raw 95% CI'
        )
        ax[0, 0].errorbar(
            true_phys, pred_phys,
            yerr=1.96 * sigma_phys,
            fmt='o', ecolor=colors[i], mfc=colors[i], mec='k',
            capsize=3, alpha=0.7, label='Calibrated 95% CI'
        )
        ax[0, 0].plot(lims, lims, 'r--', lw=2, label='Ideal (1:1)')
        ax[0, 0].set_title(f'{name}: Parity Plot')
        ax[0, 0].set_xlabel('Experimental [μm]')
        ax[0, 0].set_ylabel('Predicted [μm]')
        ax[0, 0].legend()

        sns.histplot(resid_phys, bins=15, kde=True, color=colors[i],
                     ax=ax[0, 1], edgecolor='black', alpha=0.6)
        ax[0, 1].set_title(
            f'{name}: Residual Distribution '
            f'(Kurtosis: {results[name]["Kurtosis"]:.2f})'
        )
        ax[0, 1].set_xlabel('Residual [μm]')

        stats.probplot(std_norm_r, dist="norm", plot=ax[0, 2])
        ax[0, 2].get_lines()[0].set_markerfacecolor(colors[i])
        ax[0, 2].get_lines()[0].set_markeredgecolor('k')
        ax[0, 2].set_title(f'{name}: Q-Q Plot (Standardized Residuals)')

        raw_observed_picp = [
            np.mean(np.abs(resid_norm) <= stats.norm.ppf((1 + c) / 2) * raw_sigma_n)
            for c in conf_levels
        ]
        observed_picp = [
            np.mean(np.abs(resid_norm) <= stats.norm.ppf((1 + c) / 2) * sigma_norm)
            for c in conf_levels
        ]
        ax[1, 0].plot(conf_levels, raw_observed_picp, 's--', color='gray',
                      alpha=0.4, label='Raw GPR (Uncalibrated)')
        ax[1, 0].plot(conf_levels, observed_picp, 'o-', color=colors[i],
                      label=f'Calibrated (Scale: {results[name]["Scale Factor"]:.2f})')
        ax[1, 0].plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfect Calibration')
        ax[1, 0].set_title(f'{name}: Calibration Impact')
        ax[1, 0].set_xlabel('Expected Confidence')
        ax[1, 0].set_ylabel('Observed Coverage')
        ax[1, 0].legend()

        sort_idx = np.argsort(true_norm)
        x_plot   = np.arange(len(true_norm))
        ax[1, 1].plot(x_plot, (pred_norm - 1.96 * raw_sigma_n)[sort_idx],
                      'k--', lw=1, alpha=0.2)
        ax[1, 1].plot(x_plot, (pred_norm + 1.96 * raw_sigma_n)[sort_idx],
                      'k--', lw=1, alpha=0.2, label='Raw 95% Band')
        ax[1, 1].fill_between(
            x_plot,
            (pred_norm - 1.96 * sigma_norm)[sort_idx],
            (pred_norm + 1.96 * sigma_norm)[sort_idx],
            color=colors[i], alpha=0.3, label='Calibrated 95% Band'
        )
        ax[1, 1].scatter(x_plot, true_norm[sort_idx], c='k', s=20,
                         label='Experimental', zorder=3)
        ax[1, 1].plot(x_plot, pred_norm[sort_idx], color=colors[i],
                      lw=2, label='GPR Mean')
        ax[1, 1].set_title(f'{name}: Uncertainty Envelope')
        ax[1, 1].set_xlabel('Sorted Test Samples')
        ax[1, 1].set_ylabel('Normalized Value')
        ax[1, 1].legend()

        ax[1, 2].scatter(pred_phys, resid_phys, alpha=0.7,
                         edgecolors='k', color=colors[i])
        ax[1, 2].axhline(0, color='r', linestyle='--')
        ax[1, 2].set_title(f'{name}: Residuals vs Predicted')
        ax[1, 2].set_xlabel('Predicted [μm]')
        ax[1, 2].set_ylabel('Residual [μm]')

        plt.tight_layout()
        if seed == 589 and downsample_fraction == 1.0 and not noise_test:
            plt.savefig(
                f'GPR_SingleTask_{name}_seed42_diagnostic.png',
                dpi=300, bbox_inches='tight'
            )
        plt.show()

        plt.figure(figsize=(8, 4))
        plt.plot(results[name]['_convergence'], color='#FF5733',
                 linewidth=2, marker='o', markersize=4)
        plt.title(f'GWO Convergence: {name} (Seed {seed})')
        plt.xlabel('Iteration')
        plt.ylabel('Best NLL (Fitness)')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.show()

    res_width = results['Width']['_resid_phys_1d']
    res_depth = results['Depth']['_resid_phys_1d']

    seed42_models = None
    if seed == 589 and downsample_fraction == 1.0 and not noise_test:
        seed42_models = {
            'Width':  {
                'model':      results['Width']['_model'],
                'likelihood': results['Width']['_likelihood'],
                'm_x':        results['Width']['_m_x'],
                's_x':        results['Width']['_s_x'],
                'm_y':        results['Width']['_m_y'],
                's_y':        results['Width']['_s_y'],
            },
            'Depth': {
                'model':      results['Depth']['_model'],
                'likelihood': results['Depth']['_likelihood'],
                'm_x':        results['Depth']['_m_x'],
                's_x':        results['Depth']['_s_x'],
                'm_y':        results['Depth']['_m_y'],
                's_y':        results['Depth']['_s_y'],
            },
        }

    return (y_test,
            results['Width']['_pred_phys'],
            results['Depth']['_pred_phys'],
            results['Width']['_convergence'],
            res_width, res_depth, results,
            seed42_models)



# SECTION 4: MULTI-SEED run

def run_multi_seed_experiment_gpr_separate(df, seed_list, selection_mode,
                                           downsample_fraction=1.0,
                                           noise_test=False,
                                           noise_level=0.0):
    all_res_width, all_res_depth = [], []
    results_list, conv_history   = [], {}
    seed42_models_out            = None

    idx_map = {
        's1': list(range(3)),
        's2': [3, 4, 5],
        's3': [0, 1, 5],
        's4': [4, 5, 1]
    }
    if selection_mode not in idx_map:
        raise ValueError(f"Unknown mode '{selection_mode}'. Use s1–s4.")

    X_f          = df.iloc[:, idx_map[selection_mode]].values
    y_f          = df.iloc[:, -2:].values
    input_names  = df.columns[idx_map[selection_mode]].tolist()
    output_names = df.columns[-2:].tolist()

    # Create a string for the title to know what test is running
    test_title = "100% Data"
    if downsample_fraction < 1.0:
        test_title = f"{int(downsample_fraction*100)}% Data"
    if noise_test:
        test_title += f" + {int(noise_level*100)}% Noise"

    print("\n" + f" Selected Configuration ({test_title}) ".center(50, "="))
    print(f"  Mode    : {selection_mode}")
    print(f"  Inputs  : {input_names}")
    print(f"  Outputs : {output_names}")
    print("=" * 50 + "\n")

    for s in seed_list:
        (y_true, p_w, p_d, curve,
         res_w, res_d, seed_results,
         s42_models) = run_gwo_gpr_separate_pipeline(
             X_f, y_f, s,
             downsample_fraction=downsample_fraction,
             noise_test=noise_test,
             noise_level=noise_level
         )

        if s == 589 and s42_models is not None:
            seed42_models_out = s42_models
            seed42_X_train_val, seed42_X_test, _, _ = train_test_split(
                X_f, y_f, test_size=0.20, random_state=42
            )

        conv_history[s] = curve
        all_res_width.append(res_w)
        all_res_depth.append(res_d)

        results_list.append({
            'Seed':           s,
            'W_R2':           seed_results['Width']['R2'],
            'W_RMSE':         seed_results['Width']['RMSE'],
            'W_MAE':          seed_results['Width']['MAE'],
            'W_MAPE':         seed_results['Width']['MAPE (%)'],
            'W_MPIW':         seed_results['Width']['MPIW'],
            'W_PICP':         seed_results['Width']['PICP'],
            'W_NLPD':         seed_results['Width']['NLPD'],
            'W_StdNormResid': seed_results['Width']['Std Norm Resid'],
            'W_ScaleFactor':  seed_results['Width']['Scale Factor'],
            'W_Raw_PICP':     seed_results['Width']['Raw_PICP'],
            'W_Raw_NLPD':     seed_results['Width']['Raw_NLPD'],
            'W_Raw_MPIW':     seed_results['Width']['Raw_MPIW'],
            'D_R2':           seed_results['Depth']['R2'],
            'D_RMSE':         seed_results['Depth']['RMSE'],
            'D_MAE':          seed_results['Depth']['MAE'],
            'D_MAPE':         seed_results['Depth']['MAPE (%)'],
            'D_MPIW':         seed_results['Depth']['MPIW'],
            'D_PICP':         seed_results['Depth']['PICP'],
            'D_NLPD':         seed_results['Depth']['NLPD'],
            'D_StdNormResid': seed_results['Depth']['Std Norm Resid'],
            'D_ScaleFactor':  seed_results['Depth']['Scale Factor'],
            'D_Raw_PICP':     seed_results['Depth']['Raw_PICP'],
            'D_Raw_NLPD':     seed_results['Depth']['Raw_NLPD'],
            'D_Raw_MPIW':     seed_results['Depth']['Raw_MPIW'],
        })

    all_results_df = pd.DataFrame(results_list)
    summary        = all_results_df.drop(columns='Seed').agg(['mean', 'std', 'min', 'max']).T

    print("\n" + f" FINAL STATS OVER ALL SEEDS ({selection_mode} - {test_title}) ".center(60, "="))
    print(summary.to_string())
    print("=" * 60)

    pooled_w = np.concatenate(all_res_width)
    pooled_d = np.concatenate(all_res_depth)
    print(f"\nPooled Residual Mean — Width: {np.mean(pooled_w):.4f} μm | "
          f"Depth: {np.mean(pooled_d):.4f} μm")
    print(f"Pooled Residual Std  — Width: {np.std(pooled_w):.4f} μm | "
          f"Depth: {np.std(pooled_d):.4f} μm")

    try:
        safe_title = test_title.replace(" ", "_").replace("%", "pct")
        all_results_df.to_csv(f'gpr_results_{selection_mode}_{safe_title}.csv', index=False)
        print(f"\nResults saved to CSV.")
    except:
        pass

    return (all_results_df, summary,
            seed42_models_out,
            None, None,
            X_f, y_f)



#GPR WRAPPER FOR SHAP
class GPRSHAPWrapper:
    """
    Wraps trained single-task GPR models (Width + Depth) for SHAP.
    predict() returns shape (N, 2) — required by KernelExplainer
    for multi-output SHAP. Single-output functions also provided.
    """
    model_label = 'GPR'
    shap_type   = 'kernel'

    def __init__(self, width_objs, depth_objs):
        """
        width_objs / depth_objs: dict with keys
            model, likelihood, m_x, s_x, m_y, s_y
        """
        self.w = width_objs
        self.d = depth_objs

    def _predict_single(self, X_np, objs):
        model      = objs['model']
        likelihood = objs['likelihood']
        m_x        = objs['m_x']
        s_x        = objs['s_x']
        m_y        = objs['m_y']
        s_y        = objs['s_y']

        model.eval(); likelihood.eval()
        X_t    = torch.tensor(X_np.astype(np.float32))
        X_norm = (X_t - m_x) / s_x
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            pred   = likelihood(model(X_norm))
            y_norm = pred.mean
        return (y_norm * s_y + m_y).numpy()

    def predict_width(self, X_np):
        return self._predict_single(np.atleast_2d(X_np), self.w)

    def predict_depth(self, X_np):
        return self._predict_single(np.atleast_2d(X_np), self.d)

    def predict_both(self, X_np):
        """Returns (N,2) array — used by multi-output KernelExplainer."""
        w = self.predict_width(X_np)
        d = self.predict_depth(X_np)
        return np.column_stack([w, d])



#SHAP ANALYSIS


def run_gpr_shap_analysis(wrapper, X_train_val, X_test,
                           feature_names, selection_mode,
                           n_background=50, n_samples=200):
    """
    Runs KernelSHAP for GPR Width and Depth independently.
    Uses K-means background for efficiency (Huang et al. approach).

    Returns dict: {'Width': {'shap_values', 'shap_df', 'top_features'},
                   'Depth': {'shap_values', 'shap_df', 'top_features'}}
    """
    # K-means background — more representative than random sample
    background = shap.kmeans(X_train_val, n_background)

    shap_results = {}

    for output_label, predict_fn in [
        ('Width', wrapper.predict_width),
        ('Depth', wrapper.predict_depth)
    ]:
        print(f"\n[SHAP] GPR — {output_label} ({selection_mode}) | "
              f"background={n_background} | nsamples={n_samples}...")

        explainer = shap.KernelExplainer(predict_fn, background)
        shap_vals = explainer.shap_values(X_test, nsamples=n_samples)
        shap_df   = pd.DataFrame(shap_vals, columns=feature_names)

        # mean |SHAP| ranking
        mean_abs  = np.abs(shap_df).mean().sort_values(ascending=False)
        top_feats = mean_abs.index.tolist()

        print(f"  Feature ranking ({output_label}): {top_feats}")

        # Bar + Beeswarm ─────────────────────────────────────
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        fig.suptitle(
            f'SHAP — GPR | {output_label} | {selection_mode} | Seed 42',
            fontsize=13, fontweight='bold'
        )

        # Bar chart
        mean_sorted = mean_abs.sort_values(ascending=True)
        colors_bar  = plt.cm.Blues(
            np.linspace(0.4, 0.9, len(mean_sorted)))
        axes[0].barh(mean_sorted.index, mean_sorted.values,
                     color=colors_bar, edgecolor='k', alpha=0.85)
        axes[0].set_title('Mean |SHAP| — Feature Importance')
        axes[0].set_xlabel('Mean |SHAP| Value [μm]')
        axes[0].grid(True, axis='x', linestyle='--', alpha=0.5)

        # Beeswarm
        plt.sca(axes[1])
        shap.summary_plot(
            shap_vals, X_test,
            feature_names=feature_names,
            show=False, plot_type='dot',
            color_bar_label='Feature Value'
        )
        axes[1].set_title('SHAP Beeswarm — Direction & Magnitude')
        axes[1].set_xlabel(f'SHAP Value (impact on {output_label} [μm])')

        plt.tight_layout()
        plt.savefig(
            f'SHAP_GPR_{output_label}_{selection_mode}_seed42.png',
            dpi=300, bbox_inches='tight'
        )
        plt.show()

        shap_results[output_label] = {
            'shap_values':  shap_vals,
            'shap_df':      shap_df,
            'mean_abs':     mean_abs,
            'top_features': top_feats,
        }

    return shap_results



# PDP + ICE 


def run_gpr_pdp_ice_analysis(wrapper, X_data, feature_names,
                              shap_results, selection_mode,
                              n_ice=20, n_grid=50):
    """
    Runs PDP + ICE for ALL features, for both Width and Depth.
    Layout: one row per feature, two columns (Width | Depth).
    ICE lines shown in light color, PDP mean shown bold.
    Top feature per output highlighted with annotation.

    Parameters:
    -----------
    wrapper        : GPRSHAPWrapper
    X_data         : test set (used for ICE lines + rug)
    feature_names  : list of feature name strings
    shap_results   : output of run_gpr_shap_analysis (for ranking labels)
    selection_mode : 's1', 's2', 's3', 's4'
    n_ice          : number of ICE lines to draw
    n_grid         : number of grid points for PDP sweep
    """
    n_features = len(feature_names)
    colors     = {'Width': '#1f77b4', 'Depth': '#ff7f0e'}

    predict_fns = {
        'Width': wrapper.predict_width,
        'Depth': wrapper.predict_depth,
    }

    fig, axes = plt.subplots(
        n_features, 2,
        figsize=(14, 4.5 * n_features),
        squeeze=False
    )
    fig.suptitle(
        f'PDP + ICE — GPR | {selection_mode} | Seed 42',
        fontsize=14, fontweight='bold', y=1.01
    )

    # Select ICE sample indices — for reproducibility
    rng      = np.random.RandomState(42)
    ice_idx  = rng.choice(len(X_data), size=min(n_ice, len(X_data)),
                          replace=False)

    for feat_idx, feat_name in enumerate(feature_names):

        # Grid range: 5th–95th percentile (avoids sparse boundary regions)
        lo  = np.percentile(X_data[:, feat_idx], 5)
        hi  = np.percentile(X_data[:, feat_idx], 95)
        grid = np.linspace(lo, hi, n_grid)

        for col_idx, output_label in enumerate(['Width', 'Depth']):
            ax         = axes[feat_idx][col_idx]
            predict_fn = predict_fns[output_label]
            color      = colors[output_label]

            # ── ICE lines ──────────────────────────────────────────────
            ice_preds = np.zeros((len(ice_idx), n_grid))
            for j, idx in enumerate(ice_idx):
                X_sweep               = np.tile(X_data[idx], (n_grid, 1))
                X_sweep[:, feat_idx]  = grid
                ice_preds[j]          = predict_fn(X_sweep)

            for j in range(len(ice_idx)):
                ax.plot(grid, ice_preds[j], color=color,
                        alpha=0.15, lw=1.0)

            # ── PDP mean ───────────────────────────────────────────────
            pdp_preds = np.zeros(n_grid)
            for k, val in enumerate(grid):
                X_sweep              = X_data.copy()
                X_sweep[:, feat_idx] = val
                pdp_preds[k]         = predict_fn(X_sweep).mean()

            ax.plot(grid, pdp_preds, color=color,
                    lw=2.5, zorder=5, label='PDP (mean)')

            # ── Global mean reference line ─────────────────────────────
            global_mean = predict_fn(X_data).mean()
            ax.axhline(global_mean, color='gray', linestyle='--',
                       alpha=0.6, lw=1.2, label=f'Global mean')

            # ── Rug plot — actual data distribution ────────────────────
            sns.rugplot(X_data[:, feat_idx], ax=ax,
                        color='black', alpha=0.4, height=0.04)

            # ── SHAP rank label ────────────────────────────────────────
            rank = shap_results[output_label]['top_features'].index(feat_name) + 1
            mean_shap_val = shap_results[output_label]['mean_abs'][feat_name]
            ax.set_title(
                f'{output_label}: {feat_name}  '
                f'[SHAP rank #{rank} | mean|SHAP|={mean_shap_val:.3f}]',
                fontsize=10
            )
            ax.set_xlabel(feat_name)
            ax.set_ylabel(f'Predicted {output_label} [μm]')
            ax.legend(fontsize=8, loc='best')
            ax.grid(True, linestyle='--', alpha=0.4)

    plt.tight_layout()
    plt.savefig(
        f'PDP_ICE_GPR_{selection_mode}_seed42.png',
        dpi=300, bbox_inches='tight'
    )
    plt.show()
    print(f"[PDP+ICE] Saved: PDP_ICE_GPR_{selection_mode}_seed42.png")


# =============================================================================
# SECTION 4: MASTER RUNNER
# =============================================================================

def run_gpr_shap_pdp_ice(seed42_models, X_train_val, X_test,
                          feature_names, selection_mode,
                          n_background=50, n_samples=200,
                          n_ice=20, n_grid=50):
    """
    Master function — runs full SHAP + PDP + ICE pipeline for GPR.

    Parameters:
    -----------
    seed42_models  : dict from run_multi_seed_experiment_gpr_separate
                     keys: 'Width', 'Depth' — each with model objects
    X_train_val    : training data (SHAP background source)
    X_test         : test data (explained + ICE lines)
    feature_names  : list of feature name strings
    selection_mode : 's1', 's2', 's3', 's4'
    n_background   : K-means background points for KernelExplainer
    n_samples      : KernelSHAP integration samples (higher = more accurate)
    n_ice          : number of ICE lines per plot
    n_grid         : PDP sweep resolution

    Returns:
    --------
    shap_results : dict with SHAP values, rankings, top features
    """
    print("\n" + " GPR SHAP + PDP + ICE ".center(60, "="))
    print(f"  Mode         : {selection_mode}")
    print(f"  Features     : {feature_names}")
    print(f"  Test samples : {len(X_test)}")
    print(f"  Background   : {n_background} K-means centers")
    print(f"  SHAP samples : {n_samples}")
    print(f"  ICE lines    : {n_ice}")
    print("=" * 60)

    # Build wrapper
    wrapper = GPRSHAPWrapper(
        width_objs=seed42_models['Width'],
        depth_objs=seed42_models['Depth']
    )

    # Step 1: SHAP
    shap_results = run_gpr_shap_analysis(
        wrapper, X_train_val, X_test,
        feature_names, selection_mode,
        n_background=n_background,
        n_samples=n_samples
    )

    # Step 2: PDP + ICE for all features
    run_gpr_pdp_ice_analysis(
        wrapper, X_test, feature_names,
        shap_results, selection_mode,
        n_ice=n_ice, n_grid=n_grid
    )

    print("\n[DONE] SHAP + PDP + ICE complete.")
    return shap_results


feature_map = {
    's1': ['Speed', 'Defocus length', 'Power'],
    's2': ['Spot Diameter', 'Linear ED', 'Area ED'],
    's3': ['Speed', 'Defocus length', 'Area ED'],
    's4': ['Power', 'Speed', 'Defocus length', 'Beam Diameter', 'Linear ED', 'Area ED']
}



df_main = pd.read_csv('data.csv')
seeds = [405, 166, 327, 347, 589]
selection_mode = 's1'

print("#####################################################")
print("# 1. RUNNING BASELINE 100% DATA                     #")
print("#####################################################")
all_results_df_100, summary_100, seed42_models, _, _, X_f, y_f = run_multi_seed_experiment_gpr_separate(
    df_main, seeds, selection_mode, downsample_fraction=1.0, noise_test=False
)




shap_results = run_gpr_shap_pdp_ice(
    seed42_models  = seed42_models,
    X_train_val    = X_train_val,
    X_test         = X_test,
    feature_names  = feature_map[selection_mode],
    selection_mode = selection_mode,
    n_background   = 50,    # K-means centers
    n_samples      = 200,   # increase to 500 for final paper figures
    n_ice          = 20,    # ICE lines per plot
    n_grid         = 50     # PDP sweep points
)
