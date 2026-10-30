import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.metrics import (r2_score, mean_squared_error,
                             mean_absolute_error, mean_absolute_percentage_error)
from scipy.stats import probplot
import random
import os



def set_global_determinism(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

set_global_determinism(42)


def run_gwo_svr_pipeline(X_raw, y_raw, seed):
  
    np.random.seed(seed)

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X_raw, y_raw, test_size=0.20, random_state=seed
    )

    # GWO FITNESS FUNCTION
    def fitness_function_svr(params, X_data, y_data):
        kf      = KFold(n_splits=5, shuffle=True, random_state=seed)
        C       = 10**params[0]
        epsilon = 10**params[1]
        gamma   = 10**params[2]
        fold_errors = []

        for train_idx, val_idx in kf.split(X_data):
            X_tr, X_val = X_data[train_idx], X_data[val_idx]
            y_tr, y_val = y_data[train_idx], y_data[val_idx]

            model_w = Pipeline([
                ('poly',   PolynomialFeatures(degree=2, include_bias=False)),
                ('scaler', StandardScaler()),
                ('svr',    SVR(kernel='rbf', C=C, epsilon=epsilon, gamma=gamma))
            ]).fit(X_tr, np.log(y_tr[:, 0]))

            model_d = Pipeline([
                ('poly',   PolynomialFeatures(degree=2, include_bias=False)),
                ('scaler', StandardScaler()),
                ('svr',    SVR(kernel='rbf', C=C, epsilon=epsilon, gamma=gamma))
            ]).fit(X_tr, np.log(y_tr[:, 1]))

            pred_w = np.exp(model_w.predict(X_val))
            pred_d = np.exp(model_d.predict(X_val))

            mape_w = mean_absolute_percentage_error(y_val[:, 0], pred_w)
            mape_d = mean_absolute_percentage_error(y_val[:, 1], pred_d)
            fold_errors.append(np.mean([mape_w, mape_d]))

        return np.mean(fold_errors)

    # GWO LOOP
    num_wolves, max_iter, dim = 12, 25, 3
    lb = np.array([-1.0, -4.0, -4.0])   
    ub = np.array([ 2.0, -1.0, -1.0])   

    Positions = np.random.uniform(lb, ub, (num_wolves, dim))
    alpha_pos, alpha_score = np.zeros(dim), float("inf")
    beta_pos,  beta_score  = np.zeros(dim), float("inf")
    delta_pos, delta_score = np.zeros(dim), float("inf")

    print(f"\n[Seed {seed}] Executing GWO-SVR Optimization...")
    convergence_curve = []

    for l in range(max_iter):
        for i in range(num_wolves):
            Positions[i] = np.clip(Positions[i], lb, ub)
            fitness = fitness_function_svr(Positions[i], X_train_val, y_train_val)

            if fitness < alpha_score:
                alpha_score, alpha_pos = fitness, Positions[i].copy()
            elif fitness < beta_score:
                beta_score,  beta_pos  = fitness, Positions[i].copy()
            elif fitness < delta_score:
                delta_score, delta_pos = fitness, Positions[i].copy()

        a = 2 - l * (2 / max_iter)
        for i in range(num_wolves):
            for j in range(dim):
                r1, r2 = np.random.rand(), np.random.rand()
                A1, C1 = 2*a*r1-a, 2*r2
                X1 = alpha_pos[j] - A1*abs(C1*alpha_pos[j] - Positions[i, j])

                r1, r2 = np.random.rand(), np.random.rand()
                A2, C2 = 2*a*r1-a, 2*r2
                X2 = beta_pos[j] - A2*abs(C2*beta_pos[j] - Positions[i, j])

                r1, r2 = np.random.rand(), np.random.rand()
                A3, C3 = 2*a*r1-a, 2*r2
                X3 = delta_pos[j] - A3*abs(C3*delta_pos[j] - Positions[i, j])

                Positions[i, j] = (X1 + X2 + X3) / 3

        Positions = np.clip(Positions, lb, ub)
        convergence_curve.append(alpha_score)
        if (l+1) % 1 == 0 or (l+1) == max_iter:
            print(f"Iteration {l+1}/{max_iter} | Best MAPE Fitness: {alpha_score:.6f}")

    best_C       = 10**alpha_pos[0]
    best_epsilon = 10**alpha_pos[1]
    best_gamma   = 10**alpha_pos[2]
    print(f"\n[Seed {seed}] Optimized Params: C={best_C:.4f}, "
          f"epsilon={best_epsilon:.6f}, gamma={best_gamma:.6f}")

    # FINAL MODEL TRAINING ---
    model_w = Pipeline([
        ('poly',   PolynomialFeatures(degree=2, include_bias=False)),
        ('scaler', StandardScaler()),
        ('svr',    SVR(kernel='rbf', C=best_C,
                       epsilon=best_epsilon, gamma=best_gamma))
    ]).fit(X_train_val, np.log(y_train_val[:, 0]))

    model_d = Pipeline([
        ('poly',   PolynomialFeatures(degree=2, include_bias=False)),
        ('scaler', StandardScaler()),
        ('svr',    SVR(kernel='rbf', C=best_C,
                       epsilon=best_epsilon, gamma=best_gamma))
    ]).fit(X_train_val, np.log(y_train_val[:, 1]))

    y_preds_width = np.exp(model_w.predict(X_test))
    y_preds_depth = np.exp(model_d.predict(X_test))

    res_width = y_test[:, 0] - y_preds_width
    res_depth = y_test[:, 1] - y_preds_depth

    #METRICS & VISUALIZATION
    def report_results(y_true, y_pred, label):
        residuals = y_true - y_pred
        r2   = r2_score(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae  = mean_absolute_error(y_true, y_pred)
        mape = mean_absolute_percentage_error(y_true, y_pred)
        res_std  = np.std(residuals)
        res_kurt = stats.kurtosis(residuals, fisher=True)

        print(f"\n--- {label} Final Metrics (Seed {seed}) ---")
        print(f"R²: {r2:.4f} | RMSE: {rmse:.4f} μm | MAE: {mae:.4f} μm | MAPE: {mape:.2%}")
        print(f"Res. Std Dev: {res_std:.4f} μm | Res. Kurtosis: {res_kurt:.4f}")

        fig, ax = plt.subplots(2, 2, figsize=(12, 10))

        ax[0, 0].scatter(y_pred, residuals, alpha=0.7, edgecolors='k', color='#4A90E2')
        ax[0, 0].axhline(0, color='r', linestyle='--')
        ax[0, 0].set_title(f'{label}: Residuals vs Predicted')
        ax[0, 0].set_xlabel('Predicted [μm]')
        ax[0, 0].set_ylabel('Residual [μm]')

        probplot(residuals, dist="norm", plot=ax[0, 1])
        ax[0, 1].set_title(f'{label}: Normal Q-Q Plot')

        ax[1, 0].scatter(y_true, y_pred, alpha=0.7, edgecolors='k', color='#50C878')
        line_lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
        ax[1, 0].plot(line_lims, line_lims, 'r--', lw=2)
        ax[1, 0].set_title(f'{label}: Actual vs Predicted')
        ax[1, 0].set_xlabel('Experimental [μm]')
        ax[1, 0].set_ylabel('Predicted [μm]')

        sns.histplot(residuals, bins=15, kde=True, ax=ax[1, 1],
                     color='#5C5CFF', edgecolor='black', alpha=0.6)
        ax[1, 1].text(0.05, 0.90, f'Kurtosis: {res_kurt:.4f}',
                      transform=ax[1, 1].transAxes,
                      bbox=dict(facecolor='white', alpha=0.8))
        ax[1, 1].set_title(f'{label}: Residual Distribution')

        plt.tight_layout()
        if seed == 42:
            plt.savefig(f'SVR_{label.replace(" ", "_")}_seed42_plots.png',
                        dpi=300, bbox_inches='tight')
        plt.show()

    report_results(y_test[:, 0], y_preds_width, "Width")
    report_results(y_test[:, 1], y_preds_depth, "Depth")

    if seed == 42:
        return (y_test, y_preds_width, y_preds_depth, convergence_curve,
                res_width, res_depth, model_w, model_d)
    else:
        return (y_test, y_preds_width, y_preds_depth, convergence_curve,
                res_width, res_depth, None, None)


def run_multi_seed_experiment_svr(df, seed_list, selection_mode):
    all_res_width, all_res_depth = [], []
    results_list, conv_history   = [], {}
    seed42_models = {'width': None, 'depth': None}

    # DATA SELECTION 
    if selection_mode == 's1':
        X_f, y_f = df.iloc[:, :3].values, df.iloc[:, -2:].values
    elif selection_mode == 's2':
        X_f, y_f = df.iloc[:, [3,4,5]].values, df.iloc[:, -2:].values
    elif selection_mode == 's3':
        X_f, y_f = df.iloc[:, [0,1,5]].values, df.iloc[:, -2:].values
    elif selection_mode == 's4':
        X_f, y_f = df.iloc[:, [1,4,5]].values, df.iloc[:, -2:].values
    else:
        raise ValueError("Mode Error")

    idx_map      = {'s1': list(range(3)), 's2': [3,4,5], 's3': [0,1,5], 's4': [1,4,5]}
    input_names  = df.columns[idx_map[selection_mode]].tolist()
    output_names = df.columns[-2:].tolist()

    print("\n" + "Selected Configuration ".center(50, "="))
    print(f"Mode         : {selection_mode}")
    print(f"Input Vars   : {input_names}")
    print(f"Output Vars  : {output_names}")
    print("="*50 + "\n")

    # SEED LOOP
    for s in seed_list:
        y_true, p_w, p_d, curve, res_w, res_d, m_w, m_d = run_gwo_svr_pipeline(X_f, y_f, s)
        conv_history[s] = curve
        all_res_width.append(res_w)
        all_res_depth.append(res_d)

        if s == 42:
            seed42_models['width'] = m_w
            seed42_models['depth'] = m_d

        w_mae = mean_absolute_error(y_true[:, 0], p_w)
        d_mae = mean_absolute_error(y_true[:, 1], p_d)

        plt.figure(figsize=(8, 4))
        plt.plot(curve, color='#FF5733', linewidth=2, marker='o', markersize=4)
        plt.title(f'GWO-SVR Convergence Curve (Seed {s})')
        plt.xlabel('Iteration')
        plt.ylabel('Best MAPE (Fitness)')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.show()

        results_list.append({
            'Seed':   s,
            'W_R2':   r2_score(y_true[:, 0], p_w),
            'W_RMSE': np.sqrt(mean_squared_error(y_true[:, 0], p_w)),
            'W_MAE':  w_mae,
            'W_MAPE': mean_absolute_percentage_error(y_true[:, 0], p_w),
            'D_R2':   r2_score(y_true[:, 1], p_d),
            'D_RMSE': np.sqrt(mean_squared_error(y_true[:, 1], p_d)),
            'D_MAE':  d_mae,
            'D_MAPE': mean_absolute_percentage_error(y_true[:, 1], p_d)
        })

    # CONVERGENCE PLOT 
    plt.figure(figsize=(10, 6))
    for s, curve in conv_history.items():
        plt.plot(curve, label=f'Seed {s}', alpha=0.8)
    plt.title(f'GWO-SVR Convergence Summary - {selection_mode}')
    plt.legend()
    plt.grid(True)
    plt.show()

    # SUMMARY TABLE
    all_results_df = pd.DataFrame(results_list)
    summary = all_results_df.drop(columns='Seed').agg(['mean', 'std']).T
    print("\n" + "="*50)
    print(f"FINAL STATS OVER ALL SEEDS ({selection_mode})")
    print("="*50)
    print(summary)
    print("="*50)

    # POOLED RESIDUALS
    pooled_res_width = np.concatenate(all_res_width)
    pooled_res_depth = np.concatenate(all_res_depth)
    print(f"Pooled Residual Std  — Width: {np.std(pooled_res_width):.4f} μm | Depth: {np.std(pooled_res_depth):.4f} μm")
    print(f"Pooled Residual Mean — Width: {np.mean(pooled_res_width):.4f} μm | Depth: {np.mean(pooled_res_depth):.4f} μm")

    return all_results_df, summary, seed42_models, X_f, y_f



df_main = pd.read_csv('data.csv')
seeds = [405, 166, 327, 347, 589]
raw_df, summary_df, seed42_svr, X_f_svr, y_f_svr = run_multi_seed_experiment_svr(df_main, seeds, 's1')
