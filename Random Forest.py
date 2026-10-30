import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import (r2_score, mean_squared_error,
                             mean_absolute_error, mean_absolute_percentage_error)
from scipy.stats import probplot
from joblib import Parallel, delayed
import random
import os


def set_global_determinism(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

set_global_determinism(42)



# SINGLE WOLF EVALUATION  (called in parallel)

def _evaluate_wolf(params, X_data, y_data, fold_indices, seed):
    """
    Evaluate one wolf's fitness using pre-computed fold indices.
    Runs on a single core (n_jobs=1) because parallelism is at the wolf level.
    """
    n_estimators      = int(round(params[0]))
    max_depth         = int(round(params[1]))
    min_samples_split = int(round(params[2]))
    ccp_alpha         = 10 ** params[3]

    fold_errors = []
    for train_idx, val_idx in fold_indices:
        X_tr, X_val = X_data[train_idx], X_data[val_idx]
        y_tr, y_val = y_data[train_idx], y_data[val_idx]

    
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            ccp_alpha=ccp_alpha,
            n_jobs=1,           
            random_state=seed
        ).fit(X_tr, y_tr)     

        y_hat = model.predict(X_val)   # shape (n_val, 2)
        mape_w = mean_absolute_percentage_error(y_val[:, 0], y_hat[:, 0])
        mape_d = mean_absolute_percentage_error(y_val[:, 1], y_hat[:, 1])
        fold_errors.append(np.mean([mape_w, mape_d]))

    return np.mean(fold_errors)


def run_gwo_rf_pipeline(X_raw, y_raw, seed):

  
    np.random.seed(seed)

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X_raw, y_raw, test_size=0.20, random_state=seed
    )

    # FOLD INDICES 
    kf = KFold(n_splits=5, shuffle=True, random_state=seed)
    fold_indices = list(kf.split(X_train_val))   # list of (train_idx, val_idx) tuples

    # GWO-LOOP
    num_wolves, max_iter, dim = 12, 25, 4
    lb = np.array([50,  3,  4, -4.0])
    ub = np.array([150, 8, 12, -2.0])   # n_estimator > max_depth > min_samples_split > ccp_alpha

    Positions = np.random.uniform(lb, ub, (num_wolves, dim))
    alpha_pos, alpha_score = np.zeros(dim), float("inf")
    beta_pos,  beta_score  = np.zeros(dim), float("inf")
    delta_pos, delta_score = np.zeros(dim), float("inf")

    print(f"\n[Seed {seed}] Executing GWO-RF Optimization (parallelized wolves)...")
    convergence_curve = []

    for l in range(max_iter):
        Positions = np.clip(Positions, lb, ub)

        # Evaluate all wolves 
        fitnesses = Parallel(n_jobs=-1, prefer="threads")(
            delayed(_evaluate_wolf)(
                Positions[i], X_train_val, y_train_val, fold_indices, seed
            )
            for i in range(num_wolves)
        )
        

        for i, fitness in enumerate(fitnesses):
            if fitness < alpha_score:
                alpha_score, alpha_pos = fitness, Positions[i].copy()
            elif fitness < beta_score:
                beta_score,  beta_pos  = fitness, Positions[i].copy()
            elif fitness < delta_score:
                delta_score, delta_pos = fitness, Positions[i].copy()

        # GWO position update
        a = 2 - l * (2 / max_iter)
        for i in range(num_wolves):
            for j in range(dim):
                r1, r2 = np.random.rand(), np.random.rand()
                X1 = alpha_pos[j] - (2*a*r1 - a) * abs(2*r2*alpha_pos[j] - Positions[i, j])

                r1, r2 = np.random.rand(), np.random.rand()
                X2 = beta_pos[j]  - (2*a*r1 - a) * abs(2*r2*beta_pos[j]  - Positions[i, j])

                r1, r2 = np.random.rand(), np.random.rand()
                X3 = delta_pos[j] - (2*a*r1 - a) * abs(2*r2*delta_pos[j] - Positions[i, j])

                Positions[i, j] = (X1 + X2 + X3) / 3

        Positions = np.clip(Positions, lb, ub)
        convergence_curve.append(alpha_score)

        if (l + 1) % 1 == 0 or (l + 1) == max_iter:
            print(f"  Iter {l+1:>2}/{max_iter} | Best MAPE: {alpha_score:.6f}")

    best_n   = int(round(alpha_pos[0]))
    best_d   = int(round(alpha_pos[1]))
    best_mss = int(round(alpha_pos[2]))
    best_ccp = 10 ** alpha_pos[3]
    print(f"\n[Seed {seed}] Best params → n_estimators={best_n}, max_depth={best_d}, "
          f"min_samples_split={best_mss}, ccp_alpha={best_ccp:.6f}")

    #FINAL MODEL TRAINING 
       model_w = RandomForestRegressor(
        n_estimators=best_n, max_depth=best_d,
        min_samples_split=best_mss, ccp_alpha=best_ccp,
        n_jobs=-1, random_state=seed          # n_jobs=-1 is fine for final fit
    ).fit(X_train_val, y_train_val[:, 0])

    model_d = RandomForestRegressor(
        n_estimators=best_n, max_depth=best_d,
        min_samples_split=best_mss, ccp_alpha=best_ccp,
        n_jobs=-1, random_state=seed
    ).fit(X_train_val, y_train_val[:, 1])

    y_preds_width = model_w.predict(X_test)
    y_preds_depth = model_d.predict(X_test)

    res_width = y_test[:, 0] - y_preds_width
    res_depth = y_test[:, 1] - y_preds_depth

    #REPORTING & PLOTS 
    def report_results(y_true, y_pred, label):
        residuals = y_true - y_pred
        r2   = r2_score(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae  = mean_absolute_error(y_true, y_pred)
        mape = mean_absolute_percentage_error(y_true, y_pred)
        res_std  = np.std(residuals)
        res_kurt = stats.kurtosis(residuals, fisher=True)

        print(f"\n--- {label} (Seed {seed}) ---")
        print(f"R²: {r2:.4f} | RMSE: {rmse:.4f} μm | MAE: {mae:.4f} μm | MAPE: {mape:.2%}")
        print(f"Res. Std: {res_std:.4f} μm | Res. Kurtosis: {res_kurt:.4f}")

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
            plt.savefig(f'RF_{label.replace(" ", "_")}_seed42_plots.png',
                        dpi=300, bbox_inches='tight')
        plt.show()

    report_results(y_test[:, 0], y_preds_width, "Track Width")
    report_results(y_test[:, 1], y_preds_depth, "Track Depth")

    return y_test, y_preds_width, y_preds_depth, convergence_curve, res_width, res_depth


def run_multi_seed_experiment_rf(df, seed_list, selection_mode):
    all_res_width, all_res_depth = [], []
    results_list, conv_history   = [], {}

    # idx_map
    feature_map = {
        's1': [0, 1, 2],
        's2': [3, 4, 5],
        's3': [0, 1, 5],
        's4': [1, 4, 5],
    }
    if selection_mode not in feature_map:
        raise ValueError(f"Unknown mode: {selection_mode}")

    col_idx  = feature_map[selection_mode]
    X_f      = df.iloc[:, col_idx].values
    y_f      = df.iloc[:, -2:].values

    input_names  = df.columns[col_idx].tolist()
    output_names = df.columns[-2:].tolist()

    print("\n" + " Selected Configuration ".center(50, "="))
    print(f"  Mode       : {selection_mode}")
    print(f"  Inputs     : {input_names}")
    print(f"  Outputs    : {output_names}")
    print("=" * 50 + "\n")

    #SEED LOOP
    for s in seed_list:
        y_true, p_w, p_d, curve, res_w, res_d = run_gwo_rf_pipeline(X_f, y_f, s)
        conv_history[s] = curve
        all_res_width.append(res_w)
        all_res_depth.append(res_d)

        plt.figure(figsize=(8, 4))
        plt.plot(curve, color='#FF5733', linewidth=2, marker='o', markersize=4)
        plt.title(f'GWO-RF Convergence (Seed {s})')
        plt.xlabel('Iteration')
        plt.ylabel('Best MAPE (Fitness)')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.show()

        results_list.append({
            'Seed':   s,
            'W_R2':   r2_score(y_true[:, 0], p_w),
            'W_RMSE': np.sqrt(mean_squared_error(y_true[:, 0], p_w)),
            'W_MAE':  mean_absolute_error(y_true[:, 0], p_w),
            'W_MAPE': mean_absolute_percentage_error(y_true[:, 0], p_w),
            'D_R2':   r2_score(y_true[:, 1], p_d),
            'D_RMSE': np.sqrt(mean_squared_error(y_true[:, 1], p_d)),
            'D_MAE':  mean_absolute_error(y_true[:, 1], p_d),
            'D_MAPE': mean_absolute_percentage_error(y_true[:, 1], p_d),
        })

    # CONVERGENCE PLOT
    plt.figure(figsize=(10, 6))
    for s, curve in conv_history.items():
        plt.plot(curve, label=f'Seed {s}', alpha=0.8)
    plt.title(f'GWO-RF Convergence Summary — {selection_mode}')
    plt.legend()
    plt.grid(True)
    plt.show()

    #SUMMARY TABLE
    all_results_df = pd.DataFrame(results_list)
    summary = all_results_df.drop(columns='Seed').agg(['mean', 'std']).T
    print("\n" + "=" * 50)
    print(f"FINAL STATS OVER ALL SEEDS ({selection_mode})")
    print("=" * 50)
    print(summary)
    print("=" * 50)

    #POOLED RESIDUALS
    pooled_w = np.concatenate(all_res_width)
    pooled_d = np.concatenate(all_res_depth)
    print(f"\nPooled Residual Mean — Width: {np.mean(pooled_w):.4f} μm | Depth: {np.mean(pooled_d):.4f} μm")
    print(f"Pooled Residual Std  — Width: {np.std(pooled_w):.4f}  μm | Depth: {np.std(pooled_d):.4f}  μm")

    return all_results_df, summary


df_main = pd.read_csv('data.csv')
seeds = [405, 166, 327, 347, 589]
raw_df, summary_df = run_multi_seed_experiment_rf(df_main, seeds, 's1')
