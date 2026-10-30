import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from scipy.stats import kurtosis, probplot
import seaborn as sns
import random
import os



def set_global_determinism(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

set_global_determinism(42)


def run_gwo_ridge_pipeline(X_raw, y_raw, seed):
   
    np.random.seed(seed)

   
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X_raw, y_raw, test_size=0.20, random_state=seed
    )

    #  FITNESS FUNCTION (MAPE) 
    def fitness_function_ridge(params, X_data, y_data):
        kf = KFold(n_splits=5, shuffle=True, random_state=seed)
        alpha_w, alpha_d = 10**params[0], 10**params[1]
        fold_errors = []

        for train_idx, test_idx in kf.split(X_data):
            X_tr, X_val = X_data[train_idx], X_data[test_idx]
            y_tr, y_val = y_data[train_idx], y_data[test_idx]

        
            model_w = Pipeline([
                ('poly', PolynomialFeatures(degree=2, include_bias=False)),
                ('scaler', StandardScaler()),
                ('ridge', Ridge(alpha=alpha_w))
            ]).fit(X_tr, y_tr[:, 0])

          
            model_d = Pipeline([
                ('poly', PolynomialFeatures(degree=2, include_bias=False)),
                ('scaler', StandardScaler()),
                ('ridge', Ridge(alpha=alpha_d))
            ]).fit(X_tr, y_tr[:, 1])

          
            pred_w = model_w.predict(X_val)
            pred_d = model_d.predict(X_val)

            mape_w = mean_absolute_percentage_error(y_val[:, 0], pred_w)
            mape_d = mean_absolute_percentage_error(y_val[:, 1], pred_d)

            fold_errors.append(np.mean([mape_w, mape_d]))

        return np.mean(fold_errors)

    # GWO LOOP 
    num_wolves, max_iter, dim = 12, 25, 2
    lb, ub = -5 * np.ones(dim), 1.0 * np.ones(dim)

    Positions = np.random.uniform(lb, ub, (num_wolves, dim))
    Alpha_pos, Beta_pos, Delta_pos = np.zeros(dim), np.zeros(dim), np.zeros(dim)
    Alpha_score, Beta_score, Delta_score = float("inf"), float("inf"), float("inf")

    print(f"\n[Seed {seed}] Executing GWO...")
    convergence_curve = []

    for l in range(max_iter):
        for i in range(num_wolves):
            Positions[i, :] = np.clip(Positions[i, :], lb, ub)
            fitness = fitness_function_ridge(Positions[i, :], X_train_val, y_train_val)

            if fitness < Alpha_score:
                Alpha_score, Alpha_pos = fitness, Positions[i, :].copy()
            elif fitness < Beta_score:
                Beta_score, Beta_pos = fitness, Positions[i, :].copy()
            elif fitness < Delta_score:
                Delta_score, Delta_pos = fitness, Positions[i, :].copy()

        a = 2 - l * (2 / max_iter)
        for i in range(num_wolves):
            for j in range(dim):
                r1, r2 = np.random.rand(), np.random.rand()
                A1, C1 = 2*a*r1-a, 2*r2
                D_alpha = abs(C1*Alpha_pos[j] - Positions[i, j])
                X1 = Alpha_pos[j] - A1*D_alpha

                r1, r2 = np.random.rand(), np.random.rand()
                A2, C2 = 2*a*r1-a, 2*r2
                D_beta = abs(C2*Beta_pos[j] - Positions[i, j])
                X2 = Beta_pos[j] - A2*D_beta

                r1, r2 = np.random.rand(), np.random.rand()
                A3, C3 = 2*a*r1-a, 2*r2
                D_delta = abs(C3*Delta_pos[j] - Positions[i, j])
                X3 = Delta_pos[j] - A3*D_delta

                Positions[i, j] = (X1 + X2 + X3) / 3

        convergence_curve.append(Alpha_score)
        if (l+1) % 1 == 0:
            print(f"Iteration {l+1}/{max_iter} | Best MAPE Fitness: {Alpha_score:.6f}")

    #FINAL MODEL TRAINING 
    opt_alphas = 10**Alpha_pos

    
    model_w = Pipeline([('poly', PolynomialFeatures(2, include_bias=False)), ('scaler', StandardScaler()), ('ridge', Ridge(alpha=opt_alphas[0]))]).fit(X_train_val, y_train_val[:, 0])
    model_d = Pipeline([('poly', PolynomialFeatures(2, include_bias=False)), ('scaler', StandardScaler()), ('ridge', Ridge(alpha=opt_alphas[1]))]).fit(X_train_val, y_train_val[:, 1])


    y_preds_width = model_w.predict(X_test)
    y_preds_depth = model_d.predict(X_test)

    res_width = y_test[:, 0] - y_preds_width
    res_depth = y_test[:, 1] - y_preds_depth

    # METRICS & VISUALIZATION 
    def report_results(y_true, y_pred, label):
        residuals = y_true - y_pred
        r2 = r2_score(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae  = mean_absolute_error(y_true, y_pred)
        mape = mean_absolute_percentage_error(y_true, y_pred)
        res_std, res_kurt = np.std(residuals), kurtosis(residuals)

        print(f"\n--- {label} Final Metrics (Seed {seed}) ---")
        print(f"R²: {r2:.4f} | RMSE: {rmse:.4f} μm | MAE: {mae:.4f} μm | MAPE: {mape:.2%}")
        print(f"Res. Std Dev: {res_std:.4f} μm | Res. Kurtosis: {res_kurt:.4f}")

        fig, ax = plt.subplots(2, 2, figsize=(12, 10))
        ax[0, 0].scatter(y_pred, residuals, alpha=0.7, edgecolors='k', color='#4A90E2')
        ax[0, 0].axhline(0, color='r', linestyle='--')
        ax[0, 0].set_title(f'{label}: Residuals vs Predicted')

        probplot(residuals, dist="norm", plot=ax[0, 1])
        ax[0, 1].set_title(f'{label}: Normal Q-Q Plot')

        ax[1, 0].scatter(y_true, y_pred, alpha=0.7, edgecolors='k', color='#50C878')
        line_lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
        ax[1, 0].plot(line_lims, line_lims, 'r--', lw=2)
        ax[1, 0].set_title(f'{label}: Actual vs Predicted')

        sns.histplot(residuals, bins=15, kde=True, ax=ax[1, 1], color='#5C5CFF', edgecolor='black', alpha=0.6)
        ax[1, 1].text(0.05, 0.90, f'Kurtosis: {res_kurt:.4f}', transform=ax[1, 1].transAxes, bbox=dict(facecolor='white', alpha=0.8))
        ax[1, 1].set_title(f'{label}: Residual Distribution')

        plt.tight_layout()
        plt.show()

    report_results(y_test[:, 0], y_preds_width, "Width")
    report_results(y_test[:, 1], y_preds_depth, "Depth")

    if seed == 42:
        return y_test, y_preds_width, y_preds_depth, convergence_curve, res_width, res_depth, model_w, model_d
    else:
        return y_test, y_preds_width, y_preds_depth, convergence_curve, res_width, res_depth, None, None


def run_multi_seed_experiment(df, seed_list, selection_mode):
    all_res_width, all_res_depth = [], []
    results_list, conv_history = [], {}
    seed42_models = {'width': None, 'depth': None}

 
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

    idx_map = {'s1': list(range(3)), 's2': [3,4,5], 's3': [0,1,5], 's4': [1,4,5]}
    input_names  = df.columns[idx_map[selection_mode]].tolist()
    output_names = df.columns[-2:].tolist()

    print("\n" + "Selected Configuration ".center(50, "="))
    print(f"Mode         : {selection_mode}")
    print(f"Input Vars   : {input_names}")
    print(f"Output Vars  : {output_names}")
    print("="*50 + "\n")

   
    for s in seed_list:
        y_true, p_w, p_d, curve, res_w, res_d, m_w, m_d = run_gwo_ridge_pipeline(X_f, y_f, s)
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
        plt.title(f'GWO Convergence Curve (Seed {s})')
        plt.xlabel('Iteration')
        plt.ylabel('Best MAPE (Fitness)')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.show()

        results_list.append({
            'Seed': s,
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
    plt.title(f'GWO Ridge Convergence Summary - {selection_mode}')
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


df_main = pd.read_csv('data2.csv')
seeds = [405, 166, 327, 347, 589]

raw_df, summary_df, models_42, X_feat, y_target = run_multi_seed_experiment(df_main, seeds, 's1')
