import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from scipy.stats import kurtosis, probplot
import random
import os



def set_global_determinism(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

set_global_determinism(42)


#MODEL ARCHITECTURE
class ResearchMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(ResearchMLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 2)  # [Width, Depth]
        )
    def forward(self, x):
        return self.layers(x)


def run_gwo_mlp_pipeline(X_raw, y_raw, seed):
    # --- 0. SET SEED ---
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)

    # --- 1. DATA PREPARATION ---
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X_raw, y_raw, test_size=0.20, random_state=seed
    )

    # GWO FITNESS FUNCTION
    def fitness_function_mlp(params, X_data, y_data):
        hidden_dim = int(round(params[0]))
        lr = 10**params[1]
        wd = 10**params[2]

        kf = KFold(n_splits=5, shuffle=True, random_state=seed)
        fold_errors = []

        for train_idx, val_idx in kf.split(X_data):
            scaler_x, scaler_y = StandardScaler(), StandardScaler()

            X_tr = torch.FloatTensor(scaler_x.fit_transform(X_data[train_idx]))
            y_tr = torch.FloatTensor(scaler_y.fit_transform(y_data[train_idx]))
            X_val = torch.FloatTensor(scaler_x.transform(X_data[val_idx]))

            model = ResearchMLP(X_data.shape[1], hidden_dim)
            optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=20, factor=0.5)
            criterion = nn.MSELoss()

            model.train()
            for _ in range(200):
                optimizer.zero_grad()
                loss = criterion(model(X_tr), y_tr)
                loss.backward()
                optimizer.step()
                scheduler.step(loss)

            model.eval()
            with torch.no_grad():
                preds_scaled = model(X_val).numpy()
                preds_real = scaler_y.inverse_transform(preds_scaled)

            mape_w = mean_absolute_percentage_error(y_data[val_idx, 0], preds_real[:, 0])
            mape_d = mean_absolute_percentage_error(y_data[val_idx, 1], preds_real[:, 1])
            fold_errors.append(np.mean([mape_w, mape_d]))

        return np.mean(fold_errors)

    # GWO OPTIMIZATION LOOP
    num_wolves, max_iter, dim = 12, 25, 3
    lb = np.array([5,  -4.0, -3.8])
    ub = np.array([25, -1.5, -1.0])

    Positions = np.random.uniform(lb, ub, (num_wolves, dim))
    Alpha_pos, Beta_pos, Delta_pos = np.zeros(dim), np.zeros(dim), np.zeros(dim)
    Alpha_score, Beta_score, Delta_score = float("inf"), float("inf"), float("inf")

    print(f"\n[Seed {seed}] Executing GWO-MLP Optimization...")
    convergence_curve = []

    for l in range(max_iter):
        for i in range(num_wolves):
            Positions[i] = np.clip(Positions[i], lb, ub)
            fitness = fitness_function_mlp(Positions[i], X_train_val, y_train_val)

            if fitness < Alpha_score:
                Alpha_score, Alpha_pos = fitness, Positions[i].copy()
            elif fitness < Beta_score:
                Beta_score, Beta_pos = fitness, Positions[i].copy()
            elif fitness < Delta_score:
                Delta_score, Delta_pos = fitness, Positions[i].copy()

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
        if (l+1) % 1 == 0 or (l+1) == max_iter:
            print(f"Iteration {l+1}/{max_iter} | Best MAPE Fitness: {Alpha_score:.6f}")

    # FINAL MODEL TRAINING
    best_h  = int(round(Alpha_pos[0]))
    best_lr = 10**Alpha_pos[1]
    best_wd = 10**Alpha_pos[2]
    print(f"\n[Seed {seed}] Optimized Params: Neurons={best_h}, LR={best_lr:.5f}, WD={best_wd:.5f}")

    scaler_x, scaler_y = StandardScaler(), StandardScaler()
    X_final_tr = torch.FloatTensor(scaler_x.fit_transform(X_train_val))
    y_final_tr = torch.FloatTensor(scaler_y.fit_transform(y_train_val))

    torch.manual_seed(seed)
    final_model = ResearchMLP(X_raw.shape[1], best_h)
    optimizer   = optim.Adam(final_model.parameters(), lr=best_lr, weight_decay=best_wd)
    scheduler   = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=20, factor=0.5)
    criterion   = nn.MSELoss()

    final_model.train()
    for _ in range(600):
        optimizer.zero_grad()
        loss = criterion(final_model(X_final_tr), y_final_tr)
        loss.backward()
        optimizer.step()
        scheduler.step(loss)

    final_model.eval()
    with torch.no_grad():
        X_test_tensor = torch.FloatTensor(scaler_x.transform(X_test))
        preds_scaled  = final_model(X_test_tensor).numpy()
        y_preds       = scaler_y.inverse_transform(preds_scaled)

    y_preds_width = y_preds[:, 0]
    y_preds_depth = y_preds[:, 1]

    res_width = y_test[:, 0] - y_preds_width
    res_depth = y_test[:, 1] - y_preds_depth

    # METRICS & VISUALIZATION 
    def report_results(y_true, y_pred, label):
        residuals = y_true - y_pred
        r2   = r2_score(y_true, y_pred)
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

    report_results(y_test[:, 0], y_preds_width, "Track Width")
    report_results(y_test[:, 1], y_preds_depth, "Track Depth")

    return y_test, y_preds_width, y_preds_depth, convergence_curve, res_width, res_depth


def run_multi_seed_experiment_mlp(df, seed_list, selection_mode):
    all_res_width, all_res_depth = [], []
    results_list, conv_history = [], {}

    
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

    #SEED LOOP 
    for s in seed_list:
        y_true, p_w, p_d, curve, res_w, res_d = run_gwo_mlp_pipeline(X_f, y_f, s)
        conv_history[s] = curve
        all_res_width.append(res_w)
        all_res_depth.append(res_d)

        w_mae = mean_absolute_error(y_true[:, 0], p_w)
        d_mae = mean_absolute_error(y_true[:, 1], p_d)

        plt.figure(figsize=(8, 4))
        plt.plot(curve, color='#FF5733', linewidth=2, marker='o', markersize=4)
        plt.title(f'GWO-MLP Convergence Curve (Seed {s})')
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
    plt.title(f'GWO-MLP Convergence Summary - {selection_mode}')
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

    return all_results_df, summary


df_main = pd.read_csv('data.csv')
seeds = [405, 166, 327, 347, 589]
raw_df, summary_df = run_multi_seed_experiment_mlp(df_main, seeds, 's1')
