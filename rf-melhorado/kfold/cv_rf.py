import os
import pandas as pd
import numpy as np

from itertools import product
from scipy.stats import pearsonr

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, root_mean_squared_error


# Configuração
output_dir = "rf-melhorado/kfold/results_step1_cv"
os.makedirs(output_dir, exist_ok=True)


# 1. Dados
df = pd.read_csv("data-processing/imputed-values/SeoulBikeData_clean_imputed.csv", encoding="utf-8")

features = [
    "Hour",
    "Temperature(°C)",
    "Humidity(%)",
    "Wind speed (m/s)",
    "Visibility (10m)",
    "Dew point temperature(°C)",
    "Solar Radiation (MJ/m2)",
    "Snowfall (cm)",
    "Rainfall(mm)",
    "Seasons",
    "Weekday",
    "Day",
    "Month"
]

target = "Rented Bike Count"

X = df[features]
y = df[target]


# 2. Treino/teste e normalização
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, shuffle=True
)

scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# 3. Hiperparâmetros
param_grid = {
    "n_estimators": [100, 200, 300],
    "max_depth": [None, 20, 10],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 2],
}

param_combinations = list(product(*param_grid.values()))


# 4. Validação cruzada
results = []

kf = KFold(n_splits=10, shuffle=True)

for n_est, max_d, min_split, min_leaf in param_combinations:

    r_list, rmse_list, mae_list = [], [], []

    for train_idx, val_idx in kf.split(X_train_scaled):

        X_tr = X_train_scaled[train_idx]
        X_val = X_train_scaled[val_idx]

        y_tr = y_train.iloc[train_idx]
        y_val = y_train.iloc[val_idx]

        model = RandomForestRegressor(
            n_estimators=n_est,
            max_depth=max_d,
            min_samples_split=min_split,
            min_samples_leaf=min_leaf,
            n_jobs=-1,
        )

        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_val)

        r, _ = pearsonr(y_val, y_pred)
        rmse = root_mean_squared_error(y_val, y_pred)
        mae = mean_absolute_error(y_val, y_pred)

        r_list.append(r)
        rmse_list.append(rmse)
        mae_list.append(mae)

    results.append({
        "params": {
            "n_estimators": n_est,
            "max_depth": max_d,
            "min_samples_split": min_split,
            "min_samples_leaf": min_leaf,
        },
        "R_mean": np.mean(r_list),
        "RMSE_mean": np.mean(rmse_list),
        "MAE_mean": np.mean(mae_list),
    })


# 5. Melhor combinação
best = sorted(
    results,
    key=lambda x: (-x["R_mean"], x["RMSE_mean"], x["MAE_mean"])
)[0]

print("Melhores hiperparâmetros encontrados:")
print(best["params"])
print(
    f"R: {best['R_mean']:.4f}, "
    f"RMSE: {best['RMSE_mean']:.2f}, "
    f"MAE: {best['MAE_mean']:.2f}"
)

best_df = pd.DataFrame([best])
best_df.to_csv(
    os.path.join(output_dir, "cv_best_params.csv"),
    index=False
)


# 6. Resultados completos
df_results = pd.DataFrame(results)
df_expanded = pd.json_normalize(df_results.to_dict(orient="records"))

df_expanded.to_csv(
    os.path.join(output_dir, "cv_all_results.csv"),
    index=False
)


# 7. Top 5
top5_r = df_expanded.sort_values(by="R_mean", ascending=False).head(5)
top5_r.to_csv(
    os.path.join(output_dir, "cv_top5_by_R.csv"),
    index=False
)

top5_rmse = df_expanded.sort_values(by="RMSE_mean").head(5)
top5_rmse.to_csv(
    os.path.join(output_dir, "cv_top5_by_RMSE.csv"),
    index=False
)

top5_mae = df_expanded.sort_values(by="MAE_mean").head(5)
top5_mae.to_csv(
    os.path.join(output_dir, "cv_top5_by_MAE.csv"),
    index=False
)

print("\nTop 5 combinações por R:")
print(top5_r)

print("\nTop 5 combinações por RMSE:")
print(df_expanded.sort_values(by="RMSE_mean", ascending=False).head(5))

print("\nTop 5 combinações por MAE:")
print(df_expanded.sort_values(by="MAE_mean", ascending=False).head(5))


# 8. Agrupamento por parâmetro
for param in [
    "params.n_estimators",
    "params.max_depth",
    "params.min_samples_split",
    "params.min_samples_leaf",
]:
    grouped = (
        df_expanded
        .groupby(param)
        .agg({
            "R_mean": "mean",
            "RMSE_mean": "mean",
            "MAE_mean": "mean",
        })
        .reset_index()
    )

    param_name = param.split(".")[-1]

    grouped.to_csv(
        os.path.join(output_dir, f"cv_grouped_by_{param_name}.csv"),
        index=False
    )
