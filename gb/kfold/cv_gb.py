import os
import pandas as pd
import numpy as np

from itertools import product
from scipy.stats import pearsonr

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error


# ======================================================
# Configuração de saída
# ======================================================
output_dir = "gb\kfold\gb_results"
os.makedirs(output_dir, exist_ok=True)


# ======================================================
# 1. Carregamento dos dados
# ======================================================
df = pd.read_csv("data-processing\imputed-values\SeoulBikeData_clean_imputed.csv", encoding="utf-8")

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


# ======================================================
# 2. Divisão treino/teste e normalização
# ======================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, shuffle=True
)

scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# ======================================================
# 3. Grade de hiperparâmetros 
# ======================================================
param_grid = {
    "loss": ["squared_error"],
    "learning_rate": [0.03, 0.05, 0.1],
    "max_leaf_nodes": [15, 31, 63],
    "min_samples_leaf": [10, 20, 30],
    "max_features": [0.6, 0.8, 1.0]
}

param_combinations = list(product(*param_grid.values()))


# ======================================================
# 4. Validação cruzada (Step 1)
# ======================================================
results = []
print("Iniciando...")
kf = KFold(n_splits=10, shuffle=True)

for loss, lr, max_leaf, min_leaf, max_feat in param_combinations:

    r_list, rmse_list, mae_list = [], [], []

    for train_idx, val_idx in kf.split(X_train_scaled):

        X_tr = X_train_scaled[train_idx]
        X_val = X_train_scaled[val_idx]

        y_tr = y_train.iloc[train_idx]
        y_val = y_train.iloc[val_idx]

        model = HistGradientBoostingRegressor(
            loss=loss,
            learning_rate=lr,
            max_leaf_nodes=max_leaf,
            min_samples_leaf=min_leaf,
            max_features=max_feat,
            max_iter=1000,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=30,
        )

        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_val)

        r, _ = pearsonr(y_val, y_pred)
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        mae = mean_absolute_error(y_val, y_pred)

        r_list.append(r)
        rmse_list.append(rmse)
        mae_list.append(mae)

    results.append({
        "params": {
            "loss": loss,
            "learning_rate": lr,
            "max_leaf_nodes": max_leaf,
            "min_samples_leaf": min_leaf,
            "max_features": max_feat,
        },
        "R_mean": np.mean(r_list),
        "RMSE_mean": np.mean(rmse_list),
        "MAE_mean": np.mean(mae_list),
    })


# ======================================================
# 5. Seleção da melhor combinação
# ======================================================
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

pd.DataFrame([best]).to_csv(
    os.path.join(output_dir, "cv_best_params.csv"),
    index=False
)


# ======================================================
# 6. Resultados completos
# ======================================================
df_results = pd.DataFrame(results)
df_expanded = pd.json_normalize(df_results.to_dict(orient="records"))

df_expanded.to_csv(
    os.path.join(output_dir, "cv_all_results.csv"),
    index=False
)


# ======================================================
# 7. Top 5 por métrica
# ======================================================
df_expanded.sort_values(by="R_mean", ascending=False).head(5).to_csv(
    os.path.join(output_dir, "cv_top5_by_R.csv"),
    index=False
)

df_expanded.sort_values(by="RMSE_mean").head(5).to_csv(
    os.path.join(output_dir, "cv_top5_by_RMSE.csv"),
    index=False
)

df_expanded.sort_values(by="MAE_mean").head(5).to_csv(
    os.path.join(output_dir, "cv_top5_by_MAE.csv"),
    index=False
)


# ======================================================
# 8. Análise agrupada por hiperparâmetro
# ======================================================
for param in [
    "params.loss",
    "params.learning_rate",
    "params.max_leaf_nodes",
    "params.min_samples_leaf",
    "params.max_features",
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
