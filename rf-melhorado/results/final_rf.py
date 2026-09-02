import os
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn.utils import shuffle
from sklearn.linear_model import LinearRegression

from scipy.stats import pearsonr
import matplotlib.pyplot as plt


# Configuração
output_dir = "rf-melhorado/results/final-results"
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


# 2. Métricas
r_list, rmse_list, mae_list = [], [], []
r_train_list, rmse_train_list, mae_train_list = [], [], []

preds_rows = []


# 3. Treino e teste
print("\nSTEP 2 – 10 execuções com divisão 70/30\n")

for i in range(10):

    print(f"Execução {i + 1}/10")

    X_shuffled, y_shuffled = shuffle(X, y)

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_shuffled,
        y_shuffled,
        test_size=0.3
    )

    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)

    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=20,
        min_samples_split=2,
        min_samples_leaf=2,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    y_pred_train = model.predict(X_train)

    r_train, _ = pearsonr(y_train, y_pred_train)
    rmse_train = root_mean_squared_error(y_train, y_pred_train)
    mae_train = mean_absolute_error(y_train, y_pred_train)

    print(
        f"[TREINO] R={r_train:.4f}, "
        f"RMSE={rmse_train:.2f} cnt/h, "
        f"MAE={mae_train:.2f} cnt/h"
    )

    y_pred = model.predict(X_test)

    r, _ = pearsonr(y_test, y_pred)
    rmse = root_mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    print(
        f"[TESTE ] R={r:.4f}, "
        f"RMSE={rmse:.2f} cnt/h, "
        f"MAE={mae:.2f} cnt/h"
    )

    preds_rows.append(pd.DataFrame({
        "Execucao": i + 1,
        "Conjunto": "Treino",
        "y_true": np.asarray(y_train),
        "y_pred": np.asarray(y_pred_train)
    }))

    preds_rows.append(pd.DataFrame({
        "Execucao": i + 1,
        "Conjunto": "Teste",
        "y_true": np.asarray(y_test),
        "y_pred": np.asarray(y_pred)
    }))

    r_train_list.append(r_train)
    rmse_train_list.append(rmse_train)
    mae_train_list.append(mae_train)

    r_list.append(r)
    rmse_list.append(rmse)
    mae_list.append(mae)


# 4. Resultados médios
print("\nMÉDIAS APÓS 10 EXECUÇÕES (TREINO):")
print(f"R médio   : {np.mean(r_train_list):.4f}")
print(f"RMSE médio: {np.mean(rmse_train_list):.2f}")
print(f"MAE médio : {np.mean(mae_train_list):.2f}")

print("\nMÉDIAS APÓS 10 EXECUÇÕES (TESTE):")
print(f"R médio   : {np.mean(r_list):.4f}")
print(f"RMSE médio: {np.mean(rmse_list):.2f} cnt/h")
print(f"MAE médio : {np.mean(mae_list):.2f} cnt/h")


# 5. Gráfico predito vs real
def plot_pred_vs_true(y_true, y_pred, title, ax):

    ax.scatter(y_true, y_pred, color="magenta", s=5, label="Dados")
    ax.plot([0, max(y_true)], [0, max(y_true)], "k--", label="Ideal")

    reg = LinearRegression().fit(
        np.array(y_true).reshape(-1, 1),
        y_pred
    )

    fit_line = reg.predict(
        np.array(y_true).reshape(-1, 1)
    )

    ax.plot(y_true, fit_line, "m-", label="Regressão")

    r, _ = pearsonr(y_true, y_pred)
    rmse = root_mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)

    ax.text(
        0.05,
        0.9,
        f"R={r:.4f}\nRMSE={rmse:.2f}\nMAE={mae:.2f}",
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="top"
    )

    ax.set_xlabel("Target values")
    ax.set_ylabel("Output values")
    ax.set_title(title)
    ax.legend()


# 6. Figura treino/teste
fig, axs = plt.subplots(1, 2, figsize=(12, 5))

plot_pred_vs_true(y_train, y_pred_train, "(a) Train data", axs[0])
plot_pred_vs_true(y_test, y_pred, "(b) Test data", axs[1])

plt.tight_layout()

plt.savefig(
    os.path.join(output_dir, "rf_step2_pred_vs_true_train_test.png"),
    dpi=300,
    bbox_inches="tight"
)

plt.close()



# 7. Métricas por execução
df_treino = pd.DataFrame({
    "Execucao": range(1, 11),
    "R": r_train_list,
    "RMSE": rmse_train_list,
    "MAE": mae_train_list
})

df_treino.to_csv(
    os.path.join(output_dir, "rf_step2_treino.csv"),
    index=False
)

df_teste = pd.DataFrame({
    "Execucao": range(1, 11),
    "R": r_list,
    "RMSE": rmse_list,
    "MAE": mae_list
})

df_teste.to_csv(
    os.path.join(output_dir, "rf_step2_teste.csv"),
    index=False
)

print("\nArquivos salvos: rf_step2_treino.csv e rf_step2_teste.csv")


# 8. Médias
df_medias = pd.DataFrame({
    "Conjunto": ["Treino", "Teste"],
    "R_medio": [np.mean(r_train_list), np.mean(r_list)],
    "RMSE_medio": [np.mean(rmse_train_list), np.mean(rmse_list)],
    "MAE_medio": [np.mean(mae_train_list), np.mean(mae_list)]
})

df_medias.to_csv(
    os.path.join(output_dir, "rf_step2_medias.csv"),
    index=False
)

print("Arquivo de médias salvo: rf_step2_medias.csv")


# 9. Previsões
df_preds = pd.concat(preds_rows, ignore_index=True)

df_preds.to_csv(
    os.path.join(output_dir, "rf_step2_preds.csv"),
    index=False
)

print("Arquivo de previsões salvo: rf_step2_preds.csv")

# 10. Importância das variáveis
importances = model.feature_importances_

df_importances = pd.DataFrame({
    "Feature": features,
    "Importance": importances
}).sort_values(by="Importance", ascending=False)

df_importances.to_csv(
    os.path.join(output_dir, "rf_feature_importance.csv"),
    index=False
)

print("Arquivo de feature importance salvo: rf_feature_importance.csv")
# 11. Gráfico de importância
plt.figure(figsize=(8, 5))

plt.barh(
    df_importances["Feature"],
    df_importances["Importance"],
    color="purple"
)

plt.gca().invert_yaxis()
plt.xlabel("Importance")
plt.title("Random Forest Feature Importance")

plt.tight_layout()

plt.savefig(
    os.path.join(output_dir, "rf_feature_importance.png"),
    dpi=300,
    bbox_inches="tight"
)

plt.close()
