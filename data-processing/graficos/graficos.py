import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")

# Configuração
DATA_PATH = "SeoulBikeData_original.csv"
OUTPUT_DIR = "data-processing/graficos/figures"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# 1. Carregamento dos dados
df = pd.read_csv(DATA_PATH, encoding="utf-8")


# 2. Preprocessamento
df.columns = df.columns.str.strip()
df["Date"] = pd.to_datetime(df["Date"], format="%d/%m/%Y")
df["Weekday"] = df["Date"].dt.dayofweek
df["Month"] = df["Date"].dt.month
df["Day"] = df["Date"].dt.day


# 3. Média por dia do mês
daily_mean = (
    df.groupby("Day")["Rented Bike Count"]
    .mean()
    .reset_index()
)

plt.figure(figsize=(12, 7))

plt.plot(
    daily_mean["Day"],
    daily_mean["Rented Bike Count"],
    marker="o",
    markersize=5,
    linewidth=2.5,
    color="#059669"
)

plt.xlabel("Dia do Mês", fontsize=14)
plt.ylabel("Média de Bicicletas Alugadas", fontsize=14)

plt.xticks(np.arange(1, 32, 1), fontsize=12)
plt.yticks(fontsize=12)

plt.grid(True, linestyle="--", alpha=0.5)
plt.xlim(1, 31)

plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_DIR, "media_por_dia_mes.png"),
    dpi=300
)

plt.close()
print("Figuras geradas com sucesso na pasta 'figures'.")
