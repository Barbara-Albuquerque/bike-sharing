# ============================================
# 📦 IMPORTS
# ============================================
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")
# colors = sns.color_palette("bright", 7)

# ============================================
# 📁 CONFIGURAÇÕES
# ============================================
DATA_PATH = "SeoulBikeData_original.csv" 
OUTPUT_DIR = "data-processing/graficos/figures"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================
# 📥 CARREGAMENTO DOS DADOS
# ============================================
df = pd.read_csv(DATA_PATH, encoding='utf-8')


# ============================================
# 🔧 PRÉ-PROCESSAMENTO BÁSICO
# ============================================

# Ajustar nomes das colunas (caso tenham espaços)
df.columns = df.columns.str.strip()

# Converter coluna de data
df["Date"] = pd.to_datetime(df["Date"], format="%d/%m/%Y")

# Criar variáveis temporais
df["Weekday"] = df["Date"].dt.dayofweek  # 0=Seg, ..., 6=Dom
df["Month"] = df["Date"].dt.month
df["Day"] = df["Date"].dt.day


# ============================================
# 📊 1) MÉDIA DE ALUGUÉIS POR HORA E DIA
# ============================================
# ============================================
# 📊 1) MÉDIA DE ALUGUÉIS POR HORA E DIA
# ============================================

# # Agrupar
# hourly_mean = (
#     df.groupby(["Weekday", "Hour"])["Rented Bike Count"]
#     .mean()
# )

# # Criar grade completa (7 dias x 24 horas)
# full_index = pd.MultiIndex.from_product(
#     [range(7), range(24)],
#     names=["Weekday", "Hour"]
# )

# # Reindexar para garantir todas as combinações
# hourly_mean = hourly_mean.reindex(full_index).reset_index()

# # Mapear nomes
# weekday_map = {
#     0: "Segunda",
#     1: "Terça",
#     2: "Quarta",
#     3: "Quinta",
#     4: "Sexta",
#     5: "Sábado",
#     6: "Domingo"
# }

# hourly_mean["Weekday_Name"] = hourly_mean["Weekday"].map(weekday_map)
# # Plot
# plt.figure(figsize=(12, 7))

# for i, day in enumerate(weekday_map.values()):
#     subset = hourly_mean[hourly_mean["Weekday_Name"] == day]
#     plt.plot(
#         subset["Hour"],
#         subset["Rented Bike Count"],
#         marker='o',
#         markersize=4,
#         linewidth=2,
#         # color=colors[i],  
#         label=day
#     )

# # plt.title("Média de aluguéis por hora\npara cada dia da semana")
# plt.xlabel("Hora do Dia (0–23)", fontsize=14)
# plt.ylabel("Média de Bicicletas Alugadas", fontsize=14)
# plt.tick_params(axis='both', labelsize=14)
# plt.legend(fontsize=14)
# plt.grid(True)

# plt.tight_layout()
# ax = plt.gca()
# ax.set_xticks(np.arange(0, 24, 1))
# ax.set_xlim(0, 23)
# ax.set_xlabel
# plt.savefig(os.path.join(OUTPUT_DIR, "media_por_hora_semtitulo.png"), dpi=300)
# plt.close()


# # ============================================
# # 📊 2) MATRIZ DE CORRELAÇÃO
# # ============================================

# # Selecionar variáveis numéricas
# numeric_cols = [
#     "Hour",
#     "Temperature(°C)",
#     "Humidity(%)",
#     "Wind speed (m/s)",
#     "Visibility (10m)",
#     "Dew point temperature(°C)",
#     "Solar Radiation (MJ/m2)",
#     "Snowfall (cm)",
#     "Rainfall(mm)",
#     "Seasons",
#     "Weekday",
#     "Day",
#     "Month",
#     "Rented Bike Count"
# ]

# corr_matrix = df[numeric_cols].corr()

# plt.figure(figsize=(12, 10))
# sns.heatmap(
#     corr_matrix,
#     annot=False,
#     cmap="viridis",
#     linewidths=0.5
# )

# plt.title("Matriz de Correlação")
# plt.tight_layout()
# plt.savefig(os.path.join(OUTPUT_DIR, "correlation_matrix.png"), dpi=300)
# plt.close()

# ============================================
# 📊 2) MÉDIA DE ALUGUÉIS POR DIA DO MÊS
# ============================================

# Agrupar por dia do mês
daily_mean = (
    df.groupby("Day")["Rented Bike Count"]
    .mean()
    .reset_index()
)

# Plot
plt.figure(figsize=(12, 7))

plt.plot(
    daily_mean["Day"],
    daily_mean["Rented Bike Count"],
    marker='o',
    markersize=5,
    linewidth=2.5,
    color="#059669"
)

# plt.title("Média de aluguéis por dia do mês")

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