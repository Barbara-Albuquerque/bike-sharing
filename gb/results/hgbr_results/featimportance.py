import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Leitura do CSV
df = pd.read_csv(
    "gb/results/hgbr_results/hgbr_feature_importance_permutation.csv"
)

if "column00" in df.columns:
    df = df.drop(columns=["column00"])

df = df[df.iloc[:, 0] != "Média"]


# 2. Importâncias médias
mean_importance = df.mean(numeric_only=True)
mean_importance = mean_importance.sort_values(ascending=True)


# 3. Nomes das variáveis
rename_dict = {
    "Hour": "Hour",
    "Temperature(°C)": "Temperature (°C)",
    "Humidity(%)": "Humidity (%)",
    "Wind speed (m/s)": "Wind speed (m/s)",
    "Visibility (10m)": "Visibility (10m)",
    "Dew point temperature(°C)": "Dew point temperature (°C)",
    "Solar Radiation (MJ/m2)": "Solar Radiation (MJ/m²)",
    "Rainfall(mm)": "Rainfall (mm)",
    "Snowfall (cm)": "Snowfall (cm)",
    "Month": "Month",
    "Day": "Day",
    "Weekday": "Weekday",
    "Seasons": "Seasons"
}

mean_importance.index = [
    rename_dict.get(col, col)
    for col in mean_importance.index
]


# 4. Gráfico
plt.figure(figsize=(12, 7))

bars = plt.barh(
    mean_importance.index,
    mean_importance.values
)

ax = plt.gca()
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.grid(
    axis="x",
    linestyle="--",
    alpha=0.4
)

plt.xlabel(
    "Aumento Médio do MSE após Permutação",
    fontsize=16
)

plt.ylabel(
    "Variáveis",
    fontsize=16
)

plt.tick_params(
    axis="both",
    labelsize=14
)


# 5. Valores
for bar in bars:
    width = bar.get_width()

    plt.text(
        width + (max(mean_importance.values) * 0.005),
        bar.get_y() + bar.get_height() / 2,
        f"{width:,.0f}",
        va="center",
        fontsize=12
    )

plt.tight_layout()

plt.savefig(
    "hgbr_feature_importance_beautiful.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
