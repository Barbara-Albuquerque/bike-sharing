import pandas as pd
import matplotlib.pyplot as plt

def load_model(path):
    df = pd.read_csv(path)
    df_test = df[
        (df["Conjunto"] == "Teste") &
        (df["Execucao"] == 1)
    ].reset_index(drop=True)
    return df_test.iloc[:72].reset_index(drop=True)


# Carregar dados
df_hgbr = load_model("gb/results/hgbr_results/hgbr_step2_preds.csv")
df_rf   = load_model("rf-base/results/rf_base_step2_preds.csv")
df_svr  = load_model("svm/results/svr_results/svr_step2_preds.csv")


horas = range(72)

plt.style.use("default")
fig, axes = plt.subplots(1, 3, figsize=(18,5), sharey=True)

modelos = [
    ("RF-NGO2022", df_rf),
    ("SVR", df_svr),
    ("HGBR", df_hgbr)
]

for ax, (nome, df_model) in zip(axes, modelos):
    
    ax.plot(horas, df_model["y_true"], linewidth=2, label="Real")
    ax.plot(horas, df_model["y_pred"], linewidth=2, label="Predito")
    
    ax.set_title(nome, fontsize=14)
    ax.set_xlabel("Horas", fontsize=14)
    ax.tick_params(axis='both', labelsize=14)
    ax.grid(alpha=0.3)

axes[0].set_ylabel("Bicicletas alugadas", fontsize=14)

# Colocar legenda só no último subplot
axes[-1].legend()

plt.tight_layout()
plt.savefig("data-processing/graficos/figures/modelos_comparacao_subplots5.png", dpi=300)
plt.close()