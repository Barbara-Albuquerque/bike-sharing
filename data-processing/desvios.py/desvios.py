import pandas as pd

# =========================
# 1. LER OS ARQUIVOS
# =========================

svr = pd.read_csv("svm/results/svr_results/svr_step2_treino.csv")
rf = pd.read_csv("data-processing/desvios.py/rf_base_teste.csv")
hgb = pd.read_csv("gb/results/hgbr_results/hgbr_step2_treino.csv")

# =========================
# 2. FUNÇÃO PARA CALCULAR MÉDIA E DP
# =========================

def calcular_media_dp(df, nome_modelo):
    
    media = df.mean()
    dp = df.std(ddof=1)  # desvio padrão amostral
    
    resumo = pd.DataFrame({
        "Modelo": [nome_modelo],
        "R_media": [round(media["R"], 4)],
        "R_dp": [round(dp["R"], 4)],
        "RMSE_media": [round(media["RMSE"], 2)],
        "RMSE_dp": [round(dp["RMSE"], 2)],
        "MAE_media": [round(media["MAE"], 2)],
        "MAE_dp": [round(dp["MAE"], 2)]
    })
    
    return resumo

# =========================
# 3. CALCULAR PARA CADA MODELO
# =========================

svr_resumo = calcular_media_dp(svr, "SVR")
rf_resumo = calcular_media_dp(rf, "Random Forest")
hgb_resumo = calcular_media_dp(hgb, "Histogram Gradient Boosting")

# =========================
# 4. SALVAR CSV POR MODELO
# =========================

svr_resumo.to_csv("data-processing/desvios.py/svr_resumo_media_dp2.csv", index=False)
rf_resumo.to_csv("data-processing/desvios.py/rfbase_resumo_media_dp.csv", index=False)
hgb_resumo.to_csv("data-processing/desvios.py/hgb_resumo_media_dp2.csv", index=False)

print("Arquivos gerados com sucesso.")