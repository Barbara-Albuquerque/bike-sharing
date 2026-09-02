import pandas as pd
import os

# Configuração
PATH = "SeoulBikeData_original.csv"
OUTPUT_DIR = "data-processing/imputed-values"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 1. Leitura do CSV
encodings = ["utf-8", "ISO-8859-1", "latin1", "cp1252"]
last_err = None

for enc in encodings:
    try:
        df = pd.read_csv(PATH, encoding=enc)
        break
    except Exception as e:
        last_err = e
else:
    raise RuntimeError(f"Não consegui ler {PATH}. Último erro: {last_err}")

# 2. Coluna alvo
def achar_coluna_rented(cols):
    for c in cols:
        norm = c.lower().replace("_", " ").replace("-", " ").strip()
        if norm == "rented bike count" or all(w in norm for w in ["rented", "bike", "count"]):
            return c
    raise ValueError(
        f"Não encontrei a coluna 'Rented Bike Count'. Colunas: {list(cols)}"
    )

col_rented = achar_coluna_rented(df.columns)

# 3. Zeros no alvo
rented_num = pd.to_numeric(df[col_rented], errors="coerce")
num_zeros = int((rented_num == 0).sum())
print(f"Linhas com '{col_rented}' == 0: {num_zeros}")

date_col = next((c for c in df.columns if "date" in c.lower()), None)

if date_col:
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce", dayfirst=True)

    zeros = rented_num.eq(0)
    por_data = (
        df.loc[zeros, [date_col]]
          .assign(date=df[date_col].dt.date)
          .groupby("date")
          .size()
          .reset_index(name="qtd_zeros")
          .sort_values("date")
    )

    print(por_data.to_string(index=False))
else:
    print("Coluna de data não encontrada.")

# 4. Linhas com zero
df_zeros = df.loc[rented_num.eq(0)].copy()
df_zeros.to_csv(
    os.path.join(OUTPUT_DIR, "rented_bike_count_zero_rows.csv"),
    index=False
)
print("Salvo em rented_bike_count_zero_rows.csv")

# 5. Coluna de hora
def _norm(s):
    return s.strip().lower().replace("_", " ").replace("-", " ")

hour_col = next((c for c in df.columns if _norm(c) == "hour"), None)
if hour_col is None:
    raise ValueError("Coluna 'Hour' não encontrada no dataset.")

# 6. Tipos
df[date_col]   = pd.to_datetime(df[date_col], errors="coerce", dayfirst=True)
df[hour_col]   = pd.to_numeric(df[hour_col], errors="coerce")
df[col_rented] = pd.to_numeric(df[col_rented], errors="coerce")

# 7. Variáveis temporais
df["Weekday"] = df[date_col].dt.dayofweek.astype("Int64")
df["Month"]   = df[date_col].dt.month.astype("Int64")
df["Day"]     = df[date_col].dt.day.astype("Int64")

# 8. Máscara de imputação
func_col = next(
    (c for c in df.columns if _norm(c) == "functioning day"),
    None
)

if func_col is not None:
    func_norm = df[func_col].astype(str).str.strip().str.lower()
    is_no = func_norm.isin(
        ["no", "n", "0", "false", "non-functioning", "nonfunctioning"]
    )
else:
    is_no = pd.Series(False, index=df.index)

is_zero_or_na = df[col_rented].isna() | (df[col_rented] == 0)
mask_impute = is_no & is_zero_or_na

# 9. Médias para imputação
mean_wh = (
    df.groupby(["Weekday", hour_col], dropna=False)[col_rented]
      .transform("mean")
)

mean_h = (
    df.groupby(hour_col, dropna=False)[col_rented]
      .transform("mean")
)

fill_vals = mean_wh.where(~mean_wh.isna(), mean_h)

# 10. Imputação
df.loc[mask_impute, col_rented] = fill_vals[mask_impute]
df.loc[mask_impute, col_rented] = (
    df.loc[mask_impute, col_rented]
      .round()
      .astype("Int64")
)

# 11. Formatação da data
df[date_col] = (
    pd.to_datetime(df[date_col], errors="coerce", dayfirst=True)
      .dt.strftime("%d/%m/%Y")
)

# 12. Remoção de colunas
to_drop = [
    c for c in df.columns
    if _norm(c) in {"holiday", "functioning day"}
]

df.drop(columns=to_drop, inplace=True, errors="ignore")

# 13. Codificação de Seasons
season_mapping = {
    "Winter": 1,
    "Spring": 2,
    "Summer": 3,
    "Autumn": 4
}

if "Seasons" not in df.columns:
    raise KeyError("Coluna 'Seasons' não encontrada.")

if df["Seasons"].dtype == "O":
    df["Seasons"] = df["Seasons"].map(season_mapping).astype("Int64")
else:
    df["Seasons"] = (
        pd.to_numeric(df["Seasons"], errors="coerce")
          .astype("Int64")
    )

# 14. Saída
out_path = os.path.join(
    OUTPUT_DIR,
    "SeoulBikeData_clean_imputed.csv"
)

df.to_csv(out_path, index=False, encoding="utf-8-sig")
print("Dataset limpo salvo em:", out_path)
