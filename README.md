# Melhorias Sem TI

Projeto de análise e modelagem preditiva com o dataset `SeoulBikeData_original.csv`, com foco em pré-processamento dos dados e comparação de abordagens de aprendizado de máquina para prever a variável `Rented Bike Count`.

## Estrutura

- `data-processing/`: scripts de limpeza, imputação e geração de arquivos auxiliares.
- `rf-base/`: implementação base com `RandomForestRegressor`.
- `rf-melhorado/`: variações e resultados da abordagem com Random Forest.
- `svm/`: experimentos e resultados com SVM.
- `gb/`: experimentos e resultados com Gradient Boosting.
- `SeoulBikeData_original.csv`: base de dados usada nos experimentos.

## Requisitos

O projeto usa Python e bibliotecas como:

- `pandas`
- `numpy`
- `scikit-learn`
- `scipy`
- `matplotlib`

Instalação sugerida:

```bash
pip install pandas numpy scikit-learn scipy matplotlib
```

## Como executar

### 1. Pré-processamento dos dados

```bash
python data-processing/data-process.py
```

Esse script:

- lê o arquivo original;
- identifica registros com `Rented Bike Count = 0`;
- realiza imputação em casos específicos;
- salva o dataset tratado em `data-processing/imputed-values/`.

### 2. Modelo Random Forest base

```bash
python rf-base/rf_base.py
```

Esse script:

- treina e avalia o modelo em múltiplas execuções;
- calcula métricas como `R`, `RMSE` e `MAE`;
- salva gráficos e arquivos `.csv` em `rf-base/results/`.

## Saídas

Os resultados gerados pelos scripts ficam, em geral, dentro das próprias pastas de cada abordagem, como:

- `results/`
- `kfold/`
- `imputed-values/`
- `graficos/`

## Observação

Os scripts assumem que o arquivo `SeoulBikeData_original.csv` está na raiz do projeto.
