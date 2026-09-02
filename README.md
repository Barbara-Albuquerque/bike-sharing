# Melhorias Sem TI

Projeto de análise e modelagem preditiva com o dataset `SeoulBikeData_original.csv`.
O objetivo é prever `Rented Bike Count`, usando pré-processamento dos dados,
engenharia de variáveis temporais e comparação entre modelos de regressão.

## Base de Dados

- `SeoulBikeData_original.csv`: base original com 8760 registros horários do sistema de bicicletas de Seul.
- `data-processing/imputed-values/SeoulBikeData_clean_imputed.csv`: base tratada gerada pelo script de processamento.

## Requisitos

O projeto usa Python com as seguintes bibliotecas:

```bash
pip install pandas numpy scikit-learn scipy matplotlib seaborn
```

## Pipeline

O fluxo do projeto começa com a reprodução de um modelo base usando Random Forest
diretamente sobre as variáveis originais disponíveis no dataset. Esse baseline serve como
ponto de comparação para os demais experimentos.

Depois disso, os dados passam por uma etapa de processamento, em que são tratados os
registros de indisponibilidade operacional, a variável `Seasons` é codificada e novas
features temporais são criadas a partir da data: `Weekday`, `Month` e `Day`.

Com a base tratada, são testados outros modelos de regressão. Para cada abordagem,
primeiro é feita uma busca em grade com validação cruzada 10-fold para selecionar
hiperparâmetros. Em seguida, os modelos finais são treinados e avaliados em 10 execuções
com divisão 70/30 entre treino e teste.

Os resultados finais são comparados usando `R`, `RMSE` e `MAE`, além de gráficos de
predição e importância das variáveis.

## Estrutura do Projeto

### `data-processing/`

Scripts de preparação dos dados, estatísticas auxiliares e geração de gráficos.

- `data-process.py`: lê a base original, identifica registros com `Rented Bike Count = 0`, cria `Weekday`, `Month` e `Day`, imputa valores em dias não operacionais, codifica `Seasons` e salva a base tratada.
- `imputed-values/`: arquivos gerados pelo processamento, incluindo a base imputada e as linhas originalmente zeradas.
- `desvios.py/desvios.py`: calcula média e desvio padrão das métricas dos modelos a partir dos CSVs de resultado.
- `graficos/graficos.py`: gera gráfico exploratório de média de bicicletas alugadas por dia do mês.
- `graficos/plots_preds.py`: gera comparação visual entre valores reais e previstos para RF, SVR e HGBR.
- `graficos/figures/`: imagens geradas pelos scripts de gráficos.

### `rf-base/`

Modelo baseline com `RandomForestRegressor`.

- `rf_base.py`: treina e avalia o Random Forest base em 10 execuções com divisão 70/30, calcula `R`, `RMSE` e `MAE`, salva previsões, médias, métricas por execução e importância das variáveis.
- `results/`: CSVs e figuras gerados pelo modelo baseline.

### `rf-melhorado/`

Modelo Random Forest com base tratada e conjunto ampliado de features.

- `kfold/cv_rf.py`: executa validação cruzada 10-fold para seleção de hiperparâmetros do Random Forest.
- `kfold/results_step1_cv/`: resultados da validação cruzada, melhores combinações e análises agrupadas por hiperparâmetro.
- `results/final_rf.py`: treina e avalia o Random Forest com os hiperparâmetros selecionados, usando 10 execuções com divisão 70/30.
- `results/final-results/`: métricas, previsões e importância das variáveis do modelo final.

### `svm/`

Experimentos com `SVR`.

- `kfold/cv_svm.py`: executa validação cruzada 10-fold para seleção dos hiperparâmetros do SVR.
- `kfold/results_cv_svm/`: resultados completos da busca, melhores parâmetros e rankings por métrica.
- `results/final_svm.py`: treina e avalia o SVR final em 10 execuções com divisão 70/30, salvando métricas, previsões e importância por permutação.
- `results/svr_results/`: saídas finais do SVR.

### `gb/`

Experimentos com `HistGradientBoostingRegressor`.

- `kfold/cv_gb.py`: executa validação cruzada 10-fold para seleção dos hiperparâmetros do HGBR.
- `kfold/gb_results/`: resultados completos da busca, melhores parâmetros e rankings por métrica.
- `results/final_gb.py`: treina e avalia o HGBR final em 10 execuções com divisão 70/30, salvando métricas, previsões e importância por permutação.
- `results/hgbr_results/featimportance.py`: gera uma versão formatada do gráfico de importância das variáveis.
- `results/hgbr_results/`: saídas finais do HGBR.

## Como Executar

Execute os scripts a partir da raiz do repositório.

### 1. Processar os dados

```bash
python data-processing/data-process.py
```

### 2. Rodar o baseline Random Forest

```bash
python rf-base/rf_base.py
```

### 3. Rodar validação cruzada

```bash
python rf-melhorado/kfold/cv_rf.py
python svm/kfold/cv_svm.py
python gb/kfold/cv_gb.py
```

### 4. Rodar os modelos finais

```bash
python rf-melhorado/results/final_rf.py
python svm/results/final_svm.py
python gb/results/final_gb.py
```

### 5. Gerar gráficos auxiliares

```bash
python data-processing/graficos/graficos.py
python data-processing/graficos/plots_preds.py
python gb/results/hgbr_results/featimportance.py
```

## Saídas

Os scripts geram arquivos `.csv` com métricas, médias, previsões e rankings de hiperparâmetros,
além de figuras `.png` com gráficos de desempenho e importância das variáveis.

As principais saídas ficam em:

- `rf-base/results/`
- `rf-melhorado/kfold/results_step1_cv/`
- `rf-melhorado/results/final-results/`
- `svm/kfold/results_cv_svm/`
- `svm/results/svr_results/`
- `gb/kfold/gb_results/`
- `gb/results/hgbr_results/`
- `data-processing/graficos/figures/`
