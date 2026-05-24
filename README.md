# House Prices — Repositório de Desenvolvimento

Repositório de desenvolvimento da competição de Machine Learning de Previsão de Preços de Imóveis (AV2 — Ciências de Dados).

Contém a análise exploratória completa, o ciclo de modelagem e todos os arquivos necessários para reproduzir os experimentos localmente.

---

## Estrutura

```
repositorio-desenvolvimento/
├── EDA_final.ipynb                  # Análise Exploratória de Dados (EDA)
├── Acompanhamento2_modelagem.ipynb  # Modelagem, Feature Engineering e Avaliação
├── pipeline.py                      # Função prever_precos (cópia do repositório de submissão)
├── requirements.txt                 # Dependências do projeto
├── treino.csv                       # Base de dados de treino (com SalePrice)
├── teste_publico.csv                # Base de teste pública (sem SalePrice)
├── data_description.txt             # Dicionário de dados das 79 variáveis
└── metricas_baseline.txt            # Métricas do modelo baseline do professor
```

---

## Como reproduzir

### 1. Instale as dependências

```bash
pip install -r requirements.txt
```

### 2. Execute os notebooks em ordem

| Ordem | Notebook | Conteúdo |
|---|---|---|
| 1 | `EDA_final.ipynb` | Análise exploratória: distribuição de SalePrice, correlações, nulos, outliers e compatibilidade com a base de teste |
| 2 | `Acompanhamento2_modelagem.ipynb` | Feature engineering, pré-processamento, comparação de algoritmos, busca de hiperparâmetros e salvamento do modelo |

Ao rodar o notebook 2 até o final, o arquivo `modelo_final.joblib` será gerado na mesma pasta.

---

## Resultados

### Comparação de modelos (holdout 20%, `random_state=42`)

| Modelo | Tempo treino (s) | RMSLE | RMSE ($) | MAE ($) | R² |
|---|---:|---:|---:|---:|---:|
| **Ridge** | 0.19 | **0.11249** | 20.476 | 13.814 | 0.93465 |
| Gradient Boosting | 4.89 | 0.11911 | 22.215 | 14.352 | 0.92308 |
| Random Forest | 4.64 | 0.13015 | 25.233 | 16.572 | 0.90077 |
| XGBoost | ~3.5 | ~0.12 | — | — | — |

### Modelo final escolhido: Ridge Regression (L2)

| Métrica | Valor |
|---|---|
| RMSLE (CV 5-fold) | 0.11219 |
| RMSLE (holdout 20%) | 0.11249 |
| MAE | ~$ 13.800 |
| R² | ~0.934 |
| Melhora sobre o baseline do professor | ~36% |

O baseline do professor (Regressão Linear simples) obteve RMSLE de 0.17543.

---

## Decisões técnicas

- **Remoção de outliers:** 2 imóveis com `GrLivArea > 4.000 sqft` e `SalePrice < $300.000` foram removidos apenas do treino — o pipeline de predição não filtra o teste.
- **Feature engineering:** 8 variáveis derivadas criadas a partir da EDA (`TotalSF`, `HouseAge`, `RemodAge`, `WasRemodeled`, `TotalBaths`, `HasPool`, `HasFireplace`, `HasGarage`).
- **Pré-processamento:** imputação pela mediana (numéricas), imputação com "Nenhum" (categóricas), `StandardScaler`, `OneHotEncoder(handle_unknown="ignore")` — tudo dentro de `Pipeline` e `ColumnTransformer` para evitar data leakage.
- **Transformação do alvo:** `log1p` no treino / `expm1` nas predições via `TransformedTargetRegressor`, alinhando a otimização do modelo com a métrica RMSLE.
- **Busca de hiperparâmetros:** `GridSearchCV` com 9 valores de `alpha` para o Ridge, validação cruzada 5-fold, `random_state=42`.
- **Reprodutibilidade:** `RANDOM_STATE = 42` aplicado em todos os modelos e splits.

---

## Impacto para o negócio

O modelo final erra o preço das casas, em média, em **$ 13.800 para mais ou para menos** (MAE). Para uma casa de valor mediano (~$ 165.000), a estimativa típica fica entre $ 146.500 e $ 183.500 — uma margem de ~11%, adequada para precificação inicial de imóveis residenciais.
