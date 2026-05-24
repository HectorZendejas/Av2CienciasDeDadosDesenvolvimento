# House Prices — Repositório de Desenvolvimento

Repositório de desenvolvimento para a competição de Machine Learning de Previsão de Preços de Imóveis (AV2 — Ciências de Dados).

## Estrutura

```
├── EDA_final.ipynb              # Análise Exploratória de Dados completa
├── Acompanhamento2_modelagem.ipynb  # Modelagem, Feature Engineering, Avaliação
├── requirements.txt             # Dependências do projeto
├── treino.csv                   # Base de dados de treino
├── teste_publico.csv            # Base de teste pública
├── data_description.txt         # Descrição das variáveis
└── metricas_baseline.txt        # Métricas do modelo baseline
```

## Notebooks

- **EDA_final.ipynb** — Análise exploratória completa: distribuição de SalePrice, correlações, variáveis categóricas, diagnóstico de nulos, outliers e compatibilidade com a base de teste.
- **Acompanhamento2_modelagem.ipynb** — Pipeline completo de modelagem: feature engineering (8 variáveis derivadas), pré-processamento, comparação de 4 algoritmos (Ridge, Random Forest, Gradient Boosting, XGBoost), busca de hiperparâmetros (GridSearchCV), validação cruzada 5-fold e análise de impacto de negócio.

## Modelo Final

| Métrica | Valor |
|---|---|
| Algoritmo | Ridge Regression (L2) |
| RMSLE (CV 5-fold) | 0.11219 |
| RMSLE (holdout 20%) | 0.11249 |
| MAE | ~$ 13.800 |
| R² | ~0.934 |
| Melhora sobre baseline | ~36% |

## Instalação

```bash
pip install -r requirements.txt
```

## Repositório de Submissão

O código de produção (pipeline + modelo treinado) está no repositório de submissão separado.

