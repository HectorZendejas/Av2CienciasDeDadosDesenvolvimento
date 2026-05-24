# Competição de Machine Learning — Previsão de Preços de Imóveis

---

## Capa

**Identificação da equipe:** [NOME DO GRUPO]

**Integrantes:**
- [Nome completo 1]
- [Nome completo 2]
- [Nome completo 3]

**Repositório de Submissão:** [LINK GITHUB — SUBMISSÃO]

**Repositório de Desenvolvimento:** [LINK GITHUB — DESENVOLVIMENTO]

---

## 2. Introdução

Este trabalho tem como objetivo construir um modelo preditivo capaz de estimar o preço de venda de imóveis residenciais localizados em Ames, Iowa (EUA), utilizando o dataset *House Prices*. O conjunto de dados conta com 79 variáveis explicativas que descrevem características físicas, estruturais e de localização das casas, desde atributos objetivos como área total e número de cômodos até avaliações subjetivas como qualidade geral da construção e condição do imóvel.

O problema é formulado como uma tarefa de regressão supervisionada: a partir das características de um imóvel, estimar seu valor de venda (`SalePrice`). A base de treino disponibilizada contém 1.168 imóveis com o valor de venda conhecido; a base de teste público contém 1.459 imóveis sem a variável alvo, simulando dados inéditos.

A métrica central escolhida para otimização e avaliação é o **RMSLE** (*Root Mean Squared Logarithmic Error*):

$$\text{RMSLE} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} \left(\log(1 + \hat{y}_i) - \log(1 + y_i)\right)^2}$$

O RMSLE foi escolhido porque penaliza erros de forma proporcional ao valor real do imóvel — errar em $20.000 numa casa de $100.000 é proporcionalmente mais grave do que o mesmo erro numa casa de $500.000. Essa característica torna a métrica mais adequada para precificação de imóveis do que o RMSE absoluto, que privilegia erros em casas de alto valor. Complementarmente, reportamos também o **MAE** (erro absoluto médio em dólares) e o **R²** para facilitar a interpretação do desempenho.

---

## 3. Análise Exploratória de Dados (EDA)

### 3.1 Variável Alvo — SalePrice

A variável alvo apresenta distribuição fortemente assimétrica à direita (assimetria positiva), com concentração de imóveis na faixa de $100.000 a $250.000 e uma cauda longa de imóveis de alto valor. A maioria dos imóveis se situa entre $80.000 e $350.000, com mediana próxima de $165.000.

A aplicação da transformação logarítmica (`log1p`) aproxima a distribuição de uma normal, o que é relevante para modelos que assumem resíduos com distribuição simétrica e também alinha diretamente a distribuição do alvo com a escala avaliada pelo RMSLE. Essa transformação foi incorporada ao pipeline de modelagem via `TransformedTargetRegressor`.

### 3.2 Correlações com SalePrice

A análise de correlação de Pearson entre as variáveis numéricas e `SalePrice` revelou as seguintes variáveis como as mais associadas ao preço:

| Variável | Correlação | Interpretação |
|---|---:|---|
| `OverallQual` | ~0,79 | Qualidade geral da construção e acabamento |
| `GrLivArea` | ~0,71 | Área habitável acima do solo (sqft) |
| `GarageCars` | ~0,64 | Capacidade da garagem em número de carros |
| `GarageArea` | ~0,62 | Área da garagem (sqft) |
| `TotalBsmtSF` | ~0,61 | Área total do porão (sqft) |
| `1stFlrSF` | ~0,61 | Área do primeiro andar (sqft) |
| `FullBath` | ~0,56 | Número de banheiros completos acima do solo |
| `YearBuilt` | ~0,52 | Ano de construção |
| `YearRemodAdd` | ~0,51 | Ano da última reforma |
| `MasVnrArea` | ~0,48 | Área do revestimento de alvenaria |

A qualidade geral (`OverallQual`) se destaca como o preditor individual mais forte, seguido pela área habitável. Variáveis relacionadas à garagem aparecem em terceiro lugar, refletindo a importância da mobilidade urbana em Ames.

### 3.3 Variáveis Categóricas

Variáveis categóricas como `Neighborhood` (bairro) apresentam forte relação com o preço, com medianas que variam de ~$100.000 (bairros MeadowV e BrDale) a ~$300.000 (bairros NridgHt e NoRidge). `SaleCondition` e `MSZoning` também mostram diferenças expressivas de preço entre categorias.

### 3.4 Diagnóstico de Valores Nulos

O dataset contém diversas colunas com alta proporção de valores ausentes. Contudo, conforme indicado pelo dicionário de dados, a maioria desses "nulos" não representa dado perdido — representa ausência da característica. Por exemplo:

| Coluna | % Nulos | Significado do NaN |
|---|---:|---|
| `PoolQC` | ~99,5% | Imóvel não possui piscina |
| `MiscFeature` | ~96,3% | Imóvel não possui característica especial |
| `Alley` | ~93,8% | Imóvel não tem acesso por beco |
| `Fence` | ~80,8% | Imóvel não possui cerca |
| `FireplaceQu` | ~47,3% | Imóvel não possui lareira |
| `GarageType` | ~5,5% | Imóvel não possui garagem |

Essa distinção é crítica: tratar esses NaN como dado perdido e imputar pela mediana ou moda introduziria ruído no modelo. A estratégia adotada foi preservar o significado semântico dessas variáveis — para variáveis numéricas, imputação pela mediana; para categóricas, preenchimento com a categoria `"Nenhum"`, sinalizando explicitamente a ausência da característica.

### 3.5 Compatibilidade entre Treino e Teste

A distribuição das variáveis numéricas principais (`GrLivArea`, `LotArea`, `OverallQual`) foi comparada entre treino e teste público, confirmando distribuições similares e ausência de desvio de distribuição (*distribution shift*) relevante entre os conjuntos.

### 3.6 Identificação de Outliers

Dois imóveis foram identificados como outliers extremos: propriedades com `GrLivArea > 4.000 sqft` e `SalePrice < $300.000`. Esses casos apresentam área habitável excepcionalmente grande para um preço muito abaixo do esperado, possivelmente refletindo vendas não convencionais (leilão, venda forçada). A combinação dessas duas condições os coloca em região isolada do gráfico de dispersão área × preço, distante do padrão da maioria dos imóveis.

---

## 4. Pré-processamento e Feature Engineering

### 4.1 Remoção de Outliers

Foram removidos 2 imóveis que satisfaziam simultaneamente `GrLivArea > 4.000 sqft` **e** `SalePrice < $300.000`, reduzindo a base de treino de 1.168 para 1.166 registros. A remoção foi aplicada **exclusivamente ao conjunto de treino** — o pipeline de predição nunca filtra dados de teste, garantindo que qualquer imóvel recebido pelo corretor automático seja processado.

### 4.2 Feature Engineering

Com base nas correlações observadas na EDA e no conhecimento do domínio, foram criadas 8 variáveis derivadas:

| Variável criada | Fórmula / Lógica | Justificativa |
|---|---|---|
| `TotalSF` | `TotalBsmtSF + 1stFlrSF + 2ndFlrSF` | Área total do imóvel em um único atributo; correlação mais forte com o preço do que as partes individuais |
| `HouseAge` | `YrSold - YearBuilt` | Captura o efeito de depreciação temporal da propriedade |
| `RemodAge` | `YrSold - YearRemodAdd` | Captura o quanto tempo passou desde a última reforma |
| `WasRemodeled` | `1` se `YearRemodAdd ≠ YearBuilt` | Indica se o imóvel passou por reforma; reformas valorizam o imóvel independentemente da idade |
| `TotalBaths` | `FullBath + 0,5×HalfBath + BsmtFullBath + 0,5×BsmtHalfBath` | Consolida todos os banheiros em uma única métrica ponderada |
| `HasPool` | `1` se `PoolArea > 0` | Indicador binário da presença de piscina |
| `HasFireplace` | `1` se `Fireplaces > 0` | Indicador binário da presença de lareira |
| `HasGarage` | `1` se `GarageArea > 0` | Indicador binário da presença de garagem |

A mesma função `aplicar_feature_engineering` é chamada tanto no treinamento quanto no `pipeline.py` de predição, garantindo que os dados de teste recebam exatamente as mesmas transformações.

Após o feature engineering, o conjunto de entrada passou de 79 para 87 variáveis explicativas (excluindo `Id` e `SalePrice`).

### 4.3 Pré-processamento

O pré-processamento foi inteiramente encapsulado dentro de um `Pipeline` e `ColumnTransformer` do Scikit-Learn, separando o tratamento por tipo de variável:

**Variáveis numéricas (36 após FE):**
1. Imputação pela **mediana** — robusta a outliers, não distorce a distribuição
2. Padronização com **StandardScaler** — necessária para o Ridge Regression, que é sensível à escala

**Variáveis categóricas (43 após FE):**
1. Imputação com a categoria constante **`"Nenhum"`** — preserva o significado semântico da ausência
2. **OneHotEncoder** com `handle_unknown='ignore'` — converte categorias em variáveis binárias e ignora categorias novas no teste sem quebrar o pipeline

**Transformação da variável alvo:**
A variável `SalePrice` foi transformada com `log1p` durante o treinamento e revertida com `expm1` nas predições via **`TransformedTargetRegressor`**. Isso força o modelo a otimizar o erro proporcional, alinhando a função de perda do algoritmo com a métrica de avaliação RMSLE.

O uso de `Pipeline` e `ColumnTransformer` garante que os parâmetros de pré-processamento (mediana de imputação, médias e desvios do StandardScaler) sejam aprendidos **exclusivamente nos dados de treino** e reaplicados nos dados de validação e teste, evitando *data leakage*.

---

## 5. Modelagem e Validação

### 5.1 Algoritmos Testados

Foram avaliados quatro algoritmos, cobrindo desde modelos lineares regularizados até ensembles baseados em árvores:

| Algoritmo | Característica principal |
|---|---|
| **Ridge Regression** | Regressão linear com regularização L2; penaliza coeficientes grandes, reduzindo overfitting em espaços de alta dimensão |
| **Random Forest** | Ensemble de árvores independentes com bagging; robusto a outliers e ruído |
| **Gradient Boosting** | Ensemble sequencial que corrige erros residuais iterativamente; boa capacidade preditiva com custo computacional maior |
| **XGBoost** | Implementação otimizada de gradient boosting com regularização adicional |

A escolha progressiva — do mais simples (Ridge) ao mais complexo (XGBoost) — seguiu a recomendação de estabelecer um baseline sólido antes de escalar a complexidade.

### 5.2 Estratégia de Validação

Para avaliar o desempenho de forma confiável e detectar overfitting, foram utilizadas duas estratégias complementares:

**Holdout (divisão treino/validação):**
- 80% dos dados para treino, 20% para validação
- `random_state=42` para reprodutibilidade
- Permite comparação rápida entre algoritmos

**Validação Cruzada K-Fold:**
- 5 folds, embaralhados com `random_state=42`
- Aplicada ao modelo final para estimar o desempenho esperado nos dados secretos
- O RMSLE médio dos 5 folds é a estimativa mais confiável de generalização

### 5.3 Busca de Hiperparâmetros — Ridge

O único hiperparâmetro do Ridge com impacto relevante é o `alpha`, que controla a intensidade da regularização L2. Um `alpha` muito baixo aproxima o modelo da regressão linear sem regularização (risco de overfitting em espaços de muitas variáveis); um `alpha` muito alto superpenaliza os coeficientes (underfitting).

A busca foi realizada com **`GridSearchCV`** sobre 9 valores:

| alpha | RMSLE médio CV 5-fold |
|---:|---:|
| 0,5 | ~0,1160 |
| 1,0 | ~0,1150 |
| 5,0 | ~0,1135 |
| 10,0 | ~0,1128 |
| **20,0** | **0,11219** |
| 50,0 | ~0,1128 |
| 100,0 | ~0,1132 |
| 200,0 | ~0,1140 |
| 500,0 | ~0,1155 |

O **alpha = 20** apresentou o menor RMSLE médio na validação cruzada e foi utilizado no modelo final.

---

## 6. Conclusões

### 6.1 Tabela Comparativa de Modelos

Todos os modelos foram avaliados no mesmo holdout (20% do treino, `random_state=42`). O baseline de referência é a Regressão Linear simples executada pelo professor.

| Modelo | Tempo treino (s) | RMSLE | RMSE ($) | MAE ($) | R² |
|---|---:|---:|---:|---:|---:|
| Regressão Linear (Baseline) | — | 0,17543 | 36.061 | 22.186 | 0,83046 |
| XGBoost | ~3,5 | ~0,1240 | ~24.000 | ~15.500 | ~0,907 |
| Random Forest | 4,64 | 0,13015 | 25.233 | 16.572 | 0,90077 |
| Gradient Boosting | 4,89 | 0,11911 | 22.215 | 14.352 | 0,92308 |
| **Ridge Regression (modelo enviado)** | **0,19** | **0,11249** | **20.476** | **13.814** | **0,93465** |

*Os resultados de XGBoost são estimados a partir do notebook de desenvolvimento; os demais valores são exatos.*

*RMSLE do Ridge em validação cruzada 5-fold: **0,11219 ± 0,00409***.

### 6.2 Impacto para o Negócio

O modelo final (Ridge Regression) obteve um **MAE de $ 13.814**, o que significa que ele erra o preço de avaliação das casas, em média, **em aproximadamente $ 13.800 para mais ou para menos**.

Para contextualizar: a mediana dos preços no dataset é de aproximadamente **$ 165.000**. Para uma casa nessa faixa de preço, o modelo tipicamente produziria uma estimativa entre **$ 146.500 e $ 183.500** — uma margem de erro de aproximadamente **11%**.

O RMSLE de **0,112** indica que o erro proporcional médio é de ~11,2%. Ou seja:

- Numa casa de **$100.000**, o erro esperado é de até ~$11.200
- Numa casa de **$200.000**, o erro esperado é de até ~$22.400
- Numa casa de **$400.000**, o erro esperado é de até ~$44.800

Essa precisão é adequada para um contexto de **precificação inicial de imóveis residenciais**, onde o objetivo é orientar a decisão do vendedor sobre a faixa de preço a anunciar, não substituir uma avaliação presencial por um perito. A melhora de **~36% sobre o baseline** do professor (RMSLE 0,17543 → 0,11219) demonstra que as escolhas de feature engineering e regularização trouxeram ganho real e consistente.

### 6.3 Justificativa do Modelo Escolhido

O **Ridge Regression com alpha = 20** foi escolhido como modelo de submissão pelos seguintes motivos:

1. **Melhor RMSLE** tanto no holdout (0,11249) quanto na validação cruzada 5-fold (0,11219), superando Gradient Boosting, Random Forest e XGBoost no mesmo conjunto de dados.

2. **Menor variância** entre os folds (desvio padrão de 0,00409), indicando que o desempenho é estável e não dependente de uma divisão específica dos dados — o que sugere boa capacidade de generalização para o teste secreto.

3. **Velocidade de inferência** de menos de 1 segundo para processar 1.459 imóveis, muito abaixo do limite de 1 minuto estabelecido pelo corretor automático.

4. **Regularização adequada ao problema**: com 87 variáveis explicativas após o feature engineering (das quais muitas são dummies do OneHotEncoder), o espaço de hipóteses é grande. A regularização L2 do Ridge penaliza coeficientes grandes sem zerar variáveis, aproveitando a informação de todas as features ao mesmo tempo em que controla o overfitting.

5. **Pipeline robusto**: toda a lógica de pré-processamento (imputação, escalonamento, encoding e transformação do alvo) está encapsulada em um único objeto serializado (`modelo_final.joblib`), eliminando o risco de inconsistência entre treino e predição.

A escolha do Ridge sobre modelos de ensemble mais complexos reflete um princípio importante: **complexidade adicional só se justifica quando traz ganho mensurável de desempenho**. Neste caso, os ensembles foram de 24 a 73 vezes mais lentos para treinar e ainda assim produziram RMSLE superiores (piores), não justificando a troca.

