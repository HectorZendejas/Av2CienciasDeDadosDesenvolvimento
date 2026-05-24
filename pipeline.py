import os

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_log_error


LIMITE_SUPERIOR_PRECO = 745000.0


def aplicar_feature_engineering(df):
    dados = df.copy()
    dados["TotalSF"] = dados["TotalBsmtSF"] + dados["1stFlrSF"] + dados["2ndFlrSF"]
    dados["HouseAge"] = dados["YrSold"] - dados["YearBuilt"]
    dados["RemodAge"] = dados["YrSold"] - dados["YearRemodAdd"]
    dados["WasRemodeled"] = (dados["YearRemodAdd"] != dados["YearBuilt"]).astype(int)
    dados["TotalBaths"] = (
        dados["FullBath"] + 0.5 * dados["HalfBath"]
        + dados["BsmtFullBath"] + 0.5 * dados["BsmtHalfBath"]
    )
    dados["HasPool"] = (dados["PoolArea"] > 0).astype(int)
    dados["HasFireplace"] = (dados["Fireplaces"] > 0).astype(int)
    dados["HasGarage"] = (dados["GarageArea"] > 0).astype(int)
    return dados


def prever_precos(caminho_arquivo_teste):
    """
    Funcao obrigatoria para o corretor automatico.
    Le o arquivo de teste, aplica o pipeline treinado e retorna as predicoes.

    Parametros:
    caminho_arquivo_teste (str): Caminho local para o arquivo CSV de teste.

    Retorna:
    np.array: As predicoes de precos, em dolares e nao negativas.
    """
    df_teste = pd.read_csv(caminho_arquivo_teste)
    df_teste = aplicar_feature_engineering(df_teste)

    caminho_modelo = os.path.join(os.path.dirname(__file__), "modelo_final.joblib")
    if not os.path.exists(caminho_modelo):
        caminho_modelo = os.path.join(os.path.dirname(__file__), "modelo_baseline.joblib")

    if not os.path.exists(caminho_modelo):
        raise FileNotFoundError("Arquivo de modelo nao encontrado na pasta do pipeline.")

    modelo = joblib.load(caminho_modelo)
    predicoes = modelo.predict(df_teste)

    # Formato exigido pela avaliacao: somente um array com valores continuos.
    # O limite superior evita extrapolacoes irreais em modelos lineares.
    return np.asarray(np.clip(predicoes, a_min=0, a_max=LIMITE_SUPERIOR_PRECO), dtype=float)


if __name__ == "__main__":
    arquivo_teste_exemplo = "teste_publico.csv"
    if not os.path.exists(arquivo_teste_exemplo):
        arquivo_teste_exemplo = os.path.join(os.path.dirname(__file__), "teste_publico.csv")

    print("--- Executando Validacao Local do Pipeline ---")

    if not os.path.exists(arquivo_teste_exemplo):
        print(f"[Aviso] Arquivo '{arquivo_teste_exemplo}' nao encontrado.")
    else:
        try:
            resultados = prever_precos(arquivo_teste_exemplo)

            print("\nSucesso! O pipeline rodou corretamente.")
            print("-" * 30)
            print("Primeiras 5 predicoes:")
            print(resultados[:5])
            print("-" * 30)

            df_val = pd.read_csv(arquivo_teste_exemplo)
            if "SalePrice" in df_val.columns:
                y_true = df_val["SalePrice"]
                rmsle = np.sqrt(mean_squared_log_error(y_true, resultados))
                print(f"Metrica RMSLE Local: {rmsle:.5f}")
            else:
                print("[Nota] Coluna 'SalePrice' nao encontrada. Calculo do RMSLE pulado.")

        except Exception as e:
            print("\nErro encontrado no pipeline:")
            print(str(e))
