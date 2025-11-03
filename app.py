# app.py  — API de previsão (pipeline completo)
# Rodar:  uvicorn app:app --reload

from fastapi import FastAPI, Body
import joblib
import pandas as pd
import numpy as np
from datetime import datetime

app = FastAPI(title="API - Previsão de Aceitação de Campanha")

# Carrega a pipeline (prep + modelo) salva no passo 10.1
try:
    model = joblib.load("modelo_marketing.pkl")
except Exception as e:
    raise RuntimeError(f"Erro ao carregar modelo_marketing.pkl: {e}")

def _get_input_cols_from_prep():
    """
    Descobre as COLUNAS DE ENTRADA que o ColumnTransformer 'prep' espera
    (antes de OneHot/PCA). Retorna (num_cols, cat_cols).
    """
    num_cols, cat_cols = [], []
    try:
        prep = model.named_steps["prep"]
        # transformers_ é uma lista de (name, transformer, columns)
        for name, trans, cols in getattr(prep, "transformers_", []):
            if cols == "drop" or cols is None:
                continue
            if name == "num":
                num_cols = list(cols)
            elif name == "cat":
                cat_cols = list(cols)
    except Exception:
        pass
    return num_cols, cat_cols

def _compute_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula features derivadas SE estiverem faltando e os campos crus existirem.
    - Dependents = Kidhome + Teenhome
    - IncomePerCapita = Income / (1 + Dependents)
    - TotalSpent = soma dos Mnt*
    - Age = ano_atual - Year_Birth
    - DaysSinceEnroll = 2024-01-01 - Dt_Customer
    """
    df = df.copy()

    # Dependents
    if "Dependents" not in df or df["Dependents"].isna().any():
        kid = df.get("Kidhome", 0)
        teen = df.get("Teenhome", 0)
        df["Dependents"] = pd.to_numeric(kid, errors="coerce").fillna(0) + \
                           pd.to_numeric(teen, errors="coerce").fillna(0)

    # IncomePerCapita
    if "IncomePerCapita" not in df or df["IncomePerCapita"].isna().any():
        inc = pd.to_numeric(df.get("Income", np.nan), errors="coerce")
        if inc.notna().any():
            df["IncomePerCapita"] = (inc.fillna(0) /
                                     (1 + pd.to_numeric(df.get("Dependents", 0), errors="coerce").fillna(0)))
        else:
            # mantém se já veio no payload; senão, preenche depois como 0
            if "IncomePerCapita" not in df:
                df["IncomePerCapita"] = np.nan

    # TotalSpent
    if "TotalSpent" not in df or df["TotalSpent"].isna().any():
        gastos_cols = [
            "MntWines", "MntFruits", "MntMeatProducts",
            "MntFishProducts", "MntSweetProducts", "MntGoldProds"
        ]
        soma = 0
        tem_algum = False
        for c in gastos_cols:
            if c in df.columns:
                tem_algum = True
                soma = soma + pd.to_numeric(df[c], errors="coerce").fillna(0)
        if tem_algum:
            df["TotalSpent"] = soma
        else:
            if "TotalSpent" not in df:
                df["TotalSpent"] = np.nan

    # Age
    if "Age" not in df or df["Age"].isna().any():
        if "Year_Birth" in df.columns:
            year = pd.to_numeric(df["Year_Birth"], errors="coerce")
            df["Age"] = datetime.now().year - year.fillna(datetime.now().year)
        else:
            if "Age" not in df:
                df["Age"] = np.nan

    # DaysSinceEnroll
    if "DaysSinceEnroll" not in df or df["DaysSinceEnroll"].isna().any():
        if "Dt_Customer" in df.columns:
            dt = pd.to_datetime(df["Dt_Customer"], errors="coerce", dayfirst=True)
            df["DaysSinceEnroll"] = (pd.Timestamp("2024-01-01") - dt).dt.days
        else:
            if "DaysSinceEnroll" not in df:
                df["DaysSinceEnroll"] = np.nan

    return df

def _prepare_dataframe(payload: dict) -> pd.DataFrame:
    """
    Constrói o DataFrame a partir do JSON:
      1) Cria df com os campos enviados
      2) Calcula features derivadas se possível
      3) Garante TODAS as colunas de ENTRADA do prep (num & cat)
         - numéricas faltantes -> 0
         - categóricas faltantes -> "Unknown"
      4) Reordena colunas na ordem de entrada do treino
    """
    df = pd.DataFrame([payload])

    # 1) Normaliza valores binários/categóricos simples (se vierem como string "0"/"1")
    binarios = ["AcceptedCmp1","AcceptedCmp2","AcceptedCmp3","AcceptedCmp4","AcceptedCmp5","Complain",
                "Kidhome","Teenhome","Z_CostContact","Z_Revenue","NumDealsPurchases"]
    for b in binarios:
        if b in df.columns:
            df[b] = pd.to_numeric(df[b], errors="coerce")

    # 2) Calcula derivadas (se faltarem)
    df = _compute_engineered_features(df)

    # 3) Colunas de entrada esperadas pelo prep
    num_cols, cat_cols = _get_input_cols_from_prep()
    input_cols = list(num_cols) + list(cat_cols)

    # 4) Preenche faltantes com neutros
    for col in input_cols:
        if col not in df.columns:
            if col in num_cols:
                df[col] = 0
            else:
                df[col] = "Unknown"

    # 5) Reordena
    df = df.reindex(columns=input_cols, fill_value=0)

    # 6) Tipagem: força numéricas a serem numéricas
    for col in num_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # 7) Strings nas categóricas (OneHotEncoder lida com "Unknown")
    for col in cat_cols:
        df[col] = df[col].astype(str)

    return df

@app.post("/prever")
def prever(cliente: dict = Body(...)):
    """
    Recebe QUALQUER dicionário de campos (crus e/ou derivados),
    completa o que faltar e retorna a probabilidade de aceitar (classe 1).
    """
    try:
        dados = _prepare_dataframe(cliente)
        prob = model.predict_proba(dados)[0, 1]
        return {"status": "sucesso", "probabilidade_aceitar": round(float(prob), 4)}
    except Exception as e:
        # devolve erro legível para depuração
        return {"erro": str(e)}

@app.get("/")
def raiz():
    return {"status": "ok", "mensagem": "Use POST /prever com o payload do cliente."}
