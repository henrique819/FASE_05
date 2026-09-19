"""
Feature engineering do modelo de risco de defasagem.

Usado igualmente pelo notebook de modelagem e pelo app Streamlit, garantindo
que o treino e a inferência apliquem exatamente as mesmas transformações.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Fase ideal por idade (moda observada na base PEDE 2022-2024, coerente com a
# Tabela 4 do relatório PEDE: Alfa 7-8 anos, Fase 1 8-9, Fase 2 10-11 ...)
FASE_IDEAL_POR_IDADE = {6: 0, 7: 0, 8: 0, 9: 1, 10: 2, 11: 2, 12: 3, 13: 3,
                        14: 4, 15: 5, 16: 6, 17: 7}


def fase_ideal(idade: float) -> float:
    if pd.isna(idade):
        return np.nan
    idade = int(idade)
    if idade >= 18:
        return 8.0
    return float(FASE_IDEAL_POR_IDADE.get(idade, 0))


# Variáveis de entrada (o que a equipe da Passos Mágicos informa no app)
NUM_INPUT = ["FASE", "IDADE", "ANOS_NA_PM", "DEFASAGEM", "INDE",
             "IAN", "IDA", "IEG", "IAA", "IPS", "IPV", "MAT", "POR"]
CAT_INPUT = ["GENERO", "INSTITUICAO_GRUPO"]

# Variáveis criadas
ENGINEERED = ["FASE_IDEAL_PROX_ANO", "DEFAS_PROJ_SEM_AVANCO", "ANO_DE_VIRADA_IDEAL",
              "GAP_IAA_IDA", "GAP_POR_MAT", "MEDIA_ACADEMICA_ENGAJ", "IEG_BAIXO", "IDA_BAIXO"]

FEATURES = NUM_INPUT + ENGINEERED + CAT_INPUT


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Recebe as colunas de NUM_INPUT/CAT_INPUT e devolve o dataframe de features."""
    X = df.copy()
    for c in NUM_INPUT:
        if c not in X:
            X[c] = np.nan
        X[c] = pd.to_numeric(X[c], errors="coerce")
    for c in CAT_INPUT:
        if c not in X:
            X[c] = "Não informado"
        X[c] = X[c].fillna("Não informado").astype(str)

    # Idade avança 1 ano: qual será a fase ideal no próximo ciclo?
    X["FASE_IDEAL_PROX_ANO"] = (X["IDADE"] + 1).map(fase_ideal)
    # Se o aluno NÃO avançar de fase, qual será sua defasagem no próximo ano?
    X["DEFAS_PROJ_SEM_AVANCO"] = X["FASE"] - X["FASE_IDEAL_PROX_ANO"]
    # A fase ideal "sobe" no próximo ano? (anos de transição exigem promoção)
    X["ANO_DE_VIRADA_IDEAL"] = (X["FASE_IDEAL_PROX_ANO"] > X["IDADE"].map(fase_ideal)).astype(int)

    X["GAP_IAA_IDA"] = X["IAA"] - X["IDA"]            # superestimação da autoavaliação
    X["GAP_POR_MAT"] = X["POR"] - X["MAT"]
    X["MEDIA_ACADEMICA_ENGAJ"] = X[["IDA", "IEG"]].mean(axis=1)
    X["IEG_BAIXO"] = (X["IEG"] < 7).astype(int)
    X["IDA_BAIXO"] = (X["IDA"] < 5).astype(int)
    return X[FEATURES]
