"""
Limpeza e padronização da base PEDE 2022-2024 (Associação Passos Mágicos).

Gera duas tabelas:
  * pede_long.csv  -> uma linha por aluno x ano (2022, 2023, 2024), colunas padronizadas
  * pares_t_t1.csv -> uma linha por aluno com indicadores do ano t e desfecho do ano t+1
                      (base de modelagem do risco de defasagem)

Uso:
    python src/data_prep.py
"""
from __future__ import annotations

import datetime as dt
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "BASE_DE_DADOS_PEDE_2024_-_DATATHON.xlsx"
PROCESSED = ROOT / "data" / "processed"

INDICADORES = ["IAN", "IDA", "IEG", "IAA", "IPS", "IPP", "IPV"]
PEDRAS = ["Quartzo", "Ágata", "Ametista", "Topázio"]

# Mapeamento de colunas de cada aba para um nome único
COLMAP = {
    2022: {
        "RA": "RA", "Fase": "FASE_RAW", "Turma": "TURMA", "Idade 22": "IDADE",
        "Gênero": "GENERO", "Ano ingresso": "ANO_INGRESSO",
        "Instituição de ensino": "INSTITUICAO", "Pedra 22": "PEDRA", "INDE 22": "INDE",
        "IAA": "IAA", "IEG": "IEG", "IPS": "IPS", "IDA": "IDA", "Matem": "MAT",
        "Portug": "POR", "Inglês": "ING", "IPV": "IPV", "IAN": "IAN",
        "Fase ideal": "FASE_IDEAL_RAW", "Defas": "DEFASAGEM", "Atingiu PV": "PONTO_VIRADA",
        "Indicado": "INDICADO_BOLSA", "Nº Av": "N_AVALIACOES", "Rec Psicologia": "REC_PSICO",
    },
    2023: {
        "RA": "RA", "Fase": "FASE_RAW", "Turma": "TURMA", "Idade": "IDADE",
        "Gênero": "GENERO", "Ano ingresso": "ANO_INGRESSO",
        "Instituição de ensino": "INSTITUICAO", "Pedra 2023": "PEDRA", "INDE 2023": "INDE",
        "IAA": "IAA", "IEG": "IEG", "IPS": "IPS", "IPP": "IPP", "IDA": "IDA", "Mat": "MAT",
        "Por": "POR", "Ing": "ING", "IPV": "IPV", "IAN": "IAN",
        "Fase Ideal": "FASE_IDEAL_RAW", "Defasagem": "DEFASAGEM", "Nº Av": "N_AVALIACOES",
    },
    2024: {
        "RA": "RA", "Fase": "FASE_RAW", "Turma": "TURMA", "Idade": "IDADE",
        "Gênero": "GENERO", "Ano ingresso": "ANO_INGRESSO",
        "Instituição de ensino": "INSTITUICAO", "Pedra 2024": "PEDRA", "INDE 2024": "INDE",
        "IAA": "IAA", "IEG": "IEG", "IPS": "IPS", "IPP": "IPP", "IDA": "IDA", "Mat": "MAT",
        "Por": "POR", "Ing": "ING", "IPV": "IPV", "IAN": "IAN",
        "Fase Ideal": "FASE_IDEAL_RAW", "Defasagem": "DEFASAGEM", "Nº Av": "N_AVALIACOES",
        "Escola": "ESCOLA",
    },
}


# ----------------------------------------------------------------------------
# Funções auxiliares de limpeza
# ----------------------------------------------------------------------------
def parse_fase(v) -> float:
    """'ALFA'->0, 'FASE 3'->3, '4M'->4, 7->7, 9->9 (universitários/egressos)."""
    if pd.isna(v):
        return np.nan
    s = str(v).strip().upper()
    if s.startswith("ALFA"):
        return 0.0
    m = re.search(r"\d", s)
    return float(m.group()) if m else np.nan


def parse_fase_ideal(v) -> float:
    """'ALFA (1° e 2° ano)'->0, 'Fase 3 (7° e 8° ano)'->3."""
    if pd.isna(v):
        return np.nan
    s = str(v).strip().upper()
    if s.startswith("ALFA"):
        return 0.0
    m = re.search(r"FASE\s*(\d)", s)
    return float(m.group(1)) if m else np.nan


def parse_idade(v) -> float:
    """Na aba 2023 parte das idades veio como data do Excel (1900-01-11 == 11 anos)."""
    if pd.isna(v):
        return np.nan
    if isinstance(v, (dt.datetime, pd.Timestamp)):
        return float(v.day)
    try:
        return float(v)
    except (TypeError, ValueError):
        return np.nan


def padroniza_pedra(v):
    if pd.isna(v):
        return np.nan
    s = str(v).strip().capitalize()
    mapa = {"Agata": "Ágata", "Ágata": "Ágata", "Quartzo": "Quartzo",
            "Ametista": "Ametista", "Topázio": "Topázio", "Topazio": "Topázio"}
    return mapa.get(s, np.nan)  # 'INCLUIR' e afins viram NaN


def pedra_por_inde(inde: float):
    """Classificação oficial (dicionário de dados) a partir do INDE."""
    if pd.isna(inde):
        return np.nan
    if inde < 5.506:
        return "Quartzo"
    if inde < 6.868:
        return "Ágata"
    if inde < 8.230:
        return "Ametista"
    return "Topázio"


def padroniza_genero(v):
    if pd.isna(v):
        return np.nan
    s = str(v).strip().lower()
    if s in ("menina", "feminino"):
        return "Feminino"
    if s in ("menino", "masculino"):
        return "Masculino"
    return np.nan


def padroniza_instituicao(v):
    """Agrupa as instituições em Pública / Privada (bolsa/parceria) / Outros."""
    if pd.isna(v):
        return "Não informado"
    s = str(v).lower()
    if "públic" in s or "public" in s:
        return "Pública"
    if "privada" in s or "decisão" in s or "jp ii" in s or "bolsista" in s:
        return "Privada/Bolsa"
    return "Outros/Concluiu EM"


def classe_defasagem(d: float):
    """Tabela 41 do PEDE: D>=0 em fase | -2<=D<0 moderada | D<-2 severa."""
    if pd.isna(d):
        return np.nan
    if d >= 0:
        return "Em fase"
    if d >= -2:
        return "Moderada"
    return "Severa"


def ian_por_defasagem(d: float) -> float:
    if pd.isna(d):
        return np.nan
    return 10.0 if d >= 0 else (5.0 if d >= -2 else 2.5)


# ----------------------------------------------------------------------------
# Pipeline
# ----------------------------------------------------------------------------
def carrega_ano(xls: pd.ExcelFile, ano: int) -> pd.DataFrame:
    raw = xls.parse(f"PEDE{ano}")
    df = raw[[c for c in COLMAP[ano] if c in raw.columns]].rename(columns=COLMAP[ano]).copy()
    df["ANO"] = ano

    df["FASE"] = df["FASE_RAW"].map(parse_fase)
    df["FASE_IDEAL"] = df["FASE_IDEAL_RAW"].map(parse_fase_ideal)
    df["IDADE"] = df["IDADE"].map(parse_idade)
    df["GENERO"] = df["GENERO"].map(padroniza_genero)
    df["INSTITUICAO_GRUPO"] = df["INSTITUICAO"].map(padroniza_instituicao)

    num_cols = ["INDE", "IAA", "IEG", "IPS", "IPP", "IDA", "MAT", "POR", "ING",
                "IPV", "IAN", "DEFASAGEM", "ANO_INGRESSO", "N_AVALIACOES"]
    for c in num_cols:
        if c not in df.columns:
            df[c] = np.nan
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Pedra: usa a informada; se ausente, deriva do INDE (regra oficial)
    df["PEDRA"] = df["PEDRA"].map(padroniza_pedra)
    df["PEDRA"] = df["PEDRA"].fillna(df["INDE"].map(pedra_por_inde))

    # Alunos 'INCLUIR' (2024) chegam sem avaliação: IEG=0 é ausência, não nota
    sem_avaliacao = df["IDA"].isna() & df["IPV"].isna() & df["IAA"].isna()
    df.loc[sem_avaliacao & (df["IEG"] == 0), "IEG"] = np.nan
    df["SEM_AVALIACAO"] = sem_avaliacao

    # IAA = 0 é implausível (questionário de 6 perguntas com nota mínima > 0):
    # em 2023 são 190 casos -> tratado como não respondido
    df["IAA_ZERADO"] = (df["IAA"] == 0).astype(int)
    df.loc[df["IAA"] == 0, "IAA"] = np.nan

    df["ANOS_NA_PM"] = (df["ANO"] - df["ANO_INGRESSO"]).clip(lower=0)
    df["CLASSE_DEFASAGEM"] = df["DEFASAGEM"].map(classe_defasagem)
    df["EM_DEFASAGEM"] = (df["DEFASAGEM"] < 0).astype(int)

    if "PONTO_VIRADA" in df.columns:
        df["PONTO_VIRADA"] = df["PONTO_VIRADA"].map({"Sim": 1, "Não": 0})
    else:
        df["PONTO_VIRADA"] = np.nan
    return df


def monta_base_longa() -> pd.DataFrame:
    xls = pd.ExcelFile(RAW)
    df = pd.concat([carrega_ano(xls, a) for a in (2022, 2023, 2024)], ignore_index=True)
    ordem = ["RA", "ANO", "FASE", "TURMA", "IDADE", "GENERO", "ANO_INGRESSO", "ANOS_NA_PM",
             "INSTITUICAO", "INSTITUICAO_GRUPO", "PEDRA", "INDE", *INDICADORES,
             "MAT", "POR", "ING", "FASE_IDEAL", "DEFASAGEM", "CLASSE_DEFASAGEM",
             "EM_DEFASAGEM", "PONTO_VIRADA", "N_AVALIACOES", "SEM_AVALIACAO", "IAA_ZERADO"]
    return df[ordem]


def monta_pares(longa: pd.DataFrame) -> pd.DataFrame:
    """Une o ano t com o ano t+1 do mesmo RA (2022->2023 e 2023->2024)."""
    pares = []
    for t in (2022, 2023):
        a = longa[longa["ANO"] == t]
        b = longa[longa["ANO"] == t + 1][["RA", "DEFASAGEM", "IAN", "INDE", "IDA",
                                          "IEG", "PEDRA", "FASE"]]
        b = b.rename(columns=lambda c: c if c == "RA" else f"{c}_T1")
        m = a.merge(b, on="RA", how="inner")
        pares.append(m)
    p = pd.concat(pares, ignore_index=True)

    # --- Alvo: aluno em risco de defasagem no ano seguinte -------------------
    #  risco = passa a estar (ou continua) defasado em t+1  OU  a defasagem piora
    p["DEFASADO_T1"] = (p["DEFASAGEM_T1"] < 0).astype(int)
    p["PIOROU_DEFASAGEM"] = (p["DEFASAGEM_T1"] < p["DEFASAGEM"]).astype(int)
    p["RISCO_DEFASAGEM"] = ((p["DEFASADO_T1"] == 1) | (p["PIOROU_DEFASAGEM"] == 1)).astype(int)
    p["QUEDA_INDE"] = (p["INDE_T1"] < p["INDE"]).astype(float).where(p["INDE_T1"].notna() & p["INDE"].notna())
    return p


def main():
    PROCESSED.mkdir(parents=True, exist_ok=True)
    longa = monta_base_longa()
    pares = monta_pares(longa)
    longa.to_csv(PROCESSED / "pede_long.csv", index=False)
    pares.to_csv(PROCESSED / "pares_t_t1.csv", index=False)
    print(f"pede_long.csv  -> {longa.shape}")
    print(f"pares_t_t1.csv -> {pares.shape}")


if __name__ == "__main__":
    main()
