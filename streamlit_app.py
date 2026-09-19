"""
Passos Mágicos · Preditor de Risco de Defasagem
App Streamlit do Datathon Fase 5 (POSTECH Data Analytics - FIAP)

Executar localmente:  streamlit run streamlit_app.py
"""
import io
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from features import build_features, fase_ideal, NUM_INPUT, CAT_INPUT  # noqa: E402
from data_prep import parse_fase, parse_idade, padroniza_genero, padroniza_instituicao  # noqa: E402

st.set_page_config(page_title="Passos Mágicos · Risco de Defasagem", page_icon="⭐", layout="wide")

AZUL, LARANJA, AMARELO, VERDE, VERMELHO = "#16508A", "#F58634", "#F7B32B", "#2E9E6B", "#C0392B"

st.markdown(f"""
<style>
.bloco {{background:#F4F7FB;border-radius:14px;padding:18px 22px;margin-bottom:8px}}
.prob {{font-size:56px;font-weight:800;line-height:1}}
.faixa {{display:inline-block;padding:4px 14px;border-radius:20px;color:white;font-weight:700;font-size:15px}}
.small {{color:#5A6472;font-size:13px}}
</style>""", unsafe_allow_html=True)


# ----------------------------------------------------------------------------
@st.cache_resource
def carregar_modelo():
    modelo = joblib.load(ROOT / "models" / "modelo_risco_defasagem.joblib")
    meta = json.loads((ROOT / "models" / "metadata.json").read_text(encoding="utf-8"))
    return modelo, meta


modelo, META = carregar_modelo()
LIMIAR = META["limiar"]


def faixa_risco(p: float):
    if p < 0.30:
        return "Baixo", VERDE
    if p < LIMIAR:
        return "Atenção", AMARELO
    if p < 0.70:
        return "Alto", LARANJA
    return "Muito alto", VERMELHO


def ian_de(defas: float) -> float:
    return 10.0 if defas >= 0 else (5.0 if defas >= -2 else 2.5)


def inde_estimado(fase, ian, ida, ieg, iaa, ips, ipp, ipv) -> float:
    if fase >= 8:
        return 0.1 * ian + 0.4 * ida + 0.2 * ieg + 0.1 * iaa + 0.2 * ips
    return 0.1 * ian + 0.2 * ida + 0.2 * ieg + 0.1 * iaa + 0.1 * ips + 0.1 * ipp + 0.2 * ipv


def prever(df: pd.DataFrame) -> np.ndarray:
    return modelo.predict_proba(build_features(df))[:, 1]


def alertas(r: dict) -> list[str]:
    """Explicações em linguagem simples, baseadas nos padrões encontrados na análise."""
    msgs = []
    proj = r["FASE"] - fase_ideal(r["IDADE"] + 1)
    if proj < 0:
        msgs.append(f"🎯 **Precisa avançar de fase:** se permanecer na Fase {int(r['FASE'])}, ficará "
                    f"**{int(abs(proj))} fase(s)** atrás da fase ideal para a idade no próximo ano.")
    if r["DEFASAGEM"] < 0:
        msgs.append(f"📉 Já está **{int(abs(r['DEFASAGEM']))} fase(s) abaixo** do ideal hoje (IAN {ian_de(r['DEFASAGEM']):.1f}).")
    if r["IEG"] < 7:
        msgs.append("🙋 **Engajamento baixo (IEG < 7)** — é o termômetro mais precoce de queda de desempenho.")
    if r["IDA"] < 5:
        msgs.append("📚 **Desempenho acadêmico baixo (IDA < 5)** — priorizar reforço em Matemática e Português.")
    if r["IPV"] < 6.5:
        msgs.append("🔄 **Ponto de virada distante (IPV < 6,5).**")
    if r["IPS"] < 5:
        msgs.append("💬 **IPS baixo** — alunos com IPS < 5 têm mais quedas de engajamento no ano seguinte. Avaliar acompanhamento psicológico.")
    if not pd.isna(r.get("IAA")) and r["IAA"] - r["IDA"] > 3:
        msgs.append("🪞 Autoavaliação muito acima do desempenho real — vale uma devolutiva individual.")
    return msgs


ACOES = {
    "Baixo": "Manter o acompanhamento regular e reconhecer a evolução do aluno.",
    "Atenção": "Monitorar IEG mensalmente e checar entregas de lição; conversa com o aluno.",
    "Alto": "Plano individual: reforço acadêmico nas disciplinas mais fracas + avaliação psicopedagógica para avanço de fase.",
    "Muito alto": "Prioridade máxima: reforço intensivo, avaliação de avanço de fase, contato com a família e encaminhamento psicológico se IPS baixo.",
}

# ----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## ⭐ Passos Mágicos")
    st.markdown("**Preditor de Risco de Defasagem**")
    st.caption("Datathon Fase 5 · POSTECH Data Analytics (FIAP)")
    st.divider()
    st.markdown("**O que o modelo prevê**")
    st.write("Probabilidade de o aluno estar **defasado** (abaixo da fase ideal) **ou piorar a defasagem** "
             "no **próximo ciclo do PEDE**, usando apenas os indicadores atuais.")
    mt = META["metricas_teste"]
    c1, c2 = st.columns(2)
    c1.metric("ROC-AUC", f"{mt['ROC-AUC']:.2f}")
    c2.metric("Recall", f"{mt['Recall']:.0%}")
    c1.metric("Acurácia", f"{mt['Acurácia']:.0%}")
    c2.metric("F1", f"{mt['F1']:.2f}")
    st.caption(f"{META['modelo']} · {META['n_pares_treino']} pares aluno-ano (PEDE 2022→2023 e 2023→2024) · limiar {LIMIAR:.2f}")
    st.divider()
    st.markdown("**Faixas de risco**")
    st.markdown(f"🟢 Baixo < 30% · 🟡 Atenção 30–{LIMIAR:.0%} · 🟠 Alto {LIMIAR:.0%}–70% · 🔴 Muito alto ≥ 70%")

st.title("Quem precisa de apoio antes do próximo ciclo?")
st.caption("Informe os indicadores atuais do aluno (ou envie uma planilha) para estimar a probabilidade de risco de defasagem no próximo ano.")

aba1, aba2, aba3 = st.tabs(["👤 Aluno individual", "📋 Turma / planilha", "📊 Panorama 2024 → 2025"])

# ----------------------------------------------------------------------------
with aba1:
    col_in, col_out = st.columns([1.25, 1])
    with col_in:
        st.subheader("Dados do aluno")
        a, b, c = st.columns(3)
        fase = a.selectbox("Fase atual", list(range(0, 10)), index=2, format_func=lambda f: "ALFA (0)" if f == 0 else f"Fase {f}")
        idade = b.number_input("Idade", 6, 30, 11)
        ano_ing = c.number_input("Ano de ingresso", 2010, 2026, 2022)
        a, b, c = st.columns(3)
        genero = a.selectbox("Gênero", ["Feminino", "Masculino"])
        inst = b.selectbox("Escola", ["Pública", "Privada/Bolsa", "Outros/Concluiu EM"])
        ano_ref = c.number_input("Ano da avaliação", 2022, 2030, 2024)
        defas_sug = int(fase - fase_ideal(idade))
        defas = st.number_input("Defasagem atual (fase atual − fase ideal)", -6, 4, defas_sug, key=f"def_{fase}_{idade}",
                                help="Pré-calculada pela idade; ajuste se a fase ideal (série escolar) for diferente.")

        st.markdown("**Indicadores do PEDE (0 a 10)**")
        a, b, c, d = st.columns(4)
        ida = a.number_input("IDA · aprendizagem", 0.0, 10.0, 6.0, 0.1)
        ieg = b.number_input("IEG · engajamento", 0.0, 10.0, 8.0, 0.1)
        ipv = c.number_input("IPV · ponto de virada", 0.0, 10.0, 7.3, 0.1)
        ips = d.number_input("IPS · psicossocial", 0.0, 10.0, 6.5, 0.1)
        a, b, c, d = st.columns(4)
        iaa = a.number_input("IAA · autoavaliação", 0.0, 10.0, 8.5, 0.1)
        ipp = b.number_input("IPP · psicopedagógico", 0.0, 10.0, 7.5, 0.1, help="Usado apenas para estimar o INDE.")
        mat = c.number_input("Nota Matemática", 0.0, 10.0, 6.0, 0.1)
        por = d.number_input("Nota Português", 0.0, 10.0, 6.3, 0.1)
        ian = ian_de(defas)
        inde_calc = inde_estimado(fase, ian, ida, ieg, iaa, ips, ipp, ipv)
        inde = st.number_input("INDE (pré-calculado pela fórmula oficial)", 0.0, 10.0, float(round(inde_calc, 2)), 0.01,
                               key=f"inde_{round(inde_calc, 2)}")

    registro = {"FASE": fase, "IDADE": idade, "ANOS_NA_PM": max(ano_ref - ano_ing, 0), "DEFASAGEM": defas,
                "INDE": inde, "IAN": ian, "IDA": ida, "IEG": ieg, "IAA": iaa, "IPS": ips, "IPV": ipv,
                "MAT": mat, "POR": por, "GENERO": genero, "INSTITUICAO_GRUPO": inst}
    prob = float(prever(pd.DataFrame([registro]))[0])
    faixa, cor = faixa_risco(prob)

    with col_out:
        st.subheader("Resultado")
        st.markdown(f"""<div class="bloco">
            <div class="small">Probabilidade de risco de defasagem no próximo ciclo</div>
            <div class="prob" style="color:{cor}">{prob:.0%}</div>
            <span class="faixa" style="background:{cor}">Risco {faixa}</span>
            <div class="small" style="margin-top:10px">IAN {ian:.1f} · INDE {inde:.2f} · fase ideal para {idade} anos: {int(fase_ideal(idade))}</div>
            </div>""", unsafe_allow_html=True)
        st.progress(min(prob, 1.0))
        st.markdown("**Ação sugerida**")
        st.info(ACOES[faixa])
        msgs = alertas(registro)
        if msgs:
            st.markdown("**Sinais de atenção identificados**")
            for m in msgs:
                st.markdown(f"- {m}")
        else:
            st.success("Nenhum sinal de atenção relevante nos indicadores informados.")

# ----------------------------------------------------------------------------
ALIASES = {
    "FASE": ["FASE", "Fase"], "IDADE": ["IDADE", "Idade", "Idade 22"], "ANO_INGRESSO": ["ANO_INGRESSO", "Ano ingresso"],
    "DEFASAGEM": ["DEFASAGEM", "Defasagem", "Defas"], "INDE": ["INDE", "INDE 2024", "INDE 2023", "INDE 22"],
    "IAN": ["IAN"], "IDA": ["IDA"], "IEG": ["IEG"], "IAA": ["IAA"], "IPS": ["IPS"], "IPV": ["IPV"],
    "MAT": ["MAT", "Mat", "Matem"], "POR": ["POR", "Por", "Portug"], "GENERO": ["GENERO", "Gênero"],
    "INSTITUICAO_GRUPO": ["INSTITUICAO_GRUPO", "Instituição de ensino"], "RA": ["RA"], "NOME": ["NOME", "Nome", "Nome Anonimizado"],
    "ANOS_NA_PM": ["ANOS_NA_PM"],
}


def normalizar_planilha(raw: pd.DataFrame, ano_ref: int) -> pd.DataFrame:
    df = pd.DataFrame(index=raw.index)
    for alvo, opcoes in ALIASES.items():
        for o in opcoes:
            if o in raw.columns:
                df[alvo] = raw[o]
                break
    df["FASE"] = df["FASE"].map(parse_fase) if "FASE" in df else np.nan
    df["IDADE"] = df["IDADE"].map(parse_idade) if "IDADE" in df else np.nan
    if "ANOS_NA_PM" not in df:
        df["ANOS_NA_PM"] = (ano_ref - pd.to_numeric(df.get("ANO_INGRESSO"), errors="coerce")).clip(lower=0)
    if "DEFASAGEM" not in df:
        df["DEFASAGEM"] = df["FASE"] - df["IDADE"].map(fase_ideal)
    df["DEFASAGEM"] = pd.to_numeric(df["DEFASAGEM"], errors="coerce")
    if "IAN" not in df:
        df["IAN"] = df["DEFASAGEM"].map(lambda d: np.nan if pd.isna(d) else ian_de(d))
    for c in ["INDE", "IAN", "IDA", "IEG", "IAA", "IPS", "IPV", "MAT", "POR"]:
        df[c] = pd.to_numeric(df.get(c), errors="coerce")
    df.loc[df["IAA"] == 0, "IAA"] = np.nan
    df["GENERO"] = df["GENERO"].map(padroniza_genero) if "GENERO" in df else "Não informado"
    if "INSTITUICAO_GRUPO" in df:
        ja_ok = df["INSTITUICAO_GRUPO"].isin(["Pública", "Privada/Bolsa", "Outros/Concluiu EM"])
        df.loc[~ja_ok, "INSTITUICAO_GRUPO"] = df.loc[~ja_ok, "INSTITUICAO_GRUPO"].map(padroniza_instituicao)
    return df


with aba2:
    st.subheader("Previsão em lote")
    st.write("Envie um **CSV ou Excel** com uma linha por aluno. Aceita o modelo abaixo **ou** a própria aba do PEDE "
             "(colunas `Fase`, `Idade`, `Ano ingresso`, `Defasagem`, `IDA`, `IEG`, `IPV`, `Mat`, `Por`…).")
    modelo_csv = pd.DataFrame([{"RA": "RA-EXEMPLO", "FASE": 3, "IDADE": 13, "ANO_INGRESSO": 2022, "DEFASAGEM": 0, "INDE": 7.1,
                                "IDA": 6.0, "IEG": 8.0, "IAA": 8.5, "IPS": 6.5, "IPV": 7.2, "MAT": 5.8, "POR": 6.4,
                                "GENERO": "Feminino", "INSTITUICAO_GRUPO": "Pública"}])
    st.download_button("⬇️ Baixar planilha modelo (CSV)", modelo_csv.to_csv(index=False).encode("utf-8"), "modelo_alunos.csv", "text/csv")
    ano_lote = st.number_input("Ano de referência dos dados", 2022, 2030, 2024, key="ano_lote")
    arq = st.file_uploader("Planilha de alunos", type=["csv", "xlsx"])
    if arq is not None:
        try:
            if arq.name.endswith(".csv"):
                txt = arq.getvalue().decode("utf-8", errors="ignore")
                raw = pd.read_csv(io.StringIO(txt), sep=None, engine="python")
            else:
                xls = pd.ExcelFile(arq)
                aba = st.selectbox("Aba", xls.sheet_names, index=len(xls.sheet_names) - 1)
                raw = xls.parse(aba)
            df = normalizar_planilha(raw, ano_lote)
            validos = df[["IDA", "IPV"]].notna().any(axis=1) & df["FASE"].notna() & df["IDADE"].notna()
            df = df[validos].copy()
            df["PROB_RISCO"] = prever(df)
            df["FAIXA"] = df["PROB_RISCO"].map(lambda p: faixa_risco(p)[0])
            ordem = ["Baixo", "Atenção", "Alto", "Muito alto"]
            cont = df["FAIXA"].value_counts().reindex(ordem, fill_value=0)
            k = st.columns(4)
            for i, f in enumerate(ordem):
                k[i].metric(f"Risco {f}", int(cont[f]), f"{cont[f] / len(df):.0%}", delta_color="off")
            st.caption(f"{len(df)} alunos avaliados · {int((~validos).sum())} linhas ignoradas (sem fase/idade ou sem indicadores).")
            cols = [c for c in ["RA", "NOME", "FASE", "IDADE", "DEFASAGEM", "IDA", "IEG", "IPV", "INDE", "PROB_RISCO", "FAIXA"] if c in df]
            show = df.sort_values("PROB_RISCO", ascending=False)[cols]
            st.dataframe(show.style.format({"PROB_RISCO": "{:.0%}", "IDA": "{:.1f}", "IEG": "{:.1f}", "IPV": "{:.1f}", "INDE": "{:.2f}",
                                            "FASE": "{:.0f}", "IDADE": "{:.0f}"}),
                         width="stretch", height=420)
            st.download_button("⬇️ Baixar resultado (CSV)", show.to_csv(index=False).encode("utf-8"), "previsao_risco.csv", "text/csv")
        except Exception as e:  # noqa: BLE001
            st.error(f"Não foi possível ler a planilha: {e}")

# ----------------------------------------------------------------------------
with aba3:
    arq_prev = ROOT / "data" / "processed" / "previsao_risco_2025.csv"
    if arq_prev.exists():
        prev = pd.read_csv(arq_prev)
        prev["FAIXA"] = prev["PROB_RISCO"].map(lambda p: faixa_risco(p)[0])
        st.subheader("Risco previsto para 2025 · alunos avaliados no PEDE 2024")
        ordem = ["Baixo", "Atenção", "Alto", "Muito alto"]
        cont = prev["FAIXA"].value_counts().reindex(ordem, fill_value=0)
        k = st.columns(4)
        for i, f in enumerate(ordem):
            k[i].metric(f"Risco {f}", int(cont[f]), f"{cont[f] / len(prev):.0%}", delta_color="off")
        c1, c2 = st.columns(2)
        por_fase = prev.assign(ALTO=prev["PROB_RISCO"] >= LIMIAR).groupby("FASE")["ALTO"].mean().mul(100).round(0)
        por_fase.index = ["ALFA" if f == 0 else f"Fase {int(f)}" for f in por_fase.index]
        c1.markdown("**% de alunos com risco Alto/Muito alto por fase**")
        c1.bar_chart(por_fase, color=LARANJA)
        c2.markdown("**Probabilidade média de risco por defasagem atual**")
        c2.bar_chart(prev.groupby("DEFASAGEM")["PROB_RISCO"].mean().mul(100).round(0), color=AZUL)
        fases = ["Todas"] + sorted(prev["FASE"].dropna().unique().tolist())
        filtro = st.selectbox("Filtrar fase", fases, format_func=lambda f: f if f == "Todas" else ("ALFA" if f == 0 else f"Fase {int(f)}"))
        v = prev if filtro == "Todas" else prev[prev["FASE"] == filtro]
        st.dataframe(v.sort_values("PROB_RISCO", ascending=False).head(50).style.format(
            {"PROB_RISCO": "{:.0%}", "FASE": "{:.0f}", "IDADE": "{:.0f}", "INDE": "{:.2f}", "IDA": "{:.1f}", "IEG": "{:.1f}"}),
            width="stretch", height=380)
    else:
        st.info("Arquivo de previsões não encontrado. Execute o notebook 02 para gerá-lo.")

st.divider()
st.caption("Modelo de apoio à decisão — não substitui a avaliação pedagógica e psicológica da equipe Passos Mágicos. "
           "Dados do PEDE anonimizados. Grupo: Guilherme Cutrim Batista · João Victor Castro Rodrigues · "
           "João Victor Gomes Dominici · Luis Henrique Carvalho Shikasho · Vítor Francisco Pirani Silva.")
