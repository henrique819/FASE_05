# Datathon Fase 5 · Case Passos Mágicos

**POSTECH Data Analytics — FIAP**

Análise da Pesquisa Extensiva do Desenvolvimento Educacional (PEDE 2022, 2023 e 2024) da Associação Passos Mágicos, com storytelling gerencial, respostas às 11 perguntas de negócio e um modelo preditivo de **risco de defasagem** disponibilizado em um app Streamlit.

| Integrante | RM |
|---|---|
| Guilherme Cutrim Batista | 369505 |
| João Victor Castro Rodrigues | 369462 |
| João Victor Gomes Dominici | 370163 |
| Luis Henrique Carvalho Shikasho | 368421 |
| Vítor Francisco Pirani Silva | 36916 |

## Entregáveis

| Entregável | Onde está |
|---|---|
| Código de limpeza e análise | `src/data_prep.py`, `notebooks/01_limpeza_eda_storytelling.ipynb` |
| Notebook do modelo preditivo (feature engineering, treino/teste, modelagem, avaliação) | `notebooks/02_modelo_preditivo_risco_defasagem.ipynb` |
| Apresentação gerencial | `apresentacao/Datathon_Passos_Magicos_Fase5.pptx` e `.pdf` |
| Aplicação Streamlit | `streamlit_app.py` (deploy no Community Cloud) |
| Roteiro do vídeo (até 5 min) | `docs/roteiro_video.md` |

## Estrutura

```
FASE_05/
├── streamlit_app.py                  # app de previsão (entrada do Streamlit Cloud)
├── requirements.txt                  # dependências do app (versões fixadas)
├── requirements-dev.txt              # + bibliotecas para os notebooks
├── .streamlit/config.toml            # tema do app
├── src/
│   ├── data_prep.py                  # limpeza e padronização das 3 abas do PEDE
│   └── features.py                   # feature engineering (usado no treino e no app)
├── notebooks/
│   ├── 01_limpeza_eda_storytelling.ipynb
│   └── 02_modelo_preditivo_risco_defasagem.ipynb
├── data/
│   ├── raw/BASE_DE_DADOS_PEDE_2024_-_DATATHON.xlsx
│   └── processed/                    # pede_long.csv, pares_t_t1.csv, previsao_risco_2025.csv
├── models/                           # modelo_risco_defasagem.joblib + metadata.json
├── reports/figures/                  # gráficos gerados pelos notebooks
├── apresentacao/                     # deck gerencial (PPTX e PDF)
└── docs/                             # dicionário de dados e roteiro do vídeo
```

## Limpeza dos dados

As três abas do PEDE têm nomes de colunas e formatos diferentes. O script `src/data_prep.py` padroniza tudo em uma base longa (uma linha por aluno × ano) e em uma base de pares (mesmo aluno no ano *t* e no ano *t+1*, ligados pelo RA).

| Problema | Tratamento |
|---|---|
| Colunas com nomes diferentes por ano | Mapeamento único (`COLMAP`) |
| `Fase` 2024 mistura fase e turma (`"4M"`, `"ALFA"`) e 2023 vem como `"FASE 3"` | Extração do dígito; ALFA = 0 |
| Idade 2023 parcialmente em formato data (`1900-01-11` = 11 anos) | Conversão pelo dia |
| Gênero `Menina/Menino` vs `Feminino/Masculino` | Padronização |
| Pedra `Agata` e `INCLUIR` | Padronização; pedra derivada do INDE quando ausente |
| Ingressantes sem avaliação (IEG = 0 e demais nulos) | Flag `SEM_AVALIACAO`, fora das médias e do modelo |
| IAA = 0 (190 casos em 2023) | Tratado como não respondido |
| IPP inexistente em 2022 | Mantido nulo; análises de IPP usam 2023-2024 |

## Principais resultados

1. **Defasagem (IAN):** alunos defasados caíram de **70% (2022) para 46% (2024)**; a defasagem severa foi de 3,3% para 0,3%.
2. **Desempenho (IDA):** oscila (6,09 → 6,66 → 6,35), com um vale nas Fases 2 a 4.
3. **Engajamento (IEG):** correlação de 0,46 a 0,56 com IDA; +2,5 pontos de IDA entre o quartil mais e o menos engajado.
4. **Autoavaliação (IAA):** pouco coerente com o desempenho; 47% superestimam em mais de 2 pontos.
5. **Psicossocial (IPS):** com IPS baixo, 33% têm queda de engajamento no ano seguinte (vs 19% com IPS alto).
6. **Psicopedagógico (IPP):** confirma a direção da defasagem, mas discrimina pouco (correlação 0,12 com o IAN).
7. **Ponto de virada (IPV):** puxado por IPP (0,49), IEG (0,27) e IDA (0,22).
8. **INDE:** IDA + IEG + IPS + IPP acima da mediana levam a INDE médio de 8,42 (Topázio).
9. **Modelo preditivo:** detalhes abaixo.
10. **Efetividade:** alunos Topázio foram de 15% para 31%; na coorte presente nos 3 anos, a defasagem média foi de −0,83 para −0,24.
11. **Insight extra:** só 47% dos alunos Quartzo continuam no ano seguinte, contra 84% dos Topázio.

## Modelo preditivo de risco de defasagem

- **Alvo:** aluno **defasado (fase atual < fase ideal)** ou com **defasagem piorando** no ano seguinte.
- **Entradas:** apenas informações do ano atual (sem vazamento): fase, idade, tempo de programa, defasagem, INDE, IAN, IDA, IEG, IAA, IPS, IPV, notas de Matemática e Português, gênero e tipo de escola, mais 8 variáveis criadas (por exemplo, a *defasagem projetada se o aluno não avançar de fase*).
- **Validação:** holdout 80/20 estratificado e agrupado por aluno, validação cruzada de 5 folds e validação temporal (treino 2022→2023, teste 2023→2024).
- **Algoritmo escolhido:** Random Forest (comparado com Regressão Logística e Gradient Boosting).

| Métrica (teste) | Random Forest | Regra "já está defasado" |
|---|---|---|
| ROC-AUC | 0,88 | — |
| Recall | 84% | 69% |
| Acurácia | 78% | 65% |
| F1 | 0,81 | 0,68 |

Na validação temporal a ROC-AUC é 0,84. Recomenda-se re-treinar o modelo a cada novo ciclo do PEDE.

## Como executar

```bash
# 1. Ambiente
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt

# 2. Limpeza (gera data/processed/)
python src/data_prep.py

# 3. Notebooks (análise e modelo; o 02 regrava models/)
jupyter notebook notebooks/

# 4. App
streamlit run streamlit_app.py
```

## Deploy no Streamlit Community Cloud

1. Publique este repositório no GitHub (`henrique819/FASE_05`).
2. Em [share.streamlit.io](https://share.streamlit.io), clique em **Create app** e escolha o repositório, a branch `main` e o arquivo `streamlit_app.py`.
3. Em **Advanced settings**, selecione **Python 3.12**.
4. Clique em **Deploy**.

O modelo foi salvo com scikit-learn 1.8.0; por isso as versões estão fixadas no `requirements.txt`.

## Fontes

- Base PEDE 2022-2024 e dicionário de dados fornecidos pela FIAP / Associação Passos Mágicos (dados anonimizados).
- Relatórios PEDE 2020, 2021 e 2022 (fórmula do INDE e critérios do IAN).
- [Passos Mágicos · Impacto e transparência](https://passosmagicos.org.br/impacto-e-transparencia/)
