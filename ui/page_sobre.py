import streamlit as st


def render():
    st.title("Sobre o projeto")
    st.markdown("""
**INDUSTRIAL INTELLIGENCE — Smart Predictive Maintenance** é uma aplicação de demonstração e análise do projeto académico
*Tema 7 — Automação Inteligente e Deteção de Falhas*: um sistema híbrido de dois níveis (regras industriais + Machine Learning).

**Como decide:** nível 1, regras críticas (HDF, PWF, OSF) → «FALHA (regra crítica)»; nível 2, ML sobre o resto → «ALERTA (ML)» se a probabilidade ≥ τ, senão «NORMAL».
Os limiares τ_F1 e τ_R90 são escolhidos só no treino; o conjunto de teste nunca os influencia.

**Limitações (leia antes de tirar conclusões)**
- O AI4I 2020 é um dataset **sintético** (UCI); os resultados não provam desempenho numa máquina real.
- Cada linha é independente: **não é uma série temporal**. As séries geradas na app são simuladas e não validam antecipação temporal.
- Não houve validação em equipamento real. Os resultados dependem dos dados, do perfil e dos parâmetros.
- Regras e ML podem errar; um alerta **não** é uma falha confirmada. Há falhas (por exemplo RNF) que nenhum método consegue prever.
- A «probabilidade» do SVM é uma transformação da função de decisão, não está calibrada.
- Perfis, históricos e registos ficam apenas na sessão do browser, salvo exportação manual.
- A app **não controla máquinas**, não implementa comandos PLC/SCADA e **não substitui** sistemas de segurança certificados.
""")
