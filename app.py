"""INDUSTRIAL INTELLIGENCE — Smart Predictive Maintenance. Ponto de entrada: python -m streamlit run app.py"""
import streamlit as st

st.set_page_config(page_title="Industrial Intelligence", layout="wide", initial_sidebar_state="expanded")

from ui import page_dicionario, page_fluxo, page_importar, page_modelos, page_painel, page_perfis, page_series, page_simulador, page_sobre, state, theme  # noqa: E402

PAGES = {
    "Painel geral": page_painel, "Simulador manual": page_simulador, "Importar dados": page_importar,
    "Modelos & Avaliação": page_modelos, "Simulação de séries": page_series, "Perfis de máquina": page_perfis,
    "Fluxo de manutenção": page_fluxo, "Dicionário de variáveis": page_dicionario, "Sobre o projeto": page_sobre,
}


def main():
    state.init()
    theme.inject()
    with st.sidebar:
        st.markdown('<div class="brand">INDUSTRIAL INTELLIGENCE</div>', unsafe_allow_html=True)
        st.caption("Manutenção preditiva · Tema 7")
        page = st.radio("Navegação", list(PAGES), label_visibility="collapsed")
        st.divider()
        st.caption(f"Perfil: **{st.session_state.profile_name}**  \nModelo ML: **{state.model_status()}**")
        st.caption("Demonstração e análise. Não controla máquinas.")
    PAGES[page].render()


main()
