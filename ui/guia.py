"""Guia rápido de utilização (mostrado no Painel geral e na página Sobre)."""
import streamlit as st

GUIA = """
1. **Painel geral.** A aplicação abre com o AI4I 2020 já carregado (10 000 leituras sintéticas). Veja quantas leituras estão em falha crítica ou em alerta e que regras dispararam.
2. **Simulador manual.** Escolha um cenário (Normal, Risco térmico, Potência anormal, Sobrecarga) ou altere os valores dos sensores. A caixa de estado mostra o resultado e a explicação diz que regra disparou.
3. **Modelos & Avaliação.** Carregue em «Treinar modelos» (alguns segundos a cerca de um minuto). Aparecem as métricas no conjunto de teste, as matrizes de confusão e o sistema híbrido (variantes R e G, com os limiares τ_F1 e τ_R90).
4. **Voltar ao Painel geral.** Depois de treinar, escolha um método de ML ou híbrido em «Método de deteção» e ajuste o limiar τ.
5. **Opcional.** «Importar dados» aceita o seu CSV/XLSX (há um modelo para descarregar); «Simulação de séries» gera séries simuladas; «Fluxo de manutenção» permite registar intervenções.

*Nota: os dados são sintéticos e a aplicação só apoia a análise; não controla máquinas.*
"""


def render_guide(expanded: bool = False):
    with st.expander("Como usar (1 minuto)", expanded=expanded):
        st.markdown(GUIA)
