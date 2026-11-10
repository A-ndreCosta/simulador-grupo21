import pandas as pd
import streamlit as st

ROWS = [
    ("UDI", "Identificador único da linha", "inteiro", "—", "ID", "—"),
    ("Product ID", "Código do produto (L/M/H + número)", "texto", "—", "ID", "—"),
    ("Type", "Qualidade do produto: L (baixa), M (média), H (alta)", "categórica", "—", "sensor/entrada", "—"),
    ("Air temperature [K]", "Temperatura do ar (T_ar)", "decimal", "K", "sensor", "—"),
    ("Process temperature [K]", "Temperatura do processo (T_proc)", "decimal", "K", "sensor", "—"),
    ("Rotational speed [rpm]", "Velocidade de rotação (Rot)", "inteiro", "rpm", "sensor", "—"),
    ("Torque [Nm]", "Binário (Torque)", "decimal", "Nm", "sensor", "—"),
    ("Tool wear [min]", "Desgaste acumulado da ferramenta (Desgaste)", "inteiro", "min", "sensor", "—"),
    ("Machine failure", "Falha da máquina (Falha): 1 se ocorreu alguma falha", "binária 0/1", "—", "etiqueta", "—"),
    ("TWF", "Falha por desgaste da ferramenta", "binária 0/1", "—", "etiqueta", "—"),
    ("HDF", "Falha por dissipação de calor", "binária 0/1", "—", "etiqueta", "—"),
    ("PWF", "Falha de potência", "binária 0/1", "—", "etiqueta", "—"),
    ("OSF", "Falha por sobrecarga", "binária 0/1", "—", "etiqueta", "—"),
    ("RNF", "Falha aleatória (sem regra determinística)", "binária 0/1", "—", "etiqueta", "—"),
    ("dT", "Diferença de temperatura", "decimal", "K", "derivada", "T_proc − T_ar"),
    ("Potencia", "Potência mecânica estimada", "decimal", "W", "derivada", "Torque · Rot · 2π / 60"),
    ("Sobrecarga", "Indicador de sobrecarga acumulada", "decimal", "min·Nm", "derivada", "Desgaste · Torque"),
]


def render():
    st.title("Dicionário de variáveis")
    st.caption("14 variáveis originais do AI4I 2020 e 3 derivadas (calculadas automaticamente, nunca editáveis). "
               "Os identificadores e as etiquetas de modo de falha nunca são usados como entrada do ML.")
    df = pd.DataFrame(ROWS, columns=["Nome", "Descrição", "Tipo", "Unidade", "Papel", "Fórmula"])
    st.dataframe(df, width="stretch", hide_index=True)
    st.subheader("Regras industriais")
    st.markdown("- **PWF:** Potência < 3500 W ou > 9000 W (limites do perfil).\n- **HDF:** dT < 8,6 K **e** Rot < 1380 rpm.\n"
                "- **OSF:** Sobrecarga ≥ limite do tipo (perfil «Trabalho»: L 11000, M 12337, H 13234; «UCI»: 11000 / 12000 / 13000 min·Nm).\n"
                "- **TWF:** Desgaste ≥ 198 min — opcional e desativada por defeito (pode gerar falsos alarmes).\n"
                "- **RNF:** sem regra (a falha é aleatória no dataset).")
