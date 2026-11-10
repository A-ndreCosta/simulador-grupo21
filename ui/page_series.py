import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.detection import ALL_METHODS, method_available, run_method
from src.features import add_derived_features
from src.simulation import DEFAULT_NOISE, EVOLUTIONS, SCENARIOS, SENSORS, generate_series
from src.validation import INPUT_LIMITS, validate_inputs
from ui import state, theme


def render():
    st.title("Simulação de séries")
    ss = st.session_state
    theme.synthetic_badge("Série SIMULADA — não são dados de uma máquina")
    st.caption("O AI4I não é uma série temporal (cada linha é independente). Esta simulação ilustra o comportamento do sistema numa evolução hipotética; "
               "**não** valida antecipação temporal de falhas.")
    base_name = st.selectbox("Valores iniciais", list(SCENARIOS), index=0)
    start = dict(SCENARIOS[base_name])
    c = st.columns(5)
    for (k, col) in zip(SENSORS, c):
        lo, hi, *_ , unit = INPUT_LIMITS[k]
        start[k] = col.number_input(f"{k} ({unit})", float(lo), float(hi), float(start[k]), 0.1, key=f"s0_{k}_{base_name}")
    start["Type"] = st.selectbox("Tipo de produto", ["L", "M", "H"], index=["L", "M", "H"].index(start["Type"]))
    c2 = st.columns(3)
    n = int(c2[0].number_input("Número de leituras", 10, 5000, 200, 10))
    seed = int(c2[1].number_input("Semente aleatória", 0, 999999, 42))
    evo = c2[2].selectbox("Evolução", list(EVOLUTIONS))
    drift = dict(EVOLUTIONS[evo])
    with st.expander("Ruído e deriva por passo"):
        noise, cc = {}, st.columns(5)
        for k, col in zip(SENSORS, cc):
            noise[k] = col.number_input(f"ruído σ {k}", 0.0, 1000.0, float(DEFAULT_NOISE[k]), 0.1, key=f"n_{k}")
        if evo == "Personalizado":
            cd = st.columns(5)
            for k, col in zip(SENSORS, cd):
                drift[k] = col.number_input(f"deriva {k}/passo", -100.0, 100.0, 0.0, 0.01, key=f"d_{k}")
        wear = st.number_input("Crescimento do desgaste (min/passo)", 0.0, 50.0, 1.0, 0.1)
    errs, _ = validate_inputs(**start)
    if errs:
        for e in errs:
            st.error(e)
        return
    if st.button("Gerar série", type="primary"):
        ss.series = generate_series(n, start, noise, drift, seed, wear)
        ss.series_meta = dict(seed=seed, evolucao=evo, perfil=ss.profile_name)
    if ss.series is None:
        st.info("Defina os parâmetros e clique em «Gerar série».")
        return
    avail = [m for m in ALL_METHODS if method_available(m, ss.trained, ss.hybrids)]
    method = st.selectbox("Método de deteção", avail)
    s = ss.series
    res = run_method(method, s, state.profile(), ss.trained, ss.hybrids, ss.tau)
    out = pd.concat([add_derived_features(s), res], axis=1)
    var = st.selectbox("Variável a mostrar", ["T_ar", "T_proc", "Rot", "Torque", "Desgaste", "dT", "Potencia", "Sobrecarga"], index=6)
    fig = go.Figure(go.Scatter(x=out["passo"], y=out[var], mode="lines", name=var, line=dict(color=theme.BLUE)))
    for estado, color in theme.STATE_COLORS.items():
        sub = out[out["estado"] == estado]
        if estado != "NORMAL" and len(sub):
            fig.add_trace(go.Scatter(x=sub["passo"], y=sub[var], mode="markers", name=estado, marker=dict(color=color, size=7)))
    fig.update_layout(title=f"{var} — série simulada (semente {ss.series_meta['seed']})", xaxis_title="passo", yaxis_title=var)
    st.plotly_chart(fig, width="stretch")
    first = out.index[out["estado"] != "NORMAL"]
    st.write(f"Primeira leitura com alerta/falha: **{'nenhuma' if len(first) == 0 else int(out.loc[first[0], 'passo'])}** · "
             f"leituras em alerta/falha: {int((out['estado'] != 'NORMAL').sum())} de {len(out)}.")
    st.dataframe(out.head(200), width="stretch", hide_index=True)
    st.download_button("Exportar sequência (CSV, simulada)", out.to_csv(index=False).encode("utf-8-sig"), "serie_simulada.csv", "text/csv")
