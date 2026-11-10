import pandas as pd
import streamlit as st

from src.rules import analyze_one, pt
from src.simulation import SCENARIOS
from src.validation import INPUT_LIMITS, validate_inputs
from ui import state, theme

FIELDS = [("T_ar", "Temperatura do ar (K)"), ("T_proc", "Temperatura do processo (K)"), ("Rot", "Rotação (rpm)"),
          ("Torque", "Binário (Nm)"), ("Desgaste", "Desgaste da ferramenta (min)")]


def _load(values: dict):
    st.session_state.sim_Type = values["Type"]
    for k, _ in FIELDS:
        st.session_state[f"sim_{k}"] = float(values[k])


def render():
    theme.header("Simulador manual", "Escolha um cenário ou introduza uma leitura e veja que regras dispara. As variáveis derivadas são calculadas automaticamente.")
    ss = st.session_state
    if "sim_Type" not in ss:
        _load(SCENARIOS["Normal"])
    cols = st.columns(5)
    for (name, tip), col in zip(list(SCENARIOS.items()) + [("Personalizado", None)], cols):
        if name == "Personalizado":
            col.button("Repor valores", on_click=_load, args=(SCENARIOS["Normal"],), width="stretch")
        else:
            col.button(name, on_click=_load, args=(SCENARIOS[name],), width="stretch", key=f"sc_{name}")
    left, right = st.columns([1, 1])
    with left:
        st.selectbox("Tipo de produto (qualidade)", ["L", "M", "H"], key="sim_Type")
        for key, label in FIELDS:
            lo, hi, olo, ohi, unit = INPUT_LIMITS[key]
            st.number_input(label, min_value=float(lo), max_value=float(hi), step=0.1, key=f"sim_{key}",
                            help=f"Permitido: {lo:g} a {hi:g} {unit}. Observado no AI4I: {olo:g} a {ohi:g} {unit}.")
    vals = dict(Type=ss.sim_Type, **{k: ss[f"sim_{k}"] for k, _ in FIELDS})
    errors, warnings = validate_inputs(**vals)
    with right:
        for e in errors:
            st.error(e)
        for w in warnings:
            st.warning(w)
        if errors:
            st.stop()
        profile = state.profile()
        r = analyze_one(**vals, profile=profile)
        st.caption(f"Perfil ativo: **{ss.profile_name}**")
        theme.state_box(r["estado"])
        d = r["derivadas"]
        c = st.columns(3)
        with c[0]: theme.card("dT (K)", pt(d['dT']), "T_proc − T_ar")
        with c[1]: theme.card("Potência (W)", pt(d['Potencia']), "Torque·Rot·2π/60")
        with c[2]: theme.card("Sobrecarga (min·Nm)", pt(d['Sobrecarga']), "Desgaste·Torque")
        st.write("**Explicação:** " + r["explicacao"])
        st.table(pd.DataFrame(r["detalhe"]).set_index("Regra"))
        if ss.trained is not None or ss.hybrids:
            st.markdown("**Com o ML treinado**")
            row = pd.DataFrame([vals])
            from src.detection import ALL_METHODS, method_available, run_method
            out = []
            for m in ALL_METHODS[2:]:
                if method_available(m, ss.trained, ss.hybrids):
                    res = run_method(m, row, profile, ss.trained, ss.hybrids, ss.tau).iloc[0]
                    out.append({"método": m, "estado": res["estado"], "p_ML": None if pd.isna(res["p_ML"]) else round(float(res["p_ML"]), 3)})
            st.dataframe(pd.DataFrame(out), width="stretch", hide_index=True)
        b = st.columns(3)
        if b[0].button("Guardar", help="Guardar no histórico (só nesta sessão)"):
            ss.history.append({**r["entradas"], **{k: round(v, 3) for k, v in d.items()}, "regras": "+".join(r["regras_acionadas"]) or "—",
                               "estado": r["estado"], "perfil": ss.profile_name})
            st.success("Leitura guardada (apenas nesta sessão).")
        if b[1].button("Comparar", help="Adicionar à comparação"):
            ss.compare.append({"leitura": len(ss.compare) + 1, **r["entradas"], **{k: round(v, 2) for k, v in d.items()}, "regras": "+".join(r["regras_acionadas"]) or "—", "estado": r["estado"]})
    if ss.compare:
        st.subheader("Comparação de leituras")
        st.dataframe(pd.DataFrame(ss.compare), width="stretch", hide_index=True)
        if st.button("Limpar comparação"):
            ss.compare = []
            st.rerun()
    if ss.history:
        st.subheader("Histórico (esta sessão)")
        h = pd.DataFrame(ss.history)
        st.dataframe(h, width="stretch", hide_index=True)
        st.download_button("Exportar histórico (CSV)", h.to_csv(index=False).encode("utf-8-sig"), "historico_simulador.csv", "text/csv")
