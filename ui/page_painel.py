import pandas as pd
import plotly.express as px
import streamlit as st

from src.detection import ALL_METHODS, method_available
from src.features import add_derived_features
from src.rules import STATE_NORMAL
from ui import state, theme


def render():
    st.title("Painel geral")
    ss = st.session_state
    c = st.columns([2, 2, 1])
    avail = [m for m in ALL_METHODS if method_available(m, ss.trained, ss.hybrids)]
    if ss.method not in avail:
        ss.method = avail[0]
    ss.method = c[0].selectbox("Método de deteção", avail, index=avail.index(ss.method),
                               help="Os métodos de ML e híbridos aparecem depois de treinados na página Modelos & Avaliação.")
    ss.tau = c[1].slider("Limiar τ do ML (0–1)", 0.0, 1.0, float(ss.tau), 0.01) if "Híbrido" in ss.method else ss.tau
    c[2].write(""); c[2].caption(f"Perfil: **{ss.profile_name}**")

    an = state.get_analysis()
    if an is None:
        st.info("Ainda não há dados carregados. Vá a **Importar dados** para carregar um CSV/XLSX, o AI4I 2020 ou o demo sintético, "
                "ou use o **Simulador manual** para analisar uma leitura.")
        k = st.columns(4)
        with k[0]: theme.card("Leituras", "—", "sem dados")
        with k[1]: theme.card("Estado do modelo", state.model_status(), "ML", "ml")
        with k[2]: theme.card("Perfil ativo", ss.profile_name, "limites das regras")
        with k[3]: theme.card("Última execução", ss.last_run or "—", "")
        return

    df, meta = an["df"], ss.dataset_meta
    if meta.get("sintetico"):
        theme.synthetic_badge(meta.get("rotulo", "Dados sintéticos"))
    st.caption(f"Fonte: {meta['nome']} · {meta.get('origem', '')}")
    n = len(df)
    fails = int((df["estado"] == "FALHA (regra crítica)").sum())
    alerts = int(df["estado"].isin(["ALERTA (ML)", "ALERTA (regra TWF)"]).sum())
    normal = int((df["estado"] == STATE_NORMAL).sum())
    k = st.columns(6)
    with k[0]: theme.card("Leituras", f"{n:,}".replace(",", " "), "analisadas")
    with k[1]: theme.card("Falhas (regras)", str(fails), "estado crítico", "fail")
    with k[2]: theme.card("Alertas", str(alerts), "ML / TWF", "warn")
    with k[3]: theme.card("Normais", f"{100 * normal / n:.1f} %".replace(".", ","), f"{normal} leituras", "ok")
    with k[4]: theme.card("Modelo de ML", state.model_status(), an["method"], "ml")
    with k[5]: theme.card("Última execução", an["when"].split(" ")[1], an["when"].split(" ")[0])
    if an["method"] != ss.method:
        st.warning("O método escolhido deixou de estar disponível; a mostrar regras críticas.")

    left, right = st.columns(2)
    dist = df["estado"].value_counts().reset_index()
    dist.columns = ["estado", "leituras"]
    fig = px.bar(dist, x="estado", y="leituras", color="estado", color_discrete_map=theme.STATE_COLORS, title="Distribuição de estados")
    fig.update_layout(showlegend=False, xaxis_title="", yaxis_title="Nº de leituras")
    left.plotly_chart(fig, width="stretch")
    rules = df["regras_acionadas"].astype(str).str.split("+").explode()
    rules = rules[rules != "—"].value_counts().reset_index()
    rules.columns = ["regra", "ativações"]
    if len(rules):
        fig2 = px.bar(rules, x="regra", y="ativações", title="Regras ativadas (uma leitura pode ativar várias)", color_discrete_sequence=[theme.BLUE])
        right.plotly_chart(fig2, width="stretch")
    else:
        right.info("Nenhuma regra foi ativada neste conjunto (ou o método escolhido não usa regras).")

    d = add_derived_features(df)
    cols = st.multiselect("Evolução dos sensores (ordem das leituras)", ["T_ar", "T_proc", "Rot", "Torque", "Desgaste", "dT", "Potencia", "Sobrecarga"], default=["Torque"])
    if cols:
        view = d.head(2000)
        fig3 = px.line(view.reset_index(names="leitura"), x="leitura", y=cols, title="Evolução (primeiras 2 000 leituras)")
        st.plotly_chart(fig3, width="stretch")
        if meta.get("nome", "").startswith("ai4i"):
            st.caption("Nota: no AI4I cada linha é independente; a ordem das linhas não é uma série temporal.")
    st.subheader("Leituras recentes")
    show = [c for c in ["Type", "T_ar", "T_proc", "Rot", "Torque", "Desgaste", "estado", "regras_acionadas", "p_ML"] if c in df.columns]
    st.dataframe(df.tail(20)[show], width="stretch")
    if ss.history:
        st.subheader("Histórico do simulador manual (esta sessão)")
        st.dataframe(pd.DataFrame(ss.history).tail(10), width="stretch")
