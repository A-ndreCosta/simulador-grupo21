import pandas as pd
import plotly.express as px
import streamlit as st

from src.detection import ALL_METHODS, method_available
from src.features import add_derived_features
from src.rules import STATE_NORMAL
from ui import guia, state, theme


def _bar(df, x, y, colors, title):
    fig = px.bar(df, y=x, x=y, orientation="h", color=x, color_discrete_map=colors, text=y, title=title)
    fig.update_traces(textposition="outside", cliponaxis=False)
    fig.update_layout(showlegend=False, xaxis_title="", yaxis_title="", height=300, yaxis=dict(autorange="reversed"))
    fig.update_xaxes(range=[0, float(df[y].max()) * 1.2])
    return fig


def render():
    theme.header("Painel geral", "Resumo da análise dos dados carregados com o método e o perfil escolhidos.")
    ss = st.session_state
    guia.render_guide()
    avail = [m for m in ALL_METHODS if method_available(m, ss.trained, ss.hybrids)]
    if ss.method not in avail:
        ss.method = avail[0]
    with st.container(border=True):
        c = st.columns([3, 2, 1])
        ss.method = c[0].selectbox("Método de deteção", avail, index=avail.index(ss.method),
                                   help="Os métodos de ML e híbridos aparecem depois de treinados na página Modelos & Avaliação.")
        ss.tau = c[1].slider("Limiar τ do ML", 0.0, 1.0, float(ss.tau), 0.01, help="Só se aplica aos métodos híbridos.") if "Híbrido" in ss.method else ss.tau
        c[2].caption("Perfil de regras"); c[2].markdown(f"**{ss.profile_name}**")

    an = state.get_analysis()
    if an is None:
        st.info("Ainda não há dados carregados. Vá a **Importar dados** para carregar um CSV/XLSX, o AI4I 2020 ou o demo sintético, "
                "ou use o **Simulador manual** para analisar uma leitura.")
        return

    df, meta = an["df"], ss.dataset_meta
    if meta.get("sintetico"):
        theme.synthetic_badge(meta.get("rotulo", "Dados sintéticos"))
    st.caption(f"Fonte: {meta['nome']} · {meta.get('origem', '')}")
    n = len(df)
    fails = int((df["estado"] == "FALHA (regra crítica)").sum())
    alerts = int(df["estado"].isin(["ALERTA (ML)", "ALERTA (regra TWF)"]).sum())
    normal = int((df["estado"] == STATE_NORMAL).sum())
    k = st.columns(4)
    with k[0]: theme.card("Leituras analisadas", f"{n:,}".replace(",", " "), f"{an['method']}")
    with k[1]: theme.card("Falhas críticas", str(fails), "regras HDF, PWF e OSF", "fail")
    with k[2]: theme.card("Alertas", str(alerts), "ML ou regra TWF", "warn")
    with k[3]: theme.card("Leituras normais", f"{100 * normal / n:.1f} %".replace(".", ","), f"{normal:,} leituras".replace(",", " "), "ok")
    st.caption(f"Modelo de ML: {state.model_status()} · última execução: {an['when']}")
    if an["method"] != ss.method:
        st.warning("O método escolhido deixou de estar disponível; a mostrar regras críticas.")

    left, right = st.columns(2)
    dist = df["estado"].value_counts().reset_index()
    dist.columns = ["estado", "leituras"]
    left.plotly_chart(_bar(dist, "estado", "leituras", theme.STATE_COLORS, "Estado das leituras"), width="stretch")
    rules = df["regras_acionadas"].astype(str).str.split("+").explode()
    rules = rules[rules != "—"].value_counts().reset_index()
    rules.columns = ["regra", "ativações"]
    if len(rules):
        right.plotly_chart(_bar(rules, "regra", "ativações", {r: theme.BLUE for r in rules["regra"]}, "Regras ativadas (uma leitura pode ativar várias)"), width="stretch")
    else:
        right.info("Nenhuma regra foi ativada neste conjunto (ou o método escolhido não usa regras).")

    d = add_derived_features(df)
    with st.expander("Evolução dos sensores"):
        cols = st.multiselect("Variáveis", ["T_ar", "T_proc", "Rot", "Torque", "Desgaste", "dT", "Potencia", "Sobrecarga"], default=["Torque"])
        if cols:
            fig3 = px.line(d.head(2000).reset_index(names="leitura"), x="leitura", y=cols, title="Primeiras 2 000 leituras")
            st.plotly_chart(fig3, width="stretch")
            if meta.get("nome", "").startswith("ai4i"):
                st.caption("Nota: no AI4I cada linha é independente; a ordem das linhas não é uma série temporal.")
    with st.expander("Leituras recentes"):
        show = [c for c in ["Type", "T_ar", "T_proc", "Rot", "Torque", "Desgaste", "estado", "regras_acionadas", "p_ML"] if c in df.columns]
        st.dataframe(df.tail(20)[show], width="stretch")
    if ss.history:
        with st.expander("Histórico do simulador manual (esta sessão)"):
            st.dataframe(pd.DataFrame(ss.history).tail(10), width="stretch")
