import streamlit as st

from src.maintenance import BRANCH_HELP, CONFIRMACOES, ESTADOS, add_log_entry, branches_for, flow_dot
from ui import state


def render():
    st.title("Fluxo de manutenção")
    ss = st.session_state
    st.caption("Fluxo do projeto: Máquina → Funciona? → (Não) Porque motivo? → ramo → Peça → Responsável → Arranja → Confirmação. "
               "A app apenas **apoia a análise**; não comanda máquinas.")
    an = state.get_analysis()
    active = []
    if an is not None:
        rules = an["df"]["regras_acionadas"].astype(str).str.split("+").explode()
        active = sorted(set(r for r in rules if r not in ("—", "nan")))
        if active:
            st.write("Regras ativadas nos dados atuais: " + ", ".join(f"**{r}**" for r in active))
    st.graphviz_chart(flow_dot(branches_for(active)))
    st.subheader("Correspondência regra → ramo")
    st.table({"Ramo": list(BRANCH_HELP), "Significado": list(BRANCH_HELP.values())})
    st.caption("Ramo «Random» (RNF): não existe regra determinística; um alerta de ML sem regra ativada não identifica a causa e exige inspeção humana.")
    st.subheader("Registo de intervenções (preenchido por si)")
    st.caption("Nada aqui é inventado: só aparece o que registar. Fica apenas nesta sessão; exporte em CSV para guardar.")
    with st.form("log", clear_on_submit=True):
        c = st.columns(5)
        ramo = c[0].selectbox("Ramo", list(BRANCH_HELP))
        peca = c[1].text_input("Peça")
        resp = c[2].text_input("Responsável")
        est = c[3].selectbox("Estado", ESTADOS)
        conf = c[4].selectbox("Confirmação", CONFIRMACOES)
        ok = st.form_submit_button("Adicionar registo")
    if ok:
        try:
            ss.log = add_log_entry(ss.log, ramo, peca, resp, est, conf)
        except ValueError as e:
            st.error(str(e))
    if len(ss.log):
        st.dataframe(ss.log, width="stretch", hide_index=True)
        st.download_button("Exportar registo (CSV)", ss.log.to_csv(index=False).encode("utf-8-sig"), "registo_intervencoes.csv", "text/csv")
    else:
        st.info("Ainda não há registos.")
