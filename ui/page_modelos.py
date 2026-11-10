import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.detection import compare_on_test
from src.hybrid import HybridSystem
from src.metrics import comparison_table, confusion_matrix_df, recall_vs_false_alarms
from src.ml_model import MODEL_SPECS, TrainingError, check_trainable, split_train_test, train_models
from src.rules import STATE_FAIL
from ui import state, theme


def _train():
    ss = st.session_state
    bar = st.progress(0.0, text="A preparar…")
    try:
        check_trainable(ss.dataset)
        tm = train_models(ss.dataset, progress=lambda f, t: bar.progress(min(f, 1.0) * 0.8, text=t))
        hybrids = {}
        for i, v in enumerate("RG"):
            bar.progress(0.8 + 0.1 * (i + 1), text=f"A treinar o híbrido {v} e a escolher τ no treino…")
            hybrids[v] = HybridSystem(state.profile(), v).fit(tm.train)
        ss.trained, ss.hybrids, ss.train_error = tm, hybrids, None
        ss.train_version += 1
        ss.analysis_cache = None
        bar.progress(1.0, text="Concluído")
    except TrainingError as e:
        ss.train_error = str(e)
        bar.empty()


def render():
    st.title("Modelos & Avaliação")
    ss = st.session_state
    if ss.dataset is None:
        st.info("Carregue dados etiquetados em **Importar dados** (por exemplo o AI4I 2020).")
        return
    meta = ss.dataset_meta
    if not meta.get("etiquetado"):
        st.warning("Os dados carregados não têm etiqueta de falha: não é possível treinar nem calcular precisão/recall/F1. As regras continuam a funcionar no Painel.")
        return
    st.caption(f"Dados: {meta['nome']} · {len(ss.dataset)} leituras · perfil das regras: **{ss.profile_name}** · semente 42 · divisão estratificada 80/20")
    st.write("Os hiperparâmetros são fixos (os escolhidos por validação cruzada no notebook). A app não repete a pesquisa em grelha.")
    if ss.train_error:
        st.error(ss.train_error)
    if st.button("Treinar modelos", type="primary"):
        _train()
        st.rerun()
    tm = ss.trained
    if tm is None:
        st.info("Modelos ainda não treinados. O treino demora cerca de uma dezena de segundos no AI4I.")
        return
    prof = state.profile()
    st.success(f"Treinados em {len(tm.train)} leituras; avaliados uma vez em {len(tm.test)} leituras de teste ({int(tm.test.Falha.sum())} falhas).")
    results = compare_on_test(tm, ss.hybrids, prof)
    st.subheader("Comparação no mesmo conjunto de teste")
    tab = comparison_table(results)
    disp = tab.copy()
    for c in ["precisão", "recall", "F1", "accuracy", "PR-AUC", "ROC-AUC"]:
        disp[c] = [("—" if v != v else f"{v:.3f}") for v in tab[c]]
    st.dataframe(disp, width="stretch")
    st.caption("A accuracy é só complementar (96,6 % das leituras são normais); compare por recall, precisão e F1. "
               "A «probabilidade» do SVM é uma sigmóide da função de decisão, não calibrada. Os τ do híbrido foram escolhidos no treino.")
    for v, h in ss.hybrids.items():
        if h.fallback:
            st.warning(f"Híbrido {v}: {h.fallback}")
        st.write(f"**{h.label}** — τ_F1 = {h.tau_f1:.2f} · τ_R90 = {h.tau_r90:.2f} (escolhidos no treino com validação cruzada; 1,01 = o ML nunca dispara).")
    names = list(results)
    sel = st.selectbox("Matriz de confusão de", names, index=names.index("RF — sensores + derivadas"))
    r = results[sel]
    c = st.columns([1, 1])
    cm = [[r["TN"], r["FP"]], [r["FN"], r["TP"]]]
    fig = px.imshow(cm, text_auto=True, x=["Previsto: normal", "Previsto: alerta/falha"], y=["Real: normal", "Real: falha"],
                    color_continuous_scale=[[0, "#f5f3ff"], [1, theme.VIOLET]], title=sel)
    c[0].plotly_chart(fig, width="stretch")
    c[1].markdown(f"TP **{r['TP']}** · FP **{r['FP']}** · FN **{r['FN']}** · TN **{r['TN']}**  \nLeituras: {r['leituras']} · falhas reais: {r['falhas reais']} · detetadas: {r['detetadas']}")
    st.subheader("Recall vs falsos alarmes (híbrido, no teste)")
    test, y = tm.test, tm.test["Falha"].astype(int)
    from src.rules import critical_mask
    flag = critical_mask(test, prof).to_numpy()
    fig2 = go.Figure()
    for v, h in ss.hybrids.items():
        cur = recall_vs_false_alarms(y, flag, h.proba_ml(test), [t for t in h.curve["tau"]])
        fig2.add_trace(go.Scatter(x=cur["FP"], y=cur["recall"], mode="lines+markers", name=f"Híbrido {v}", marker=dict(size=4)))
    fig2.update_layout(xaxis_title="Falsos alarmes (nº de leituras)", yaxis_title="Recall", title="Cada ponto é um τ; as regras críticas estão sempre incluídas")
    st.plotly_chart(fig2, width="stretch")
    st.caption("Esta curva mostra o compromisso para todos os τ, mas só como ilustração: os τ usados acima foram escolhidos no treino.")
    st.download_button("Exportar comparação (CSV)", tab.to_csv().encode("utf-8-sig"), "comparacao_teste.csv", "text/csv")
