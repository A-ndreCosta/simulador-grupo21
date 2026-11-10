import pandas as pd
import streamlit as st

from src.features import REQUIRED_COLUMNS
from src.file_import import ALIASES, REQUIRED_TARGETS, ImportFileError, apply_mapping, detect_mapping, read_table, template_csv
from src.validation import clean_dataframe, validate_dataframe
from ui import state

LABEL_NOTE = "Sem coluna de etiqueta (Falha): as regras funcionam, mas **não** são apresentadas métricas de avaliação."


def _finish(df: pd.DataFrame, meta: dict, drop_invalid: bool):
    rep = validate_dataframe(df)
    for e in rep.errors:
        st.error(e)
    for w in rep.warnings:
        st.warning(w)
    if rep.errors:
        return
    if rep.n_invalid:
        st.error(f"{rep.n_invalid} linhas inválidas (relatório abaixo). Nenhuma linha é descartada em silêncio.")
        st.dataframe(rep.row_issues.head(500), width="stretch", hide_index=True)
        st.download_button("Descarregar relatório de erros (CSV)", rep.row_issues.to_csv(index=False).encode("utf-8-sig"), "erros_importacao.csv")
        if not drop_invalid:
            st.info("Corrija o ficheiro ou marque a opção para excluir explicitamente as linhas inválidas.")
            return
    clean, rejected = clean_dataframe(df, rep, drop_invalid=drop_invalid)
    if len(rejected):
        st.warning(f"{len(rejected)} linhas excluídas por escolha sua.")
    if not rep.has_label:
        st.info(LABEL_NOTE)
    else:
        st.caption(f"Etiquetas: {rep.label_counts}")
    st.dataframe(clean.head(20), width="stretch")
    if st.button("Usar estes dados", type="primary"):
        state.set_dataset(clean.reset_index(drop=True), {**meta, "linhas": len(clean), "etiquetado": rep.has_label})
        st.success("Dados carregados. Os modelos anteriores foram invalidados.")


def render():
    st.title("Importar dados")
    st.caption("CSV (separador , ou ;) ou XLSX. Formato AI4I reconhecido automaticamente; ficheiros .xls não são suportados.")
    c = st.columns(3)
    if c[0].button("Carregar AI4I 2020 (UCI)", width="stretch"):
        path = state.DATA_DIR / "ai4i2020.csv"
        if not path.exists():
            st.error("Ficheiro data/ai4i2020.csv não encontrado.")
        else:
            st.session_state.pending = ("ai4i2020.csv", path.read_bytes(), dict(nome="ai4i2020.csv", origem="UCI AI4I 2020 Predictive Maintenance Dataset — dataset SINTÉTICO publicado pela UCI, não vem de uma máquina real.", sintetico=True, rotulo="AI4I 2020 — dataset sintético (UCI)"))
    if c[1].button("Carregar demo sintético (sem etiquetas)", width="stretch"):
        path = state.DATA_DIR / "demo_sintetico.csv"
        if not path.exists():
            st.error("Ficheiro data/demo_sintetico.csv não encontrado.")
        else:
            st.session_state.pending = ("demo_sintetico.csv", path.read_bytes(), dict(nome="demo_sintetico.csv", origem="Gerado por simulação (sementes fixas). NÃO é um dataset real.", sintetico=True, rotulo="Demo sintético — dados simulados"))
    c[2].download_button("Descarregar modelo CSV", template_csv(), "modelo_importacao.csv", "text/csv", width="stretch")
    up = st.file_uploader("Ou carregue o seu ficheiro", type=["csv", "xlsx"])
    if up is not None:
        st.session_state.pending = (up.name, up.getvalue(), dict(nome=up.name, origem="Ficheiro carregado pelo utilizador.", sintetico=False, rotulo=""))
    pend = st.session_state.get("pending")
    if not pend:
        return
    name, raw, meta = pend
    try:
        df, info = read_table(raw, name)
    except ImportFileError as e:
        st.error(str(e))
        return
    st.success(f"Lido: {info.get('formato')} · {len(df)} linhas · {len(df.columns)} colunas" + (f" · codificação {info.get('codificacao')} · separador «{info.get('separador')}»" if info.get("formato") == "CSV" else ""))
    mapping, missing, ambiguous = detect_mapping(list(df.columns))
    with st.expander("Mapeamento de colunas", expanded=bool(missing or ambiguous)):
        options = [""] + list(df.columns)
        for tgt in REQUIRED_TARGETS + ["Falha"]:
            cur = mapping.get(tgt, "")
            sel = st.selectbox(tgt + (" (opcional — etiqueta de falha 0/1)" if tgt == "Falha" else ""), options,
                               index=options.index(cur) if cur in options else 0, key=f"map_{tgt}_{name}")
            if sel:
                mapping[tgt] = sel
            else:
                mapping.pop(tgt, None)
    miss = [t for t in REQUIRED_TARGETS if t not in mapping]
    if miss:
        st.error("Faltam colunas obrigatórias: " + ", ".join(miss) + ". Atribua-as no mapeamento.")
        return
    try:
        mapped = apply_mapping(df, mapping)
    except ImportFileError as e:
        st.error(str(e))
        return
    drop = st.checkbox("Excluir explicitamente as linhas inválidas (por defeito, o ficheiro é recusado)")
    _finish(mapped, meta, drop)
