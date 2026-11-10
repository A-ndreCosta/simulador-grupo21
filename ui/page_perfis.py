import json

import streamlit as st

from src.config import ProfileError, compare_profiles, duplicate_profile, load_default_profiles, profile_from_json, profile_to_json, validate_profile
from ui import state


def render():
    st.title("Perfis de máquina")
    ss = st.session_state
    st.info("As alterações aplicam-se de imediato às análises, mas **só ficam guardadas nesta sessão do browser**. Use «Exportar perfil» para as guardar num ficheiro.")
    defaults = load_default_profiles()
    names = list(ss.profiles)
    ss.profile_name = st.selectbox("Perfil ativo", names, index=names.index(ss.profile_name))
    p = ss.profiles[ss.profile_name]
    st.caption(p.get("descricao", "") + " · Origem: " + p.get("origem", ""))
    with st.form("edit"):
        c = st.columns(3)
        pmin = c[0].number_input("PWF — potência mínima (W)", 0.0, 1e6, float(p["pwf"]["potencia_min_w"]))
        pmax = c[1].number_input("PWF — potência máxima (W)", 0.0, 1e6, float(p["pwf"]["potencia_max_w"]))
        dt = c[2].number_input("HDF — dT máximo (K)", 0.0, 100.0, float(p["hdf"]["dT_max_k"]))
        c = st.columns(4)
        rot = c[0].number_input("HDF — rotação máxima (rpm)", 0.0, 20000.0, float(p["hdf"]["rot_max_rpm"]))
        osf = {t: c[i + 1].number_input(f"OSF — limite {t} (min·Nm)", 0.0, 1e6, float(p["osf"][t])) for i, t in enumerate("LMH")}
        c = st.columns(2)
        twf = c[0].checkbox("Ativar TWF (desgaste)", value=bool(p["twf"]["ativa"]),
                            help="A TWF é probabilística: um desgaste elevado não é falha certa e pode gerar muitos falsos alarmes.")
        tw = c[1].number_input("TWF — desgaste mínimo (min)", 0.0, 1000.0, float(p["twf"]["desgaste_min"]))
        ok = st.form_submit_button("Aplicar alterações")
    if ok:
        new = json.loads(json.dumps(p))
        new["pwf"].update(potencia_min_w=pmin, potencia_max_w=pmax)
        new["hdf"].update(dT_max_k=dt, rot_max_rpm=rot)
        new["osf"].update(osf)
        new["twf"].update(ativa=twf, desgaste_min=tw)
        errs = validate_profile(new)
        if errs:
            for e in errs:
                st.error(e)
        else:
            ss.profiles[ss.profile_name] = new
            state.reset_models()
            st.success("Perfil atualizado. Os modelos treinados foram invalidados (o nível 1 do híbrido depende das regras).")
            st.rerun()
    if twf:
        st.warning("TWF ativa: só afeta o método «Regras críticas + TWF» (as regras críticas do híbrido são sempre HDF, PWF e OSF).")
    c = st.columns(4)
    new_name = c[0].text_input("Nome para duplicar", value=f"{ss.profile_name} (cópia)")
    if c[1].button("Duplicar perfil"):
        try:
            ss.profiles[new_name.strip()] = duplicate_profile(ss.profiles, ss.profile_name, new_name)
            ss.profile_name = new_name.strip()
            state.reset_models()
            st.rerun()
        except ProfileError as e:
            st.error(str(e))
    if c[2].button("Restaurar default deste perfil"):
        if ss.profile_name in defaults:
            ss.profiles[ss.profile_name] = defaults[ss.profile_name]
            state.reset_models()
            st.rerun()
        else:
            st.warning("Este perfil foi criado por si; não tem default.")
    c[3].download_button("Exportar perfil (JSON)", profile_to_json(p, ss.profile_name), f"perfil_{ss.profile_name}.json", "application/json")
    up = st.file_uploader("Importar perfil (JSON)", type=["json"])
    if up is not None and st.button("Importar perfil"):
        try:
            name, prof = profile_from_json(up.getvalue())
            ss.profiles[name] = prof
            ss.profile_name = name
            state.reset_models()
            st.rerun()
        except ProfileError as e:
            st.error(str(e))
    base = ss.profile_name if ss.profile_name in defaults else "Trabalho"
    st.subheader(f"Comparação: original ({base}) vs atual")
    st.dataframe(compare_profiles(defaults[base], p), width="stretch", hide_index=True)
