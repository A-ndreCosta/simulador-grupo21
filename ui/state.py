"""Estado de sessão (st.session_state) e funções partilhadas entre páginas."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

from src.config import load_default_profiles
from src.detection import ALL_METHODS, method_available, run_method
from src.maintenance import empty_log

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"


def init():
    defaults = {
        "profiles": load_default_profiles(), "profile_name": "Trabalho", "dataset": None, "dataset_meta": None,
        "trained": None, "hybrids": {}, "train_version": 0, "method": ALL_METHODS[0], "tau": 0.5,
        "history": [], "series": None, "log": empty_log(), "last_run": None, "analysis_cache": None,
        "data_version": 0, "train_error": None, "compare": [],
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def profile() -> dict:
    return st.session_state.profiles[st.session_state.profile_name]


def set_dataset(df: pd.DataFrame, meta: dict):
    st.session_state.dataset, st.session_state.dataset_meta = df, meta
    st.session_state.data_version += 1
    reset_models()


def reset_models():
    """Invalida modelos treinados (dados ou perfil mudaram)."""
    st.session_state.trained, st.session_state.hybrids = None, {}
    st.session_state.train_version += 1
    st.session_state.analysis_cache = None


def model_status() -> str:
    if st.session_state.trained is not None:
        return "Treinado"
    if st.session_state.train_error:
        return "Indisponível"
    return "Não treinado"


def get_analysis():
    """Analisa o conjunto de dados atual com o método e o perfil escolhidos (memoizado na sessão)."""
    ss = st.session_state
    if ss.dataset is None:
        return None
    method = ss.method
    if not method_available(method, ss.trained, ss.hybrids):
        method = ALL_METHODS[0]
    key = (ss.data_version, json.dumps(profile(), sort_keys=True), method, round(float(ss.tau), 4), ss.train_version, ss.profile_name)
    cache = ss.analysis_cache
    if cache and cache["key"] == key:
        return cache
    res = run_method(method, ss.dataset, profile(), ss.trained, ss.hybrids, ss.tau)
    full = pd.concat([ss.dataset.reset_index(drop=True), res.reset_index(drop=True)], axis=1)
    ss.analysis_cache = {"key": key, "df": full, "method": method, "when": datetime.now().strftime("%d/%m/%Y %H:%M:%S")}
    ss.last_run = ss.analysis_cache["when"]
    return ss.analysis_cache
