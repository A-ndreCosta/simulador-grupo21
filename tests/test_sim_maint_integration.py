import compileall
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.detection import ALL_METHODS, compare_on_test, run_method
from src.hybrid import HybridSystem
from src.maintenance import add_log_entry, branches_for, empty_log, flow_dot
from src.ml_model import train_models
from src.rules import analyze_one
from src.simulation import EVOLUTIONS, SCENARIOS, generate_series, validate_scenarios

ROOT = Path(__file__).resolve().parent.parent


# ---------- simulação ----------
def test_scenarios_valid_and_trigger_intended_rule(profiles):
    assert validate_scenarios() == []
    expected = {"Normal": [], "Risco térmico": ["HDF"], "Potência anormal": ["PWF"], "Sobrecarga": ["OSF"]}
    for pname, P in profiles.items():
        for name, vals in SCENARIOS.items():
            assert analyze_one(**vals, profile=P)["regras_acionadas"] == expected[name], (pname, name)


def test_series_reproducible_and_bounded():
    a = generate_series(100, SCENARIOS["Normal"], seed=5, drift=EVOLUTIONS["Sobrecarga progressiva"])
    b = generate_series(100, SCENARIOS["Normal"], seed=5, drift=EVOLUTIONS["Sobrecarga progressiva"])
    c = generate_series(100, SCENARIOS["Normal"], seed=6, drift=EVOLUTIONS["Sobrecarga progressiva"])
    pd.testing.assert_frame_equal(a, b)
    assert not a["Torque"].equals(c["Torque"])
    assert a["simulado"].all() and a["Desgaste"].is_monotonic_increasing and a["Torque"].between(0, 500).all()


def test_series_progressive_change_triggers_rule(profiles):
    s = generate_series(200, SCENARIOS["Normal"], noise={"Torque": 0, "Rot": 0, "T_ar": 0, "T_proc": 0}, drift=EVOLUTIONS["Sobrecarga progressiva"], seed=1)
    st = run_method("Regras críticas (HDF+PWF+OSF)", s, profiles["Trabalho"])["estado"]
    assert st.iloc[0] == "NORMAL" and (st != "NORMAL").any() and st[st != "NORMAL"].index[0] > 0


def test_series_invalid_args():
    with pytest.raises(ValueError):
        generate_series(1, SCENARIOS["Normal"])
    with pytest.raises(ValueError):
        generate_series(10, {**SCENARIOS["Normal"], "Torque": -4})
    with pytest.raises(ValueError):
        generate_series(10, SCENARIOS["Normal"], noise={"Torque": -1})
    with pytest.raises(ValueError):
        generate_series(10, SCENARIOS["Normal"], drift={"X": 1})


# ---------- manutenção ----------
def test_flow_branches_and_dot():
    assert branches_for(["HDF", "PWF", "OSF", "TWF"]) == ["Quente", "Potência", "Sobrecarga", "Desgaste"]
    dot = flow_dot(["Quente"])
    for w in ("Máquina", "Funciona?", "Porque motivo?", "Quente", "Desgaste", "Potência", "Random", "Sobrecarga", "Peça", "Responsável", "Arranja", "Confirmação"):
        assert w in dot


def test_intervention_log_only_user_data():
    log = empty_log()
    assert len(log) == 0
    log = add_log_entry(log, "Quente", "Ventoinha", "Ana", "Pendente", "Por confirmar")
    assert log.iloc[0]["peça"] == "Ventoinha" and len(log) == 1
    with pytest.raises(ValueError):
        add_log_entry(log, "Quente", "", "Ana", "Pendente", "Por confirmar")
    with pytest.raises(ValueError):
        add_log_entry(log, "Inexistente", "x", "y", "Pendente", "Por confirmar")
    with pytest.raises(ValueError):
        add_log_entry(log, "Quente", "x" * 500, "y", "Pendente", "Por confirmar")


# ---------- integração ----------
@pytest.fixture(scope="module")
def pipeline(small_df, profiles):
    tm = train_models(small_df)
    hy = {v: HybridSystem(profiles["Trabalho"], v).fit(tm.train) for v in "RG"}
    return tm, hy


def test_main_flow_all_methods(pipeline, small_df, profiles):
    tm, hy = pipeline
    for m in ALL_METHODS:
        out = run_method(m, small_df.head(50), profiles["Trabalho"], tm, hy, 0.5)
        assert len(out) == 50 and {"estado", "regras_acionadas", "p_ML", "explicacao"} <= set(out.columns)


def test_unavailable_method_raises(small_df, profiles):
    with pytest.raises(RuntimeError):
        run_method("Híbrido R", small_df.head(5), profiles["Trabalho"])
    with pytest.raises(ValueError):
        run_method("inexistente", small_df.head(5), profiles["Trabalho"])


def test_compare_same_test_set_and_names(pipeline, profiles):
    tm, hy = pipeline
    res = compare_on_test(tm, hy, profiles["Trabalho"])
    assert {"Regras críticas (HDF+PWF+OSF)", "Regras críticas + TWF", "RF — sensores", "SVM — sensores", "RF — sensores + derivadas",
            "SVM — sensores + derivadas"} <= set(res)
    assert any("τ_F1" in k for k in res) and any("τ_R90" in k for k in res)
    assert {r["leituras"] for r in res.values()} == {len(tm.test)}
    assert {r["falhas reais"] for r in res.values()} == {int(tm.test.Falha.sum())}


def test_unlabelled_data_rules_only(small_df, profiles):
    out = run_method("Regras críticas (HDF+PWF+OSF)", small_df.drop(columns="Falha").head(20), profiles["Trabalho"])
    assert "Falha" not in out.columns


def test_modules_compile():
    for sub in ("src", "ui"):
        assert compileall.compile_dir(str(ROOT / sub), quiet=1, force=True)
    assert compileall.compile_file(str(ROOT / "app.py"), quiet=1, force=True)


def test_no_dangerous_calls():
    import re
    pat = re.compile(r"\beval\(|\bexec\(|pickle\.load|os\.system|subprocess|except\s*:|except Exception")
    for f in list((ROOT / "src").glob("*.py")) + list((ROOT / "ui").glob("*.py")) + [ROOT / "app.py"]:
        assert not pat.search(f.read_text(encoding="utf-8")), f
        assert not re.search(r"[\"'](/Users/|/home/|C:\\\\)", f.read_text(encoding="utf-8")), f


def test_data_files_identified():
    assert "SINTETICO" in (ROOT / "data" / "LEIA-ME.txt").read_text(encoding="utf-8")
    demo = pd.read_csv(ROOT / "data" / "demo_sintetico.csv")
    assert "Machine failure" not in demo.columns and demo["Product ID"].str.startswith("DEMO").all()


# ---------- interface (AppTest) ----------
def _app():
    from streamlit.testing.v1 import AppTest
    return AppTest.from_file(str(ROOT / "app.py"), default_timeout=180).run()


PAGES = ["Painel geral", "Simulador manual", "Importar dados", "Modelos & Avaliação", "Simulação de séries", "Perfis de máquina",
         "Fluxo de manutenção", "Dicionário de variáveis", "Sobre o projeto"]


def test_every_page_renders_without_exception():
    at = _app()
    for p in PAGES:
        at.sidebar.radio[0].set_value(p).run()
        assert not at.exception, (p, at.exception)


def test_ui_simulator_scenario_buttons_change_result():
    at = _app()
    at.sidebar.radio[0].set_value("Simulador manual").run()
    next(b for b in at.button if b.label == "Risco térmico").click().run()
    assert any("FALHA" in m.value for m in at.markdown if "statebox" in m.value)
    assert any("HDF" in m.value for m in at.markdown)
    next(b for b in at.button if b.label == "Normal").click().run()
    assert any("NORMAL" in m.value for m in at.markdown if "statebox" in m.value)


def test_ui_full_flow_with_demo_unlabelled():
    at = _app()
    at.sidebar.radio[0].set_value("Importar dados").run()
    next(b for b in at.button if "demo sintético" in b.label).click().run()
    next(b for b in at.button if b.label == "Usar estes dados").click().run()
    at.sidebar.radio[0].set_value("Painel geral").run()
    assert not at.exception and any("sintético" in m.value.lower() for m in at.markdown)
    at.sidebar.radio[0].set_value("Modelos & Avaliação").run()
    assert any("etiqueta" in w.value for w in at.warning)


def test_ui_profile_change_invalidates_state():
    at = _app()
    at.sidebar.radio[0].set_value("Perfis de máquina").run()
    assert not at.exception
    assert at.session_state["profile_name"] == "Trabalho"
