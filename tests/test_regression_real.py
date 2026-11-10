"""Regressão contra os valores publicados no notebook/relatório (AI4I 2020, perfil «Trabalho», semente 42)."""
import pytest

from src.hybrid import HybridSystem
from src.metrics import classification_metrics
from src.ml_model import train_models
from src.rules import critical_mask


def test_dataset_shape(real_df):
    assert len(real_df) == 10000 and int(real_df.Falha.sum()) == 339


def test_critical_rules_full_dataset(real_df, profiles):
    m = classification_metrics(real_df.Falha, critical_mask(real_df, profiles["Trabalho"]).astype(int))
    assert (m["TP"], m["FP"], m["FN"]) == (287, 0, 52)


@pytest.fixture(scope="module")
def trained(real_df):
    return train_models(real_df, names=["RF — sensores", "RF — sensores + derivadas"])


def test_rf_metrics_match_notebook(trained):
    a, b = trained.test_metrics["RF — sensores"], trained.test_metrics["RF — sensores + derivadas"]
    assert (round(a["precisão"], 3), round(a["recall"], 3), round(a["F1"], 3)) == (0.714, 0.662, 0.687)
    assert round(b["F1"], 3) == 0.882


def test_hybrid_R_matches_notebook(trained, profiles):
    h = HybridSystem(profiles["Trabalho"], "R").fit(trained.train)
    y = trained.test.Falha.astype(int)
    m = classification_metrics(y, h.predict(trained.test, h.tau_r90))
    assert (round(m["recall"], 3), round(m["precisão"], 3)) == (0.882, 0.556)
    m0 = classification_metrics(y, h.predict(trained.test, 1.01))          # só regras
    assert m0["FP"] == 0
