"""Despacho único dos métodos de deteção e comparação no MESMO conjunto de teste."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .hybrid import HybridSystem
from .metrics import classification_metrics
from .ml_model import MODEL_SPECS, TrainedModels
from .rules import STATE_FAIL, STATE_ML, STATE_NORMAL, analyze_dataframe

RULE_METHODS = ["Regras críticas (HDF+PWF+OSF)", "Regras críticas + TWF"]
ML_METHODS = list(MODEL_SPECS)
HYBRID_METHODS = ["Híbrido R", "Híbrido G"]
ALL_METHODS = RULE_METHODS + ML_METHODS + HYBRID_METHODS


def method_available(method: str, trained: TrainedModels | None, hybrids: dict | None) -> bool:
    if method in RULE_METHODS:
        return True
    if method in ML_METHODS:
        return trained is not None and method in trained.models
    if method in HYBRID_METHODS:
        return bool(hybrids) and method[-1] in hybrids
    return False


def run_method(method: str, df: pd.DataFrame, profile: dict, trained: TrainedModels | None = None,
               hybrids: dict | None = None, tau: float = 0.5) -> pd.DataFrame:
    """Devolve DataFrame com ``estado``, ``regras_acionadas``, ``p_ML`` (NaN se não aplicável) e ``explicacao``."""
    if method not in ALL_METHODS:
        raise ValueError(f"Método desconhecido: {method}")
    if not method_available(method, trained, hybrids):
        raise RuntimeError(f"O método '{method}' ainda não está disponível: treine os modelos primeiro.")
    if method in RULE_METHODS:
        res = analyze_dataframe(df, profile, usar_twf=(method == RULE_METHODS[1]))
        out = res[["estado", "regras_acionadas", "explicacao"]].copy()
        out["p_ML"] = np.nan
        return out[["estado", "regras_acionadas", "p_ML", "explicacao"]]
    if method in ML_METHODS:
        pr = trained.predict(method, df)
        alert = pr["previsao"].to_numpy() == 1
        return pd.DataFrame({"estado": np.where(alert, STATE_ML, STATE_NORMAL), "regras_acionadas": "—",
                             "p_ML": pr["score_ML"], "explicacao": np.where(alert, "Modelo de ML prevê falha.", "Modelo de ML prevê normal.")},
                            index=df.index)
    return hybrids[method[-1]].decide(df, tau)


def predicted_failure(states: pd.Series) -> np.ndarray:
    return (states != STATE_NORMAL).astype(int).to_numpy()


def compare_on_test(trained: TrainedModels, hybrids: dict, profile: dict) -> dict:
    """Todas as abordagens sobre o mesmo teste. Os τ do híbrido vêm do treino (nunca do teste)."""
    test, y = trained.test, trained.test["Falha"].astype(int)
    results = {}
    for m in RULE_METHODS:
        st = run_method(m, test, profile)["estado"]
        results[m] = classification_metrics(y, predicted_failure(st))
    for m in ML_METHODS:
        results[m] = trained.test_metrics[m]
    for var, h in hybrids.items():
        p = h.proba_ml(test)
        crit = run_method("Regras críticas (HDF+PWF+OSF)", test, profile)["estado"].eq(STATE_FAIL).to_numpy()
        for name, tau in ((f"Híbrido {var} (τ=0,5)", 0.5), (f"Híbrido {var} (τ_F1={h.tau_f1:.2f})", h.tau_f1),
                          (f"Híbrido {var} (τ_R90={h.tau_r90:.2f})", h.tau_r90)):
            pred = (crit | (p >= tau)).astype(int)
            score = np.where(crit, 1.0, p)
            results[name] = classification_metrics(y, pred, score)
    return results
