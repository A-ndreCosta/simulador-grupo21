"""Métricas de classificação para dados desequilibrados (a classe positiva é a falha)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


def _binary(y) -> np.ndarray:
    arr = np.asarray(y).astype(float)
    if arr.size == 0 or not np.isin(arr, [0.0, 1.0]).all():
        raise ValueError("As etiquetas e as previsões têm de ser 0/1 e não podem estar vazias.")
    return arr.astype(int)


def confusion_counts(y_true, y_pred) -> dict:
    y, p = _binary(y_true), _binary(y_pred)
    if y.shape != p.shape:
        raise ValueError("Etiquetas e previsões com tamanhos diferentes.")
    return {"TP": int(((y == 1) & (p == 1)).sum()), "FP": int(((y == 0) & (p == 1)).sum()),
            "FN": int(((y == 1) & (p == 0)).sum()), "TN": int(((y == 0) & (p == 0)).sum())}


def classification_metrics(y_true, y_pred, score=None) -> dict:
    """Precisão, recall, F1, accuracy (complementar) e contagens. AP e ROC-AUC só se houver ``score`` e as duas classes."""
    c = confusion_counts(y_true, y_pred)
    tp, fp, fn, tn = c["TP"], c["FP"], c["FN"], c["TN"]
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    out = {"precisão": prec, "recall": rec, "F1": f1, "accuracy": (tp + tn) / (tp + fp + fn + tn), **c,
           "leituras": tp + fp + fn + tn, "falhas reais": tp + fn, "detetadas": tp + fp,
           "PR-AUC": np.nan, "ROC-AUC": np.nan}
    if score is not None:
        y = _binary(y_true)
        if len(set(y.tolist())) == 2:
            s = np.asarray(score, dtype=float)
            out["PR-AUC"] = float(average_precision_score(y, s))
            out["ROC-AUC"] = float(roc_auc_score(y, s))
    return out


def confusion_matrix_df(y_true, y_pred) -> pd.DataFrame:
    c = confusion_counts(y_true, y_pred)
    return pd.DataFrame([[c["TN"], c["FP"]], [c["FN"], c["TP"]]],
                        index=["Real: normal", "Real: falha"], columns=["Previsto: normal", "Previsto: alerta/falha"])


def comparison_table(results: dict) -> pd.DataFrame:
    """``results``: nome → dict de ``classification_metrics``. Colunas numéricas com tipos corretos (contagens inteiras)."""
    cols = ["precisão", "recall", "F1", "accuracy", "PR-AUC", "ROC-AUC", "TP", "FP", "FN", "TN"]
    df = pd.DataFrame({k: {c: v[c] for c in cols} for k, v in results.items()}).T
    for c in ("TP", "FP", "FN", "TN"):
        df[c] = df[c].astype(int)
    for c in cols[:6]:
        df[c] = df[c].astype(float)
    return df


def recall_vs_false_alarms(y_true, flag, score, taus) -> pd.DataFrame:
    """Para cada τ: decisão = regras ∨ (score ≥ τ). Devolve recall e nº de falsos alarmes."""
    y = _binary(y_true)
    flag = np.asarray(flag, dtype=bool)
    score = np.asarray(score, dtype=float)
    rows = []
    for t in taus:
        m = classification_metrics(y, (flag | (score >= t)).astype(int))
        rows.append({"tau": float(t), "recall": m["recall"], "FP": m["FP"], "precisão": m["precisão"], "F1": m["F1"]})
    return pd.DataFrame(rows)
