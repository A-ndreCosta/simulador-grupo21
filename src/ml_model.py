"""Treino, previsão e avaliação dos 4 modelos de ML (RF/SVM × sensores/sensores+derivadas).

Sem fugas de informação: só entram Type, sensores (e derivadas); UDI, Product ID e etiquetas de modo de falha
nunca chegam ao modelo. Pré-processamento aprendido apenas no treino (Pipeline).
Hiperparâmetros fixos, os escolhidos por validação cruzada (F1) no notebook original — a app não repete a grelha.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC

from .features import CATEGORY_COLUMN, DERIVED_COLUMNS, SENSOR_COLUMNS, VALID_TYPES, add_derived_features
from .metrics import classification_metrics

SEED = 42
MIN_ROWS = 100
MIN_POSITIVES = 10
FORBIDDEN_FEATURES = {"UDI", "Product ID", "Falha", "TWF", "HDF", "PWF", "OSF", "RNF", "Machine failure"}

NUM_A = list(SENSOR_COLUMNS)
NUM_B = list(SENSOR_COLUMNS) + list(DERIVED_COLUMNS)

# nome → (tipo, variáveis numéricas, hiperparâmetros)
MODEL_SPECS = {
    "RF — sensores": ("rf", NUM_A, dict(n_estimators=200, max_depth=None, min_samples_leaf=1)),
    "SVM — sensores": ("svm", NUM_A, dict(C=1000, gamma="scale")),
    "RF — sensores + derivadas": ("rf", NUM_B, dict(n_estimators=400, max_depth=None, min_samples_leaf=1)),
    "SVM — sensores + derivadas": ("svm", NUM_B, dict(C=1000, gamma="scale")),
}


class TrainingError(ValueError):
    """Não há condições para treinar (sem etiquetas, uma só classe, poucos dados…)."""


def features_for(num_cols: list[str]) -> list[str]:
    return [CATEGORY_COLUMN] + list(num_cols)


def assert_no_leakage(cols) -> None:
    bad = FORBIDDEN_FEATURES.intersection(cols)
    if bad:
        raise TrainingError(f"Variáveis proibidas (fuga de informação): {sorted(bad)}")


def make_pipeline(kind: str, num_cols: list[str], params: dict, seed: int = SEED) -> Pipeline:
    assert_no_leakage(num_cols)
    scale = StandardScaler() if kind == "svm" else "passthrough"
    pre = ColumnTransformer([("cat", OneHotEncoder(categories=[sorted(VALID_TYPES)]), [CATEGORY_COLUMN]),
                             ("num", scale, list(num_cols))])
    if kind == "rf":
        est = RandomForestClassifier(class_weight="balanced", random_state=seed, n_jobs=-1, **params)
    elif kind == "svm":
        est = SVC(kernel="rbf", class_weight="balanced", random_state=seed, **params)
    else:
        raise ValueError(f"Tipo de modelo desconhecido: {kind}")
    return Pipeline([("pre", pre), ("m", est)])


def check_trainable(df: pd.DataFrame, y_col: str = "Falha") -> None:
    if y_col not in df.columns:
        raise TrainingError("Os dados não têm etiqueta de falha ('Falha'): não é possível treinar nem avaliar o ML.")
    y = pd.to_numeric(df[y_col], errors="coerce")
    if y.isna().any() or not y.isin([0, 1]).all():
        raise TrainingError("A etiqueta 'Falha' tem de ser 0 ou 1, sem valores em falta.")
    if len(df) < MIN_ROWS:
        raise TrainingError(f"Poucas leituras para treinar ({len(df)}; mínimo {MIN_ROWS}).")
    counts = y.astype(int).value_counts()
    if len(counts) < 2:
        raise TrainingError("Os dados têm uma só classe (só normais ou só falhas): não é possível treinar.")
    if counts.min() < MIN_POSITIVES:
        raise TrainingError(f"A classe menos frequente tem só {int(counts.min())} exemplos (mínimo {MIN_POSITIVES}).")


def prepare_xy(df: pd.DataFrame, num_cols: list[str]) -> pd.DataFrame:
    data = add_derived_features(df)
    cols = features_for(num_cols)
    assert_no_leakage(cols)
    return data[cols]


def split_train_test(df: pd.DataFrame, seed: int = SEED, test_size: float = 0.2):
    """Divisão estratificada 80/20. Devolve (treino, teste) com o índice original."""
    check_trainable(df)
    tr, te = train_test_split(df.index, test_size=test_size, stratify=df["Falha"].astype(int), random_state=seed)
    return df.loc[tr].copy(), df.loc[te].copy()


def score_of(model: Pipeline, X: pd.DataFrame) -> np.ndarray:
    """Pontuação em [0,1]. RF: probabilidade. SVM: sigmóide da função de decisão (NÃO calibrada)."""
    if hasattr(model.named_steps["m"], "predict_proba"):
        return model.predict_proba(X)[:, 1]
    d = model.decision_function(X)
    return 1.0 / (1.0 + np.exp(-d))


@dataclass
class TrainedModels:
    models: dict
    train: pd.DataFrame
    test: pd.DataFrame
    seed: int
    test_metrics: dict = field(default_factory=dict)
    test_scores: dict = field(default_factory=dict)
    test_pred: dict = field(default_factory=dict)
    cv_f1_train: dict = field(default_factory=dict)

    def predict(self, name: str, df: pd.DataFrame) -> pd.DataFrame:
        """Previsões para ``df`` (esquema: score_ML, previsao 0/1). Nunca usa etiquetas."""
        if name not in self.models:
            raise KeyError(f"Modelo não treinado: {name}")
        _, num, _ = MODEL_SPECS[name]
        X = prepare_xy(df, num)
        if X.isna().any().any():
            raise ValueError("Os dados têm valores em falta.")
        s = score_of(self.models[name], X)
        return pd.DataFrame({"score_ML": s, "previsao": (self.models[name].predict(X)).astype(int)}, index=df.index)


def train_models(df: pd.DataFrame, seed: int = SEED, names: list[str] | None = None, progress=None,
                 cv_folds: int = 0) -> TrainedModels:
    """Treina os modelos pedidos no conjunto de treino (80 %) e avalia UMA vez no teste (20 %)."""
    train, test = split_train_test(df, seed)
    names = names or list(MODEL_SPECS)
    out = TrainedModels(models={}, train=train, test=test, seed=seed)
    y_tr, y_te = train["Falha"].astype(int), test["Falha"].astype(int)
    for i, name in enumerate(names):
        if progress:
            progress(i / len(names), f"A treinar: {name}")
        kind, num, params = MODEL_SPECS[name]
        model = make_pipeline(kind, num, params, seed).fit(prepare_xy(train, num), y_tr)
        Xte = prepare_xy(test, num)
        s = score_of(model, Xte)
        pred = model.predict(Xte).astype(int)
        out.models[name] = model
        out.test_scores[name] = s
        out.test_pred[name] = pred
        out.test_metrics[name] = classification_metrics(y_te, pred, s)
        if cv_folds:
            from sklearn.model_selection import cross_val_score
            cv = StratifiedKFold(cv_folds, shuffle=True, random_state=seed)
            out.cv_f1_train[name] = float(cross_val_score(make_pipeline(kind, num, params, seed), prepare_xy(train, num), y_tr, cv=cv, scoring="f1").mean())
    if progress:
        progress(1.0, "Treino concluído")
    return out
