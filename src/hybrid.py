"""Sistema híbrido de dois níveis: regras críticas (nível 1) + ML sobre o resto (nível 2).

Variante R (residual): ML treinado só onde as regras críticas NÃO disparam.
Variante G (global): ML treinado em todas as leituras; a decisão continua a ser regras ∨ ML.
τ_F1 e τ_R90 escolhem-se SÓ no treino, com probabilidades fora da amostra (CV estratificada); o teste nunca interfere.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline

from .features import add_derived_features
from .metrics import classification_metrics
from .ml_model import NUM_B, SEED, make_pipeline, prepare_xy
from .rules import CRITICAL_RULES, STATE_FAIL, STATE_ML, STATE_NORMAL, analyze_dataframe, critical_mask

TAUS = np.append(np.round(np.arange(0.05, 1.0, 0.01), 2), 1.01)   # 1.01 = «o ML nunca dispara»
RF_PARAMS = dict(n_estimators=200, min_samples_leaf=3, class_weight="balanced_subsample")
MIN_POS_CV = 5          # mínimo de falhas na amostra do ML para fazer CV de 5 partes
VARIANT_NAMES = {"R": "Híbrido R (ML residual)", "G": "Híbrido G (ML global ∪ regras)"}


def _ml_pipeline(seed: int) -> Pipeline:
    p = make_pipeline("rf", NUM_B, dict(n_estimators=200, min_samples_leaf=3), seed)
    p.set_params(m__class_weight="balanced_subsample")
    return p


def _usable(y: pd.Series, n_folds: int = 5) -> bool:
    return y.nunique() == 2 and int(y.sum()) >= MIN_POS_CV and int((1 - y).sum()) >= MIN_POS_CV


def choose_tau(flag, p, y):
    """(τ_F1, τ_R90, curva) — avaliado só nos dados de treino fornecidos (probabilidades fora da amostra)."""
    flag, p, y = np.asarray(flag, bool), np.asarray(p, float), np.asarray(y, int)
    cur = pd.DataFrame([{"tau": t, **classification_metrics(y, (flag | (p >= t)).astype(int))} for t in TAUS])
    t_f1 = cur.sort_values(["F1", "tau"], ascending=[False, False]).iloc[0]["tau"]
    ok = cur[cur["recall"] >= 0.90]
    t_r90 = ok["tau"].max() if len(ok) else cur.sort_values(["recall", "tau"], ascending=[False, False]).iloc[0]["tau"]
    return float(t_f1), float(t_r90), cur


@dataclass
class HybridSystem:
    profile: dict
    requested: str = "R"
    seed: int = SEED
    tau: float = 0.5
    effective: str = ""            # variante realmente usada
    fallback: str = ""             # motivo, se houve fallback
    ml: Pipeline | None = None
    tau_f1: float = 1.01
    tau_r90: float = 1.01
    curve: pd.DataFrame = field(default_factory=pd.DataFrame)
    n_train_ml: int = 0

    @property
    def ml_available(self) -> bool:
        return self.ml is not None

    @property
    def label(self) -> str:
        if not self.fallback:
            return VARIANT_NAMES[self.effective]
        if not self.ml_available:
            return f"Só regras críticas (a variante {self.requested} não pôde ser treinada)"
        return f"Híbrido G (fallback — pedida a variante {self.requested})"

    def fit(self, train: pd.DataFrame) -> "HybridSystem":
        if self.requested not in ("R", "G"):
            raise ValueError("A variante tem de ser 'R' ou 'G'.")
        if "Falha" not in train.columns:
            raise ValueError("O treino do híbrido precisa da coluna 'Falha'.")
        train = add_derived_features(train)
        y = train["Falha"].astype(int)
        flag = critical_mask(train, self.profile).to_numpy()
        mode = self.requested
        sel = ~flag if mode == "R" else np.ones(len(train), bool)
        if mode == "R" and not _usable(y[sel]):
            self.fallback = ("Nas leituras onde as regras críticas não disparam não há falhas/normais suficientes "
                             f"(mín. {MIN_POS_CV} de cada) para treinar o ML residual; usada a variante G.")
            mode = "G"
            sel = np.ones(len(train), bool)
        if mode == "G" and not _usable(y[sel]):
            self.fallback = self.fallback or "Dados insuficientes para treinar o ML."
            self.fallback += " O ML global também não pôde ser treinado: só as regras críticas decidem."
            self.effective = mode
            self.ml = None
            self.tau_f1 = self.tau_r90 = 1.01
            self.tau = 1.01
            self.curve = choose_tau(flag, np.zeros(len(train)), y.to_numpy())[2]
            return self
        self.effective = mode
        X = prepare_xy(train, NUM_B)
        idx = np.flatnonzero(sel)
        cv = StratifiedKFold(5, shuffle=True, random_state=self.seed)
        p = np.zeros(len(train))
        p[idx] = cross_val_predict(_ml_pipeline(self.seed), X.iloc[idx], y.iloc[idx], cv=cv, method="predict_proba")[:, 1]
        self.tau_f1, self.tau_r90, self.curve = choose_tau(flag, p, y.to_numpy())
        self.ml = _ml_pipeline(self.seed).fit(X.iloc[idx], y.iloc[idx])
        self.n_train_ml = len(idx)
        return self

    def proba_ml(self, df: pd.DataFrame) -> np.ndarray:
        if self.ml is None:
            return np.zeros(len(df))
        return self.ml.predict_proba(prepare_xy(df, NUM_B))[:, 1]

    def decide(self, df: pd.DataFrame, tau: float | None = None) -> pd.DataFrame:
        """Estado por leitura. Prioridade: regra crítica > ALERTA (ML) > NORMAL. Nunca usa a etiqueta."""
        tau = self.tau if tau is None else tau
        if not 0.0 <= tau <= 1.01:
            raise ValueError("τ tem de estar entre 0 e 1.")
        res = analyze_dataframe(df, self.profile, usar_twf=False)
        p = self.proba_ml(df)
        crit = res["estado"].eq(STATE_FAIL).to_numpy()
        ml_alert = (~crit) & (p >= tau)
        estado = np.where(crit, STATE_FAIL, np.where(ml_alert, STATE_ML, STATE_NORMAL))
        expl = np.where(crit, res["explicacao"], np.where(ml_alert, [f"Probabilidade do ML = {v:.2f} ≥ τ = {tau:.2f}" for v in p], "Nenhuma regra ativada; ML abaixo de τ."))
        return pd.DataFrame({"estado": estado, "regras_acionadas": res["regras_acionadas"].to_numpy(), "p_ML": p,
                             "explicacao": expl}, index=df.index)

    def predict(self, df: pd.DataFrame, tau: float | None = None) -> np.ndarray:
        return (self.decide(df, tau)["estado"] != STATE_NORMAL).astype(int).to_numpy()
