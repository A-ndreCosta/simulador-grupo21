"""Cenários validados e séries simuladas (SEMPRE identificadas como simuladas).

AI4I 2020 não é uma série temporal: cada linha é independente. As séries aqui geradas servem para
demonstrar o comportamento das regras/ML ao longo de uma evolução hipotética; NÃO validam antecipação temporal.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .validation import INPUT_LIMITS, validate_inputs

SCENARIOS = {
    "Normal": dict(Type="L", T_ar=300.0, T_proc=310.0, Rot=1500.0, Torque=40.0, Desgaste=100.0),
    "Risco térmico": dict(Type="M", T_ar=302.0, T_proc=309.0, Rot=1300.0, Torque=40.0, Desgaste=50.0),
    "Potência anormal": dict(Type="L", T_ar=300.0, T_proc=310.0, Rot=2500.0, Torque=70.0, Desgaste=50.0),
    "Sobrecarga": dict(Type="L", T_ar=300.0, T_proc=310.0, Rot=1400.0, Torque=55.0, Desgaste=220.0),
}
SENSORS = ["T_ar", "T_proc", "Rot", "Torque", "Desgaste"]

# deriva por passo de cada cenário de evolução (unidade/passo)
EVOLUTIONS = {
    "Estável (só desgaste normal)": {},
    "Degradação térmica": {"T_proc": -0.05, "Rot": -6.0},
    "Sobrecarga progressiva": {"Torque": 0.25, "Desgaste": 1.5},
    "Potência a subir": {"Torque": 0.3, "Rot": 12.0},
    "Personalizado": {},
}
DEFAULT_NOISE = {"T_ar": 0.3, "T_proc": 0.3, "Rot": 20.0, "Torque": 1.5, "Desgaste": 0.0}
MAX_STEPS = 5000


def validate_scenarios() -> list[str]:
    """Devolve erros de validação dos cenários (lista vazia = todos válidos)."""
    errs = []
    for name, v in SCENARIOS.items():
        e, _ = validate_inputs(**v)
        errs += [f"{name}: {x}" for x in e]
    return errs


def generate_series(n: int, start: dict, noise: dict | None = None, drift: dict | None = None, seed: int = 42,
                    wear_per_step: float = 1.0) -> pd.DataFrame:
    """Gera ``n`` leituras: valor inicial + deriva·passo + ruído gaussiano (semente fixa → reprodutível).

    Valores são limitados aos intervalos físicos permitidos (clip); o desgaste só cresce.
    """
    if not isinstance(n, (int, np.integer)) or not 2 <= n <= MAX_STEPS:
        raise ValueError(f"O número de leituras tem de estar entre 2 e {MAX_STEPS}.")
    errs, _ = validate_inputs(**start)
    if errs:
        raise ValueError("Valores iniciais inválidos: " + "; ".join(errs))
    noise = {**DEFAULT_NOISE, **(noise or {})}
    drift = dict(drift or {})
    for d in (noise, drift):
        for k, v in d.items():
            if k not in SENSORS:
                raise ValueError(f"Variável desconhecida: {k}")
            if not np.isfinite(v):
                raise ValueError(f"Valor não finito em {k}.")
    if any(v < 0 for v in noise.values()):
        raise ValueError("O ruído (desvio-padrão) não pode ser negativo.")
    rng = np.random.default_rng(seed)
    steps = np.arange(n, dtype=float)
    out = {"passo": np.arange(n), "Type": [start["Type"]] * n}
    for col in SENSORS:
        base = start[col] + drift.get(col, 0.0) * steps
        if col == "Desgaste":
            base = base + wear_per_step * steps
        vals = base + rng.normal(0.0, noise[col], n)
        lo, hi = INPUT_LIMITS[col][0], INPUT_LIMITS[col][1]
        out[col] = np.clip(vals, lo, hi)
    df = pd.DataFrame(out)
    df["Desgaste"] = np.maximum.accumulate(df["Desgaste"].to_numpy())
    df["simulado"] = True
    return df
