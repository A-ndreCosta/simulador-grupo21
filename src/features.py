"""Variáveis derivadas do Tema 7 — implementação central e única das fórmulas.

dT         = T_proc − T_ar                    [K]
Potencia   = Torque · Rot · 2π / 60           [W]
Sobrecarga = Desgaste · Torque                [min·Nm]
"""
from __future__ import annotations

import numpy as np
import pandas as pd

SENSOR_COLUMNS = ["T_ar", "T_proc", "Rot", "Torque", "Desgaste"]
CATEGORY_COLUMN = "Type"
REQUIRED_COLUMNS = [CATEGORY_COLUMN] + SENSOR_COLUMNS
DERIVED_COLUMNS = ["dT", "Potencia", "Sobrecarga"]
VALID_TYPES = ("L", "M", "H")


def calc_dT(T_proc, T_ar):
    """Diferença de temperatura (K). Aceita escalares, arrays ou Series."""
    return T_proc - T_ar


def calc_potencia(Torque, Rot):
    """Potência mecânica estimada (W) a partir do binário (Nm) e da rotação (rpm)."""
    return Torque * Rot * 2 * np.pi / 60


def calc_sobrecarga(Desgaste, Torque):
    """Indicador de sobrecarga acumulada (min·Nm)."""
    return Desgaste * Torque


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Devolve uma cópia de ``df`` com ``dT``, ``Potencia`` e ``Sobrecarga``.

    Não altera o DataFrame original. Substitui sempre colunas derivadas já existentes,
    para que nunca haja divergência com os sensores (as derivadas não são editáveis).
    """
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Faltam colunas obrigatórias: {', '.join(missing)}")
    out = df.copy()
    for col in SENSOR_COLUMNS:
        if not pd.api.types.is_numeric_dtype(out[col]):
            raise ValueError(f"A coluna {col} tem de ser numérica (valores em {col}: ver dicionário de variáveis).")
    out["dT"] = calc_dT(out["T_proc"], out["T_ar"])
    out["Potencia"] = calc_potencia(out["Torque"], out["Rot"])
    out["Sobrecarga"] = calc_sobrecarga(out["Desgaste"], out["Torque"])
    return out
