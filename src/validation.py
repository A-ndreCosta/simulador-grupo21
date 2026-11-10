"""Validação de entradas manuais e de ficheiros. Nunca descarta registos em silêncio."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .features import REQUIRED_COLUMNS, SENSOR_COLUMNS, VALID_TYPES

MAX_ROWS = 100_000

# (mínimo permitido, máximo permitido, mínimo observado no AI4I, máximo observado no AI4I, unidade)
INPUT_LIMITS = {
    "T_ar": (250.0, 400.0, 295.3, 304.5, "K"),
    "T_proc": (250.0, 450.0, 305.7, 313.8, "K"),
    "Rot": (0.0, 10000.0, 1168.0, 2886.0, "rpm"),
    "Torque": (0.0, 500.0, 3.8, 76.6, "Nm"),
    "Desgaste": (0.0, 1000.0, 0.0, 253.0, "min"),
}


def to_numeric_pt(series: pd.Series) -> pd.Series:
    """Converte para número, aceitando vírgula decimal. Valores inválidos ficam NaN (e são reportados, não escondidos)."""
    if pd.api.types.is_numeric_dtype(series):
        return series.astype(float)
    text = series.astype("string").str.strip().str.replace(",", ".", regex=False)
    return pd.to_numeric(text, errors="coerce").astype(float)


def validate_inputs(Type, T_ar, T_proc, Rot, Torque, Desgaste) -> tuple[list[str], list[str]]:
    """Valida uma leitura manual. Devolve (erros, avisos)."""
    errors, warnings = [], []
    if Type not in VALID_TYPES:
        errors.append("O tipo de produto tem de ser L, M ou H.")
    for name, value in zip(SENSOR_COLUMNS, [T_ar, T_proc, Rot, Torque, Desgaste]):
        lo, hi, olo, ohi, unit = INPUT_LIMITS[name]
        try:
            v = float(value)
        except (TypeError, ValueError):
            errors.append(f"{name}: valor não numérico.")
            continue
        if not np.isfinite(v):
            errors.append(f"{name}: valor não finito.")
        elif v < lo or v > hi:
            errors.append(f"{name} = {v:g} {unit} fora do intervalo permitido [{lo:g}; {hi:g}] {unit}.")
        elif v < olo or v > ohi:
            warnings.append(f"{name} = {v:g} {unit} está fora do intervalo observado no AI4I [{olo:g}; {ohi:g}] {unit}.")
    return errors, warnings


@dataclass
class ValidationReport:
    n_rows: int = 0
    errors: list[str] = field(default_factory=list)          # problemas globais (colunas em falta, vazio…)
    warnings: list[str] = field(default_factory=list)
    row_issues: pd.DataFrame = field(default_factory=lambda: pd.DataFrame(columns=["linha", "coluna", "valor", "problema"]))
    invalid_positions: set = field(default_factory=set)       # posições (0-based) das linhas inválidas
    has_label: bool = False
    label_counts: dict = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.errors and not self.invalid_positions

    @property
    def n_invalid(self) -> int:
        return len(self.invalid_positions)

    @property
    def can_train(self) -> bool:
        return self.has_label and set(self.label_counts) == {0, 1}


def validate_dataframe(df: pd.DataFrame, max_rows: int = MAX_ROWS) -> ValidationReport:
    rep = ValidationReport(n_rows=len(df))
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        rep.errors.append(f"Colunas obrigatórias em falta: {', '.join(missing)}.")
        return rep
    if df.empty:
        rep.errors.append("O ficheiro não contém leituras.")
        return rep
    if len(df) > max_rows:
        rep.errors.append(f"O ficheiro tem {len(df)} linhas; o máximo suportado é {max_rows}.")
        return rep

    issues = []

    def add(mask, col, problem):
        for pos in np.flatnonzero(mask.to_numpy()):
            rep.invalid_positions.add(int(pos))
            issues.append({"linha": int(pos) + 2, "coluna": col, "valor": str(df[col].iloc[pos]), "problema": problem})

    for col in SENSOR_COLUMNS:
        raw = df[col]
        num = to_numeric_pt(raw)
        was_null = raw.isna() | (raw.astype("string").str.strip() == "")
        add(was_null, col, "valor em falta")
        add(num.isna() & ~was_null, col, "valor não numérico")
        add(pd.Series(np.isinf(num.to_numpy(dtype=float)), index=df.index), col, "valor não finito")
        lo, hi, olo, ohi, unit = INPUT_LIMITS[col]
        add((num < lo) | (num > hi), col, f"fora do intervalo permitido [{lo:g}; {hi:g}] {unit}")
        out_obs = int((((num < olo) | (num > ohi)) & ~((num < lo) | (num > hi))).sum())
        if out_obs:
            rep.warnings.append(f"{col}: {out_obs} valores fora do intervalo observado no AI4I [{olo:g}; {ohi:g}] {unit}.")
    types = df["Type"].astype("string").str.strip().str.upper()
    add(types.isna() | ~types.isin(VALID_TYPES), "Type", "categoria desconhecida (só L, M ou H)")

    if "Falha" in df.columns:
        rep.has_label = True
        lab = to_numeric_pt(df["Falha"])
        add(~lab.isin([0.0, 1.0]), "Falha", "etiqueta inválida (só 0 ou 1)")
        counts = lab[lab.isin([0.0, 1.0])].astype(int).value_counts().to_dict()
        rep.label_counts = {int(k): int(v) for k, v in counts.items()}
    if issues:
        rep.row_issues = pd.DataFrame(issues)
    return rep


def clean_dataframe(df: pd.DataFrame, report: ValidationReport, drop_invalid: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Converte para tipos numéricos. Se houver linhas inválidas só as remove com ``drop_invalid=True`` (escolha explícita).

    Devolve (dados limpos, linhas rejeitadas). Com linhas inválidas e ``drop_invalid=False`` levanta ValueError.
    """
    if report.errors:
        raise ValueError("; ".join(report.errors))
    if report.invalid_positions and not drop_invalid:
        raise ValueError(f"Existem {report.n_invalid} linhas inválidas; corrija o ficheiro ou escolha excluí-las explicitamente.")
    out = df.copy()
    for col in SENSOR_COLUMNS + (["Falha"] if report.has_label else []):
        out[col] = to_numeric_pt(out[col])
    out["Type"] = out["Type"].astype("string").str.strip().str.upper().astype(object)
    mask = np.zeros(len(out), dtype=bool)
    if report.invalid_positions:
        mask[list(report.invalid_positions)] = True
    rejected = df.iloc[np.flatnonzero(mask)].copy()
    clean = out.iloc[np.flatnonzero(~mask)].reset_index(drop=True)
    if report.has_label:
        clean["Falha"] = clean["Falha"].astype(int)
    return clean, rejected
