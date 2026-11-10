"""Motor de regras industriais (HDF, PWF, OSF críticas; TWF opcional). RNF não tem regra.

Uma única implementação vetorizada (``evaluate_rules``) serve o simulador manual, a análise de ficheiros,
o treino (marcação do nível 1 do híbrido) e as séries simuladas. Operadores exatamente como no notebook:
PWF: P < min  OU  P > max        HDF: dT < limite  E  Rot < limite        OSF: Sobrecarga >= limite(Type)
TWF: Desgaste >= limite (opcional).  Nenhum valor é arredondado antes de comparar.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .features import SENSOR_COLUMNS, VALID_TYPES, add_derived_features
from .config import check_profile

CRITICAL_RULES = ("HDF", "PWF", "OSF")
ALL_RULES = CRITICAL_RULES + ("TWF",)
STATE_FAIL = "FALHA (regra crítica)"
STATE_TWF = "ALERTA (regra TWF)"
STATE_ML = "ALERTA (ML)"
STATE_NORMAL = "NORMAL"

RULE_INFO = {
    "HDF": ("Heat Dissipation Failure", "dT < limite e Rot < limite: pouco gradiente térmico e pouca ventilação."),
    "PWF": ("Power Failure", "Potência fora dos limites de segurança (abaixo do mínimo ou acima do máximo)."),
    "OSF": ("Overstrain Failure", "Sobrecarga (desgaste × binário) igual ou superior ao limite do tipo de produto."),
    "TWF": ("Tool Wear Failure (opcional)", "Desgaste ≥ limite. Regra probabilística: desgaste elevado não é falha certa e pode gerar muitos falsos alarmes."),
}


def pt(value, nd: int = 2) -> str:
    """Formata um número com vírgula decimal e espaço fino como separador de milhares."""
    s = f"{value:,.{nd}f}"
    return s.replace(",", " ").replace(".", ",")


def evaluate_rules(df: pd.DataFrame, profile: dict, usar_twf: bool | None = None) -> pd.DataFrame:
    """Devolve um DataFrame booleano (colunas HDF, PWF, OSF e, se ativa, TWF) alinhado com o índice de ``df``."""
    check_profile(profile)
    data = add_derived_features(df)
    if not data["Type"].isin(VALID_TYPES).all():
        raise ValueError("O tipo de produto só pode ser L, M ou H.")
    values = data[SENSOR_COLUMNS].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Os sensores contêm valores em falta ou não finitos (NaN/infinito).")
    limit = data["Type"].map(profile["osf"]).astype(float)
    out = pd.DataFrame(index=data.index)
    out["HDF"] = (data["dT"] < profile["hdf"]["dT_max_k"]) & (data["Rot"] < profile["hdf"]["rot_max_rpm"])
    out["PWF"] = (data["Potencia"] < profile["pwf"]["potencia_min_w"]) | (data["Potencia"] > profile["pwf"]["potencia_max_w"])
    out["OSF"] = data["Sobrecarga"] >= limit
    twf_on = profile["twf"]["ativa"] if usar_twf is None else usar_twf
    if twf_on:
        out["TWF"] = data["Desgaste"] >= profile["twf"]["desgaste_min"]
    return out.astype(bool)


def critical_mask(df: pd.DataFrame, profile: dict) -> pd.Series:
    """True onde alguma regra crítica (HDF, PWF, OSF) dispara — nível 1 do sistema híbrido."""
    flags = evaluate_rules(df, profile, usar_twf=False)
    return flags[list(CRITICAL_RULES)].any(axis=1)


def explain_activation(row: pd.Series, rule: str, profile: dict) -> str:
    """Texto com os valores que provocaram a ativação de ``rule`` numa leitura (``row`` com derivadas)."""
    if rule == "PWF":
        lo, hi = profile["pwf"]["potencia_min_w"], profile["pwf"]["potencia_max_w"]
        p = row["Potencia"]
        cond = f"< {pt(lo, 0)} W" if p < lo else f"> {pt(hi, 0)} W"
        return f"PWF: potência = {pt(p)} W {cond}"
    if rule == "HDF":
        return (f"HDF: dT = {pt(row['dT'])} K < {pt(profile['hdf']['dT_max_k'], 1)} K e "
                f"Rot = {pt(row['Rot'], 0)} rpm < {pt(profile['hdf']['rot_max_rpm'], 0)} rpm")
    if rule == "OSF":
        lim = profile["osf"][row["Type"]]
        return f"OSF ({row['Type']}): sobrecarga = {pt(row['Sobrecarga'])} min·Nm ≥ {pt(lim, 0)} min·Nm"
    if rule == "TWF":
        return f"TWF: desgaste = {pt(row['Desgaste'], 0)} min ≥ {pt(profile['twf']['desgaste_min'], 0)} min"
    raise ValueError(f"Regra desconhecida: {rule}")


def rule_state(active: list[str]) -> str:
    if any(r in CRITICAL_RULES for r in active):
        return STATE_FAIL
    if "TWF" in active:
        return STATE_TWF
    return STATE_NORMAL


def analyze_dataframe(df: pd.DataFrame, profile: dict, usar_twf: bool | None = None) -> pd.DataFrame:
    """Aplica as regras a todas as leituras. Preserva identificadores e etiquetas reais (nunca usadas pelas regras)."""
    data = add_derived_features(df)
    flags = evaluate_rules(df, profile, usar_twf)
    rules = [c for c in ALL_RULES if c in flags.columns]
    active_lists = [[r for r in rules if flags.at[i, r]] for i in flags.index]
    out = data.copy()
    out["regras_acionadas"] = ["+".join(a) if a else "—" for a in active_lists]
    out["estado"] = [rule_state(a) for a in active_lists]
    expl = []
    for (idx, row), act in zip(data.iterrows(), active_lists):
        expl.append("; ".join(explain_activation(row, r, profile) for r in act) if act else "Nenhuma regra ativada.")
    out["explicacao"] = expl
    return out


def analyze_one(Type: str, T_ar: float, T_proc: float, Rot: float, Torque: float, Desgaste: float,
                profile: dict, usar_twf: bool | None = None) -> dict:
    """Analisa uma leitura. Devolve entradas, derivadas, regras ativadas, estado e detalhe auditável por regra."""
    df = pd.DataFrame([{"Type": Type, "T_ar": float(T_ar), "T_proc": float(T_proc), "Rot": float(Rot),
                        "Torque": float(Torque), "Desgaste": float(Desgaste)}])
    res = analyze_dataframe(df, profile, usar_twf).iloc[0]
    flags = evaluate_rules(df, profile, usar_twf).iloc[0]
    detail = []
    for rule in ALL_RULES:
        if rule not in flags.index:
            detail.append({"Regra": rule, "Aplicada": "não (desativada)", "Ativada": "—", "Descrição": RULE_INFO[rule][1], "Valores": "—"})
            continue
        detail.append({"Regra": rule, "Aplicada": "sim", "Ativada": "SIM" if flags[rule] else "não",
                       "Descrição": RULE_INFO[rule][1],
                       "Valores": explain_activation(res, rule, profile) if flags[rule] else "—"})
    active = [] if res["regras_acionadas"] == "—" else res["regras_acionadas"].split("+")
    return {
        "entradas": {"Type": Type, "T_ar": float(T_ar), "T_proc": float(T_proc), "Rot": float(Rot),
                     "Torque": float(Torque), "Desgaste": float(Desgaste)},
        "derivadas": {"dT": float(res["dT"]), "Potencia": float(res["Potencia"]), "Sobrecarga": float(res["Sobrecarga"])},
        "regras_acionadas": active, "estado": res["estado"], "explicacao": res["explicacao"], "detalhe": detail,
    }
