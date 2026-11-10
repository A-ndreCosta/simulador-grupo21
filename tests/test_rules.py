import copy
import math

import pandas as pd
import pytest

from src.config import ProfileError
from src.rules import STATE_FAIL, STATE_NORMAL, STATE_TWF, analyze_dataframe, analyze_one, evaluate_rules


def rules_of(P, Type="L", T_ar=300.0, T_proc=310.0, Rot=1500.0, Torque=40.0, Desgaste=10.0, twf=None):
    return analyze_one(Type, T_ar, T_proc, Rot, Torque, Desgaste, P, twf)["regras_acionadas"]


def test_normal(P):
    assert rules_of(P) == []
    assert analyze_one("L", 300, 310, 1500, 40, 100, P)["estado"] == STATE_NORMAL


# PWF: P = Torque*Rot*2π/60
def torque_for_power(power, rot):
    return power * 60 / (2 * math.pi * rot)


def test_pwf_below_at_above_lower_limit(P):
    rot = 1500.0
    assert "PWF" in rules_of(P, Rot=rot, Torque=torque_for_power(3499.0, rot))
    assert "PWF" not in rules_of(P, Rot=rot, Torque=torque_for_power(3501.0, rot))


def test_pwf_upper(P):
    rot = 1500.0
    assert "PWF" in rules_of(P, Rot=rot, Torque=torque_for_power(9001.0, rot))
    assert "PWF" not in rules_of(P, Rot=rot, Torque=torque_for_power(8999.0, rot))


def test_pwf_limit_exact_strict_with_custom_integer_profile(P):
    # perfil com limite exatamente atingível: Rot=60 rpm → P = Torque·2π; usamos limites que coincidem com o valor calculado
    df = pd.DataFrame([dict(Type="L", T_ar=300.0, T_proc=310.0, Rot=1500.0, Torque=40.0, Desgaste=0.0)])
    from src.features import calc_potencia
    p = float(calc_potencia(40.0, 1500.0))
    P["pwf"]["potencia_min_w"], P["pwf"]["potencia_max_w"] = p, p + 1
    assert not evaluate_rules(df, P)["PWF"].iloc[0]          # P == mínimo → não dispara (operador <)
    P["pwf"]["potencia_min_w"], P["pwf"]["potencia_max_w"] = p - 1, p
    assert not evaluate_rules(df, P)["PWF"].iloc[0]          # P == máximo → não dispara (operador >)


def test_hdf_both_conditions(P):
    assert rules_of(P, T_ar=302, T_proc=309, Rot=1300) == ["HDF"]


def test_hdf_only_one_condition(P):
    assert "HDF" not in rules_of(P, T_ar=302, T_proc=309, Rot=1500)   # só dT baixo
    assert "HDF" not in rules_of(P, T_ar=300, T_proc=310, Rot=1300)   # só rotação baixa


def test_hdf_exact_limits_not_active(P):
    df = pd.DataFrame([dict(Type="L", T_ar=0.0 + 300, T_proc=300 + 8.6, Rot=1380.0, Torque=40.0, Desgaste=0.0)])
    d = float((df.T_proc - df.T_ar).iloc[0])
    P["hdf"]["dT_max_k"] = d
    assert not evaluate_rules(df, P)["HDF"].iloc[0]          # dT == limite → não dispara
    P["hdf"]["dT_max_k"] = d + 1e-9
    P["hdf"]["rot_max_rpm"] = 1380.0
    assert not evaluate_rules(df, P)["HDF"].iloc[0]          # Rot == limite → não dispara
    P["hdf"]["rot_max_rpm"] = 1380.0001
    assert evaluate_rules(df, P)["HDF"].iloc[0]


@pytest.mark.parametrize("Type,limit", [("L", 11000), ("M", 12337), ("H", 13234)])
def test_osf_per_type(P, Type, limit):
    below = analyze_one(Type, 300, 310, 1500, 40.0, (limit - 1) / 40.0, P)
    assert "OSF" not in below["regras_acionadas"]
    at = analyze_one(Type, 300, 310, 1500, 40.0, limit / 40.0, P)
    assert "OSF" in at["regras_acionadas"] and at["estado"] == STATE_FAIL


def test_osf_uci_profile_limits(profiles):
    U = profiles["UCI"]
    # Sobrecarga 12100: H(UCI 13000) não dispara; M (UCI 12000) dispara; no perfil Trabalho M=12337 não dispara
    assert "OSF" not in analyze_one("H", 300, 310, 1500, 55, 220, U)["regras_acionadas"]
    assert "OSF" in analyze_one("M", 300, 310, 1500, 55, 220, U)["regras_acionadas"]
    assert "OSF" not in analyze_one("M", 300, 310, 1500, 55, 220, profiles["Trabalho"])["regras_acionadas"]


def test_twf_off_by_default_and_on(P):
    assert "TWF" not in rules_of(P, Desgaste=240)
    r = analyze_one("L", 300, 310, 1500, 30, 240, P, usar_twf=True)
    assert r["regras_acionadas"] == ["TWF"] and r["estado"] == STATE_TWF
    P["twf"]["ativa"] = True
    assert "TWF" in rules_of(P, Desgaste=198)
    assert "TWF" not in rules_of(P, Desgaste=197)


def test_no_rnf_rule(P):
    df = pd.DataFrame([dict(Type="L", T_ar=300.0, T_proc=310.0, Rot=1500.0, Torque=40.0, Desgaste=10.0)])
    assert "RNF" not in evaluate_rules(df, P, usar_twf=True).columns


def test_custom_profile_changes_result(P):
    assert rules_of(P, Rot=1500, Torque=40) == []
    P["pwf"]["potencia_max_w"] = 5000.0
    assert rules_of(P, Rot=1500, Torque=40) == ["PWF"]


def test_invalid_profile_rejected(P):
    bad = copy.deepcopy(P)
    bad["pwf"]["potencia_min_w"], bad["pwf"]["potencia_max_w"] = 9000, 3500
    with pytest.raises(ProfileError):
        analyze_one("L", 300, 310, 1500, 40, 10, bad)


def test_invalid_data_raise(P):
    with pytest.raises(ValueError):
        analyze_one("X", 300, 310, 1500, 40, 10, P)
    df = pd.DataFrame([dict(Type="L", T_ar=float("nan"), T_proc=310.0, Rot=1500.0, Torque=40.0, Desgaste=1.0)])
    with pytest.raises(ValueError):
        evaluate_rules(df, P)


def test_explanation_contains_values(P):
    r = analyze_one("M", 302, 309, 1300, 40, 10, P)
    assert "HDF" in r["explicacao"] and "7,00" in r["explicacao"]


def test_dataframe_preserves_columns_and_state(P):
    df = pd.DataFrame([dict(Type="L", T_ar=300.0, T_proc=310.0, Rot=1500.0, Torque=40.0, Desgaste=1.0, extra="x")])
    out = analyze_dataframe(df, P)
    assert out["extra"].iloc[0] == "x" and out["estado"].iloc[0] == STATE_NORMAL


def test_derived_formulas():
    from src.features import calc_dT, calc_potencia, calc_sobrecarga
    assert calc_dT(310.0, 300.0) == 10.0
    assert calc_potencia(40.0, 1500.0) == pytest.approx(40 * 1500 * 2 * math.pi / 60)
    assert calc_sobrecarga(100, 40) == 4000


def test_derived_not_editable_overwritten():
    from src.features import add_derived_features
    df = pd.DataFrame([dict(Type="L", T_ar=300.0, T_proc=310.0, Rot=1500.0, Torque=40.0, Desgaste=100.0, dT=999.0)])
    assert add_derived_features(df)["dT"].iloc[0] == 10.0
