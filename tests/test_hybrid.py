import numpy as np
import pandas as pd
import pytest

from src.hybrid import TAUS, HybridSystem, choose_tau
from src.ml_model import split_train_test
from src.rules import STATE_FAIL, STATE_ML, STATE_NORMAL, critical_mask


@pytest.fixture(scope="module")
def split(small_df):
    return split_train_test(small_df)


@pytest.fixture(scope="module")
def hR(split, profiles):
    return HybridSystem(profiles["Trabalho"], "R").fit(split[0])


@pytest.fixture(scope="module")
def hG(split, profiles):
    return HybridSystem(profiles["Trabalho"], "G").fit(split[0])


def test_rules_have_priority(hR, split, profiles):
    te = split[1]
    out = hR.decide(te, tau=0.0)                       # τ=0: o ML dispararia em tudo o que as regras não decidem
    crit = critical_mask(te, profiles["Trabalho"])
    assert (out.loc[crit, "estado"] == STATE_FAIL).all()
    assert (out.loc[~crit, "estado"] == STATE_ML).all()


def test_states_normal_and_ml_alert(hR, split):
    te = split[1]
    none = hR.decide(te, tau=1.01)
    assert set(none["estado"]) <= {STATE_FAIL, STATE_NORMAL}
    some = hR.decide(te, tau=0.3)
    assert (some["p_ML"] >= 0.3)[some["estado"] == STATE_ML].all()


def test_tau_applied(hR, split):
    te = split[1]
    n = [int((hR.decide(te, t)["estado"] == STATE_ML).sum()) for t in (0.1, 0.5, 0.9)]
    assert n[0] >= n[1] >= n[2]
    with pytest.raises(ValueError):
        hR.decide(te, tau=-0.1)


def test_explanation_for_rule_and_ml(hR, split):
    out = hR.decide(split[1], 0.2)
    assert out.loc[out.estado == STATE_FAIL, "explicacao"].str.contains("[HPO][DWS]F", regex=True).all()
    if (out.estado == STATE_ML).any():
        assert out.loc[out.estado == STATE_ML, "explicacao"].str.contains("τ").all()


def test_decision_consistent_with_rules_and_prediction(hR, hG, split, profiles):
    te = split[1]
    for h in (hR, hG):
        d = h.decide(te, h.tau_f1)
        assert (d["estado"].eq(STATE_FAIL) == critical_mask(te, profiles["Trabalho"])).all()
        assert (h.predict(te, h.tau_f1) == (d["estado"] != STATE_NORMAL).astype(int)).all()


def test_variant_R_trained_only_on_non_critical(hR, hG, split, profiles):
    tr = split[0]
    n_non_crit = int((~critical_mask(tr, profiles["Trabalho"])).sum())
    assert hR.effective == "R" and hR.n_train_ml == n_non_crit
    assert hG.effective == "G" and hG.n_train_ml == len(tr)
    assert not hR.fallback and not hG.fallback


def test_tau_selected_without_test_set(split, profiles):
    tr, te = split
    h1 = HybridSystem(profiles["Trabalho"], "R").fit(tr)
    te_flipped = te.assign(Falha=1 - te.Falha)           # alterar o teste não pode mudar τ
    h2 = HybridSystem(profiles["Trabalho"], "R").fit(tr)
    assert (h1.tau_f1, h1.tau_r90) == (h2.tau_f1, h2.tau_r90)
    assert 0.05 <= h1.tau_f1 <= 1.01 and 0.05 <= h1.tau_r90 <= 1.01
    assert len(h1.curve) == len(TAUS)
    _ = te_flipped


def test_choose_tau_rules():
    y = np.array([1] * 10 + [0] * 90)
    flag = np.zeros(100, bool)
    p = np.concatenate([np.linspace(0.6, 0.95, 10), np.linspace(0.0, 0.3, 90)])
    t_f1, t_r90, cur = choose_tau(flag, p, y)
    assert t_f1 > 0.3 and t_r90 == pytest.approx(0.63)      # 9 das 10 falhas têm p ≥ 0,63 (recall 0,9)
    row = cur[cur.tau == t_r90].iloc[0]
    assert row["recall"] >= 0.9
    assert cur[(cur.tau > t_r90) & (cur.recall >= 0.9)].empty


def test_residual_fallback_explicit(profiles):
    from tests.conftest import make_labelled
    df = make_labelled(600, seed=3)
    # todas as falhas são apanhadas pelas regras → o resíduo não tem classe positiva
    from src.rules import critical_mask as cm
    df["Falha"] = cm(df, profiles["Trabalho"]).astype(int)
    h = HybridSystem(profiles["Trabalho"], "R").fit(df)
    assert h.requested == "R" and h.effective == "G" and h.fallback
    assert "G" in h.label and "fallback" in h.label.lower()
    assert not h.label.startswith("Híbrido R")


def test_no_ml_possible_rules_only(profiles):
    from tests.conftest import make_labelled
    df = make_labelled(300, seed=4)
    df["Falha"] = 0
    h = HybridSystem(profiles["Trabalho"], "G").fit(df)
    assert not h.ml_available and h.fallback
    out = h.decide(df)
    assert set(out.estado) <= {STATE_FAIL, STATE_NORMAL}


def test_invalid_args(profiles, small_df):
    with pytest.raises(ValueError):
        HybridSystem(profiles["Trabalho"], "X").fit(small_df)
    with pytest.raises(ValueError):
        HybridSystem(profiles["Trabalho"], "R").fit(small_df.drop(columns="Falha"))


def test_profile_change_changes_level1(split, hR, profiles):
    import copy
    P2 = copy.deepcopy(profiles["Trabalho"])
    P2["pwf"]["potencia_max_w"] = 5000.0
    h2 = HybridSystem(P2, "R").fit(split[0])
    d1, d2 = hR.decide(split[1], 1.01), h2.decide(split[1], 1.01)
    assert (d2.estado == STATE_FAIL).sum() > (d1.estado == STATE_FAIL).sum()
