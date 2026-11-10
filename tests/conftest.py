import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import load_default_profiles  # noqa: E402
from src.features import add_derived_features  # noqa: E402
from src.rules import critical_mask  # noqa: E402


@pytest.fixture(scope="session")
def profiles():
    return load_default_profiles()


@pytest.fixture
def P(profiles):
    import copy
    return copy.deepcopy(profiles["Trabalho"])


def make_labelled(n=1500, seed=0, profile=None):
    """Conjunto SINTÉTICO de teste: sensores aleatórios; falha = regra crítica OU ruído raro. Só para testes."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "Type": rng.choice(["L", "M", "H"], n, p=[.6, .3, .1]),
        "T_ar": rng.normal(300, 2, n), "T_proc": rng.normal(310, 1.5, n),
        "Rot": rng.normal(1540, 170, n).clip(1100, 2900), "Torque": rng.normal(40, 10, n).clip(4, 77),
        "Desgaste": rng.integers(0, 254, n).astype(float)})
    prof = profile or load_default_profiles()["Trabalho"]
    flag = critical_mask(df, prof).to_numpy()
    extra = (add_derived_features(df)["Sobrecarga"].to_numpy() > 9500) & (rng.random(n) < 0.5)
    df["Falha"] = (flag | extra).astype(int)
    return df


@pytest.fixture(scope="session")
def small_df():
    return make_labelled()


@pytest.fixture(scope="session")
def real_df():
    path = ROOT / "data" / "ai4i2020.csv"
    if not path.exists():
        pytest.skip("data/ai4i2020.csv não existe")
    from src.file_import import apply_mapping, detect_mapping, read_table
    from src.validation import clean_dataframe, validate_dataframe
    df, _ = read_table(path.read_bytes(), path.name)
    m, _, _ = detect_mapping(list(df.columns))
    df = apply_mapping(df, m)
    clean, _ = clean_dataframe(df, validate_dataframe(df))
    return clean
