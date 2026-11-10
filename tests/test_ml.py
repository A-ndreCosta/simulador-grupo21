import numpy as np
import pandas as pd
import pytest

from src.metrics import classification_metrics, confusion_matrix_df
from src.ml_model import (FORBIDDEN_FEATURES, MODEL_SPECS, NUM_A, NUM_B, TrainingError, assert_no_leakage, check_trainable,
                          make_pipeline, prepare_xy, split_train_test, train_models)


@pytest.fixture(scope="module")
def trained(small_df):
    return train_models(small_df, names=["RF — sensores", "RF — sensores + derivadas", "SVM — sensores"])


def test_four_models_defined():
    assert set(MODEL_SPECS) == {"RF — sensores", "SVM — sensores", "RF — sensores + derivadas", "SVM — sensores + derivadas"}


def test_split_stratified_80_20_no_overlap(small_df):
    tr, te = split_train_test(small_df)
    assert len(tr) + len(te) == len(small_df) and abs(len(te) / len(small_df) - 0.2) < 0.01
    assert set(tr.index).isdisjoint(te.index)
    assert abs(tr.Falha.mean() - te.Falha.mean()) < 0.01


def test_train_and_predict_schema(trained, small_df):
    out = trained.predict("RF — sensores + derivadas", small_df.head(10))
    assert list(out.columns) == ["score_ML", "previsao"] and len(out) == 10
    assert out["score_ML"].between(0, 1).all() and set(out["previsao"]) <= {0, 1}
    assert not out.isna().any().any()


def test_predict_does_not_need_or_use_label(trained, small_df):
    a = trained.predict("RF — sensores", small_df.head(20))
    b = trained.predict("RF — sensores", small_df.head(20).drop(columns="Falha"))
    pd.testing.assert_frame_equal(a, b)


def test_predict_invalid_data(trained, small_df):
    bad = small_df.head(3).copy()
    bad.loc[bad.index[0], "Torque"] = np.nan
    with pytest.raises(ValueError):
        trained.predict("RF — sensores", bad)
    with pytest.raises(KeyError):
        trained.predict("modelo inexistente", small_df.head(2))
    with pytest.raises(ValueError):
        trained.predict("RF — sensores", small_df.head(2).drop(columns="Torque"))


def test_metrics_consistent_with_confusion(trained):
    m = trained.test_metrics["RF — sensores + derivadas"]
    assert m["TP"] + m["FP"] + m["FN"] + m["TN"] == len(trained.test) == m["leituras"]
    assert m["TP"] + m["FN"] == int(trained.test.Falha.sum()) == m["falhas reais"]
    assert m["precisão"] == pytest.approx(m["TP"] / (m["TP"] + m["FP"]))
    assert m["recall"] == pytest.approx(m["TP"] / (m["TP"] + m["FN"]))
    assert 0 <= m["PR-AUC"] <= 1 and 0 <= m["ROC-AUC"] <= 1


def test_metrics_known_values():
    y = [1, 1, 0, 0, 0, 1]
    p = [1, 0, 0, 1, 0, 1]
    m = classification_metrics(y, p)
    assert (m["TP"], m["FP"], m["FN"], m["TN"]) == (2, 1, 1, 2)
    assert m["F1"] == pytest.approx(2 / 3)
    cm = confusion_matrix_df(y, p)
    assert cm.to_numpy().tolist() == [[2, 1], [1, 2]]
    assert np.isnan(m["PR-AUC"])


def test_metrics_reject_bad_input():
    with pytest.raises(ValueError):
        classification_metrics([0, 2], [0, 1])
    with pytest.raises(ValueError):
        classification_metrics([], [])


def test_no_leakage_features():
    for _, num, _ in MODEL_SPECS.values():
        assert not FORBIDDEN_FEATURES.intersection(num)
    for bad in ("UDI", "Product ID", "HDF", "TWF", "Falha", "RNF"):
        with pytest.raises(TrainingError):
            assert_no_leakage(NUM_A + [bad])
    X = prepare_xy(pd.DataFrame({"Type": ["L"], "T_ar": [300.], "T_proc": [310.], "Rot": [1500.], "Torque": [40.], "Desgaste": [1.],
                                 "UDI": [1], "HDF": [1], "Falha": [1]}), NUM_B)
    assert set(X.columns) == {"Type"} | set(NUM_B)


def test_trained_pipeline_inputs_exclude_labels(trained):
    pre = trained.models["RF — sensores"].named_steps["pre"]
    assert "Falha" not in pre.feature_names_in_ and "HDF" not in pre.feature_names_in_


def test_preprocessing_fitted_on_train_only(small_df):
    from src.ml_model import make_pipeline
    tr, te = split_train_test(small_df)
    pipe = make_pipeline("svm", NUM_A, dict(C=1, gamma="scale")).fit(prepare_xy(tr, NUM_A), tr.Falha)
    scaler = pipe.named_steps["pre"].named_transformers_["num"]
    assert scaler.mean_[0] == pytest.approx(tr["T_ar"].mean())


def test_training_errors(small_df):
    with pytest.raises(TrainingError):
        train_models(small_df.drop(columns="Falha"))
    with pytest.raises(TrainingError):
        train_models(small_df.assign(Falha=0))
    with pytest.raises(TrainingError):
        train_models(small_df.head(50))
    few = small_df.copy()
    few["Falha"] = 0
    few.loc[few.index[:3], "Falha"] = 1
    with pytest.raises(TrainingError):
        check_trainable(few)
    with pytest.raises(TrainingError):
        check_trainable(small_df.assign(Falha=2))


def test_reproducible_with_seed(small_df):
    a = train_models(small_df, names=["RF — sensores"], seed=7)
    b = train_models(small_df, names=["RF — sensores"], seed=7)
    assert a.test_metrics["RF — sensores"] == b.test_metrics["RF — sensores"]


def test_comparison_table_dtypes():
    from src.metrics import comparison_table
    m = classification_metrics([1, 0, 1, 0], [1, 0, 0, 0])
    tab = comparison_table({"a": m})
    assert str(tab["TP"].dtype).startswith("int") and tab["PR-AUC"].dtype == float and np.isnan(tab.loc["a", "ROC-AUC"])
