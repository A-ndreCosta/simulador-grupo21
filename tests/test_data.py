import io

import pandas as pd
import pytest

from src.file_import import ImportFileError, apply_mapping, detect_mapping, read_table, template_csv
from src.validation import clean_dataframe, validate_dataframe, validate_inputs

HEADER = "UDI,Product ID,Type,Air temperature [K],Process temperature [K],Rotational speed [rpm],Torque [Nm],Tool wear [min],Machine failure,TWF,HDF,PWF,OSF,RNF\n"
ROW_OK = "1,M1,L,300.0,310.0,1500,40.0,100,0,0,0,0,0,0\n"
ROW_F = "2,M2,M,302.0,309.0,1300,40.0,10,1,0,1,0,0,0\n"


def load(raw: bytes, name="x.csv"):
    df, _ = read_table(raw, name)
    m, miss, amb = detect_mapping(list(df.columns))
    return apply_mapping(df, m), m, miss


def test_valid_csv_ai4i_names():
    df, m, miss = load((HEADER + ROW_OK + ROW_F).encode())
    assert not miss and {"Type", "T_ar", "T_proc", "Rot", "Torque", "Desgaste", "Falha"} <= set(df.columns)
    rep = validate_dataframe(df)
    assert rep.ok and rep.has_label and rep.label_counts == {0: 1, 1: 1}


def test_semicolon_and_decimal_comma_and_cp1252():
    txt = "Tipo;Temperatura do ar [K];Temperatura do processo [K];Rotação [rpm];Binário [Nm];Desgaste [min]\nL;300,5;310,2;1500;40,5;100\n"
    df, m, miss = load(txt.encode("cp1252"))
    assert not miss
    rep = validate_dataframe(df)
    assert rep.ok and not rep.has_label
    clean, _ = clean_dataframe(df, rep)
    assert clean["T_ar"].iloc[0] == 300.5


def test_valid_excel(tmp_path):
    p = tmp_path / "a.xlsx"
    pd.read_csv(io.StringIO(HEADER + ROW_OK + ROW_F)).to_excel(p, index=False)
    df, _, _ = load(p.read_bytes(), "a.xlsx")
    assert validate_dataframe(df).ok


def test_xls_rejected_and_unknown_extension():
    with pytest.raises(ImportFileError):
        read_table(b"abc", "a.xls")
    with pytest.raises(ImportFileError):
        read_table(b"abc", "a.pdf")


def test_empty_and_corrupt_files():
    with pytest.raises(ImportFileError):
        read_table(b"", "a.csv")
    with pytest.raises(ImportFileError):
        read_table(HEADER.encode(), "a.csv")                  # só cabeçalho
    with pytest.raises(ImportFileError):
        read_table(b"PK\x03\x04lixo", "a.xlsx")               # xlsx corrompido


def test_missing_columns_reported():
    df, m, miss = load("Type,Torque [Nm]\nL,40\n".encode())
    assert "T_ar" in miss
    rep = validate_dataframe(df)
    assert rep.errors and not rep.ok


def test_invalid_values_reported_per_row_and_not_dropped_silently():
    bad = "3,M3,X,300.0,310.0,1500,40.0,100,0,0,0,0,0,0\n4,M4,L,abc,310.0,1500,40.0,100,0,0,0,0,0,0\n5,M5,L,300,310,,40,100,0,0,0,0,0,0\n6,M6,L,300,310,1500,-5,100,0,0,0,0,0,0\n"
    df, _, _ = load((HEADER + ROW_OK + bad).encode())
    rep = validate_dataframe(df)
    assert rep.n_invalid == 4 and not rep.ok
    assert set(rep.row_issues["linha"]) == {3, 4, 5, 6}
    with pytest.raises(ValueError):
        clean_dataframe(df, rep)                                 # não descarta em silêncio
    clean, rejected = clean_dataframe(df, rep, drop_invalid=True)
    assert len(clean) == 1 and len(rejected) == 4


def test_nan_inf_detected():
    df, _, _ = load((HEADER + ROW_OK + "7,M7,L,inf,310,1500,40,100,0,0,0,0,0,0\n").encode())
    assert validate_dataframe(df).n_invalid == 1


def test_unknown_category():
    df, _, _ = load((HEADER + "8,M8,Z,300,310,1500,40,100,0,0,0,0,0,0\n").encode())
    assert "categoria" in validate_dataframe(df).row_issues["problema"].iloc[0]


def test_single_class_label_cannot_train():
    df, _, _ = load((HEADER + ROW_OK + ROW_OK).encode())
    rep = validate_dataframe(df)
    assert rep.has_label and not rep.can_train


def test_invalid_label_value():
    df, _, _ = load((HEADER + "9,M9,L,300,310,1500,40,100,2,0,0,0,0,0\n").encode())
    assert validate_dataframe(df).n_invalid == 1


def test_duplicate_mapping_rejected():
    df, _ = read_table((HEADER + ROW_OK).encode(), "x.csv")
    with pytest.raises(ImportFileError):
        apply_mapping(df, {"Type": "Type", "T_ar": "Type"})


def test_template_roundtrip_and_marked_example():
    raw = template_csv()
    assert b"EXEMPLO" in raw
    df, m, miss = load(raw)
    assert not miss and validate_dataframe(df).ok


def test_validate_inputs():
    assert validate_inputs("L", 300, 310, 1500, 40, 100)[0] == []
    assert validate_inputs("Q", 300, 310, 1500, 40, 100)[0]
    assert validate_inputs("L", float("nan"), 310, 1500, 40, 100)[0]
    assert validate_inputs("L", 300, 310, 1500, -1, 100)[0]
    assert validate_inputs("L", 300, 310, 1500, 40, 5000)[0]
