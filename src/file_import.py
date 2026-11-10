"""Importação de CSV/XLSX: encoding, separador, nomes AI4I, mapeamento de colunas e modelo para download."""
from __future__ import annotations

import csv
import io
import zipfile

import pandas as pd
from openpyxl.utils.exceptions import InvalidFileException

MAX_FILE_BYTES = 25 * 1024 * 1024
ENCODINGS = ("utf-8-sig", "utf-8", "cp1252", "latin-1")
SEPARATORS = [",", ";", "\t", "|"]

# nome interno → nomes aceites (minúsculas, espaços normalizados)
ALIASES = {
    "Type": ["type", "tipo", "product type", "tipo de produto"],
    "T_ar": ["t_ar", "air temperature [k]", "air temperature", "temperatura do ar [k]", "temperatura do ar"],
    "T_proc": ["t_proc", "process temperature [k]", "process temperature", "temperatura do processo [k]", "temperatura do processo"],
    "Rot": ["rot", "rotational speed [rpm]", "rotational speed", "rotação [rpm]", "rotação", "rotacao"],
    "Torque": ["torque", "torque [nm]", "binário [nm]", "binário", "binario"],
    "Desgaste": ["desgaste", "tool wear [min]", "tool wear", "desgaste [min]", "desgaste da ferramenta"],
    "Falha": ["falha", "machine failure"],
    "UDI": ["udi"],
    "Product ID": ["product id", "productid"],
    "TWF": ["twf"], "HDF": ["hdf"], "PWF": ["pwf"], "OSF": ["osf"], "RNF": ["rnf"],
}
REQUIRED_TARGETS = ["Type", "T_ar", "T_proc", "Rot", "Torque", "Desgaste"]
AI4I_HEADERS = ["UDI", "Product ID", "Type", "Air temperature [K]", "Process temperature [K]", "Rotational speed [rpm]",
                "Torque [Nm]", "Tool wear [min]", "Machine failure", "TWF", "HDF", "PWF", "OSF", "RNF"]


class ImportFileError(ValueError):
    """Ficheiro ilegível, vazio ou inválido (mensagem em português para o utilizador)."""


def _norm(name: str) -> str:
    return " ".join(str(name).strip().lower().split())


def decode_text(raw: bytes) -> tuple[str, str]:
    for enc in ENCODINGS:
        try:
            return raw.decode(enc), enc
        except UnicodeDecodeError:
            continue
    raise ImportFileError("Não foi possível descodificar o ficheiro de texto (codificação desconhecida).")


def detect_separator(text: str) -> str:
    sample = "\n".join(text.splitlines()[:20])
    try:
        return csv.Sniffer().sniff(sample, delimiters="".join(SEPARATORS)).delimiter
    except csv.Error:
        counts = {s: sample.count(s) for s in SEPARATORS}
        best = max(counts, key=counts.get)
        return best if counts[best] else ","


def read_table(raw: bytes, filename: str) -> tuple[pd.DataFrame, dict]:
    """Lê CSV/XLSX como texto (sem converter silenciosamente). Devolve (DataFrame bruto, metadados)."""
    if not raw:
        raise ImportFileError("O ficheiro está vazio.")
    if len(raw) > MAX_FILE_BYTES:
        raise ImportFileError(f"O ficheiro excede o limite de {MAX_FILE_BYTES // (1024 * 1024)} MB.")
    name = filename.lower()
    meta = {"ficheiro": filename, "bytes": len(raw)}
    try:
        if name.endswith(".csv") or name.endswith(".txt"):
            text, enc = decode_text(raw)
            if not text.strip():
                raise ImportFileError("O ficheiro está vazio.")
            sep = detect_separator(text)
            df = pd.read_csv(io.StringIO(text), sep=sep, dtype=str, keep_default_na=True, skip_blank_lines=True)
            meta.update({"formato": "CSV", "codificacao": enc, "separador": "tab" if sep == "\t" else sep})
        elif name.endswith(".xlsx"):
            df = pd.read_excel(io.BytesIO(raw), engine="openpyxl", dtype=object)
            meta.update({"formato": "Excel (.xlsx)"})
        elif name.endswith(".xls"):
            raise ImportFileError("O formato .xls antigo não é suportado; guarde o ficheiro como .xlsx ou CSV.")
        else:
            raise ImportFileError("Formato não suportado. Use CSV ou XLSX.")
    except ImportFileError:
        raise
    except (pd.errors.EmptyDataError, pd.errors.ParserError, ValueError, OSError, KeyError, ImportError) as exc:
        raise ImportFileError(f"Não foi possível ler o ficheiro ({type(exc).__name__}): {exc}") from exc
    except (zipfile.BadZipFile, InvalidFileException) as exc:
        raise ImportFileError(f"O ficheiro Excel está corrompido ou não é um .xlsx válido ({type(exc).__name__}).") from exc
    if df.empty or df.shape[1] == 0:
        raise ImportFileError("O ficheiro não contém linhas de dados.")
    df.columns = [str(c).strip() for c in df.columns]
    return df, meta


def detect_mapping(columns) -> tuple[dict, list[str], dict]:
    """Devolve (mapeamento alvo→coluna de origem, alvos obrigatórios em falta, ambiguidades alvo→[colunas])."""
    mapping, ambiguous = {}, {}
    for target, names in ALIASES.items():
        found = [c for c in columns if _norm(c) in names]
        if len(found) == 1:
            mapping[target] = found[0]
        elif len(found) > 1:
            ambiguous[target] = found
    missing = [t for t in REQUIRED_TARGETS if t not in mapping]
    return mapping, missing, ambiguous


def apply_mapping(df: pd.DataFrame, mapping: dict) -> pd.DataFrame:
    """Renomeia as colunas de origem para o esquema interno, mantendo as restantes colunas."""
    sources = [s for s in mapping.values() if s]
    if len(set(sources)) != len(sources):
        raise ImportFileError("A mesma coluna do ficheiro foi atribuída a mais do que uma variável.")
    for s in sources:
        if s not in df.columns:
            raise ImportFileError(f"A coluna '{s}' não existe no ficheiro.")
    rename = {src: tgt for tgt, src in mapping.items() if src}
    clashes = [t for t, s in rename.items() if t in df.columns and t != s and t not in rename]
    # um alvo que já existe como outra coluna (não mapeada) seria sobrescrito
    existing = [t for t in rename.values() if t in df.columns and t not in rename]
    if existing:
        raise ImportFileError(f"A coluna '{existing[0]}' já existe no ficheiro e não foi mapeada; remova-a ou mapeie-a.")
    return df.rename(columns=rename)


def template_csv() -> bytes:
    """Modelo para download: cabeçalhos AI4I e 3 linhas de EXEMPLO (identificadas, não são dados reais)."""
    rows = [
        ["1", "EXEMPLO-1", "L", "300.0", "310.0", "1500", "40.0", "100", "0", "0", "0", "0", "0", "0"],
        ["2", "EXEMPLO-2", "M", "302.0", "310.0", "1300", "40.0", "10", "1", "0", "1", "0", "0", "0"],
        ["3", "EXEMPLO-3", "H", "300.0", "310.0", "2000", "60.0", "50", "1", "0", "0", "1", "0", "0"],
    ]
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(AI4I_HEADERS)
    w.writerows(rows)
    return ("﻿" + buf.getvalue()).encode("utf-8")


UNITS_TEXT = {
    "Type": "L, M ou H (qualidade do produto)", "Air temperature [K]": "K", "Process temperature [K]": "K",
    "Rotational speed [rpm]": "rpm", "Torque [Nm]": "Nm", "Tool wear [min]": "min",
    "Machine failure": "0/1 (opcional; etiqueta real)", "TWF/HDF/PWF/OSF/RNF": "0/1 (opcional)",
}
