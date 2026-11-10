"""Perfis de máquina: carregamento, validação, duplicação, comparação e (de)serialização segura.

Os perfis predefinidos vêm de ``config/machine_profiles.json`` (ficheiro só de leitura para a aplicação).
As alterações do utilizador vivem na sessão; podem ser exportadas/importadas como JSON, sempre validadas.
"""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
PROFILES_PATH = ROOT / "config" / "machine_profiles.json"
DEFAULT_PROFILE_NAME = "Trabalho"
MAX_JSON_BYTES = 100_000


class ProfileError(ValueError):
    """Perfil inválido. ``args[0]`` é uma lista de mensagens em português."""

    def __init__(self, messages):
        self.messages = list(messages)
        super().__init__("; ".join(self.messages))


def _num(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def validate_profile(profile: dict) -> list[str]:
    """Devolve a lista de problemas (vazia se o perfil é válido)."""
    errors: list[str] = []
    try:
        pwf, hdf, osf, twf = profile["pwf"], profile["hdf"], profile["osf"], profile["twf"]
    except (KeyError, TypeError):
        return ["O perfil tem de conter as secções pwf, hdf, osf e twf."]
    try:
        lo, hi = pwf["potencia_min_w"], pwf["potencia_max_w"]
        dT, rot = hdf["dT_max_k"], hdf["rot_max_rpm"]
        lims = {t: osf[t] for t in ("L", "M", "H")}
        tw, active = twf["desgaste_min"], twf["ativa"]
    except (KeyError, TypeError) as exc:
        return [f"Falta um campo no perfil: {exc}"]
    for label, val in [("Potência mínima PWF (W)", lo), ("Potência máxima PWF (W)", hi), ("dT máximo HDF (K)", dT),
                       ("Rotação máxima HDF (rpm)", rot), ("Limite TWF (min)", tw)] + \
                      [(f"Limite OSF tipo {t} (min·Nm)", v) for t, v in lims.items()]:
        if not _num(val):
            errors.append(f"{label}: valor não numérico ou não finito.")
        elif val <= 0:
            errors.append(f"{label}: tem de ser maior que zero.")
    if not errors and lo >= hi:
        errors.append("PWF: a potência mínima tem de ser menor que a máxima.")
    if not isinstance(active, bool):
        errors.append("TWF: o campo 'ativa' tem de ser verdadeiro ou falso.")
    return errors


def check_profile(profile: dict) -> dict:
    errors = validate_profile(profile)
    if errors:
        raise ProfileError(errors)
    return profile


def load_default_profiles(path: Path = PROFILES_PATH) -> dict[str, dict]:
    with open(path, encoding="utf-8") as fh:
        raw = json.load(fh)
    profiles = raw["perfis"]
    for name, profile in profiles.items():
        errors = validate_profile(profile)
        if errors:
            raise ProfileError([f"Perfil predefinido '{name}' inválido: {e}" for e in errors])
    return profiles


def duplicate_profile(profiles: dict, source: str, new_name: str) -> dict:
    """Devolve uma cópia (não altera ``profiles``)."""
    new_name = (new_name or "").strip()
    if not new_name or len(new_name) > 60:
        raise ProfileError(["O nome do perfil tem de ter entre 1 e 60 caracteres."])
    if new_name in profiles:
        raise ProfileError([f"Já existe um perfil com o nome '{new_name}'."])
    clone = copy.deepcopy(profiles[source])
    clone["descricao"] = f"Cópia de '{source}'."
    clone["origem"] = f"Personalizado (base: {source})"
    return clone


def profile_to_json(profile: dict, name: str) -> str:
    return json.dumps({"versao_esquema": 1, "nome": name, "perfil": profile}, ensure_ascii=False, indent=2)


def profile_from_json(text: str | bytes) -> tuple[str, dict]:
    """Lê um perfil exportado. Só aceita JSON puro (nunca executa código) e valida os limites."""
    raw = text if isinstance(text, bytes) else text.encode("utf-8")
    if len(raw) > MAX_JSON_BYTES:
        raise ProfileError(["Ficheiro de perfil demasiado grande."])
    try:
        data = json.loads(raw.decode("utf-8-sig"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProfileError([f"O ficheiro não é um JSON válido: {exc}"]) from exc
    if not isinstance(data, dict) or "perfil" not in data or not isinstance(data.get("nome"), str):
        raise ProfileError(["Formato inesperado: esperado {'nome': ..., 'perfil': {...}}."])
    clean = {
        "descricao": str(data["perfil"].get("descricao", ""))[:300] if isinstance(data["perfil"], dict) else "",
        "origem": "Importado de ficheiro JSON",
    }
    prof = data["perfil"]
    if not isinstance(prof, dict):
        raise ProfileError(["O campo 'perfil' tem de ser um objecto."])
    for key in ("pwf", "hdf", "osf", "twf"):
        clean[key] = prof.get(key)
    check_profile(clean)
    return data["nome"].strip()[:60], clean


def profile_limits_table(profile: dict) -> pd.DataFrame:
    """Tabela plana (limite → valor, unidade) usada na comparação e na interface."""
    rows = [
        ("PWF — potência mínima", profile["pwf"]["potencia_min_w"], "W"),
        ("PWF — potência máxima", profile["pwf"]["potencia_max_w"], "W"),
        ("HDF — dT máximo", profile["hdf"]["dT_max_k"], "K"),
        ("HDF — rotação máxima", profile["hdf"]["rot_max_rpm"], "rpm"),
        ("OSF — tipo L", profile["osf"]["L"], "min·Nm"),
        ("OSF — tipo M", profile["osf"]["M"], "min·Nm"),
        ("OSF — tipo H", profile["osf"]["H"], "min·Nm"),
        ("TWF — desgaste mínimo", profile["twf"]["desgaste_min"], "min"),
        ("TWF — regra ativa", "sim" if profile["twf"]["ativa"] else "não", "—"),
    ]
    return pd.DataFrame(rows, columns=["Limite", "Valor", "Unidade"])


def compare_profiles(original: dict, custom: dict) -> pd.DataFrame:
    a, b = profile_limits_table(original), profile_limits_table(custom)
    out = a.rename(columns={"Valor": "Original"})
    out["Original"] = out["Original"].astype(str)
    out["Personalizado"] = b["Valor"].astype(str).values
    out["Alterado"] = [("sim" if str(x) != str(y) else "não") for x, y in zip(out["Original"], out["Personalizado"])]
    return out
