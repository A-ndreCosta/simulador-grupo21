"""Fluxo de manutenção (diagrama do projeto) e registo de intervenções preenchido pelo utilizador.

Nada é inventado: o registo só contém o que o utilizador escreve. A app NÃO comanda máquinas.
"""
from __future__ import annotations

import pandas as pd

# regra → ramo do fluxograma
FLOW_BRANCH = {"HDF": "Quente", "TWF": "Desgaste", "PWF": "Potência", "RNF": "Random", "OSF": "Sobrecarga"}
BRANCH_HELP = {
    "Quente": "Falha de dissipação de calor (HDF): verificar arrefecimento e ventilação.",
    "Desgaste": "Desgaste da ferramenta (TWF): verificar estado da ferramenta.",
    "Potência": "Falha de potência (PWF): verificar binário e rotação.",
    "Random": "Falha aleatória (RNF): não tem regra determinística; investigação manual.",
    "Sobrecarga": "Sobrecarga (OSF): verificar carga e desgaste acumulado.",
}
LOG_COLUMNS = ["registo", "ramo", "peça", "responsável", "estado", "confirmação"]
ESTADOS = ["Pendente", "Em curso", "Concluída"]
CONFIRMACOES = ["Por confirmar", "Confirmada"]
MAX_TEXT = 120


def branches_for(rules: list[str]) -> list[str]:
    return [FLOW_BRANCH[r] for r in rules if r in FLOW_BRANCH]


def flow_dot(highlight: list[str] | None = None) -> str:
    """Graphviz DOT do fluxo: Máquina → Funciona? → (Não) Porquê? → ramos → Peça → Responsável → Arranja → Confirmação."""
    hl = set(highlight or [])
    def node(name):
        color = ' color="#dc2626" fontcolor="#991b1b" fillcolor="#fee2e2" style="rounded,filled,bold"' if name in hl else ' fillcolor="#eff6ff"'
        return f'"{name}" [label="{name}"{color}];'
    lines = ['digraph G { rankdir=LR; bgcolor="transparent"; node [shape=box, style="rounded,filled", fillcolor="#eff6ff", '
             'color="#93c5fd", fontcolor="#0f172a", fontname="Helvetica"]; edge [color="#94a3b8", fontcolor="#64748b", fontname="Helvetica"];',
             '"Máquina" -> "Funciona?";', '"Funciona?" [shape=diamond];',
             '"Funciona?" -> "Confirmação" [label="Sim"];', '"Funciona?" -> "Porque motivo?" [label="Não"];', '"Porque motivo?" [shape=diamond];']
    for b in FLOW_BRANCH.values():
        lines.append(node(b))
        lines.append(f'"Porque motivo?" -> "{b}";')
        lines.append(f'"{b}" -> "Peça";')
    lines += ['"Peça" -> "Responsável";', '"Responsável" -> "Arranja";', '"Arranja" -> "Confirmação";', "}"]
    return "\n".join(lines)


def empty_log() -> pd.DataFrame:
    return pd.DataFrame(columns=LOG_COLUMNS)


def add_log_entry(log: pd.DataFrame, ramo: str, peca: str, responsavel: str, estado: str, confirmacao: str) -> pd.DataFrame:
    if ramo not in BRANCH_HELP:
        raise ValueError("Ramo desconhecido.")
    if estado not in ESTADOS or confirmacao not in CONFIRMACOES:
        raise ValueError("Estado ou confirmação inválidos.")
    peca, responsavel = (peca or "").strip(), (responsavel or "").strip()
    if not peca or not responsavel:
        raise ValueError("Preencha a peça e o responsável.")
    if len(peca) > MAX_TEXT or len(responsavel) > MAX_TEXT:
        raise ValueError(f"Texto demasiado longo (máx. {MAX_TEXT} caracteres).")
    row = {"registo": len(log) + 1, "ramo": ramo, "peça": peca, "responsável": responsavel, "estado": estado, "confirmação": confirmacao}
    return pd.concat([log, pd.DataFrame([row])], ignore_index=True)
