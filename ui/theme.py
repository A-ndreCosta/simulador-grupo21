"""Tema escuro industrial: CSS, cores e modelo Plotly."""
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

GREEN, AMBER, RED, BLUE, VIOLET, GREY = "#22c55e", "#f59e0b", "#ef4444", "#38bdf8", "#8b5cf6", "#9ca3af"
STATE_COLORS = {"NORMAL": GREEN, "ALERTA (regra TWF)": AMBER, "ALERTA (ML)": VIOLET, "FALHA (regra crítica)": RED}

pio.templates["industrial"] = go.layout.Template(layout=go.Layout(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#1d2229", font=dict(color="#e5e7eb", family="Inter, Helvetica, Arial"),
    colorway=[BLUE, VIOLET, AMBER, GREEN, RED, GREY], margin=dict(l=40, r=20, t=50, b=40),
    xaxis=dict(gridcolor="#2b323b", zerolinecolor="#2b323b"), yaxis=dict(gridcolor="#2b323b", zerolinecolor="#2b323b"),
    legend=dict(bgcolor="rgba(0,0,0,0)")))
pio.templates.default = "industrial"

CSS = """
<style>
.card{background:#1d2229;border:1px solid #2b323b;border-left:4px solid #6366f1;border-radius:8px;padding:14px 16px;margin-bottom:8px}
.card .k{font-size:.78rem;color:#9ca3af;text-transform:uppercase;letter-spacing:.04em}
.card .v{font-size:1.25rem;font-weight:600;color:#f3f4f6}
.card .s{font-size:.8rem;color:#9ca3af}
.card.ok{border-left-color:#22c55e}.card.warn{border-left-color:#f59e0b}.card.fail{border-left-color:#ef4444}.card.ml{border-left-color:#8b5cf6}
.badge{display:inline-block;padding:2px 10px;border-radius:12px;font-size:.78rem;font-weight:600}
.badge.syn{background:#78350f;color:#fde68a}
.statebox{border-radius:8px;padding:14px 18px;font-size:1.2rem;font-weight:600;margin:6px 0}
.statebox.ok{background:#14532d;color:#bbf7d0}.statebox.warn{background:#78350f;color:#fde68a}
.statebox.fail{background:#7f1d1d;color:#fecaca}.statebox.ml{background:#4c1d95;color:#ddd6fe}
</style>
"""


def inject():
    st.markdown(CSS, unsafe_allow_html=True)


def card(label: str, value: str, sub: str = "", kind: str = ""):
    st.markdown(f'<div class="card {kind}"><div class="k">{label}</div><div class="v">{value}</div><div class="s">{sub}</div></div>',
                unsafe_allow_html=True)


def state_kind(estado: str) -> str:
    return {"NORMAL": "ok", "ALERTA (regra TWF)": "warn", "ALERTA (ML)": "ml", "FALHA (regra crítica)": "fail"}.get(estado, "")


def state_box(estado: str):
    st.markdown(f'<div class="statebox {state_kind(estado)}">{estado}</div>', unsafe_allow_html=True)


def synthetic_badge(text="Dados sintéticos / simulados"):
    st.markdown(f'<span class="badge syn">{text}</span>', unsafe_allow_html=True)
