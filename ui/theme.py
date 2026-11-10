"""Tema claro e sóbrio: CSS, cores e modelo Plotly."""
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

GREEN, AMBER, RED, BLUE, VIOLET, GREY = "#16a34a", "#d97706", "#dc2626", "#2563eb", "#7c3aed", "#64748b"
STATE_COLORS = {"NORMAL": GREEN, "ALERTA (regra TWF)": AMBER, "ALERTA (ML)": VIOLET, "FALHA (regra crítica)": RED}

pio.templates["industrial"] = go.layout.Template(layout=go.Layout(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#ffffff", font=dict(color="#0f172a", family="Inter, Helvetica, Arial", size=13),
    colorway=[BLUE, VIOLET, AMBER, GREEN, RED, GREY], margin=dict(l=10, r=20, t=50, b=30),
    title=dict(font=dict(size=15), x=0),
    xaxis=dict(gridcolor="#eef0f3", zerolinecolor="#e2e8f0", linecolor="#e2e8f0"),
    yaxis=dict(gridcolor="#eef0f3", zerolinecolor="#e2e8f0", linecolor="#e2e8f0"),
    legend=dict(bgcolor="rgba(0,0,0,0)")))
pio.templates.default = "industrial"

CSS = """
<style>
.block-container{padding-top:2.2rem;max-width:1180px}
h1{font-weight:700;letter-spacing:-.02em;font-size:1.9rem!important;margin-bottom:.2rem}
h2,h3{letter-spacing:-.01em}
[data-testid="stSidebar"]{background:#0f172a}
[data-testid="stSidebar"] *{color:#e2e8f0}
[data-testid="stSidebar"] hr{border-color:#1e293b}
[data-testid="stSidebar"] .stCaption,[data-testid="stSidebar"] small{color:#94a3b8!important}
.brand{font-weight:700;font-size:1.05rem;letter-spacing:.02em;color:#fff}
.sub{color:#64748b;font-size:.95rem;margin:0 0 1rem 0}
.card{background:#fff;border:1px solid #e5e7eb;border-top:3px solid #2563eb;border-radius:10px;padding:14px 16px;margin-bottom:8px;box-shadow:0 1px 2px rgba(15,23,42,.04)}
.card .k{font-size:.74rem;color:#64748b;text-transform:uppercase;letter-spacing:.05em;font-weight:600}
.card .v{font-size:clamp(1.05rem,1.7vw,1.5rem);font-weight:700;color:#0f172a;line-height:1.25}
.card .s{font-size:.8rem;color:#64748b}
.card.ok{border-top-color:#16a34a}.card.warn{border-top-color:#d97706}.card.fail{border-top-color:#dc2626}.card.ml{border-top-color:#7c3aed}
.badge{display:inline-block;padding:3px 11px;border-radius:12px;font-size:.78rem;font-weight:600}
.badge.syn{background:#fef3c7;color:#92400e;border:1px solid #fde68a}
.statebox{border-radius:10px;padding:14px 18px;font-size:1.2rem;font-weight:700;margin:6px 0 10px 0;border:1px solid}
.statebox.ok{background:#f0fdf4;color:#166534;border-color:#bbf7d0}.statebox.warn{background:#fffbeb;color:#92400e;border-color:#fde68a}
.statebox.fail{background:#fef2f2;color:#991b1b;border-color:#fecaca}.statebox.ml{background:#f5f3ff;color:#5b21b6;border-color:#ddd6fe}
</style>
"""


def inject():
    st.markdown(CSS, unsafe_allow_html=True)


def header(title: str, subtitle: str = ""):
    st.title(title)
    if subtitle:
        st.markdown(f'<p class="sub">{subtitle}</p>', unsafe_allow_html=True)


def card(label: str, value: str, sub: str = "", kind: str = ""):
    st.markdown(f'<div class="card {kind}"><div class="k">{label}</div><div class="v">{value}</div><div class="s">{sub}</div></div>',
                unsafe_allow_html=True)


def state_kind(estado: str) -> str:
    return {"NORMAL": "ok", "ALERTA (regra TWF)": "warn", "ALERTA (ML)": "ml", "FALHA (regra crítica)": "fail"}.get(estado, "")


def state_box(estado: str):
    st.markdown(f'<div class="statebox {state_kind(estado)}">{estado}</div>', unsafe_allow_html=True)


def synthetic_badge(text="Dados sintéticos / simulados"):
    st.markdown(f'<span class="badge syn">{text}</span>', unsafe_allow_html=True)
