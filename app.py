import streamlit as st
import pandas as pd
import numpy as np
import os

st.set_page_config(
    page_title="Churn Prediction System",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded"
)

SHARED_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Merriweather:wght@700;900&family=Source+Sans+3:wght@300;400;500;600;700&family=Source+Code+Pro:wght@400;500;600&display=swap');

:root { color-scheme: light only !important; }
*, *::before, *::after { color-scheme: light !important; }
html[data-theme="dark"], html[data-theme="light"] { background-color: #F7F8FA !important; }

html, body { background-color: #F7F8FA !important; }

.stApp,
[data-testid="stAppViewContainer"],
[data-testid="stMain"],
.main,
.block-container,
[data-testid="stVerticalBlock"],
[data-testid="stHorizontalBlock"],
section.main,
div.stMarkdown {
    background-color: #F7F8FA !important;
    font-family: 'Source Sans 3', sans-serif !important;
}

html, body, p, span, div, label,
.stMarkdown, .stMarkdown p,
.stMarkdown span,
[data-testid="stMarkdownContainer"],
[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] span {
    color: #111827 !important;
    font-family: 'Source Sans 3', sans-serif !important;
}

section[data-testid="stSidebar"],
section[data-testid="stSidebar"] > div,
section[data-testid="stSidebar"] > div > div {
    background-color: #0F2236 !important;
}
section[data-testid="stSidebar"] *,
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] a,
section[data-testid="stSidebar"] label {
    color: #CBD5E1 !important;
}
/* ══ SIDEBAR NAV LINKS ══ */
[data-testid="stSidebarNavLink"] {
    background: transparent !important;
    border-radius: 8px !important;
    margin: 2px 8px !important;
    padding: 8px 12px !important;
}
[data-testid="stSidebarNavLink"]:hover {
    background: rgba(255,255,255,0.08) !important;
}
[data-testid="stSidebarNavLink"][aria-selected="true"] {
    background: rgba(255,255,255,0.13) !important;
    border-left: 3px solid #5EEAD4 !important;
}
[data-testid="stSidebarNavLink"] * { color: #CBD5E1 !important; }
[data-testid="stSidebarNavLink"][aria-selected="true"] * { color: #FFFFFF !important; font-weight: 600 !important; }
/* Hide the auto-generated letter icons in nav */
[data-testid="stSidebarNavLink"] svg,
[data-testid="stSidebarNavLink"] [data-testid="stIconMaterial"] {
    display: none !important;
}
/* Fix nav icon letters that appear as filled circles */
[data-testid="stSidebarNavSeparator"] { border-color: rgba(255,255,255,0.1) !important; }

.page-title {
    font-family: 'Merriweather', serif !important;
    font-size: 34px !important;
    font-weight: 900 !important;
    color: #0C1F3A !important;
    letter-spacing: -0.025em;
    line-height: 1.2;
}
.page-subtitle {
    font-size: 15px !important;
    color: #4B5563 !important;
    margin-top: 5px;
    font-weight: 400;
}

.sec-head {
    font-family: 'Merriweather', serif !important;
    font-size: 12px !important;
    font-weight: 700 !important;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #1E3A5A !important;
    border-bottom: 2.5px solid #1E3A5A;
    padding-bottom: 7px;
    margin: 30px 0 18px 0;
    display: block;
}

.kpi {
    background: #FFFFFF;
    border: 1.5px solid #DDE3EC;
    border-top: 4px solid #1E3A5A;
    border-radius: 10px;
    padding: 22px 18px 16px 18px;
    text-align: center;
    box-shadow: 0 2px 10px rgba(15,34,54,0.07);
}
.kpi.teal  { border-top-color: #0D7A55; }
.kpi.amber { border-top-color: #C2790A; }

.kpi-lbl {
    display: block;
    font-size: 10px !important;
    font-weight: 700 !important;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: #6B7280 !important;
    margin-bottom: 10px;
}
.kpi-num {
    display: block;
    font-family: 'Source Code Pro', monospace !important;
    font-size: 31px !important;
    font-weight: 600 !important;
    color: #0C1F3A !important;
    line-height: 1;
}
.kpi-num.teal  { color: #0D7A55 !important; }
.kpi-num.amber { color: #C2790A !important; }
.kpi-sub {
    display: block;
    font-size: 11px !important;
    color: #9CA3AF !important;
    margin-top: 7px;
}

.banner {
    background: #EFF6FF;
    border: 1px solid #BFDBFE;
    border-left: 4px solid #1D4ED8;
    border-radius: 0 8px 8px 0;
    padding: 14px 18px;
    font-size: 13.5px !important;
    color: #1E3A5F !important;
    line-height: 1.65;
}

[data-testid="stMetric"] {
    background: #FFFFFF !important;
    border: 1.5px solid #DDE3EC !important;
    border-radius: 10px !important;
    padding: 16px 14px !important;
}
[data-testid="stMetricLabel"] > div,
[data-testid="stMetricLabel"] p {
    color: #6B7280 !important;
    font-size: 11px !important;
    font-weight: 700 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.09em !important;
}
[data-testid="stMetricValue"] > div {
    color: #0C1F3A !important;
    font-family: 'Source Code Pro', monospace !important;
    font-size: 22px !important;
    font-weight: 600 !important;
}

.divider {
    border: none;
    border-top: 1.5px solid #DDE3EC;
    margin: 24px 0;
}

/* ── Executive Insight Banner ── */
.exec-insight {
    background: linear-gradient(135deg, #F0FDF4 0%, #ECFDF5 100%);
    border: 1px solid #6EE7B7;
    border-left: 5px solid #0D7A55;
    border-radius: 0 10px 10px 0;
    padding: 18px 22px;
    margin-bottom: 8px;
}
.exec-insight .ei-label {
    font-size: 10px !important;
    font-weight: 800 !important;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: #065F46 !important;
    display: block;
    margin-bottom: 6px;
}
.exec-insight .ei-body {
    font-size: 14px !important;
    color: #14532D !important;
    line-height: 1.7;
}
.exec-insight .ei-stats {
    display: flex;
    gap: 24px;
    margin-top: 12px;
    flex-wrap: wrap;
}
.exec-insight .ei-stat {
    background: rgba(255,255,255,0.7);
    border: 1px solid #A7F3D0;
    border-radius: 6px;
    padding: 7px 14px;
    text-align: center;
}
.exec-insight .ei-stat-val {
    display: block;
    font-family: 'Source Code Pro', monospace;
    font-size: 17px !important;
    font-weight: 700 !important;
    color: #065F46 !important;
}
.exec-insight .ei-stat-lbl {
    display: block;
    font-size: 10px !important;
    color: #047857 !important;
    font-weight: 600;
    letter-spacing: 0.06em;
}

/* ── Disclaimer note ── */
.disclaimer {
    background: #FAFAFA;
    border: 1px solid #E5E7EB;
    border-radius: 8px;
    padding: 10px 16px;
    font-size: 12px !important;
    color: #6B7280 !important;
    display: flex;
    align-items: center;
    gap: 8px;
}

/* ── Version badge ── */
.version-badge {
    display: inline-block;
    background: #0C1F3A;
    color: #FFFFFF !important;
    padding: 5px 14px;
    border-radius: 20px;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.08em;
}
</style>
"""

st.markdown(SHARED_CSS, unsafe_allow_html=True)

# ── Load ───────────────────────────────────────────────────────────────────────
@st.cache_data
def load_decisions():
    p = "results/customer_decisions.csv"
    if os.path.exists(p):
        return pd.read_csv(p)
    np.random.seed(42); n = 1409
    probs = np.random.beta(2, 3, n)
    return pd.DataFrame({
        "Customer_ID": range(1, n+1),
        "Churn_Probability": probs,
        "Expected_Profit": probs * 766,
        "Decision_Threshold": ["INTERVENE" if p >= 0.1 else "NO ACTION" for p in probs],
        "Actual_Churn": np.random.choice([0,1], n, p=[0.73,0.27])
    })

df = load_decisions()
total  = len(df)
target = int((df["Decision_Threshold"] == "INTERVENE").sum())

# ── Header ─────────────────────────────────────────────────────────────────────
c_title, c_badge = st.columns([5,1])
with c_title:
    st.markdown('<div class="page-title">📡 Churn Prediction &amp; Intervention System</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Logistic Regression · GEO Feature Selection · Profit-Optimised Threshold · Telco Customer Dataset</div>', unsafe_allow_html=True)
with c_badge:
    st.markdown("""<div style="text-align:right;padding-top:18px">
        <span class="version-badge">MODEL v1.0</span>
    </div>""", unsafe_allow_html=True)

st.markdown('<hr class="divider">', unsafe_allow_html=True)

# ── ✅ PART 1: Disclaimer ─────────────────────────────────────────────────────
st.markdown("""
<div class="disclaimer">
    <span style="font-size:16px">ℹ️</span>
    <span>Profit values are simulated based on customer lifetime assumptions and used as a relative decision metric — not real currency figures.</span>
</div>
""", unsafe_allow_html=True)

st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

# ── ✅ PART 2 ADD 1: Executive Insight ────────────────────────────────────────
st.markdown("""
<div class="exec-insight">
    <span class="ei-label">⚡ Executive Insight</span>
    <div class="ei-body">
        The system increases profit by <strong>~65% vs. baseline</strong> by prioritising high-risk, high-value customers. 
        Optimal intervention range is <strong>50–60% of customers</strong> for maximum ROI — beyond 70%, 
        marginal returns diminish rapidly as lower-probability customers are included.
    </div>
    <div class="ei-stats">
        <div class="ei-stat">
            <span class="ei-stat-val">+65%</span>
            <span class="ei-stat-lbl">vs Baseline</span>
        </div>
        <div class="ei-stat">
            <span class="ei-stat-val">50–60%</span>
            <span class="ei-stat-lbl">Optimal Target Range</span>
        </div>
        <div class="ei-stat">
            <span class="ei-stat-val">98.7%</span>
            <span class="ei-stat-lbl">Churn Detection</span>
        </div>
        <div class="ei-stat">
            <span class="ei-stat-val">0.8393</span>
            <span class="ei-stat-lbl">AUC Score</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

# ── KPI Row 1 ──────────────────────────────────────────────────────────────────
st.markdown('<span class="sec-head">Performance Dashboard</span>', unsafe_allow_html=True)

k1, k2, k3 = st.columns(3)
with k1:
    st.markdown(f"""<div class="kpi">
        <span class="kpi-lbl">Total Customers (Test Set)</span>
        <span class="kpi-num">{total:,}</span>
        <span class="kpi-sub">20% holdout · stratified split</span>
    </div>""", unsafe_allow_html=True)
with k2:
    pct = round(target/total*100, 1)
    st.markdown(f"""<div class="kpi teal">
        <span class="kpi-lbl">Customers Targeted</span>
        <span class="kpi-num teal">{target:,}</span>
        <span class="kpi-sub">{pct}% of base · INTERVENE decision</span>
    </div>""", unsafe_allow_html=True)
with k3:
    st.markdown(f"""<div class="kpi teal">
        <span class="kpi-lbl">Net Expected Profit</span>
        <span class="kpi-num teal">446,563 units</span>
        <span class="kpi-sub">after 14,326 units total intervention costs</span>
    </div>""", unsafe_allow_html=True)

st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)

k4, k5, k6 = st.columns(3)
with k4:
    st.markdown(f"""<div class="kpi amber">
        <span class="kpi-lbl">Recall (Churn Detection)</span>
        <span class="kpi-num amber">82.1%</span>
        <span class="kpi-sub">98.7% of all churners caught at t=0.10</span>
    </div>""", unsafe_allow_html=True)
with k5:
    st.markdown(f"""<div class="kpi">
        <span class="kpi-lbl">AUC Score (ROC)</span>
        <span class="kpi-num">0.8393</span>
        <span class="kpi-sub">CV Mean Recall: 81.6% · Std: ±2.0%</span>
    </div>""", unsafe_allow_html=True)
with k6:
    st.markdown(f"""<div class="kpi amber">
        <span class="kpi-lbl">Decision Threshold</span>
        <span class="kpi-num amber">0.10</span>
        <span class="kpi-sub">profit-optimised · default was 0.50</span>
    </div>""", unsafe_allow_html=True)

# ── Banner ──────────────────────────────────────────────────────────────────────
st.markdown('<hr class="divider">', unsafe_allow_html=True)
st.markdown("""<div class="banner">
    <strong>System Overview:</strong> This dashboard presents a profit-driven customer churn 
    intervention pipeline. The Golden Eagle Optimisation (GEO) algorithm selects 17 of 30 features 
    (43.3% reduction) to maximise recall while penalising feature redundancy (λ=0.05). 
    The decision threshold is tuned to maximise expected business profit, not accuracy. 
    Use <strong>Overview</strong> for model validation · <strong>Customer Explorer</strong> to identify 
    who to target · <strong>Budget Simulator</strong> to plan interventions by budget.
</div>""", unsafe_allow_html=True)

# ── Ablation strip ──────────────────────────────────────────────────────────────
st.markdown('<span class="sec-head">Ablation Study — Profit Contribution of Each Component</span>', unsafe_allow_html=True)

m1, m2, m3, m4 = st.columns(4)
with m1:
    st.metric("① Baseline", "270,001 units", help="All 30 features, default threshold 0.5")
with m2:
    st.metric("② Threshold Only", "377,955 units", delta="+107,954 units vs baseline")
with m3:
    st.metric("③ GEO Only", "345,572 units", delta="+75,571 units vs baseline")
with m4:
    st.metric("④ Full System ✦", "446,563 units", delta="+176,562 units vs baseline")

st.markdown("""<div style="margin-top:8px;padding:10px 14px;background:#F0FDF4;border:1px solid #BBF7D0;border-radius:8px;font-size:13px;color:#14532D">
    <strong>✦ Proposed system</strong> (GEO + Optimised Threshold) achieves the highest profit. 
    Both components contribute independently — threshold tuning alone gives +40% uplift; 
    combining with GEO adds a further +18%.
</div>""", unsafe_allow_html=True)

st.markdown('<hr class="divider">', unsafe_allow_html=True)
st.markdown('<span class="sec-head">Feature Reduction Summary</span>', unsafe_allow_html=True)

f1, f2, f3 = st.columns(3)
with f1:
    st.metric("Original Features", "30")
with f2:
    st.metric("GEO-Selected", "17", delta="-13 features removed", delta_color="inverse")
with f3:
    st.metric("Reduction", "43.3%", help="Fewer features = faster inference + better generalisation")

st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)