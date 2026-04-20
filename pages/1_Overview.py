import streamlit as st
import pandas as pd
import os

st.set_page_config(page_title="Overview · Churn System", page_icon="📊", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Merriweather:wght@700;900&family=Source+Sans+3:wght@300;400;500;600;700&family=Source+Code+Pro:wght@400;500;600&display=swap');

:root { color-scheme: light only !important; }
*, *::before, *::after { color-scheme: light !important; }
html[data-theme="dark"], html[data-theme="light"] { background-color: #F7F8FA !important; }

html, body,
.stApp,
[data-testid="stAppViewContainer"],
[data-testid="stMain"],
.main, .block-container,
[data-testid="stVerticalBlock"],
[data-testid="stHorizontalBlock"],
div.stMarkdown,
[data-testid="stMarkdownContainer"] {
    background-color: #F7F8FA !important;
    color: #111827 !important;
    font-family: 'Source Sans 3', sans-serif !important;
}
p, span, div, label,
.stMarkdown p, .stMarkdown span,
[data-testid="stMarkdownContainer"] p {
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
section[data-testid="stSidebar"] span {
    color: #CBD5E1 !important;
}

.page-title {
    font-family: 'Merriweather', serif !important;
    font-size: 30px !important; font-weight: 900 !important;
    color: #0C1F3A !important; letter-spacing: -0.02em;
}
.page-sub { font-size: 15px !important; color: #4B5563 !important; margin-top: 5px; }

.sec-head {
    font-family: 'Merriweather', serif !important;
    font-size: 12px !important; font-weight: 700 !important;
    letter-spacing: 0.12em; text-transform: uppercase;
    color: #1E3A5A !important;
    border-bottom: 2.5px solid #1E3A5A;
    padding-bottom: 7px; margin: 30px 0 18px 0; display: block;
}

.rtable {
    width: 100%; border-collapse: collapse;
    font-size: 14px; font-family: 'Source Sans 3', sans-serif;
    background: #FFFFFF;
    border-radius: 10px; overflow: hidden;
    box-shadow: 0 2px 10px rgba(15,34,54,0.07);
}
.rtable thead tr { background: #0C1F3A !important; }
.rtable thead th {
    padding: 12px 16px; text-align: left;
    color: #F1F5F9 !important;
    font-size: 11px !important; font-weight: 700 !important;
    letter-spacing: 0.1em; text-transform: uppercase;
}
.rtable tbody td {
    padding: 11px 16px;
    border-bottom: 1px solid #EFF2F6;
    color: #1F2937 !important;
    font-size: 14px !important;
}
.rtable tbody tr:last-child td {
    background: #F0FDF4 !important;
    font-weight: 700 !important; color: #064E3B !important;
    border-bottom: none;
}
.rtable tbody tr:hover td { background: #F8FAFC !important; }
.rtable .mono { font-family: 'Source Code Pro', monospace !important; font-size: 13px !important; }

.delta-pos { color: #0D7A55 !important; font-size: 12px !important; font-weight: 600 !important; }
.delta-base { color: #9CA3AF !important; font-size: 12px !important; }

.rtable tbody tr.ours td {
    background: #F0FDF4 !important;
    font-weight: 700 !important; color: #064E3B !important;
}

.card {
    background: #FFFFFF; border: 1.5px solid #DDE3EC;
    border-radius: 10px; padding: 22px 20px;
    box-shadow: 0 2px 8px rgba(15,34,54,0.06);
}
.card-title { font-size: 14px !important; font-weight: 700 !important; color: #0C1F3A !important; margin-bottom: 8px; }
.card-body  { font-size: 13.5px !important; color: #374151 !important; line-height: 1.65; }

.feat-row { display: flex; align-items: center; gap: 10px; margin: 6px 0; }
.feat-name { font-family: 'Source Code Pro', monospace; font-size: 12px; color: #1F2937 !important; width: 260px; flex-shrink: 0; }
.bar-track  { background: #E5E7EB; border-radius: 3px; height: 7px; flex: 1; }
.bar-fill   { height: 7px; border-radius: 3px; }

.divider { border: none; border-top: 1.5px solid #DDE3EC; margin: 24px 0; }

[data-testid="stMetric"] {
    background: #FFFFFF !important; border: 1.5px solid #DDE3EC !important;
    border-radius: 10px !important; padding: 14px !important;
}
[data-testid="stMetricLabel"] > div,
[data-testid="stMetricLabel"] p {
    color: #6B7280 !important; font-size: 11px !important;
    font-weight: 700 !important; text-transform: uppercase !important; letter-spacing: 0.09em !important;
}
[data-testid="stMetricValue"] > div {
    color: #0C1F3A !important;
    font-family: 'Source Code Pro', monospace !important;
    font-size: 22px !important; font-weight: 600 !important;
}

/* ── Executive Insight ── */
.exec-insight {
    background: linear-gradient(135deg, #F0FDF4 0%, #ECFDF5 100%);
    border: 1px solid #6EE7B7;
    border-left: 5px solid #0D7A55;
    border-radius: 0 10px 10px 0;
    padding: 18px 22px;
    margin-bottom: 4px;
}
.exec-insight .ei-label {
    font-size: 10px !important; font-weight: 800 !important;
    letter-spacing: 0.18em; text-transform: uppercase;
    color: #065F46 !important; display: block; margin-bottom: 6px;
}
.exec-insight .ei-body { font-size: 14px !important; color: #14532D !important; line-height: 1.7; }

/* ── Disclaimer ── */
.disclaimer {
    background: #FAFAFA; border: 1px solid #E5E7EB; border-radius: 8px;
    padding: 9px 15px; font-size: 12px !important; color: #6B7280 !important;
    display: flex; align-items: center; gap: 8px; margin-bottom: 6px;
}

/* ── Final Action Plan ── */
.action-plan {
    background: linear-gradient(135deg, #0C1F3A 0%, #1E3A5A 100%);
    border-radius: 12px;
    padding: 22px 26px;
    margin: 16px 0 6px 0;
}
.action-plan .ap-label {
    font-size: 10px !important; font-weight: 800 !important;
    letter-spacing: 0.20em; text-transform: uppercase;
    color: #5EEAD4 !important; display: block; margin-bottom: 14px;
}
.action-plan .ap-grid {
    display: grid; grid-template-columns: repeat(4, 1fr); gap: 14px;
}
.action-plan .ap-item {
    background: rgba(255,255,255,0.07);
    border: 1px solid rgba(255,255,255,0.13);
    border-radius: 8px; padding: 13px 14px;
}
.action-plan .ap-icon { font-size: 18px; display: block; margin-bottom: 5px; }
.action-plan .ap-val {
    font-family: 'Source Code Pro', monospace;
    font-size: 16px !important; font-weight: 700 !important;
    color: #FFFFFF !important; display: block; margin-bottom: 3px;
}
.action-plan .ap-desc {
    font-size: 11.5px !important; color: #94A3B8 !important; line-height: 1.4;
}
.action-plan .ap-footer {
    margin-top: 14px; padding-top: 12px;
    border-top: 1px solid rgba(255,255,255,0.10);
    font-size: 13px !important; color: #CBD5E1 !important;
    font-style: italic;
}

/* ── Accuracy Warning ── */
.acc-warning {
    background: #FFF7ED;
    border: 1px solid #FED7AA;
    border-left: 5px solid #EA580C;
    border-radius: 0 10px 10px 0;
    padding: 16px 22px;
    margin: 12px 0;
}
.acc-warning .aw-label {
    font-size: 10px !important; font-weight: 800 !important;
    letter-spacing: 0.18em; text-transform: uppercase;
    color: #9A3412 !important; display: block; margin-bottom: 8px;
}
.acc-warning .aw-body { font-size: 13.5px !important; color: #7C2D12 !important; line-height: 1.7; }
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown('<div class="page-title">📊 System Overview</div>', unsafe_allow_html=True)
st.markdown('<div class="page-sub">Model validation · Ablation evidence · Comparative benchmarks · Feature analysis</div>', unsafe_allow_html=True)
st.markdown('<hr class="divider">', unsafe_allow_html=True)

# ── Why This Matters ─────────────────────────────────────────────────────────
st.success("🚀 **What does this system do?** Identifies high-risk, high-value customers to target — delivering **~65% higher profit** vs. targeting at random. Optimised for business decisions, not model accuracy.")
st.caption("Trained on ~7,000 customers · Evaluated on 1,409 unseen test customers · Profit-optimised threshold · 5-fold CV validated.")

# ── Disclaimer ────────────────────────────────────────────────────────────────
st.markdown("""
<div class="disclaimer">
    <span style="font-size:16px">ℹ️</span>
    <span>Profit values are simulated using customer lifetime assumptions — used as a relative decision metric, not real currency.</span>
</div>
""", unsafe_allow_html=True)

# ── Recommended Strategy ──────────────────────────────────────────────────────
st.markdown("""
<div class="action-plan">
    <span class="ap-label">🚀 Recommended Strategy</span>
    <div class="ap-grid">
        <div class="ap-item">
            <span class="ap-icon">🎯</span>
            <span class="ap-val">50–60%</span>
            <span class="ap-desc">Optimal share of customers to target</span>
        </div>
        <div class="ap-item">
            <span class="ap-icon">⚡</span>
            <span class="ap-val">&gt; 0.7</span>
            <span class="ap-desc">Focus on customers above this risk score</span>
        </div>
        <div class="ap-item">
            <span class="ap-icon">💰</span>
            <span class="ap-val">~450K units</span>
            <span class="ap-desc">Expected estimated value at optimal range</span>
        </div>
        <div class="ap-item">
            <span class="ap-icon">🚫</span>
            <span class="ap-val">&gt; 70%</span>
            <span class="ap-desc">Beyond this, ROI drops — stop here</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

# ── ROC Curve ─────────────────────────────────────────────────────────────────
st.markdown('<span class="sec-head">ROC Curve &amp; AUC</span>', unsafe_allow_html=True)

col_roc, col_roc_info = st.columns([3, 2])
with col_roc:
    roc_path = "results/roc_curve.png"
    if os.path.exists(roc_path):
        st.image(roc_path, use_container_width=True)
    else:
        st.info("📂 Place `results/roc_curve.png` here to display the ROC curve.")

with col_roc_info:
    st.markdown("""
    <div class="card">
        <div class="card-title">AUC = 0.8393</div>
        <div class="card-body">
            Strongly outperforms random chance (AUC = 0.50) — the model reliably ranks churners above non-churners before applying the decision threshold.<br><br>
            <strong>5-Fold CV Recall:</strong> 0.840 · 0.829 · 0.824 · 0.791 · 0.797<br>
            <strong>Mean: 81.6%</strong> — stable across all folds.
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)
    st.markdown("""
    <div class="card">
        <div class="card-title">Stability — 10 Seeds</div>
        <div class="card-body">
            Mean Profit: <strong>377,548 units</strong> · Std: <strong>7,524</strong><br>
            CV (σ/μ): <strong>0.0199</strong> — consistent across all splits.
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown('<hr class="divider">', unsafe_allow_html=True)

# ── Ablation ─────────────────────────────────────────────────────────────────
st.caption("Each component below is tested in isolation — confirming both GEO and threshold tuning contribute independently to profit.")
st.markdown('<span class="sec-head">Ablation Study — Profit Contribution of Each Component</span>', unsafe_allow_html=True)

ablation = [
    ("① Baseline (all features, t=0.5)",         "270,001 units",  "—",              "baseline"),
    ("② Threshold Optimisation Only",             "377,955 units",  "+107,954 units", "pos"),
    ("③ GEO Feature Selection Only (t=0.5)",      "345,572 units",  "+75,571 units",  "pos"),
    ("④ Full System — GEO + Optimised Threshold", "446,563 units",  "+176,562 units", "full"),
]

rows = ""
for i, (s, p, d, cls) in enumerate(ablation):
    if cls == "full":
        rows += f"""<tr>
            <td style="background:#F0FDF4;font-weight:700;color:#064E3B">{s}</td>
            <td class="mono" style="background:#F0FDF4;font-weight:700;color:#064E3B">{p}</td>
            <td class="delta-pos" style="background:#F0FDF4">{d}</td>
        </tr>"""
    elif cls == "baseline":
        rows += f"""<tr>
            <td>{s}</td>
            <td class="mono">{p}</td>
            <td class="delta-base">{d}</td>
        </tr>"""
    else:
        rows += f"""<tr>
            <td>{s}</td>
            <td class="mono">{p}</td>
            <td class="delta-pos">{d}</td>
        </tr>"""

st.markdown(f"""
<div style="overflow:hidden;border-radius:10px;border:1.5px solid #DDE3EC;box-shadow:0 2px 10px rgba(15,34,54,0.07)">
<table class="rtable">
<thead><tr><th>Setup</th><th>Estimated Value</th><th>vs Baseline</th></tr></thead>
<tbody>{rows}</tbody>
</table>
</div>
""", unsafe_allow_html=True)

st.markdown('<hr class="divider">', unsafe_allow_html=True)

# ── Model Comparison ──────────────────────────────────────────────────────────
st.caption("Comparing against standard ML models — higher accuracy does not mean higher business value.")
st.markdown('<span class="sec-head">Model Comparison Benchmark</span>', unsafe_allow_html=True)

models = [
    ("LR + GEO (Proposed System) ✦", "0.7317", "82.1%", "446,563 units", True),
    ("Random Forest",                  "0.7878", "49.7%", "164,483 units", False),
    ("XGBoost",                        "0.7850", "53.5%", "183,578 units", False),
    ("SVM",                            "0.7928", "48.9%", "137,574 units", False),
]

model_rows = ""
for name, acc, rec, profit, ours in models:
    if ours:
        model_rows += f"""<tr class="ours">
            <td style="background:#F0FDF4;font-weight:700;color:#064E3B">{name}</td>
            <td class="mono" style="background:#F0FDF4;color:#064E3B">{acc}</td>
            <td class="mono" style="background:#F0FDF4;color:#064E3B">{rec}</td>
            <td class="mono" style="background:#F0FDF4;font-weight:700;color:#064E3B">{profit}</td>
        </tr>"""
    else:
        model_rows += f"""<tr>
            <td>{name}</td>
            <td class="mono">{acc}</td>
            <td class="mono">{rec}</td>
            <td class="mono">{profit}</td>
        </tr>"""

st.markdown(f"""
<div style="overflow:hidden;border-radius:10px;border:1.5px solid #DDE3EC;box-shadow:0 2px 10px rgba(15,34,54,0.07)">
<table class="rtable">
<thead><tr><th>Model</th><th>Accuracy</th><th>Recall</th><th>Estimated Value (t=0.1)</th></tr></thead>
<tbody>{model_rows}</tbody>
</table>
</div>
""", unsafe_allow_html=True)

# ── Accuracy insight (tight) ──────────────────────────────────────────────────
st.markdown("""
<div class="acc-warning">
    <span class="aw-label">⚠️ Key Insight — Accuracy Is Misleading Here</span>
    <div class="aw-body">
        RF and SVM score higher on accuracy (78–79%) yet generate <strong>3× less profit</strong> than our system (73% accuracy, 446K units).
        For churn intervention, <strong>recall and estimated value are the right metrics</strong> — not accuracy.
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown('<hr class="divider">', unsafe_allow_html=True)

# ── Feature Analysis ──────────────────────────────────────────────────────────
st.caption("GEO selected 17 of 30 features — validated against SHAP importance to confirm alignment with model interpretability.")
st.markdown('<span class="sec-head">GEO Feature Selection — 17 / 30 Features</span>', unsafe_allow_html=True)

selected_feats = [
    ('tenure', True),
    ('MonthlyCharges', True),
    ('TotalCharges', True),
    ('Contract_Two year', True),
    ('Contract_One year', True),
    ('OnlineSecurity_Yes', True),
    ('PaymentMethod_Electronic check', True),
    ('SeniorCitizen', True),
    ('MultipleLines_Yes', True),
    ('MultipleLines_No phone service', True),
    ('Partner_Yes', False),
    ('InternetService_No', False),
    ('OnlineSecurity_No internet service', False),
    ('DeviceProtection_No internet service', False),
    ('StreamingTV_No internet service', False),
    ('StreamingMovies_Yes', False),
    ('PaymentMethod_Credit card (automatic)', False),
]

col_feat, col_legend = st.columns([3, 1])
with col_feat:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">All 17 GEO-Selected Features</div>', unsafe_allow_html=True)
    rows_feat = ""
    for feat, in_shap in selected_feats:
        color = "#0D7A55" if in_shap else "#93C5FD"
        icon  = "🟢" if in_shap else "🔵"
        rows_feat += f"""
        <div class="feat-row">
            <span style="width:18px;flex-shrink:0">{icon}</span>
            <span class="feat-name">{feat}</span>
            <div class="bar-track">
                <div class="bar-fill" style="width:{'85%' if in_shap else '45%'};background:{color}"></div>
            </div>
        </div>"""
    st.markdown(rows_feat, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col_legend:
    st.markdown("""
    <div class="card">
        <div class="card-title">SHAP Overlap</div>
        <div class="card-body">
            <strong>10 / 10</strong> top SHAP features match GEO selections — 100% alignment without requiring model introspection.<br><br>
            🟢 <strong>In top-10 SHAP</strong> — statistically important<br>
            🔵 <strong>GEO only</strong> — recall maximisation
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

# ── Limitations ───────────────────────────────────────────────────────────────
st.markdown('<hr class="divider">', unsafe_allow_html=True)
st.markdown('<span class="sec-head">System Limitations</span>', unsafe_allow_html=True)
st.info("""
⚠️ **Before real-world deployment, consider:**

- **Simulated profit** — values use CLV assumptions, not real revenue; recalibrate with actual campaign cost data
- **Threshold sensitivity** — the 0.10 decision threshold was optimised for this dataset; adjust for different churn base rates
- **No causal guarantee** — high predicted risk does not mean intervention will succeed; some customers may churn regardless
""")
st.caption("Trained on ~7,000 customers · Evaluated on 1,409 unseen test customers · 10-seed stability · 5-fold CV.")