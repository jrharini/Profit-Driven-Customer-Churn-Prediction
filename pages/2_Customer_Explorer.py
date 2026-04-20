import streamlit as st
import pandas as pd
import numpy as np
import os

st.set_page_config(page_title="Customer Explorer · Churn System", page_icon="🎯", layout="wide")

# ── AGGRESSIVE LIGHT THEME OVERRIDE ──────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Merriweather:wght@700;900&family=Source+Sans+3:wght@300;400;500;600;700&family=Source+Code+Pro:wght@400;500;600&display=swap');

/* ════ NUCLEAR LIGHT MODE ════ */
:root { color-scheme: light only !important; }
*, *::before, *::after { color-scheme: light !important; }

html, body,
html[data-theme="dark"], html[data-theme="light"],
.stApp, .stApp > div,
[data-testid="stAppViewContainer"],
[data-testid="stAppViewContainer"] > div,
[data-testid="stMain"], [data-testid="stMain"] > div,
.main, .main > div, .block-container,
[data-testid="stVerticalBlock"], [data-testid="stVerticalBlock"] > div,
[data-testid="stHorizontalBlock"],
div.stMarkdown, [data-testid="stMarkdownContainer"],
.element-container, div[class*="stColumn"] {
    background-color: #F7F8FA !important;
    color: #111827 !important;
}

p, span, div, label, h1, h2, h3, h4, h5, h6, li, td, th,
.stMarkdown p, .stMarkdown span, .stMarkdown div,
[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] span {
    color: #111827 !important;
    font-family: 'Source Sans 3', sans-serif !important;
}

/* ════ SIDEBAR ════ */
section[data-testid="stSidebar"],
section[data-testid="stSidebar"] > div,
section[data-testid="stSidebar"] > div > div,
[data-testid="stSidebarContent"] {
    background-color: #0F2236 !important;
}
section[data-testid="stSidebar"] *,
[data-testid="stSidebarContent"] * { color: #CBD5E1 !important; background-color: transparent !important; }
[data-testid="stSidebarNavLink"] { background: transparent !important; border-radius: 8px !important; padding: 8px 12px !important; }
[data-testid="stSidebarNavLink"]:hover { background: rgba(255,255,255,0.08) !important; }
[data-testid="stSidebarNavLink"][aria-selected="true"] { background: rgba(255,255,255,0.13) !important; border-left: 3px solid #5EEAD4 !important; }
[data-testid="stSidebarNavLink"] * { color: #CBD5E1 !important; }
[data-testid="stSidebarNavLink"][aria-selected="true"] * { color: #FFFFFF !important; font-weight: 700 !important; }

/* ════ CONTROLS ════ */
[data-testid="stSlider"] > div > div > div { background: #1E3A5A !important; }
div[data-baseweb="select"] > div { background-color: #FFFFFF !important; border-color: #DDE3EC !important; color: #111827 !important; }
div[data-baseweb="popover"] > div, div[data-baseweb="menu"] { background-color: #FFFFFF !important; }
div[data-baseweb="menu"] li, div[data-baseweb="menu"] li span { color: #111827 !important; }
[data-testid="stCheckbox"] label span { color: #111827 !important; }
input, textarea { background-color: #FFFFFF !important; color: #111827 !important; }

/* ════ DOWNLOAD BUTTON ════ */
[data-testid="stDownloadButton"] > button {
    background: #0C1F3A !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 700 !important;
    font-size: 13px !important;
    padding: 10px 20px !important;
    width: 100%;
}
[data-testid="stDownloadButton"] > button:hover {
    background: #1E3A5A !important;
}

/* ════ SCROLLBAR ════ */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: #F1F5F9; }
::-webkit-scrollbar-thumb { background: #CBD5E1; border-radius: 3px; }

/* ════ TYPOGRAPHY ════ */
.page-title { font-family: 'Merriweather', serif !important; font-size: 30px !important; font-weight: 900 !important; color: #0C1F3A !important; letter-spacing: -0.02em; }
.page-sub { font-size: 15px !important; color: #4B5563 !important; margin-top: 5px; }
.sec-head { font-family: 'Merriweather', serif !important; font-size: 12px !important; font-weight: 700 !important; letter-spacing: 0.12em; text-transform: uppercase; color: #1E3A5A !important; border-bottom: 2.5px solid #1E3A5A; padding-bottom: 7px; margin: 28px 0 16px 0; display: block; }
.divider { border: none; border-top: 1.5px solid #DDE3EC; margin: 22px 0; }

/* ════ CHIPS ════ */
.chip-row { display: flex; gap: 10px; flex-wrap: wrap; margin: 14px 0 20px 0; }
.chip { background: #FFFFFF !important; border: 1.5px solid #DDE3EC; border-radius: 8px; padding: 10px 18px; font-size: 13px !important; color: #374151 !important; box-shadow: 0 1px 4px rgba(15,34,54,0.06); }
.chip strong { color: #0C1F3A !important; font-family: 'Source Code Pro', monospace; font-size: 15px; }

/* ════ TABLE ════ */
.ctable-wrap { background: #FFFFFF !important; border: 1.5px solid #DDE3EC; border-radius: 12px; overflow: hidden; box-shadow: 0 3px 14px rgba(15,34,54,0.09); }
.ctable { width: 100%; border-collapse: collapse; font-family: 'Source Sans 3', sans-serif; font-size: 13.5px; background: #FFFFFF !important; }
.ctable thead { background: #0C1F3A !important; }
.ctable thead th { padding: 13px 16px; text-align: left; color: #F1F5F9 !important; font-size: 11px !important; font-weight: 700 !important; letter-spacing: 0.12em; text-transform: uppercase; background: #0C1F3A !important; }
.ctable tbody tr { border-bottom: 1px solid #EEF1F5; }
.ctable tbody tr:hover td { background: #F0F9FF !important; }
.ctable tbody td { padding: 10px 16px; color: #1F2937 !important; font-size: 13.5px !important; }
.ctable tbody tr.intervene-row td { background: #F9FFFE !important; }
.ctable tbody tr.no-action-row td { background: #FAFAFA !important; color: #6B7280 !important; }

.badge-int { display: inline-block; background: #D1FAE5 !important; color: #065F46 !important; border: 1px solid #6EE7B7; padding: 3px 11px; border-radius: 20px; font-size: 11px !important; font-weight: 700 !important; }
.badge-no { display: inline-block; background: #F3F4F6 !important; color: #6B7280 !important; border: 1px solid #D1D5DB; padding: 3px 11px; border-radius: 20px; font-size: 11px !important; font-weight: 600 !important; }
.reason-high-churn { display: inline-block; background: #FEF2F2 !important; color: #991B1B !important; border: 1px solid #FECACA; padding: 3px 10px; border-radius: 20px; font-size: 11px !important; font-weight: 600 !important; }
.reason-high-val { display: inline-block; background: #EFF6FF !important; color: #1E40AF !important; border: 1px solid #BFDBFE; padding: 3px 10px; border-radius: 20px; font-size: 11px !important; font-weight: 600 !important; }
.reason-moderate { display: inline-block; background: #FFFBEB !important; color: #92400E !important; border: 1px solid #FDE68A; padding: 3px 10px; border-radius: 20px; font-size: 11px !important; font-weight: 600 !important; }
.reason-loyal { display: inline-block; background: #F5F3FF !important; color: #5B21B6 !important; border: 1px solid #DDD6FE; padding: 3px 10px; border-radius: 20px; font-size: 11px !important; font-weight: 600 !important; }

.pb { display: flex; align-items: center; gap: 9px; }
.pb-val { font-family: 'Source Code Pro', monospace; font-size: 13px; width: 46px; color: #0C1F3A !important; }
.pb-track { background: #E5E7EB !important; border-radius: 3px; height: 7px; width: 88px; flex-shrink: 0; }
.pb-fill { height: 7px; border-radius: 3px; }
.profit-val { font-family: 'Source Code Pro', monospace !important; color: #0D7A55 !important; font-weight: 600 !important; }
.profit-val.dim { color: #9CA3AF !important; font-weight: 400 !important; }
.cid { font-family: 'Source Code Pro', monospace; color: #374151 !important; font-size: 13px; }
.footer-note { font-size: 12px !important; color: #9CA3AF !important; padding: 10px 16px 14px 16px; text-align: center; border-top: 1px solid #EEF1F5; background: #FAFAFA !important; }

/* ════ MISC ════ */
.disclaimer { background: #FAFAFA !important; border: 1px solid #E5E7EB; border-radius: 8px; padding: 9px 15px; font-size: 12px !important; color: #6B7280 !important; display: flex; align-items: center; gap: 8px; margin-bottom: 6px; }
.legend-box { background: #FFFFFF !important; border: 1.5px solid #DDE3EC; border-radius: 8px; padding: 12px 16px; font-size: 12px !important; color: #374151 !important; margin-top: 8px; }
.page-controls { display: flex; align-items: center; justify-content: space-between; background: #FFFFFF !important; border: 1.5px solid #DDE3EC; border-radius: 10px; padding: 12px 18px; margin: 12px 0; flex-wrap: wrap; gap: 8px; }
.page-info { font-size: 13px !important; color: #374151 !important; }
.page-info strong { color: #0C1F3A !important; font-family: 'Source Code Pro', monospace; }

/* ════ EXPORT STRIP ════ */
.export-strip {
    background: linear-gradient(135deg, #F0FDF4 0%, #ECFDF5 100%);
    border: 1px solid #6EE7B7;
    border-radius: 10px;
    padding: 16px 20px;
    display: flex; align-items: center; justify-content: space-between;
    flex-wrap: wrap; gap: 12px;
    margin: 16px 0;
}
.export-strip .es-text { font-size: 13.5px !important; color: #14532D !important; }
.export-strip .es-text strong { color: #065F46 !important; }
</style>
""", unsafe_allow_html=True)


# ── Helper functions ──────────────────────────────────────────────────────────
def explain(row):
    """FIX 2: Combined multi-reason explainer — captures all relevant signals."""
    reasons = []
    if row["Churn_Probability"] > 0.8:
        reasons.append("High churn risk")
    if row["Expected_Profit"] > 600:
        reasons.append("High value")
    # tenure check only if column exists
    if "tenure" in row.index and row["tenure"] > 12:
        reasons.append("Loyal customer")
    if not reasons:
        reasons.append("Moderate candidate")
    return ", ".join(reasons)

def primary_reason(row):
    """Returns the single most important reason for badge display."""
    if row["Churn_Probability"] > 0.8:
        return "High churn risk"
    elif row["Expected_Profit"] > 600:
        return "High value customer"
    elif "tenure" in row.index and row["tenure"] > 12:
        return "Loyal customer"
    else:
        return "Moderate candidate"

def reason_badge(reason):
    if "High churn" in reason:
        return f'<span class="reason-high-churn">🔴 {reason}</span>'
    elif "High value" in reason:
        return f'<span class="reason-high-val">🔵 {reason}</span>'
    elif "Loyal" in reason:
        return f'<span class="reason-loyal">🟣 {reason}</span>'
    else:
        return f'<span class="reason-moderate">🟡 {reason}</span>'

def multi_reason_badges(reason_str):
    """Render all reasons as stacked badges."""
    parts = [r.strip() for r in reason_str.split(",")]
    badges = []
    for p in parts:
        if "High churn" in p:
            badges.append(f'<span class="reason-high-churn">🔴 {p}</span>')
        elif "High value" in p:
            badges.append(f'<span class="reason-high-val">🔵 {p}</span>')
        elif "Loyal" in p:
            badges.append(f'<span class="reason-loyal">🟣 {p}</span>')
        else:
            badges.append(f'<span class="reason-moderate">🟡 {p}</span>')
    return " ".join(badges)

def prob_color(p):
    if p >= 0.7: return "#EF4444"
    if p >= 0.4: return "#F59E0B"
    return "#10B981"


# ── Load data — supports any CSV size ────────────────────────────────────────
@st.cache_data
def load_decisions():
    for path in ["results/customer_decisions.csv", "customer_decisions.csv"]:
        if os.path.exists(path):
            df = pd.read_csv(path)
            df.columns = df.columns.str.strip()
            return df

    # Fallback: full 7,043-row synthetic telco dataset
    np.random.seed(42)
    n = 7043
    high_risk = np.random.beta(5, 2, int(n * 0.265))
    low_risk  = np.random.beta(1, 6, n - int(n * 0.265))
    probs = np.concatenate([high_risk, low_risk])
    np.random.shuffle(probs)
    probs = np.clip(probs, 0.001, 0.999)
    clv = np.random.normal(780, 180, n).clip(200, 1800)
    ep  = np.round(probs * 0.5 * clv - 13, 3)
    actual = np.array([1 if np.random.random() < p else 0 for p in probs])
    tenure = np.random.randint(1, 73, n)
    return pd.DataFrame({
        "Customer_ID":        range(1, n + 1),
        "Churn_Probability":  np.round(probs, 4),
        "Expected_Profit":    ep,
        "Decision_Threshold": ["INTERVENE" if p >= 0.1 else "NO ACTION" for p in probs],
        "Actual_Churn":       actual,
        "tenure":             tenure,
    })

df = load_decisions()
TOTAL_CUSTOMERS = len(df)

# Pre-compute for status strip (uses full unfiltered df)
_n_int_all  = int((df["Decision_Threshold"] == "INTERVENE").sum())
_ep_all     = df.loc[df["Decision_Threshold"] == "INTERVENE", "Expected_Profit"].sum()

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown('<div class="page-title">🎯 Customer Explorer</div>', unsafe_allow_html=True)
st.markdown(f'<div class="page-sub">Filter · Sort · Export — <strong>{TOTAL_CUSTOMERS:,} customers</strong> loaded</div>', unsafe_allow_html=True)
st.markdown('<hr class="divider">', unsafe_allow_html=True)

# ── Issue 4: System Status Strip ─────────────────────────────────────────────
st.success(f"""
📊 **System Recommendation:** Target **{_n_int_all:,} customers ({_n_int_all/TOTAL_CUSTOMERS*100:.1f}%)**
→ Estimated Value: **{_ep_all:,.0f} units** · Focus on high-risk, high-value segment for optimal ROI
""")

st.markdown("""<div class="disclaimer">
    <span style="font-size:16px">ℹ️</span>
    <span>Estimated value is simulated using CLV assumptions — used as a relative decision metric, not real currency.</span>
</div>""", unsafe_allow_html=True)


# ── Controls ──────────────────────────────────────────────────────────────────
st.markdown('<span class="sec-head">Filters &amp; Controls</span>', unsafe_allow_html=True)

cc1, cc2, cc3, cc4 = st.columns([3, 1.2, 1.4, 1.2])
with cc1:
    prob_range = st.slider("Risk Score Range", 0.0, 1.0, (0.0, 1.0), step=0.01)
with cc2:
    intervene_only = st.checkbox("INTERVENE only", value=False)
with cc3:
    sort_by = st.selectbox("Sort by", ["Estimated Value ↓", "Risk Score ↓", "Customer ID ↑"])
with cc4:
    rows_per_page = st.selectbox("Rows/page", [25, 50, 100, 200, 500], index=2)


# ── Filter & sort ─────────────────────────────────────────────────────────────
filtered = df[
    (df["Churn_Probability"] >= prob_range[0]) &
    (df["Churn_Probability"] <= prob_range[1])
].copy()
if intervene_only:
    filtered = filtered[filtered["Decision_Threshold"] == "INTERVENE"]
if sort_by == "Estimated Value ↓":
    filtered = filtered.sort_values("Expected_Profit", ascending=False)
elif sort_by == "Risk Score ↓":
    filtered = filtered.sort_values("Churn_Probability", ascending=False)
else:
    filtered = filtered.sort_values("Customer_ID", ascending=True)

# FIX 2: Apply combined multi-signal reasoning
filtered["Reason"] = filtered.apply(explain, axis=1)
filtered = filtered.reset_index(drop=True)


# ── Chips ─────────────────────────────────────────────────────────────────────
n_int  = int((filtered["Decision_Threshold"] == "INTERVENE").sum())
ep_sum = filtered.loc[filtered["Decision_Threshold"] == "INTERVENE", "Expected_Profit"].sum()
n_churn = int(filtered["Actual_Churn"].sum()) if "Actual_Churn" in filtered.columns else 0

st.markdown(f"""<div class="chip-row">
    <div class="chip">Filtered: <strong>{len(filtered):,}</strong> / {TOTAL_CUSTOMERS:,}</div>
    <div class="chip">Intervene: <strong>{n_int:,}</strong> ({n_int/max(len(filtered),1)*100:.1f}%)</div>
    <div class="chip">Est. Value: <strong>{ep_sum:,.0f} units</strong></div>
    <div class="chip">Actual Churners: <strong>{n_churn:,}</strong></div>
</div>""", unsafe_allow_html=True)

st.markdown("""<div class="legend-box">
    <strong style="color:#0C1F3A;display:block;margin-bottom:5px;font-size:12px">Why This Customer? — Reason Column</strong>
    🔴 <strong>High churn risk</strong> — prob &gt; 80% &nbsp;·&nbsp;
    🔵 <strong>High value customer</strong> — profit &gt; 600 units &nbsp;·&nbsp;
    🟣 <strong>Loyal customer</strong> — tenure &gt; 12 months &nbsp;·&nbsp;
    🟡 <strong>Moderate candidate</strong> — standard risk profile<br>
    <span style="color:#6B7280;font-size:11px;margin-top:4px;display:block">Multiple badges may appear for a single customer — each signal is independently evaluated.</span>
</div>""", unsafe_allow_html=True)


# ── FIX 5: Export Button ──────────────────────────────────────────────────────
st.markdown('<span class="sec-head">Export Intervention List</span>', unsafe_allow_html=True)

intervene_df = filtered[filtered["Decision_Threshold"] == "INTERVENE"].copy()
export_cols = [c for c in ["Customer_ID", "Churn_Probability", "Expected_Profit", "Decision_Threshold", "Actual_Churn", "Reason"] if c in intervene_df.columns]
export_df = intervene_df[export_cols]

ex1, ex2 = st.columns([3, 1])
with ex1:
    st.markdown(f"""<div style="background:#F0FDF4;border:1px solid #BBF7D0;border-radius:8px;padding:12px 16px;font-size:13.5px;color:#14532D">
        <strong>{len(export_df):,} customers</strong> flagged for intervention in current view — 
        expected combined profit: <strong>{export_df['Expected_Profit'].sum():,.0f} units</strong>. 
        Download the full list to action with your retention team.
    </div>""", unsafe_allow_html=True)
with ex2:
    csv = export_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Download Target Customers",
        data=csv,
        file_name="target_customers.csv",
        mime="text/csv",
    )

st.markdown('<hr class="divider">', unsafe_allow_html=True)


# ── Pagination ────────────────────────────────────────────────────────────────
st.markdown('<span class="sec-head">Customer Decision Table</span>', unsafe_allow_html=True)

# Fix 3: Action-focused guidance above table
st.warning("""
🎯 **Focus on TOP rows** — these customers deliver the highest profit impact per intervention.
Targeting the top 50–60% of the list yields maximum ROI. Rows highlighted in green = INTERVENE. Rows with 🔴 or 🔵 badges are your highest-priority targets.
""")

total_rows  = len(filtered)
total_pages = max(1, (total_rows + rows_per_page - 1) // rows_per_page)

tc1, tc2, tc3 = st.columns([2, 1, 2])
with tc2:
    page_num = st.number_input(f"Page (1–{total_pages:,})", min_value=1, max_value=total_pages, value=1, step=1)

start_idx  = (page_num - 1) * rows_per_page
end_idx    = min(start_idx + rows_per_page, total_rows)
display_df = filtered.iloc[start_idx:end_idx]

st.markdown(f"""<div class="page-controls">
    <div class="page-info">Rows <strong>{start_idx+1:,}–{end_idx:,}</strong> of <strong>{total_rows:,}</strong></div>
    <div class="page-info">Page <strong>{page_num}</strong> / <strong>{total_pages}</strong></div>
    <div class="page-info">Total: <strong>{TOTAL_CUSTOMERS:,}</strong> customers</div>
</div>""", unsafe_allow_html=True)


# ── Build table HTML ──────────────────────────────────────────────────────────
rows_html = ""
for _, row in display_df.iterrows():
    cid    = int(row["Customer_ID"])
    prob   = float(row["Churn_Probability"])
    ep     = float(row["Expected_Profit"])
    dec    = str(row["Decision_Threshold"])
    churn  = int(row["Actual_Churn"]) if "Actual_Churn" in row else -1
    reason = str(row["Reason"])
    bar_w     = int(prob * 100)
    bar_c     = prob_color(prob)
    badge     = '<span class="badge-int">🟢 INTERVENE</span>' if dec == "INTERVENE" else '<span class="badge-no">⚪ NO ACTION</span>'
    ep_cls    = "profit-val" if dec == "INTERVENE" else "profit-val dim"
    churn_txt = ("✓ Yes" if churn == 1 else "✗ No") if churn >= 0 else "—"
    churn_col = "#EF4444" if churn == 1 else "#6B7280"
    row_cls   = "intervene-row" if dec == "INTERVENE" else "no-action-row"
    rows_html += f"""<tr class="{row_cls}">
        <td><span class="cid">#{cid:05d}</span></td>
        <td><div class="pb"><span class="pb-val">{prob:.3f}</span><div class="pb-track"><div class="pb-fill" style="width:{bar_w}%;background:{bar_c}"></div></div></div></td>
        <td><span class="{ep_cls}">{ep:,.2f} units</span></td>
        <td>{badge}</td>
        <td style="color:{churn_col} !important;font-weight:{'600' if churn==1 else '400'}">{churn_txt}</td>
        <td>{multi_reason_badges(reason)}</td>
    </tr>"""

st.markdown(f"""<div class="ctable-wrap">
    <table class="ctable">
    <thead><tr>
        <th>Customer ID</th><th>Risk Score</th><th>Estimated Value</th>
        <th>Decision</th><th>Actual Churn</th><th>Why This Customer</th>
    </tr></thead>
    <tbody>{rows_html}</tbody>
    </table>
    <div class="footer-note">
        Decision threshold = 0.10 &nbsp;·&nbsp; Intervention cost ≈ 13 units/customer &nbsp;·&nbsp;
        Page {page_num}/{total_pages} &nbsp;·&nbsp; {TOTAL_CUSTOMERS:,} total customers
    </div>
</div>""", unsafe_allow_html=True)

st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
st.markdown('<hr class="divider">', unsafe_allow_html=True)

# ── Decision Scatter Chart ────────────────────────────────────────────────────
st.markdown('<span class="sec-head">Decision Visualisation — Risk Score vs Estimated Value</span>', unsafe_allow_html=True)
st.markdown("""<div style="background:#EFF6FF;border:1px solid #BFDBFE;border-left:4px solid #1D4ED8;border-radius:0 8px 8px 0;padding:12px 18px;font-size:13px;color:#1E3A5F;margin-bottom:14px">
    <strong>How to read this:</strong> Each dot is one customer.
    🟢 Green = INTERVENE · ⚫ Grey = NO ACTION · 🔴 Red = High Priority (risk &gt; 0.8 or value &gt; 600).
    The <strong>top-right cluster</strong> is your highest-value target zone.
</div>""", unsafe_allow_html=True)

try:
    import plotly.graph_objects as go

    intervene_pts = df[df["Decision_Threshold"] == "INTERVENE"].copy()
    no_action_pts = df[df["Decision_Threshold"] != "INTERVENE"].copy()
    high_priority = intervene_pts[
        (intervene_pts["Churn_Probability"] > 0.8) | (intervene_pts["Expected_Profit"] > 600)
    ]

    fig_scatter = go.Figure()
    fig_scatter.add_trace(go.Scatter(
        x=no_action_pts["Churn_Probability"], y=no_action_pts["Expected_Profit"],
        mode="markers", marker=dict(color="#CBD5E1", size=4, opacity=0.5),
        name="⚪ No Action",
        hovertemplate="Customer #%{customdata}<br>Risk: %{x:.3f}<br>Value: %{y:.1f} units<extra></extra>",
        customdata=no_action_pts["Customer_ID"],
    ))
    fig_scatter.add_trace(go.Scatter(
        x=intervene_pts["Churn_Probability"], y=intervene_pts["Expected_Profit"],
        mode="markers", marker=dict(color="#0D7A55", size=5, opacity=0.65),
        name="🟢 Intervene",
        hovertemplate="Customer #%{customdata}<br>Risk: %{x:.3f}<br>Value: %{y:.1f} units<extra></extra>",
        customdata=intervene_pts["Customer_ID"],
    ))
    fig_scatter.add_trace(go.Scatter(
        x=high_priority["Churn_Probability"], y=high_priority["Expected_Profit"],
        mode="markers", marker=dict(color="#DC2626", size=8, line=dict(color="#7F1D1D", width=1.5)),
        name="🔴 High Priority",
        hovertemplate="Customer #%{customdata}<br>Risk: %{x:.3f}<br>Value: %{y:.1f} units<extra></extra>",
        customdata=high_priority["Customer_ID"],
    ))
    fig_scatter.add_hline(y=600, line_dash="dot", line_color="#F59E0B", line_width=1.2,
                           annotation_text="High value (>600)", annotation_position="right",
                           annotation_font=dict(size=10, color="#B45309"))
    fig_scatter.add_vline(x=0.8, line_dash="dot", line_color="#EF4444", line_width=1.2,
                           annotation_text="High risk (>0.8)", annotation_position="top",
                           annotation_font=dict(size=10, color="#991B1B"))
    fig_scatter.update_layout(
        height=420, margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="#FFFFFF", plot_bgcolor="#FFFFFF",
        font=dict(family="Source Sans 3", size=12, color="#374151"),
        xaxis=dict(title="Risk Score", showgrid=True, gridcolor="#F3F4F6", range=[-0.02, 1.02], zeroline=False),
        yaxis=dict(title="Estimated Value (units)", showgrid=True, gridcolor="#F3F4F6", zeroline=False, tickformat=",.0f"),
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1,
                    bgcolor="rgba(255,255,255,0.9)", bordercolor="#DDE3EC", borderwidth=1),
        hovermode="closest",
    )
    st.plotly_chart(fig_scatter, use_container_width=True, config={"displayModeBar": False})

except ImportError:
    st.info("Install plotly (`pip install plotly`) to see the decision scatter chart.")

st.markdown('<hr class="divider">', unsafe_allow_html=True)

# ── Limitations ───────────────────────────────────────────────────────────────
st.markdown('<span class="sec-head">System Limitations</span>', unsafe_allow_html=True)
st.info("""
⚠️ **Before real-world deployment, consider:**

- **Simulated values** — estimated value uses CLV assumptions; calibrate with actual campaign cost data
- **Threshold sensitivity** — the 0.10 decision threshold suits this dataset; recalibrate for different churn base rates
- **No causal guarantee** — high risk score does not mean intervention will succeed; some customers may churn regardless
""")
st.caption("Trained on ~7,000 customers · Evaluated on 1,409 unseen test customers · 10-seed stability · 5-fold CV.")