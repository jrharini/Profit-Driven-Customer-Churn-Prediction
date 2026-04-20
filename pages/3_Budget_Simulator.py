import streamlit as st
import pandas as pd
import numpy as np
import os

st.set_page_config(page_title="Budget Simulator · Churn System", page_icon="💰", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Merriweather:wght@700;900&family=Source+Sans+3:wght@300;400;500;600;700&family=Source+Code+Pro:wght@400;500;600&display=swap');

:root { color-scheme: light only !important; }
*, *::before, *::after { color-scheme: light !important; }
html[data-theme="dark"], html[data-theme="light"] { background-color: #F7F8FA !important; }

html, body,
.stApp, [data-testid="stAppViewContainer"],
[data-testid="stMain"], .main, .block-container,
[data-testid="stVerticalBlock"], [data-testid="stHorizontalBlock"],
div.stMarkdown, [data-testid="stMarkdownContainer"] {
    background-color: #F7F8FA !important;
    color: #111827 !important;
    font-family: 'Source Sans 3', sans-serif !important;
}
p, span, div, label,
.stMarkdown p, .stMarkdown span,
[data-testid="stMarkdownContainer"] p { color: #111827 !important; }

section[data-testid="stSidebar"],
section[data-testid="stSidebar"] > div,
section[data-testid="stSidebar"] > div > div { background-color: #0F2236 !important; }
section[data-testid="stSidebar"] * { color: #CBD5E1 !important; }

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
    color: #1E3A5A !important; border-bottom: 2.5px solid #1E3A5A;
    padding-bottom: 7px; margin: 28px 0 16px 0; display: block;
}

.out-card {
    background: #FFFFFF; border: 1.5px solid #DDE3EC;
    border-top: 4px solid #1E3A5A; border-radius: 10px;
    padding: 24px 18px 18px 18px; text-align: center;
    box-shadow: 0 2px 10px rgba(15,34,54,0.07);
}
.out-card.teal  { border-top-color: #0D7A55; }
.out-card.amber { border-top-color: #C2790A; }
.out-lbl {
    display: block; font-size: 10.5px !important;
    font-weight: 700 !important; letter-spacing: 0.13em;
    text-transform: uppercase; color: #6B7280 !important; margin-bottom: 10px;
}
.out-val {
    display: block; font-family: 'Source Code Pro', monospace !important;
    font-size: 33px !important; font-weight: 600 !important;
    color: #0C1F3A !important; line-height: 1;
}
.out-val.teal  { color: #0D7A55 !important; }
.out-val.amber { color: #C2790A !important; }
.out-sub { display: block; font-size: 11.5px !important; color: #9CA3AF !important; margin-top: 8px; }

.insight {
    border-radius: 9px; padding: 14px 18px;
    font-size: 13.5px !important; line-height: 1.65;
    margin: 16px 0;
}
.insight.green  { background: #F0FDF4; border: 1px solid #BBF7D0; color: #14532D !important; }
.insight.blue   { background: #EFF6FF; border: 1px solid #BFDBFE; color: #1E3A5F !important; }
.insight.amber  { background: #FFFBEB; border: 1px solid #FDE68A; color: #78350F !important; }

/* ── FIX 3: Diminishing Returns Warning ── */
.dim-returns-warning {
    background: #FFF7ED;
    border: 1.5px solid #FED7AA;
    border-left: 5px solid #EA580C;
    border-radius: 0 10px 10px 0;
    padding: 16px 22px;
    margin: 0 0 16px 0;
}
.dim-returns-warning .drw-label {
    font-size: 10px !important; font-weight: 800 !important;
    letter-spacing: 0.18em; text-transform: uppercase;
    color: #9A3412 !important; display: block; margin-bottom: 8px;
}
.dim-returns-warning .drw-body { font-size: 13.5px !important; color: #7C2D12 !important; line-height: 1.7; }
.dim-returns-warning .drw-zones {
    display: flex; gap: 12px; margin-top: 12px; flex-wrap: wrap;
}
.dim-returns-warning .drw-zone {
    flex: 1; min-width: 130px;
    border-radius: 7px; padding: 9px 13px; text-align: center;
}
.drw-zone.good  { background: #F0FDF4; border: 1px solid #6EE7B7; }
.drw-zone.ok    { background: #FFFBEB; border: 1px solid #FDE68A; }
.drw-zone.bad   { background: #FEF2F2; border: 1px solid #FECACA; }
.drw-zone .dz-label { font-size: 10px !important; font-weight: 700 !important; letter-spacing: 0.1em; text-transform: uppercase; display: block; margin-bottom: 4px; }
.drw-zone.good .dz-label { color: #065F46 !important; }
.drw-zone.ok   .dz-label { color: #92400E !important; }
.drw-zone.bad  .dz-label { color: #991B1B !important; }
.drw-zone .dz-range { font-family: 'Source Code Pro', monospace; font-size: 15px !important; font-weight: 700 !important; display: block; }
.drw-zone.good .dz-range { color: #0D7A55 !important; }
.drw-zone.ok   .dz-range { color: #B45309 !important; }
.drw-zone.bad  .dz-range { color: #DC2626 !important; }
.drw-zone .dz-note { font-size: 11px !important; display: block; margin-top: 2px; }
.drw-zone.good .dz-note { color: #047857 !important; }
.drw-zone.ok   .dz-note { color: #92400E !important; }
.drw-zone.bad  .dz-note { color: #991B1B !important; }

.reftable {
    width: 100%; border-collapse: collapse;
    font-family: 'Source Sans 3', sans-serif; font-size: 14px;
    background: #FFFFFF;
}
.reftable thead tr { background: #0C1F3A; }
.reftable thead th {
    padding: 12px 16px; text-align: left;
    color: #F1F5F9 !important; font-size: 11px !important;
    font-weight: 700 !important; letter-spacing: 0.10em; text-transform: uppercase;
}
.reftable tbody td {
    padding: 11px 16px; border-bottom: 1px solid #EFF2F6;
    color: #1F2937 !important; font-family: 'Source Code Pro', monospace; font-size: 13px !important;
}
.reftable tbody td.label { font-family: 'Source Sans 3', sans-serif; font-size: 14px !important; }
.reftable tbody tr.sel td { background: #F0FDF4 !important; font-weight: 700 !important; color: #065F46 !important; }
.reftable tbody tr:hover td { background: #F8FAFC !important; }

.divider { border: none; border-top: 1.5px solid #DDE3EC; margin: 22px 0; }

/* ── Disclaimer ── */
.disclaimer {
    background: #FAFAFA; border: 1px solid #E5E7EB; border-radius: 8px;
    padding: 9px 15px; font-size: 12px !important; color: #6B7280 !important;
    display: flex; align-items: center; gap: 8px; margin-bottom: 6px;
}

/* ── Strategy Box ── */
.strategy-box {
    background: linear-gradient(135deg, #EFF6FF 0%, #F0F9FF 100%);
    border: 1px solid #93C5FD;
    border-left: 5px solid #1D4ED8;
    border-radius: 0 10px 10px 0;
    padding: 18px 22px;
    margin-top: 16px;
}
.strategy-box .sb-label {
    font-size: 10px !important; font-weight: 800 !important;
    letter-spacing: 0.18em; text-transform: uppercase;
    color: #1E3A8A !important; display: block; margin-bottom: 10px;
}
.strategy-box .sb-body { font-size: 13.5px !important; color: #1E3A5F !important; line-height: 1.8; }
.strategy-box .sb-body .sb-row { display: flex; align-items: flex-start; gap: 10px; margin: 4px 0; }
.strategy-box .sb-body .sb-icon { font-size: 15px; flex-shrink: 0; margin-top: 1px; }
</style>
""", unsafe_allow_html=True)

# ── Known data ────────────────────────────────────────────────────────────────
KNOWN = pd.DataFrame({
    "Budget_Pct":  [10,  20,  50,  100],
    "N_Customers": [140, 281, 704, 1409],
    "Profit":      [93073.84, 179333.37, 374579.46, 446563.33]
})

@st.cache_data
def load_budget():
    p = "results/budget_analysis.csv"
    if os.path.exists(p):
        df = pd.read_csv(p)
        if "Budget_Pct" not in df.columns and "Budget" in df.columns:
            df["Budget_Pct"] = df["Budget"].str.extract(r'(\d+)').astype(float)
        return df[["Budget_Pct","N_Customers","Profit"]].dropna()
    return KNOWN

bdf = load_budget()
COST_PER = 14326 / 1102   # ≈ 13.00 units

def interp(pct):
    x  = bdf["Budget_Pct"].values.astype(float)
    yp = bdf["Profit"].values.astype(float)
    yn = bdf["N_Customers"].values.astype(float)
    return float(np.interp(pct, x, yp)), int(np.interp(pct, x, yn))

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown('<div class="page-title">💰 Budget Simulator</div>', unsafe_allow_html=True)
st.markdown('<div class="page-sub">Adjust your intervention budget and instantly see expected financial impact — plan your strategy before deployment</div>', unsafe_allow_html=True)
st.markdown('<hr class="divider">', unsafe_allow_html=True)

# ── Disclaimer ────────────────────────────────────────────────────────────────
st.markdown("""
<div class="disclaimer">
    <span style="font-size:16px">ℹ️</span>
    <span>Profit values are simulated based on customer lifetime assumptions and used as a relative decision metric — not real currency figures.</span>
</div>
""", unsafe_allow_html=True)

# ── FIX 3: Diminishing Returns Warning ───────────────────────────────────────
st.markdown("""
<div class="dim-returns-warning">
    <span class="drw-label">⚠️ Diminishing Returns — Read Before Setting Your Budget</span>
    <div class="drw-body">
        Profit peaks around <strong>50–60% targeting</strong>. Beyond this threshold, ROI drops sharply 
        as lower-probability customers are included — intervention costs begin to outweigh expected savings. 
        Use the slider below to explore the curve, but stay within the recommended zone for optimal efficiency.
    </div>
    <div class="drw-zones">
        <div class="drw-zone good">
            <span class="dz-label">✅ Optimal Zone</span>
            <span class="dz-range">50–60%</span>
            <span class="dz-note">Best profit-to-cost ratio</span>
        </div>
        <div class="drw-zone ok">
            <span class="dz-label">⚠️ Acceptable</span>
            <span class="dz-range">20–50%</span>
            <span class="dz-note">Good ROI, lower total profit</span>
        </div>
        <div class="drw-zone bad">
            <span class="dz-label">🚫 Avoid</span>
            <span class="dz-range">&gt; 70%</span>
            <span class="dz-note">Marginal returns diminish rapidly</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Slider ────────────────────────────────────────────────────────────────────
st.markdown('<span class="sec-head">Set Your Budget Coverage</span>', unsafe_allow_html=True)

budget_pct = st.slider(
    "Percentage of total customer base to target with interventions",
    min_value=5, max_value=100, value=50, step=5, format="%d%%"
)

profit_est, n_est = interp(budget_pct)
cost_est = COST_PER * n_est
roi_est  = (profit_est / max(cost_est, 1)) * 100

# ── Output cards ──────────────────────────────────────────────────────────────
st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
o1, o2, o3 = st.columns(3)

with o1:
    st.markdown(f"""<div class="out-card">
        <span class="out-lbl">Customers Targeted</span>
        <span class="out-val">{n_est:,}</span>
        <span class="out-sub">{budget_pct}% of 1,409 test customers</span>
    </div>""", unsafe_allow_html=True)
with o2:
    st.markdown(f"""<div class="out-card teal">
        <span class="out-lbl">Expected Profit</span>
        <span class="out-val teal">{profit_est:,.0f} units</span>
        <span class="out-sub">net of intervention costs</span>
    </div>""", unsafe_allow_html=True)
with o3:
    st.markdown(f"""<div class="out-card amber">
        <span class="out-lbl">Est. Cost · ROI</span>
        <span class="out-val amber">{roi_est:.0f}%</span>
        <span class="out-sub">{cost_est:,.0f} units total cost · ~13 units/customer</span>
    </div>""", unsafe_allow_html=True)

# ── Insight banner ────────────────────────────────────────────────────────────
if budget_pct <= 20:
    cls = "amber"
    msg = f"⚠️ At {budget_pct}% budget you capture <strong>{profit_est:,.0f} units</strong> in profit. Increasing to 50% would add ~{374579-profit_est:,.0f} more units. Consider scaling up if intervention capacity allows."
elif budget_pct <= 65:
    cls = "green"
    msg = f"✅ At {budget_pct}% budget you achieve an excellent profit-to-cost balance (<strong>ROI {roi_est:.0f}%</strong>). This is the recommended operating range for this system."
else:
    cls = "blue"
    msg = f"ℹ️ At {budget_pct}% budget, marginal returns are decreasing — lower-probability customers are included. ROI is <strong>{roi_est:.0f}%</strong>. Consider stopping at 50–60% for optimal efficiency."

st.markdown(f'<div class="insight {cls}">{msg}</div>', unsafe_allow_html=True)
st.markdown('<hr class="divider">', unsafe_allow_html=True)

# ── Chart ─────────────────────────────────────────────────────────────────────
st.markdown('<span class="sec-head">Profit Curve — Budget vs Expected Return</span>', unsafe_allow_html=True)

try:
    import plotly.graph_objects as go

    x_all = np.linspace(bdf["Budget_Pct"].min(), 100, 300)
    y_all = np.array([interp(xx)[0] for xx in x_all])
    y_cost = np.array([COST_PER * interp(xx)[1] for xx in x_all])

    fig = go.Figure()

    # Optimal zone shading (50–60%)
    fig.add_vrect(
        x0=50, x1=60,
        fillcolor="rgba(13,122,85,0.08)",
        layer="below", line_width=0,
        annotation_text="Optimal Zone",
        annotation_position="top left",
        annotation_font=dict(size=10, color="#0D7A55"),
    )

    fig.add_trace(go.Scatter(
        x=np.concatenate([x_all, x_all[::-1]]),
        y=np.concatenate([y_all, np.zeros(len(y_all))]),
        fill='toself', fillcolor='rgba(13,122,85,0.06)',
        line=dict(color='rgba(0,0,0,0)'), showlegend=False, hoverinfo='skip'
    ))

    fig.add_trace(go.Scatter(
        x=x_all, y=y_cost, mode='lines',
        line=dict(color='#C2790A', width=1.5, dash='dot'),
        name='Intervention Cost',
        hovertemplate='Budget: %{x:.0f}%<br>Cost: %{y:,.0f} units<extra></extra>'
    ))

    fig.add_trace(go.Scatter(
        x=x_all, y=y_all, mode='lines',
        line=dict(color='#0D7A55', width=2.8),
        name='Expected Profit',
        hovertemplate='Budget: %{x:.0f}%<br>Profit: %{y:,.0f} units<extra></extra>'
    ))

    fig.add_trace(go.Scatter(
        x=bdf["Budget_Pct"], y=bdf["Profit"],
        mode='markers', marker=dict(color='#0C1F3A', size=9),
        name='Breakpoints', hovertemplate='%{x}% → %{y:,.0f} units<extra></extra>'
    ))

    fig.add_trace(go.Scatter(
        x=[budget_pct], y=[profit_est],
        mode='markers',
        marker=dict(color='#0D7A55', size=15, symbol='circle',
                    line=dict(color='#064E3B', width=2.5)),
        name=f'Your selection',
        hovertemplate=f'{budget_pct}% → {profit_est:,.0f} units<extra></extra>'
    ))

    fig.add_vline(x=budget_pct, line_dash="dot", line_color="#CBD5E1", line_width=1.5)
    # Diminishing returns boundary marker
    fig.add_vline(x=70, line_dash="dash", line_color="#EF4444", line_width=1.2,
                  annotation_text="⚠️ ROI drops here", annotation_position="top right",
                  annotation_font=dict(size=10, color="#DC2626"))

    fig.update_layout(
        height=420, margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor='#FFFFFF', plot_bgcolor='#FFFFFF',
        font=dict(family='Source Sans 3', size=12, color='#374151'),
        xaxis=dict(
            title=dict(text='Budget Coverage (%)', font=dict(size=12, color='#374151')),
            showgrid=True, gridcolor='#F3F4F6',
            ticksuffix='%', range=[0, 105], zeroline=False,
        ),
        yaxis=dict(
            title=dict(text='Expected Value (units)', font=dict(size=12, color='#374151')),
            showgrid=True, gridcolor='#F3F4F6',
            ticksuffix=' units', tickformat=',.0f', zeroline=False,
        ),
        legend=dict(
            orientation='h', yanchor='bottom', y=1.01, xanchor='right', x=1,
            bgcolor='rgba(255,255,255,0.9)', bordercolor='#DDE3EC', borderwidth=1
        ),
        hovermode='x unified'
    )

    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

except ImportError:
    chart_df = pd.DataFrame({"Budget %": bdf["Budget_Pct"].values, "Profit": bdf["Profit"].values}).set_index("Budget %")
    st.line_chart(chart_df)

st.markdown('<hr class="divider">', unsafe_allow_html=True)

# ── Reference table ───────────────────────────────────────────────────────────
st.markdown('<span class="sec-head">Breakpoint Reference Table</span>', unsafe_allow_html=True)

ref_rows = ""
for _, row in bdf.iterrows():
    pct = int(row["Budget_Pct"])
    n   = int(row["N_Customers"])
    p   = float(row["Profit"])
    c   = COST_PER * n
    roi = (p / max(c, 1)) * 100
    is_sel = abs(pct - budget_pct) == min(abs(bdf["Budget_Pct"] - budget_pct))
    cls = "sel" if is_sel else ""
    ref_rows += f"""<tr class="{cls}">
        <td class="label">{pct}% of customers</td>
        <td>{n:,}</td>
        <td>{p:,.2f} units</td>
        <td>{c:,.2f} units</td>
        <td>{roi:.0f}%</td>
    </tr>"""

st.markdown(f"""
<div style="overflow:hidden;border-radius:10px;border:1.5px solid #DDE3EC;box-shadow:0 2px 10px rgba(15,34,54,0.07)">
<table class="reftable">
<thead>
    <tr>
        <th>Budget Level</th>
        <th>Customers Targeted</th>
        <th>Expected Profit</th>
        <th>Est. Intervention Cost</th>
        <th>ROI</th>
    </tr>
</thead>
<tbody>{ref_rows}</tbody>
</table>
</div>
<div style="margin-top:10px;font-size:12px;color:#9CA3AF;padding:0 4px">
    Highlighted row = closest match to your selected budget. 
    ROI = Profit ÷ Intervention Cost. Profit-per-customer = Churn Prob × 0.5 × (65 × 24) − 13 units.
</div>
""", unsafe_allow_html=True)

# ── Strategy Box ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="strategy-box">
    <span class="sb-label">🎯 Recommended Strategy</span>
    <div class="sb-body">
        <div class="sb-row"><span class="sb-icon">✅</span><span>Target the <strong>top 50–60% of customers</strong> ranked by churn probability for optimal ROI — this range captures the majority of high-risk churners before diminishing returns set in.</span></div>
        <div class="sb-row"><span class="sb-icon">🎯</span><span>Focus intervention efforts on customers with <strong>churn probability &gt; 0.7</strong> — these high-risk individuals drive the bulk of expected profit and have the clearest business case for intervention.</span></div>
        <div class="sb-row"><span class="sb-icon">⚠️</span><span>Avoid targeting beyond <strong>70% of the customer base</strong> — beyond this threshold, marginal returns diminish rapidly as low-probability customers dominate and intervention costs outweigh the expected savings.</span></div>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)