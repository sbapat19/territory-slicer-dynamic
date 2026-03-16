import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np

# ─── PAGE CONFIG ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Territory Slicer",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── LIGHT MODE STYLING ─────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&display=swap');
    /* Force light mode everywhere */
    .stApp {
        font-family: 'DM Sans', sans-serif;
        background-color: #ffffff !important;
        color: #1a1d23 !important;
    }

    /* Kill the dark top bar */
    header[data-testid="stHeader"] {
        background-color: #ffffff !important;
        border-bottom: 1px solid #e2e4e9;
    }

    /* Main content padding */
    .block-container {
        padding-top: 2rem !important;
        padding-bottom: 2rem !important;
    }

    /* Header - no box */
    .main-header {
        padding: 0 0 16px;
        margin-bottom: 20px;
        border-bottom: 1px solid #e2e4e9;
    }
    .main-header h1 {
        font-size: 32px; font-weight: 700; margin: 0; color: #1a1d23;
        letter-spacing: -0.5px;
    }
    .main-header p { font-size: 14px; color: #6b7280; margin: 6px 0 0; }

    /* Section headers */
    .section-header {
        font-size: 17px; font-weight: 700; color: #1a1d23;
        margin: 28px 0 6px; padding-bottom: 8px;
        border-bottom: 2px solid #e2e4e9;
    }
    .section-desc {
        font-size: 13px; color: #6b7280; margin: 0 0 16px;
        line-height: 1.6;
    }

    /* Segment overview inline metrics */
    .overview-row {
        display: flex; gap: 32px; flex-wrap: wrap;
        margin-bottom: 8px; padding: 4px 0;
    }
    .overview-item {
        font-size: 14px; color: #1a1d23;
    }
    .overview-item .ov-label { color: #6b7280; font-size: 12px; }
    .overview-item .ov-value { font-weight: 700; font-size: 20px; }

    /* CV badge */
    .cv-badge {
        display: inline-block; font-size: 13px; font-weight: 600;
        padding: 4px 14px; border-radius: 20px; margin-top: 4px;
    }
    .cv-good { background: #e8f8ef; color: #1a9a4a; }
    .cv-ok { background: #fef9e7; color: #b7791f; }
    .cv-bad { background: #fdedec; color: #c0392b; }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: #f8f9fb !important;
        padding-top: 1.5rem;
    }
    section[data-testid="stSidebar"] > div {
        padding: 0 1.2rem;
    }
    section[data-testid="stSidebar"] .stMarkdown p,
    section[data-testid="stSidebar"] .stMarkdown li,
    section[data-testid="stSidebar"] .stMarkdown h1,
    section[data-testid="stSidebar"] .stMarkdown h2,
    section[data-testid="stSidebar"] .stMarkdown h3 {
        color: #1a1d23 !important;
    }

    /* Force all Plotly chart containers to white */
    .stPlotlyChart, .js-plotly-plot, .plot-container {
        background-color: #ffffff !important;
    }

    /* Methodology boxes */
    .method-box {
        background: #ffffff; border-radius: 12px; padding: 22px 26px;
        box-shadow: 0 1px 4px rgba(0,0,0,0.06);
        border-left: 4px solid #4fc3f7;
        margin-bottom: 16px;
    }
    .method-box h3 { font-size: 15px; font-weight: 700; margin: 0 0 10px; color: #1a1d23; }
    .method-box p { font-size: 13px; color: #444; line-height: 1.8; margin: 0 0 10px; }
</style>
""", unsafe_allow_html=True)

# ─── LOAD DATA ───────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    reps = pd.read_csv("data/reps.csv")
    reps.columns = reps.columns.str.strip()

    accounts = pd.read_csv("data/accounts.csv")
    accounts.columns = accounts.columns.str.strip()
    # Keep only the columns we need (ignore any extra summary columns)
    keep_cols = ["Account_ID", "Account_Name", "Current_Rep", "ARR",
                 "Location", "Num_Employees", "Num_Marketers", "Risk_Score"]
    accounts = accounts[[c for c in keep_cols if c in accounts.columns]]

    accounts["ARR"] = pd.to_numeric(accounts["ARR"], errors="coerce")
    accounts["Num_Employees"] = pd.to_numeric(accounts["Num_Employees"], errors="coerce")
    accounts["Num_Marketers"] = pd.to_numeric(accounts["Num_Marketers"], errors="coerce")
    accounts["Risk_Score"] = pd.to_numeric(accounts["Risk_Score"], errors="coerce")
    accounts = accounts.dropna(subset=["ARR", "Num_Employees"])

    return reps, accounts

reps_df, accounts_df = load_data()

# ─── ALGORITHM ───────────────────────────────────────────────────────────────
def greedy_distribute(accounts, rep_names, reps_df=None,
                      risk_weight=False, churn_penalty=0, geo_bonus=0):
    """
    Enhanced LPT greedy heuristic.
    For each unassigned account (sorted by ARR desc), calculate an effective cost
    for each rep, then assign to the rep with the lowest adjusted total.

    - risk_weight: if True, multiply ARR by risk factor (high=1.3, med=1.0, low=0.8)
    - churn_penalty: $ added to effective cost when assigning to a DIFFERENT rep
      than the account's current rep (incentivizes keeping existing relationships)
    - geo_bonus: $ subtracted from effective cost when rep and account share a state
      (incentivizes geographic alignment)
    """
    if len(rep_names) == 0 or len(accounts) == 0:
        return pd.DataFrame()
    sorted_accts = accounts.sort_values("ARR", ascending=False).copy()
    rep_totals = {name: 0.0 for name in rep_names}

    # Build rep location lookup
    rep_locations = {}
    if reps_df is not None and geo_bonus > 0:
        for _, row in reps_df.iterrows():
            rep_locations[row["Rep_Name"]] = str(row["Location"]).strip().upper()

    assignments = []
    for _, acct in sorted_accts.iterrows():
        # Calculate effective ARR for this account (used for load tracking)
        arr = acct["ARR"]
        effective_arr = arr
        if risk_weight:
            risk = acct.get("Risk_Score", 50)
            if risk >= 75:
                effective_arr = arr * 1.3
            elif risk < 26:
                effective_arr = arr * 0.8

        # For each candidate rep, calculate the cost of assigning this account
        best_rep = None
        best_cost = float("inf")
        for rep in rep_names:
            cost = rep_totals[rep] + effective_arr

            # Churn penalty: costs more to switch away from current rep
            if churn_penalty > 0 and "Current_Rep" in acct.index:
                current = str(acct["Current_Rep"]).strip()
                if current != rep:
                    cost += churn_penalty

            # Geo bonus: cheaper to assign to a co-located rep
            if geo_bonus > 0 and rep in rep_locations:
                acct_loc = str(acct.get("Location", "")).strip().upper()
                if acct_loc and acct_loc == rep_locations[rep]:
                    cost -= geo_bonus

            if cost < best_cost:
                best_cost = cost
                best_rep = rep

        rep_totals[best_rep] += effective_arr
        assignments.append(best_rep)

    sorted_accts["Assigned_Rep"] = assignments
    return sorted_accts

def calc_cv(values):
    """Coefficient of variation: stdev / mean × 100."""
    if len(values) < 2:
        return 0.0
    mean = np.mean(values)
    if mean == 0:
        return 0.0
    return (np.std(values) / mean) * 100

# ─── HELPERS ─────────────────────────────────────────────────────────────────
def fmt_arr(n):
    if n >= 1_000_000:
        return f"${n/1_000_000:.2f}M"
    elif n >= 1_000:
        return f"${n/1_000:.0f}K"
    return f"${n:,.0f}"

ENT_COLORS = ["#1a5276", "#21618c", "#2874a6", "#2e86c1"]
MM_COLORS = ["#7d5a38", "#936a42", "#a97b4c", "#bf8c56", "#d5a06a", "#eab580"]
ENT_MAIN = "#1a5276"
MM_MAIN = "#7d5a38"
CHART_BG = "white"

def cv_badge_html(cv_val, label="Workload Variation"):
    css = "cv-good" if cv_val < 5 else "cv-ok" if cv_val < 10 else "cv-bad"
    return (
        f'<div style="text-align:center; margin: 0 0 8px;">'
        f'<span class="cv-badge {css}">{label}: {cv_val:.1f}%</span>'
        f'<br><span style="font-size:11px; color:#999;">Lower = more balanced. Under 5% is strong.</span>'
        f'</div>'
    )

def std_layout(title_text, height=340):
    return dict(
        title=dict(text=title_text, font=dict(size=14, color="#1a1d23")),
        height=height, margin=dict(t=50, b=50, l=40, r=20),
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
        font=dict(color="#1a1d23"),
        yaxis=dict(gridcolor="#f0f0f0", tickfont=dict(color="#1a1d23")),
        xaxis=dict(tickangle=-20, tickfont=dict(color="#1a1d23")),
        showlegend=False,
    )

# ─── SIDEBAR ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🎯 Territory Slicer")
    st.markdown("---")

    st.markdown("#### Employee Count Threshold")
    st.markdown(
        "<p style='font-size:12px; color:#6b7280;'>"
        "Accounts <b>at or above</b> this threshold → Enterprise<br>"
        "Accounts <b>below</b> → Mid-Market</p>",
        unsafe_allow_html=True,
    )

    threshold = st.slider(
        "Threshold (slider)",
        min_value=1_000, max_value=180_000, value=50_000, step=1_000,
        format="%d employees",
        label_visibility="collapsed",
    )
    threshold = st.number_input(
        "Or type a threshold",
        min_value=1_000, max_value=180_000, value=threshold, step=1_000,
    )

    st.markdown(f"**Threshold: {threshold:,} employees**")
    st.markdown("---")

    # Territory rules
    st.markdown("#### Territory Rules")

    use_risk = st.checkbox(
        "Weight by risk",
        help="High-risk accounts (75+) are treated as 1.3× their ARR in the algorithm. "
             "Low-risk accounts (1-25) are treated as 0.8×. This spreads high-risk accounts "
             "more evenly since they consume more rep capacity.",
    )

    use_churn = st.checkbox(
        "Prefer keeping current rep",
        help="Adds a penalty when reassigning an account to a different rep than who manages "
             "it today. Higher penalty = stickier assignments. Set to $0 to disable.",
    )
    churn_penalty = 0
    if use_churn:
        churn_penalty = st.number_input(
            "Churn penalty ($ equivalent)",
            min_value=0, max_value=500_000, value=50_000, step=10_000,
            help="Dollar amount added to the effective cost of assigning an account to a rep "
                 "other than its current one. A $50K penalty means the algorithm will only "
                 "reassign if the ARR balance improvement exceeds $50K.",
        )

    use_geo = st.checkbox(
        "Prefer location alignment",
        help="Gives a bonus when a rep and account are in the same state. "
             "Higher bonus = stronger geographic preference.",
    )
    geo_bonus = 0
    if use_geo:
        geo_bonus = st.number_input(
            "Location bonus ($ equivalent)",
            min_value=0, max_value=500_000, value=30_000, step=10_000,
            help="Dollar amount subtracted from the effective cost when rep and account "
                 "share a state. Makes the algorithm prefer co-located assignments.",
        )

    st.markdown("---")

    # Detect segment labels
    segment_values = reps_df["Segment"].unique().tolist()
    ent_label = [s for s in segment_values if "ent" in s.lower()][0]
    mm_label = [s for s in segment_values if "mid" in s.lower()][0]

    ent_reps = reps_df[reps_df["Segment"] == ent_label]["Rep_Name"].tolist()
    mm_reps = reps_df[reps_df["Segment"] == mm_label]["Rep_Name"].tolist()

    st.markdown(f"**Enterprise Reps:** {len(ent_reps)}")
    st.markdown(f"**Mid-Market Reps:** {len(mm_reps)}")

# ─── SEGMENTATION & DISTRIBUTION ─────────────────────────────────────────────
ent_accounts = accounts_df[accounts_df["Num_Employees"] >= threshold].copy()
mm_accounts = accounts_df[accounts_df["Num_Employees"] < threshold].copy()

ent_assigned = greedy_distribute(ent_accounts, ent_reps, reps_df=reps_df,
                                  risk_weight=use_risk, churn_penalty=churn_penalty,
                                  geo_bonus=geo_bonus)
mm_assigned = greedy_distribute(mm_accounts, mm_reps, reps_df=reps_df,
                                risk_weight=use_risk, churn_penalty=churn_penalty,
                                geo_bonus=geo_bonus)

def rep_summary(assigned_df, rep_names):
    rows = []
    for rep in rep_names:
        ra = assigned_df[assigned_df["Assigned_Rep"] == rep] if len(assigned_df) > 0 else pd.DataFrame()
        n = len(ra)
        arr = ra["ARR"].sum() if n > 0 else 0
        high = len(ra[ra["Risk_Score"] >= 75]) if n > 0 else 0
        med = len(ra[(ra["Risk_Score"] >= 26) & (ra["Risk_Score"] < 75)]) if n > 0 else 0
        low = len(ra[ra["Risk_Score"] < 26]) if n > 0 else 0
        mktr = ra["Num_Marketers"].sum() if n > 0 else 0
        rows.append({
            "Rep": rep, "Accounts": n, "Total ARR": arr,
            "High Risk (75+)": high, "Med Risk (26-74)": med, "Low Risk (1-25)": low,
            "Total Marketers": mktr,
        })
    return pd.DataFrame(rows)

ent_summary = rep_summary(ent_assigned, ent_reps)
mm_summary = rep_summary(mm_assigned, mm_reps)

ent_cv = calc_cv(ent_summary["Total ARR"].values)
mm_cv = calc_cv(mm_summary["Total ARR"].values)
ent_risk_cv = calc_cv(ent_summary["High Risk (75+)"].values)
mm_risk_cv = calc_cv(mm_summary["High Risk (75+)"].values)

# ─── HEADER ──────────────────────────────────────────────────────────────────
st.markdown(
    '<div class="main-header">'
    '<h1>🎯 Territory Slicer</h1>'
    f'<p>Balance sales territory planning by toggling the employee threshold to segment territories based on size and ARR. '
    f'{len(accounts_df)} accounts · {len(reps_df)} reps · {fmt_arr(accounts_df["ARR"].sum())} total ARR</p>'
    '</div>',
    unsafe_allow_html=True,
)

# ─── SEGMENT OVERVIEW ────────────────────────────────────────────────────────
st.markdown('<div class="section-header">Segment Overview</div>', unsafe_allow_html=True)
st.markdown(
    '<p class="section-desc">How accounts and ARR split at the current threshold. '
    'Enterprise reps handle fewer, larger accounts; Mid-Market reps handle more, smaller ones.</p>',
    unsafe_allow_html=True,
)

arr_per_ent = ent_accounts["ARR"].sum() / len(ent_reps) if len(ent_reps) > 0 else 0
arr_per_mm = mm_accounts["ARR"].sum() / len(mm_reps) if len(mm_reps) > 0 else 0

st.markdown(
    f'<div class="overview-row">'
    f'<div class="overview-item"><div class="ov-label">Total Accounts</div><div class="ov-value">{len(accounts_df)}</div></div>'
    f'<div class="overview-item"><div class="ov-label">Total ARR</div><div class="ov-value">{fmt_arr(accounts_df["ARR"].sum())}</div></div>'
    f'<div class="overview-item" style="margin-left:16px; padding-left:16px; border-left:2px solid {ENT_MAIN};">'
    f'<div class="ov-label">Enterprise Accounts</div><div class="ov-value">{len(ent_accounts)}</div></div>'
    f'<div class="overview-item"><div class="ov-label">Enterprise ARR</div><div class="ov-value">{fmt_arr(ent_accounts["ARR"].sum())}</div></div>'
    f'<div class="overview-item" style="margin-left:16px; padding-left:16px; border-left:2px solid {MM_MAIN};">'
    f'<div class="ov-label">Mid-Market Accounts</div><div class="ov-value">{len(mm_accounts)}</div></div>'
    f'<div class="overview-item"><div class="ov-label">Mid-Market ARR</div><div class="ov-value">{fmt_arr(mm_accounts["ARR"].sum())}</div></div>'
    f'</div>',
    unsafe_allow_html=True,
)

# ─── TOTAL ARR PER REP ──────────────────────────────────────────────────────
st.markdown('<div class="section-header">Total ARR per Rep</div>', unsafe_allow_html=True)
st.markdown(
    '<p class="section-desc">Total managed ARR assigned to each rep after the greedy balancing algorithm. '
    'The dashed line shows the segment average. Workload Variation (CV) measures how evenly ARR is spread.</p>',
    unsafe_allow_html=True,
)

col_l, col_r = st.columns(2)

with col_l:
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=ent_summary["Rep"], y=ent_summary["Total ARR"],
        marker_color=ENT_COLORS * 2,
        text=[fmt_arr(v) for v in ent_summary["Total ARR"]],
        textposition="outside", textfont=dict(size=11),
    ))
    fig.update_layout(**std_layout("Enterprise Reps — Total ARR"))
    fig.update_yaxes(tickformat="$,.0f")
    st.plotly_chart(fig, use_container_width=True)
    st.markdown(cv_badge_html(ent_cv), unsafe_allow_html=True)

with col_r:
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=mm_summary["Rep"], y=mm_summary["Total ARR"],
        marker_color=MM_COLORS,
        text=[fmt_arr(v) for v in mm_summary["Total ARR"]],
        textposition="outside", textfont=dict(size=11),
    ))
    fig.update_layout(**std_layout("Mid-Market Reps — Total ARR"))
    fig.update_yaxes(tickformat="$,.0f")
    st.plotly_chart(fig, use_container_width=True)
    st.markdown(cv_badge_html(mm_cv), unsafe_allow_html=True)

# ─── ACCOUNTS PER REP ────────────────────────────────────────────────────────
st.markdown('<div class="section-header">Accounts per Rep</div>', unsafe_allow_html=True)
st.markdown(
    '<p class="section-desc">Number of accounts each rep manages. Enterprise reps should have fewer, '
    'higher-value accounts; Mid-Market reps handle a higher volume of smaller accounts.</p>',
    unsafe_allow_html=True,
)

col_l, col_r = st.columns(2)

with col_l:
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=ent_summary["Rep"], y=ent_summary["Accounts"],
        marker_color=ENT_COLORS * 2,
        text=ent_summary["Accounts"], textposition="outside", textfont=dict(size=12),
    ))
    fig.update_layout(**std_layout("Enterprise Reps — Account Count"))
    fig.update_yaxes(title="# of Accounts")
    st.plotly_chart(fig, use_container_width=True)

with col_r:
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=mm_summary["Rep"], y=mm_summary["Accounts"],
        marker_color=MM_COLORS,
        text=mm_summary["Accounts"], textposition="outside", textfont=dict(size=12),
    ))
    fig.update_layout(**std_layout("Mid-Market Reps — Account Count"))
    fig.update_yaxes(title="# of Accounts")
    st.plotly_chart(fig, use_container_width=True)

# ─── RISK EXPOSURE ───────────────────────────────────────────────────────────
st.markdown('<div class="section-header">Risk Exposure by Rep</div>', unsafe_allow_html=True)
st.markdown(
    '<p class="section-desc">Even when ARR is balanced, risk may not be. A rep loaded with high-risk (75+) '
    'accounts spends more time on retention and less on expansion — their effective capacity is lower '
    'than the ARR alone suggests. Risk Concentration CV measures how evenly high-risk accounts are spread.</p>',
    unsafe_allow_html=True,
)

col_l, col_r = st.columns(2)

with col_l:
    fig = go.Figure()
    fig.add_trace(go.Bar(name="Low (1-25)", x=ent_summary["Rep"],
                         y=ent_summary["Low Risk (1-25)"], marker_color="#27ae60"))
    fig.add_trace(go.Bar(name="Med (26-74)", x=ent_summary["Rep"],
                         y=ent_summary["Med Risk (26-74)"], marker_color="#f39c12"))
    fig.add_trace(go.Bar(name="High (75+)", x=ent_summary["Rep"],
                         y=ent_summary["High Risk (75+)"], marker_color="#e74c3c"))
    fig.update_layout(
        barmode="stack",
        title=dict(text="Enterprise Reps — Risk Breakdown", font=dict(size=14, color="#1a1d23")),
        height=360, margin=dict(t=80, b=50, l=40, r=20),
        xaxis=dict(tickangle=-20), plot_bgcolor="#ffffff", paper_bgcolor="#ffffff", font=dict(color="#1a1d23"),
        yaxis=dict(title=dict(text="# of Accounts", font=dict(color="#1a1d23")), gridcolor="#f0f0f0", tickfont=dict(color="#1a1d23")),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=11, color="#1a1d23")),
    )
    st.plotly_chart(fig, use_container_width=True)
    st.markdown(cv_badge_html(ent_risk_cv, "Risk Concentration CV (High-Risk Accounts)"), unsafe_allow_html=True)

with col_r:
    fig = go.Figure()
    fig.add_trace(go.Bar(name="Low (1-25)", x=mm_summary["Rep"],
                         y=mm_summary["Low Risk (1-25)"], marker_color="#27ae60"))
    fig.add_trace(go.Bar(name="Med (26-74)", x=mm_summary["Rep"],
                         y=mm_summary["Med Risk (26-74)"], marker_color="#f39c12"))
    fig.add_trace(go.Bar(name="High (75+)", x=mm_summary["Rep"],
                         y=mm_summary["High Risk (75+)"], marker_color="#e74c3c"))
    fig.update_layout(
        barmode="stack",
        title=dict(text="Mid-Market Reps — Risk Breakdown", font=dict(size=14, color="#1a1d23")),
        height=360, margin=dict(t=80, b=50, l=40, r=20),
        xaxis=dict(tickangle=-20), plot_bgcolor="#ffffff", paper_bgcolor="#ffffff", font=dict(color="#1a1d23"),
        yaxis=dict(title=dict(text="# of Accounts", font=dict(color="#1a1d23")), gridcolor="#f0f0f0", tickfont=dict(color="#1a1d23")),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=11, color="#1a1d23")),
    )
    st.plotly_chart(fig, use_container_width=True)
    st.markdown(cv_badge_html(mm_risk_cv, "Risk Concentration CV (High-Risk Accounts)"), unsafe_allow_html=True)

# ─── SEAT PENETRATION ────────────────────────────────────────────────────────
st.markdown('<div class="section-header">Seat Penetration by Rep</div>', unsafe_allow_html=True)
st.markdown(
    '<p class="section-desc">Assuming per-seat pricing, ARR ÷ Marketers approximates how many seats '
    'have been sold at each account. Lower values suggest low seat adoption — these accounts have '
    'significant room for expansion. Higher values indicate deeper adoption with less immediate growth headroom.</p>',
    unsafe_allow_html=True,
)

def penetration_buckets(assigned_df, rep_names):
    """For each rep, count accounts in each ARR/marketer bucket."""
    rows = []
    for rep in rep_names:
        ra = assigned_df[assigned_df["Assigned_Rep"] == rep] if len(assigned_df) > 0 else pd.DataFrame()
        if len(ra) == 0:
            rows.append({"Rep": rep, "Barely Landed (<$20)": 0, "Early Adoption ($20-50)": 0,
                         "Growing ($50-100)": 0, "Well-Penetrated ($100+)": 0})
            continue
        ra = ra.copy()
        ra["ARR_per_Mktr"] = ra.apply(
            lambda r: r["ARR"] / r["Num_Marketers"] if r["Num_Marketers"] > 0 else 0, axis=1
        )
        rows.append({
            "Rep": rep,
            "Barely Landed (<$20)": len(ra[ra["ARR_per_Mktr"] < 20]),
            "Early Adoption ($20-50)": len(ra[(ra["ARR_per_Mktr"] >= 20) & (ra["ARR_per_Mktr"] < 50)]),
            "Growing ($50-100)": len(ra[(ra["ARR_per_Mktr"] >= 50) & (ra["ARR_per_Mktr"] < 100)]),
            "Well-Penetrated ($100+)": len(ra[ra["ARR_per_Mktr"] >= 100]),
        })
    return pd.DataFrame(rows)

ent_pen = penetration_buckets(ent_assigned, ent_reps)
mm_pen = penetration_buckets(mm_assigned, mm_reps)

PEN_COLORS = {"Barely Landed (<$20)": "#3498db", "Early Adoption ($20-50)": "#2ecc71",
              "Growing ($50-100)": "#f39c12", "Well-Penetrated ($100+)": "#e74c3c"}

col_l, col_r = st.columns(2)

with col_l:
    fig = go.Figure()
    for bucket, color in PEN_COLORS.items():
        fig.add_trace(go.Bar(name=bucket, x=ent_pen["Rep"], y=ent_pen[bucket], marker_color=color))
    fig.update_layout(
        barmode="stack",
        title=dict(text="Enterprise Reps — Seat Penetration", font=dict(size=14, color="#1a1d23")),
        height=380, margin=dict(t=80, b=50, l=40, r=20),
        xaxis=dict(tickangle=-20), plot_bgcolor="#ffffff", paper_bgcolor="#ffffff", font=dict(color="#1a1d23"),
        yaxis=dict(title=dict(text="# of Accounts", font=dict(color="#1a1d23")), gridcolor="#f0f0f0", tickfont=dict(color="#1a1d23")),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=10, color="#1a1d23")),
    )
    st.plotly_chart(fig, use_container_width=True)

with col_r:
    fig = go.Figure()
    for bucket, color in PEN_COLORS.items():
        fig.add_trace(go.Bar(name=bucket, x=mm_pen["Rep"], y=mm_pen[bucket], marker_color=color))
    fig.update_layout(
        barmode="stack",
        title=dict(text="Mid-Market Reps — Seat Penetration", font=dict(size=14, color="#1a1d23")),
        height=380, margin=dict(t=80, b=50, l=40, r=20),
        xaxis=dict(tickangle=-20), plot_bgcolor="#ffffff", paper_bgcolor="#ffffff", font=dict(color="#1a1d23"),
        yaxis=dict(title=dict(text="# of Accounts", font=dict(color="#1a1d23")), gridcolor="#f0f0f0", tickfont=dict(color="#1a1d23")),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=10, color="#1a1d23")),
    )
    st.plotly_chart(fig, use_container_width=True)

# ─── BEFORE VS AFTER ─────────────────────────────────────────────────────────
st.markdown(
    '<div class="section-header">Current Generalist Model vs Proposed Segmented Model</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<p class="section-desc">Today, all 10 reps cover all account sizes with no specialization. '
    'The segmented model assigns Enterprise reps to large accounts and Mid-Market reps to smaller ones — '
    'each rep develops the right skills and motions for their segment, and workload is balanced within each tier.</p>',
    unsafe_allow_html=True,
)

current_data = pd.DataFrame([
    {"Rep": "Ariel", "Accounts": 56, "ARR": 12518119},
    {"Rep": "Daisy Duck", "Accounts": 44, "ARR": 10588810},
    {"Rep": "Donald Duck", "Accounts": 53, "ARR": 13214694},
    {"Rep": "Elsa", "Accounts": 41, "ARR": 9985641},
    {"Rep": "Goofy", "Accounts": 46, "ARR": 13764464},
    {"Rep": "Mickey Mouse", "Accounts": 57, "ARR": 14905304},
    {"Rep": "Minnie Mouse", "Accounts": 51, "ARR": 13204673},
    {"Rep": "Moana", "Accounts": 61, "ARR": 15249451},
    {"Rep": "Pluto", "Accounts": 43, "ARR": 10737834},
    {"Rep": "Simba", "Accounts": 48, "ARR": 12982979},
])
current_cv = calc_cv(current_data["ARR"].values)

col_l, col_r = st.columns(2)

with col_l:
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=current_data["Rep"], y=current_data["ARR"],
        marker_color="#888",
        text=[fmt_arr(v) for v in current_data["ARR"]],
        textposition="outside", textfont=dict(size=10),
    ))
    fig.update_layout(
        title=dict(text=f"BEFORE — Generalist (CV: {current_cv:.1f}%)", font=dict(size=13, color="#1a1d23")),
        height=340, margin=dict(t=50, b=50, l=40, r=20),
        yaxis=dict(tickformat="$,.0f", gridcolor="#f0f0f0", tickfont=dict(color="#1a1d23")),
        xaxis=dict(tickangle=-25, tickfont=dict(color="#1a1d23")),
        plot_bgcolor="#ffffff", paper_bgcolor="#ffffff", font=dict(color="#1a1d23"),
    )
    st.plotly_chart(fig, use_container_width=True)

with col_r:
    all_new = pd.concat([ent_summary, mm_summary], ignore_index=True)
    colors_list = [ENT_MAIN] * len(ent_summary) + [MM_MAIN] * len(mm_summary)
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=all_new["Rep"], y=all_new["Total ARR"],
        marker_color=colors_list,
        text=[fmt_arr(v) for v in all_new["Total ARR"]],
        textposition="outside", textfont=dict(size=10),
    ))
    fig.update_layout(
        title=dict(text=f"AFTER — Segmented (Ent CV: {ent_cv:.1f}% · MM CV: {mm_cv:.1f}%)", font=dict(size=13, color="#1a1d23")),
        height=340, margin=dict(t=50, b=50, l=40, r=20),
        yaxis=dict(tickformat="$,.0f", gridcolor="#f0f0f0", tickfont=dict(color="#1a1d23")),
        xaxis=dict(tickangle=-25, tickfont=dict(color="#1a1d23")),
        plot_bgcolor="#ffffff", paper_bgcolor="#ffffff", font=dict(color="#1a1d23"),
    )
    st.plotly_chart(fig, use_container_width=True)

# ─── ACCOUNT DETAIL TABLES ──────────────────────────────────────────────────
st.markdown('<div class="section-header">Account Assignments — Detail View</div>', unsafe_allow_html=True)
st.markdown(
    '<p class="section-desc">Full list of account-to-rep assignments. Sort by any column to explore.</p>',
    unsafe_allow_html=True,
)

tab_ent, tab_mm = st.tabs(["Enterprise Accounts", "Mid-Market Accounts"])

with tab_ent:
    if len(ent_assigned) > 0:
        d = ent_assigned[["Assigned_Rep", "Account_Name", "Num_Employees", "ARR",
                          "Num_Marketers", "Risk_Score"]].copy()
        d.columns = ["Assigned Rep", "Account", "Employees", "ARR", "Marketers", "Risk Score"]
        d = d.sort_values(["Assigned Rep", "ARR"], ascending=[True, False])
        st.dataframe(
            d.style.format({"ARR": "${:,.0f}", "Employees": "{:,.0f}", "Marketers": "{:,.0f}"}),
            use_container_width=True, height=400,
        )
    else:
        st.info("No Enterprise accounts at this threshold.")

with tab_mm:
    if len(mm_assigned) > 0:
        d = mm_assigned[["Assigned_Rep", "Account_Name", "Num_Employees", "ARR",
                          "Num_Marketers", "Risk_Score"]].copy()
        d.columns = ["Assigned Rep", "Account", "Employees", "ARR", "Marketers", "Risk Score"]
        d = d.sort_values(["Assigned Rep", "ARR"], ascending=[True, False])
        st.dataframe(
            d.style.format({"ARR": "${:,.0f}", "Employees": "{:,.0f}", "Marketers": "{:,.0f}"}),
            use_container_width=True, height=400,
        )
    else:
        st.info("No Mid-Market accounts at this threshold.")

# ─── DOWNLOAD ────────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">Download Assignments</div>', unsafe_allow_html=True)
st.markdown(
    '<p class="section-desc">Export the current territory assignments as a CSV file.</p>',
    unsafe_allow_html=True,
)

# Combine enterprise and mid-market assignments
all_assigned = pd.DataFrame()
if len(ent_assigned) > 0:
    ent_dl = ent_assigned.copy()
    ent_dl["Segment"] = "Enterprise"
    all_assigned = pd.concat([all_assigned, ent_dl])
if len(mm_assigned) > 0:
    mm_dl = mm_assigned.copy()
    mm_dl["Segment"] = "Mid-Market"
    all_assigned = pd.concat([all_assigned, mm_dl])

if len(all_assigned) > 0:
    export_cols = ["Assigned_Rep", "Segment", "Account_Name", "Num_Employees",
                   "ARR", "Num_Marketers", "Risk_Score", "Location"]
    export_cols = [c for c in export_cols if c in all_assigned.columns]
    if "Current_Rep" in all_assigned.columns:
        export_cols.insert(2, "Current_Rep")
    export_df = all_assigned[export_cols].sort_values(["Segment", "Assigned_Rep", "ARR"],
                                                       ascending=[True, True, False])
    csv_data = export_df.to_csv(index=False)
    st.download_button(
        label="⬇ Download assignments as CSV",
        data=csv_data,
        file_name=f"territory_assignments_{threshold}.csv",
        mime="text/csv",
    )

# ─── METHODOLOGY ─────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">Methodology</div>', unsafe_allow_html=True)
st.markdown(
    '<p class="section-desc">How the algorithm works and why these design choices are sound.</p>',
    unsafe_allow_html=True,
)

st.markdown(
    f"""
<div class="method-box">
<h3>The Algorithm: Longest Processing Time (LPT) Greedy Heuristic</h3>
<p>
Accounts are sorted by ARR from highest to lowest within each segment. Each account is assigned
to whichever rep currently has the lowest total ARR — the same logic a thoughtful RevOps lead
would use intuitively, formalized as an algorithm.
</p>
<p>
LPT is a well-studied heuristic for the <em>multiprocessor scheduling problem</em>, proven to
produce results within 4/3 of optimal in the worst case. With {len(accounts_df)} accounts across
4–6 reps per segment, actual results are significantly tighter — typically under 1% variation.
</p>
</div>

<div class="method-box" style="border-left-color: #27ae60;">
<h3>Why Balance on ARR?</h3>
<p>
ARR is the closest single proxy for the stakes and complexity of a rep's book. Higher-ARR accounts
typically demand more strategic attention, executive engagement, and complex renewal negotiations.
Balancing by account count alone could leave one rep managing $20M while another manages $8M —
not equitable even with equal account counts.
</p>
</div>

<div class="method-box" style="border-left-color: #f39c12;">
<h3>What the Workload Variation (CV) Tells You</h3>
<p>
Coefficient of variation = standard deviation ÷ mean × 100. It answers: <em>"How far does the
typical rep deviate from the average?"</em> Under 5% is a tight, fair split. Above 10% signals
meaningful imbalance. CV is scale-independent — it works whether total ARR is $1M or $100M.
</p>
</div>

<div class="method-box" style="border-left-color: #e74c3c;">
<h3>Territory Rules — Optional Constraints</h3>
<p>
<strong>Risk weighting:</strong> When enabled, high-risk accounts (75+) are treated as 1.3× their ARR
in the algorithm, and low-risk accounts (1-25) as 0.8×. This spreads high-risk accounts more evenly
because they "weigh" more — reflecting the reality that a high-churn account consumes more rep capacity
than a stable one at the same dollar value.
</p>
<p>
<strong>Churn penalty:</strong> Adds a dollar penalty when the algorithm considers reassigning an account
to a different rep than who manages it today (using the Current_Rep field). A $50K penalty means the algorithm
only reassigns if the ARR balance improvement exceeds $50K — keeping existing relationships intact unless
the imbalance justifies the disruption.
</p>
<p>
<strong>Location bonus:</strong> Subtracts a dollar bonus when a rep and account share a state. This makes
the algorithm prefer co-located assignments — reducing travel costs and enabling in-person relationship
building — while still allowing cross-state assignments when ARR balance requires it.
</p>
</div>

<div class="method-box" style="border-left-color: #9b59b6;">
<h3>Limitations &amp; Future Enhancements</h3>
<p>
<strong>Dynamic rep allocation:</strong> Currently, rep counts per segment are fixed. A production
system would reallocate reps proportional to how much ARR falls in each segment as the threshold moves.
</p>
<p>
<strong>Multi-objective optimization:</strong> The current approach layers penalties and bonuses onto a
single greedy pass. A more sophisticated system could use constraint optimization to jointly minimize
ARR imbalance, risk concentration, geographic distance, and relationship disruption simultaneously.
</p>
</div>
""",
    unsafe_allow_html=True,
)

# ─── ALGORITHM TRACE ─────────────────────────────────────────────────────────
rules_active = []
if use_risk:
    rules_active.append("Risk weighting (high=1.3×, low=0.8×)")
if use_churn and churn_penalty > 0:
    rules_active.append(f"Churn penalty (${churn_penalty:,})")
if use_geo and geo_bonus > 0:
    rules_active.append(f"Location bonus (${geo_bonus:,})")
rules_str = ", ".join(rules_active) if rules_active else "None (pure ARR balancing)"

st.markdown(
    f"""
<div class="method-box" style="border-left-color: #4fc3f7;">
<h3>Live Algorithm Trace (threshold: {threshold:,} employees)</h3>
<p>
<code>
1. Split {len(accounts_df)} accounts → {len(ent_accounts)} Enterprise (≥{threshold:,} emp)
   + {len(mm_accounts)} Mid-Market (&lt;{threshold:,} emp)<br>
2. Enterprise pool: {fmt_arr(ent_accounts["ARR"].sum())} ÷ {len(ent_reps)} reps
   = {fmt_arr(arr_per_ent)} target per rep<br>
3. Mid-Market pool: {fmt_arr(mm_accounts["ARR"].sum())} ÷ {len(mm_reps)} reps
   = {fmt_arr(arr_per_mm)} target per rep<br>
4. Active rules: {rules_str}<br>
5. Sort each pool by ARR descending → assign to rep with lowest adjusted cost<br>
6. Result → Enterprise CV: {ent_cv:.1f}% · Mid-Market CV: {mm_cv:.1f}%
</code>
</p>
</div>
""",
    unsafe_allow_html=True,
)
