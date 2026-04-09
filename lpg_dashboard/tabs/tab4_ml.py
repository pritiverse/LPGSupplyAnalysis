# tabs/tab4_ml.py
"""
Advanced ML Analytics Tab — Restructured for analyst efficiency.

Architecture:
  Layer 1 — Command center (KPI strip, global filter, benchmark table)
  Layer 2 — Primary analysis (consensus heatmap, risk ranking, causal network)
  Layer 3 — Method deep dives (7 tabs: Structural, Temporal, Forecasting,
             Clustering, Survival, Change points, Raw data)
  Layer 4 — Utility (model guide, methodology notes)

All methodology explanations are inside collapsed expanders.
All debug/diagnostic code removed.
Flag convention normalised once at load via helpers.standardize_flags.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
import glob
import warnings

from pathlib import Path
from utils import helpers

warnings.filterwarnings("ignore", category=UserWarning)


# ══════════════════════════════════════════════════════════════════════════════
#  UTILITY FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def load_parquet_folder(folder_path: str) -> pd.DataFrame | None:
    if not os.path.exists(folder_path):
        return None
    parts = glob.glob(os.path.join(folder_path, "part-*.parquet"))
    if not parts:
        return None
    dfs = []
    for f in sorted(parts):
        try:
            dfs.append(pd.read_parquet(f))
        except Exception:
            continue
    return pd.concat(dfs, ignore_index=True) if dfs else None


def fcol(df, *names) -> str | None:
    if df is None or (isinstance(df, pd.DataFrame) and df.empty):
        return None
    for n in names:
        if n in df.columns:
            return n
    return None


def ok(d) -> bool:
    return d is not None and not (isinstance(d, pd.DataFrame) and d.empty)


def anomaly_flag_count(df, *cols) -> int | None:
    for c in cols:
        if ok(df) and c in df.columns:
            v = df[c]
            if (v == -1).any():
                return int((v == -1).sum())
            if (v == 1).any():
                return int((v == 1).sum())
    return None


def get_countries(df) -> list[str]:
    c = fcol(df, "country", "Country", "COUNTRY")
    return sorted(df[c].dropna().unique().tolist()) if c else []


def filter_df(df: pd.DataFrame, selected: list, search: str) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    c = fcol(df, "country", "Country", "COUNTRY")
    if c is None:
        return df
    mask = pd.Series([True] * len(df), index=df.index)
    if search:
        mask &= df[c].str.contains(search, case=False, na=False)
    if selected and "All Countries" not in selected:
        mask &= df[c].isin(selected)
    return df[mask]


def get_supply_col(df: pd.DataFrame) -> str | None:
    for col in ("supply", "total_supply", "value", "supply_volume", "supply_mt"):
        if col in df.columns:
            return col
    return None


def country_controls(key: str, all_c: list, default_top: int = 10):
    c1, c2, c3, c4 = st.columns([2, 3, 1, 1])
    with c1:
        srch = st.text_input("Search", key=f"{key}_s",
                             placeholder="🔍 Country…", label_visibility="collapsed")
    with c2:
        sel = st.multiselect("Filter", ["All Countries"] + all_c,
                             default=["All Countries"], key=f"{key}_m",
                             label_visibility="collapsed", placeholder="Select countries…")
    with c3:
        top = st.number_input("Top N", 5, max(len(all_c), 50) if all_c else 50,
                              default_top, 5, key=f"{key}_n", label_visibility="collapsed")
    with c4:
        show_all = st.toggle("All", key=f"{key}_a", value=False)
    return srch, sel, int(top), show_all


# ══════════════════════════════════════════════════════════════════════════════
#  THEME
# ══════════════════════════════════════════════════════════════════════════════

T = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="'Sora', sans-serif", color="#94a3b8", size=11),
    margin=dict(l=8, r=8, t=44, b=8),
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=10)),
    coloraxis_colorbar=dict(tickfont=dict(size=9), thickness=10),
)

TEAL    = "#22d3ee"
ROSE    = "#fb7185"
VIOLET  = "#a78bfa"
AMBER   = "#fbbf24"
EMERALD = "#34d399"
SLATE   = "#64748b"
BLUE    = "#60a5fa"
PINK    = "#f472b6"
ACCENT_SEQ = [TEAL, ROSE, VIOLET, AMBER, EMERALD, BLUE, PINK, "#4ade80"]

KPI_CSS = """
<style>
.kpi-grid{display:grid;grid-template-columns:repeat(5,1fr);gap:.8rem;margin-bottom:1.2rem;}
.kpi{background:rgba(15,23,42,.6);border:1px solid #1e2d40;border-radius:14px;
  padding:1rem .9rem .8rem;position:relative;overflow:hidden;transition:border-color .2s;}
.kpi:hover{border-color:#22d3ee;}
.kpi::before{content:'';position:absolute;top:0;left:0;right:0;height:2px;border-radius:14px 14px 0 0;}
.kpi.c0::before{background:linear-gradient(90deg,#22d3ee,transparent);}
.kpi.c1::before{background:linear-gradient(90deg,#fb7185,transparent);}
.kpi.c2::before{background:linear-gradient(90deg,#a78bfa,transparent);}
.kpi.c3::before{background:linear-gradient(90deg,#fbbf24,transparent);}
.kpi.c4::before{background:linear-gradient(90deg,#34d399,transparent);}
.kpi-label{font-size:.58rem;letter-spacing:.15em;color:#475569;text-transform:uppercase;margin-bottom:.4rem;}
.kpi-val{font-size:1.6rem;font-weight:700;color:#e2e8f0;line-height:1;margin-bottom:.2rem;}
.kpi-sub{font-size:.68rem;color:#475569;}
</style>
"""


def kpi_html(cc, label, value, sub) -> str:
    v = f"{value:,}" if isinstance(value, int) else (str(value) if value is not None else "—")
    return (f'<div class="kpi {cc}"><div class="kpi-label">{label}</div>'
            f'<div class="kpi-val">{v}</div><div class="kpi-sub">{sub}</div></div>')


def empty_state(msg="Checkpoint not found. Run: python pipeline.py --full --ml"):
    st.info(f"📭 {msg}")


# ══════════════════════════════════════════════════════════════════════════════
#  SHARED CHART HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def plot_ts_anomalies(df, date_col, value_col, flag_col, title, color):
    if not all([date_col, value_col, flag_col]):
        return None
    if not all(c in df.columns for c in [date_col, value_col]):
        return None
    fig = go.Figure()
    # Detect anomaly convention
    anom_val = 1 if (df[flag_col] == 1).any() else -1
    normal = df[df[flag_col] != anom_val] if flag_col in df.columns else df
    fig.add_trace(go.Scatter(x=normal[date_col], y=normal[value_col],
                             mode="lines+markers", name="Supply",
                             line=dict(color=color, width=2),
                             marker=dict(size=4, color=color)))
    if flag_col in df.columns:
        anom = df[df[flag_col] == anom_val]
        if not anom.empty:
            fig.add_trace(go.Scatter(x=anom[date_col], y=anom[value_col],
                                     mode="markers", name="Anomaly",
                                     marker=dict(symbol="x", size=10, color=ROSE,
                                                 line=dict(width=1, color="white"))))
    fig.update_layout(title=title, height=320, **T,
                      xaxis_title="Year", yaxis_title="Supply",
                      hovermode="x unified")
    return fig


def show_structural_model(df_model, model_name, score_col, flag_col, color_scale, accent_color):
    if not ok(df_model):
        empty_state(f"{model_name} checkpoint not found.")
        return
    ctrl_s, ctrl_sel, ctrl_top, ctrl_all = country_controls(
        model_name.lower().replace(" ", "_"), get_countries(df_model))
    df_f = filter_df(df_model, ctrl_sel, ctrl_s)
    c_col = fcol(df_f, "country", "Country")
    if not c_col:
        st.warning(f"No country column in {model_name}.")
        return

    anom_val = 1 if (flag_col and flag_col in df_f.columns and
                     (df_f[flag_col] == 1).any()) else -1
    if flag_col and flag_col in df_f.columns:
        df_anom = df_f[df_f[flag_col] == anom_val]
    else:
        df_anom = df_f

    if score_col and score_col in df_anom.columns:
        n = ctrl_top if not ctrl_all else len(df_anom)
        df_top = df_anom.nlargest(n, score_col)
    else:
        df_top = df_anom if ctrl_all else df_anom.head(ctrl_top)

    col1, col2 = st.columns([3, 2])
    with col1:
        if score_col and score_col in df_top.columns:
            fig = px.bar(df_top.sort_values(score_col, ascending=True),
                         x=score_col, y=c_col, orientation="h",
                         color=score_col, color_continuous_scale=color_scale,
                         title=f"<b>{model_name} — Anomaly Scores</b>")
            fig.update_layout(height=400, **{**T, "margin": dict(l=120)})
            st.plotly_chart(fig, use_container_width=True)
        else:
            if ok(df_anom) and c_col in df_anom.columns:
                cnt = df_anom[c_col].value_counts().reset_index()
                cnt.columns = [c_col, "Anomaly Count"]
                cnt = cnt if ctrl_all else cnt.head(ctrl_top)
                fig = px.bar(cnt, x="Anomaly Count", y=c_col, orientation="h",
                             color="Anomaly Count", color_continuous_scale=color_scale,
                             title=f"<b>{model_name} — Anomaly Count</b>")
                fig.update_layout(height=400, **T)
                st.plotly_chart(fig, use_container_width=True)
    with col2:
        display_cols = [c for c in [c_col, score_col, flag_col]
                        if c and c in df_top.columns]
        if not display_cols:
            display_cols = list(df_top.columns[:3])
        st.dataframe(df_top[display_cols], use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN RENDER
# ══════════════════════════════════════════════════════════════════════════════

def render(data: dict, df_sev, filters):
    try:
        from utils.helpers import section_tag
        section_tag("MACHINE LEARNING & ADVANCED ANALYTICS")
    except ImportError:
        st.markdown("## Machine Learning & Advanced Analytics")

    # ── Load all checkpoints ──────────────────────────────────────────────
    iso_forest  = helpers.safe_get_data(data, "anomaly_isolation_forest")
    lof         = helpers.safe_get_data(data, "anomaly_lof")
    ocsvm_df    = helpers.safe_get_data(data, "anomaly_ocsvm")
    comp_df     = helpers.safe_get_data(data, "anomaly_composite")
    iqr_df      = helpers.safe_get_data(data, "anomaly_iqr")
    percentile  = helpers.safe_get_data(data, "anomaly_percentile")
    mov_avg     = helpers.safe_get_data(data, "anomaly_moving_avg_robust")
    prophet     = helpers.safe_get_data(data, "anomaly_prophet")
    prophet     = prophet if ok(prophet) else helpers.safe_get_data(data, "prophet_anomalies")
    lstm        = helpers.safe_get_data(data, "anomaly_lstm")
    adash       = helpers.safe_get_data(data, "anomaly_dashboard")
    combined_yr = helpers.safe_get_data(data, "combined_yearly_anomalies")
    combined_co = helpers.safe_get_data(data, "combined_country_anomalies")
    clusters    = helpers.safe_get_data(data, "clusters")
    dbscan_df   = helpers.safe_get_data(data, "dbscan_results")
    temporal_cl = helpers.safe_get_data(data, "temporal_clusters")
    pca_feat    = helpers.safe_get_data(data, "country_pca_features")
    arima       = helpers.safe_get_data(data, "arima_forecast_results")
    change_pts  = helpers.safe_get_data(data, "change_points")
    rf_pred     = helpers.safe_get_data(data, "rf_predictions")
    rf_meta     = helpers.safe_get_data(data, "rf_model_metadata")
    granger     = helpers.safe_get_data(data, "granger_causality_results")
    cox_coef    = helpers.safe_get_data(data, "coxph_coefficients")
    cox_prep    = helpers.safe_get_data(data, "coxph_prepared_data")
    net_metrics = helpers.safe_get_data(data, "network_metrics")
    net_comm    = helpers.safe_get_data(data, "network_communities")
    shock_score = helpers.safe_get_data(data, "shock_score")
    shock_valid = helpers.safe_get_data(data, "shock_validation_advanced")
    contagion   = helpers.safe_get_data(data, "contagion_features")
    features_df = helpers.safe_get_data(data, "features_timeseries")
    features_df = features_df if ok(features_df) else helpers.safe_get_data(data, "04_features")
    yearly_composite = helpers.safe_get_data(data, "yearly_composite")

    # ── Normalise flag conventions once ──────────────────────────────────
    ewma_raw = helpers.safe_get_data(data, "anomaly_ewma")
    ewma = helpers.standardize_flags(ewma_raw, ["ewma_anomaly_flag", "shock_flag"]) \
        if ewma_raw is not None else None

    # ── Derived KPI values ────────────────────────────────────────────────
    if_n      = anomaly_flag_count(iso_forest, "if_anomaly_flag", "anomaly")
    lof_n     = anomaly_flag_count(lof,        "lof_anomaly_flag", "anomaly")
    lstm_n    = anomaly_flag_count(lstm,        "lstm_anomaly_flag", "anomaly", "is_anomaly")
    prop_n    = anomaly_flag_count(prophet,     "prophet_anomaly_flag", "anomaly", "is_anomaly")
    ewma_n    = anomaly_flag_count(ewma,        "ewma_anomaly_flag_std", "ewma_anomaly_flag", "shock_flag")
    cp_n      = len(change_pts) if ok(change_pts) else None
    n_clusters = (clusters["cluster_label"].nunique()
                  if ok(clusters) and "cluster_label" in clusters.columns else None)

    caus_col  = fcol(granger, "is_causal_1_to_2", "is_causal_2_to_1", "is_causal")
    granger_n = (int(granger[granger[caus_col]].shape[0])
                 if ok(granger) and caus_col else None)

    auc_val = None
    if ok(rf_pred) and "auc" in rf_pred.columns:
        v = rf_pred["auc"].dropna()
        if not v.empty:
            auc_val = f"{v.iloc[0]:.0%}"

    tc_switches = None
    if ok(temporal_cl) and "cluster_switch" in temporal_cl.columns:
        tc_switches = int(temporal_cl["cluster_switch"].sum())

    # ── Global country list ───────────────────────────────────────────────
    df_for_countries = next(
        (d for d in [features_df, ewma, iso_forest, clusters] if ok(d)), None)
    all_countries = get_countries(df_for_countries)

    # ══════════════════════════════════════════════════════════════════════
    # LAYER 1 — COMMAND CENTER
    # ══════════════════════════════════════════════════════════════════════

    st.caption("LPG Supply Chain · Global Risk Intelligence · 1990–2024 · 172 countries · 15+ ML models")

    # KPI strip
    st.markdown(KPI_CSS, unsafe_allow_html=True)
    row = "".join([
        kpi_html("c0", "Detection Methods",  10,         "Ensemble anomaly layers"),
        kpi_html("c1", "LSTM Anomalies",      lstm_n,     "Deep learning flags"),
        kpi_html("c2", "Causal Links",        granger_n,  "Granger significant pairs"),
        kpi_html("c3", "Regime Shifts",       cp_n,       "Structural break-points"),
        kpi_html("c4", "Country Clusters",    n_clusters, "KMeans archetypes"),
    ])
    st.markdown(f'<div class="kpi-grid">{row}</div>', unsafe_allow_html=True)

    # Global filter
    with st.expander("🌍 Global filter — applies to all sections", expanded=False):
        gf1, gf2, gf3, gf4 = st.columns([2, 3, 1, 1])
        with gf1:
            g_srch = st.text_input("Search", key="g_s",
                                   placeholder="🔍 Country…", label_visibility="collapsed")
        with gf2:
            g_sel = st.multiselect("Countries", ["All Countries"] + all_countries,
                                   default=["All Countries"], key="g_m",
                                   label_visibility="collapsed")
        with gf3:
            g_top = st.number_input("Top N", 5, 50, 15, 5, key="g_n",
                                    label_visibility="collapsed")
        with gf4:
            g_yr_min, g_yr_max = 1990, 2024
            for d in [features_df, ewma, combined_yr]:
                yc = fcol(d, "year", "Year")
                if yc and ok(d):
                    g_yr_min = int(d[yc].min())
                    g_yr_max = int(d[yc].max())
                    break
            g_yr = st.select_slider("Year range", options=list(range(g_yr_min, g_yr_max + 1)),
                                    value=(g_yr_min, g_yr_max), key="g_yr",
                                    label_visibility="collapsed")

    GF = {"srch": g_srch, "sel": g_sel, "top": int(g_top), "yr": g_yr}

    def gf(df):
        if not ok(df):
            return df
        d = filter_df(df, GF["sel"], GF["srch"])
        yc = fcol(d, "year", "Year")
        if yc and GF["yr"]:
            d = d[(d[yc] >= GF["yr"][0]) & (d[yc] <= GF["yr"][1])]
        return d

    # Benchmark status table
    rows = [
        ("Isolation Forest",     "Ensemble Tree",     "Structural",  ok(iso_forest), if_n),
        ("Local Outlier Factor", "Density",           "Structural",  ok(lof),        lof_n),
        ("One-Class SVM",        "Kernel SVM",        "Structural",  ok(ocsvm_df),   anomaly_flag_count(ocsvm_df, "ocsvm_anomaly_flag", "anomaly")),
        ("EWMA",                 "Statistical",       "Temporal",    ok(ewma),       ewma_n),
        ("IQR",                  "Non-parametric",    "Temporal",    ok(iqr_df),     None),
        ("Percentile",           "Non-parametric",    "Temporal",    ok(percentile), None),
        ("Moving Avg Robust",    "Statistical",       "Temporal",    ok(mov_avg),    None),
        ("Prophet",              "Bayesian",          "Temporal",    ok(prophet),    prop_n),
        ("LSTM Autoencoder",     "Deep Learning",     "Temporal",    ok(lstm),       lstm_n),
        ("KMeans",               "Clustering",        "Clustering",  ok(clusters),   None),
        ("DBSCAN",               "Clustering",        "Clustering",  ok(dbscan_df),  None),
        ("Random Forest",        "Ensemble ML",       "Predictive",  ok(rf_pred),    None),
        ("Granger Causality",    "Causal Inference",  "Network",     ok(granger),    granger_n),
        ("Cox PH Survival",      "Survival Model",    "Risk",        ok(cox_coef),   None),
        ("ARIMA",                "Time-Series",       "Forecasting", ok(arima),      None),
    ]
    df_bench = pd.DataFrame(rows, columns=["Model", "Family", "Category", "Available", "Detections"])
    df_bench["Status"]     = df_bench["Available"].map({True: "✅ Loaded", False: "⬜ Missing"})
    df_bench["Detections"] = df_bench["Detections"].apply(
        lambda x: f"{x:,}" if isinstance(x, int) else "—")
    st.dataframe(
        df_bench[["Model", "Family", "Category", "Detections", "Status"]],
        use_container_width=True, hide_index=True,
        column_config={
            "Model":      st.column_config.TextColumn(width="medium"),
            "Family":     st.column_config.TextColumn(width="medium"),
            "Category":   st.column_config.TextColumn(width="medium"),
            "Detections": st.column_config.TextColumn(width="small"),
            "Status":     st.column_config.TextColumn(width="small"),
        })

    st.divider()

    # ══════════════════════════════════════════════════════════════════════
    # LAYER 2 — PRIMARY ANALYSIS
    # ══════════════════════════════════════════════════════════════════════

    col1, col2, col3 = st.columns([1, 1, 1])

    # ── Col 1: Consensus anomaly heatmap ─────────────────────────────────
    with col1:
        st.markdown("#### Where are the anomalies?")
        src_cy = next((d for d in [combined_yr, adash, yearly_composite] if ok(d)), None)
        if ok(src_cy):
            c_col   = fcol(src_cy, "country", "Country")
            yr_col  = fcol(src_cy, "year", "Year")
            risk_col = fcol(src_cy,
                            "composite_yearly_anomaly_score", "composite_score", "risk_score",
                            "total_flags", "anomaly_count", "score", "total_anomalies")
            if c_col and yr_col and risk_col:
                df_cy = gf(src_cy)
                top25 = (df_cy.groupby(c_col)[risk_col].sum()
                         .nlargest(25).index.tolist())
                heat_df = df_cy[df_cy[c_col].isin(top25)]
                pivot = heat_df.pivot_table(index=c_col, columns=yr_col,
                                            values=risk_col, aggfunc="mean")
                if not pivot.empty:
                    fig = px.imshow(pivot, color_continuous_scale="Reds",
                                    aspect="auto",
                                    title="<b>Consensus risk — country × year</b>",
                                    labels={"color": "Risk score"})
                    fig.update_layout(height=480, **T)
                    st.plotly_chart(fig, use_container_width=True)

                high_risk   = int((df_cy[risk_col] >= 4).sum()) if risk_col in df_cy.columns else 0
                medium_risk = int(((df_cy[risk_col] >= 2) & (df_cy[risk_col] < 4)).sum()) \
                              if risk_col in df_cy.columns else 0
                m1, m2 = st.columns(2)
                m1.metric("High risk events (score ≥ 4)", high_risk)
                m2.metric("Medium risk events (score 2–3)", medium_risk)
            else:
                empty_state("Risk score column not found in combined anomaly data.")
        else:
            empty_state("Combined yearly anomaly checkpoint not found.")

    # ── Col 2: Country risk ranking ───────────────────────────────────────
    with col2:
        st.markdown("#### Which countries are highest risk?")
        rank_parts = []

        # RF probability
        if ok(rf_pred):
            c_col_rf = fcol(rf_pred, "country", "Country")
            prob_col  = fcol(rf_pred, "probability", "shock_probability", "prob")
            if c_col_rf and prob_col:
                rf_agg = (rf_pred.groupby(c_col_rf)[prob_col].mean()
                          .reset_index().rename(columns={c_col_rf: "Country",
                                                         prob_col: "Shock Prob (avg)"}))
                rank_parts.append(rf_agg)

        # Anomaly count from combined country
        if ok(combined_co):
            c_col_co  = fcol(combined_co, "country", "Country")
            cnt_col   = fcol(combined_co, "total_anomaly_count", "anomaly_count",
                             "total_flags", "count")
            if c_col_co and cnt_col:
                co_agg = combined_co[[c_col_co, cnt_col]].rename(
                    columns={c_col_co: "Country", cnt_col: "Total Anomaly Events"})
                rank_parts.append(co_agg)

        # Cluster label
        if ok(clusters):
            c_col_cl = fcol(clusters, "country", "Country")
            lbl_col  = fcol(clusters, "cluster_label", "cluster", "label")
            if c_col_cl and lbl_col:
                cl_agg = clusters[[c_col_cl, lbl_col]].drop_duplicates().rename(
                    columns={c_col_cl: "Country", lbl_col: "Cluster"})
                rank_parts.append(cl_agg)

        if rank_parts:
            merged = rank_parts[0]
            for part in rank_parts[1:]:
                merged = merged.merge(part, on="Country", how="outer")

            # Apply global country filter
            if GF["sel"] and "All Countries" not in GF["sel"]:
                merged = merged[merged["Country"].isin(GF["sel"])]
            if GF["srch"]:
                merged = merged[merged["Country"].str.contains(
                    GF["srch"], case=False, na=False)]

            sort_col = "Shock Prob (avg)" if "Shock Prob (avg)" in merged.columns \
                       else "Total Anomaly Events" if "Total Anomaly Events" in merged.columns \
                       else merged.columns[-1]
            merged = merged.sort_values(sort_col, ascending=False).head(GF["top"])

            if "Shock Prob (avg)" in merged.columns:
                merged["Shock Prob (avg)"] = merged["Shock Prob (avg)"].apply(
                    lambda v: f"{v:.1%}" if pd.notna(v) else "—")

            st.dataframe(merged.fillna("—"), use_container_width=True, hide_index=True)
            st.caption("RF probability = next-year shock likelihood. "
                       "Anomaly events = consensus flag count 1990–2024.")
        else:
            empty_state("No ranking data available. Need rf_predictions or combined_country_anomalies.")

    # ── Col 3: Causal network summary ─────────────────────────────────────
    with col3:
        st.markdown("#### Who influences whom?")
        if ok(granger) and caus_col:
            causal_df = granger[granger[caus_col] == True].copy()  # noqa: E712
            src_col   = fcol(causal_df, "source", "country_1", "country_x", "country")
            tgt_col   = fcol(causal_df, "target", "country_2", "country_y")

            # Apply search/select to source
            if GF["srch"] and src_col:
                causal_df = causal_df[causal_df[src_col].str.contains(
                    GF["srch"], case=False, na=False)]
            if GF["sel"] and "All Countries" not in GF["sel"] and src_col:
                causal_df = causal_df[causal_df[src_col].isin(GF["sel"])]

            if src_col:
                hubs = (causal_df.groupby(src_col).size()
                        .sort_values(ascending=False).head(GF["top"])
                        .reset_index(name="Outgoing"))
                fig = go.Figure(go.Bar(
                    x=hubs["Outgoing"], y=hubs[src_col], orientation="h",
                    marker=dict(color=hubs["Outgoing"],
                                colorscale=[[0, "#0d2a2a"], [1, TEAL]],
                                showscale=False),
                    text=hubs["Outgoing"], textposition="outside",
                    textfont=dict(size=9, color="#94a3b8")))
                fig.update_layout(title="<b>Top causal hubs</b>", height=220, **T,
                                  xaxis_title="# influenced")
                st.plotly_chart(fig, use_container_width=True)

            if tgt_col:
                recv = (granger[granger[caus_col] == True]
                        .groupby(tgt_col).size()
                        .sort_values(ascending=False).head(GF["top"])
                        .reset_index(name="Incoming"))
                fig2 = go.Figure(go.Bar(
                    x=recv["Incoming"], y=recv[tgt_col], orientation="h",
                    marker=dict(color=recv["Incoming"],
                                colorscale=[[0, "#1a0d2e"], [1, VIOLET]],
                                showscale=False),
                    text=recv["Incoming"], textposition="outside",
                    textfont=dict(size=9, color="#94a3b8")))
                fig2.update_layout(title="<b>Most influenced</b>", height=220, **T,
                                   xaxis_title="# influencers")
                st.plotly_chart(fig2, use_container_width=True)

            # Network summary metrics
            if ok(net_metrics):
                pr_col  = fcol(net_metrics, "pagerank", "page_rank")
                btw_col = fcol(net_metrics, "betweenness_centrality", "betweenness")
                c_nm    = fcol(net_metrics, "country", "Country")
                if pr_col and c_nm:
                    top_pr  = net_metrics.nlargest(1, pr_col)[c_nm].iloc[0]
                    top_btw = (net_metrics.nlargest(1, btw_col)[c_nm].iloc[0]
                               if btw_col else "—")
                    total_edges = len(granger[granger[caus_col] == True])
                    n1, n2, n3 = st.columns(3)
                    n1.metric("Top PageRank", top_pr)
                    n2.metric("Top Betweenness", top_btw)
                    n3.metric("Causal edges", f"{total_edges:,}")
        else:
            empty_state("Granger causality checkpoint not found.")

    st.divider()

    # ══════════════════════════════════════════════════════════════════════
    # LAYER 3 — METHOD DEEP DIVES
    # ══════════════════════════════════════════════════════════════════════

    st.markdown("### Method deep dives")

    (tab_struct, tab_temp, tab_forecast,
     tab_cluster, tab_survival, tab_chgpt, tab_raw) = st.tabs([
        "🌲 Structural", "📈 Temporal", "📊 Forecasting",
        "🏷️ Clustering", "⏱️ Survival", "📐 Change points", "📁 Raw data",
    ])

    # ──────────────────────────────────────────────────────────────────────
    # TAB: STRUCTURAL
    # ──────────────────────────────────────────────────────────────────────
    with tab_struct:
        with st.expander("How structural models work", expanded=False):
            st.markdown("""
**Isolation Forest** isolates anomalies by randomly splitting features — countries that require
fewer splits are far from the majority in multi-dimensional space (import dependency, volatility,
growth, trade patterns). Fast and scalable.

**Local Outlier Factor** compares each country's local density to its k-nearest neighbours.
Flags countries that are anomalous *within their peer group* — finding emerging-market outliers
that look fine globally.

**One-Class SVM** learns a non-linear boundary enclosing the "normal" supply feature space.
Countries outside this boundary have rare feature combinations. The RBF kernel captures
curved interactions that linear methods miss.

**Composite score** averages the three — countries flagged by all three have the highest
confidence of being genuine structural outliers.
            """)

        st_if, st_lof, st_svm, st_cons = st.tabs([
            "Isolation Forest", "LOF", "One-Class SVM", "Consensus"])

        with st_if:
            show_structural_model(
                gf(iso_forest), "Isolation Forest",
                fcol(iso_forest, "if_anomaly_score", "anomaly_score", "score"),
                fcol(iso_forest, "if_anomaly_flag", "anomaly"),
                "Blues_r", TEAL)

        with st_lof:
            show_structural_model(
                gf(lof), "Local Outlier Factor",
                fcol(lof, "lof_anomaly_score", "anomaly_score", "score"),
                fcol(lof, "lof_anomaly_flag", "anomaly"),
                "Oranges_r", ROSE)

        with st_svm:
            show_structural_model(
                gf(ocsvm_df), "One-Class SVM",
                fcol(ocsvm_df, "ocsvm_anomaly_score", "anomaly_score", "score"),
                fcol(ocsvm_df, "ocsvm_anomaly_flag", "anomaly"),
                "Purples_r", VIOLET)

        with st_cons:
            show_structural_model(
                gf(comp_df), "Composite Structural Score",
                fcol(comp_df, "composite_score_avg", "composite_score", "score"),
                fcol(comp_df, "composite_flag", "flag"),
                "Reds_r", AMBER)

            # High-confidence table — flagged by all 3
            if ok(iso_forest) and ok(lof) and ok(ocsvm_df):
                c_if  = fcol(iso_forest, "country", "Country")
                c_lof = fcol(lof,        "country", "Country")
                c_sv  = fcol(ocsvm_df,   "country", "Country")
                f_if  = fcol(iso_forest, "if_anomaly_flag",   "anomaly")
                f_lof = fcol(lof,        "lof_anomaly_flag",  "anomaly")
                f_sv  = fcol(ocsvm_df,   "ocsvm_anomaly_flag","anomaly")

                if all([c_if, c_lof, c_sv, f_if, f_lof, f_sv]):
                    def get_flagged_countries(df, c, f):
                        anom_val = 1 if (df[f] == 1).any() else -1
                        return set(df[df[f] == anom_val][c].dropna().unique())

                    flagged_all = (get_flagged_countries(iso_forest, c_if, f_if) &
                                   get_flagged_countries(lof, c_lof, f_lof) &
                                   get_flagged_countries(ocsvm_df, c_sv, f_sv))
                    if flagged_all:
                        st.markdown("**High-confidence anomalies — flagged by all 3 structural methods**")
                        st.dataframe(
                            pd.DataFrame(sorted(flagged_all), columns=["Country"]),
                            use_container_width=True, hide_index=True)

    # ──────────────────────────────────────────────────────────────────────
    # TAB: TEMPORAL
    # ──────────────────────────────────────────────────────────────────────
    with tab_temp:
        with st.expander("How temporal models work", expanded=False):
            st.markdown("""
**EWMA** (Exponentially Weighted Moving Average) gives exponentially decreasing weight to older
observations and flags years where actual supply deviates significantly from the smoothed trend.
Best for short-duration infrastructure shocks.

**IQR** flags country-years outside the interquartile range — non-parametric, no Gaussian
assumption, best as a first-pass filter.

**Percentile** flags years below a country's own historical 5th percentile or above its 95th —
country-relative rather than global, so small stable nations are held to their own standard.

**Robust Moving Average** uses median-based rolling windows and classifies shocks by severity
(Normal / Moderate / High / Extreme) and direction (surge / collapse).

**Prophet** fits piecewise linear trend + Fourier seasonality. Flags country-years outside the
95% prediction interval. Handles missing data and irregular patterns well.

**LSTM Autoencoder** learns to compress and reconstruct supply time-series. High reconstruction
error means the model could not explain the pattern — captures non-linear temporal disruptions
that statistical methods miss entirely.
            """)

        # Shared country selector for all temporal sub-tabs
        all_temp = get_countries(ewma if ok(ewma) else None or prophet if ok(prophet) else None or lstm if ok(lstm) else None or iqr_df if ok(iqr_df) else None)
        sel_temp = st.selectbox("Focus country (applies to all temporal sub-tabs)",
                                ["All"] + all_temp, key="temp_country_global")

        def temp_filter(df):
            d = gf(df)
            if not ok(d):
                return d
            c = fcol(d, "country", "Country")
            if sel_temp != "All" and c:
                d = d[d[c] == sel_temp]
            return d

        t_ewma, t_iqr_pct, t_ma, t_prop, t_lstm = st.tabs([
            "EWMA", "IQR + Percentile", "Moving Avg", "Prophet", "LSTM"])

        # ── EWMA ─────────────────────────────────────────────────────────
        with t_ewma:
            if not ok(ewma):
                empty_state("EWMA checkpoint not found.")
            else:
                df_e  = temp_filter(ewma)
                c_col = fcol(df_e, "country", "Country")
                yr_col = fcol(df_e, "year", "Year")
                flag_col = fcol(df_e,
                                "ewma_anomaly_flag_std", "ewma_anomaly_flag",
                                "shock_flag", "anomaly", "flag")
                supply_col = get_supply_col(df_e)

                ctrl_s, ctrl_sel, ctrl_top, ctrl_all = country_controls(
                    "ewma", get_countries(df_e))
                df_e = filter_df(df_e, ctrl_sel, ctrl_s)

                if not df_e.empty:
                    col1, col2 = st.columns([3, 2])
                    with col1:
                        if supply_col and yr_col and flag_col:
                            fig = plot_ts_anomalies(
                                df_e, yr_col, supply_col, flag_col,
                                f"<b>EWMA: {sel_temp if sel_temp != 'All' else 'All Countries'}</b>",
                                TEAL)
                            if fig:
                                st.plotly_chart(fig, use_container_width=True)

                        if yr_col and flag_col and flag_col in df_e.columns:
                            anom_val = 1 if (df_e[flag_col] == 1).any() else -1
                            yr_agg = ((df_e[flag_col] == anom_val)
                                      .groupby(df_e[yr_col]).sum()
                                      .reset_index())
                            yr_agg.columns = ["year", "anomalies"]
                            if yr_agg["anomalies"].sum() > 0:
                                fig2 = go.Figure(go.Bar(
                                    x=yr_agg["year"], y=yr_agg["anomalies"],
                                    marker_color=TEAL))
                                fig2.update_layout(
                                    title="<b>Anomalies per year</b>", height=240, **T)
                                st.plotly_chart(fig2, use_container_width=True)

                    with col2:
                        if c_col and flag_col and flag_col in df_e.columns:
                            anom_val = 1 if (df_e[flag_col] == 1).any() else -1
                            anom_df = df_e[df_e[flag_col] == anom_val]
                            if not anom_df.empty:
                                top_e = (anom_df.groupby(c_col).size()
                                         .sort_values(ascending=False)
                                         .head(ctrl_top if not ctrl_all else 9999)
                                         .reset_index(name="Anomaly Years"))
                                st.dataframe(
                                    top_e.rename(columns={c_col: "Country"}),
                                    use_container_width=True, hide_index=True)
                            else:
                                st.info("No anomaly rows found with current filter.")

        # ── IQR + Percentile ──────────────────────────────────────────────
        with t_iqr_pct:
            iqr_col1, pct_col2 = st.columns(2)

            with iqr_col1:
                st.markdown("**IQR**")
                if not ok(iqr_df):
                    empty_state("IQR checkpoint not found.")
                else:
                    df_iq = temp_filter(iqr_df)
                    c_col_iq = fcol(df_iq, "country", "Country")
                    yr_col_iq = fcol(df_iq, "year", "Year")
                    sc_iq = fcol(df_iq, "iqr_anomaly_score", "anomaly_score", "score")
                    sup_iq = get_supply_col(df_iq)

                    _, ctrl_sel_iq, ctrl_top_iq, ctrl_all_iq = country_controls(
                        "iqr", get_countries(df_iq))
                    df_iq = filter_df(df_iq, ctrl_sel_iq, "")

                    if not df_iq.empty:
                        if sup_iq and yr_col_iq:
                            fig = px.scatter(df_iq, x=yr_col_iq, y=sup_iq,
                                             color=sc_iq if sc_iq else None,
                                             color_continuous_scale="Viridis",
                                             title="<b>IQR: Supply with anomaly score</b>")
                            fig.update_layout(height=300, **T)
                            st.plotly_chart(fig, use_container_width=True)
                        if sc_iq and c_col_iq and sc_iq in df_iq.columns:
                            n = ctrl_top_iq if not ctrl_all_iq else len(df_iq)
                            top_iq = df_iq.nlargest(n, sc_iq)
                            st.dataframe(
                                top_iq[[c_col_iq, sc_iq]].rename(
                                    columns={c_col_iq: "Country", sc_iq: "IQR Score"}),
                                use_container_width=True, hide_index=True)

            with pct_col2:
                st.markdown("**Percentile (p5 / p95)**")
                if not ok(percentile):
                    empty_state("Percentile checkpoint not found.")
                else:
                    df_p = temp_filter(percentile)
                    c_col_p  = fcol(df_p, "country", "Country")
                    yr_col_p = fcol(df_p, "year", "Year")
                    sev_col  = fcol(df_p, "anomaly_severity", "severity")
                    sc_p     = fcol(df_p, "pct_anomaly_score", "score")

                    _, ctrl_sel_p, ctrl_top_p, ctrl_all_p = country_controls(
                        "pct", get_countries(df_p))
                    df_p = filter_df(df_p, ctrl_sel_p, "")

                    if not df_p.empty:
                        if sev_col and sev_col in df_p.columns and \
                                pd.api.types.is_object_dtype(df_p[sev_col]):
                            vc = df_p[sev_col].value_counts()
                            if not vc.empty:
                                fig = px.pie(values=vc.values, names=vc.index, hole=0.55,
                                             color_discrete_sequence=ACCENT_SEQ,
                                             title="<b>Severity distribution</b>")
                                fig.update_layout(height=280, **T)
                                st.plotly_chart(fig, use_container_width=True)
                        if c_col_p and sc_p and sc_p in df_p.columns:
                            n = ctrl_top_p if not ctrl_all_p else len(df_p)
                            top_p = df_p.nlargest(n, sc_p)
                            show_c = [x for x in [c_col_p, yr_col_p, sev_col, sc_p]
                                      if x and x in top_p.columns]
                            st.dataframe(top_p[show_c],
                                         use_container_width=True, hide_index=True)

        # ── Moving Avg ────────────────────────────────────────────────────
        with t_ma:
            if not ok(mov_avg):
                empty_state("Moving Average checkpoint not found.")
            else:
                df_ma = temp_filter(mov_avg)
                c_col_ma = fcol(df_ma, "country", "Country")
                yr_col_ma = fcol(df_ma, "year", "Year")
                sev_col_ma = fcol(df_ma, "ma_severity", "severity", "anomaly_severity")
                dir_col_ma = fcol(df_ma, "anomaly_direction", "direction")
                sup_ma = get_supply_col(df_ma)

                _, ctrl_sel_ma, ctrl_top_ma, ctrl_all_ma = country_controls(
                    "ma", get_countries(df_ma))
                df_ma = filter_df(df_ma, ctrl_sel_ma, "")

                if not df_ma.empty:
                    ma1, ma2 = st.columns(2)
                    with ma1:
                        if sev_col_ma and sev_col_ma in df_ma.columns:
                            vc = df_ma[sev_col_ma].value_counts()
                            if not vc.empty:
                                fig = px.pie(values=vc.values, names=vc.index, hole=0.55,
                                             color_discrete_sequence=ACCENT_SEQ,
                                             title="<b>Severity distribution</b>")
                                fig.update_layout(height=260, **T)
                                st.plotly_chart(fig, use_container_width=True)
                    with ma2:
                        if dir_col_ma and dir_col_ma in df_ma.columns:
                            d2 = df_ma[df_ma[dir_col_ma].notna()]
                            if not d2.empty:
                                ud = d2[dir_col_ma].unique()
                                cmap = {d: TEAL if "pos" in str(d).lower() else ROSE
                                        for d in ud}
                                cmap["neutral"] = SLATE
                                fig = px.pie(d2, names=dir_col_ma, hole=0.55,
                                             color=dir_col_ma, color_discrete_map=cmap,
                                             title="<b>Shock direction</b>")
                                fig.update_layout(height=260, **T)
                                st.plotly_chart(fig, use_container_width=True)

                    if sup_ma and yr_col_ma and sev_col_ma and sev_col_ma in df_ma.columns:
                        fig = px.scatter(df_ma, x=yr_col_ma, y=sup_ma,
                                         color=sev_col_ma,
                                         color_discrete_sequence=ACCENT_SEQ,
                                         title="<b>Supply with severity</b>")
                        fig.update_layout(height=320, **T)
                        st.plotly_chart(fig, use_container_width=True)

                    if c_col_ma and sev_col_ma and sev_col_ma in df_ma.columns:
                        sv = df_ma[sev_col_ma]
                        if pd.api.types.is_object_dtype(sv):
                            mask = sv.str.lower().isin(["high", "extreme"])
                        else:
                            num_sv = pd.to_numeric(sv, errors="coerce")
                            thresh = num_sv.quantile(0.75) if num_sv.notna().any() else 0
                            mask = num_sv >= thresh
                        top_ma = df_ma[mask] if ctrl_all_ma else df_ma[mask].head(ctrl_top_ma)
                        if not top_ma.empty:
                            show_c = [x for x in [c_col_ma, yr_col_ma, sev_col_ma, dir_col_ma]
                                      if x and x in top_ma.columns]
                            st.dataframe(top_ma[show_c],
                                         use_container_width=True, hide_index=True)

        # ── Prophet ───────────────────────────────────────────────────────
        with t_prop:
            if not ok(prophet):
                empty_state("Prophet checkpoint not found.")
            else:
                df_pr = temp_filter(prophet)
                c_col_pr = fcol(df_pr, "country", "Country")
                yr_col_pr = fcol(df_pr, "year", "Year")
                flag_col_pr = fcol(df_pr, "prophet_anomaly_flag", "anomaly",
                                   "is_anomaly", "flag")
                sup_pr = get_supply_col(df_pr)

                _, ctrl_sel_pr, ctrl_top_pr, ctrl_all_pr = country_controls(
                    "prop", get_countries(df_pr))
                df_pr = filter_df(df_pr, ctrl_sel_pr, "")

                if not df_pr.empty and flag_col_pr:
                    col1, col2 = st.columns([3, 2])
                    with col1:
                        if sup_pr and yr_col_pr:
                            fig = plot_ts_anomalies(
                                df_pr, yr_col_pr, sup_pr, flag_col_pr,
                                f"<b>Prophet: {sel_temp if sel_temp != 'All' else 'All Countries'}</b>",
                                EMERALD)
                            if fig:
                                st.plotly_chart(fig, use_container_width=True)

                        if yr_col_pr and flag_col_pr in df_pr.columns:
                            anom_val = 1 if (df_pr[flag_col_pr] == 1).any() else -1
                            yr_pr = ((df_pr[flag_col_pr] == anom_val)
                                     .groupby(df_pr[yr_col_pr]).sum()
                                     .reset_index())
                            yr_pr.columns = ["year", "anomalies"]
                            if not yr_pr.empty:
                                fig2 = px.bar(yr_pr, x="year", y="anomalies",
                                              color="anomalies",
                                              color_continuous_scale=[[0, "#0d2a20"],
                                                                       [1, EMERALD]],
                                              title="<b>Prophet anomalies per year</b>")
                                fig2.update_layout(height=240, **T,
                                                   coloraxis_showscale=False)
                                st.plotly_chart(fig2, use_container_width=True)

                    with col2:
                        if c_col_pr and flag_col_pr in df_pr.columns:
                            anom_val = 1 if (df_pr[flag_col_pr] == 1).any() else -1
                            anom_pr = df_pr[df_pr[flag_col_pr] == anom_val]
                            if not anom_pr.empty:
                                top_pr = (anom_pr.groupby(c_col_pr).size()
                                          .sort_values(ascending=False)
                                          .head(ctrl_top_pr if not ctrl_all_pr else 9999)
                                          .reset_index(name="Flagged Years"))
                                st.dataframe(
                                    top_pr.rename(columns={c_col_pr: "Country"}),
                                    use_container_width=True, hide_index=True)
                elif not flag_col_pr:
                    st.warning(f"No flag column found. Available: {prophet.columns.tolist()}")

        # ── LSTM ──────────────────────────────────────────────────────────
        with t_lstm:
            if not ok(lstm):
                empty_state("LSTM checkpoint not found.")
            else:
                df_lm = temp_filter(lstm)
                c_col_lm  = fcol(df_lm, "country", "Country")
                yr_col_lm = fcol(df_lm, "year", "Year")
                flag_col_lm = fcol(df_lm, "lstm_anomaly_flag", "anomaly", "is_anomaly")
                err_col_lm  = fcol(df_lm, "reconstruction_error", "lstm_error", "error")
                sup_lm = get_supply_col(df_lm)

                _, ctrl_sel_lm, ctrl_top_lm, ctrl_all_lm = country_controls(
                    "lstm", get_countries(df_lm))
                df_lm = filter_df(df_lm, ctrl_sel_lm, "")

                if not df_lm.empty:
                    col1, col2 = st.columns([2, 1])
                    with col1:
                        if sup_lm and yr_col_lm and flag_col_lm and \
                                flag_col_lm in df_lm.columns:
                            fig = plot_ts_anomalies(
                                df_lm, yr_col_lm, sup_lm, flag_col_lm,
                                "<b>LSTM: Supply & anomalies</b>", ROSE)
                            if fig:
                                st.plotly_chart(fig, use_container_width=True)

                        if err_col_lm and err_col_lm in df_lm.columns:
                            fig_h = go.Figure()
                            if flag_col_lm and flag_col_lm in df_lm.columns:
                                anom_val = 1 if (df_lm[flag_col_lm] == 1).any() else -1
                                for label, color, val in [
                                        ("Normal", TEAL, 1 if anom_val == 1 else 0),
                                        ("Anomaly", ROSE, anom_val)]:
                                    m = df_lm[flag_col_lm] == val
                                    if m.any():
                                        fig_h.add_trace(go.Histogram(
                                            x=df_lm.loc[m, err_col_lm],
                                            name=label, marker_color=color,
                                            opacity=0.75, nbinsx=50))
                            else:
                                fig_h.add_trace(go.Histogram(
                                    x=df_lm[err_col_lm], marker_color=TEAL, nbinsx=50))
                            fig_h.update_layout(
                                title="<b>Reconstruction error distribution</b>",
                                barmode="overlay", height=260, **T)
                            st.plotly_chart(fig_h, use_container_width=True)

                    with col2:
                        if flag_col_lm and c_col_lm and flag_col_lm in df_lm.columns:
                            anom_val = 1 if (df_lm[flag_col_lm] == 1).any() else -1
                            anom_lm = df_lm[df_lm[flag_col_lm] == anom_val]
                            if not anom_lm.empty:
                                top_lm = (anom_lm.groupby(c_col_lm).size()
                                          .sort_values(ascending=False)
                                          .head(ctrl_top_lm if not ctrl_all_lm else 9999)
                                          .reset_index(name="Flagged Years"))
                                st.dataframe(
                                    top_lm.rename(columns={c_col_lm: "Country"}),
                                    use_container_width=True, hide_index=True)

    # ──────────────────────────────────────────────────────────────────────
    # TAB: FORECASTING
    # ──────────────────────────────────────────────────────────────────────
    with tab_forecast:
        with st.expander("How forecasting models work", expanded=False):
            st.markdown("""
**ARIMA** (Auto-Regressive Integrated Moving Average) models each country's supply as a
function of its own lagged values (AR), differencing for stationarity (I), and lagged forecast
errors (MA). Auto-ARIMA selects optimal (p,d,q) parameters per country via AIC minimisation.
Provides a 5-year supply projection under no-shock conditions — the counterfactual baseline.

**Prophet trend forecast** (if forecast columns are available): extends the fitted Prophet model
forward and shows the predicted trend with uncertainty bands. Widening confidence intervals
indicate higher structural uncertainty in that country's supply trajectory.
            """)

        if not ok(arima):
            empty_state("ARIMA forecast checkpoint not found.")
        else:
            c_col_ar = fcol(arima, "country", "Country")
            yr_col_ar = fcol(arima, "year", "Year", "forecast_year")
            sup_col_ar = fcol(
                arima,
                "forecasted_supply",
                "forecast",
                "forecast_value",
                "predicted_supply",
                "predicted",
                "supply",
                "value",
            )

            if c_col_ar and yr_col_ar and sup_col_ar:
                _, ctrl_sel_ar, ctrl_top_ar, ctrl_all_ar = country_controls(
                    "ar", get_countries(arima), 10)
                df_ar = filter_df(gf(arima), ctrl_sel_ar, "")
                countries_ar = sorted(df_ar[c_col_ar].dropna().unique().tolist())

                if not countries_ar:
                    st.info("No ARIMA rows available after current filters.")
                else:
                    if "All Countries" in ctrl_sel_ar or not ctrl_sel_ar:
                        sel_c = countries_ar if ctrl_all_ar else countries_ar[:ctrl_top_ar]
                    else:
                        sel_c = [c for c in ctrl_sel_ar
                                 if c in countries_ar and c != "All Countries"] \
                                or (countries_ar if ctrl_all_ar else countries_ar[:ctrl_top_ar])

                    df_plot = df_ar[df_ar[c_col_ar].isin(sel_c)].copy()
                    df_plot[yr_col_ar] = pd.to_numeric(df_plot[yr_col_ar], errors="coerce")
                    df_plot[sup_col_ar] = pd.to_numeric(df_plot[sup_col_ar], errors="coerce")
                    df_plot = df_plot.dropna(subset=[yr_col_ar, sup_col_ar, c_col_ar]).sort_values(
                        [c_col_ar, yr_col_ar]
                    )

                    col1, col2 = st.columns([3, 2])
                    with col1:
                        if df_plot.empty:
                            st.info("ARIMA plot has no numeric forecast rows after filtering.")
                        else:
                            fig = px.line(
                                df_plot,
                                x=yr_col_ar,
                                y=sup_col_ar,
                                color=c_col_ar,
                                markers=True,
                                color_discrete_sequence=ACCENT_SEQ,
                                title=f"<b>ARIMA forecast trajectories</b> · {len(sel_c)} countries",
                                labels={
                                    yr_col_ar: "Year",
                                    sup_col_ar: "Forecasted Supply",
                                    c_col_ar: "",
                                },
                            )
                            fig.update_traces(line=dict(width=1.8), marker=dict(size=5))
                            fig.update_layout(
                                height=400,
                                xaxis=dict(type="linear", tickmode="linear", dtick=1),
                                **T,
                            )
                            st.plotly_chart(fig, use_container_width=True)

                    with col2:
                        df_latest = df_ar.copy()
                        df_latest[yr_col_ar] = pd.to_numeric(df_latest[yr_col_ar], errors="coerce")
                        df_latest[sup_col_ar] = pd.to_numeric(df_latest[sup_col_ar], errors="coerce")
                        df_latest = df_latest.dropna(subset=[yr_col_ar, sup_col_ar, c_col_ar])

                        latest_year = int(df_latest[yr_col_ar].max()) if not df_latest.empty else None
                        latest_df = (
                            df_latest[df_latest[yr_col_ar] == latest_year]
                            .sort_values(sup_col_ar, ascending=False)
                            if latest_year is not None
                            else df_latest
                        )
                        latest_df = latest_df if ctrl_all_ar else latest_df.head(ctrl_top_ar)
                        st.markdown(f"**Latest forecast year: {latest_year if latest_year is not None else '—'}**")
                        st.dataframe(
                            latest_df[[c_col_ar, yr_col_ar, sup_col_ar]].rename(
                                columns={
                                    c_col_ar: "Country",
                                    yr_col_ar: "Year",
                                    sup_col_ar: "Forecasted Supply",
                                }
                            ),
                            use_container_width=True,
                            hide_index=True,
                        )

                    with st.expander("Show all ARIMA forecast rows", expanded=False):
                        st.dataframe(
                            df_plot[[c_col_ar, yr_col_ar, sup_col_ar]].rename(
                                columns={
                                    c_col_ar: "Country",
                                    yr_col_ar: "Year",
                                    sup_col_ar: "Forecasted Supply",
                                }
                            ).sort_values(["Country", "Year"]),
                            use_container_width=True,
                            hide_index=True,
                        )
            else:
                st.warning(f"ARIMA columns available: {list(arima.columns)}")
                empty_state("Required columns not found in ARIMA data.")

    # ──────────────────────────────────────────────────────────────────────
    # TAB: CLUSTERING
    # ──────────────────────────────────────────────────────────────────────
    
    with tab_cluster:
        with st.expander("How clustering works", expanded=False):
            st.markdown("""
**KMeans** partitions countries into k groups (k=4, optimal via silhouette score) based on
5 supply features: average supply, import dependency, volatility, growth rate, shock frequency.
Identifies strategic archetypes — analysts can apply archetype-specific interventions at scale
rather than examining 172 countries individually.

**DBSCAN** finds clusters based on feature-space density without requiring k to be specified.
Countries in sparse regions are labelled as noise (Cluster −1) — these have no statistical peers
and require truly bespoke risk frameworks.

**Temporal clustering** applies KMeans to rolling 5-year supply windows to detect when a country
switches regime (e.g. Stable → Volatile). High switch frequency indicates structural fragility
rather than a single shock.
            """)

        c_km, c_db, c_tc = st.tabs(["KMeans archetypes", "DBSCAN", "Temporal regimes"])

        with c_km:
            if not ok(clusters):
                empty_state("KMeans clusters checkpoint not found.")
            else:
                c_col_km = fcol(clusters, "country", "Country")
                lbl_km   = fcol(clusters, "cluster_label", "cluster", "label")
                x_km     = fcol(clusters, "avg_import_dep", "avg_import_dependency",
                                "import_dependency")
                y_km     = fcol(clusters, "avg_volatility", "supply_volatility",
                                "volatility")
                s_km     = fcol(clusters, "avg_supply", "avg_total_supply",
                                "total_supply")

                _, ctrl_sel_km, ctrl_top_km, ctrl_all_km = country_controls(
                    "km", get_countries(clusters))
                df_km = filter_df(gf(clusters), ctrl_sel_km, "")

                col1, col2 = st.columns([3, 2])
                with col1:
                    if x_km and y_km and lbl_km:
                        fig = px.scatter(df_km, x=x_km, y=y_km, color=lbl_km,
                                         hover_name=c_col_km,
                                         size=s_km if s_km else None, size_max=20,
                                         color_discrete_sequence=ACCENT_SEQ,
                                         title="<b>Country archetype separation</b>",
                                         labels={x_km: "Avg Import Dependency",
                                                 y_km: "Avg Volatility"})
                        fig.update_traces(marker=dict(opacity=0.8, line=dict(width=0)))
                        fig.update_layout(height=360, **T)
                        st.plotly_chart(fig, use_container_width=True)

                with col2:
                    if lbl_km:
                        sizes = df_km[lbl_km].value_counts()
                        fig = px.pie(values=sizes.values, names=sizes.index, hole=0.55,
                                     color_discrete_sequence=ACCENT_SEQ,
                                     title="<b>Archetype distribution</b>")
                        fig.update_layout(height=280, **T)
                        st.plotly_chart(fig, use_container_width=True)

                if c_col_km and lbl_km:
                    with st.expander("Browse all countries by cluster"):
                        for lv in sorted(df_km[lbl_km].dropna().unique()):
                            sub = df_km[df_km[lbl_km] == lv][[c_col_km]]
                            st.markdown(f"**{lv}** — {len(sub)} countries")
                            st.dataframe(sub.rename(columns={c_col_km: "Country"}),
                                         use_container_width=True, hide_index=True)

        with c_db:
            if not ok(dbscan_df):
                empty_state("DBSCAN results checkpoint not found.")
            else:
                c_col_db = fcol(dbscan_df, "country", "Country")
                lbl_col_db = fcol(
                    dbscan_df,
                    "dbscan_cluster",
                    "dbscan_label",
                    "cluster",
                    "label",
                    "prediction",
                )
                x_db = fcol(dbscan_df, "import_dep", "avg_import_dep", "import_dependency")
                y_db = fcol(dbscan_df, "volatility", "avg_volatility", "supply_volatility")
                size_db = fcol(dbscan_df, "avg_shock", "shock_score", "avg_supply", "value")

                _, ctrl_sel_db, ctrl_top_db, ctrl_all_db = country_controls(
                    "db", get_countries(dbscan_df), 10
                )
                df_db = filter_df(gf(dbscan_df), ctrl_sel_db, "")

                if lbl_col_db and c_col_db:
                    noise = df_db[df_db[lbl_col_db] == -1].copy()
                    cluster_sizes = (
                        df_db.groupby(lbl_col_db)[c_col_db]
                        .count()
                        .reset_index(name="Country Count")
                        .sort_values(["Country Count", lbl_col_db], ascending=[False, True])
                    )

                    col1, col2 = st.columns([3, 2])
                    with col1:
                        if x_db and y_db:
                            df_db_plot = df_db.copy()
                            df_db_plot["Cluster"] = df_db_plot[lbl_col_db].astype(str)
                            fig = px.scatter(
                                df_db_plot,
                                x=x_db,
                                y=y_db,
                                color="Cluster",
                                hover_name=c_col_db,
                                size=size_db if size_db else None,
                                size_max=22,
                                color_discrete_sequence=ACCENT_SEQ,
                                title="<b>DBSCAN cluster map</b>",
                                labels={
                                    x_db: "Import Dependency",
                                    y_db: "Volatility",
                                },
                            )
                            fig.update_traces(marker=dict(opacity=0.85, line=dict(width=0)))
                            fig.update_layout(height=380, **T)
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            fig = px.bar(
                                cluster_sizes,
                                x=lbl_col_db,
                                y="Country Count",
                                color="Country Count",
                                color_continuous_scale=[[0, "#0d1830"], [1, BLUE]],
                                title="<b>DBSCAN cluster sizes</b>",
                            )
                            fig.update_layout(height=320, **T, coloraxis_showscale=False)
                            st.plotly_chart(fig, use_container_width=True)

                    with col2:
                        st.markdown("**Cluster summary**")
                        st.dataframe(
                            cluster_sizes.rename(columns={lbl_col_db: "Cluster"}),
                            use_container_width=True,
                            hide_index=True,
                        )
                        st.markdown(f"**No-peer countries (Cluster −1): {len(noise)}**")
                        st.caption(
                            "These supply profiles have no statistical peers and need bespoke risk handling."
                        )
                        noise_show = noise[[c_col_db]].rename(columns={c_col_db: "Country"})
                        if not ctrl_all_db:
                            noise_show = noise_show.head(ctrl_top_db)
                        st.dataframe(noise_show, use_container_width=True, hide_index=True)

                    with st.expander("Show countries inside each DBSCAN cluster", expanded=False):
                        for cluster_id in sorted(df_db[lbl_col_db].dropna().unique()):
                            sub = (
                                df_db[df_db[lbl_col_db] == cluster_id]
                                .sort_values(c_col_db)
                                [[c_col_db] + [c for c in [x_db, y_db, size_db] if c]]
                            )
                            if not ctrl_all_db:
                                sub = sub.head(ctrl_top_db)
                            st.markdown(f"**Cluster {cluster_id}** — {len(df_db[df_db[lbl_col_db] == cluster_id])} countries")
                            st.dataframe(
                                sub.rename(columns={c_col_db: "Country"}),
                                use_container_width=True,
                                hide_index=True,
                            )

        with c_tc:
            if not ok(temporal_cl):
                empty_state("Temporal clusters checkpoint not found.")
            else:
                period_col = fcol(temporal_cl, "period", "year", "time_period")
                sw_col     = fcol(temporal_cl, "cluster_switch", "switch")
                c_col_tc   = fcol(temporal_cl, "country", "Country")

                _, ctrl_sel_tc, ctrl_top_tc, ctrl_all_tc = country_controls(
                    "tc", get_countries(temporal_cl))
                df_tc = filter_df(gf(temporal_cl), ctrl_sel_tc, "")

                if period_col and sw_col:
                    agg = df_tc.groupby(period_col)[sw_col].sum().reset_index()
                    total_sw = int(df_tc[sw_col].sum())
                    fig = go.Figure(go.Scatter(
                        x=agg[period_col], y=agg[sw_col],
                        fill="tozeroy", fillcolor="rgba(244,114,182,0.08)",
                        line=dict(color=PINK, width=2),
                        mode="lines+markers", marker=dict(size=5, color=PINK)))
                    fig.update_layout(
                        title=f"<b>Regime switches per period</b> (total: {total_sw:,})",
                        height=300, **T)
                    st.plotly_chart(fig, use_container_width=True)

                if c_col_tc and sw_col:
                    with st.expander("Countries with most regime switches"):
                        most = (df_tc.groupby(c_col_tc)[sw_col].sum()
                                .sort_values(ascending=False)
                                .head(ctrl_top_tc if not ctrl_all_tc else 9999)
                                .reset_index())
                        st.dataframe(most.rename(columns={c_col_tc: "Country",
                                                           sw_col: "Total Switches"}),
                                     use_container_width=True, hide_index=True)

    # ──────────────────────────────────────────────────────────────────────
    # TAB: SURVIVAL / COX
    # ──────────────────────────────────────────────────────────────────────
    with tab_survival:
        with st.expander("How Cox PH survival analysis works", expanded=False):
            st.markdown("""
**Cox Proportional Hazards** estimates how covariates affect the hazard rate — the instantaneous
rate at which a stable country transitions into a supply shock state.

The model outputs **log hazard ratios** (coefficients β). A **positive β** means that feature
accelerates time-to-shock (risk-increasing). A **negative β** means it delays shocks (protective).

**Hazard Ratio exp(β)**: a value of 2.0 means the feature doubles the shock rate. A value of 0.5
means the feature halves it. The baseline hazard h₀(t) is left unspecified — the model is
semi-parametric.

**Key question answered**: "Given a country's current covariate profile, how long until its next
supply shock, and which features are driving that timing?"
            """)

        if not ok(cox_coef):
            empty_state("Cox PH coefficients checkpoint not found.")
        else:
            cov_col  = fcol(cox_coef, "covariate", "feature", "variable")
            coef_col = fcol(cox_coef, "coef", "coefficient", "beta")
            hr_col   = fcol(cox_coef, "exp(coef)", "hazard_ratio", "HR")
            p_col    = fcol(cox_coef, "p_value", "pval", "p")

            col1, col2 = st.columns([3, 2])
            with col1:
                if cov_col and coef_col:
                    dc = cox_coef.sort_values(coef_col)
                    colors = [ROSE if v > 0 else TEAL for v in dc[coef_col]]
                    fig = go.Figure(go.Bar(
                        x=dc[coef_col], y=dc[cov_col], orientation="h",
                        marker=dict(color=colors, line=dict(width=0)),
                        text=dc[coef_col].apply(lambda v: f"{v:+.3f}"),
                        textposition="outside",
                        textfont=dict(size=9, color="#94a3b8")))
                    fig.add_vline(x=0, line_color="#334155", line_width=1.5)
                    fig.update_layout(
                        title="<b>Cox PH coefficients</b>  "
                              "(🔴 risk-increasing · 🔵 protective)",
                        height=380, **T,
                        xaxis_title="Log hazard ratio (coefficient)",
                        yaxis=dict(gridcolor="rgba(0,0,0,0)"))
                    st.plotly_chart(fig, use_container_width=True)

            with col2:
                if hr_col and cov_col:
                    df_hr = cox_coef.sort_values(hr_col, ascending=False)
                    fig = px.bar(df_hr, x=cov_col, y=hr_col, color=hr_col,
                                 color_continuous_scale=[[0, "#0d2a2a"],
                                                          [0.5, AMBER], [1, ROSE]],
                                 title="<b>Hazard ratios exp(β)</b>  (> 1 = ↑ risk)")
                    fig.add_hline(y=1, line_dash="dot", line_color="#334155")
                    fig.update_layout(height=340, **T, coloraxis_showscale=False,
                                      xaxis_title="")
                    st.plotly_chart(fig, use_container_width=True)

            with st.expander("Full coefficient table"):
                show_c = [x for x in [cov_col, coef_col, hr_col, p_col] if x]
                if show_c:
                    st.dataframe(cox_coef[show_c], use_container_width=True, hide_index=True)

    # ──────────────────────────────────────────────────────────────────────
    # TAB: CHANGE POINTS
    # ──────────────────────────────────────────────────────────────────────
    with tab_chgpt:
        with st.expander("How change-point detection works", expanded=False):
            st.markdown("""
**Change-point detection** (PELT — Pruned Exact Linear Time) identifies moments where a country's
supply distribution fundamentally shifts: a new mean, new variance, or both. This is distinct from
a single-year spike — a change-point marks a **persistent regime shift** that may last years or
decades.

The RBF cost function is used, which is sensitive to both mean and variance changes. PELT finds the
optimal number of breakpoints exactly in O(n) time.

**Strategic use**: A country with a single change-point in 2014 probably had a structural reform
or infrastructure investment. A country with 4+ change-points is oscillating — indicating fragility
that requires dynamic rather than static procurement strategy.
            """)

        if not ok(change_pts):
            empty_state("Change-points checkpoint not found.")
        else:
            _, ctrl_sel_cp, ctrl_top_cp, ctrl_all_cp = country_controls(
                "cp", get_countries(change_pts))
            df_cp  = filter_df(gf(change_pts), ctrl_sel_cp, "")
            c_col_cp  = fcol(df_cp, "country", "Country")
            yr_col_cp = fcol(df_cp, "year_change_point", "year", "change_year")

            col1, col2 = st.columns([3, 2])
            with col1:
                if yr_col_cp:
                    # Use full dataset for global histogram, not filtered
                    yc_all = change_pts[yr_col_cp].value_counts().sort_index().reset_index()
                    yc_all.columns = ["year", "breaks"]
                    fig = go.Figure(go.Bar(
                        x=yc_all["year"], y=yc_all["breaks"],
                        marker=dict(color=yc_all["breaks"],
                                    colorscale=[[0, "#1e1a0d"], [1, AMBER]],
                                    showscale=False, line=dict(width=0))))
                    fig.update_layout(
                        title="<b>Structural breaks per year</b>",
                        height=300, **T, xaxis_title="Year",
                        yaxis_title="# Breaks")
                    st.plotly_chart(fig, use_container_width=True)

            with col2:
                if c_col_cp and yr_col_cp:
                    mb = df_cp.groupby(c_col_cp)[yr_col_cp].apply(list).reset_index()
                    mb.columns = ["Country", "Break Years"]
                    mb["# Breaks"] = mb["Break Years"].apply(len)
                    mb = mb.sort_values("# Breaks", ascending=False)
                    top_cp = mb if ctrl_all_cp else mb.head(ctrl_top_cp)
                    st.dataframe(top_cp[["Country", "# Breaks", "Break Years"]],
                                 use_container_width=True, hide_index=True)

            # Country break detail
            if c_col_cp and yr_col_cp:
                with st.expander("Country break detail"):
                    countries_cp = sorted(df_cp[c_col_cp].dropna().unique())
                    sel_cp = st.selectbox("Select country", countries_cp,
                                          key="cp_detail") if countries_cp else None
                    if sel_cp:
                        sub_cp = df_cp[df_cp[c_col_cp] == sel_cp]
                        break_years = sorted(sub_cp[yr_col_cp].dropna().tolist())
                        st.markdown(f"**{sel_cp}** — {len(break_years)} structural "
                                    f"break(s): {break_years}")

                        # Show supply time-series with break lines if available
                        sup_src = features_df if ok(features_df) else ewma if ok(ewma) else None
                        if ok(sup_src):
                            c_feat = fcol(sup_src, "country", "Country")
                            y_feat = fcol(sup_src, "year", "Year")
                            s_feat = get_supply_col(sup_src)
                            if all([c_feat, y_feat, s_feat]):
                                sub_f = sup_src[sup_src[c_feat] == sel_cp]
                                if not sub_f.empty:
                                    fig = px.line(sub_f, x=y_feat, y=s_feat,
                                                  title=f"<b>{sel_cp} — supply with break lines</b>")
                                    fig.update_traces(line=dict(color=TEAL))
                                    for by in break_years:
                                        fig.add_vline(x=by, line_dash="dash",
                                                      line_color=AMBER,
                                                      annotation_text=str(by),
                                                      annotation_font_size=9)
                                    fig.update_layout(height=300, **T)
                                    st.plotly_chart(fig, use_container_width=True)

    # ──────────────────────────────────────────────────────────────────────
    # TAB: RAW DATA
    # ──────────────────────────────────────────────────────────────────────
    with tab_raw:
        all_cp = {
            "anomaly_isolation_forest":   iso_forest,
            "anomaly_lof":                lof,
            "anomaly_ocsvm":              ocsvm_df,
            "anomaly_composite":          comp_df,
            "anomaly_ewma":               ewma,
            "anomaly_iqr":                iqr_df,
            "anomaly_percentile":         percentile,
            "anomaly_moving_avg_robust":  mov_avg,
            "anomaly_prophet":            prophet,
            "anomaly_lstm":               lstm,
            "anomaly_dashboard":          adash,
            "combined_yearly_anomalies":  combined_yr,
            "combined_country_anomalies": combined_co,
            "clusters":                   clusters,
            "dbscan_results":             dbscan_df,
            "temporal_clusters":          temporal_cl,
            "country_pca_features":       pca_feat,
            "arima_forecast_results":     arima,
            "change_points":              change_pts,
            "rf_predictions":             rf_pred,
            "granger_causality_results":  granger,
            "coxph_coefficients":         cox_coef,
            "coxph_prepared_data":        cox_prep,
            "network_metrics":            net_metrics,
            "network_communities":        net_comm,
            "shock_score":                shock_score,
            "shock_validation_advanced":  shock_valid,
            "contagion_features":         contagion,
        }
        found = {k: v for k, v in all_cp.items() if ok(v)}
        st.caption(f"**{len(found)}** of **{len(all_cp)}** checkpoints loaded")

        for name, df in found.items():
            r1, r2, r3 = st.columns([3, 2, 1])
            with r1:
                st.markdown(f"`{name}` — **{len(df):,}** rows × **{len(df.columns)}** cols")
            with r2:
                st.caption(", ".join(df.columns[:4].tolist()) + ("…" if len(df.columns) > 4 else ""))
            with r3:
                try:
                    csv_bytes = df.to_csv(index=False).encode("utf-8")
                    st.download_button("↓ CSV", csv_bytes, f"{name}.csv",
                                       key=f"dl_{name}")
                except Exception as exc:
                    st.caption(f"Error: {exc}")

    st.divider()

    # ══════════════════════════════════════════════════════════════════════
    # LAYER 4 — UTILITY
    # ══════════════════════════════════════════════════════════════════════

    u1, u2 = st.columns(2)

    with u1:
        with st.expander("📐 When to use each model", expanded=False):
            guide = [
                ("Find structural outliers",     "Isolation Forest + LOF + OCSVM",
                 "Countries unlike all others across multiple features. Use all 3 for high confidence."),
                ("Pinpoint historical disruptions", "EWMA + Prophet + LSTM",
                 "Specific country-years with abnormal supply. EWMA=short shocks, Prophet=seasonal, LSTM=non-linear."),
                ("Predict next-year shock risk",  "Random Forest",
                 "Forward-looking probability to prioritise procurement or policy interventions."),
                ("Trace contagion pathways",      "Granger Causality + Network",
                 "Which countries' shocks cascade to others. Critical for diversification strategy."),
                ("Estimate time-to-shock",        "Cox PH Survival",
                 "How long a country is likely to remain stable given its current risk profile."),
                ("Segment countries",             "KMeans + DBSCAN",
                 "Group similar supply profiles, or identify nations with no peers (DBSCAN noise)."),
                ("Identify regime changes",       "Change-Point Detection",
                 "Find the specific year a country's supply fundamentally changed — persistent shift, not spike."),
                ("Project future baseline",       "ARIMA",
                 "5-year statistical baseline forecast under no-shock conditions."),
            ]
            df_guide = pd.DataFrame(guide,
                                    columns=["Use case", "Model(s)", "When to choose it"])
            st.dataframe(df_guide, use_container_width=True, hide_index=True,
                         column_config={
                             "Use case":       st.column_config.TextColumn(width="medium"),
                             "Model(s)":       st.column_config.TextColumn(width="medium"),
                             "When to choose it": st.column_config.TextColumn(width="large"),
                         })

    with u2:
        with st.expander("ℹ️ Methodology notes", expanded=False):
            st.markdown("""
**Data source**: IEA / FAO LPG supply data 1990–2024, 172 countries.

**Feature engineering**: Import dependency = imports / total supply. Volatility = coefficient of
variation of annual supply. Shock label = year where supply dropped > 2σ from country trend.

**Flag conventions**:
- sklearn models (IF, LOF, OCSVM): −1 = anomaly, 1 = normal
- Standardized output via `helpers.standardize_flags`: 1 = anomaly, 0 = normal
- Always check `_std` suffix columns for the normalized convention

**Contamination parameters**: IF and LOF use `contamination='auto'`. OCSVM uses `nu=0.05`.

**ARIMA selection**: Auto-selected per country via AIC minimisation over (p,d,q) ∈ [0,3]³.

**Granger causality**: Bivariate VAR, lag=2, significance threshold α=0.05.

**Regenerate all checkpoints**: `python pipeline.py --full --ml`
            """)

    st.caption("All metrics from live pipeline checkpoints · No synthetic data · "
               "Regenerate: python pipeline.py --full --ml")
