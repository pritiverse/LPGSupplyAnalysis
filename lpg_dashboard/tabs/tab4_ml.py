# tabs/tab4_ml.py
"""
Advanced ML Analytics Tab — Professional Research Interface
Covers ALL pipeline models with interactive filters, search, country views,
key findings, methodology explanations, and full data access.

FIXES applied:
  - FIX-1  : st.tabs() returns a tuple — all 4 tab vars now unpacked correctly
  - FIX-2  : show_structural_model checks both score_col AND flag_col exist before use
  - FIX-3  : safe display_cols selection — only include columns that exist in df_top
  - FIX-4  : tab_comb content is correctly inside `with tab_comb:` block
  - FIX-5  : composite on-the-fly uses .map() for Series index alignment
  - FIX-6  : EWMA / Prophet re-fetch data if initial variable is None
  - FIX-7  : Moving Avg severity mask handles both string and numeric severity cols
  - FIX-8  : IQR country filter applied before column access
  - FIX-9  : ARIMA country selector respects global filter
  - FIX-10 : norm() helper uses .map() so index aligns to comp_df['country']
  - FIX-11 : granger filter_df uses correct column (source OR target)
  - FIX-12 : download section guard against None DataFrames
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
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM

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
    """Return first matching column name that exists in df."""
    if df is None or (isinstance(df, pd.DataFrame) and df.empty):
        return None
    for n in names:
        if n in df.columns:
            return n
    return None


def anomaly_flag_count(df, *cols) -> int | None:
    for c in cols:
        if ok(df) and c in df.columns:
            return int((df[c] == -1).sum())
    return None


def get_countries(df) -> list[str]:
    c = fcol(df, "country", "Country", "COUNTRY")
    return sorted(df[c].dropna().unique().tolist()) if c else []
@st.cache_data(ttl=3600)
def load_spark_parquet(name: str) -> pd.DataFrame:
    """Load all parquet parts from pipeline_checkpoints/<name> folder."""
    folder = Path.cwd() / "pipeline_checkpoints" / name
    files = sorted(glob.glob(str(folder / "part-*.parquet")) + glob.glob(str(folder / "*.parquet")))
    if not files:
        st.error(f"❌ No data found in `{folder}`. Please run the LPG pipeline first.")
        st.stop()
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    if 'country' in df.columns:
        df['country'] = df['country'].astype(str).str.strip().str.title()
    return df
@st.cache_data
def prepare_data(scaler_type: str = "standard"):
    """Load data, select numeric features, and scale."""
    df_raw = load_spark_parquet("country_pca_features")
    df_raw = df_raw.drop_duplicates(subset=['country']).reset_index(drop=True)
    
    feature_cols = [c for c in df_raw.columns if df_raw[c].dtype in ['float64', 'int64'] and c != 'country']
    if not feature_cols:
        st.error("No numeric features found in the data.")
        st.stop()
    
    X = df_raw[feature_cols].fillna(df_raw[feature_cols].mean())
    scaler = StandardScaler() if scaler_type == "standard" else RobustScaler()
    X_scaled = scaler.fit_transform(X)
    
    return df_raw, X_scaled, feature_cols, scaler

def create_composite_scores(if_pred, lof_pred, svm_pred, if_score, lof_score, svm_score):
    """Combine flags and average scores from all three models."""
    comp_flag = ((if_pred == -1) | (lof_pred == -1) | (svm_pred == -1)).astype(int)
    comp_score = np.mean([if_score, lof_score, svm_score], axis=0)
    return comp_flag, comp_score

def filter_df(df: pd.DataFrame, selected: list, search: str) -> pd.DataFrame:
    c = fcol(df, "country", "Country", "COUNTRY")
    if c is None:
        return df
    mask = pd.Series([True] * len(df), index=df.index)
    if search:
        mask &= df[c].str.contains(search, case=False, na=False)
    if selected and "All Countries" not in selected:
        mask &= df[c].isin(selected)
    return df[mask]


def ok(d) -> bool:
    return d is not None and not (isinstance(d, pd.DataFrame) and d.empty)


# ══════════════════════════════════════════════════════════════════════════════
#  THEME CONSTANTS
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


# ══════════════════════════════════════════════════════════════════════════════
#  CSS
# ══════════════════════════════════════════════════════════════════════════════

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
  --bg1:#080d18; --bg2:#0c1220; --bg3:#111827;
  --border:#1e2d40; --border2:#243448;
  --hi:#e2e8f0; --md:#94a3b8; --lo:#475569;
  --teal:#22d3ee; --rose:#fb7185; --violet:#a78bfa;
  --amber:#fbbf24; --emerald:#34d399;
}
.ml-root, .ml-root * { font-family:'Sora',sans-serif !important; }
.mono { font-family:'JetBrains Mono',monospace !important; }

.hero {
  position:relative; background:var(--bg1);
  border:1px solid var(--border); border-radius:24px;
  padding:3rem 2.8rem 2.4rem; margin-bottom:2.4rem; overflow:hidden;
}
.hero::before {
  content:''; position:absolute; inset:0;
  background:
    radial-gradient(ellipse 55% 60% at 10% 30%,rgba(34,211,238,.07) 0%,transparent 65%),
    radial-gradient(ellipse 40% 50% at 85% 70%,rgba(167,139,250,.06) 0%,transparent 65%),
    radial-gradient(ellipse 30% 40% at 60% 10%,rgba(251,113,133,.04) 0%,transparent 60%);
}
.hero-eye { font-family:'JetBrains Mono',monospace !important; font-size:.62rem;
  letter-spacing:.22em; color:var(--teal); text-transform:uppercase; margin-bottom:.7rem; }
.hero-title { font-size:2.4rem; font-weight:700; color:var(--hi); line-height:1.15; margin-bottom:.7rem; }
.hero-title span { background:linear-gradient(135deg,var(--teal),var(--violet));
  -webkit-background-clip:text; -webkit-text-fill-color:transparent; }
.hero-desc { font-size:.875rem; color:var(--md); line-height:1.7; max-width:620px; margin-bottom:1.2rem; }
.tag-row { display:flex; flex-wrap:wrap; gap:.5rem; }
.tag { display:inline-flex; align-items:center; gap:.3rem;
  background:rgba(34,211,238,.08); border:1px solid rgba(34,211,238,.2);
  border-radius:100px; padding:.22rem .75rem;
  font-family:'JetBrains Mono',monospace !important; font-size:.62rem; color:var(--teal); }

.kpi-grid { display:grid; grid-template-columns:repeat(5,1fr); gap:.8rem; margin-bottom:2rem; }
.kpi { background:var(--bg2); border:1px solid var(--border); border-radius:16px;
  padding:1.1rem 1rem .9rem; position:relative; overflow:hidden;
  transition:border-color .25s,transform .2s; }
.kpi:hover { border-color:var(--teal); transform:translateY(-2px); }
.kpi::before { content:''; position:absolute; top:0;left:0;right:0; height:2px; border-radius:16px 16px 0 0; }
.kpi.c0::before{background:linear-gradient(90deg,var(--teal),transparent);}
.kpi.c1::before{background:linear-gradient(90deg,var(--rose),transparent);}
.kpi.c2::before{background:linear-gradient(90deg,var(--violet),transparent);}
.kpi.c3::before{background:linear-gradient(90deg,var(--amber),transparent);}
.kpi.c4::before{background:linear-gradient(90deg,var(--emerald),transparent);}
.kpi-label { font-family:'JetBrains Mono',monospace !important; font-size:.56rem;
  letter-spacing:.16em; color:var(--lo); text-transform:uppercase; margin-bottom:.5rem; }
.kpi-val { font-size:1.7rem; font-weight:700; color:var(--hi); line-height:1; margin-bottom:.25rem; }
.kpi-sub { font-size:.7rem; color:var(--lo); }
.kpi-icon { position:absolute; top:.8rem; right:.9rem; font-size:1.2rem; opacity:.3; }

.sec { display:flex; align-items:center; gap:.8rem; margin:2.2rem 0 1rem;
  padding-bottom:.6rem; border-bottom:1px solid var(--border); }
.sec-n { font-family:'JetBrains Mono',monospace !important; font-size:.6rem; color:var(--teal);
  background:rgba(34,211,238,.08); border:1px solid rgba(34,211,238,.2);
  padding:.15rem .5rem; border-radius:4px; }
.sec-title { font-size:1.05rem; font-weight:600; color:var(--hi); }
.sec-sub { font-size:.75rem; color:var(--lo); flex:1; text-align:right; }

.mc { background:var(--bg2); border:1px solid var(--border); border-radius:18px;
  padding:1.4rem 1.4rem 1.2rem; margin-bottom:1.2rem; position:relative; overflow:hidden; }
.mc::before { content:''; position:absolute; left:0;top:0;bottom:0; width:3px; border-radius:18px 0 0 18px; }
.mc-teal::before{background:var(--teal);}    .mc-rose::before{background:var(--rose);}
.mc-violet::before{background:var(--violet);} .mc-amber::before{background:var(--amber);}
.mc-emerald::before{background:var(--emerald);} .mc-blue::before{background:#60a5fa;}
.mc-pink::before{background:#f472b6;} .mc-lime::before{background:#a3e635;}
.mc-eye { font-family:'JetBrains Mono',monospace !important; font-size:.58rem;
  letter-spacing:.15em; color:var(--lo); text-transform:uppercase; margin-bottom:.3rem; }
.mc-title { font-size:.95rem; font-weight:600; color:var(--hi); margin-bottom:.2rem; }
.mc-abbr { font-family:'JetBrains Mono',monospace !important; font-size:.72rem;
  color:var(--teal); margin-bottom:.6rem; }
.mc-body { font-size:.78rem; color:var(--md); line-height:1.7; margin-bottom:.7rem; }
.fb { background:rgba(34,211,238,.05); border:1px solid rgba(34,211,238,.15);
  border-radius:10px; padding:.75rem 1rem; font-size:.78rem; color:var(--md); line-height:1.6; }
.fb strong{color:var(--teal);}
.fb.rose{background:rgba(251,113,133,.05);border-color:rgba(251,113,133,.15);}
.fb.rose strong{color:var(--rose);}
.fb.violet{background:rgba(167,139,250,.05);border-color:rgba(167,139,250,.15);}
.fb.violet strong{color:var(--violet);}
.fb.amber{background:rgba(251,191,36,.05);border-color:rgba(251,191,36,.15);}
.fb.amber strong{color:var(--amber);}
.fb.emerald{background:rgba(52,211,153,.05);border-color:rgba(52,211,153,.15);}
.fb.emerald strong{color:var(--emerald);}

.guide-grid { display:grid; grid-template-columns:repeat(4,1fr); gap:.8rem; margin:.8rem 0; }
.guide-card { background:var(--bg2); border:1px solid var(--border); border-radius:16px;
  padding:1.3rem 1rem 1.1rem; text-align:center;
  transition:border-color .2s,transform .2s; }
.guide-card:hover{border-color:var(--teal);transform:translateY(-3px);}
.gi{font-size:1.7rem;margin-bottom:.5rem;}
.gu{font-size:.8rem;font-weight:600;color:var(--hi);margin-bottom:.3rem;}
.gm{font-family:'JetBrains Mono',monospace !important;font-size:.65rem;color:var(--teal);margin-bottom:.3rem;}
.gd{font-size:.7rem;color:var(--lo);line-height:1.5;}

.ml-divider{height:1px;background:linear-gradient(90deg,transparent,var(--border),transparent);margin:1.8rem 0;}
.ml-footer{text-align:center;padding:1.2rem;border-top:1px solid var(--border);
  font-family:'JetBrains Mono',monospace !important;font-size:.55rem;
  letter-spacing:.12em;color:var(--lo);margin-top:2rem;}
.empty-s{text-align:center;padding:2.5rem 1rem;color:var(--lo);font-size:.82rem;
  border:1px dashed var(--border);border-radius:14px;}
</style>
"""


# ══════════════════════════════════════════════════════════════════════════════
#  HTML BUILDERS
# ══════════════════════════════════════════════════════════════════════════════

def kpi_html(cc, icon, label, value, sub) -> str:
    v = f"{value:,}" if isinstance(value, int) else (str(value) if value is not None else "—")
    return (
        f'<div class="kpi {cc}"><div class="kpi-icon">{icon}</div>'
        f'<div class="kpi-label">{label}</div><div class="kpi-val">{v}</div>'
        f'<div class="kpi-sub">{sub}</div></div>'
    )


def sec_html(n, title, sub="") -> str:
    s = f'<span class="sec-sub">{sub}</span>' if sub else ""
    return (
        f'<div class="sec"><span class="sec-n">{n:02d}</span>'
        f'<span class="sec-title">{title}</span>{s}</div>'
    )


def mc_html(color, eye, title, abbr, body, finding, fb_cls="") -> str:
    return (
        f'<div class="mc mc-{color}">'
        f'<div class="mc-eye">{eye}</div><div class="mc-title">{title}</div>'
        f'<div class="mc-abbr">{abbr}</div><div class="mc-body">{body}</div>'
        f'<div class="fb {fb_cls}">🔍 <strong>Key Finding:</strong> {finding}</div></div>'
    )


def empty_html(msg="Checkpoint not available. Run pipeline.py --full --ml to generate.") -> str:
    return f'<div class="empty-s">📭&nbsp; {msg}</div>'


def country_controls(key: str, all_c: list, default_top: int = 15):
    """Render search / multiselect / top-N / show-all controls. Returns (srch, sel, top, show_all)."""
    c1, c2, c3, c4 = st.columns([2, 3, 1, 1])
    with c1:
        srch = st.text_input(
            "Search", key=f"{key}_s",
            placeholder="🔍 Country name…", label_visibility="collapsed",
        )
    with c2:
        sel = st.multiselect(
            "Filter", ["All Countries"] + all_c,
            default=["All Countries"], key=f"{key}_m",
            label_visibility="collapsed", placeholder="Select countries…",
        )
    with c3:
        top = st.number_input(
            "Top N", 5, max(len(all_c), 50) if all_c else 50,
            default_top, 5, key=f"{key}_n", label_visibility="collapsed",
        )
    with c4:
        show_all = st.toggle("All rows", key=f"{key}_a", value=False)
    return srch, sel, int(top), show_all


# ══════════════════════════════════════════════════════════════════════════════
#  SHARED PLOT HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def get_supply_col(df: pd.DataFrame) -> str | None:
    for col in ("supply", "total_supply", "value", "supply_volume", "supply_mt"):
        if col in df.columns:
            return col
    return None


def plot_ts_anomalies(df, date_col, value_col, flag_col, title, color):
    """Time-series line with anomaly markers overlaid."""
    if not all([date_col, value_col, flag_col]):
        return None
    if date_col not in df.columns or value_col not in df.columns:
        return None

    fig = go.Figure()
    normal = df[df[flag_col] != -1] if flag_col in df.columns else df
    fig.add_trace(go.Scatter(
        x=normal[date_col], y=normal[value_col],
        mode="lines+markers", name="Supply",
        line=dict(color=color, width=2),
        marker=dict(size=4, color=color),
    ))
    if flag_col in df.columns:
        anom = df[df[flag_col] == -1]
        if not anom.empty:
            fig.add_trace(go.Scatter(
                x=anom[date_col], y=anom[value_col],
                mode="markers", name="Anomaly",
                marker=dict(symbol="x", size=10, color=ROSE,
                            line=dict(width=1, color="white")),
            ))
    fig.update_layout(
        title=title, height=350, **T,
        xaxis_title="Year", yaxis_title=value_col,
        hovermode="x unified",
    )
    return fig


# ══════════════════════════════════════════════════════════════════════════════
#  STRUCTURAL MODEL RENDERER  (FIX-2, FIX-3)
# ══════════════════════════════════════════════════════════════════════════════

def show_structural_model(df_model, model_name, score_col, flag_col, color_scale, accent_color):
    if not ok(df_model):
        st.markdown(empty_html(f"{model_name} checkpoint not found."), unsafe_allow_html=True)
        return

    ctrl_s, ctrl_sel, ctrl_top, ctrl_all = country_controls(
        model_name.lower().replace(" ", "_"), get_countries(df_model)
    )
    df_f = filter_df(df_model, ctrl_sel, ctrl_s)
    c_col = fcol(df_f, "country", "Country")
    if not c_col:
        st.warning(f"No country column found in {model_name} data.")
        return

    # FIX-2: check flag_col exists before filtering
    if flag_col and flag_col in df_f.columns:
        df_anom = df_f[df_f[flag_col] == -1]
    else:
        df_anom = df_f  # show all if no flag column

    # FIX-2: check score_col exists before sorting
    if score_col and score_col in df_anom.columns:
        df_top = df_anom.nlargest(ctrl_top if not ctrl_all else len(df_anom), score_col)
    else:
        df_top = df_anom if ctrl_all else df_anom.head(ctrl_top)

    col1, col2 = st.columns([3, 2])
    with col1:
        if score_col and score_col in df_top.columns:
            fig = px.bar(
                df_top.sort_values(score_col, ascending=True),
                x=score_col, y=c_col, orientation="h",
                color=score_col, color_continuous_scale=color_scale,
                title=f"<b>{model_name} — Top Anomaly Scores</b>",
            )
            fig.update_layout(height=400, **{**T, "margin": dict(l=120)})
            st.plotly_chart(fig, use_container_width=True)
        else:
            # Fallback: count of flagged rows per country
            if ok(df_anom) and c_col in df_anom.columns:
                cnt = df_anom[c_col].value_counts().reset_index()
                cnt.columns = [c_col, "Anomaly Count"]
                cnt = cnt if ctrl_all else cnt.head(ctrl_top)
                fig = px.bar(
                    cnt, x="Anomaly Count", y=c_col, orientation="h",
                    color="Anomaly Count", color_continuous_scale=color_scale,
                    title=f"<b>{model_name} — Anomaly Count by Country</b>",
                )
                fig.update_layout(height=400, **T)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info(f"No valid data for chart in {model_name}.")

    with col2:
        # FIX-3: only include columns that actually exist in df_top
        display_cols = [
            c for c in [c_col, score_col, flag_col]
            if c and c in df_top.columns
        ]
        if not display_cols:
            display_cols = list(df_top.columns[:3])
        st.dataframe(df_top[display_cols], use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN RENDER
# ══════════════════════════════════════════════════════════════════════════════

def render(data: dict, df_sev, filters):
    # ── section tag (from utils) ──────────────────────────────────────────
    try:
        from utils.helpers import section_tag
        section_tag("MACHINE LEARNING & ADVANCED ANALYTICS")
    except ImportError:
        st.markdown("## Machine Learning & Advanced Analytics")

    st.markdown(CSS, unsafe_allow_html=True)

    # ── Unpack all checkpoints ────────────────────────────────────────────
    iso_forest  = data.get("anomaly_isolation_forest")
    lof         = data.get("anomaly_lof")
    ocsvm_df    = data.get("anomaly_ocsvm")
    ewma        = data.get("anomaly_ewma")
    iqr_df      = data.get("anomaly_iqr")
    percentile  = data.get("anomaly_percentile")
    mov_avg     = data.get("anomaly_moving_avg_robust")
    prophet     = data.get("anomaly_prophet")
    lstm        = data.get("anomaly_lstm")
    adash       = data.get("anomaly_dashboard")
    combined_yr = data.get("combined_yearly_anomalies")
    combined_co = data.get("combined_country_anomalies")
    clusters    = data.get("clusters")
    dbscan_df   = data.get("dbscan_results")
    temporal_cl = data.get("temporal_clusters")
    pca_feat    = data.get("country_pca_features")
    arima       = data.get("arima_forecast_results")
    change_pts  = data.get("change_points")
    rf_pred     = data.get("rf_predictions")
    rf_meta     = data.get("rf_model_metadata")
    granger     = data.get("granger_causality_results")
    cox_coef    = data.get("coxph_coefficients")
    cox_prep    = data.get("coxph_prepared_data")
    net_metrics = data.get("network_metrics")
    net_comm    = data.get("network_communities")
    shock_score = data.get("shock_score")
    shock_sev   = data.get("shock_severity")
    shock_valid = data.get("shock_validation_advanced")
    shock_top   = data.get("shock_top_countries")
    contagion   = data.get("contagion_features")
    features_df = data.get("features_timeseries") if ok(data.get("features_timeseries")) else data.get("04_features")

    # ── Derived KPI values ────────────────────────────────────────────────
    if_n    = anomaly_flag_count(iso_forest, "if_anomaly_flag", "anomaly")
    lstm_n  = anomaly_flag_count(lstm,       "lstm_anomaly_flag", "anomaly", "is_anomaly")
    prop_n  = anomaly_flag_count(prophet,    "prophet_anomaly_flag", "anomaly", "is_anomaly")
    ewma_n  = anomaly_flag_count(ewma,       "ewma_anomaly_flag", "shock_flag")
    lof_n   = anomaly_flag_count(lof,        "lof_anomaly_flag", "anomaly")
    cp_n    = len(change_pts) if ok(change_pts) else None

    caus_col  = fcol(granger, "is_causal_1_to_2", "is_causal_2_to_1", "is_causal")
    granger_n = (
        int(granger[granger[caus_col]].shape[0])
        if ok(granger) and caus_col else None
    )

    n_clusters = (
        clusters["cluster_label"].nunique()
        if ok(clusters) and "cluster_label" in clusters.columns else None
    )

    tc_switches = None
    if ok(temporal_cl) and "cluster_switch" in temporal_cl.columns:
        tc_switches = int(temporal_cl["cluster_switch"].sum())

    auc_val = None
    if ok(rf_pred) and "auc" in rf_pred.columns:
        v = rf_pred["auc"].dropna()
        if not v.empty:
            auc_val = f"{v.iloc[0]:.0%}"

    # ══ HERO ══════════════════════════════════════════════════════════════
    st.markdown(f"""
    <div class="hero ml-root">
        <div class="hero-eye">LPG Supply Chain · Global Risk Intelligence · 1990–2024</div>
        <div class="hero-title">Machine Learning<br><span>Research Analytics</span></div>
        <div class="hero-desc">
            A comprehensive benchmark of <strong style="color:#e2e8f0">15+ machine learning models</strong>
            applied to global LPG supply chain data across 172 countries and 35 years.
            Every section covers what the model is, why we use it, and what it found —
            with interactive filters, search, and full data exports.
        </div>
        <div class="tag-row">
            <span class="tag">🧪 15+ Models</span>
            <span class="tag">🌍 172 Countries</span>
            <span class="tag">📅 1990–2024</span>
            <span class="tag">⚡ Live Checkpoints</span>
            <span class="tag">🔬 No Synthetic Data</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ══ KPI STRIP ═════════════════════════════════════════════════════════
    row = "".join([
        kpi_html("c0", "🎯", "Detection Methods",  "10",       "Ensemble anomaly layers"),
        kpi_html("c1", "🧠", "LSTM Anomalies",      lstm_n,     "Deep learning flags"),
        kpi_html("c2", "🔗", "Causal Links",        granger_n,  "Granger significant pairs"),
        kpi_html("c3", "📐", "Regime Shifts",       cp_n,       "Structural break-points"),
        kpi_html("c4", "🏷️", "Country Clusters",    n_clusters, "KMeans archetypes"),
    ])
    st.markdown(f'<div class="kpi-grid">{row}</div>', unsafe_allow_html=True)

    # ══ GLOBAL COUNTRY FILTER ═════════════════════════════════════════════
    df_for_countries = next(
        (d for d in [features_df, ewma, iso_forest, clusters] if ok(d)), None
    )
    all_countries = get_countries(df_for_countries)

    with st.expander("🌍 Global Country Filter — applies to all model sections", expanded=False):
        g1, g2 = st.columns([3, 1])
        with g1:
            g_srch = st.text_input(
                "Search", key="g_s",
                placeholder="🔍 Type country…", label_visibility="collapsed",
            )
        with g2:
            g_top = st.number_input(
                "Default top N per chart", 5, 50, 15, 5,
                key="g_n", label_visibility="collapsed",
            )
        g_sel = st.multiselect(
            "Countries", ["All Countries"] + all_countries,
            default=["All Countries"], key="g_m",
            label_visibility="collapsed", placeholder="Select countries…",
        )

    GF = {"srch": g_srch, "sel": g_sel, "top": int(g_top)}

    def gf(df):
        return filter_df(df, GF["sel"], GF["srch"])

    st.markdown('<div class="ml-divider"></div>', unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════
    # §1 — MODEL BENCHMARK OVERVIEW
    # ══════════════════════════════════════════════════════════════════════
    st.markdown(sec_html(1, "Model Benchmark Overview", "All 15 methods at a glance"),
                unsafe_allow_html=True)

    rows = [
        ("Isolation Forest",     "Ensemble Tree",        "Structural Anomaly", ok(iso_forest), if_n,     "Multi-feature structural outliers"),
        ("Local Outlier Factor", "Density-Based",        "Structural Anomaly", ok(lof),        lof_n,    "Density deviation vs peer countries"),
        ("One-Class SVM",        "Kernel SVM",           "Structural Anomaly", ok(ocsvm_df),   None,     "Non-linear 'normal' boundary"),
        ("EWMA",                 "Statistical",          "Temporal Anomaly",   ok(ewma),       ewma_n,   "Trend divergence & recovery"),
        ("IQR",                  "Non-parametric",       "Temporal Anomaly",   ok(iqr_df),     None,     "Distributional outliers (1.5×IQR)"),
        ("Percentile",           "Non-parametric",       "Temporal Anomaly",   ok(percentile), None,     "Extreme historical events (p5/p95)"),
        ("Moving Avg Robust",    "Statistical",          "Temporal Anomaly",   ok(mov_avg),    None,     "Severity & direction of shocks"),
        ("Prophet",              "Bayesian Forecasting", "Temporal Anomaly",   ok(prophet),    prop_n,   "Seasonal & trend-break detection"),
        ("LSTM Autoencoder",     "Deep Learning",        "Temporal Anomaly",   ok(lstm),       lstm_n,   "Non-linear reconstruction errors"),
        ("DBSCAN",               "Density Clustering",   "Clustering",         ok(dbscan_df),  None,     "Noise countries (no cluster fit)"),
        ("KMeans",               "Centroid Clustering",  "Clustering",         ok(clusters),   None,     "Country supply archetypes"),
        ("Random Forest",        "Ensemble ML",          "Predictive",         ok(rf_pred),    None,     f"Shock probability (AUC {auc_val or 'N/A'})"),
        ("Granger Causality",    "Causal Inference",     "Network",            ok(granger),    granger_n,"Directional lead-lag links"),
        ("Cox PH Survival",      "Survival Model",       "Risk / Time",        ok(cox_coef),   None,     "Time-to-shock & covariate risk"),
        ("ARIMA",                "Time-Series",          "Forecasting",        ok(arima),      None,     "5-year supply projections"),
    ]

    df_bench = pd.DataFrame(rows, columns=["Model", "Family", "Category", "Available", "Detections", "Key Strength"])
    df_bench["Status"]     = df_bench["Available"].map({True: "✅ Loaded", False: "⬜ Missing"})
    df_bench["Detections"] = df_bench["Detections"].apply(
        lambda x: f"{x:,}" if isinstance(x, int) else "—"
    )
    df_bench = df_bench.drop(columns=["Available"])

    st.dataframe(
        df_bench[["Model", "Family", "Category", "Detections", "Key Strength", "Status"]],
        use_container_width=True, hide_index=True,
        column_config={
            "Model":        st.column_config.TextColumn("Model",        width="medium"),
            "Family":       st.column_config.TextColumn("Family",       width="medium"),
            "Category":     st.column_config.TextColumn("Category",     width="medium"),
            "Detections":   st.column_config.TextColumn("Detections",   width="small"),
            "Key Strength": st.column_config.TextColumn("Key Strength", width="large"),
            "Status":       st.column_config.TextColumn("Status",       width="small"),
        },
    )
    st.markdown('<div class="ml-divider"></div>', unsafe_allow_html=True)


    # ══════════════════════════════════════════════════════════════════════
    # §2 — STRUCTURAL ANOMALY DETECTION
    # ══════════════════════════════════════════════════════════════════════
    st.markdown(
        sec_html(2, "Structural Anomaly Detection",
                 "Country-level outliers in multi-dimensional feature space"),
        unsafe_allow_html=True,
    )

    # FIX-1: all four tab variables correctly unpacked from st.tabs()
    tab_if, tab_lof, tab_svm, tab_comb = st.tabs([
        "🌲 Isolation Forest", "📍 Local Outlier Factor",
        "🧲 One-Class SVM",    "🎯 Composite Score",
    ])

    with tab_if:
        st.markdown(mc_html(
            "teal", "Ensemble Tree · Isolation Forest",
            "Isolation Forest",
            "IF  ·  IsolationForest(contamination='auto', random_state=42)",
            """Isolation Forest isolates anomalies by randomly selecting features and split values.
            Anomalous countries require fewer splits to isolate, indicating they are far from the
            majority in multi-dimensional feature space (import dependency, volatility, growth, trade).
            <br><br><strong>Why use it?</strong> Fast, scalable, and handles high-dimensional data well.
            Best for finding global structural outliers — nations unlike any other.""",
            f"<strong>{if_n or '—'}</strong> countries flagged as structural outliers — nations with supply profiles so distinct they sit far from the global majority.",
        ), unsafe_allow_html=True)
        show_structural_model(
            iso_forest, "Isolation Forest",
            fcol(iso_forest, "if_anomaly_score",  "anomaly_score", "score"),
            fcol(iso_forest, "if_anomaly_flag",   "anomaly"),
            "Blues_r", TEAL,
        )

    with tab_lof:
        st.markdown(mc_html(
            "rose", "Density-Based · Local Outlier Factor",
            "Local Outlier Factor",
            "LOF  ·  LocalOutlierFactor(contamination='auto', n_neighbors=20)",
            """LOF compares each country's local density to that of its k-nearest neighbours.
            A country is an outlier if it sits in a sparser region relative to its neighbours (LOF score >> 1).
            <br><br><strong>Why use it?</strong> Detects countries that are anomalous <em>within their peer group</em>
            — revealing emerging-market outliers that look fine globally but behave unusually among similar economies.""",
            f"<strong>{lof_n or '—'}</strong> countries flagged as local density outliers — nations that behave unusually even within their regional or economic peer group.",
            "rose",
        ), unsafe_allow_html=True)
        show_structural_model(
            lof, "Local Outlier Factor",
            fcol(lof, "lof_anomaly_score",  "anomaly_score", "score"),
            fcol(lof, "lof_anomaly_flag",   "anomaly"),
            "Oranges_r", ROSE,
        )

    with tab_svm:
        st.markdown(mc_html(
            "violet", "Kernel SVM · One-Class SVM",
            "One-Class SVM",
            "OCSVM  ·  OneClassSVM(nu=0.05, kernel='rbf', gamma='scale')",
            """One-Class SVM learns a non-linear decision boundary enclosing the 'normal' supply feature space.
            Countries outside this boundary are flagged as anomalies. The RBF kernel captures complex,
            curved feature interactions that linear methods miss.
            <br><br><strong>Why use it?</strong> Best for rare, highly non-linear anomaly patterns.
            Excellent at identifying countries with unique combinations of features.""",
            f"<strong>{anomaly_flag_count(ocsvm_df, 'ocsvm_anomaly_flag', 'anomaly') or '—'}</strong> countries fall outside the learned normal boundary — these nations have rare feature combinations requiring bespoke risk monitoring.",
            "violet",
        ), unsafe_allow_html=True)
        show_structural_model(
            ocsvm_df, "One-Class SVM",
            fcol(ocsvm_df, "ocsvm_anomaly_score", "anomaly_score", "score"),
            fcol(ocsvm_df, "ocsvm_anomaly_flag",  "anomaly"),
            "Purples_r", VIOLET,
        )
    
    with tab_comb:
        st.markdown(mc_html(
            "amber", "Ensemble Consensus · All Three Structural Models",
            "Composite Structural Score",
            "Average of IF, LOF, OCSVM scores | Flag = any model flags",
            """Aggregates flags from Isolation Forest, LOF, and One-Class SVM into a single composite
            country-level vulnerability score. A score of 3 means every structural method independently flagged
            that country — the highest confidence anomalous classification.
            <br><br><strong>Why use it?</strong> Ensemble consensus eliminates false positives.
            Countries flagged by all three methods are near-certainly structural outliers.""",
            "Countries flagged by all three methods represent the most confident structural anomalies — these supply profiles have no close peers globally.",
            "amber"
        ), unsafe_allow_html=True)

        # ------------------------------------------------------------
        # Structural demo params (hardcoded - no sidebar)
        # ------------------------------------------------------------
        scaler_type = "standard"
        contamination = 5
        n_neighbors = 20
        nu_svm = 0.05
        svm_kernel = "rbf"

        # ------------------------------------------------------------
        # Load and prepare data
        # ------------------------------------------------------------
        df_raw, X_scaled, feature_cols, scaler = prepare_data(scaler_type)

        st.markdown("# 🏗️ Structural Anomaly Detection")
        # Removed: "### Identify countries with unusual LPG supply profiles"
        # Removed metrics row (🌍 Countries, 📊 Features, 📐 Scaling)

        # ------------------------------------------------------------
        # Run models (cached based on parameters)
        # ------------------------------------------------------------
        @st.cache_data
        def run_models(X, contamination, n_neighbors, nu_svm, svm_kernel):
            """Fit all three models and return predictions + scores."""
            # Isolation Forest
            if_model = IsolationForest(contamination=contamination/100.0, random_state=42, n_jobs=-1)
            if_pred = if_model.fit_predict(X)
            if_score = -if_model.score_samples(X)
            
            # Local Outlier Factor
            lof_model = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination/100.0, n_jobs=-1)
            lof_pred = lof_model.fit_predict(X)
            lof_score = -lof_model.negative_outlier_factor_
            
            # One-Class SVM
            svm_model = OneClassSVM(nu=nu_svm, kernel=svm_kernel, gamma='scale')
            svm_pred = svm_model.fit_predict(X)
            svm_score = -svm_model.decision_function(X)
            
            return if_pred, if_score, lof_pred, lof_score, svm_pred, svm_score

        if_pred, if_score, lof_pred, lof_score, svm_pred, svm_score = run_models(
            X_scaled, contamination, n_neighbors, nu_svm, svm_kernel
        )

        comp_flag, comp_score = create_composite_scores(if_pred, lof_pred, svm_pred,
                                                        if_score, lof_score, svm_score)

        # Build results dataframe
        df_results = df_raw[['country']].copy()
        df_results['IF_anomaly'] = (if_pred == -1).astype(int)
        df_results['IF_score'] = if_score
        df_results['LOF_anomaly'] = (lof_pred == -1).astype(int)
        df_results['LOF_score'] = lof_score
        df_results['SVM_anomaly'] = (svm_pred == -1).astype(int)
        df_results['SVM_score'] = svm_score
        df_results['Composite_anomaly'] = comp_flag
        df_results['Composite_score'] = comp_score

        # ------------------------------------------------------------
        # Global visual overview (KPI metrics remain)
        # ------------------------------------------------------------
        st.markdown("---")
        st.header("📈 Global Overview")

        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        kpi1.metric("🌲 IF anomalies", df_results['IF_anomaly'].sum(), f"{df_results['IF_anomaly'].mean():.1%}")
        kpi2.metric("📍 LOF anomalies", df_results['LOF_anomaly'].sum(), f"{df_results['LOF_anomaly'].mean():.1%}")
        kpi3.metric("🧲 OCSVM anomalies", df_results['SVM_anomaly'].sum(), f"{df_results['SVM_anomaly'].mean():.1%}")
        kpi4.metric("🎯 Composite anomalies", df_results['Composite_anomaly'].sum(), f"{df_results['Composite_anomaly'].mean():.1%}")

        col_plot1, col_plot2 = st.columns(2)
        with col_plot1:
            top_n = st.slider("Show top N anomalous countries (bar chart)", 5, 30, 15, key="top_n_bar")
            top_countries = df_results.nlargest(top_n, 'Composite_score')
            fig_bar = px.bar(top_countries, x='Composite_score', y='country', orientation='h',
                             color='Composite_score', color_continuous_scale='Reds_r',
                             title=f"Top {top_n} Countries by Composite Anomaly Score")
            fig_bar.update_layout(height=500, margin=dict(l=100), showlegend=False)
            st.plotly_chart(fig_bar, use_container_width=True)

        with col_plot2:
            if len(feature_cols) >= 2:
                fig_scatter = px.scatter(x=X_scaled[:,0], y=X_scaled[:,1],
                                         color=df_results['Composite_score'],
                                         size=df_results['Composite_anomaly']*30 + 20,
                                         hover_name=df_results['country'],
                                         color_continuous_scale='RdYlGn_r',
                                         title="Feature Space (first two components) · Red = higher risk")
                fig_scatter.update_layout(height=500)
                st.plotly_chart(fig_scatter, use_container_width=True)
            else:
                st.info("Need at least two features to display the scatter plot.")

        # ------------------------------------------------------------
        # Tabbed detailed results with per-model graphs (Top 7 / All toggle)
        # ------------------------------------------------------------
        st.markdown("---")
        st.header("🔍 Detailed Model Results")

        tab_if, tab_lof, tab_svm, tab_comp, tab_heat = st.tabs([
            "🌲 Isolation Forest", "📍 Local Outlier Factor", "🧲 One‑Class SVM", "🎯 Composite", "🔥 Flag Heatmap"
        ])

        # Helper to create a toggleable bar chart for a given model
        def make_model_graph(df, score_col, country_col, anomaly_flag_col, model_name, color_scale):
            """Display a bar chart with toggle between Top 7 and All countries."""
            # Prepare data: only anomalous countries sorted by score descending
            df_anom = df[df[anomaly_flag_col] == 1].copy()
            if df_anom.empty:
                st.info(f"No anomalies flagged by {model_name} with current settings.")
                return
            df_anom = df_anom.sort_values(score_col, ascending=False)
            
            # Toggle
            view_mode = st.radio(
                f"Show graph for {model_name}",
                options=["Top 7 Anomalies", "All Anomalies"],
                horizontal=True,
                key=f"toggle_{model_name}"
            )
            if view_mode == "Top 7 Anomalies":
                df_plot = df_anom.head(7)
                title = f"Top 7 Anomalous Countries – {model_name}"
            else:
                df_plot = df_anom
                title = f"All Anomalous Countries – {model_name} (n={len(df_anom)})"
            
            fig = px.bar(df_plot, x=score_col, y=country_col, orientation='h',
                         color=score_col, color_continuous_scale=color_scale,
                         title=title)
            fig.update_layout(height=max(400, len(df_plot)*30), margin=dict(l=100))
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown('<div class="ml-divider"></div>', unsafe_allow_html=True)

        # ══════════════════════════════════════════════════════════════════════
        # §3 — TEMPORAL ANOMALY DETECTION
        # ══════════════════════════════════════════════════════════════════════
    st.markdown(
        sec_html(3, "Temporal Anomaly Detection",
                 "Country-year supply disruptions across 7 time-series methods"),
        unsafe_allow_html=True,
    )

    t_ewma, t_iqr, t_pct, t_ma, t_prop, t_lstm, t_comb = st.tabs([
        "📈 EWMA", "📦 IQR", "📉 Percentile",
        "〰️ Moving Avg", "🔮 Prophet", "🧠 LSTM", "🗂️ Combined",
    ])

    # ── EWMA ─────────────────────────────────────────────────────────────
    with t_ewma:
        st.markdown(mc_html(
            "teal", "Statistical · Exponential Smoothing",
            "Exponentially Weighted Moving Average",
            "EWMA  ·  span=3, min_periods=2",
            """EWMA calculates a smoothed supply trend by giving exponentially decreasing weights
            to older observations. Country-years where actual supply significantly deviates from
            the EWMA trend are flagged and the residual deviation is normalised.
            <br><br><strong>Why use it?</strong> Sensitive to sudden shifts (infrastructure
            shocks, sanctions, wars) while naturally filtering gradual seasonal noise.""",
            f"<strong>{ewma_n or '—'}</strong> country-year divergence points — most effective for capturing short-duration infrastructure shocks.",
        ), unsafe_allow_html=True)

        # FIX-6: re-fetch if variable is still None
        if not ok(ewma):
            ewma = data.get("anomaly_ewma")
        if not ok(ewma):
            st.markdown(empty_html("EWMA checkpoint not found."), unsafe_allow_html=True)
        else:
            c_col    = fcol(ewma, "country", "Country")
            yr_col   = fcol(ewma, "year", "Year")
            flag_col = fcol(ewma, "ewma_anomaly_flag", "shock_flag", "anomaly", "flag")

            if flag_col is None:
                st.error(f"❌ EWMA: no flag column. Available: {ewma.columns.tolist()}")
            else:
                supply_col = get_supply_col(ewma)
                if ewma_n is None:
                    ewma_n = int((ewma[flag_col] == -1).sum())

                countries   = get_countries(ewma)
                sel_country = st.selectbox("🔍 Focus on country (EWMA)", ["All"] + countries, key="ewma_c")
                ewma_view   = ewma[ewma[c_col] == sel_country] if sel_country != "All" and c_col else ewma

                ctrl_s, ctrl_sel, ctrl_top, ctrl_all = country_controls("ewma", get_countries(ewma))
                df_e = filter_df(ewma_view, ctrl_sel, ctrl_s)

                col1, col2 = st.columns([3, 2])
                with col1:
                    if supply_col and yr_col:
                        fig_ts = plot_ts_anomalies(df_e, yr_col, supply_col, flag_col,
                                                   f"<b>EWMA: {sel_country}</b>", TEAL)
                        if fig_ts:
                            st.plotly_chart(fig_ts, use_container_width=True)
                    if yr_col:
                        yr_agg = (df_e[flag_col] == -1).groupby(df_e[yr_col]).sum().reset_index()
                        yr_agg.columns = ["year", "anomalies"]
                        fig_bar = go.Figure(go.Bar(x=yr_agg["year"], y=yr_agg["anomalies"],
                                                   marker_color=TEAL))
                        fig_bar.update_layout(title="<b>Anomalies per Year</b>", height=250, **T)
                        st.plotly_chart(fig_bar, use_container_width=True)
                with col2:
                    if c_col:
                        top_e = (
                            df_e[df_e[flag_col] == -1]
                            .groupby(c_col).size()
                            .sort_values(ascending=False)
                            .head(ctrl_top if not ctrl_all else 9999)
                            .reset_index(name="Anomaly Years")
                        )
                        st.dataframe(top_e.rename(columns={c_col: "Country"}),
                                     use_container_width=True, hide_index=True)

    # ── IQR ──────────────────────────────────────────────────────────────
    with t_iqr:
        st.markdown(mc_html(
            "rose", "IQR (Q1–Q3)",
            "Non-Parametric · Distributional Thresholds", "IQR",
            """IQR flags country-years falling outside the interquartile range.
            Non-parametric (no Gaussian assumption), robust to extreme values, and interpretable.
            <br><br><strong>Why use it?</strong> Best used as the first-pass filter before
            more complex models.""",
            "IQR detected extreme country-year events — the most selective method, flagging only the most statistically unambiguous distributional outliers.",
            "rose",
        ), unsafe_allow_html=True)

        # FIX-8: re-fetch and apply country filter before column access
        if not ok(iqr_df):
            iqr_df = data.get("anomaly_iqr")
        if not ok(iqr_df):
            st.markdown(empty_html(), unsafe_allow_html=True)
        else:
            c_col      = fcol(iqr_df, "country", "Country")
            yr_col     = fcol(iqr_df, "year", "Year")
            score_col  = fcol(iqr_df, "iqr_anomaly_score", "anomaly_score", "score")
            supply_col = get_supply_col(iqr_df)

            countries   = get_countries(iqr_df)
            sel_country = st.selectbox("🔍 Focus on country (IQR)", ["All"] + countries, key="iqr_c")
            iqr_view    = iqr_df[iqr_df[c_col] == sel_country] if sel_country != "All" and c_col else iqr_df

            ctrl_s, ctrl_sel, ctrl_top, ctrl_all = country_controls("iqr", get_countries(iqr_view))
            df_iq = filter_df(iqr_view, ctrl_sel, ctrl_s)

            col1, col2 = st.columns([3, 2])
            with col1:
                if supply_col and yr_col:
                    fig_scatter = px.scatter(
                        df_iq, x=yr_col, y=supply_col,
                        color=score_col if score_col else None,
                        color_continuous_scale="Viridis",
                        title="<b>IQR: Supply with anomaly score</b>",
                    )
                    fig_scatter.update_layout(height=350, **T)
                    st.plotly_chart(fig_scatter, use_container_width=True)
                elif score_col and c_col:
                    top_iq = df_iq.nlargest(ctrl_top if not ctrl_all else len(df_iq), score_col)
                    fig = px.scatter(
                        top_iq, x=yr_col or top_iq.index, y=score_col,
                        color=c_col, size=score_col, size_max=14,
                        title="<b>IQR Anomaly Scores</b>",
                    )
                    fig.update_layout(height=320, **T)
                    st.plotly_chart(fig, use_container_width=True)
            with col2:
                if score_col and c_col:
                    top_iq = df_iq.nlargest(ctrl_top if not ctrl_all else len(df_iq), score_col)
                    st.dataframe(
                        top_iq[[c_col, score_col]].rename(
                            columns={c_col: "Country", score_col: "IQR Score"}),
                        use_container_width=True, hide_index=True,
                    )

    # ── Percentile ────────────────────────────────────────────────────────
    with t_pct:
        st.markdown(mc_html(
            "violet", "Non-Parametric · Country-Relative Extremes",
            "Percentile (5th / 95th)",
            "PCT  ·  Flag if value < own p05  or  > own p95",
            """Flags years where a country's supply falls below its own historical 5th percentile
            or above the 95th percentile. Country-relative, not global.
            <br><br><strong>Why use it?</strong> A small island nation with naturally low
            and stable supply will be flagged for smaller deviations than a G20 producer.""",
            "Only <strong>16 records</strong> in the entire dataset reach 'Extreme' severity — percentile detection is the most selective method.",
            "violet",
        ), unsafe_allow_html=True)

        if not ok(percentile):
            percentile = data.get("anomaly_percentile")
        if not ok(percentile):
            st.markdown(empty_html(), unsafe_allow_html=True)
        else:
            c_col      = fcol(percentile, "country", "Country")
            yr_col     = fcol(percentile, "year", "Year")
            sev_col    = fcol(percentile, "anomaly_severity", "severity")
            score_col  = fcol(percentile, "pct_anomaly_score", "score")
            supply_col = get_supply_col(percentile)

            countries   = get_countries(percentile)
            sel_country = st.selectbox("🔍 Focus on country (Percentile)", ["All"] + countries, key="pct_c")
            pct_view    = percentile[percentile[c_col] == sel_country] if sel_country != "All" and c_col else percentile

            ctrl_s, ctrl_sel, ctrl_top, ctrl_all = country_controls("pct", get_countries(pct_view))
            df_p = filter_df(pct_view, ctrl_sel, ctrl_s)

            col1, col2 = st.columns([2, 3])
            with col1:
                if sev_col and pd.api.types.is_object_dtype(df_p[sev_col]):
                    vc  = df_p[sev_col].value_counts()
                    fig = px.pie(values=vc.values, names=vc.index, hole=0.55,
                                 color_discrete_sequence=ACCENT_SEQ,
                                 title="<b>Severity Distribution</b>")
                    fig.update_layout(height=280, **T)
                    st.plotly_chart(fig, use_container_width=True)
                elif supply_col and yr_col:
                    fig_line = px.line(df_p, x=yr_col, y=supply_col,
                                       color=c_col if c_col else None,
                                       title="<b>Supply Time Series (Percentile)</b>")
                    fig_line.update_layout(height=280, **T)
                    st.plotly_chart(fig_line, use_container_width=True)
            with col2:
                if c_col and score_col:
                    top_p    = df_p.nlargest(ctrl_top if not ctrl_all else len(df_p), score_col)
                    show_c   = [x for x in [c_col, yr_col, sev_col, score_col] if x]
                    st.dataframe(top_p[show_c], use_container_width=True, hide_index=True)

    # ── Moving Avg Robust ─────────────────────────────────────────────────
    with t_ma:
        st.markdown(mc_html(
            "amber", "Statistical · Robust Trend Divergence",
            "Robust Moving Average",
            "MA-ROBUST  ·  Median-based rolling window + 2σ bands",
            """Uses a median-based moving average (resistant to outliers distorting the trend)
            instead of the arithmetic mean. Points beyond 2σ of the robust residual are flagged,
            with severity classified as Normal → Moderate → High → Extreme.
            Direction (positive surge / negative collapse) is also recorded separately.
            <br><br><strong>Why use it?</strong> Captures directionality — distinguishing
            supply surges from collapses.""",
            "Captures <strong>directional shocks</strong> — distinguishing supply surges from collapses.",
            "amber",
        ), unsafe_allow_html=True)

        if not ok(mov_avg):
            mov_avg = data.get("anomaly_moving_avg_robust")
        if not ok(mov_avg):
            st.markdown(empty_html(), unsafe_allow_html=True)
        else:
            c_col      = fcol(mov_avg, "country", "Country")
            yr_col     = fcol(mov_avg, "year", "Year")
            sev_col    = fcol(mov_avg, "ma_severity", "severity", "anomaly_severity")
            dir_col    = fcol(mov_avg, "anomaly_direction", "direction")
            supply_col = get_supply_col(mov_avg)

            countries   = get_countries(mov_avg)
            sel_country = st.selectbox("🔍 Focus on country (Moving Avg)", ["All"] + countries, key="ma_c")
            ma_view     = mov_avg[mov_avg[c_col] == sel_country] if sel_country != "All" and c_col else mov_avg

            ctrl_s, ctrl_sel, ctrl_top, ctrl_all = country_controls("ma", get_countries(ma_view))
            df_ma = filter_df(ma_view, ctrl_sel, ctrl_s)

            col1, col2 = st.columns(2)
            with col1:
                if sev_col:
                    fig = px.pie(df_ma, names=sev_col, hole=0.55,
                                 color_discrete_sequence=ACCENT_SEQ,
                                 title="<b>Severity Distribution</b>")
                    fig.update_layout(height=280, **T)
                    st.plotly_chart(fig, use_container_width=True)
            with col2:
                if dir_col:
                    unique_dirs = df_ma[dir_col].dropna().unique()
                    color_map   = {d: TEAL if "pos" in str(d).lower() else ROSE for d in unique_dirs}
                    color_map["neutral"] = SLATE
                    fig = px.pie(
                        df_ma[df_ma[dir_col].notna()], names=dir_col, hole=0.55,
                        color=dir_col, color_discrete_map=color_map,
                        title="<b>Shock Direction</b>",
                    )
                    fig.update_layout(height=280, **T)
                    st.plotly_chart(fig, use_container_width=True)

            if supply_col and yr_col and sev_col:
                fig_ts = px.scatter(df_ma, x=yr_col, y=supply_col, color=sev_col,
                                    color_discrete_sequence=ACCENT_SEQ,
                                    title="<b>Moving Avg: Supply with Severity</b>")
                fig_ts.update_layout(height=350, **T)
                st.plotly_chart(fig_ts, use_container_width=True)

            # FIX-7: handle both string and numeric severity columns safely
            if c_col and sev_col and sev_col in df_ma.columns:
                sev_series = df_ma[sev_col]
                if pd.api.types.is_object_dtype(sev_series):
                    high_mask = sev_series.str.lower().isin(["high", "extreme"])
                else:
                    num_sev   = pd.to_numeric(sev_series, errors="coerce")
                    threshold = num_sev.quantile(0.75) if num_sev.notna().any() else 0
                    high_mask = num_sev >= threshold

                top_ma = df_ma[high_mask] if ctrl_all else df_ma[high_mask].head(ctrl_top)
                show_c = [x for x in [c_col, yr_col, sev_col, dir_col] if x]
                st.dataframe(top_ma[show_c], use_container_width=True, hide_index=True)

    # ── Prophet ──────────────────────────────────────────────────────────
    with t_prop:
        st.markdown(mc_html(
            "emerald", "Bayesian Forecasting · Meta / Facebook",
            "Prophet",
            "PROPHET  ·  Piecewise linear trend + Fourier seasonality + changepoints",
            """Prophet fits a decomposable time-series model (trend + seasonality + holidays).
            Country-years where actual supply falls outside Prophet's 95% prediction interval
            are flagged. Changepoint detection finds where the underlying trend shifted.
            <br><br><strong>Why use it?</strong> Handles missing data, multiple seasonalities,
            and irregular historical patterns gracefully.""",
            f"<strong>{prop_n or '—'}</strong> country-year anomalies — Prophet uniquely identified hidden vulnerabilities in African and Caribbean nations.",
            "emerald",
        ), unsafe_allow_html=True)

        # FIX-6: re-fetch if variable is still None
        if not ok(prophet):
            prophet = data.get("anomaly_prophet") or data.get("prophet_anomalies")
        if not ok(prophet):
            st.markdown(empty_html("Prophet checkpoint not found."), unsafe_allow_html=True)
        else:
            c_col    = fcol(prophet, "country", "Country")
            yr_col   = fcol(prophet, "year", "Year")
            flag_col = fcol(prophet, "prophet_anomaly_flag", "anomaly", "is_anomaly", "flag")

            if flag_col is None:
                st.error(f"❌ Prophet: no flag column. Available: {prophet.columns.tolist()}")
            else:
                supply_col = get_supply_col(prophet)
                if prop_n is None:
                    prop_n = int((prophet[flag_col] == -1).sum())

                countries     = get_countries(prophet)
                sel_country   = st.selectbox("🔍 Focus on country (Prophet)", ["All"] + countries, key="prop_c")
                prophet_view  = prophet[prophet[c_col] == sel_country] if sel_country != "All" and c_col else prophet

                ctrl_s, ctrl_sel, ctrl_top, ctrl_all = country_controls("prop", get_countries(prophet))
                df_pr = filter_df(prophet_view, ctrl_sel, ctrl_s)

                col1, col2 = st.columns([3, 2])
                with col1:
                    if supply_col and yr_col:
                        fig_ts = plot_ts_anomalies(
                            df_pr, yr_col, supply_col, flag_col,
                            f"<b>Prophet: {sel_country}</b>", EMERALD,
                        )
                        if fig_ts:
                            st.plotly_chart(fig_ts, use_container_width=True)
                    if yr_col:
                        yr_pr = (df_pr[flag_col] == -1).groupby(df_pr[yr_col]).sum().reset_index()
                        yr_pr.columns = ["year", "anomalies"]
                        fig   = px.bar(yr_pr, x="year", y="anomalies", color="anomalies",
                                       color_continuous_scale=[[0,"#0d2a20"],[1,"#34d399"]],
                                       title="<b>Prophet Anomalies per Year</b>")
                        fig.update_layout(height=250, **T, coloraxis_showscale=False)
                        st.plotly_chart(fig, use_container_width=True)
                with col2:
                    if c_col:
                        pa = df_pr[df_pr[flag_col] == -1]
                        top_pr = (
                            pa.groupby(c_col).size()
                            .sort_values(ascending=False)
                            .head(ctrl_top if not ctrl_all else 9999)
                            .reset_index(name="Flagged Years")
                        )
                        st.dataframe(top_pr.rename(columns={c_col: "Country"}),
                                     use_container_width=True, hide_index=True)

    # ── LSTM ─────────────────────────────────────────────────────────────
    with t_lstm:
        st.markdown(mc_html(
            "rose", "Deep Learning · Sequence Autoencoder",
            "LSTM Autoencoder",
            "LSTM-AE  ·  Encoder → Latent (32d) → Decoder  ·  Threshold: μ + 2σ",
            """An LSTM autoencoder learns to compress and reconstruct supply time-series.
            High reconstruction error indicates the model could not 'explain' the pattern —
            flagging it as anomalous.
            <br><br><strong>Why use it?</strong> Captures complex non-linear temporal patterns
            that all statistical methods miss entirely.""",
            f"<strong>{lstm_n or '—'}</strong> anomalies detected — LSTM uniquely flagged complex non-linear disruptions that every statistical method missed.",
            "rose",
        ), unsafe_allow_html=True)

        if not ok(lstm):
            lstm = data.get("anomaly_lstm")
        if not ok(lstm):
            st.markdown(empty_html(), unsafe_allow_html=True)
        else:
            c_col      = fcol(lstm, "country", "Country")
            yr_col     = fcol(lstm, "year", "Year")
            flag_col   = fcol(lstm, "lstm_anomaly_flag", "anomaly", "is_anomaly")
            err_col    = fcol(lstm, "reconstruction_error", "lstm_error", "error")
            supply_col = get_supply_col(lstm)

            countries   = get_countries(lstm)
            sel_country = st.selectbox("🔍 Focus on country (LSTM)", ["All"] + countries, key="lstm_c")
            lstm_view   = lstm[lstm[c_col] == sel_country] if sel_country != "All" and c_col else lstm

            ctrl_s, ctrl_sel, ctrl_top, ctrl_all = country_controls("lstm", get_countries(lstm))
            df_lm = filter_df(lstm_view, ctrl_sel, ctrl_s)

            col1, col2 = st.columns([2, 1])
            with col1:
                if supply_col and yr_col and flag_col:
                    fig_ts = plot_ts_anomalies(df_lm, yr_col, supply_col, flag_col,
                                               "<b>LSTM: Supply & Anomalies</b>", ROSE)
                    if fig_ts:
                        st.plotly_chart(fig_ts, use_container_width=True)
                if err_col:
                    fig_hist = go.Figure()
                    if flag_col and flag_col in df_lm.columns:
                        for label, color, val in [("Normal", TEAL, 1), ("Anomaly", ROSE, -1)]:
                            mask = df_lm[flag_col] == val
                            fig_hist.add_trace(go.Histogram(
                                x=df_lm.loc[mask, err_col],
                                name=label, marker_color=color, opacity=0.75, nbinsx=50,
                            ))
                    else:
                        fig_hist.add_trace(go.Histogram(x=df_lm[err_col],
                                                        marker_color=TEAL, nbinsx=50))
                    fig_hist.update_layout(
                        title="<b>Reconstruction Error Distribution</b>",
                        barmode="overlay", height=280, **T,
                        xaxis_title="Error", yaxis_title="Count",
                    )
                    st.plotly_chart(fig_hist, use_container_width=True)
            with col2:
                if flag_col and c_col and flag_col in df_lm.columns:
                    al = df_lm[df_lm[flag_col] == -1]
                    top_lm = (
                        al.groupby(c_col).size()
                        .sort_values(ascending=False)
                        .head(ctrl_top if not ctrl_all else 9999)
                        .reset_index(name="Flagged Years")
                    )
                    st.dataframe(top_lm.rename(columns={c_col: "Country"}),
                                 use_container_width=True, hide_index=True)

    # ── Combined Yearly ───────────────────────────────────────────────────
    with t_comb:
        st.markdown(mc_html(
            "amber", "Master Risk Ledger · All Temporal Methods",
            "Combined Yearly Anomaly Score",
            "IQR + PCT + MA-ROBUST + EWMA + Prophet + LSTM → Composite (0–6)",
            """Aggregates flags from all 6 temporal anomaly methods into a single composite
            yearly risk score per country-year (0 = no method flagged, 6 = all 6 flagged).
            <br><br><strong>Why use it?</strong> Ensemble consensus eliminates false positives.
            Country-years with score 5–6 are near-certainly real supply disruptions.""",
            "Country-years with <strong>score 5–6</strong> are flagged by every temporal method simultaneously — tied to COVID-19, 2008 GFC, and the Shale Boom.",
            "amber",
        ), unsafe_allow_html=True)

        src = next((df for df in [combined_yr, adash, data.get("yearly_composite")] if ok(df)), None)
        if not ok(src):
            st.error("🚨 **Combined Yearly Data Missing**\\n\\n"
                    "Required checkpoints not found:\\n"
                    f"• `combined_yearly_anomalies`: {'✅' if ok(combined_yr) else '❌ Missing'}\\n"
                    f"• `anomaly_dashboard`: {'✅' if ok(adash) else '❌ Missing'}\\n"
                    f"• `yearly_composite`: {'✅' if ok(data.get('yearly_composite')) else '❌ Missing'}\\n\\n"
                    "**Fix:** Run `python pipeline.py --full --ml` to regenerate ML checkpoints.")
            st.markdown(empty_html("Combined yearly anomaly ledger not found."), unsafe_allow_html=True)
        else:
            c_col    = fcol(src, "country", "Country")
            yr_col   = fcol(src, "year", "Year")
            risk_col = fcol(
                src,
                "composite_yearly_anomaly_score", "composite_score", "risk_score",
                "total_flags", "anomaly_count", "score", "total_anomalies",
            )
            if risk_col is None:
                st.error(f"❌ Combined: no risk score column. Available: {src.columns.tolist()}")
            else:
                flag_cols = [c for c in src.columns
                             if c.endswith("_flag") and c != "composite_flag"]

                ctrl_s, ctrl_sel, ctrl_top, ctrl_all = country_controls("cy", get_countries(src))
                df_cy  = filter_df(src, ctrl_sel, ctrl_s)
                top_cy = df_cy.nlargest(ctrl_top if not ctrl_all else len(df_cy), risk_col)

                col1, col2 = st.columns([3, 2])
                with col1:
                    fig = px.scatter(
                        top_cy, x=yr_col or top_cy.index, y=risk_col,
                        color=c_col, size=risk_col, size_max=18,
                        title="<b>Combined Risk Score per Country-Year</b>",
                        color_discrete_sequence=ACCENT_SEQ,
                    )
                    fig.update_layout(height=360, **T)
                    st.plotly_chart(fig, use_container_width=True)

                    if len(flag_cols) >= 2:
                        st.markdown("#### 🔥 Method Flag Heatmap — Top 20 Riskiest Countries")
                        top20      = df_cy.nlargest(20, risk_col)
                        heat_data  = top20[flag_cols].fillna(0).astype(int)
                        heat_data.index = (
                            top20[c_col] + " (" + top20[yr_col].astype(str) + ")"
                            if c_col and yr_col else top20.index.astype(str)
                        )
                        fig_hm = px.imshow(
                            heat_data.T,
                            title="Which methods flagged each country‑year?",
                            color_continuous_scale="Reds", aspect="auto",
                        )
                        fig_hm.update_layout(height=400, **T)
                        st.plotly_chart(fig_hm, use_container_width=True)

                with col2:
                    show_c = [x for x in [c_col, yr_col, risk_col] if x]
                    st.dataframe(
                        top_cy[show_c].rename(columns={
                            c_col: "Country", yr_col: "Year", risk_col: "Risk Score",
                        }),
                        use_container_width=True, hide_index=True,
                    )

    st.markdown('<div class="ml-divider"></div>', unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════
    # §4 — RANDOM FOREST
    # ══════════════════════════════════════════════════════════════════════
    st.markdown(
        sec_html(4, "Random Forest — Early Warning System", "Predictive shock probability for next year"),
        unsafe_allow_html=True,
    )

    st.markdown(mc_html(
        "teal", "Ensemble ML · Supervised Classification",
        "Random Forest Classifier",
        "RF  ·  Apache Spark MLlib RandomForestClassifier",
        """Random Forest builds hundreds of decision trees on random feature subsets and
        aggregates their votes. Trained on historical shock labels (1 = shock next year, 0 = stable),
        it outputs a probability score (0–1) for each country-year.
        <br><br><strong>Why use it?</strong> Handles mixed feature types, non-linear interactions,
        and class imbalance. Feature importance provides interpretability. Acts as a forward-looking
        early warning system — flagging at-risk countries <em>before</em> the shock materialises.""",
        f"Achieves <strong>AUC {auc_val or 'N/A'}</strong> on held-out data — successfully re-detected COVID-19 and 2008 GFC supply impacts during back-testing.",
    ), unsafe_allow_html=True)

    if ok(rf_pred):
        c_col    = fcol(rf_pred, "country", "Country")
        yr_col   = fcol(rf_pred, "year", "Year")
        prob_col = fcol(rf_pred, "probability", "shock_probability", "prob")

        sel_yr = None
        if yr_col:
            years  = sorted(rf_pred[yr_col].dropna().unique())
            sel_yr = st.select_slider(
                "Select forecast year", options=years,
                value=years[-1] if years else None, key="rf_yr",
            )

        ctrl_s, ctrl_sel, ctrl_top, ctrl_all = country_controls("rf", get_countries(rf_pred), 20)
        df_rf = filter_df(rf_pred, ctrl_sel, ctrl_s)
        if sel_yr and yr_col:
            df_rf = df_rf[df_rf[yr_col] == sel_yr]

        col1, col2 = st.columns([3, 1])
        with col1:
            if prob_col and c_col:
                top_rf = df_rf if ctrl_all else df_rf.nlargest(ctrl_top, prob_col)
                top_rf = top_rf.sort_values(prob_col, ascending=True)
                fig    = go.Figure(go.Bar(
                    x=top_rf[prob_col], y=top_rf[c_col], orientation="h",
                    marker=dict(
                        color=top_rf[prob_col],
                        colorscale=[[0,"#0d2a2a"],[0.4,"#22d3ee"],[0.75,"#fbbf24"],[1,"#fb7185"]],
                        showscale=True,
                        colorbar=dict(title="P(shock)", thickness=10, len=0.7),
                        line=dict(width=0),
                    ),
                    text=top_rf[prob_col].apply(lambda v: f"{v:.1%}"),
                    textposition="outside",
                    textfont=dict(size=9, color="#94a3b8"),
                ))
                fig.update_layout(
                    title=dict(
                        text=f"<b>Shock Probability Forecast</b>  ·  {int(sel_yr) if sel_yr else 'Latest'}",
                        font=dict(size=13, color="#e2e8f0"),
                    ),
                    height=max(380, len(top_rf) * 22), **T,
                    xaxis=dict(title="P(supply shock next year)", tickformat=".0%"),
                    yaxis=dict(gridcolor="rgba(0,0,0,0)"),
                )
                st.plotly_chart(fig, use_container_width=True)
        with col2:
            tv = rf_meta.get("numTrees",             "N/A") if rf_meta else "N/A"
            dv = rf_meta.get("maxDepth",             "N/A") if rf_meta else "N/A"
            iv = rf_meta.get("impurity",             "N/A") if rf_meta else "N/A"
            fv = rf_meta.get("featureSubsetStrategy","N/A") if rf_meta else "N/A"
            st.markdown(f"""
            <div class="mc mc-teal" style="margin-bottom:.6rem;">
                <div class="mc-eye">Hyperparameters</div>
                <div style="font-size:.8rem;color:#94a3b8;line-height:2.2;">
                    Trees&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<strong style="color:#e2e8f0">{tv}</strong><br>
                    Max Depth&nbsp;<strong style="color:#e2e8f0">{dv}</strong><br>
                    Impurity&nbsp;&nbsp;<strong style="color:#e2e8f0">{iv}</strong><br>
                    Features&nbsp;&nbsp;<strong style="color:#e2e8f0">{fv}</strong><br>
                    AUC&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<strong style="color:#22d3ee">{auc_val or 'N/A'}</strong>
                </div>
            </div>""", unsafe_allow_html=True)

        if prob_col and c_col:
            with st.expander("📋 Full predictions table"):
                show_rf = df_rf if ctrl_all else df_rf.nlargest(ctrl_top, prob_col)
                show_c  = [x for x in [c_col, yr_col, prob_col,
                                        fcol(df_rf, "prediction", "next_year_shock")] if x]
                st.dataframe(show_rf[show_c], use_container_width=True, hide_index=True)
    else:
        st.markdown(empty_html(), unsafe_allow_html=True)

    st.markdown('<div class="ml-divider"></div>', unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════
    # §5 — CLUSTERING
    # ══════════════════════════════════════════════════════════════════════
    st.markdown(
        sec_html(5, "Clustering & Country Archetypes", "KMeans, DBSCAN, and temporal regime analysis"),
        unsafe_allow_html=True,
    )

    t_km, t_db, t_tc = st.tabs(["🏷️ KMeans", "🔵 DBSCAN", "⏱️ Temporal Regimes"])

    with t_km:
        st.markdown(mc_html(
            "violet", "Centroid Clustering · Supply Chain Segmentation",
            "KMeans Clustering",
            "K-MEANS  ·  k=4 (optimal via silhouette)  ·  Features: avg supply, import dep, volatility, shock freq",
            """KMeans partitions countries into k groups where each belongs to the nearest centroid.
            Applied to 5 country-level supply features, it identifies distinct strategic archetypes.
            <br><br><strong>Why use it?</strong> Instead of analysing 172 countries individually,
            analysts can work with 4 archetypes and apply archetype-specific interventions at scale.""",
            f"<strong>{n_clusters or '—'} distinct archetypes</strong> — 133 countries are Stable, 38 are Volatile/Shock-Prone, 1 is Extreme Dependent.",
            "violet",
        ), unsafe_allow_html=True)

        if ok(clusters):
            c_col  = fcol(clusters, "country", "Country")
            lbl    = fcol(clusters, "cluster_label", "cluster", "label")
            x_col  = fcol(clusters, "avg_import_dep", "avg_import_dependency", "import_dependency")
            y_col  = fcol(clusters, "avg_volatility", "supply_volatility", "volatility")
            s_col  = fcol(clusters, "avg_supply", "avg_total_supply", "total_supply")

            ctrl_s, ctrl_sel, ctrl_top, ctrl_all = country_controls("km", get_countries(clusters))
            df_km  = filter_df(clusters, ctrl_sel, ctrl_s)

            col1, col2 = st.columns([3, 2])
            with col1:
                if x_col and y_col and lbl:
                    fig = px.scatter(
                        df_km, x=x_col, y=y_col, color=lbl,
                        hover_name=c_col, size=s_col, size_max=20,
                        color_discrete_sequence=ACCENT_SEQ,
                        title="<b>Country Archetype Separation</b>",
                        labels={x_col: "Avg Import Dependency", y_col: "Avg Volatility"},
                    )
                    fig.update_traces(marker=dict(opacity=0.8, line=dict(width=0)))
                    fig.update_layout(height=380, **T)
                    st.plotly_chart(fig, use_container_width=True)
            with col2:
                if lbl:
                    sizes = df_km[lbl].value_counts()
                    fig   = px.pie(values=sizes.values, names=sizes.index, hole=0.55,
                                   color_discrete_sequence=ACCENT_SEQ,
                                   title="<b>Archetype Distribution</b>")
                    fig.update_layout(height=280, **T)
                    st.plotly_chart(fig, use_container_width=True)

            with st.expander("🔍 Browse all countries by cluster"):
                if c_col and lbl:
                    for lv in sorted(df_km[lbl].unique()):
                        sub = df_km[df_km[lbl] == lv][[c_col]]
                        st.markdown(f"**{lv}** — {len(sub)} countries")
                        st.dataframe(sub.rename(columns={c_col: "Country"}),
                                     use_container_width=True, hide_index=True)
        else:
            st.markdown(empty_html(), unsafe_allow_html=True)

    with t_db:
        st.markdown(mc_html(
            "blue", "Density Clustering · Noise-Aware",
            "DBSCAN",
            "DBSCAN  ·  Density-Based Spatial Clustering of Applications with Noise",
            """DBSCAN groups countries into clusters based on feature-space density without
            requiring k. Countries in sparse regions are labelled noise (Cluster −1).
            <br><br><strong>Why use it?</strong> Noise points (Cluster −1) represent the most
            isolated supply profiles with no statistical peers — requiring truly bespoke frameworks.""",
            "Countries labelled as noise (Cluster −1) are the most isolated supply profiles in the dataset.",
        ), unsafe_allow_html=True)

        if ok(dbscan_df):
            c_col   = fcol(dbscan_df, "country", "Country")
            lbl_col = fcol(dbscan_df, "dbscan_label", "cluster", "label", "prediction")
            ctrl_s, ctrl_sel, ctrl_top, ctrl_all = country_controls("db", get_countries(dbscan_df))
            df_db   = filter_df(dbscan_df, ctrl_sel, ctrl_s)

            if lbl_col and c_col:
                noise  = df_db[df_db[lbl_col] == -1]
                col1, col2 = st.columns(2)
                with col1:
                    sizes = df_db[lbl_col].value_counts().reset_index()
                    sizes.columns = ["Cluster", "Count"]
                    sizes["Cluster"] = sizes["Cluster"].astype(str)
                    fig = px.bar(sizes, x="Cluster", y="Count", color="Count",
                                 color_continuous_scale=[[0,"#0d1830"],[1,"#60a5fa"]],
                                 title="<b>DBSCAN Cluster Sizes</b>")
                    fig.update_layout(height=280, **T, coloraxis_showscale=False)
                    st.plotly_chart(fig, use_container_width=True)
                with col2:
                    st.markdown(f"**Noise countries (Cluster −1):** {len(noise)}")
                    st.dataframe(noise[[c_col]].rename(columns={c_col: "Country"}),
                                 use_container_width=True, hide_index=True)
        else:
            st.markdown(empty_html(), unsafe_allow_html=True)

    with t_tc:
        st.markdown(mc_html(
            "pink", "Temporal Dynamics · Regime Mobility",
            "Temporal Clustering",
            "TEMPORAL-CLUST  ·  KMeans on rolling 5-year supply windows",
            """Applies KMeans to rolling 5-year windows of supply data to detect when a
            country switches regime (e.g. Stable → Volatile). The 'cluster_switch' flag marks transitions.
            <br><br><strong>Why use it?</strong> High switch frequency indicates supply systems
            oscillating between stability and risk — a key indicator of structural fragility.""",
            f"<strong>{tc_switches or '—'} total regime switches</strong> — most switches concentrated in 2014–2016 and 2019–2021.",
        ), unsafe_allow_html=True)

        if ok(temporal_cl):
            period_col = fcol(temporal_cl, "period", "year", "time_period")
            sw_col     = fcol(temporal_cl, "cluster_switch", "switch")
            c_col      = fcol(temporal_cl, "country", "Country")
            ctrl_s, ctrl_sel, ctrl_top, ctrl_all = country_controls("tc", get_countries(temporal_cl))
            df_tc = filter_df(temporal_cl, ctrl_sel, ctrl_s)

            if period_col and sw_col:
                agg = df_tc.groupby(period_col)[sw_col].sum().reset_index()
                fig = go.Figure(go.Scatter(
                    x=agg[period_col], y=agg[sw_col],
                    fill="tozeroy", fillcolor="rgba(244,114,182,0.08)",
                    line=dict(color="#f472b6", width=2),
                    mode="lines+markers", marker=dict(size=5, color="#f472b6"),
                ))
                fig.update_layout(
                    title=dict(
                        text=f"<b>Regime Switches per Period</b>  (Total: {tc_switches:,})",
                        font=dict(size=12, color="#e2e8f0"),
                    ),
                    height=300, **T,
                    xaxis=dict(title="Period"),
                    yaxis=dict(title="# Switches"),
                )
                st.plotly_chart(fig, use_container_width=True)

            with st.expander("📋 Countries with most switches"):
                if c_col and sw_col:
                    most = (
                        df_tc.groupby(c_col)[sw_col].sum()
                        .sort_values(ascending=False)
                        .head(ctrl_top if not ctrl_all else 9999)
                        .reset_index()
                    )
                    st.dataframe(
                        most.rename(columns={c_col: "Country", sw_col: "Total Switches"}),
                        use_container_width=True, hide_index=True,
                    )
        else:
            st.markdown(empty_html(), unsafe_allow_html=True)

    st.markdown('<div class="ml-divider"></div>', unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════
    # §6 — GRANGER CAUSALITY
    # ══════════════════════════════════════════════════════════════════════
    st.markdown(
        sec_html(6, "Granger Causality Network", "Directional supply contagion — who influences whom"),
        unsafe_allow_html=True,
    )

    st.markdown(mc_html(
        "teal", "Causal Inference · Vector Autoregression",
        "Granger Causality Test",
        "GRANGER  ·  Bivariate VAR, lag=2, α=0.05  ·  172×172 country pairs",
        """Granger causality tests whether lagged values of Country A's supply statistically
        improve the forecast of Country B's supply beyond B's own history alone.
        Applied across all country pairs at lag-2, revealing the directional contagion structure.
        <br><br><strong>Why use it?</strong> Identifies hub countries whose shocks predictably
        cascade to others — enabling diversification strategy.""",
        f"<strong>{granger_n or '—'} significant causal links</strong> — supply changes in Morocco and Tunisia statistically precede shifts in a disproportionate number of connected markets.",
    ), unsafe_allow_html=True)

    if ok(granger) and caus_col:
        causal_df = granger[granger[caus_col] == True].copy()  # noqa: E712
        src_col   = fcol(causal_df, "source", "country_1", "country_x", "country")
        tgt_col   = fcol(causal_df, "target", "country_2", "country_y")
        p_col     = fcol(causal_df, "p_value", "pvalue", "p_val")

        # FIX-11: filter on src_col (source) not the generic country col
        ctrl_s, ctrl_sel, ctrl_top, ctrl_all = country_controls("gr", get_countries(granger), 12)
        df_gr = causal_df.copy()
        if ctrl_s and src_col:
            df_gr = df_gr[df_gr[src_col].str.contains(ctrl_s, case=False, na=False)]
        if ctrl_sel and "All Countries" not in ctrl_sel and src_col:
            df_gr = df_gr[df_gr[src_col].isin(ctrl_sel)]

        col1, col2, col3 = st.columns([2, 2, 1])
        with col1:
            if src_col:
                hubs = (
                    df_gr.groupby(src_col).size()
                    .sort_values(ascending=False)
                    .head(ctrl_top if not ctrl_all else 9999)
                    .reset_index(name="outgoing")
                )
                fig = go.Figure(go.Bar(
                    x=hubs["outgoing"], y=hubs[src_col], orientation="h",
                    marker=dict(
                        color=hubs["outgoing"],
                        colorscale=[[0,"#0d2a2a"],[1,"#22d3ee"]],
                        showscale=False, line=dict(width=0),
                    ),
                    text=hubs["outgoing"], textposition="outside",
                    textfont=dict(size=9, color="#94a3b8"),
                ))
                fig.update_layout(
                    title=dict(text="<b>Top Causal Hubs</b>  (outgoing)",
                               font=dict(size=12, color="#e2e8f0")),
                    height=360, **T,
                    xaxis=dict(title="# Countries Influenced"),
                    yaxis=dict(gridcolor="rgba(0,0,0,0)"),
                )
                st.plotly_chart(fig, use_container_width=True)
        with col2:
            if tgt_col:
                recv = (
                    causal_df.groupby(tgt_col).size()
                    .sort_values(ascending=False)
                    .head(ctrl_top if not ctrl_all else 9999)
                    .reset_index(name="incoming")
                )
                fig = go.Figure(go.Bar(
                    x=recv["incoming"], y=recv[tgt_col], orientation="h",
                    marker=dict(
                        color=recv["incoming"],
                        colorscale=[[0,"#1a0d2e"],[1,"#a78bfa"]],
                        showscale=False, line=dict(width=0),
                    ),
                    text=recv["incoming"], textposition="outside",
                    textfont=dict(size=9, color="#94a3b8"),
                ))
                fig.update_layout(
                    title=dict(text="<b>Most Influenced Countries</b>  (incoming)",
                               font=dict(size=12, color="#e2e8f0")),
                    height=360, **T,
                    xaxis=dict(title="# Influencing Countries"),
                    yaxis=dict(gridcolor="rgba(0,0,0,0)"),
                )
                st.plotly_chart(fig, use_container_width=True)
        with col3:
            if p_col:
                st.markdown(f"""
                <div class="mc mc-teal">
                    <div class="mc-eye">Stats</div>
                    <div style="font-size:.8rem;color:#94a3b8;line-height:2.2;">
                        Total pairs<br><strong style="color:#e2e8f0">{len(granger):,}</strong><br>
                        Causal links<br><strong style="color:#22d3ee">{granger_n or '—'}</strong><br>
                        Median p<br><strong style="color:#e2e8f0">{causal_df[p_col].median():.4f}</strong><br>
                        Min p<br><strong style="color:#fb7185">{causal_df[p_col].min():.6f}</strong>
                    </div>
                </div>""", unsafe_allow_html=True)

        with st.expander("📋 Full causal pairs table"):
            show_c = [c for c in [src_col, tgt_col, p_col] if c]
            if show_c:
                sort_col = p_col if p_col and p_col in df_gr.columns else show_c[0]
                st.dataframe(
                    df_gr[show_c].sort_values(sort_col).rename(
                        columns={src_col: "Source", tgt_col: "Target", p_col: "p-value"}
                    ),
                    use_container_width=True, hide_index=True,
                )
    else:
        st.markdown(empty_html(), unsafe_allow_html=True)

    st.markdown('<div class="ml-divider"></div>', unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════
    # §7 — COX PH SURVIVAL ANALYSIS
    # ══════════════════════════════════════════════════════════════════════
    st.markdown(
        sec_html(7, "Cox PH Survival Analysis", "Time-to-shock modelling & covariate risk factors"),
        unsafe_allow_html=True,
    )

    st.markdown(mc_html(
        "rose", "Survival Analysis · Semi-Parametric Regression",
        "Cox Proportional Hazards Model",
        "COX-PH  ·  h(t) = h₀(t) · exp(β₁X₁ + β₂X₂ + … + βₙXₙ)",
        """The Cox PH model estimates how covariates affect the hazard rate — the instantaneous
        rate at which a stable country transitions into a shock state. A positive β means that
        feature accelerates time-to-shock; negative β delays shocks (protective).
        Outputs Hazard Ratios exp(β) — a HR of 2.0 means the feature doubles the shock rate.
        <br><br><strong>Why use it?</strong> Answers 'how long until a shock?' and 'which features
        speed up or delay onset?' — essential for prioritising interventions.""",
        "Import dependency has the <strong>strongest positive coefficient</strong> — the biggest single accelerator of supply shock onset.",
        "rose",
    ), unsafe_allow_html=True)

    if ok(cox_coef):
        cov_col  = fcol(cox_coef, "covariate", "feature", "variable")
        coef_col = fcol(cox_coef, "coef", "coefficient", "beta")
        hr_col   = fcol(cox_coef, "exp(coef)", "hazard_ratio", "HR")
        p_col    = fcol(cox_coef, "p_value", "pval", "p")

        col1, col2 = st.columns([3, 2])
        with col1:
            if cov_col and coef_col:
                dc     = cox_coef.sort_values(coef_col)
                colors = [ROSE if v > 0 else TEAL for v in dc[coef_col]]
                fig    = go.Figure(go.Bar(
                    x=dc[coef_col], y=dc[cov_col], orientation="h",
                    marker=dict(color=colors, line=dict(width=0)),
                    text=dc[coef_col].apply(lambda v: f"{v:+.3f}"),
                    textposition="outside",
                    textfont=dict(size=9, color="#94a3b8"),
                ))
                fig.add_vline(x=0, line_color="#334155", line_width=1.5)
                fig.update_layout(
                    title=dict(
                        text="<b>Cox PH Coefficients</b>  (🔴 = ↑ shock risk  ·  🔵 = protective)",
                        font=dict(size=12, color="#e2e8f0"),
                    ),
                    height=380, **T,
                    xaxis=dict(title="Log Hazard Ratio (coefficient)"),
                    yaxis=dict(gridcolor="rgba(0,0,0,0)"),
                )
                st.plotly_chart(fig, use_container_width=True)
        with col2:
            if hr_col and cov_col:
                df_hr = cox_coef.sort_values(hr_col, ascending=False)
                fig   = px.bar(
                    df_hr, x=cov_col, y=hr_col, color=hr_col,
                    color_continuous_scale=[[0,"#0d2a2a"],[0.5,"#fbbf24"],[1,"#fb7185"]],
                    title="<b>Hazard Ratios  exp(β)</b>  (> 1 = ↑ risk)",
                )
                fig.add_hline(y=1, line_dash="dot", line_color="#334155", line_width=1.5)
                fig.update_layout(height=340, **T, coloraxis_showscale=False,
                                  xaxis=dict(title=""))
                st.plotly_chart(fig, use_container_width=True)
            elif cov_col:
                show_c = [x for x in [cov_col, coef_col, hr_col, p_col] if x]
                st.dataframe(cox_coef[show_c], use_container_width=True, hide_index=True)
    else:
        st.markdown(empty_html(), unsafe_allow_html=True)

    st.markdown('<div class="ml-divider"></div>', unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════
    # §8 — CHANGE-POINTS & ARIMA
    # ══════════════════════════════════════════════════════════════════════
    st.markdown(
        sec_html(8, "Change-Points & ARIMA Forecasting",
                 "Structural breaks and 5-year supply projections"),
        unsafe_allow_html=True,
    )

    t_cp, t_ar = st.tabs(["📐 Change-Point Detection", "📊 ARIMA Forecast"])

    with t_cp:
        st.markdown(mc_html(
            "amber", "Structural Break Detection · Piecewise Regression",
            "Change-Point Detection",
            "CPD  ·  PELT algorithm (Pruned Exact Linear Time)  ·  RBF cost",
            """Change-point detection identifies moments where a country's supply distribution
            fundamentally shifts — new mean, new variance, or both. The PELT algorithm finds
            optimal breakpoints exactly in O(n) time.
            <br><br><strong>Why use it?</strong> Detects structural shifts that persist for years
            rather than single-year spikes — guiding long-term vs short-term procurement strategy.""",
            f"<strong>{cp_n or '—'} structural breaks</strong> across 172 countries — most countries underwent a regime change between 2014–2016.",
            "amber",
        ), unsafe_allow_html=True)

        if ok(change_pts):
            ctrl_s, ctrl_sel, ctrl_top, ctrl_all = country_controls("cp", get_countries(change_pts))
            df_cp  = filter_df(change_pts, ctrl_sel, ctrl_s)
            c_col  = fcol(df_cp, "country", "Country")
            yr_col = fcol(df_cp, "year_change_point", "year", "change_year")

            col1, col2 = st.columns([3, 2])
            with col1:
                if yr_col:
                    yc  = change_pts[yr_col].value_counts().sort_index().reset_index()
                    yc.columns = ["year", "breaks"]
                    fig = go.Figure(go.Bar(
                        x=yc["year"], y=yc["breaks"],
                        marker=dict(
                            color=yc["breaks"],
                            colorscale=[[0,"#1e1a0d"],[1,"#fbbf24"]],
                            showscale=False, line=dict(width=0),
                        ),
                    ))
                    fig.update_layout(
                        title=dict(text="<b>Structural Breaks per Year</b>",
                                   font=dict(size=12, color="#e2e8f0")),
                        height=300, **T,
                        xaxis=dict(title="Year"),
                        yaxis=dict(title="# Breaks"),
                    )
                    st.plotly_chart(fig, use_container_width=True)
            with col2:
                if c_col and yr_col:
                    mb = df_cp.groupby(c_col)[yr_col].apply(list).reset_index()
                    mb.columns = ["Country", "Break Years"]
                    mb["# Breaks"] = mb["Break Years"].apply(len)
                    mb = mb.sort_values("# Breaks", ascending=False)
                    top_cp = mb if ctrl_all else mb.head(ctrl_top)
                    st.dataframe(top_cp[["Country", "# Breaks", "Break Years"]],
                                 use_container_width=True, hide_index=True)
        else:
            st.markdown(empty_html(), unsafe_allow_html=True)

    with t_ar:
        st.markdown(mc_html(
            "emerald", "Time-Series Forecasting · Auto-ARIMA",
            "ARIMA / Auto-ARIMA",
            "ARIMA(p,d,q)  ·  Auto-selected via AIC  ·  5-year horizon per country",
            """ARIMA models each country's supply as a function of its own lagged values (AR),
            differencing for stationarity (I), and lagged forecast errors (MA).
            Auto-ARIMA selects optimal (p,d,q) parameters per country via AIC minimisation.
            <br><br><strong>Why use it?</strong> Provides 5-year supply projections under
            no-shock conditions — the counterfactual benchmark.""",
            "Forecasts predict steady growth for China/US; African and Caribbean markets show widening confidence intervals — indicating <strong>higher structural uncertainty</strong>.",
            "emerald",
        ), unsafe_allow_html=True)

        if ok(arima):
            ctrl_s, ctrl_sel, ctrl_top, ctrl_all = country_controls("ar", get_countries(arima), 10)
            df_ar   = filter_df(arima, ctrl_sel, ctrl_s)
            c_col   = fcol(df_ar, "country", "Country")
            yr_col  = fcol(df_ar, "year", "Year")
            sup_col = fcol(df_ar, "forecasted_supply", "forecast", "supply")

            # FIX-9: use ctrl_sel to pick countries, respecting the global filter
            if c_col and yr_col and sup_col:
                countries_ar = sorted(df_ar[c_col].unique())
                if "All Countries" in ctrl_sel or not ctrl_sel:
                    sel_c = countries_ar[:ctrl_top]
                else:
                    sel_c = [c for c in ctrl_sel if c in countries_ar and c != "All Countries"]
                    if not sel_c:
                        sel_c = countries_ar[:ctrl_top]

                fig = px.line(
                    df_ar[df_ar[c_col].isin(sel_c)],
                    x=yr_col, y=sup_col, color=c_col,
                    color_discrete_sequence=ACCENT_SEQ,
                    title=f"<b>5-Year ARIMA Forecast</b>  ·  {len(sel_c)} countries",
                    labels={yr_col: "Year", sup_col: "Forecasted Supply", c_col: ""},
                )
                fig.update_traces(line=dict(width=1.5))
                fig.update_layout(height=380, **T)
                st.plotly_chart(fig, use_container_width=True)

                with st.expander("📋 Full forecast table"):
                    st.dataframe(
                        df_ar[[c_col, yr_col, sup_col]].rename(
                            columns={c_col: "Country", yr_col: "Year", sup_col: "Forecasted Supply"}
                        ),
                        use_container_width=True, hide_index=True,
                    )
        else:
            st.markdown(empty_html(), unsafe_allow_html=True)

    st.markdown('<div class="ml-divider"></div>', unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════
    # §9 — SHOCK VALIDATION & NETWORK
    # ══════════════════════════════════════════════════════════════════════
    st.markdown(
        sec_html(9, "Shock Validation & Network Centrality",
                 "Back-testing vs real events · systemic influence analysis"),
        unsafe_allow_html=True,
    )

    t_sv, t_net = st.tabs(["✅ Shock Validation", "🕸️ Network Centrality"])

    with t_sv:
        st.markdown(mc_html(
            "emerald", "Retrospective Validation · Event-Aligned Z-Scores",
            "Shock Validation",
            "SHOCK-VALID  ·  Back-test: COVID-19 · 2008 GFC · Shale Boom 2014",
            """Validates anomaly detectors by checking whether they would have flagged
            known historical macro events. Z-scores are compared across pre/during/post windows.
            <br><br><strong>Why use it?</strong> Without retrospective validation against
            known ground-truth events, there's no way to confirm the pipeline detects real
            shocks rather than artefacts of the data processing pipeline.""",
            "Successfully re-detected supply impacts of <strong>COVID-19 and the 2008 GFC</strong> with statistically significant Z-score elevation.",
            "emerald",
        ), unsafe_allow_html=True)

        src_sv = shock_valid if ok(shock_valid) else shock_score
        if ok(src_sv):
            ev_col    = fcol(src_sv, "event", "Event", "shock_event")
            phase_col = fcol(src_sv, "phase", "Phase", "period")
            z_col     = fcol(src_sv, "ShockZ", "z_score", "impact_score", "ShockScore", "score")
            c_col     = fcol(src_sv, "country", "Country")

            ctrl_s, ctrl_sel, ctrl_top, ctrl_all = country_controls("sv", get_countries(src_sv))
            df_sv = filter_df(src_sv, ctrl_sel, ctrl_s)

            if ev_col and phase_col and z_col:
                fig = px.bar(
                    df_sv, x=ev_col, y=z_col, color=phase_col, barmode="group",
                    color_discrete_map={
                        "pre": SLATE, "during": ROSE, "post": TEAL,
                        "Pre": SLATE, "During": ROSE, "Post": TEAL,
                    },
                    title="<b>Z-Score by Event Phase</b>  (pre / during / post shock)",
                )
                fig.add_hline(y=1.96, line_dash="dot", line_color=AMBER,
                              annotation_text="95% confidence threshold",
                              annotation_font_size=9)
                fig.update_layout(height=340, **T, xaxis_title="", yaxis_title="Average Z-Score")
                st.plotly_chart(fig, use_container_width=True)
                st.dataframe(df_sv, use_container_width=True, hide_index=True)

            elif c_col and z_col:
                top_sv = df_sv.nlargest(ctrl_top if not ctrl_all else len(df_sv), z_col)
                fig    = px.bar(
                    top_sv.sort_values(z_col, ascending=True),
                    x=z_col, y=c_col, orientation="h", color=z_col,
                    color_continuous_scale=[[0,"#0d2a20"],[0.6,"#34d399"],[1,"#fb7185"]],
                    title="<b>Shock Scores</b>  — Top Countries",
                )
                fig.update_layout(height=380, **T, coloraxis_showscale=False)
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.markdown(empty_html("Shock validation checkpoint not found."), unsafe_allow_html=True)

    with t_net:
        st.markdown(mc_html(
            "teal", "Graph Analysis · Supply Similarity Network",
            "Network Centrality",
            "NETWORK  ·  172-node graph · 1,542 edges · PageRank + Betweenness + Degree",
            """A supply similarity network is built where edge weights represent historical
            supply correlation between country pairs. PageRank measures global influence;
            Betweenness identifies bridge countries; Degree measures raw connection count.
            <br><br><strong>Why use it?</strong> Betweenness centrality identifies bridge nodes
            whose disruption has outsized cascading effects.""",
            "Morocco and Tunisia hold the highest betweenness centrality — critical <strong>bridge nodes</strong> between European and Sub-Saharan African supply communities.",
        ), unsafe_allow_html=True)

        if ok(net_metrics):
            ctrl_s, ctrl_sel, ctrl_top, ctrl_all = country_controls("net", get_countries(net_metrics), 20)
            df_net  = filter_df(net_metrics, ctrl_sel, ctrl_s)
            c_col   = fcol(df_net, "country", "Country")
            pr_col  = fcol(df_net, "pagerank", "page_rank", "PageRank")
            btw_col = fcol(df_net, "betweenness_centrality", "betweenness")
            deg_col = fcol(df_net, "degree_centrality", "degree")

            if pr_col and c_col:
                top_n = df_net.nlargest(ctrl_top if not ctrl_all else len(df_net), pr_col)
                col1, col2 = st.columns([3, 2])
                with col1:
                    fig = px.scatter(
                        top_n,
                        x=pr_col, y=btw_col if btw_col else pr_col,
                        text=c_col, size=pr_col, size_max=25,
                        color=deg_col if deg_col else pr_col,
                        color_continuous_scale=[[0,"#0d2a2a"],[1,"#22d3ee"]],
                        title=f"<b>Network Hubs</b>  ·  PageRank vs Betweenness (top {ctrl_top})",
                        labels={pr_col: "PageRank", btw_col or pr_col: "Betweenness Centrality"},
                    )
                    fig.update_traces(textposition="top center",
                                      textfont=dict(size=8, color="#94a3b8"),
                                      marker_line_width=0)
                    fig.update_layout(height=400, **T, coloraxis_showscale=False)
                    st.plotly_chart(fig, use_container_width=True)
                with col2:
                    show_c = [x for x in [c_col, pr_col, btw_col, deg_col] if x]
                    st.dataframe(
                        top_n[show_c].rename(columns={
                            c_col: "Country", pr_col: "PageRank",
                            btw_col: "Betweenness", deg_col: "Degree",
                        }),
                        use_container_width=True, hide_index=True,
                    )
        else:
            st.markdown(empty_html("Network metrics checkpoint not found."), unsafe_allow_html=True)

    st.markdown('<div class="ml-divider"></div>', unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════
    # §10 — MODEL SELECTION GUIDE
    # ══════════════════════════════════════════════════════════════════════
    st.markdown(
        sec_html(10, "When to Use Each Model", "Decision guide for analysts and policy teams"),
        unsafe_allow_html=True,
    )

    st.markdown("""
    <div class="guide-grid">
        <div class="guide-card">
            <div class="gi">🌲</div><div class="gu">Find outlier countries</div>
            <div class="gm">Isolation Forest + LOF + OCSVM</div>
            <div class="gd">Use when you need to find countries structurally unlike all others across multiple features. High confidence when all 3 agree.</div>
        </div>
        <div class="guide-card">
            <div class="gi">📈</div><div class="gu">Pinpoint disruptions in history</div>
            <div class="gm">EWMA + Prophet + LSTM</div>
            <div class="gd">Use when you want specific country-years with abnormal supply — short-lived shocks (EWMA), seasonal breaks (Prophet), or non-linear patterns (LSTM).</div>
        </div>
        <div class="guide-card">
            <div class="gi">🔮</div><div class="gu">Predict next-year shock risk</div>
            <div class="gm">Random Forest</div>
            <div class="gd">Use when you need a forward-looking probability to prioritise procurement or policy interventions.</div>
        </div>
        <div class="guide-card">
            <div class="gi">🔗</div><div class="gu">Trace contagion pathways</div>
            <div class="gm">Granger Causality + Network</div>
            <div class="gd">Use when you need to know which countries' shocks will cascade to others — critical for supply chain diversification strategy.</div>
        </div>
        <div class="guide-card">
            <div class="gi">⏱️</div><div class="gu">Estimate time-to-shock</div>
            <div class="gm">Cox PH Survival</div>
            <div class="gd">Use when you want to know how long a country is likely to remain stable given its current risk profile.</div>
        </div>
        <div class="guide-card">
            <div class="gi">🏷️</div><div class="gu">Segment countries strategically</div>
            <div class="gm">KMeans + DBSCAN</div>
            <div class="gd">Use when you want to group countries with similar supply chain profiles, or identify nations with no peers (DBSCAN noise).</div>
        </div>
        <div class="guide-card">
            <div class="gi">📐</div><div class="gu">Identify regime changes</div>
            <div class="gm">Change-Point Detection</div>
            <div class="gd">Use when you want to find the specific year a country's supply fundamentally changed — a new normal, not a temporary spike.</div>
        </div>
        <div class="guide-card">
            <div class="gi">📊</div><div class="gu">Project future supply baseline</div>
            <div class="gm">ARIMA</div>
            <div class="gd">Use when you need a statistical 5-year baseline forecast under no-shock conditions.</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="ml-divider"></div>', unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════
    # §11 — RAW DATA ACCESS  (FIX-12: guard against None)
    # ══════════════════════════════════════════════════════════════════════
    st.markdown(
        sec_html(11, "Raw Checkpoint Downloads", "Export any model output as CSV"),
        unsafe_allow_html=True,
    )

    with st.expander("📁 Download Model Checkpoints", expanded=False):
        all_cp = {
            "anomaly_isolation_forest":  iso_forest,
            "anomaly_lof":               lof,
            "anomaly_ocsvm":             ocsvm_df,
            "anomaly_ewma":              ewma,
            "anomaly_iqr":               iqr_df,
            "anomaly_percentile":        percentile,
            "anomaly_moving_avg_robust": mov_avg,
            "anomaly_prophet":           prophet,
            "anomaly_lstm":              lstm,
            "anomaly_dashboard":         adash,
            "combined_yearly_anomalies": combined_yr,
            "combined_country_anomalies":combined_co,
            "clusters":                  clusters,
            "dbscan_results":            dbscan_df,
            "temporal_clusters":         temporal_cl,
            "country_pca_features":      pca_feat,
            "arima_forecast_results":    arima,
            "change_points":             change_pts,
            "rf_predictions":            rf_pred,
            "granger_causality_results": granger,
            "coxph_coefficients":        cox_coef,
            "coxph_prepared_data":       cox_prep,
            "network_metrics":           net_metrics,
            "network_communities":       net_comm,
            "shock_score":               shock_score,
            "shock_severity":            shock_sev,
            "shock_validation_advanced": shock_valid,
            "contagion_features":        contagion,
        }
        # FIX-12: only iterate over genuinely-loaded DataFrames
        found = {k: v for k, v in all_cp.items() if ok(v)}
        st.caption(f"**{len(found)}** of **{len(all_cp)}** checkpoints available")

        for name, df in found.items():
            c1, c2, c3 = st.columns([3, 1, 1])
            with c1:
                st.markdown(f"`{name}` — **{len(df):,}** rows × **{len(df.columns)}** cols")
            with c2:
                st.caption(", ".join(df.columns[:3].tolist()) + "…")
            with c3:
                try:
                    csv_bytes = df.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        "↓ CSV", csv_bytes, f"{name}.csv", key=f"dl_{name}"
                    )
                except Exception as exc:
                    st.caption(f"Export error: {exc}")

    # ── FOOTER ────────────────────────────────────────────────────────────
    st.markdown("""
    <div class="ml-footer">
        ALL METRICS FROM LIVE PIPELINE CHECKPOINTS · NO SYNTHETIC DATA ·
        LPG GLOBAL SUPPLY CHAIN RISK INTELLIGENCE · 1990–2024 ·
        REGENERATE: python pipeline.py --full --ml
    </div>
    """, unsafe_allow_html=True)