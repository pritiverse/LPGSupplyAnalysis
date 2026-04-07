"""LPG Supply Chain Shock Analysis Dashboard - Main Entry Point"""

import streamlit as st
import pandas as pd

from utils.css import apply_css
from utils.data_loader import load_all_data, get_sample_data
from utils.helpers import add_region, safe_sort_severity
from utils.constants import SEVERITY_ORDER

from tabs.tab1_global import render as render_global
from tabs import render_global, render_country, render_events, render_ml, render_detection


# =========================================================
# PAGE CONFIG (must be first)
# =========================================================
st.set_page_config(
    page_title="LPG Supply Chain Shock Analysis",
    page_icon="🛢️",
    layout="wide",
    initial_sidebar_state="expanded"
)

apply_css()


# =========================================================
# HEADER
# =========================================================
def render_header():
    st.markdown("""
        <div style="background: linear-gradient(135deg, #0d1b2a 0%, #1e3a5f 100%);
                    padding: 24px;
                    border-radius: 12px;
                    margin-bottom: 24px;
                    border: 1px solid #3a5a7a;">
            <div style="display: flex; align-items: center; gap: 20px; flex-wrap: wrap;">
                <span style="font-size: 48px;">🛢️</span>
                <div style="flex: 1;">
                    <h1 style="margin: 0;">LPG Supply Chain Shock Analysis</h1>
                    <p style="color: #8bbdd9; font-size: 13px; margin-top: 6px;">
                        172 COUNTRIES · 1990–2024 · MULTI-MODEL ANALYSIS
                    </p>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)


# =========================================================
# SIDEBAR
# =========================================================
def render_sidebar(df_sev):

    with st.sidebar:
        st.markdown("### 🔧 Control Panel")
        st.markdown("---")

        regions = sorted(df_sev['region'].dropna().unique()) if 'region' in df_sev.columns else []

        selected_regions = st.multiselect(
            "🌍 Region",
            regions,
            default=regions
        )

        severity_filter = st.multiselect(
            "⚠️ Severity",
            SEVERITY_ORDER,
            default=SEVERITY_ORDER
        )

        st.markdown("---")

        if st.button("🔄 Refresh"):
            st.cache_data.clear()
            st.rerun()

        return {
            "regions": selected_regions,
            "severity": severity_filter
        }


# =========================================================
# FILTERING
# =========================================================
def apply_filters(df, filters):

    if df is None or df.empty:
        return df

    out = df.copy()

    if filters["regions"] and 'region' in out.columns:
        out = out[out["region"].isin(filters["regions"])]

    if filters["severity"] and 'ShockSeverity' in out.columns:
        out = out[out["ShockSeverity"].isin(filters["severity"])]

    return out


# =========================================================
# KPI STRIP
# =========================================================
def render_kpi_strip(df):

    if df is None or df.empty:
        return

    strong = (df['ShockSeverity'] == 'Strong').sum() if 'ShockSeverity' in df.columns else 0
    moderate = (df['ShockSeverity'] == 'Moderate').sum() if 'ShockSeverity' in df.columns else 0

    most_affected = "N/A"
    if {'region', 'ShockScore'}.issubset(df.columns):
        try:
            most_affected = df.groupby("region")["ShockScore"].mean().idxmax()
        except Exception:
            pass

    st.markdown(f"""
    <div style="background:#112233;padding:15px;border-radius:10px;margin-bottom:20px">
        <b>🚨 Global Shock Status:</b> {int(strong)} Strong | {int(moderate)} Moderate
        &nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;
        <b>🌍 Most Affected Region:</b> {most_affected}
    </div>
    """, unsafe_allow_html=True)


# =========================================================
# MAIN
# =========================================================
def main():

    render_header()

    # ---------------- LOAD DATA ----------------
    with st.spinner("Loading data..."):
        data = load_all_data()
# Add after loading data
    with st.expander("📊 Data Checkpoint Status"):
        from utils.data_loader import get_data_status
        status = get_data_status(data)
        for name, stat in status.items():
            st.write(f"**{name}**: {stat}")
    # Safe extraction
    df_sev = data.get("shock_severity")

    # ---------------- FALLBACK ----------------
    if df_sev is None or not isinstance(df_sev, pd.DataFrame):
        st.warning("Using sample data (real pipeline not found)")
        data = get_sample_data()
        df_sev = data.get("shock_severity")

    # ---------------- VALIDATION ----------------
    if df_sev is None or df_sev.empty:
        st.error("No data available")
        st.stop()

    # ---------------- CLEANING ----------------
    df_sev = add_region(df_sev)
    df_sev = safe_sort_severity(df_sev, 'ShockSeverity')

    # ---------------- SIDEBAR ----------------
    filters = render_sidebar(df_sev)

    # ---------------- FILTER ----------------
    filtered_df = apply_filters(df_sev, filters)

    if filtered_df.empty:
        st.warning("No data after filtering")
        st.stop()

    # ---------------- KPI ----------------
    render_kpi_strip(filtered_df)

    # =========================================================
    # TABS
    # =========================================================
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🌍 Global Overview",
        "🔍 Country",
        "📅 Events",
        "🤖 Models",
        "🧠 Detection"
    ])

    # ---------------- TAB 1 ----------------
    with tab1:
        render_global(data, filtered_df, filters)

    # ---------------- PLACEHOLDERS ----------------
    with tab2:
        render_country(data, filtered_df, filters)

    with tab3:
        render_events(data, filtered_df, filters)

    with tab4:
        render_ml(data, filtered_df, filters)

    with tab5:
        render_detection(data, filtered_df, filters)


# =========================================================
# RUN
# =========================================================
if __name__ == "__main__":
    main()