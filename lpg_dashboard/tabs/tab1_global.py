"""Global Overview Tab - Complete Implementation (Fixed & Robust)"""

import streamlit as st
import pandas as pd
import plotly.express as px

from utils.helpers import (
    section_tag, table_limit, apply_plotly_dark, add_iso
)

from utils.constants import (
    SEVERITY_COLORS, SEVERITY_ORDER, PLOTLY_DARK
)


def render(data, df_sev, filters=None):
    """Render Global Overview tab"""

    section_tag("GLOBAL RISK LANDSCAPE")

    # =========================================================
    # SAFETY CHECK
    # =========================================================
    if df_sev is None or not isinstance(df_sev, pd.DataFrame) or df_sev.empty:
        st.warning("No severity data available")
        return

    # =========================================================
    # KPI ROW
    # =========================================================
    # --- Severity counts ---
    if 'ShockSeverity' in df_sev.columns:
        strong = (df_sev['ShockSeverity'] == 'Strong').sum()
        moderate = (df_sev['ShockSeverity'] == 'Moderate').sum()
        weak = (df_sev['ShockSeverity'] == 'Weak').sum()
    else:
        strong = moderate = weak = 0

    # --- Change points (robust) ---
    change_points = data.get("change_points", None)

    if change_points is None:
        n_cp = 0
    elif isinstance(change_points, (list, tuple, set)):
        n_cp = len(change_points)
    elif isinstance(change_points, pd.DataFrame):
        n_cp = len(change_points)
    elif isinstance(change_points, dict):
        n_cp = sum(len(v) for v in change_points.values())
    else:
        try:
            n_cp = len(change_points)
        except Exception:
            n_cp = 0

    # --- Network communities ---
    n_communities = 0
    nc = data.get('network_communities')
    if isinstance(nc, pd.DataFrame) and 'community_id' in nc.columns:
        n_communities = nc['community_id'].nunique()

    # --- Anomaly count ---
    anomaly_count = 0
    ad = data.get('anomaly_dashboard')
    if isinstance(ad, pd.DataFrame):
        flag_cols = [c for c in ad.columns if c.endswith('_flag')]
        if flag_cols:
            anomaly_count = int(ad[flag_cols].sum().sum())

    # --- KPI display ---
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("🌍 Countries monitored", len(df_sev))
    c2.metric("🔴 Strong risk", int(strong))
    c3.metric("🟠 Moderate risk", int(moderate))
    c4.metric("⚡ Structural breaks", int(n_cp))
    c5.metric("🔍 Anomalies detected", f"{anomaly_count:,}")

    st.divider()

    # =========================================================
    # WORLD MAP
    # =========================================================
    st.subheader("🌎 Shock Severity by Country")

    if 'country' not in df_sev.columns:
        st.error("Missing 'country' column for map")
    else:
        map_data = df_sev.copy()
        map_data = add_iso(map_data, 'country')
        map_data = map_data[map_data['iso'].notna()]

        if not map_data.empty:
            fig_map = px.choropleth(
                map_data,
                locations='iso',
                color='ShockSeverity',
                hover_name='country',
                hover_data={
                    'ShockScore': ':.2f' if 'ShockScore' in map_data.columns else False,
                    'region': True if 'region' in map_data.columns else False,
                    'iso': False
                },
                color_discrete_map=SEVERITY_COLORS,
                category_orders={"ShockSeverity": SEVERITY_ORDER},
                projection='natural earth',
                title="<b>Global Supply Chain Shock Severity</b>"
            )

            fig_map.update_layout(
                height=500,
                margin=dict(l=0, r=0, t=40, b=0),
                geo=dict(
                    showframe=False,
                    showcoastlines=True,
                    coastlinecolor="#1e3a5f",
                    showland=True,
                    landcolor="#0d1b2a",
                    showocean=True,
                    oceancolor="#07111c"
                )
            )
            apply_plotly_dark(fig_map)
            st.plotly_chart(fig_map, use_container_width=True)
        else:
            st.warning("No map data available - missing ISO codes")

    st.divider()

    # =========================================================
    # SEVERITY DISTRIBUTION & TOP COUNTRIES
    # =========================================================
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("📊 Severity Distribution")

        if 'ShockSeverity' in df_sev.columns:
            counts = df_sev['ShockSeverity'].value_counts()

            for cat in SEVERITY_ORDER:
                if cat not in counts:
                    counts[cat] = 0

            counts = counts[SEVERITY_ORDER]

            fig_pie = px.pie(
                values=counts.values,
                names=counts.index,
                color=counts.index,
                color_discrete_map=SEVERITY_COLORS,
                hole=0.45
            )

            fig_pie.update_traces(textinfo='percent+label')
            fig_pie.update_layout(**PLOTLY_DARK)
            st.plotly_chart(fig_pie, use_container_width=True)

    with col_right:
        st.subheader("⚠️ Highest-Risk Countries")

        if 'ShockScore' in df_sev.columns:
            n_top = table_limit("risk_tab1", default=15)

            top_df = (
                df_sev.nlargest(n_top, 'ShockScore')
                if n_top > 0 else
                df_sev.sort_values('ShockScore', ascending=False)
            )

            top_df = top_df.reset_index(drop=True)
            top_df['rank'] = top_df.index + 1

            fig_top = px.bar(
                top_df,
                x='ShockScore',
                y='country',
                orientation='h',
                color='ShockSeverity',
                color_discrete_map=SEVERITY_COLORS,
                category_orders={"ShockSeverity": SEVERITY_ORDER}
            )

            fig_top.update_layout(
                height=max(400, len(top_df) * 28)
            )
            apply_plotly_dark(fig_top)
            st.plotly_chart(fig_top, use_container_width=True)

    st.divider()

    # =========================================================
    # REGION SUMMARY
    # =========================================================
    st.subheader("📋 Regional Risk Summary")

    if {'region', 'ShockScore'}.issubset(df_sev.columns):

        region_summary = (
            df_sev.groupby('region')
            .agg(
                Countries=('country', 'count'),
                Avg_ShockScore=('ShockScore', 'mean'),
                Max_ShockScore=('ShockScore', 'max')
            )
            .round(2)
            .sort_values('Avg_ShockScore', ascending=False)
            .reset_index()
        )

        st.dataframe(region_summary, use_container_width=True)

    st.divider()
# tabs/tab1_global.py (Updated anomaly hotspot section)
# ... (previous code remains the same until the anomaly hotspots section)

    # =========================================================
    # ANOMALY HOTSPOTS - UPDATED FOR ACTUAL COLUMN NAMES
    # =========================================================
    st.subheader("🔥 Structural Anomaly Hotspots")
    st.caption("Countries flagged by structural detection methods (LOF, OCSVM, Isolation Forest, Change Points)")
    
    cca = data.get('anomaly_dashboard')
    if cca is not None:
        # Define structural flags and create composite score on the fly
        structural_flags = ['lof_flag', 'ocsvm_flag', 'if_flag', 'cp_flag']
        for flag in structural_flags:
            if flag not in cca.columns:
                cca[flag] = 0  # Assume 0 if flag column is missing

        cca['composite_country_anomaly_score'] = cca[structural_flags].sum(axis=1)

        # Use the actual composite score
        cca_show = cca.sort_values('composite_country_anomaly_score', ascending=False)
        cca_show = cca_show[cca_show['composite_country_anomaly_score'] > 0]  # Only show flagged countries
        
        if not cca_show.empty:
            n_hot = table_limit("hotspots", default=15)
            cca_show = cca_show if n_hot == 0 else cca_show.head(n_hot)
            cca_show = cca_show.sort_values('composite_country_anomaly_score', ascending=True)
            
            # Build tooltip text with individual flags (using corrected flag names)
            cca_show['flags'] = cca_show.apply(
                lambda x: ', '.join(filter(None, [
                    'LOF' if x.get('lof_flag') else None,
                    'OCSVM' if x.get('ocsvm_flag') else None,
                    'IF' if x.get('if_flag') else None,
                    'CP' if x.get('cp_flag') else None,
                ])),
                axis=1
            )
            
            fig_hot = px.bar(
                cca_show,
                x='composite_country_anomaly_score',
                y='country',
                orientation='h',
                color='composite_country_anomaly_score',
                color_continuous_scale='YlOrRd',
                hover_data={'flags': True},
                labels={
                    'composite_country_anomaly_score': 'Structural Anomaly Score',
                    'country': '',
                    'flags': 'Detection Methods'
                },
                title="<b>Countries with Structural Anomalies</b>"
            )
            
            fig_hot.update_layout(
                yaxis={'categoryorder': 'total ascending'},
                height=max(400, len(cca_show) * 28),
                xaxis_title="Number of Structural Methods Flagging Country"
            )
            apply_plotly_dark(fig_hot)
            st.plotly_chart(fig_hot, use_container_width=True)
            
            st.markdown(
                '<div class="insight-box">'
                '<b>💡 Structural vs. Temporal Anomalies:</b><br>'
                '• <b>Structural anomalies</b> (LOF, OCSVM, IF) identify countries fundamentally different from global norms.<br>'
                '• <b>Temporal anomalies</b> (EWMA, Prophet, LSTM) identify specific years with supply disruptions.<br>'
                '• <b>Change Points (CP)</b> identify regime shifts in a country\'s supply trajectory.'
                '</div>',
                unsafe_allow_html=True
            )
        else:
            st.info("No structural anomalies detected in the data")
    else:
        st.info("Structural anomaly data not available")
    # =========================================================
    # EXECUTIVE INSIGHTS
    # =========================================================
    st.subheader("📌 Executive Insights")

    interpretation = data.get('final_interpretation')

    if isinstance(interpretation, str) and len(interpretation) > 50:
        st.markdown(interpretation)
    else:
        st.info("No AI interpretation available")