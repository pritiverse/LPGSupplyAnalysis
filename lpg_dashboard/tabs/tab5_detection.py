# =========================================================================
# TAB 5 — RISK INTELLIGENCE DASHBOARD
# Clean, professional build with future shock predictions as the hero section
# =========================================================================

def render_detection(data, filtered_df, filters):
    import streamlit as st
    import pandas as pd
    import plotly.graph_objects as go
    import plotly.express as px
    from datetime import datetime
    from io import BytesIO

    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import inch
        from reportlab.platypus import (
            SimpleDocTemplate, Paragraph, Spacer,
            Table, TableStyle, Image, PageBreak
        )
        REPORTLAB_AVAILABLE = True
    except ImportError:
        REPORTLAB_AVAILABLE = False

    from utils.constants import PLOTLY_DARK

    # ------------------------------------------------------------------
    # STYLING
    # ------------------------------------------------------------------
    st.markdown("""
    <style>
    .kpi-row { display: flex; gap: 1rem; margin-bottom: 1.5rem; flex-wrap: wrap; }

    .kpi-card {
        flex: 1; min-width: 140px;
        background: var(--background-color, #0f172a);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 12px;
        padding: 1.1rem 1.3rem;
    }
    .kpi-label {
        font-size: 0.72rem; font-weight: 600; letter-spacing: 0.06em;
        text-transform: uppercase; color: #94a3b8; margin-bottom: 0.4rem;
    }
    .kpi-value { font-size: 2rem; font-weight: 700; line-height: 1; }
    .kpi-sub   { font-size: 0.75rem; color: #64748b; margin-top: 0.3rem; }

    .kpi-red    { color: #f87171; }
    .kpi-amber  { color: #fbbf24; }
    .kpi-blue   { color: #60a5fa; }
    .kpi-violet { color: #a78bfa; }

    .section-title {
        font-size: 0.78rem; font-weight: 700; letter-spacing: 0.08em;
        text-transform: uppercase; color: #64748b;
        border-bottom: 1px solid rgba(255,255,255,0.07);
        padding-bottom: 0.5rem; margin-bottom: 1.2rem;
    }

    .alert-row { display: flex; gap: 0.6rem; flex-wrap: wrap; margin-bottom: 1rem; }
    .badge {
        font-size: 0.7rem; font-weight: 700; letter-spacing: 0.05em;
        text-transform: uppercase; padding: 0.25rem 0.75rem;
        border-radius: 100px;
    }
    .badge-red    { background: rgba(239,68,68,0.15);  color: #f87171; border: 1px solid rgba(239,68,68,0.3); }
    .badge-amber  { background: rgba(251,191,36,0.12); color: #fbbf24; border: 1px solid rgba(251,191,36,0.3); }
    .badge-green  { background: rgba(52,211,153,0.12); color: #34d399; border: 1px solid rgba(52,211,153,0.3); }

    .insight-block {
        border-left: 3px solid #3b82f6;
        background: rgba(59,130,246,0.06);
        border-radius: 0 8px 8px 0;
        padding: 0.75rem 1rem;
        margin-bottom: 0.75rem;
        font-size: 0.88rem;
        color: #cbd5e1;
        line-height: 1.55;
    }
    .insight-block.warn { border-left-color: #f59e0b; background: rgba(245,158,11,0.06); }
    .insight-block.crit { border-left-color: #ef4444; background: rgba(239,68,68,0.06); }

    /* forecast heat table */
    .heat-table { width: 100%; border-collapse: collapse; font-size: 0.82rem; }
    .heat-table th {
        text-align: left; padding: 0.5rem 0.6rem;
        font-size: 0.7rem; font-weight: 700; letter-spacing: 0.06em;
        text-transform: uppercase; color: #64748b;
        border-bottom: 1px solid rgba(255,255,255,0.07);
    }
    .heat-table td { padding: 0.45rem 0.6rem; border-bottom: 1px solid rgba(255,255,255,0.04); color: #e2e8f0; }
    .heat-table tr:last-child td { border-bottom: none; }

    .prob-bar-wrap { position: relative; height: 6px; background: rgba(255,255,255,0.07); border-radius: 3px; min-width: 80px; }
    .prob-bar { position: absolute; left: 0; top: 0; height: 100%; border-radius: 3px; }

    .risk-pill {
        font-size: 0.68rem; font-weight: 700; padding: 0.15rem 0.55rem;
        border-radius: 100px; display: inline-block;
    }
    </style>
    """, unsafe_allow_html=True)

    # ------------------------------------------------------------------
    # DATA SOURCES
    # ------------------------------------------------------------------
    def _get(key):
        df = data.get(key)
        if isinstance(df, pd.DataFrame) and len(df) > 0:
            return df
        return None

    anomaly_dash   = _get('anomaly_dashboard')
    rf_pred        = _get('rf_predictions')
    shock_sev      = _get('shock_severity')
    global_rank    = _get('shock_global_country_rank')
    yearly_summary = _get('combined_yearly_summary')
    rf_future_preds = _get('rf_future_predictions')

    has_data = any([anomaly_dash is not None, rf_pred is not None,
                    shock_sev is not None, rf_future_preds is not None])

    if not has_data:
        st.error("No risk data available. Ensure the pipeline has run and checkpoints exist.")
        return

    def _safe(v, default=0.0):
        return float(v) if pd.notna(v) else default

    def _fig_png(fig):
        try:
            return fig.to_image(format="png", width=1400, height=700, scale=2)
        except Exception:
            return None

    def _df_pdf(df, max_rows=20):
        if df is None or len(df) == 0:
            return [["No data"]]
        d = df.head(max_rows).copy()
        for c in d.columns:
            d[c] = d[c].apply(
                lambda x: f"{x:.3f}" if isinstance(x, (int, float)) and not isinstance(x, bool) else str(x)
            )
        return [list(d.columns)] + d.astype(str).values.tolist()

    # ------------------------------------------------------------------
    # KPI COMPUTATION
    # ------------------------------------------------------------------
    kpi = dict(critical=0, avg_risk=0.0, shock_avg=0.0, high_prob=0, n_countries=0)

    if anomaly_dash is not None and 'risk_score' in anomaly_dash.columns:
        kpi['n_countries'] = len(anomaly_dash)
        kpi['avg_risk']    = anomaly_dash['risk_score'].mean()
        kpi['critical']    = int((anomaly_dash['risk_score'] >= 10).sum())

    if rf_future_preds is not None and 'probability_shock' in rf_future_preds.columns:
        kpi['shock_avg'] = rf_future_preds['probability_shock'].mean()
        kpi['high_prob'] = int((rf_future_preds['probability_shock'] > 0.6).sum())
    elif rf_pred is not None and 'probability' in rf_pred.columns:
        kpi['shock_avg'] = rf_pred['probability'].mean()
        kpi['high_prob'] = int((rf_pred['probability'] > 0.6).sum())

    # ------------------------------------------------------------------
    # HEADER
    # ------------------------------------------------------------------
    st.markdown("## Risk Intelligence Dashboard")
    st.caption(f"Global vulnerability assessment · Predictive shock forecasting · {datetime.now().strftime('%d %b %Y, %H:%M')}")

    st.markdown(f"""
    <div class="kpi-row">
        <div class="kpi-card">
            <div class="kpi-label">Critical countries</div>
            <div class="kpi-value kpi-red">{kpi['critical']}</div>
            <div class="kpi-sub">Risk score ≥ 10</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">Global avg risk</div>
            <div class="kpi-value kpi-amber">{kpi['avg_risk']:.1f}</div>
            <div class="kpi-sub">Out of 20</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">Avg shock probability</div>
            <div class="kpi-value kpi-blue">{kpi['shock_avg']:.1%}</div>
            <div class="kpi-sub">Across all forecasts</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">High-risk forecasts</div>
            <div class="kpi-value kpi-violet">{kpi['high_prob']}</div>
            <div class="kpi-sub">P(shock) > 60%</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ------------------------------------------------------------------
    # ALERTS & INSIGHTS
    # ------------------------------------------------------------------
    with st.expander("Early warning signals", expanded=True):
        col_ins, col_alrt = st.columns(2)

        with col_ins:
            st.markdown('<div class="section-title">System insights</div>', unsafe_allow_html=True)
            insights_list = []

            if anomaly_dash is not None and 'risk_score' in anomaly_dash.columns:
                std = anomaly_dash['risk_score'].std()
                concentration = "concentrated in a few regions" if std > 4 else "distributed broadly"
                insights_list.append(("info", f"Risk exposure is <b>{concentration}</b> — std dev {std:.1f}"))

            if rf_future_preds is not None and 'probability_shock' in rf_future_preds.columns:
                med = rf_future_preds['probability_shock'].median()
                label = "elevated global vulnerability" if med > 0.5 else "moderate conditions"
                insights_list.append(("info", f"Future shock median ({med:.1%}) indicates <b>{label}</b>"))
            elif rf_pred is not None and 'probability' in rf_pred.columns:
                med = rf_pred['probability'].median()
                label = "elevated" if med > 0.5 else "moderate"
                insights_list.append(("info", f"Current shock median: <b>{med:.1%}</b> — {label} outlook"))

            if anomaly_dash is not None and 'risk_score' in anomaly_dash.columns:
                n10 = max(1, len(anomaly_dash) // 10)
                top10sum = anomaly_dash.nlargest(n10, 'risk_score')['risk_score'].sum()
                total = anomaly_dash['risk_score'].sum()
                conc = (top10sum / total * 100) if total > 0 else 0
                insights_list.append(("warn", f"Top 10% of countries hold <b>{conc:.0f}%</b> of total systemic risk"))

            cls_map = {"info": "", "warn": "warn", "crit": "crit"}
            for kind, msg in insights_list:
                st.markdown(f'<div class="insight-block {cls_map[kind]}">{msg}</div>', unsafe_allow_html=True)

        with col_alrt:
            st.markdown('<div class="section-title">Active alerts</div>', unsafe_allow_html=True)
            alerts_list = []

            if anomaly_dash is not None and 'risk_score' in anomaly_dash.columns:
                n = int((anomaly_dash['risk_score'] >= 10).sum())
                if n:
                    alerts_list.append(("crit", f"{n} countries exceed critical threshold (≥ 10)"))

            if rf_future_preds is not None and 'probability_shock' in rf_future_preds.columns:
                n = int((rf_future_preds['probability_shock'] > 0.65).sum())
                if n:
                    alerts_list.append(("crit", f"{n} country-year forecasts show >65% shock probability"))

            if (shock_sev is not None and len(shock_sev) > 1
                    and 'year' in shock_sev.columns and 'shock_severity' in shock_sev.columns):
                tail = shock_sev.sort_values('year').tail(2)
                if len(tail) == 2:
                    delta = tail.iloc[-1]['shock_severity'] - tail.iloc[-2]['shock_severity']
                    if pd.notna(delta) and delta > 0:
                        alerts_list.append(("warn", f"Global shock severity trending up (+{delta:.2f} YoY)"))

            if anomaly_dash is not None:
                flag_cols = [c for c in anomaly_dash.columns if 'flag' in c.lower()]
                if flag_cols:
                    n = int((anomaly_dash[flag_cols].sum(axis=1) >= 7).sum())
                    if n:
                        alerts_list.append(("warn", f"{n} countries flagged by 7+ anomaly models simultaneously"))

            if not alerts_list:
                st.markdown('<div class="insight-block" style="border-left-color:#34d399;background:rgba(52,211,153,0.06);">No critical alerts — system within normal parameters.</div>', unsafe_allow_html=True)
            else:
                for kind, msg in alerts_list:
                    st.markdown(f'<div class="insight-block {kind if kind != "info" else ""}">{msg}</div>', unsafe_allow_html=True)

    # ------------------------------------------------------------------
    # SECTION: FUTURE SHOCK PREDICTIONS (HERO SECTION)
    # ------------------------------------------------------------------
    with st.expander("Future shock probability forecasts", expanded=True):
        _render_future_forecasts(rf_future_preds, rf_pred, PLOTLY_DARK)

    # ------------------------------------------------------------------
    # SECTION: GLOBAL RISK MAP
    # ------------------------------------------------------------------
    with st.expander("Global risk map", expanded=True):
        if anomaly_dash is not None and 'risk_score' in anomaly_dash.columns:
            fig_map = px.choropleth(
                anomaly_dash,
                locations='country',
                locationmode='country names',
                color='risk_score',
                color_continuous_scale='RdYlGn_r',
                range_color=[0, 20],
                hover_name='country',
                hover_data={'country': False, 'risk_score': ':.1f'},
                labels={'risk_score': 'Risk score'}
            )
            fig_map.update_layout(
                **PLOTLY_DARK,
                height=480,
                margin=dict(l=0, r=0, t=20, b=0),
                coloraxis_colorbar=dict(title="Risk", thickness=12, len=0.65, x=1.01)
            )
            fig_map.update_geos(
                showland=True, showcoastlines=True, showcountries=True,
                landcolor='rgba(22,33,45,0.9)',
                countrycolor='rgba(100,120,140,0.25)',
                coastlinecolor='rgba(120,140,160,0.3)'
            )
            st.plotly_chart(fig_map, use_container_width=True)
        else:
            st.info("Risk map data unavailable.")

    # ------------------------------------------------------------------
    # SECTION: COUNTRY DEEP DIVE
    # ------------------------------------------------------------------
    with st.expander("Country risk profile", expanded=False):
        if anomaly_dash is not None and len(anomaly_dash) > 0:
            countries = sorted(anomaly_dash['country'].unique())
            sel = st.selectbox("Select country", countries, key="t5_country_sel")
            row = anomaly_dash[anomaly_dash['country'] == sel]

            if not row.empty:
                row = row.iloc[0]
                rs = _safe(row.get('risk_score', 0))

                if rs >= 10:   risk_label, risk_color = "CRITICAL",  "#f87171"
                elif rs >= 8:  risk_label, risk_color = "ELEVATED",  "#fbbf24"
                elif rs >= 5:  risk_label, risk_color = "MODERATE",  "#60a5fa"
                else:          risk_label, risk_color = "STABLE",    "#34d399"

                flag_cols = [c for c in row.index if 'flag' in c.lower()]
                flag_n = sum(1 for c in flag_cols if row.get(c) == 1)

                # shock prob from future preds or rf_pred
                shock_prob = 0.0; prob_label = "N/A"; prob_color = "#64748b"
                if rf_future_preds is not None and 'probability_shock' in rf_future_preds.columns:
                    fp = rf_future_preds[rf_future_preds['country'] == sel]
                    if not fp.empty:
                        shock_prob = fp[fp['year'] == fp['year'].min()].iloc[0].get('probability_shock', 0)
                        if shock_prob > 0.6: prob_label, prob_color = "HIGH", "#f87171"
                        elif shock_prob > 0.5: prob_label, prob_color = "ELEVATED", "#fbbf24"
                        else: prob_label, prob_color = "MODERATE", "#34d399"
                elif rf_pred is not None:
                    fp = rf_pred[rf_pred['country'] == sel]
                    if not fp.empty:
                        shock_prob = _safe(fp.iloc[0].get('probability', 0))
                        if shock_prob > 0.7: prob_label, prob_color = "EXTREME", "#f87171"
                        elif shock_prob > 0.5: prob_label, prob_color = "HIGH", "#fbbf24"
                        else: prob_label, prob_color = "MODERATE", "#34d399"

                c1, c2, c3 = st.columns(3)
                with c1:
                    st.metric("Composite risk score", f"{rs:.1f} / 20", delta=risk_label)
                with c2:
                    st.metric("Shock probability", f"{shock_prob:.1%}", delta=prob_label)
                with c3:
                    consensus = "High" if flag_n >= 7 else "Moderate" if flag_n >= 5 else "Low"
                    st.metric("Model agreement", f"{flag_n} / 9", delta=f"{consensus} consensus")

                # Risk drivers
                driver_cols = ['import_dependency', 'price_volatility', 'supply_gap',
                               'debt_to_gdp', 'political_stability', 'inflation_rate', 'gdp_growth']
                drivers = {c: row[c] for c in driver_cols if c in row.index and pd.notna(row[c])}
                if drivers:
                    st.divider()
                    st.caption("Risk drivers")
                    dcols = st.columns(min(4, len(drivers)))
                    for i, (k, v) in enumerate(list(drivers.items())[:8]):
                        with dcols[i % 4]:
                            st.metric(k.replace('_', ' ').title(), f"{v:.2f}")

                # Model flags
                if flag_cols:
                    st.divider()
                    st.caption("Model flags (9 systems)")
                    fcols = st.columns(5)
                    for i, c in enumerate(flag_cols[:10]):
                        with fcols[i % 5]:
                            val = row.get(c, 0)
                            flagged = int(val) == 1 if pd.notna(val) else False
                            name = c.replace('_flag', '').replace('_', ' ').title()[:14]
                            st.metric(name, "Flagged" if flagged else "OK",
                                      delta="⚠" if flagged else None,
                                      delta_color="inverse" if flagged else "normal")
        else:
            st.info("No country data available.")

    # ------------------------------------------------------------------
    # SECTION: RISK SEGMENTATION
    # ------------------------------------------------------------------
    with st.expander("Risk segmentation & watchlist", expanded=False):
        if anomaly_dash is not None and 'risk_score' in anomaly_dash.columns:
            def _seg(s):
                if s >= 10: return "Critical (≥10)"
                elif s >= 8: return "Elevated (8–9)"
                elif s >= 5: return "Moderate (5–7)"
                return "Stable (<5)"

            seg = anomaly_dash.copy()
            seg['Segment'] = seg['risk_score'].apply(_seg)
            seg_counts = seg['Segment'].value_counts().reset_index()
            seg_counts.columns = ['Segment', 'Countries']

            col_s1, col_s2 = st.columns([1, 1])
            with col_s1:
                fig_pie = px.pie(
                    seg_counts, values='Countries', names='Segment',
                    color_discrete_sequence=['#f87171', '#fbbf24', '#60a5fa', '#34d399']
                )
                fig_pie.update_layout(**PLOTLY_DARK, height=280, showlegend=True,
                                      margin=dict(l=0, r=0, t=20, b=0))
                st.plotly_chart(fig_pie, use_container_width=True)

            with col_s2:
                n_show = st.slider("Top countries to show", 5, min(30, len(anomaly_dash)), 15, key="t5_wl")
                wl = anomaly_dash.nlargest(n_show, 'risk_score')[['country', 'risk_score']].copy()
                wl['Level'] = wl['risk_score'].apply(
                    lambda x: 'Critical' if x >= 10 else 'Elevated' if x >= 8 else 'Moderate'
                )
                wl = wl.rename(columns={'country': 'Country', 'risk_score': 'Score'})
                st.dataframe(wl, hide_index=True, use_container_width=True,
                             column_config={'Score': st.column_config.NumberColumn(format="%.1f")})

            fig_bar = px.bar(
                wl.sort_values('Score'),
                x='Score', y='Country', orientation='h',
                color='Score', color_continuous_scale='Reds',
                labels={'Score': 'Composite risk score'}
            )
            fig_bar.update_layout(
                **PLOTLY_DARK,
                height=max(350, 30 * len(wl) + 100),
                coloraxis_showscale=False,
                yaxis={'categoryorder': 'array', 'categoryarray': wl.sort_values('Score')['Country'].tolist()}
            )
            st.plotly_chart(fig_bar, use_container_width=True)

    # ------------------------------------------------------------------
    # SECTION: TEMPORAL TRENDS
    # ------------------------------------------------------------------
    with st.expander("Risk momentum & trends", expanded=False):
        if yearly_summary is not None and not yearly_summary.empty and 'total_anomalies' in yearly_summary.columns:
            ys = yearly_summary.sort_values('year')
            fig_tl = px.line(ys, x='year', y='total_anomalies', markers=True, line_shape='spline',
                             labels={'year': 'Year', 'total_anomalies': 'Anomaly count'})
            fig_tl.update_traces(line=dict(color='#60a5fa', width=2.5), marker=dict(size=8, color='#f87171'))
            fig_tl.update_layout(**PLOTLY_DARK, height=320)
            st.plotly_chart(fig_tl, use_container_width=True)

            ys = ys.copy()
            ys['YoY'] = ys['total_anomalies'].diff()
            fig_mom = px.bar(ys, x='year', y='YoY', color='YoY', color_continuous_scale='RdYlGn_r',
                             labels={'year': 'Year', 'YoY': 'YoY change'})
            fig_mom.update_layout(**PLOTLY_DARK, height=280)
            st.plotly_chart(fig_mom, use_container_width=True)
        else:
            st.info("Yearly trend data not available.")

    # ------------------------------------------------------------------
    # SECTION: DIAGNOSTICS & EXPORT
    # ------------------------------------------------------------------
    with st.expander("Diagnostics & export", expanded=False):
        if anomaly_dash is not None and 'risk_score' in anomaly_dash.columns:
            col_d1, col_d2 = st.columns(2)
            with col_d1:
                st.caption("Risk score statistics")
                rs = anomaly_dash['risk_score']
                stat_cols = st.columns(3)
                for i, (lbl, val) in enumerate([
                    ("Mean", rs.mean()), ("Median", rs.median()), ("Std dev", rs.std()),
                    ("Min", rs.min()), ("Max", rs.max()), ("IQR", rs.quantile(0.75) - rs.quantile(0.25))
                ]):
                    with stat_cols[i % 3]:
                        st.metric(lbl, f"{val:.2f}")

            with col_d2:
                st.caption("Data quality")
                st.metric("Total records", len(anomaly_dash))
                st.metric("Missing values", int(anomaly_dash.isnull().sum().sum()))
                if 'country' in anomaly_dash.columns:
                    st.metric("Unique countries", anomaly_dash['country'].nunique())

        st.divider()
        st.caption("Export")
        ecols = st.columns(3)

        with ecols[0]:
            if anomaly_dash is not None:
                st.download_button(
                    "Anomaly dashboard (CSV)",
                    anomaly_dash.to_csv(index=False),
                    f"anomaly_dashboard_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )

        with ecols[1]:
            if rf_future_preds is not None:
                st.download_button(
                    "Future predictions (CSV)",
                    rf_future_preds.to_csv(index=False),
                    f"rf_future_predictions_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )
            elif rf_pred is not None:
                st.download_button(
                    "RF predictions (CSV)",
                    rf_pred.to_csv(index=False),
                    f"rf_predictions_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )

        with ecols[2]:
            if anomaly_dash is not None:
                combined = anomaly_dash.copy()
                if rf_pred is not None and 'country' in rf_pred.columns:
                    combined = combined.merge(
                        rf_pred[['country', 'probability']].groupby('country', as_index=False).max(),
                        on='country', how='left'
                    )
                st.download_button(
                    "Combined report (CSV)",
                    combined.to_csv(index=False),
                    f"combined_risk_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )

    st.markdown(
        f"<div style='text-align:center;padding:1.5rem 0 0.5rem;color:#475569;font-size:0.78rem;'>"
        f"Risk Intelligence Dashboard · Pipeline data · {datetime.now().strftime('%Y-%m-%d %H:%M')} UTC"
        f"</div>",
        unsafe_allow_html=True
    )


# =========================================================================
# FUTURE FORECASTS — hero visualisation (called inline above)
# =========================================================================

def _render_future_forecasts(rf_future_preds, rf_pred, PLOTLY_DARK):
    """
    Dedicated future shock-probability section.
    Priority: rf_future_preds (multi-year); fallback: rf_pred.
    """
    import streamlit as st
    import pandas as pd
    import plotly.graph_objects as go
    import plotly.express as px

    # ---- case A: multi-year future predictions -------------------------
    if rf_future_preds is not None and 'probability_shock' in rf_future_preds.columns:
        df = rf_future_preds.copy()
        years = sorted(df['year'].unique())

        st.markdown('<div class="section-title">Shock probability by year · 2025–2029</div>',
                    unsafe_allow_html=True)

        # --- View A1: animated bubble map over years ---
        # Aggregate top-N per year for a ranked heatmap
        top_n = 10
        rows = []
        for yr in years:
            sub = df[df['year'] == yr].nlargest(top_n, 'probability_shock')[['country', 'probability_shock']].copy()
            sub['year'] = yr
            sub['rank'] = range(1, len(sub) + 1)
            rows.append(sub)

        ranked = pd.concat(rows, ignore_index=True)

        # Tab layout: Racing bar | Heatmap | Country trajectories
        tab_rb, tab_heat, tab_traj, tab_raw = st.tabs([
            "Forecast ranking", "Probability heatmap",
            "Country trajectories", "Full data table"
        ])

        # ---- TAB 1: Animated bar race ----
        with tab_rb:
            st.caption("Top 10 highest-risk countries by forecast year — use the slider to step through years.")
            sel_yr = st.select_slider(
                "Forecast year", options=years,
                key="t5_yr_slider"
            )
            sub = df[df['year'] == sel_yr].nlargest(15, 'probability_shock').sort_values('probability_shock')

            colors_bar = [
                '#f87171' if p > 0.6 else '#fbbf24' if p > 0.5 else '#60a5fa'
                for p in sub['probability_shock']
            ]
            fig_yr = go.Figure(go.Bar(
                x=sub['probability_shock'],
                y=sub['country'],
                orientation='h',
                marker_color=colors_bar,
                text=[f"{p:.1%}" for p in sub['probability_shock']],
                textposition='outside',
                hovertemplate='<b>%{y}</b><br>P(shock): %{x:.2%}<extra></extra>'
            ))
            fig_yr.add_vline(x=0.5, line_dash='dot', line_color='rgba(255,255,255,0.25)',
                             annotation_text="50% threshold", annotation_position="top right")
            fig_yr.add_vline(x=0.6, line_dash='dot', line_color='rgba(248,113,113,0.5)',
                             annotation_text="60% high-risk", annotation_position="top right")
            fig_yr.update_layout(
                **PLOTLY_DARK,
                height=480,
                xaxis=dict(range=[0, min(1.0, sub['probability_shock'].max() * 1.15)],
                           tickformat='.0%', title='Shock probability'),
                yaxis=dict(title=''),
                margin=dict(l=120, r=80, t=40, b=40),
                showlegend=False
            )
            st.plotly_chart(fig_yr, use_container_width=True)

            # Summary badges for selected year
            n_high = int((df[df['year'] == sel_yr]['probability_shock'] > 0.6).sum())
            n_watch = int((df[df['year'] == sel_yr]['probability_shock'] > 0.5).sum())
            st.markdown(f"""
            <div class="alert-row">
                <span class="badge badge-red">{n_high} countries P > 60%</span>
                <span class="badge badge-amber">{n_watch} countries P > 50%</span>
                <span class="badge badge-green">{sel_yr} forecast</span>
            </div>
            """, unsafe_allow_html=True)

        # ---- TAB 2: Heatmap — countries × years ----
        with tab_heat:
            st.caption("Shock probability heatmap — rows are countries, columns are years. Only the top 30 most-volatile countries are shown.")

            # Select countries with highest mean probability
            mean_prob = df.groupby('country')['probability_shock'].mean()
            top_countries = mean_prob.nlargest(30).index.tolist()

            pivot = df[df['country'].isin(top_countries)].pivot_table(
                index='country', columns='year', values='probability_shock', aggfunc='mean'
            ).round(4)
            pivot = pivot.loc[pivot.mean(axis=1).sort_values(ascending=False).index]

            fig_heat = go.Figure(go.Heatmap(
                z=pivot.values,
                x=[str(y) for y in pivot.columns],
                y=pivot.index.tolist(),
                colorscale=[
                    [0.0,  '#1e3a5f'],
                    [0.4,  '#1d4ed8'],
                    [0.55, '#fbbf24'],
                    [0.7,  '#f87171'],
                    [1.0,  '#7f1d1d']
                ],
                zmin=0, zmax=1,
                hovertemplate='<b>%{y}</b><br>Year: %{x}<br>P(shock): %{z:.2%}<extra></extra>',
                colorbar=dict(
                    title='P(shock)', tickformat='.0%',
                    thickness=12, len=0.7
                )
            ))
            fig_heat.update_layout(
                **PLOTLY_DARK,
                height=max(500, 22 * len(pivot) + 120),
                xaxis=dict(title='', side='top'),
                yaxis=dict(title='', autorange='reversed'),
                margin=dict(l=110, r=60, t=60, b=20)
            )
            st.plotly_chart(fig_heat, use_container_width=True)

        # ---- TAB 3: Country trajectories ----
        with tab_traj:
            st.caption("Track how shock probability evolves year-by-year for selected countries.")

            all_countries = sorted(df['country'].unique())
            # Default: top 5 by max probability
            defaults = df.groupby('country')['probability_shock'].max().nlargest(5).index.tolist()
            sel_countries = st.multiselect(
                "Select countries", all_countries,
                default=defaults, key="t5_traj_sel"
            )

            if sel_countries:
                traj_df = df[df['country'].isin(sel_countries)]
                fig_traj = px.line(
                    traj_df, x='year', y='probability_shock',
                    color='country', markers=True, line_shape='spline',
                    labels={'probability_shock': 'P(shock)', 'year': 'Year', 'country': 'Country'},
                    color_discrete_sequence=px.colors.qualitative.Vivid
                )
                fig_traj.add_hline(y=0.5, line_dash='dot', line_color='rgba(255,255,255,0.2)',
                                   annotation_text="50% line")
                fig_traj.add_hline(y=0.6, line_dash='dot', line_color='rgba(248,113,113,0.4)',
                                   annotation_text="60% high-risk")
                fig_traj.update_traces(line_width=2.5, marker_size=9)
                fig_traj.update_layout(
                    **PLOTLY_DARK, height=420,
                    yaxis=dict(tickformat='.0%', range=[0, 1.0]),
                    xaxis=dict(dtick=1),
                    legend=dict(orientation='h', y=-0.15)
                )
                st.plotly_chart(fig_traj, use_container_width=True)
            else:
                st.info("Select at least one country above.")

        # ---- TAB 4: Raw sortable table ----
        with tab_raw:
            st.caption("Full future predictions dataset. Filter and sort below.")

            col_f1, col_f2, col_f3 = st.columns([1, 1, 1])
            with col_f1:
                yr_filter = st.multiselect("Year", sorted(df['year'].unique()),
                                           default=list(df['year'].unique()), key="t5_raw_yr")
            with col_f2:
                min_p = st.slider("Min P(shock)", 0.0, 1.0, 0.0, 0.05, key="t5_raw_minp")
            with col_f3:
                sort_asc = st.radio("Sort", ["Highest first", "Lowest first"],
                                    horizontal=True, key="t5_raw_sort")

            filt = df[df['year'].isin(yr_filter) & (df['probability_shock'] >= min_p)].copy()
            filt = filt.sort_values('probability_shock', ascending=(sort_asc == "Lowest first"))

            def _risk_label(p):
                if p > 0.6: return "High"
                elif p > 0.5: return "Elevated"
                return "Watch"

            filt['Risk level'] = filt['probability_shock'].apply(_risk_label)
            display_cols = ['country', 'year', 'probability_shock', 'Risk level']
            if 'prediction' in filt.columns:
                display_cols.insert(2, 'prediction')

            st.dataframe(
                filt[display_cols].reset_index(drop=True),
                hide_index=True,
                use_container_width=True,
                column_config={
                    'country': 'Country',
                    'year': 'Year',
                    'prediction': st.column_config.NumberColumn('Prediction', format="%.0f"),
                    'probability_shock': st.column_config.ProgressColumn(
                        'P(shock)', format="%.2f", min_value=0, max_value=1
                    ),
                    'Risk level': 'Risk level'
                }
            )
            st.caption(f"{len(filt):,} rows shown")

    # ---- case B: single-year rf_pred fallback -------------------------
    elif rf_pred is not None and 'probability' in rf_pred.columns:
        st.markdown('<div class="section-title">Shock probability forecast</div>', unsafe_allow_html=True)
        rp = rf_pred.dropna(subset=['country', 'probability']).copy()

        time_col = next((c for c in ['date', 'Date', 'year', 'Year'] if c in rp.columns), None)
        agg = st.radio("Aggregation", ["Maximum", "Average", "Latest"],
                       horizontal=True, key="t5_agg_fallback")

        if agg == "Latest" and time_col:
            base = rp[rp[time_col] == rp[time_col].max()].groupby('country', as_index=False)['probability'].max()
        elif agg == "Average":
            base = rp.groupby('country', as_index=False)['probability'].mean()
        else:
            base = rp.groupby('country', as_index=False)['probability'].max()

        col_p1, col_p2 = st.columns([2, 1])
        with col_p1:
            n = st.slider("Countries to show", 5, min(25, len(base)), 12, key="t5_fb_n")
            top = base.nlargest(n, 'probability').sort_values('probability')
            bar_colors = ['#f87171' if p > 0.7 else '#fbbf24' if p > 0.5 else '#60a5fa'
                          for p in top['probability']]
            fig_fb = go.Figure(go.Bar(
                x=top['probability'], y=top['country'], orientation='h',
                marker_color=bar_colors,
                text=[f"{p:.1%}" for p in top['probability']],
                textposition='outside'
            ))
            fig_fb.update_layout(
                **PLOTLY_DARK, height=max(350, 28 * n + 100),
                xaxis=dict(tickformat='.0%'),
                yaxis=dict(categoryorder='array', categoryarray=top['country'].tolist()),
                showlegend=False
            )
            st.plotly_chart(fig_fb, use_container_width=True)
        with col_p2:
            st.metric("Mean P(shock)", f"{base['probability'].mean():.1%}")
            st.metric("Median P(shock)", f"{base['probability'].median():.1%}")
            st.metric("Countries > 70%", int((base['probability'] > 0.7).sum()))
            st.metric("Countries > 50%", int((base['probability'] > 0.5).sum()))
    else:
        st.info("No forecast data available. Run the prediction pipeline to generate forecasts.")


# =========================================================================
# END OF TAB 5
# =========================================================================
