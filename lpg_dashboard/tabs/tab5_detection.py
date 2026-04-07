# =========================================================================
# TAB 5 — RISK DASHBOARD & EARLY WARNING SYSTEM (FULLY ENHANCED)
# =========================================================================

def render_detection(data, filtered_df, filters):
    """Tab5 Detection render function with map, time slider, collapsible sections, and exports"""
    import streamlit as st
    import pandas as pd
    import plotly.express as px
    from utils.constants import PLOTLY_DARK

    # Custom CSS for dashboard styling
    st.markdown("""
    <style>
    .dashboard-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #0f3460 100%);
        padding: 1.8rem;
        border-radius: 20px;
        margin-bottom: 1.5rem;
        text-align: center;
        border-bottom: 3px solid #ff6b6b;
    }
    .risk-card {
        background: #1e1e2f;
        border-radius: 16px;
        padding: 1.2rem;
        margin: 0.5rem 0;
        transition: transform 0.2s;
    }
    .risk-card:hover {
        transform: translateY(-3px);
        background: #2a2a3e;
    }
    .risk-high { border-left: 4px solid #ea5455; }
    .risk-medium { border-left: 4px solid #ffd93d; }
    .risk-low { border-left: 4px solid #4ecdc4; }
    .risk-score {
        font-size: 2rem;
        font-weight: bold;
    }
    .warning-badge {
        display: inline-block;
        background: #ea5455;
        color: white;
        padding: 0.2rem 0.8rem;
        border-radius: 20px;
        font-size: 0.7rem;
        font-weight: bold;
        animation: pulse 1.5s infinite;
    }
    @keyframes pulse {
        0% { opacity: 0.6; }
        50% { opacity: 1; }
        100% { opacity: 0.6; }
    }
    .metric-tile {
        background: #1a1a2e;
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)

    # =====================================================================
    # DASHBOARD HEADER
    # =====================================================================
    st.markdown("""
    <div class="dashboard-header">
        <div style="font-size: 1.8rem; font-weight: 700;">⚠️ RISK DASHBOARD & EARLY WARNING SYSTEM</div>
        <div style="font-size: 0.9rem; opacity: 0.8; margin-top: 0.5rem;">
            Real‑time risk scores | Predictive alerts | Country vulnerability heatmap
        </div>
        <div style="margin-top: 0.8rem;">
            <span class="warning-badge">🔴 HIGH RISK ALERT</span>
            <span class="warning-badge" style="background:#ffd93d; color:#1a1a2e;">🟡 MEDIUM RISK</span>
            <span class="warning-badge" style="background:#4ecdc4;">🟢 LOW RISK</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Load required data
    shock_sev = data.get('shock_severity')
    rf_pred = data.get('rf_predictions')
    anomaly_dash = data.get('anomaly_dashboard')
    global_rank = data.get('shock_global_country_rank')
    yearly_summary = data.get('combined_yearly_summary')  # for time slider

    # -----------------------------------------------------------------
    # GLOBAL RISK METRICS (always visible)
    # -----------------------------------------------------------------
    st.markdown("## 📊 GLOBAL RISK OVERVIEW")
    
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    
    with col_m1:
        high_risk_count = 0
        if global_rank is not None and 'total_impact' in global_rank.columns:
            high_risk_count = len(global_rank[global_rank['total_impact'] > global_rank['total_impact'].quantile(0.9)])
        st.markdown(f"""
        <div class="metric-tile">
            <div style="font-size: 1.8rem;">🔴</div>
            <div style="font-size: 1.6rem; font-weight: bold;">{high_risk_count}</div>
            <div>High‑Risk Countries</div>
            <div style="font-size: 0.7rem;">Top 10% vulnerability</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col_m2:
        total_anomalies = 0
        if anomaly_dash is not None and 'risk_score' in anomaly_dash.columns:
            total_anomalies = int(anomaly_dash['risk_score'].sum())
        st.markdown(f"""
        <div class="metric-tile">
            <div style="font-size: 1.8rem;">⚠️</div>
            <div style="font-size: 1.6rem; font-weight: bold;">{total_anomalies:,}</div>
            <div>Total Risk Score (Dashboard)</div>
            <div style="font-size: 0.7rem;">172 countries</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col_m3:
        shock_prob_avg = 0
        if rf_pred is not None and 'probability' in rf_pred.columns:
            shock_prob_avg = rf_pred['probability'].mean()
        st.markdown(f"""
        <div class="metric-tile">
            <div style="font-size: 1.8rem;">🔮</div>
            <div style="font-size: 1.6rem; font-weight: bold;">{shock_prob_avg:.1%}</div>
            <div>Avg. Next‑Year Shock Probability</div>
            <div style="font-size: 0.7rem;">Global forecast</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col_m4:
        high_risk_prob = 0
        if rf_pred is not None and 'probability' in rf_pred.columns:
            high_risk_prob = (rf_pred['probability'] > 0.7).mean()
        st.markdown(f"""
        <div class="metric-tile">
            <div style="font-size: 1.8rem;">🚨</div>
            <div style="font-size: 1.6rem; font-weight: bold;">{high_risk_prob:.1%}</div>
            <div>Countries with p(Shock) > 0.7</div>
            <div style="font-size: 0.7rem;">Immediate attention needed</div>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    # -----------------------------------------------------------------
    # SECTION 1: GLOBAL RISK MAP & TIME SLIDER (collapsible)
    # -----------------------------------------------------------------
    with st.expander("🗺️ Global Risk Map & Historical Trend", expanded=True):
        # --- Time slider (if yearly data exists) ---
        if yearly_summary is not None and not yearly_summary.empty:
            st.markdown("#### 📅 Historical Risk Evolution")
            fig_line = px.line(
                yearly_summary, x='year', y='total_anomalies',
                markers=True, title='Global Anomaly Count Over Years'
            )
            fig_line.update_layout(**PLOTLY_DARK, height=300)
            st.plotly_chart(fig_line, use_container_width=True)
            
            # Slider to filter data (optional, can be used later)
            min_year = int(yearly_summary['year'].min())
            max_year = int(yearly_summary['year'].max())
            selected_years = st.slider("Filter data by year range", min_year, max_year, (min_year, max_year))
        else:
            selected_years = None
            st.info("Yearly anomaly summary not available – time slider disabled.")

        # --- Choropleth map (click to select country) ---
        if anomaly_dash is not None and 'risk_score' in anomaly_dash.columns:
            st.markdown("#### 🌍 Interactive Risk Map – click on any country")
            # Prepare data: ensure country names are recognised by plotly
            map_data = anomaly_dash.copy()
            # If region column missing, we can still plot
            fig_map = px.choropleth(
                map_data,
                locations='country',
                locationmode='country names',
                color='risk_score',
                color_continuous_scale='Reds',
                range_color=[0, 20],
                title='Composite Risk Score by Country',
                labels={'risk_score': 'Risk Score'}
            )
            fig_map.update_layout(height=500, **PLOTLY_DARK)
            st.plotly_chart(fig_map, use_container_width=True, key="risk_map")
            
            clicked_country = None
            if st.session_state.get("risk_map", {}).get("points"):
                clicked_country = st.session_state["risk_map"]["points"][0]["location"]
                st.session_state["selected_country"] = clicked_country
            elif "selected_country" in st.session_state:
                clicked_country = st.session_state["selected_country"]

        else:
            st.warning("Anomaly dashboard data missing – cannot render map.")
            clicked_country = None

    # -----------------------------------------------------------------
    # SECTION 2: EXECUTIVE INSIGHTS & EARLY WARNINGS (collapsible)
    # -----------------------------------------------------------------
    with st.expander("🧠 System Insights & Early Warning Triggers", expanded=True):
        # --- Key System Insights ---
        st.markdown("#### 📌 Key System Insights")
        insights = []
        if anomaly_dash is not None and 'risk_score' in anomaly_dash.columns:
            avg_risk = anomaly_dash['risk_score'].mean()
            high_cluster = (anomaly_dash['risk_score'] >= 10).sum()
            insights.append(f"• System-wide average risk score is {avg_risk:.2f}, indicating {'elevated' if avg_risk > 7 else 'stable'} baseline conditions.")
            insights.append(f"• {high_cluster} countries fall into critical risk tier (score ≥ 10), forming a concentrated vulnerability cluster.")
        if rf_pred is not None and 'probability' in rf_pred.columns:
            spike = (rf_pred['probability'] > 0.7).sum()
            insights.append(f"• {spike} countries show high forward-looking shock probability (>70%), signaling potential near-term disruptions.")
        if anomaly_dash is not None:
            flag_cols = [c for c in anomaly_dash.columns if 'flag' in c.lower()]
            if flag_cols:
                avg_flags = anomaly_dash[flag_cols].sum(axis=1).mean()
                insights.append(f"• Average model agreement is {avg_flags:.2f} flags per country, reflecting ensemble confidence level.")
        for i in insights:
            st.markdown(i)
        
        st.divider()
        
        # --- Early Warning Triggers ---
        st.markdown("#### 🚨 Early Warning Signals")
        alerts = []
        if anomaly_dash is not None and 'risk_score' in anomaly_dash.columns:
            if (anomaly_dash['risk_score'] >= 10).any():
                alerts.append("🔴 Critical risk threshold breached (score ≥ 10 detected)")
        if rf_pred is not None and 'probability' in rf_pred.columns:
            if (rf_pred['probability'] > 0.8).any():
                alerts.append("🔴 Extreme shock probability (>80%) detected")
        if anomaly_dash is not None:
            flag_cols = [c for c in anomaly_dash.columns if 'flag' in c.lower()]
            if flag_cols:
                if (anomaly_dash[flag_cols].sum(axis=1) >= 7).any():
                    alerts.append("🔴 High model consensus anomaly (≥7 flags)")
        if alerts:
            for a in alerts:
                st.error(a)
        else:
            st.success("🟢 No critical early warning triggers detected")

    # -----------------------------------------------------------------
    # SECTION 3: COUNTRY DEEP DIVE (from map click or selector)
    # -----------------------------------------------------------------
    with st.expander("🔍 Country Risk Profile (click on map or select below)", expanded=True):
        # If no country clicked, allow manual selection
        if not clicked_country and anomaly_dash is not None:
            all_countries = sorted(anomaly_dash['country'].unique())
            selected_country = st.selectbox("Or select a country manually", all_countries)
        else:
            selected_country = clicked_country
        
        if selected_country and anomaly_dash is not None:
            country_row = anomaly_dash[anomaly_dash['country'] == selected_country]
            if not country_row.empty:
                # Three risk cards
                risk_score_val = country_row['risk_score'].iloc[0] if 'risk_score' in country_row.columns else None
                risk_level = "Low"
                risk_color = "#4ecdc4"
                if risk_score_val:
                    if risk_score_val >= 10:
                        risk_level = "High"
                        risk_color = "#ea5455"
                    elif risk_score_val >= 8:
                        risk_level = "Medium"
                        risk_color = "#ffd93d"
                
                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    st.markdown(f"""
                    <div class="risk-card risk-{risk_level.lower()}">
                        <div style="font-size: 0.8rem;">COMPOSITE RISK SCORE</div>
                        <div class="risk-score" style="color:{risk_color};">{risk_score_val if risk_score_val else 'N/A'}</div>
                        <div>Risk Level: <b>{risk_level}</b></div>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Next-year probability
                next_prob = None
                if rf_pred is not None and 'probability' in rf_pred.columns:
                    pred_row = rf_pred[rf_pred['country'] == selected_country]
                    if not pred_row.empty:
                        next_prob = pred_row['probability'].max()
                with col_b:
                    prob_color = "#ea5455" if next_prob and next_prob > 0.7 else "#ffd93d" if next_prob and next_prob > 0.4 else "#4ecdc4"
                    st.markdown(f"""
                    <div class="risk-card">
                        <div style="font-size: 0.8rem;">NEXT-YEAR SHOCK PROB.</div>
                        <div class="risk-score" style="color:{prob_color};">{f"{next_prob:.1%}" if next_prob else "N/A"}</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Anomaly flag count
                flag_cols = [c for c in country_row.columns if 'flag' in c.lower()]
                flag_count = country_row[flag_cols].sum().sum() if flag_cols else 0
                with col_c:
                    fc_color = "#e74c3c" if flag_count >= 8 else "#f39c12" if flag_count >= 5 else "#2ecc71"
                    st.markdown(f"""
                    <div class="risk-card">
                        <div style="font-size: 0.8rem;">ANOMALY FLAGS (9 Models)</div>
                        <div class="risk-score" style="color:{fc_color};">{int(flag_count)}</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Risk Drivers (if columns exist)
                st.markdown("#### 🔍 Risk Drivers")
                driver_cols = ['import_dependency', 'price_volatility', 'supply_gap', 'debt_to_gdp', 'political_stability']
                drivers_found = []
                for col in driver_cols:
                    if col in country_row.columns:
                        val = country_row[col].iloc[0]
                        if pd.notna(val):
                            drivers_found.append(f"{col.replace('_', ' ').title()}: {val:.2f}")
                if drivers_found:
                    for d in drivers_found:
                        st.write("•", d)
                else:
                    st.info("No detailed driver data for this country.")
                
                # Model flags breakdown
                if flag_cols:
                    st.markdown("#### 📊 Model Flags Breakdown")
                    flags_dict = {col: int(country_row[col].iloc[0]) for col in flag_cols if pd.notna(country_row[col].iloc[0])}
                    col_flags = st.columns(min(4, len(flags_dict)))
                    for i, (model, val) in enumerate(flags_dict.items()):
                        with col_flags[i % 4]:
                            color = "🟢" if val == 0 else "🟡" if val == 1 else "🔴"
                            st.metric(color + model.replace('_flag', '').title(), val)
        else:
            st.info("Click on a country in the map or select one from the dropdown to see detailed risk profile.")

    # -----------------------------------------------------------------
    # SECTION 4: RISK SEGMENTATION & WATCHLIST (collapsible)
    # -----------------------------------------------------------------
    with st.expander("📊 Risk Segmentation & High‑Risk Watchlist", expanded=False):
        if anomaly_dash is not None and 'risk_score' in anomaly_dash.columns:
            # Segmentation
            def segment(x):
                if x >= 10: return "Critical"
                elif x >= 8: return "Vulnerable"
                elif x >= 5: return "Watchlist"
                else: return "Stable"
            anomaly_dash['segment'] = anomaly_dash['risk_score'].apply(segment)
            seg_counts = anomaly_dash['segment'].value_counts().reset_index()
            seg_counts.columns = ['Risk Segment', 'Number of Countries']
            st.markdown("#### 🧩 Risk Segments")
            st.dataframe(seg_counts, use_container_width=True, hide_index=True)
            
            # Watchlist
            st.markdown("#### 🚨 Top 10 Highest Risk Countries")
            watch_df = anomaly_dash.nlargest(10, 'risk_score')[['country', 'risk_score']].copy()
            watch_df['risk_level'] = watch_df['risk_score'].apply(
                lambda x: '🔴 HIGH' if x >= 10 else '🟡 MEDIUM' if x >= 8 else '🟢 MODERATE'
            )
            st.dataframe(watch_df, use_container_width=True, hide_index=True,
                         column_config={'risk_score': st.column_config.NumberColumn("Risk Score", format="%d")})
            
            # Bar chart
            fig_bar = px.bar(
                watch_df.sort_values('risk_score').tail(10),
                x='risk_score', y='country', orientation='h',
                color='risk_score', color_continuous_scale='Reds',
                title="Top 10 Riskiest Countries"
            )
            fig_bar.update_layout(height=350, **PLOTLY_DARK, coloraxis_showscale=False)
            st.plotly_chart(fig_bar, use_container_width=True)
        else:
            st.info("Anomaly dashboard missing – cannot show segmentation.")

    # -----------------------------------------------------------------
    # SECTION 5: CASCADE RISK & MOMENTUM (collapsible)
    # -----------------------------------------------------------------
    with st.expander("🌐 Cascade Risk & Risk Momentum", expanded=False):
        # Cascade risk
        if anomaly_dash is not None and rf_pred is not None:
            if 'country' in anomaly_dash.columns and 'country' in rf_pred.columns:
                merged = anomaly_dash[['country', 'risk_score']].merge(
                    rf_pred[['country', 'probability']], on='country', how='inner'
                )
                merged['cascade_risk'] = merged['risk_score'] * merged['probability']
                top_cascade = merged.nlargest(10, 'cascade_risk')
                st.markdown("#### 🌊 Top Potential Shock Amplifiers")
                st.dataframe(top_cascade[['country', 'cascade_risk']], use_container_width=True)
                st.warning("⚠️ These countries are high-impact + high-probability → potential global shock amplifiers.")
        else:
            st.info("Cascade risk requires anomaly_dashboard and rf_predictions.")
        
        st.divider()
        
        # Risk momentum
        st.markdown("#### 📈 Risk Momentum (Year‑over‑Year Change)")
        if shock_sev is not None and 'country' in shock_sev.columns and 'year' in shock_sev.columns and 'shock_severity' in shock_sev.columns:
            shock_sev['year'] = pd.to_numeric(shock_sev['year'], errors='coerce')
            shock_sev = shock_sev.dropna(subset=['year', 'shock_severity'])
            recent = shock_sev.sort_values('year').groupby('country').tail(2)
            if len(recent) >= 2:
                trend = recent.groupby('country')['shock_severity'].diff().dropna()
                rising = (trend > 0).sum()
                falling = (trend < 0).sum()
                col_mom1, col_mom2 = st.columns(2)
                col_mom1.metric("📈 Countries with Rising Risk", rising)
                col_mom2.metric("📉 Countries with Improving Risk", falling)
            else:
                st.info("Insufficient historical data to compute momentum.")
        else:
            st.info("Shock severity time series not available.")

    # -----------------------------------------------------------------
    # SECTION 6: MODEL DIAGNOSTICS & EXPORT (collapsible)
    # -----------------------------------------------------------------
    with st.expander("📈 Model Diagnostics & Data Export", expanded=False):
        if rf_pred is not None and 'probability' in rf_pred.columns:
            st.markdown("#### 🔮 Random Forest Predictions")
            col_rf1, col_rf2 = st.columns(2)
            with col_rf1:
                top_rf = rf_pred.nlargest(10, 'probability')
                fig_rf = px.bar(
                    top_rf.sort_values('probability'), x='probability', y='country',
                    orientation='h', color='probability',
                    title="Highest Next‑Year Shock Probabilities",
                    color_continuous_scale='RdYlGn_r'
                )
                fig_rf.update_layout(height=400, **PLOTLY_DARK)
                st.plotly_chart(fig_rf, use_container_width=True)
            with col_rf2:
                st.metric("Average P(Shock)", f"{rf_pred['probability'].mean():.1%}")
                st.metric("High Risk (>70%)", f"{(rf_pred['probability'] > 0.7).sum()} countries")
        
        # Risk score distribution
        if anomaly_dash is not None and 'risk_score' in anomaly_dash.columns:
            st.markdown("#### 📊 Risk Score Distribution")
            fig_hist = px.histogram(
                anomaly_dash, x='risk_score', nbins=12,
                title="Distribution of Composite Risk Scores",
                labels={'risk_score': 'Risk Score'}
            )
            fig_hist.update_layout(height=350, **PLOTLY_DARK)
            st.plotly_chart(fig_hist, use_container_width=True)
        
        # Download button
        if anomaly_dash is not None:
            csv_data = anomaly_dash.to_csv(index=False)
            st.download_button(
                label="📥 Download Full Risk Data (CSV)",
                data=csv_data,
                file_name="risk_dashboard_data.csv",
                mime="text/csv"
            )

    # FINAL SUMMARY
    st.markdown("""
    <div style="text-align:center; padding:2rem; background:rgba(13,27,42,0.5); border-radius:16px; margin-top:2rem;">
        <h3>✅ Dashboard Complete</h3>
        <p><b>Tab5:</b> Risk dashboard with map, time slider, EWS, segmentation, cascade, momentum, and exports.<br>
        Data from <code>pipeline_checkpoints</code> → fully functional!</p>
    </div>
    """, unsafe_allow_html=True)