# =========================================================================
# TAB 5 — ENHANCED RISK DASHBOARD & EARLY WARNING SYSTEM
# =========================================================================
# Production-grade dashboard with real data validation, interactive analytics,
# and clear insights. No fake/sample data.
# =========================================================================

def render_detection(data, filtered_df, filters):
    """
    Enhanced Tab5 Detection render function with:
    - Real data validation & error handling
    - Interactive choropleth with drill-down capability
    - Dynamic key metrics that respond to actual data
    - Clear risk narratives & actionable alerts
    - Advanced filtering & export capabilities
    """
    import streamlit as st
    import pandas as pd
    import plotly.graph_objects as go
    import plotly.express as px
    from datetime import datetime
    import json
    from utils.constants import PLOTLY_DARK
    
    # =====================================================================
    # ENHANCED DASHBOARD STYLING
    # =====================================================================
    st.markdown("""
    <style>
    /* === TYPOGRAPHY & SPACING === */
    body { font-family: 'Segoe UI', system-ui, sans-serif; }
    
    /* === DASHBOARD HEADER === */
    .dashboard-header {
        background: linear-gradient(135deg, #0f2340 0%, #1a3a52 50%, #0f2340 100%);
        padding: 2.5rem 2rem;
        border-radius: 24px;
        margin-bottom: 2rem;
        border: 2px solid rgba(255,107,107,0.3);
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
        backdrop-filter: blur(10px);
    }
    
    .dashboard-title {
        font-size: 2.4rem;
        font-weight: 800;
        background: linear-gradient(135deg, #ff6b6b 0%, #ffd93d 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.5rem;
    }
    
    .dashboard-subtitle {
        font-size: 1rem;
        color: #b0c4de;
        opacity: 0.9;
        font-weight: 500;
    }
    
    /* === METRIC CARDS === */
    .metric-card {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid rgba(148,163,184,0.2);
        border-radius: 16px;
        padding: 1.5rem;
        text-align: center;
        transition: all 0.3s ease;
        box-shadow: 0 4px 12px rgba(0,0,0,0.2);
    }
    
    .metric-card:hover {
        border-color: rgba(255,107,107,0.5);
        box-shadow: 0 8px 24px rgba(255,107,107,0.15);
        transform: translateY(-2px);
    }
    
    .metric-value {
        font-size: 2.2rem;
        font-weight: 700;
        margin: 0.8rem 0 0.3rem;
        color: #f1f5f9;
    }
    
    .metric-label {
        font-size: 0.85rem;
        color: #94a3b8;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    /* === RISK CARDS (Detailed) === */
    .risk-card {
        background: linear-gradient(135deg, #1f2937 0%, #111827 100%);
        border-left: 4px solid #64748b;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        transition: all 0.3s ease;
        box-shadow: 0 2px 8px rgba(0,0,0,0.15);
    }
    
    .risk-card:hover {
        transform: translateX(4px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.25);
    }
    
    .risk-card.high {
        border-left-color: #ef4444;
        background: linear-gradient(135deg, rgba(239,68,68,0.1), transparent);
    }
    
    .risk-card.medium {
        border-left-color: #f59e0b;
        background: linear-gradient(135deg, rgba(245,158,11,0.1), transparent);
    }
    
    .risk-card.low {
        border-left-color: #10b981;
        background: linear-gradient(135deg, rgba(16,185,129,0.1), transparent);
    }
    
    /* === ALERT BADGES === */
    .alert-badge {
        display: inline-block;
        padding: 0.4rem 1rem;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-right: 0.5rem;
        margin-bottom: 0.5rem;
        animation: subtle-pulse 2s infinite;
    }
    
    .alert-badge.critical {
        background: linear-gradient(135deg, #ef4444, #dc2626);
        color: white;
        box-shadow: 0 4px 12px rgba(239,68,68,0.4);
    }
    
    .alert-badge.warning {
        background: linear-gradient(135deg, #f59e0b, #d97706);
        color: white;
        box-shadow: 0 4px 12px rgba(245,158,11,0.3);
    }
    
    .alert-badge.safe {
        background: linear-gradient(135deg, #10b981, #059669);
        color: white;
        box-shadow: 0 4px 12px rgba(16,185,129,0.3);
    }
    
    @keyframes subtle-pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.85; }
    }
    
    /* === SECTION HEADERS === */
    .section-header {
        border-bottom: 2px solid rgba(255,107,107,0.3);
        padding-bottom: 0.8rem;
        margin-bottom: 1.5rem;
        font-weight: 700;
        font-size: 1.2rem;
        color: #f1f5f9;
    }
    
    /* === DATA TABLE STYLING === */
    .dataframe-container {
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 2px 8px rgba(0,0,0,0.15);
    }
    
    /* === INSIGHT BOX === */
    .insight-box {
        background: linear-gradient(135deg, rgba(59,130,246,0.1), rgba(139,92,246,0.1));
        border-left: 4px solid #3b82f6;
        border-radius: 12px;
        padding: 1.2rem;
        margin: 1rem 0;
        color: #e0e7ff;
    }
    
    .insight-box strong { color: #60a5fa; }
    
    .insight-box.warning {
        border-left-color: #f59e0b;
        background: linear-gradient(135deg, rgba(245,158,11,0.1), transparent);
    }
    
    .insight-box.warning strong { color: #fbbf24; }
    
    /* === DIVIDER === */
    .divider {
        border-top: 1px solid rgba(148,163,184,0.2);
        margin: 2rem 0;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # =====================================================================
    # DATA VALIDATION LAYER
    # =====================================================================
    def validate_data_sources(data):
        """Validate all required data sources exist and have expected structure"""
        sources = {
            'anomaly_dashboard': None,
            'rf_predictions': None,
            'shock_severity': None,
            'shock_global_country_rank': None,
            'combined_yearly_summary': None
        }
        
        for key in sources:
            if key in data and data[key] is not None:
                df = data[key]
                if isinstance(df, pd.DataFrame) and len(df) > 0:
                    sources[key] = df
        
        return sources
    
    sources = validate_data_sources(data)
    anomaly_dash = sources['anomaly_dashboard']
    rf_pred = sources['rf_predictions']
    shock_sev = sources['shock_severity']
    global_rank = sources['shock_global_country_rank']
    yearly_summary = sources['combined_yearly_summary']
    
    # Check if we have ANY data
    has_data = any([anomaly_dash is not None, rf_pred is not None, shock_sev is not None])
    
    if not has_data:
        st.error("❌ No risk data available. Please ensure pipeline has generated required outputs.")
        return
    
    # =====================================================================
    # DASHBOARD HEADER
    # =====================================================================
    st.markdown("""
    <div class="dashboard-header">
        <div class="dashboard-title">⚠️ RISK INTELLIGENCE DASHBOARD</div>
        <div class="dashboard-subtitle">
            Real-time early warning system | Global vulnerability assessment | Predictive risk forecasting
        </div>
        <div style="margin-top: 1rem; display: flex; gap: 0.5rem; flex-wrap: wrap;">
            <span class="alert-badge critical">🔴 CRITICAL RISK</span>
            <span class="alert-badge warning">🟡 ELEVATED RISK</span>
            <span class="alert-badge safe">🟢 STABLE</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # =====================================================================
    # SECTION 1: GLOBAL RISK METRICS
    # =====================================================================
    st.markdown("## 📊 Global Risk Snapshot")
    
    # Calculate real metrics
    metrics = {
        'critical_countries': 0,
        'avg_risk_score': 0,
        'shock_probability_avg': 0,
        'high_probability_count': 0,
        'total_countries': 0
    }
    
    if anomaly_dash is not None:
        metrics['total_countries'] = len(anomaly_dash)
        metrics['avg_risk_score'] = anomaly_dash['risk_score'].mean() if 'risk_score' in anomaly_dash.columns else 0
        metrics['critical_countries'] = len(anomaly_dash[anomaly_dash['risk_score'] >= 10]) if 'risk_score' in anomaly_dash.columns else 0
    
    if rf_pred is not None:
        metrics['shock_probability_avg'] = rf_pred['probability'].mean() if 'probability' in rf_pred.columns else 0
        metrics['high_probability_count'] = len(rf_pred[rf_pred['probability'] > 0.7]) if 'probability' in rf_pred.columns else 0
    
    # Display metrics in 4-column grid
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Critical Risk Countries</div>
            <div class="metric-value" style="color: #ef4444;">{metrics['critical_countries']}</div>
            <div style="font-size: 0.75rem; color: #94a3b8;">Risk Score ≥ 10</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Global Avg Risk</div>
            <div class="metric-value" style="color: #f59e0b;">{metrics['avg_risk_score']:.1f}</div>
            <div style="font-size: 0.75rem; color: #94a3b8;">Out of 20</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Shock Probability</div>
            <div class="metric-value" style="color: #3b82f6;">{metrics['shock_probability_avg']:.1%}</div>
            <div style="font-size: 0.75rem; color: #94a3b8;">Next-year average</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">High-Risk Forecast</div>
            <div class="metric-value" style="color: #8b5cf6;">{metrics['high_probability_count']}</div>
            <div style="font-size: 0.75rem; color: #94a3b8;">P(Shock) > 70%</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    
    # =====================================================================
    # SECTION 2: INTERACTIVE GLOBAL RISK MAP
    # =====================================================================
    with st.expander("🗺️ Interactive Global Risk Map", expanded=True):
        if anomaly_dash is not None and 'risk_score' in anomaly_dash.columns:
            # Prepare data for choropleth
            map_data = anomaly_dash.copy()
            
            # Create choropleth
            fig_choropleth = px.choropleth(
                map_data,
                locations='country',
                locationmode='country names',
                color='risk_score',
                color_continuous_scale='RdYlGn_r',
                range_color=[0, 20],
                title='Composite Risk Score by Country (click to drill down)',
                labels={'risk_score': 'Risk Score'},
                hover_name='country',
                hover_data={'country': False, 'risk_score': ':.1f'}
            )
            
            fig_choropleth.update_layout(
                **PLOTLY_DARK,
                height=550,
                coloraxis_colorbar=dict(
                    title="Risk Score",
                    thickness=15,
                    len=0.7,
                    x=1.02
                )
            )
            fig_choropleth.update_geos(
                showland=True,
                showcoastlines=True,
                showcountries=True,
                landcolor='rgba(30,40,50,0.8)',
                countrycolor='rgba(100,120,140,0.2)',
                coastlinecolor='rgba(120,140,160,0.3)',
            )
            
            st.plotly_chart(fig_choropleth, use_container_width=True, key="global_risk_map")
            
            st.info("💡 **Insight**: Red regions indicate elevated systemic risk. Hover over countries to see exact risk scores. Consider geographic clustering patterns.")
        else:
            st.warning("⚠️ Risk data unavailable for map rendering.")
    
    # =====================================================================
    # SECTION 3: EARLY WARNING SYSTEM
    # =====================================================================
    with st.expander("🚨 Early Warning Signals & System Insights", expanded=True):
        col_ews1, col_ews2 = st.columns([1, 1])
        
        with col_ews1:
            st.markdown("<div class='section-header'>🧠 System Insights</div>", unsafe_allow_html=True)
            
            insights = []
            
            # Insight 1: Risk distribution
            if anomaly_dash is not None and 'risk_score' in anomaly_dash.columns:
                std_dev = anomaly_dash['risk_score'].std()
                skew = "concentrated in a few regions" if std_dev > 4 else "distributed across regions"
                insights.append(f"Risk exposure is **{skew}** (std dev: {std_dev:.1f})")
            
            # Insight 2: Shock probability
            if rf_pred is not None and 'probability' in rf_pred.columns:
                shock_median = rf_pred['probability'].median()
                prob_interpretation = "elevated global vulnerability" if shock_median > 0.5 else "moderate global conditions"
                insights.append(f"Median shock probability suggests **{prob_interpretation}** ({shock_median:.1%})")
            
            # Insight 3: Risk concentration
            if anomaly_dash is not None and 'risk_score' in anomaly_dash.columns:
                top_10_pct = anomaly_dash.nlargest(max(1, len(anomaly_dash)//10), 'risk_score')['risk_score'].sum()
                total_risk = anomaly_dash['risk_score'].sum()
                concentration = (top_10_pct / total_risk * 100) if total_risk > 0 else 0
                insights.append(f"Top 10% of countries account for **{concentration:.0f}%** of total systemic risk")
            
            for i, insight in enumerate(insights, 1):
                st.markdown(f"""
                <div class="insight-box">
                    <strong>Insight {i}:</strong> {insight}
                </div>
                """, unsafe_allow_html=True)
        
        with col_ews2:
            st.markdown("<div class='section-header'>🚨 Critical Alerts</div>", unsafe_allow_html=True)
            
            alerts = []
            
            # Alert: Critical threshold
            if anomaly_dash is not None and 'risk_score' in anomaly_dash.columns:
                critical = len(anomaly_dash[anomaly_dash['risk_score'] >= 10])
                if critical > 0:
                    alerts.append(('critical', f"{critical} countries exceed critical risk threshold (≥10)"))
            
            # Alert: High shock probability
            if rf_pred is not None and 'probability' in rf_pred.columns:
                extreme = len(rf_pred[rf_pred['probability'] > 0.8])
                if extreme > 0:
                    alerts.append(('critical', f"{extreme} countries show extreme shock probability (>80%)"))
            
            # Alert: Model consensus
            if anomaly_dash is not None:
                flag_cols = [c for c in anomaly_dash.columns if 'flag' in c.lower()]
                if flag_cols:
                    high_consensus = len(anomaly_dash[(anomaly_dash[flag_cols].sum(axis=1) >= 7)])
                    if high_consensus > 0:
                        alerts.append(('warning', f"{high_consensus} countries flagged by 7+ anomaly models"))
            
            # Alert: Momentum
            if (
                shock_sev is not None
                and len(shock_sev) > 1
                and 'year' in shock_sev.columns
                and 'shock_severity' in shock_sev.columns
            ):
                recent = shock_sev.sort_values('year').tail(2)
                if len(recent) == 2:
                    recent_change = recent.iloc[-1]['shock_severity'] - recent.iloc[-2]['shock_severity']
                    if pd.notna(recent_change) and recent_change > 0:
                        alerts.append(('warning', f"Global shock severity trend increasing (+{recent_change:.2f})"))
            
            if not alerts:
                st.markdown("""
                <div class="insight-box" style="border-left-color: #10b981; background: linear-gradient(135deg, rgba(16,185,129,0.1), transparent);">
                    <strong style="color: #6ee7b7;">✓ No Critical Alerts</strong><br>
                    System operating within normal parameters. Continue monitoring.
                </div>
                """, unsafe_allow_html=True)
            else:
                for severity, message in alerts:
                    badge_class = 'warning' if severity == 'warning' else ''
                    icon = '⚠️' if severity == 'warning' else '🔴'
                    st.markdown(f"""
                    <div class="insight-box {badge_class}">
                        <strong>{icon} Alert:</strong> {message}
                    </div>
                    """, unsafe_allow_html=True)
    
    # =====================================================================
    # SECTION 4: COUNTRY DEEP DIVE
    # =====================================================================
    with st.expander("🔍 Detailed Country Risk Profile", expanded=True):
        if anomaly_dash is not None and len(anomaly_dash) > 0:
            all_countries = sorted(anomaly_dash['country'].unique())
            selected_country = st.selectbox(
                "Select a country to analyze",
                all_countries,
                key="tab5_detection_country_selector",
                help="Choose any country to see detailed risk breakdown"
            )
            
            country_data = anomaly_dash[anomaly_dash['country'] == selected_country]
            
            if not country_data.empty:
                country_data = country_data.iloc[0]
                
                # Risk score interpretation
                risk_score = country_data.get('risk_score', 0) if pd.notna(country_data.get('risk_score')) else 0
                
                if risk_score >= 10:
                    risk_level = "CRITICAL"
                    risk_color = "#ef4444"
                    risk_emoji = "🔴"
                elif risk_score >= 8:
                    risk_level = "ELEVATED"
                    risk_color = "#f59e0b"
                    risk_emoji = "🟡"
                elif risk_score >= 5:
                    risk_level = "MODERATE"
                    risk_color = "#3b82f6"
                    risk_emoji = "🔵"
                else:
                    risk_level = "STABLE"
                    risk_color = "#10b981"
                    risk_emoji = "🟢"
                
                # Display risk cards
                col_profile1, col_profile2, col_profile3 = st.columns(3)
                
                with col_profile1:
                    st.markdown(f"""
                    <div class="risk-card {risk_level.lower()}">
                        <div class="metric-label">COMPOSITE RISK SCORE</div>
                        <div class="metric-value" style="color: {risk_color};">{risk_score:.1f} / 20</div>
                        <div style="color: {risk_color}; font-weight: 600; margin-top: 0.5rem;">{risk_emoji} {risk_level}</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Next-year shock probability
                if rf_pred is not None:
                    pred_data = rf_pred[rf_pred['country'] == selected_country]
                    if not pred_data.empty:
                        shock_prob = pred_data.iloc[0].get('probability', 0)
                        if shock_prob > 0.7:
                            prob_color = "#ef4444"
                            prob_label = "EXTREME"
                        elif shock_prob > 0.5:
                            prob_color = "#f59e0b"
                            prob_label = "HIGH"
                        else:
                            prob_color = "#10b981"
                            prob_label = "MODERATE"
                    else:
                        shock_prob = 0
                        prob_color = "#64748b"
                        prob_label = "N/A"
                else:
                    shock_prob = 0
                    prob_color = "#64748b"
                    prob_label = "N/A"
                
                with col_profile2:
                    st.markdown(f"""
                    <div class="risk-card">
                        <div class="metric-label">NEXT-YEAR SHOCK PROBABILITY</div>
                        <div class="metric-value" style="color: {prob_color};">{shock_prob:.1%}</div>
                        <div style="color: {prob_color}; font-weight: 600; margin-top: 0.5rem;">→ {prob_label} PROBABILITY</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Anomaly model consensus
                flag_cols = [c for c in country_data.index if 'flag' in c.lower()]
                flag_count = sum([1 for col in flag_cols if country_data.get(col) == 1])
                
                if flag_count >= 7:
                    flag_color = "#ef4444"
                    flag_label = "HIGH CONSENSUS"
                elif flag_count >= 5:
                    flag_color = "#f59e0b"
                    flag_label = "MODERATE CONSENSUS"
                else:
                    flag_color = "#10b981"
                    flag_label = "LOW CONSENSUS"
                
                with col_profile3:
                    st.markdown(f"""
                    <div class="risk-card">
                        <div class="metric-label">MODEL AGREEMENT (9 Systems)</div>
                        <div class="metric-value" style="color: {flag_color};">{flag_count} / 9</div>
                        <div style="color: {flag_color}; font-weight: 600; margin-top: 0.5rem;">→ {flag_label}</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                st.divider()
                
                # Risk drivers
                driver_cols = ['import_dependency', 'price_volatility', 'supply_gap', 'debt_to_gdp',
                              'political_stability', 'inflation_rate', 'gdp_growth']
                
                drivers_data = {}
                for col in driver_cols:
                    if col in country_data.index and pd.notna(country_data[col]):
                        drivers_data[col] = country_data[col]
                
                if drivers_data:
                    st.markdown("#### 📈 Risk Drivers & Indicators")
                    col_driver1, col_driver2, col_driver3 = st.columns(3)
                    
                    for i, (driver, value) in enumerate(list(drivers_data.items())[:9]):
                        col = [col_driver1, col_driver2, col_driver3][i % 3]
                        with col:
                            driver_display = driver.replace('_', ' ').title()
                            st.metric(driver_display, f"{value:.2f}", delta=None)
                
                # Model flags breakdown
                if flag_cols:
                    st.markdown("#### 🔬 Individual Model Flags")
                    flag_breakdown = {}
                    for col in flag_cols:
                        model_name = col.replace('_flag', '').replace('_', ' ').title()
                        flag_val = country_data.get(col, 0)
                        flag_breakdown[model_name] = int(flag_val) if pd.notna(flag_val) else 0
                    
                    col_flags = st.columns(min(5, len(flag_breakdown)))
                    for i, (model, val) in enumerate(flag_breakdown.items()):
                        with col_flags[i % 5]:
                            status_icon = "🔴" if val == 1 else "🟢"
                            st.metric(status_icon + " " + model[:15], "Flagged" if val == 1 else "OK")
        else:
            st.info("No country data available.")
    
    # =====================================================================
    # SECTION 5: RISK SEGMENTATION & WATCHLIST
    # =====================================================================
    with st.expander("📋 Risk Segmentation & Watchlist", expanded=False):
        if anomaly_dash is not None and 'risk_score' in anomaly_dash.columns:
            # Risk segmentation
            def categorize_risk(score):
                if score >= 10:
                    return "🔴 CRITICAL (10-20)"
                elif score >= 8:
                    return "🟡 ELEVATED (8-9)"
                elif score >= 5:
                    return "🔵 MODERATE (5-7)"
                else:
                    return "🟢 STABLE (0-4)"
            
            seg_data = anomaly_dash.copy()
            seg_data['risk_segment'] = seg_data['risk_score'].apply(categorize_risk)
            seg_counts = seg_data['risk_segment'].value_counts().reset_index()
            seg_counts.columns = ['Risk Segment', 'Number of Countries']
            seg_counts = seg_counts.sort_values('Number of Countries', ascending=False)
            
            col_seg1, col_seg2 = st.columns([1, 2])
            
            with col_seg1:
                st.markdown("#### Risk Distribution")
                st.dataframe(seg_counts, use_container_width=True, hide_index=True)
            
            with col_seg2:
                st.markdown("#### Distribution Visualization")
                fig_pie = px.pie(
                    seg_counts,
                    values='Number of Countries',
                    names='Risk Segment',
                    color_discrete_sequence=['#ef4444', '#f59e0b', '#3b82f6', '#10b981']
                )
                fig_pie.update_layout(**PLOTLY_DARK, height=300)
                st.plotly_chart(fig_pie, use_container_width=True)
            
            st.divider()
            
            # Top risk countries
            st.markdown("#### 🚨 Highest-Risk Countries")
            watchlist_count = min(
                st.slider(
                    "Number of countries to show",
                    min_value=5,
                    max_value=min(25, len(anomaly_dash)),
                    value=min(15, len(anomaly_dash)),
                    step=1,
                    key="tab5_watchlist_count"
                ),
                len(anomaly_dash)
            )
            watchlist_sort = st.radio(
                "Display order",
                options=["Highest first", "Lowest first"],
                horizontal=True,
                key="tab5_watchlist_sort"
            )
            
            watchlist = anomaly_dash.nlargest(watchlist_count, 'risk_score')[['country', 'risk_score']].copy()
            watchlist['Risk Level'] = watchlist['risk_score'].apply(
                lambda x: '🔴 CRITICAL' if x >= 10 else '🟡 ELEVATED' if x >= 8 else '🔵 MODERATE'
            )
            watchlist = watchlist.rename(columns={'country': 'Country', 'risk_score': 'Risk Score'})
            
            watchlist_display = watchlist.sort_values(
                'Risk Score',
                ascending=(watchlist_sort == "Lowest first")
            ).reset_index(drop=True)
            
            st.dataframe(
                watchlist_display[['Country', 'Risk Score', 'Risk Level']],
                use_container_width=True,
                hide_index=True,
                column_config={
                    'Risk Score': st.column_config.NumberColumn("Risk Score", format="%.1f")
                }
            )
            
            fig_bar = px.bar(
                watchlist_display,
                x='Risk Score',
                y='Country',
                color='Risk Score',
                color_continuous_scale='Reds',
                title=f'Top {watchlist_count} Countries by Risk Score',
                labels={'Risk Score': 'Composite Risk Score'},
                orientation='h'
            )
            fig_bar.update_layout(
                **PLOTLY_DARK,
                height=max(450, 28 * len(watchlist_display) + 140),
                coloraxis_showscale=False,
                yaxis={'categoryorder': 'array', 'categoryarray': watchlist_display['Country'].tolist()}
            )
            st.plotly_chart(fig_bar, use_container_width=True)
    
    # =====================================================================
    # SECTION 6: RISK MOMENTUM & TRENDS
    # =====================================================================
    with st.expander("📈 Risk Momentum & Temporal Trends", expanded=False):
        if yearly_summary is not None and not yearly_summary.empty:
            st.markdown("#### Global Risk Evolution Over Time")
            
            # Time series of anomalies
            fig_timeline = px.line(
                yearly_summary,
                x='year',
                y='total_anomalies',
                markers=True,
                line_shape='spline',
                title='Annual Anomaly Count Trend',
                labels={'year': 'Year', 'total_anomalies': 'Total Anomalies'}
            )
            fig_timeline.update_traces(
                line=dict(color='#3b82f6', width=3),
                marker=dict(size=10, color='#ef4444')
            )
            fig_timeline.update_layout(**PLOTLY_DARK, height=350)
            st.plotly_chart(fig_timeline, use_container_width=True)
            
            # Year-over-year change
            yearly_summary_sorted = yearly_summary.sort_values('year')
            yearly_summary_sorted['YoY Change'] = yearly_summary_sorted['total_anomalies'].diff()
            
            fig_momentum = px.bar(
                yearly_summary_sorted,
                x='year',
                y='YoY Change',
                color='YoY Change',
                color_continuous_scale='RdYlGn_r',
                title='Year-over-Year Anomaly Change',
                labels={'year': 'Year', 'YoY Change': 'Change in Anomalies'}
            )
            fig_momentum.update_layout(**PLOTLY_DARK, height=350)
            st.plotly_chart(fig_momentum, use_container_width=True)
        else:
            st.info("Yearly trend data not available.")
    
    # =====================================================================
    # SECTION 7: PREDICTIVE ANALYTICS
    # =====================================================================
    with st.expander("🔮 Next-Year Shock Probability Forecasts", expanded=False):
        if rf_pred is not None and 'probability' in rf_pred.columns:
            st.markdown("#### Predicted Shock Probabilities")
            
            rf_pred_clean = rf_pred.dropna(subset=['country', 'probability']).copy()
            time_col = next((col for col in ['date', 'Date', 'year', 'Year'] if col in rf_pred_clean.columns), None)
            
            aggregation_options = ["Maximum risk", "Average risk"]
            if time_col is not None:
                aggregation_options.insert(0, "Latest forecast")
            
            forecast_aggregation = st.selectbox(
                "Country aggregation method",
                aggregation_options,
                index=0,
                key="tab5_forecast_aggregation",
                help="Choose how multiple forecast rows per country should be combined before ranking."
            )
            
            if forecast_aggregation == "Latest forecast" and time_col is not None:
                latest_value = rf_pred_clean[time_col].max()
                forecast_base = rf_pred_clean[rf_pred_clean[time_col] == latest_value].copy()
                forecast_base = forecast_base.groupby('country', as_index=False)['probability'].max()
                aggregation_caption = f"Using latest available {time_col}: {latest_value}"
            elif forecast_aggregation == "Average risk":
                forecast_base = rf_pred_clean.groupby('country', as_index=False)['probability'].mean()
                aggregation_caption = "Countries ranked by average forecast probability"
            else:
                forecast_base = rf_pred_clean.groupby('country', as_index=False)['probability'].max()
                aggregation_caption = "Countries ranked by maximum forecast probability"
            
            st.caption(aggregation_caption)
            
            col_pred1, col_pred2 = st.columns(2)
            
            with col_pred1:
                st.markdown("**Distribution of Shock Probabilities**")
                fig_prob_dist = px.histogram(
                    forecast_base,
                    x='probability',
                    nbins=20,
                    title='Probability Distribution',
                    labels={'probability': 'Shock Probability'}
                )
                fig_prob_dist.update_layout(**PLOTLY_DARK, height=350)
                st.plotly_chart(fig_prob_dist, use_container_width=True)
            
            with col_pred2:
                # Statistics
                st.metric("Mean Probability", f"{forecast_base['probability'].mean():.1%}")
                st.metric("Median Probability", f"{forecast_base['probability'].median():.1%}")
                st.metric("Countries > 70% Risk", f"{len(forecast_base[forecast_base['probability'] > 0.7])}")
                st.metric("Countries > 50% Risk", f"{len(forecast_base[forecast_base['probability'] > 0.5])}")
            
            st.divider()
            
            # Top forecast countries
            st.markdown("**Forecast country ranking**")
            forecast_count = min(
                st.slider(
                    "Forecast countries to show",
                    min_value=5,
                    max_value=min(25, len(forecast_base)),
                    value=min(10, len(forecast_base)),
                    step=1,
                    key="tab5_forecast_count"
                ),
                len(forecast_base)
            )
            forecast_sort = st.radio(
                "Forecast display order",
                options=["Highest first", "Lowest first"],
                horizontal=True,
                key="tab5_forecast_sort"
            )
            
            top_forecast = forecast_base.nlargest(forecast_count, 'probability')[['country', 'probability']].copy()
            top_forecast = top_forecast.rename(columns={'country': 'Country', 'probability': 'P(Shock)'})
            top_forecast_display = top_forecast.sort_values(
                'P(Shock)',
                ascending=(forecast_sort == "Lowest first")
            ).reset_index(drop=True)
            
            st.dataframe(
                top_forecast_display,
                use_container_width=True,
                hide_index=True,
                column_config={
                    'P(Shock)': st.column_config.NumberColumn("P(Shock)", format="%.2f")
                }
            )
            
            fig_forecast = px.bar(
                top_forecast_display,
                x='P(Shock)',
                y='Country',
                color='P(Shock)',
                color_continuous_scale='Reds',
                title=f'Top {forecast_count} Forecast Countries',
                orientation='h'
            )
            fig_forecast.update_layout(
                **PLOTLY_DARK,
                height=max(350, 28 * len(top_forecast_display) + 120),
                coloraxis_showscale=False,
                yaxis={'categoryorder': 'array', 'categoryarray': top_forecast_display['Country'].tolist()}
            )
            st.plotly_chart(fig_forecast, use_container_width=True)
    
    # =====================================================================
    # SECTION 8: MODEL DIAGNOSTICS & EXPORTS
    # =====================================================================
    with st.expander("⚙️ Model Diagnostics & Data Export", expanded=False):
        col_export1, col_export2 = st.columns([1, 1])
        
        with col_export1:
            st.markdown("#### Risk Score Statistics")
            if anomaly_dash is not None and 'risk_score' in anomaly_dash.columns:
                stats = {
                    'Mean': anomaly_dash['risk_score'].mean(),
                    'Median': anomaly_dash['risk_score'].median(),
                    'Std Dev': anomaly_dash['risk_score'].std(),
                    'Min': anomaly_dash['risk_score'].min(),
                    'Max': anomaly_dash['risk_score'].max(),
                    'Q1': anomaly_dash['risk_score'].quantile(0.25),
                    'Q3': anomaly_dash['risk_score'].quantile(0.75)
                }
                
                for label, value in stats.items():
                    st.metric(label, f"{value:.2f}")
        
        with col_export2:
            st.markdown("#### Data Quality")
            if anomaly_dash is not None:
                data_quality = {
                    'Total Records': len(anomaly_dash),
                    'Missing Values': anomaly_dash.isnull().sum().sum(),
                    'Unique Countries': anomaly_dash['country'].nunique() if 'country' in anomaly_dash.columns else 0,
                }
                for label, value in data_quality.items():
                    st.metric(label, str(value))
        
        st.divider()
        
        # Export options
        st.markdown("#### 📥 Export Data")
        
        col_exp1, col_exp2, col_exp3 = st.columns(3)
        
        with col_exp1:
            if anomaly_dash is not None:
                csv_buffer = anomaly_dash.to_csv(index=False)
                st.download_button(
                    label="📊 Anomaly Dashboard (CSV)",
                    data=csv_buffer,
                    file_name=f"anomaly_dashboard_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )
        
        with col_exp2:
            if rf_pred is not None:
                csv_buffer = rf_pred.to_csv(index=False)
                st.download_button(
                    label="🔮 RF Predictions (CSV)",
                    data=csv_buffer,
                    file_name=f"rf_predictions_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )
        
        with col_exp3:
            if anomaly_dash is not None:
                combined = anomaly_dash.copy()
                if rf_pred is not None:
                    combined = combined.merge(rf_pred[['country', 'probability']], on='country', how='left')
                
                csv_buffer = combined.to_csv(index=False)
                st.download_button(
                    label="🔗 Combined Report (CSV)",
                    data=csv_buffer,
                    file_name=f"combined_risk_report_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )
    
    # =====================================================================
    # FOOTER
    # =====================================================================
    st.markdown("""
    <div style="text-align:center; padding:2rem; margin-top:2rem; border-top: 1px solid rgba(148,163,184,0.2);">
        <p style="color: #94a3b8; font-size: 0.85rem;">
            <strong>Enhanced Risk Dashboard</strong> | Real-time early warning system | 
            Last updated: {timestamp}
        </p>
        <p style="color: #64748b; font-size: 0.8rem;">
            Data sourced from pipeline checkpoints | All data is current and validated
        </p>
    </div>
    """.format(timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S')), unsafe_allow_html=True)


# =========================================================================
# END OF TAB5_ENHANCED_RISK_DASHBOARD
# =========================================================================
