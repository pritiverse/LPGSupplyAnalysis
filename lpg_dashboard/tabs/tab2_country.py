# tabs/tab2_country.py
"""Country Deep-Dive Tab - Complete Implementation with Proper UI Spacing"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from utils.helpers import (
    section_tag, apply_plotly_dark, severity_badge,
    get_structural_anomaly_score
)
from utils.constants import (
    SEVERITY_COLORS, SEVERITY_ORDER, PLOTLY_DARK,
    get_region, normalize_country
)


def render(data, df_sev, filters):
    """Render Country Deep-Dive tab"""
    
    # Initialize all variables that might be used later
    flagged_count = 0
    cp_count = 0
    structural_score = 0
    shock_score = 0
    selected_country = None
    region_name = ""
    

    
    section_tag("COUNTRY INTELLIGENCE PROFILE")
    
    # Get all countries from filtered data
    all_countries = sorted(df_sev['country'].tolist()) if not df_sev.empty else []
    
    if not all_countries:
        st.warning("No country data available")
        return
    
    # Country selector - Better layout
    col_select1, col_select2 = st.columns([2, 1])
    
    with col_select1:
        selected_country = st.selectbox(
            "🔍 Select a country to analyze",
            all_countries,
            key="country_selector",
            help="Choose a country to view its detailed supply chain risk profile"
        )
    
    with col_select2:
        years = list(range(1990, 2025))
        year_range = st.slider(
            "📅 Year range",
            min_value=1990,
            max_value=2024,
            value=(2000, 2024),
            step=5,
            help="Select time period for detailed analysis"
        )
    
    if not selected_country:
        st.info("Please select a country to view its profile")
        return
    
    # Get country data
    country_row = df_sev[df_sev['country'] == selected_country]
    if country_row.empty:
        st.error(f"No data available for {selected_country}")
        return
    
    row = country_row.iloc[0]
    severity = row.get('ShockSeverity', 'Normal')
    shock_score = row.get('ShockScore', 0)
    region_name = row.get('region', get_region(selected_country))
    
    # =========================================================
    # COUNTRY HEADER WITH BADGES
    # =========================================================
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, #0d1b2a, #112233); 
                padding: 20px; 
                border-radius: 12px; 
                margin-bottom: 24px;
                border: 1px solid #1e3a5f;">
        <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap;">
            <div>
                <h2 style="margin: 0; color: #e8f4ff;">{selected_country}</h2>
                <p style="margin: 8px 0 0 0; color: #8bbdd9;">
                    🌍 {region_name} &nbsp;|&nbsp;
                    ⚠️ Severity: {severity_badge(severity)} &nbsp;|&nbsp;
                    📊 Shock Score: {shock_score:.2f}
                </p>
            </div>
            <div style="text-align: right;">
                <span style="font-size: 42px;">
                    {'🔴' if severity == 'Strong' else '🟠' if severity == 'Moderate' else '🟡' if severity == 'Weak' else '🟢'}
                </span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # =========================================================
    # KPI ROW - 5 columns
    # =========================================================
    global_rank = int((df_sev['ShockScore'] > shock_score).sum()) + 1 if 'ShockScore' in df_sev.columns else 0
    
    cp_df = data.get('change_points')
    cp_count = 0
    if cp_df is not None:
        cp_col = 'year_change_point' if 'year_change_point' in cp_df.columns else 'year'
        cp_count = len(cp_df[cp_df['country'] == selected_country])
    
    # Calculate structural score directly from anomaly_dashboard
    structural_score = 0
    anom_df_for_score = data.get('anomaly_dashboard')
    if anom_df_for_score is not None:
        country_anom_row = anom_df_for_score[anom_df_for_score['country'] == selected_country]
        if not country_anom_row.empty:
            structural_flags = ['lof_flag', 'ocsvm_flag', 'if_flag', 'cp_flag']
            score = 0
            for flag in structural_flags:
                if flag in country_anom_row.columns:
                    score += country_anom_row.iloc[0].get(flag, 0)
            structural_score = score
    
    net_df = data.get('network_metrics')
    pagerank = 0
    degree_centrality = 0
    if net_df is not None and not net_df.empty:
        country_net = net_df[net_df['country'] == selected_country]
        if not country_net.empty:
            pagerank = country_net.iloc[0].get('pagerank', 0)
            degree_centrality = country_net.iloc[0].get('degree_centrality', 0)
    
    # Display 5 KPIs in a row
    col_k1, col_k2, col_k3, col_k4, col_k5 = st.columns(5)
    col_k1.metric("🌍 Global Rank", f"#{global_rank} / {len(df_sev)}")
    col_k2.metric("⚡ Structural Breaks", cp_count, help="Number of times supply regime fundamentally changed")
    col_k3.metric("🔬 Structural Anomaly Score", f"{int(structural_score)}/4", help="0-4: How many structural methods flagged this country")
    col_k4.metric("🔗 PageRank", f"{pagerank:.4f}" if pagerank else "—", help="Network influence score (higher = more central)")
    col_k5.metric("📊 Degree Centrality", f"{degree_centrality:.3f}" if degree_centrality else "—", help="How connected in trade network")
    
    st.markdown("<br>", unsafe_allow_html=True)  # Add spacing
    
    # =========================================================
    # TIME SERIES & ANOMALIES SECTION
    # =========================================================
    st.subheader("📈 Supply Trend & Anomaly Detection")
    st.caption("Historical total supply (ktoe) with anomaly flags from multiple detection methods")
    
    # Create two columns for main content
    col_left, col_right = st.columns([2, 1])
    
    with col_left:
        # Get time series data
        ts_df = data.get('features_timeseries')
        years_full = list(range(year_range[0], year_range[1] + 1))
        
        if ts_df is not None and 'country' in ts_df.columns:
            country_ts = ts_df[ts_df['country'] == selected_country].sort_values('year')
            country_ts = country_ts[country_ts['year'].between(year_range[0], year_range[1])]
            
            if not country_ts.empty:
                ts_years = country_ts['year'].tolist()
                ts_supply = country_ts['total_supply'].tolist() if 'total_supply' in country_ts.columns else None
            else:
                ts_years = years_full
                ts_supply = None
        else:
            ts_years = years_full
            ts_supply = None
        
        if ts_supply is None:
            st.warning(f"No time series supply data available for {selected_country} in the selected range.")
            fig_ts = go.Figure()
            fig_ts.update_layout(
                height=450,
                xaxis_title='Year',
                yaxis_title='Supply (ktoe)',
                annotations=[dict(text="No Data Available", showarrow=False, font=dict(size=16))]
            )
            apply_plotly_dark(fig_ts)
            st.plotly_chart(fig_ts, use_container_width=True)
        else:
            # Create time series plot with improved spacing
            fig_ts = go.Figure()
            
            # Main supply line with area fill
            fig_ts.add_trace(go.Scatter(
                x=ts_years,
                y=ts_supply,
                mode='lines',
                name='Total Supply',
                line=dict(color='#5bc0de', width=2.5),
                fill='tozeroy',
                fillcolor='rgba(91,192,222,0.15)',
                hovertemplate='Year: %{x}<br>Supply: %{y:,.0f} ktoe<extra></extra>'
            ))
            
            # Add moving average
            if len(ts_supply) > 3:
                ma_window = min(5, len(ts_supply) // 2)
                ma = pd.Series(ts_supply).rolling(window=ma_window, center=True).mean()
                fig_ts.add_trace(go.Scatter(
                    x=ts_years,
                    y=ma,
                    mode='lines',
                    name=f'{ma_window}-Year Trend',
                    line=dict(color='#f0ad4e', width=2, dash='dash'),
                    hovertemplate='Trend: %{y:,.0f} ktoe<extra></extra>'
                ))
            
            # LSTM anomalies
            lstm_df = data.get('anomaly_lstm')
            if lstm_df is not None and filters.get('show_lstm', True):
                lstm_country = lstm_df[
                    (lstm_df['country'] == selected_country) & 
                    (lstm_df.get('lstm_anomaly_flag', pd.Series()) == 1)
                ]
                for _, row in lstm_country.iterrows():
                    year = row.get('year', 0)
                    if year and year_range[0] <= year <= year_range[1]:
                        fig_ts.add_vline(
                            x=year, line_dash='dot', line_color='#ea5455', line_width=2,
                            annotation_text='LSTM', annotation_position='top',
                            annotation_font_size=9
                        )
            
            # Prophet anomalies
            prophet_df = data.get('anomaly_prophet')
            if prophet_df is not None and filters.get('show_prophet', True):
                prophet_country = prophet_df[
                    (prophet_df['country'] == selected_country) & 
                    (prophet_df.get('prophet_anomaly_flag', pd.Series()) == 1)
                ]
                for _, row in prophet_country.iterrows():
                    year = row.get('year', 0)
                    if year and year_range[0] <= year <= year_range[1]:
                        fig_ts.add_vline(
                            x=year, line_dash='dot', line_color='#f0ad4e', line_width=2,
                            annotation_text='Prophet', annotation_position='bottom',
                            annotation_font_size=9
                        )
            
            # EWMA anomalies as markers
            ewma_df = data.get('anomaly_ewma')
            if ewma_df is not None and filters.get('show_ewma', True):
                ewma_country = ewma_df[
                    (ewma_df['country'] == selected_country) & 
                    (ewma_df.get('ewma_anomaly_flag', pd.Series()) == 1)
                ]
                for _, row in ewma_country.iterrows():
                    year = row.get('year', 0)
                    if year and year_range[0] <= year <= year_range[1]:
                        fig_ts.add_trace(go.Scatter(
                            x=[year], y=[max(ts_supply) * 1.05],
                            mode='markers', 
                            marker=dict(symbol='diamond', size=10, color='#28c76f'),
                            name='EWMA', 
                            showlegend=False,
                            hovertemplate=f'EWMA Anomaly: {year}<extra></extra>'
                        ))
            
            # Change points (Regime Shifts)
            if cp_df is not None:
                cp_country = cp_df[cp_df['country'] == selected_country]
                cp_col = 'year_change_point' if 'year_change_point' in cp_df.columns else 'year'
                for _, row in cp_country.iterrows():
                    cp_year = row.get(cp_col, 0)
                    if cp_year and year_range[0] <= cp_year <= year_range[1]:
                        fig_ts.add_vline(
                            x=cp_year, line_dash='dash', line_color='#ea5455', line_width=2.5,
                            annotation_text='Regime Shift', annotation_position='bottom',
                            annotation_font_size=10, annotation_font_color='#ea5455'
                        )
            
            # Update layout with proper x-axis spacing
            fig_ts.update_layout(
                xaxis_title='Year',
                yaxis_title='Supply (ktoe)',
                height=450,
                hovermode='x unified',
                legend=dict(
                    orientation='h', 
                    yanchor='bottom', 
                    y=1.02, 
                    xanchor='center', 
                    x=0.5,
                    font=dict(size=10)
                ),
                xaxis=dict(
                    tickmode='linear',
                    tick0=year_range[0],
                    dtick=5,  # Tick every 5 years
                    tickangle=0,
                    gridcolor='#1e3a5f',
                    showgrid=True,
                    showline=True,
                    linecolor='#1e3a5f'
                ),
                yaxis=dict(
                    gridcolor='#1e3a5f',
                    showgrid=True,
                    showline=True,
                    linecolor='#1e3a5f'
                ),
                margin=dict(l=50, r=50, t=50, b=50)
            )
            apply_plotly_dark(fig_ts)
            st.plotly_chart(fig_ts, use_container_width=True)
            
            # Supply statistics with proper spacing
            if len(ts_supply) >= 2:
                start_val = ts_supply[0]
                end_val = ts_supply[-1]
                pct_change = ((end_val - start_val) / start_val) * 100 if start_val > 0 else 0
                max_val = max(ts_supply)
                min_val = min(ts_supply)
                volatility = np.std(ts_supply)
                
                # Use columns for statistics
                col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
                col_stat1.metric(f"Supply ({year_range[0]})", f"{start_val:,.0f}")
                col_stat2.metric(f"Supply ({year_range[1]})", f"{end_val:,.0f}", delta=f"{pct_change:+.1f}%")
                col_stat3.metric("Peak Supply", f"{max_val:,.0f}")
                col_stat4.metric("Volatility (Std)", f"{volatility:,.0f}")
                
                st.markdown("---")
                
                # Insight box with proper spacing
                st.markdown("""
                <div style="background: linear-gradient(135deg, #0d2137, #0a2744); 
                            padding: 16px; 
                            border-radius: 10px; 
                            margin: 16px 0;
                            border-left: 4px solid #5bc0de;">
                    <b>💡 Understanding This Chart</b><br><br>
                    • <b>Blue area:</b> Total supply trajectory over time<br>
                    • <b>Orange dashed line:</b> Rolling average showing underlying trend<br>
                    • <b>Colored markers/lines:</b> Anomalies detected by different methods<br>
                    • <b>Red dashed vertical lines:</b> Regime shifts (fundamental supply changes)<br><br>
                    <b>Why it matters:</b> Countries with frequent anomalies or regime shifts have less predictable supply chains and require more active risk management.
                </div>
                """, unsafe_allow_html=True)
    
    # =========================================================
    # RIGHT COLUMN - ANOMALY FLAGS & RADAR CHART
    # =========================================================
    with col_right:
        # Anomaly Detection Summary
        st.markdown("### 🚨 Anomaly Detection")
        
        anom_df = data.get('anomaly_dashboard')
        flagged_count = 0  # Initialize here
        total_methods = 10
        
        if anom_df is not None and not anom_df.empty:
            country_anom = anom_df[anom_df['country'] == selected_country]
            
            if not country_anom.empty:
                a_row = country_anom.iloc[0]
                
                flag_mappings = {
                    '📊 IQR': a_row.get('iqr_flag', 0),
                    '📈 Moving Avg': a_row.get('ma_flag', 0),
                    '🎯 Percentile': a_row.get('pct_flag', 0),
                    '🔍 LOF': a_row.get('lof_flag', 0),
                    '🔄 OCSVM': a_row.get('ocsvm_flag', 0),
                    '🗂️ DBSCAN': a_row.get('db_flag', 0),
                    '🧠 LSTM': a_row.get('lstm_flag', 0),
                    '📅 Prophet': a_row.get('prophet_flag', 0),
                    '📉 EWMA': a_row.get('ewma_flag', 0),
                    '🌲 Isolation F': a_row.get('if_flag', 0),
                }
                
                flagged_count = sum(flag_mappings.values())
                
                # Risk score card
                risk_color = "#ea5455" if flagged_count >= 5 else "#f0ad4e" if flagged_count >= 3 else "#28c76f"
                risk_text = "HIGH RISK" if flagged_count >= 5 else "MODERATE RISK" if flagged_count >= 3 else "LOW RISK"
                
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, #1a1a2e, #16213e); 
                            padding: 20px; 
                            border-radius: 12px; 
                            text-align: center;
                            margin-bottom: 20px;">
                    <div style="font-size: 48px; font-weight: bold; color: {risk_color};">
                        {flagged_count} / {total_methods}
                    </div>
                    <div style="font-size: 14px; color: #8bbdd9; margin: 8px 0;">Detection Methods Flagged</div>
                    <div style="margin-top: 12px;">
                        <span style="background: {risk_color}; padding: 6px 16px; border-radius: 20px; font-size: 14px; font-weight: bold;">
                            {risk_text}
                        </span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Flags table
                flags_df = pd.DataFrame({
                    'Method': list(flag_mappings.keys()),
                    'Status': ['⚠️ Flagged' if v else '✅ Clear' for v in flag_mappings.values()],
                    'Type': ['Statistical', 'Statistical', 'Statistical', 'ML', 'ML', 'ML', 'Deep Learning', 'Time-Series', 'Time-Series', 'ML']
                })
                
                st.dataframe(
                    flags_df,
                    use_container_width=True,
                    hide_index=True,
                    height=300
                )
                
                st.markdown("""
                <div style="background: #0d1b2a; padding: 12px; border-radius: 8px; margin: 12px 0;">
                    <b>📌 Method Types:</b><br>
                    • <b>Statistical</b> = Distribution outliers<br>
                    • <b>ML</b> = Structural anomalies<br>
                    • <b>Deep Learning</b> = Complex patterns<br>
                    • <b>Time-Series</b> = Trend deviations
                </div>
                """, unsafe_allow_html=True)
            else:
                st.info("No anomaly data for this country")
                # Set default flagged_count
                flagged_count = 0
        else:
            st.info("Anomaly dashboard not available")
            flagged_count = 0
    
     # =========================================================
    # FULL WIDTH SECTION - Regional Comparison (ALL COUNTRIES)
    # =========================================================
    st.markdown("---")
    st.subheader(f"🌍 Regional Comparison: {region_name}")
    
    # Initialize region_df
    region_df = pd.DataFrame()
    
    if 'region' in df_sev.columns and 'ShockScore' in df_sev.columns:
        region_df = df_sev[df_sev['region'] == region_name].copy()
        
        if not region_df.empty:
            st.caption(f"Comparing {selected_country} with all {len(region_df)} countries in {region_name}")
            
            # Sort by ShockScore for better visualization
            region_df = region_df.sort_values('ShockScore', ascending=False).reset_index(drop=True)
            
            # Add rank column
            region_df['rank'] = region_df.index + 1
            
            # Color coding: red for selected country, gradient for others
            colors = []
            for _, row in region_df.iterrows():
                if row['country'] == selected_country:
                    colors.append('#ea5455')  # Red for selected
                else:
                    # Gradient based on ShockScore
                    score = row['ShockScore']
                    if score > 4:
                        colors.append('#ff6b6b')
                    elif score > 3:
                        colors.append('#f0ad4e')
                    elif score > 2:
                        colors.append('#ffd700')
                    else:
                        colors.append('#28c76f')
            
            # Create horizontal bar chart showing ALL countries
            fig_reg = go.Figure(go.Bar(
                x=region_df['ShockScore'],
                y=region_df['country'],
                orientation='h',
                marker_color=colors,
                text=region_df['ShockSeverity'],
                textposition='outside',
                textfont=dict(size=10),
                hovertemplate='<b>%{y}</b><br>' +
                              'Shock Score: %{x:.2f}<br>' +
                              'Severity: %{text}<br>' +
                              'Rank: #' + region_df['rank'].astype(str) + '<extra></extra>'
            ))
            
            # Add a vertical line for regional average
            region_avg = region_df['ShockScore'].mean()
            fig_reg.add_vline(
                x=region_avg, 
                line_dash='dash', 
                line_color='#5bc0de',
                line_width=2,
                annotation_text=f'Regional Avg: {region_avg:.2f}',
                annotation_position='top right',
                annotation_font_size=11
            )
            
            fig_reg.update_layout(
                xaxis_title='Shock Score (Higher = More Risk)',
                yaxis_title='',
                height=max(500, len(region_df) * 28),
                showlegend=False,
                xaxis=dict(
                    gridcolor='#1e3a5f',
                    range=[0, max(region_df['ShockScore']) * 1.1],
                    title_font=dict(size=12)
                ),
                yaxis=dict(
                    tickfont=dict(size=11),
                    automargin=True
                ),
                margin=dict(l=10, r=80, t=40, b=30)
            )
            apply_plotly_dark(fig_reg)
            st.plotly_chart(fig_reg, use_container_width=True)
            
            # Regional statistics table - show ALL countries
            st.markdown(f"### 📊 Complete Ranking for {region_name}")
            
            # Create display dataframe
            display_df = region_df[['rank', 'country', 'ShockScore', 'ShockSeverity']].copy()
            
            st.dataframe(
                display_df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    'rank': st.column_config.NumberColumn('Rank', width='small'),
                    'country': st.column_config.TextColumn('Country', width='medium'),
                    'ShockScore': st.column_config.NumberColumn('Shock Score', format='%.2f', width='small'),
                    'ShockSeverity': st.column_config.TextColumn('Severity', width='small')
                }
            )
            
            # Statistics cards
            col_s1, col_s2, col_s3, col_s4, col_s5 = st.columns(5)
            
            with col_s1:
                st.metric("Total Countries", len(region_df))
            with col_s2:
                st.metric("Regional Average", f"{region_avg:.2f}")
            with col_s3:
                st.metric(f"{selected_country}'s Score", f"{shock_score:.2f}")
            with col_s4:
                diff = shock_score - region_avg
                st.metric("vs Regional Avg", f"{diff:+.2f}", 
                         delta_color="inverse" if diff > 0 else "normal")
            with col_s5:
                percentile = (region_df['ShockScore'] < shock_score).sum() / len(region_df) * 100
                st.metric("Percentile", f"{percentile:.0f}th")
            
            # Insight based on ranking
            country_rank = region_df[region_df['country'] == selected_country]['rank'].values[0]
            total_countries = len(region_df)
            
            if country_rank <= total_countries * 0.25:
                insight = f"⚠️ {selected_country} is in the top 25% most at-risk countries in {region_name}, ranking #{country_rank} out of {total_countries}."
            elif country_rank <= total_countries * 0.5:
                insight = f"📊 {selected_country} is in the upper-middle risk tier in {region_name}, ranking #{country_rank} out of {total_countries}."
            elif country_rank <= total_countries * 0.75:
                insight = f"✅ {selected_country} is in the lower-middle risk tier in {region_name}, ranking #{country_rank} out of {total_countries}."
            else:
                insight = f"🟢 {selected_country} is in the lowest risk quartile in {region_name}, ranking #{country_rank} out of {total_countries}."
            
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #0d2137, #0a2744); 
                        padding: 14px; 
                        border-radius: 10px; 
                        margin-top: 16px;
                        border-left: 4px solid #5bc0de;">
                <b>📌 Regional Insight:</b> {insight}
            </div>
            """, unsafe_allow_html=True)
        else:
            st.warning(f"No data available for region: {region_name}")
    else:
        st.warning("Region or ShockScore data not available")
    
    # =========================================================
    # Shock Propagation Section
    # =========================================================
    st.markdown("---")
    st.subheader("🔄 Shock Propagation Network")
    
    prop_df = data.get('shock_propagation')
    if prop_df is not None and not prop_df.empty:
        src_col = 'source_country' if 'source_country' in prop_df.columns else 'source'
        tgt_col = 'target_country' if 'target_country' in prop_df.columns else 'target'
        
        if src_col in prop_df.columns and tgt_col in prop_df.columns:
            col_p1, col_p2 = st.columns(2)
            
            with col_p1:
                as_source = prop_df[prop_df[src_col] == selected_country]
                st.metric("📤 Propagated TO others", len(as_source))
                
                if len(as_source) > 0:
                    st.caption("Top recipients:")
                    partners = as_source[tgt_col].value_counts().head(5)
                    for country, count in partners.items():
                        st.write(f"• {country}: {count} events")
            
            with col_p2:
                as_target = prop_df[prop_df[tgt_col] == selected_country]
                st.metric("📥 Received FROM others", len(as_target))
                
                if len(as_target) > 0:
                    st.caption("Top sources:")
                    sources = as_target[src_col].value_counts().head(5)
                    for country, count in sources.items():
                        st.write(f"• {country}: {count} events")
        else:
            st.info("Propagation data structure not recognized")
    else:
        st.info("No propagation data available")
    
# Replace the Executive Summary section with this:

    # =========================================================
    # EXECUTIVE SUMMARY - CLEAN NATIVE STREAMLIT
    # =========================================================
    st.markdown("---")
    st.subheader("📋 Executive Summary & Recommendations")
    
    # Build recommendation
    if shock_score > 3:
        recommendation = "🔴 **Immediate Action Required:** Implement enhanced supply chain monitoring and diversification strategies immediately. Consider strategic reserves and alternative supplier relationships."
        rec_type = "error"
    elif shock_score > 2:
        recommendation = "🟡 **Active Monitoring Needed:** Maintain active monitoring with quarterly risk reviews. Develop contingency plans for supply disruptions."
        rec_type = "warning"
    else:
        recommendation = "🟢 **Standard Monitoring:** Continue regular monitoring with annual comprehensive review. Current risk levels are manageable."
        rec_type = "success"
    
    # Display findings
    col_a, col_b = st.columns([3, 2])
    
    with col_a:
        st.markdown(f"**🎯 Key Findings for {selected_country}**")
        
        # Risk level
        if shock_score <= 2:
            st.success("✅ **Risk Level:** Low Risk - Standard monitoring sufficient")
        elif shock_score <= 3:
            st.warning("⚠️ **Risk Level:** Moderate Risk - Regular review suggested")
        else:
            st.error("🔴 **Risk Level:** High Risk - Immediate attention required")
        
        # Regime shifts
        if cp_count > 0:
            st.info(f"📊 **Regime Shifts:** {cp_count} change(s) detected - Supply pattern has changed")
        
        # Structural anomaly
        if structural_score >= 3:
            st.warning("🔬 **Structural Anomaly:** Fundamentally different from global norms")
        
        # Multi-method consensus
        if flagged_count >= 5:
            st.error("🚨 **Multi-Method Consensus:** High confidence in risk assessment")
    
    with col_b:
        st.markdown("**📊 Risk Indicators**")
        st.metric("Shock Score", f"{shock_score:.2f}", 
                 delta="High Risk" if shock_score > 3 else "Moderate" if shock_score > 2 else "Low")
        st.metric("Global Rank", f"#{global_rank}")
        st.metric("Methods Flagging", f"{flagged_count}/10")
    
    # Recommendation box
    if rec_type == "error":
        st.error(recommendation)
    elif rec_type == "warning":
        st.warning(recommendation)
    else:
        st.success(recommendation)