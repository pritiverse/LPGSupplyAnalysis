import streamlit as st
import pandas as pd
import plotly.express as px
from utils.constants import PLOTLY_DARK

def render(data, df_sev, filters):
    """Renders the Crisis & Event Analysis tab."""

    # =========================================================================
    # TAB 3 — CRISIS & EVENT ANALYSIS (INTERACTIVE ANALYSIS REPORT)
    # =========================================================================
    # Custom CSS for report-style interface
    st.markdown("""
    <style>
    /* Report styling */
    .report-header {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
        padding: 2rem;
        border-radius: 20px;
        margin-bottom: 2rem;
        text-align: center;
        border-bottom: 3px solid #ff6b6b;
    }
    .report-title {
        font-size: 2.2rem;
        font-weight: 700;
        margin: 0;
        background: linear-gradient(90deg, #ff6b6b, #ffd93d);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .report-subtitle {
        font-size: 1rem;
        opacity: 0.8;
        margin-top: 0.5rem;
    }
    .finding-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        padding: 1.5rem;
        border-radius: 16px;
        margin: 1rem 0;
        border-left: 4px solid #ff6b6b;
    }
    .finding-title {
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 2px;
        color: #ff6b6b;
        margin-bottom: 0.5rem;
    }
    .finding-text {
        font-size: 1.1rem;
        line-height: 1.5;
    }
    .insight-highlight {
        background: rgba(255,107,107,0.15);
        padding: 0.2rem 0.4rem;
        border-radius: 6px;
        font-weight: 600;
        color: #ff6b6b;
    }
    .checkpoint-card {
        background: #1e1e2f;
        border-radius: 12px;
        padding: 1rem;
        margin-bottom: 0.8rem;
        border: 1px solid #2a2a3e;
        transition: all 0.2s;
    }
    .checkpoint-card:hover {
        border-color: #ff6b6b;
        transform: translateX(4px);
    }
    .checkpoint-name {
        font-family: monospace;
        font-size: 0.85rem;
        color: #ffd93d;
    }
    .stat-badge {
        display: inline-block;
        background: #2a2a3e;
        padding: 0.2rem 0.8rem;
        border-radius: 20px;
        font-size: 0.75rem;
    }
    hr {
        margin: 1.5rem 0;
        border-color: #2a2a3e;
    }
    </style>
    """, unsafe_allow_html=True)

    # =====================================================================
    # REPORT HEADER
    # =====================================================================
    st.markdown("""
    <div class="report-header">
        <div class="report-title">⚠️ SHOCK VALIDATION ARTIFACTS</div>
        <div class="report-subtitle">Statistical Event Alignment | 1990–2024 | 10 Global Events | 150+ Countries</div>
        <div style="margin-top: 1rem;">
            <span class="stat-badge">📊 Phase: Pre / During / Post</span>
            <span class="stat-badge">📈 Metric: ShockZ → Impact Score</span>
            <span class="stat-badge">🎯 Ranking: Weighted Impact</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # =====================================================================
    # FINDING 1: Event Impact Ranking (shock_event_weighted.parquet)
    # =====================================================================
    st.markdown("### 🔍 FINDING 1: Global Events by Disruption Magnitude")
    st.caption("`shock_event_weighted.parquet` — Ranks global events by aggregate supply system disruption")
    
    df_ev = data.get('shock_event_weighted')
    if df_ev is not None and len(df_ev) > 0 and 'shock_increase' in df_ev.columns and 'event' in df_ev.columns:
        df_ev_sorted = df_ev.sort_values('shock_increase', ascending=True)
        
        col_f1_left, col_f1_right = st.columns([2.5, 1.5])
        
        with col_f1_left:
            fig_ev = px.bar(
                df_ev_sorted, x='shock_increase', y='event', orientation='h',
                color='shock_increase', color_continuous_scale='Reds',
                labels={'shock_increase': 'Net Global Shock Increase', 'event': ''},
                text='shock_increase') # type: ignore
            fig_ev.update_layout(
                coloraxis_showscale=False,
                **PLOTLY_DARK,
                height=400,
                title="<b>Event Impact Ranking</b>",
                xaxis_title="Δ Shock Magnitude (During - Pre)",
                yaxis_title="")
            fig_ev.update_traces(texttemplate='%{text:.1f}', textposition='outside', textfont=dict(color='white'))
            st.plotly_chart(fig_ev, use_container_width=True)
            
        with col_f1_right:
            top_event = df_ev.loc[df_ev['shock_increase'].idxmax()]
            
            # Robust second largest - avoids nlargest index issues
            second_event = None
            if len(df_ev) >= 2:
                sorted_idx = df_ev.sort_values('shock_increase', ascending=False).index
                second_idx = sorted_idx[1]
                second_event = df_ev.loc[second_idx]
            elif len(df_ev) == 1:
                second_event = df_ev.iloc[0]
            
            gfc_event = None
            financial_mask = df_ev['event'].str.contains('Financial', case=False)
            if financial_mask.any():
                gfc_event = df_ev[financial_mask].iloc[0]
                
            # Safe rendering
            if top_event is not None:
                second_name = second_event['event'] if second_event is not None else "No second event"
                gfc_note = ""
                if gfc_event is not None and top_event['shock_increase'] > gfc_event['shock_increase']:
                    gfc_note = "Both exceeded the impact of the 2008 Financial Crisis."
                
                st.markdown(f"""
                <div class="finding-card">
                    <div class="finding-title">📌 ANALYST CONCLUSION</div>
                    <div class="finding-text">
                        The <span class="insight-highlight">{top_event['event']}</span> caused the highest net supply regime shift 
                        (<strong>+{top_event['shock_increase']:.1f}</strong> points), followed by <span class="insight-highlight">{second_name}</span>.
                        {gfc_note}
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Show pre/during comparison for top event
                if 'pre_shock' in top_event and 'during_shock' in top_event:
                    pre_val = top_event['pre_shock']
                    during_val = top_event['during_shock']
                    pct_change = ((during_val - pre_val) / pre_val) * 100 if pre_val > 0 else 0
                    st.metric(
                        label=f"{top_event['event']} — Shock Intensity Change",
                        value=f"{during_val:.1f} → {pre_val:.1f}",
                        delta=f"{pct_change:+.0f}%"
                    )
            else:
                st.warning("No valid shock events found in data.")
    else:
        st.info("`shock_event_weighted.parquet` not available. Run event alignment pipeline first.") # type: ignore

    st.divider()

    # =====================================================================
    # FINDING 2: Yearly Anomaly Timeline
    # =====================================================================
    st.markdown("### 🔍 FINDING 2: Temporal Distribution of Global Shocks")
    st.caption("`combined_yearly_anomalies` — Annual anomaly detection across all countries and models")

    cya = data.get('combined_yearly_anomalies')
    if cya is not None and 'total_anomalies' in cya.columns and 'year' in cya.columns:
        cya_sorted = cya.sort_values('year')
        
        # Calculate statistics for annotation
        mean_anomaly = cya_sorted['total_anomalies'].mean()
        std_anomaly = cya_sorted['total_anomalies'].std()
        peak_year = cya_sorted.loc[cya_sorted['total_anomalies'].idxmax(), 'year']
        peak_value = cya_sorted['total_anomalies'].max()
        
        col_f2_left, col_f2_right = st.columns([2.5, 1.5])
        
        with col_f2_left:
            fig_cya = px.area(
                cya_sorted, x='year', y='total_anomalies',
                color_discrete_sequence=['#ff6b6b'],
                labels={'total_anomalies': 'Total Anomalies', 'year': 'Year'},
                title="<b>Global Anomaly Timeline</b>") # type: ignore
            fig_cya.add_hline(y=mean_anomaly, line_dash="dash", line_color="#ffd93d", 
                              annotation_text=f"Mean: {mean_anomaly:.1f}")
            fig_cya.add_hrect(y0=mean_anomaly + std_anomaly, y1=mean_anomaly + 2*std_anomaly, 
                              line_width=0, fillcolor="red", opacity=0.1)
            fig_cya.update_layout(**PLOTLY_DARK, height=350, xaxis=dict(dtick=4))
            st.plotly_chart(fig_cya, use_container_width=True)
        
        with col_f2_right:
            st.markdown(f"""
            <div class="finding-card">
                <div class="finding-title">📈 STATISTICAL INSIGHT</div>
                <div class="finding-text">
                    • <strong>Peak disruption year:</strong> {peak_year} ({peak_value} anomalies)<br>
                    • <strong>Mean annual anomalies:</strong> {mean_anomaly:.1f}<br>
                    • <strong>Years above +1σ threshold:</strong> {len(cya_sorted[cya_sorted['total_anomalies'] > mean_anomaly + std_anomaly])}<br><br>
                    Spikes correlate with major crises but also reveal periods of widespread, low-grade instability.
                </div>
            </div>
            """, unsafe_allow_html=True)
    else: # type: ignore
        st.info("Yearly anomaly data not available.")

    st.divider()

    # =====================================================================
    # FINDING 3: Country-Level Event Vulnerability (shock_country_impact.parquet)
    # =====================================================================
    st.markdown("### 🔍 FINDING 3: Granular Country-Event Vulnerability")
    st.caption("`shock_country_impact.parquet` — Average Shock Z-scores (pre vs during) per country per event")

    sci = data.get('shock_country_impact')
    if sci is not None and 'country' in sci.columns and 'event' in sci.columns:
        # Interactive selector
        col_f3a, col_f3b = st.columns([1, 2])
        
        with col_f3a:
            analysis_mode = st.radio(
                "Analysis Mode",
                ["🔍 By Event", "🌍 By Country"],
                horizontal=True,
                help="Explore vulnerability by selecting an event or a country"
            )
            
            if analysis_mode == "🔍 By Event":
                events_list = sorted(sci['event'].unique())
                selected_event = st.selectbox("Select Global Event", events_list, 
                                              format_func=lambda x: f"⚡ {x}")
                filtered_df = sci[sci['event'] == selected_event].sort_values('weighted_impact', ascending=False)
                metric_col = 'weighted_impact'
                metric_label = "Weighted Impact Score"
                title_prefix = f"Most Vulnerable Countries: {selected_event}"
            else:
                countries_list = sorted(sci['country'].unique())
                selected_country = st.selectbox("Select Country", countries_list,
                                                format_func=lambda x: f"📍 {x}")
                filtered_df = sci[sci['country'] == selected_country].sort_values('weighted_impact', ascending=False)
                metric_col = 'weighted_impact'
                metric_label = "Weighted Impact Score"
                title_prefix = f"Events Impacting: {selected_country}"
            
            # Show top 10
            n_display = st.slider("Number of results", 5, 20, 10)
            display_df = filtered_df.head(n_display) # type: ignore
            
            # Display as compact table
            st.dataframe(
                display_df[['country' if analysis_mode == "🔍 By Event" else 'event', 
                           'impact_score', metric_col]].reset_index(drop=True),
                use_container_width=True,
                column_config={
                    'impact_score': st.column_config.NumberColumn("Raw Δ", format="%.2f"),
                    metric_col: st.column_config.NumberColumn("Weighted", format="%.2f")
                }
            )
        
        with col_f3b: # type: ignore
            if len(display_df) > 0:
                # Create comparison chart
                fig_vuln = px.bar(
                    display_df.sort_values(metric_col, ascending=True),
                    x=metric_col, 
                    y='country' if analysis_mode == "🔍 By Event" else 'event',
                    orientation='h',
                    color=metric_col,
                    color_continuous_scale='Oranges',
                    labels={metric_col: metric_label, 
                            'country' if analysis_mode == "🔍 By Event" else 'event': ''}, # type: ignore
                    title=title_prefix)
                fig_vuln.update_layout(
                    height=400,
                    coloraxis_showscale=False,
                    **PLOTLY_DARK)
                fig_vuln.update_traces(texttemplate='%{x:.2f}', textposition='outside')
                st.plotly_chart(fig_vuln, use_container_width=True)
                
                # Highlight finding
                top_result = display_df.iloc[0]
                if analysis_mode == "🔍 By Event": # type: ignore
                    st.info(f"💡 **Finding:** {top_result['country']} shows the highest weighted impact ({top_result[metric_col]:.2f}) during {selected_event}, indicating exceptional vulnerability to this specific disruption.")
                else: # type: ignore
                    st.info(f"💡 **Finding:** {selected_country} is most vulnerable to {top_result['event']} (Δ = {top_result[metric_col]:.2f}), suggesting event-specific supply chain sensitivities.")
    else:
        st.info("`shock_country_impact.parquet` not available.")

    st.divider()

    # =====================================================================
    # FINDING 4: Top Countries Per Event (shock_top_countries_advanced.parquet)
    # =====================================================================
    st.markdown("### 🔍 FINDING 4: Top 5 Most Impacted Nations Per Event")
    st.caption("`shock_top_countries_advanced.parquet` — Filtered ranking of heavily impacted nations")

    top_c = data.get('shock_top_countries_advanced')
    if top_c is not None and 'event' in top_c.columns and 'rank' in top_c.columns:
        events_list = sorted(top_c['event'].unique())
        
        # Multi-event comparison
        selected_events_compare = st.multiselect(
            "Compare events (select 2-4 for ranking visualization)",
            events_list,
            default=events_list[:2] if len(events_list) >= 2 else events_list, # type: ignore
            max_selections=4
        )
        
        if len(selected_events_compare) >= 1:
            compare_df = top_c[top_c['event'].isin(selected_events_compare)]
            
            # Pivot for comparison
            pivot_df = compare_df.pivot_table(
                index='country', columns='event', values='weighted_impact', aggfunc='first'
            ).fillna(0)
            
            # Get top countries by average impact
            pivot_df['avg_impact'] = pivot_df.mean(axis=1)
            pivot_df = pivot_df.sort_values('avg_impact', ascending=False).head(15)
            
            fig_compare = px.bar( # type: ignore
                pivot_df.reset_index().melt(id_vars='country', value_vars=selected_events_compare,
                                            var_name='event', value_name='weighted_impact'),
                x='weighted_impact', y='country', color='event', orientation='h',
                barmode='group',
                labels={'weighted_impact': 'Weighted Impact', 'country': '', 'event': 'Global Event'},
                title="<b>Cross-Event Country Vulnerability Comparison</b>")
            fig_compare.update_layout(height=500, **PLOTLY_DARK, legend=dict(orientation='h', yanchor='bottom', y=1.02))
            st.plotly_chart(fig_compare, use_container_width=True)
            
            # Insight box
            st.markdown("""
            <div class="finding-card">
                <div class="finding-title">🎯 KEY OBSERVATION</div>
                <div class="finding-text">
                    This comparison reveals which countries appear vulnerable across <strong>multiple</strong> crisis types 
                    (systemic vulnerability) vs. those that are only sensitive to <strong>specific</strong> events 
                    (event-driven vulnerability). Countries appearing in the top ranks across multiple events represent 
                    structurally fragile supply chains.
                </div>
            </div>
            """, unsafe_allow_html=True)
    else: # type: ignore
        st.info("`shock_top_countries_advanced.parquet` not available.")

    st.divider()

    # =====================================================================
    # FINDING 5: Global Vulnerability Ranking (shock_global_country_rank.parquet)
    # =====================================================================
    st.markdown("### 🔍 FINDING 5: Perpetually Vulnerable Supply Chains")
    st.caption("`shock_global_country_rank.parquet` — Aggregated sensitivity across all 10 global events")

    gcr = data.get('shock_global_country_rank')
    if gcr is not None and 'total_impact' in gcr.columns and 'country' in gcr.columns:
        col_f5_left, col_f5_right = st.columns([2, 1])
        
        with col_f5_left:
            n_display = st.slider("Ranking depth", 10, 40, 20, key="global_rank_depth") # type: ignore
            gcr_sorted = gcr.sort_values('total_impact', ascending=False).head(n_display)
            
            fig_gcr = px.bar(
                gcr_sorted.sort_values('total_impact', ascending=True),
                x='total_impact', y='country', orientation='h',
                color='total_impact', color_continuous_scale='Reds',
                labels={'total_impact': 'Cumulative Weighted Impact', 'country': ''},
                title="<b>Most Consistently Vulnerable Nations</b>") # type: ignore
            fig_gcr.update_layout(
                height=max(400, n_display * 25),
                coloraxis_showscale=False,
                **PLOTLY_DARK)
            fig_gcr.update_traces(texttemplate='%{x:.1f}', textposition='outside')
            st.plotly_chart(fig_gcr, use_container_width=True)
        
        with col_f5_right: # type: ignore
            top5 = gcr_sorted.head(5)['country'].tolist()
            top1 = top5[0]
            bottom5 = gcr_sorted.tail(5)['country'].tolist()
            
            st.markdown(f"""
            <div class="finding-card">
                <div class="finding-title">🏆 VULNERABILITY RANKING</div>
                <div class="finding-text">
                    <strong>Top 5 Most Vulnerable:</strong><br>
                    {', '.join(f'<span class="insight-highlight">{c}</span>' for c in top5)}<br><br>
                    <strong>Bottom 5 (Most Resilient):</strong><br>
                    {', '.join(bottom5)}<br><br>
                    <span style="font-size:0.9rem;">Countries with the highest cumulative weighted sensitivity are consistently rattled by external global factors, regardless of crisis type.</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
    else: # type: ignore
        st.info("`shock_global_country_rank.parquet` not available.")

    st.divider()

    # =====================================================================
    # FINDING 6: Shock Propagation Network
    # =====================================================================
    st.markdown("### 🔍 FINDING 6: Shock Propagation Dynamics")
    st.caption("`shock_propagation` — How supply disruptions cascade between countries")

    prop = data.get('shock_propagation')
    if prop is not None:
        src_col = 'source_country' if 'source_country' in prop.columns else 'source'
        evt_col = 'propagation_event' if 'propagation_event' in prop.columns else 'events_count'
        
        if src_col in prop.columns and evt_col in prop.columns:
            col_f6_left, col_f6_right = st.columns([2, 1])
            
            with col_f6_left:
                n_prop = st.slider("Top propagators", 5, 20, 10, key="prop_depth") # type: ignore
                propagators = prop.groupby(src_col)[evt_col].sum().sort_values(ascending=False).head(n_prop)
                
                fig_prop = px.bar(
                    x=propagators.values, y=propagators.index, orientation='h',
                    color=propagators.values, color_continuous_scale='Blues',
                    labels={'x': 'Propagation Events', 'y': ''},
                    title="<b>Top Shock Transmitters</b>") # type: ignore
                fig_prop.update_layout(
                    height=max(350, n_prop * 28),
                    coloraxis_showscale=False,
                    **PLOTLY_DARK)
                fig_prop.update_traces(texttemplate='%{x}', textposition='outside')
                st.plotly_chart(fig_prop, use_container_width=True)
            
            with col_f6_right: # type: ignore
                top_transmitter = propagators.index[0] if len(propagators) > 0 else "N/A"
                st.markdown(f"""
                <div class="finding-card">
                    <div class="finding-title">🔗 NETWORK INSIGHT</div>
                    <div class="finding-text">
                        <strong>{top_transmitter}</strong> is the most frequent shock propagator in the network.<br><br>
                        Countries with high betweenness centrality (e.g., Morocco, Tunisia) sit on shortest paths between many country pairs, acting as transmission hubs for supply disruptions.
                    </div>
                </div>
                """, unsafe_allow_html=True)
    else:
        st.info("Shock propagation data not available.") # type: ignore

    st.divider()

    # =====================================================================
    # FINDING 7: Temporal Cluster Instability
    # =====================================================================
    st.markdown("### 🔍 FINDING 7: Behavioral Archetype Shifts Over Time")
    st.caption("`temporal_clusters` — Countries that changed behavior most frequently (1990–2024)")

    tc = data.get('temporal_clusters')
    if tc is not None and 'cluster_switch' in tc.columns:
        col_f7_left, col_f7_right = st.columns([2, 1])
        
        with col_f7_left:
            n_tc = st.slider("Countries to analyze", 5, 25, 12, key="tc_depth") # type: ignore
            switches = tc.groupby('country')['cluster_switch'].sum().sort_values(ascending=False).head(n_tc)
            
            fig_sw = px.bar(
                x=switches.values, y=switches.index, orientation='h',
                color=switches.values, color_continuous_scale='Purples',
                labels={'x': 'Number of Cluster Switches (1990–2024)', 'y': ''},
                title="<b>Most Structurally Unstable Countries</b>") # type: ignore
            fig_sw.update_layout(
                height=max(400, n_tc * 28),
                coloraxis_showscale=False,
                **PLOTLY_DARK)
            fig_sw.update_traces(texttemplate='%{x}', textposition='outside')
            st.plotly_chart(fig_sw, use_container_width=True)
        
        with col_f7_right: # type: ignore
            total_switches = switches.sum()
            top_unstable = switches.index[0] if len(switches) > 0 else "N/A"
            st.markdown(f"""
            <div class="finding-card">
                <div class="finding-title">🔄 BEHAVIORAL DYNAMICS</div>
                <div class="finding-text">
                    • <strong>Total cluster switches:</strong> {total_switches}<br>
                    • <strong>Most unstable:</strong> {top_unstable}<br><br>
                    These countries frequently move between archetypes (e.g., Stable → Shock-Prone → Recovering) over time, indicating dynamic supply chain adaptation or structural fragility.
                </div>
            </div>
            """, unsafe_allow_html=True)
    else: # type: ignore
        st.info("Temporal cluster data not available.")

    # =====================================================================
    # RAW DATA ACCESS
    # =====================================================================
    with st.expander("📁 Raw Data Access — Download Checkpoint Artifacts", expanded=False):
        st.markdown("Access the underlying data frames for custom analysis.")
        
        artifacts = {
            'shock_validation_advanced.parquet': data.get('shock_validation_advanced'),
            'shock_country_impact.parquet': data.get('shock_country_impact'),
            'shock_top_countries_advanced.parquet': data.get('shock_top_countries_advanced'),
            'shock_global_country_rank.parquet': data.get('shock_global_country_rank'),
            'shock_event_weighted.parquet': data.get('shock_event_weighted')
        }
        
        for name, df in artifacts.items():
            if df is not None:
                with st.container():
                    col_a, col_b = st.columns([3, 1]) # type: ignore
                    with col_a:
                        st.markdown(f"**`{name}`** — {len(df):,} rows × {len(df.columns)} cols")
                    with col_b:
                        csv = df.to_csv(index=False).encode('utf-8') # type: ignore
                        st.download_button("📥 Download", csv, f"{name}.csv", key=name) # type: ignore
            
        st.caption("💡 These artifacts represent the complete output of the Statistical Event Alignment pipeline. Each file corresponds to a specific checkpoint documented in the technical specification.") # type: ignore