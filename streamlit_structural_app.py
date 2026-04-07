#!/usr/bin/env python3
"""
Structural Anomaly Detection Dashboard – Ultimate Edition
Detects country-level outliers using Isolation Forest, LOF, and One‑Class SVM.
Interactive | 172+ countries | Full model explanations | Toggleable graphs | Export ready
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import glob
import os
from pathlib import Path
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler, RobustScaler

# ------------------------------------------------------------
# Page configuration
# ------------------------------------------------------------
st.set_page_config(
    page_title="Structural Anomaly Detection",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------------------------------------------------
# Custom CSS for better aesthetics & info boxes
# ------------------------------------------------------------
st.markdown("""
<style>
    .reportview-container .main .block-container { padding-top: 2rem; }
    .stMetric > label { font-size: 1rem; color: #cbd5e1; }
    .stMetric > div > div { font-size: 2.2rem !important; font-weight: bold; }
    h1 { color: #facc15; }
    h2, h3, h4 { color: #e2e8f0; }
    .stTabs [data-baseweb="tab-list"] { gap: 2rem; }
    .stTabs [data-baseweb="tab"] { font-size: 1rem; background-color: #1e293b; border-radius: 0.5rem; padding: 0.5rem 1rem; }
    .stTabs [aria-selected="true"] { background-color: #facc15; color: #0f172a; }
    .info-box {
        background: linear-gradient(135deg, #1e293b, #0f172a);
        border-left: 6px solid #facc15;
        border-radius: 0.75rem;
        padding: 1rem 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.3);
    }
    .info-box h4 { margin-top: 0; color: #facc15; }
    .info-box p { margin-bottom: 0; }
    .model-card {
        background: #1e293b;
        border-radius: 1rem;
        padding: 1rem 1.2rem;
        margin: 0.5rem 0;
        border-top: 4px solid;
    }
    .model-card h4 { margin: 0 0 0.5rem 0; }
    .model-card p { font-size: 0.9rem; }
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------
def model_card(color, title, subtitle, description, insight):
    """Styled markdown box for model explanation."""
    color_map = {
        "teal": "#14b8a6", "rose": "#f43f5e", "violet": "#a78bfa",
        "amber": "#fbbf24", "blue": "#3b82f6"
    }
    border_color = color_map.get(color, "#facc15")
    html = f"""
    <div class="model-card" style="border-top-color: {border_color};">
        <h4>{title}</h4>
        <p><em>{subtitle}</em></p>
        <p>{description}</p>
        <p><strong>💡 Key insight:</strong> {insight}</p>
    </div>
    """
    return html

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

# ------------------------------------------------------------
# Sidebar controls
# ------------------------------------------------------------
st.sidebar.header("🎛️ Model Configuration")

contamination = st.sidebar.slider("Anomaly rate (%)", 5, 25, 10, 5,
                                  help="Expected proportion of outliers in the data.")
n_neighbors = st.sidebar.slider("LOF n_neighbors", 10, 50, 20, 5,
                                help="Number of neighbours for Local Outlier Factor.")
nu_svm = st.sidebar.slider("OCSVM nu", 0.01, 0.3, 0.05, 0.01,
                           help="Upper bound on the fraction of outliers (One‑Class SVM).")
svm_kernel = st.sidebar.selectbox("OCSVM kernel", ["rbf", "linear", "poly", "sigmoid"], index=0)
scaler_type = st.sidebar.selectbox("Scaling method", ["standard", "robust"], index=0)

if st.sidebar.button("🔄 Recompute models", type="primary", use_container_width=True):
    st.cache_data.clear()
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.info(
    "**Data source:** `pipeline_checkpoints/country_pca_features`\n\n"
    "**Models:** Isolation Forest, Local Outlier Factor, One‑Class SVM\n\n"
    "**Composite score** = average of all three anomaly scores."
)

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

# ---- Isolation Forest tab ----
with tab_if:
    st.markdown(model_card(
        "teal", "🌲 Isolation Forest – Ensemble Tree Method",
        "sklearn.ensemble.IsolationForest | contamination = {:.1%}".format(contamination/100),
        "Isolation Forest detects anomalies by randomly selecting features and split values. "
        "Anomalous countries — those with unusual supply profiles — require fewer splits to isolate "
        "(shorter average path length). Applied to the full multi‑dimensional feature vector "
        "(import dependency, volatility, growth rate, trade imbalance), it identifies countries "
        "that are structurally unlike the global majority.",
        f"Identified <strong>{df_results['IF_anomaly'].sum()}</strong> countries as structural outliers – "
        "nations with supply profiles so distinct they sit far from the majority in multi‑dimensional feature space."
    ), unsafe_allow_html=True)
    
    col1, col2 = st.columns([3, 2])
    with col1:
        make_model_graph(df_results, 'IF_score', 'country', 'IF_anomaly', 'Isolation Forest', 'Reds_r')
    with col2:
        st.dataframe(
            df_results[['country', 'IF_anomaly', 'IF_score']]
            .sort_values('IF_score', ascending=False)
            .rename(columns={'country':'Country', 'IF_anomaly':'Anomaly', 'IF_score':'Score'}),
            use_container_width=True, height=500, hide_index=True
        )

# ---- LOF tab ----
with tab_lof:
    st.markdown(model_card(
        "rose", "📍 Local Outlier Factor – Density‑Based Method",
        f"sklearn.neighbors.LocalOutlierFactor | n_neighbors = {n_neighbors} | contamination = {contamination/100:.1%}",
        "LOF compares each country's local density to that of its k‑nearest neighbours. "
        "A country is an outlier if it sits in a sparser region relative to its neighbours (LOF score >> 1). "
        "Unlike global methods, LOF detects countries that are anomalous *within their peer group* — "
        "revealing emerging‑market outliers that look fine globally but behave unusually among similar economies.",
        f"Flagged <strong>{df_results['LOF_anomaly'].sum()}</strong> countries as local density outliers – "
        "nations that behave unusually even within their own regional or economic peer group."
    ), unsafe_allow_html=True)
    
    col1, col2 = st.columns([3, 2])
    with col1:
        make_model_graph(df_results, 'LOF_score', 'country', 'LOF_anomaly', 'Local Outlier Factor', 'Oranges_r')
    with col2:
        st.dataframe(
            df_results[['country', 'LOF_anomaly', 'LOF_score']]
            .sort_values('LOF_score', ascending=False)
            .rename(columns={'country':'Country', 'LOF_anomaly':'Anomaly', 'LOF_score':'Score'}),
            use_container_width=True, height=500, hide_index=True
        )

# ---- OCSVM tab ----
with tab_svm:
    st.markdown(model_card(
        "violet", "🧲 One‑Class SVM – Kernel Boundary Learning",
        f"sklearn.svm.OneClassSVM | kernel = {svm_kernel} | nu = {nu_svm:.2f}",
        "One‑Class SVM learns a non‑linear decision boundary enclosing the 'normal' supply feature space. "
        "Countries outside this boundary are flagged as anomalies. The RBF kernel allows the model to capture "
        "complex, curved feature interactions that linear methods miss entirely. The *nu* parameter controls "
        "the upper bound on the fraction of outliers.",
        f"<strong>{df_results['SVM_anomaly'].sum()}</strong> countries fall outside the learned normal boundary – "
        "these nations have rare feature combinations the model cannot associate with typical supply behaviour, "
        "requiring bespoke risk monitoring."
    ), unsafe_allow_html=True)
    
    col1, col2 = st.columns([3, 2])
    with col1:
        make_model_graph(df_results, 'SVM_score', 'country', 'SVM_anomaly', 'One‑Class SVM', 'Purples_r')
    with col2:
        st.dataframe(
            df_results[['country', 'SVM_anomaly', 'SVM_score']]
            .sort_values('SVM_score', ascending=False)
            .rename(columns={'country':'Country', 'SVM_anomaly':'Anomaly', 'SVM_score':'Score'}),
            use_container_width=True, height=500, hide_index=True
        )

# ---- Composite tab ----
with tab_comp:
    st.markdown(model_card(
        "amber", "🎯 Composite Anomaly Score – Ensemble Consensus",
        "Average of IF_score, LOF_score, SVM_score | Flag = any model flags",
        "Aggregates flags from Isolation Forest, LOF, and One‑Class SVM into a single composite "
        "country‑level vulnerability score. A score of 3 means every structural method independently flagged "
        "that country — the highest confidence anomalous classification in the dataset. Ensemble consensus "
        "dramatically reduces false positives.",
        f"<strong>{df_results['Composite_anomaly'].sum()}</strong> countries flagged by at least one model. "
        "The composite score is the most robust indicator for further investigation."
    ), unsafe_allow_html=True)
    
    col1, col2 = st.columns([3, 2])
    with col1:
        # For composite, we show top 7 or all anomalies based on composite score
        df_comp_anom = df_results[df_results['Composite_anomaly'] == 1].sort_values('Composite_score', ascending=False)
        view_comp = st.radio(
            "Show graph for Composite Score",
            options=["Top 7 Anomalies", "All Anomalies"],
            horizontal=True, key="toggle_comp"
        )
        if view_comp == "Top 7 Anomalies":
            df_plot = df_comp_anom.head(7)
            title = "Top 7 Countries by Composite Anomaly Score"
        else:
            df_plot = df_comp_anom
            title = f"All Anomalous Countries by Composite Score (n={len(df_comp_anom)})"
        fig_comp = px.bar(df_plot, x='Composite_score', y='country', orientation='h',
                          color='Composite_score', color_continuous_scale='Reds_r', title=title)
        fig_comp.update_layout(height=max(400, len(df_plot)*30), margin=dict(l=100))
        st.plotly_chart(fig_comp, use_container_width=True)
    with col2:
        st.dataframe(
            df_results[['country', 'Composite_anomaly', 'Composite_score']]
            .sort_values('Composite_score', ascending=False)
            .rename(columns={'country':'Country', 'Composite_anomaly':'Anomaly', 'Composite_score':'Score'}),
            use_container_width=True, height=500, hide_index=True
        )

# ---- Heatmap tab ----
with tab_heat:
    st.markdown("**Model agreement for top 30 riskiest countries (by composite score)**")
    top30 = df_results.nlargest(30, 'Composite_score')
    flag_cols = ['IF_anomaly', 'LOF_anomaly', 'SVM_anomaly']
    heatmap_df = top30[['country'] + flag_cols].set_index('country')
    fig_heat = px.imshow(heatmap_df.T, aspect="auto", color_continuous_scale="Reds",
                         title="Flagged by model (red = anomaly)")
    fig_heat.update_layout(height=400, xaxis_title="Country", yaxis_title="Model")
    st.plotly_chart(fig_heat, use_container_width=True)
    


st.markdown("---")
