# utils/helpers.py
"""Helper functions for the dashboard"""

import streamlit as st
import pandas as pd
import numpy as np
from .constants import (
    get_region, normalize_country, get_iso,
    SEVERITY_COLORS, SEVERITY_ORDER, PLOTLY_DARK
)


def extract_spark_vector_prob(x):
    """Extract probability from PySpark VectorUDT"""
    try:
        if isinstance(x, dict) and 'values' in x:
            return float(x['values'][-1])
        elif isinstance(x, (list, np.ndarray)):
            return float(x[-1])
        elif hasattr(x, 'item'):
            return float(x.item())
        return float(x)
    except (TypeError, ValueError, IndexError):
        return 0.0


def severity_badge(severity):
    """Return HTML badge for severity level"""
    badge_classes = {
        'Strong': 'badge-strong',
        'Moderate': 'badge-moderate',
        'Weak': 'badge-weak',
        'Normal': 'badge-normal'
    }
    cls = badge_classes.get(severity, 'badge-normal')
    return f'<span class="{cls}">{severity}</span>'


def col_sum(df, col):
    """Safe column sum"""
    if df is not None and col in df.columns and not df.empty:
        return int(df[col].sum())
    return 0


def table_limit(label, default=15):
    """Get table row limit based on toggle"""
    show_all = st.toggle("Show all rows", value=False, key=f"showall_{label}")
    return 0 if show_all else default


def add_region(df, col='country'):
    """Add region column to dataframe using safe get_region()"""
    if df is not None and col in df.columns and not df.empty:
        df = df.copy()
        df['region'] = df[col].apply(lambda x: get_region(normalize_country(x)))
    return df


def add_iso(df, col='country'):
    """Add ISO codes for plotting"""
    if df is not None and col in df.columns and not df.empty:
        df = df.copy()
        df['iso'] = df[col].apply(get_iso)
    return df


def section_tag(label):
    """Display section tag"""
    st.markdown(f'<div class="section-tag">{label}</div>', unsafe_allow_html=True)


def apply_plotly_dark(fig, height=None):
    """Apply dark theme to plotly figure"""
    fig.update_layout(**PLOTLY_DARK)
    if height:
        fig.update_layout(height=height)
    return fig


def safe_sort_severity(df, severity_col='ShockSeverity'):
    """Sort dataframe by severity order"""
    if severity_col in df.columns:
        df = df.copy()
        df[severity_col] = pd.Categorical(
            df[severity_col],
            categories=SEVERITY_ORDER,
            ordered=True
        )
        df = df.sort_values(severity_col)
    return df


def get_structural_anomaly_score(df, country):
    """Get structural anomaly score for a country"""
    if df is None or df.empty:
        return 0
    
    country_row = df[df['country'] == country]
    if len(country_row) == 0:
        return 0
    
    # Use actual column names from your data
    score_col = 'composite_country_anomaly_score'
    if score_col in country_row.columns:
        return int(country_row.iloc[0][score_col])
    
    # Fallback: calculate from individual flags
    flag_cols = ['lof_country_flag', 'ocsvm_country_flag', 'if_country_flag', 'cp_country_flag']
    existing_flags = [c for c in flag_cols if c in country_row.columns]
    if existing_flags:
        return int(country_row[existing_flags].sum(axis=1).iloc[0])
    
    return 0


def safe_get_data(data: dict | None, key: str, default=None):
    """Safely get data from dict with ok() check. Returns default if missing/empty."""
    if data is None:
        return default
    val = data.get(key, default)
    if val is None or (isinstance(val, pd.DataFrame) and (val.empty or len(val) == 0)):
        return default
    return val


def standardize_flags(df, flag_candidates=None):
    """Normalize anomaly flags to 1=True across models (flag=-1/1/True → std=1)"""
    if df is None or df.empty:
        return df
    df = df.copy()
    candidates = flag_candidates or ['anomaly_flag', 'flag', 'shock_flag', 'anomaly']
    for cand in candidates:
        if cand in df.columns:
            col = df[cand]
            if col.dtype == 'bool':
                df[f'{cand}_std'] = col.astype(int)
            elif (col == -1).any():
                df[f'{cand}_std'] = (col == -1).astype(int)
            elif (col == 1).any() or col.nunique() <= 2:
                df[f'{cand}_std'] = (col == 1).astype(int)
            elif col.nunique() > 2:
                df[f'{cand}_std'] = (col > col.quantile(0.95)).astype(int)
            return df
    return df  # no flags found


def detect_anomaly_col(df, candidates=None):
    """Heuristically find best flag/score column"""
    if df is None or df.empty:
        return None
    candidates = candidates or ['anomaly_flag', 'flag', 'shock_flag', 'anomaly', 'score']
    for c in candidates:
        if c in df.columns:
            return c
    # Fallback to high-deviation cols
    score_like = [c for c in df.columns if any(x in c.lower() for x in ['score', 'deviation', 'error'])]
    return score_like[0] if score_like else None


def get_yearly_anomaly_count(df, country, year=None):
    """Get yearly anomaly count for a country"""
    if df is None or df.empty:
        return 0
    
    country_df = df[df['country'] == country]
    if year:
        country_df = country_df[country_df['year'] == year]
    
    if len(country_df) == 0:
        return 0
    
    # Look for composite score column
    score_col = 'composite_yearly_anomaly_score'
    if score_col in country_df.columns:
        return int(country_df[score_col].sum())
    
    # Fallback: count flag columns
    flag_cols = [c for c in country_df.columns if c.endswith('_flag') and c != 'cp_country_flag']
    if flag_cols:
        return int(country_df[flag_cols].sum(axis=1).iloc[0] if len(country_df) == 1 else country_df[flag_cols].sum().sum())
    
    return 0
