# utils/css.py
"""Custom CSS styling"""

import streamlit as st


def apply_css():
    """Apply all custom CSS to the app"""
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@300;400;500&family=Sora:wght@300;400;600;700;800&display=swap');
        
        html, body, [class*="css"] {
            font-family: 'Sora', sans-serif;
        }
        
        /* Sidebar */
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0d1b2a 0%, #1b2838 60%, #0d1b2a 100%);
            border-right: 1px solid #1e3a5f;
        }
        
        [data-testid="stSidebar"] * { color: #c9d8e8 !important; }
        
        /* Main area */
        .main { background: #07111c; }
        .block-container { padding: 1.5rem 2rem; }
        
        /* KPI cards */
        [data-testid="metric-container"] {
            background: linear-gradient(135deg, #0d1b2a, #112233);
            border: 1px solid #1e3a5f;
            border-radius: 10px;
            padding: 16px !important;
        }
        
        [data-testid="metric-container"] label { 
            color: #8bbdd9 !important; 
            font-size: 11px !important; 
            text-transform: uppercase;
        }
        
        [data-testid="metric-container"] [data-testid="stMetricValue"] { 
            color: #e8f4ff !important; 
            font-size: 26px !important; 
            font-weight: 700;
        }
        
        /* Badges */
        .badge-strong { 
            background: #3d0010; 
            color: #ff6b6b; 
            padding: 4px 12px; 
            border-radius: 20px; 
            font-weight: 700; 
            font-size: 12px; 
            border: 1px solid #ff4444; 
        }
        .badge-moderate { 
            background: #2a1500; 
            color: #ffa040; 
            padding: 4px 12px; 
            border-radius: 20px; 
            font-weight: 700; 
            font-size: 12px; 
            border: 1px solid #f0703a; 
        }
        .badge-weak { 
            background: #1f1a00; 
            color: #ffd050; 
            padding: 4px 12px; 
            border-radius: 20px; 
            font-weight: 700; 
            font-size: 12px; 
            border: 1px solid #d4a000; 
        }
        .badge-normal { 
            background: #001a10; 
            color: #50e090; 
            padding: 4px 12px; 
            border-radius: 20px; 
            font-weight: 700; 
            font-size: 12px; 
            border: 1px solid #20c060; 
        }
        
        /* Insight boxes */
        .insight-box {
            background: linear-gradient(135deg, #0d2137, #0a2744);
            padding: 16px 20px;
            border-radius: 10px;
            border-left: 4px solid #5bc0de;
            margin: 12px 0;
            color: #c9d8e8;
        }
        
        /* Section tag */
        .section-tag {
            display: inline-block;
            background: #1e3a5f;
            color: #5bc0de;
            font-family: 'DM Mono', monospace;
            font-size: 10px;
            letter-spacing: 0.1em;
            padding: 3px 10px;
            border-radius: 4px;
            margin-bottom: 8px;
            text-transform: uppercase;
        }
        
        /* Tabs */
        .stTabs [data-baseweb="tab-list"] {
            gap: 2px;
            background: #0d1b2a;
            border-radius: 8px;
            padding: 4px;
            border: 1px solid #1e3a5f;
        }
        
        .stTabs [data-baseweb="tab"] {
            padding: 8px 16px;
            border-radius: 6px;
            font-size: 13px;
            font-weight: 500;
            color: #8bbdd9;
            font-family: 'DM Mono', monospace;
        }
        
        .stTabs [aria-selected="true"] {
            background: #1e3a5f !important;
            color: #5bc0de !important;
        }
        
        /* Headers */
        h1 { color: #e8f4ff !important; font-weight: 800 !important; }
        h2 { color: #c9d8e8 !important; font-weight: 700 !important; }
        h3 { color: #8bbdd9 !important; font-weight: 600 !important; }
    </style>
    """, unsafe_allow_html=True)