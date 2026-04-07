#!/usr/bin/env python3
"""
Precompute Composite Structural Anomaly Score from IF + LOF + OCSVM checkpoints.
Saves to pipeline_checkpoints/anomaly_composite/ for dashboard loading.
"""

import pandas as pd
import numpy as np
import os
import glob

def load_parquet_folder(folder_path):
    """Load all parquet parts from folder."""
    if not os.path.exists(folder_path):
        print(f"Missing: {folder_path}")
        return None
    parts = glob.glob(os.path.join(folder_path, "part-*.parquet"))
    if not parts:
        print(f"No parquet parts in: {folder_path}")
        return None
    dfs = []
    for f in sorted(parts):
        try:
            df = pd.read_parquet(f)
            print(f"Loaded {os.path.basename(f)}: {len(df)} rows")
            dfs.append(df)
        except Exception as e:
            print(f"Failed {os.path.basename(f)}: {e}")
    return pd.concat(dfs, ignore_index=True) if dfs else None

print("=== LOADING STRUCTURAL ANOMALY CHECKPOINTS ===")
base_path = "pipeline_checkpoints"
if_df = load_parquet_folder(os.path.join(base_path, "anomaly_isolation_forest"))
lof_df = load_parquet_folder(os.path.join(base_path, "anomaly_lof"))
ocsvm_df = load_parquet_folder(os.path.join(base_path, "anomaly_ocsvm"))

if if_df is None or lof_df is None or ocsvm_df is None:
    print("Missing checkpoints. Run pipeline.py --ml first.")
    exit(1)

c_country = 'country'
common_countries = sorted(set(if_df[c_country]) & set(lof_df[c_country]) & set(ocsvm_df[c_country]))
print(f"Common countries: {len(common_countries)}")

# Known columns from output
if_score_col = 'if_anomaly_score'
if_flag_col = 'if_anomaly_flag'
lof_score_col = 'lof_score'
lof_flag_col = 'lof_flag'
ocsvm_score_col = 'ocsvm_score'
ocsvm_flag_col = 'ocsvm_flag'

print(f"Columns: IF({if_score_col}/{if_flag_col}), LOF({lof_score_col}/{lof_flag_col}), OCSVM({ocsvm_score_col}/{ocsvm_flag_col})")

# Aggregate: max score, any flag == True
def get_agg(df, score_col, flag_col):
    agg_score = df.groupby(c_country)[score_col].max()
    agg_flag = df.groupby(c_country).apply(lambda g: (g[flag_col] == -1).any())
    return pd.DataFrame({'score': agg_score, 'flag': agg_flag})

if_agg = get_agg(if_df, if_score_col, if_flag_col)
lof_agg = get_agg(lof_df, lof_score_col, lof_flag_col)
ocsvm_agg = get_agg(ocsvm_df, ocsvm_score_col, ocsvm_flag_col)

comp_df = pd.DataFrame(index=pd.Index(common_countries, name=c_country))

comp_df['IF_score'] = if_agg.loc[comp_df.index, 'score']
comp_df['LOF_score'] = lof_agg.loc[comp_df.index, 'score']
comp_df['SVM_score'] = ocsvm_agg.loc[comp_df.index, 'score']

# Normalize 0-1
for col in ['IF_score', 'LOF_score', 'SVM_score']:
    s = comp_df[col]
    comp_df[f'{col}_norm'] = (s - s.min()) / (s.max() - s.min())

comp_df['Composite_score'] = comp_df[[c + '_norm' for c in ['IF_score', 'LOF_score', 'SVM_score']]].mean(axis=1)

# Flags
comp_df['IF_flag'] = if_agg.loc[comp_df.index, 'flag']
comp_df['LOF_flag'] = lof_agg.loc[comp_df.index, 'flag']
comp_df['SVM_flag'] = ocsvm_agg.loc[comp_df.index, 'flag']
comp_df['Vote_count'] = comp_df[['IF_flag', 'LOF_flag', 'SVM_flag']].sum(axis=1)
comp_df['Composite_flag'] = comp_df['Vote_count'] > 0

print(f"Composite results: {comp_df['Composite_flag'].sum()} flagged countries, {comp_df['Vote_count'].value_counts().to_dict()} votes")

# Save
output_dir = os.path.join(base_path, 'anomaly_composite')
os.makedirs(output_dir, exist_ok=True)
output_file = os.path.join(output_dir, 'part-00000-composite.snappy.parquet')
comp_df.reset_index().to_parquet(output_file, index=False)
print(f"Saved: {output_file}")

print("Success! Composite checkpoint created.")
print("Test in dashboard: cd lpg_dashboard && streamlit run app.py")

