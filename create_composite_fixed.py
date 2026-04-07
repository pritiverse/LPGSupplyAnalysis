#!/usr/bin/env python3
"""Create composite anomaly checkpoint from IF+LOF+OCSVM+CP"""

import pandas as pd
import numpy as np
import glob
import os
from pathlib import Path

CHECKPOINT_BASE = Path.cwd() / 'pipeline_checkpoints'

def load_checkpoint(name):
    """Load Spark parquet folder to pandas DF"""  
    folder = CHECKPOINT_BASE / name
    if not folder.exists():
        print(f"Missing: {folder}")
        return None
    
    parts = glob.glob(str(folder / 'part-*.parquet'))
    if not parts:
        print(f"No parts in {folder}")
        return None
    
    dfs = []
    for part in parts:
        try:
            df = pd.read_parquet(part)
            dfs.append(df)
        except Exception as e:
            print(f"Error loading {part}: {e}")
    
    if not dfs:
        return None
    
    df = pd.concat(dfs, ignore_index=True)
    
    # Normalize country
    if 'country' in df.columns:
        df['country'] = df['country'].astype(str).str.strip()
    
    print(f"Loaded {name}: {df.shape}, unique countries={len(df['country'].unique())}")
    print(f"Columns: {list(df.columns)}")
    return df

def standardize_cols(df, method):
    """Standardize flag and score columns"""  
    flag_col = None
    score_col = None
    
    # Common flag names
    flag_candidates = [f'{method}_flag', f'{method}_anomaly_flag', 'anomaly', 'flag', 'is_anomaly']
    for cand in flag_candidates:
        if cand in df.columns:
            flag_col = cand
            break
    
    # Common score names  
    score_candidates = [f'{method}_score', f'{method}_anomaly_score', 'anomaly_score', 'score', 'decision_function']
    for cand in score_candidates:
        if cand in df.columns:
            score_col = cand
            break
    
    df_out = df.copy()
    
    if flag_col:
        # Ensure flag is 0/1, 1=anomaly (handle sklearn -1 convention)
        flag_series = pd.to_numeric(df[flag_col], errors='coerce')
        df_out[f'{method}_flag'] = ((flag_series != 0) | (flag_series == -1)).astype(int)
    
    if score_col:
        df_out[f'{method}_score'] = pd.to_numeric(df[score_col], errors='coerce').fillna(0)
    
    print(f"{method}: flag={flag_col}, score={score_col}")
    return df_out

def main():
    print("=== Creating Composite Anomaly Checkpoint ===")
    
    # Load raw data
    print("\n1. Loading checkpoints...")
    if_raw = load_checkpoint('anomaly_isolation_forest')
    lof_raw = load_checkpoint('anomaly_lof')
    ocsvm_raw = load_checkpoint('anomaly_ocsvm')
    cp_raw = load_checkpoint('change_points')
    
    # Check if all loaded successfully
    if any(d is None for d in [if_raw, lof_raw, ocsvm_raw, cp_raw]):
        print("\n❌ Missing checkpoint(s)! Run anomaly pipeline first.")
        return
    
    # Standardize
    print("\n2. Standardizing columns...")
    if_df = standardize_cols(if_raw, 'if')
    lof_df = standardize_cols(lof_raw, 'lof')
    ocsvm_df = standardize_cols(ocsvm_raw, 'ocsvm')
    cp_df = standardize_cols(cp_raw, 'cp')
    
    # Prepare merge cols
    methods = ['if', 'lof', 'ocsvm', 'cp']
    data_list = [if_df, lof_df, ocsvm_df, cp_df]
    
    # Start with countries
    merged = data_list[0][['country']].drop_duplicates().copy()
    
    # Merge each method
    for i, (method, df) in enumerate(zip(methods, data_list)):
        cols_to_merge = ['country']
        if f'{method}_flag' in df.columns:
            cols_to_merge.append(f'{method}_flag')
        if f'{method}_score' in df.columns:
            cols_to_merge.append(f'{method}_score')
        
        merged = merged.merge(df[cols_to_merge], on='country', how='outer')
    
    # Fill NaN flags/scores with 0
    flag_cols = [f'{m}_flag' for m in methods]
    score_cols = [f'{m}_score' for m in methods]
    
    for col in flag_cols + score_cols:
        if col in merged.columns:
            merged[col] = merged[col].fillna(0)
        else:
            merged[col] = 0
            print(f"Added missing col: {col}")
    
    # Compute composites
    flag_sum = merged[flag_cols].sum(axis=1)
    merged['composite_flag'] = (flag_sum > 0).astype(int)
    merged['composite_score'] = flag_sum  # 0-4 severity count
    merged['composite_score_avg'] = merged[score_cols].mean(axis=1)
    
    print(f"\n3. Merged result: {merged.shape}")
    print("Countries:", len(merged))
    print("Columns:", list(merged.columns))
    print("\nFlag counts:")
    print(merged[flag_cols + ['composite_flag']].sum())
    print("\nHigh-risk (score>=2):", (merged['composite_score'] >= 2).sum())
    
    # Save as Spark-compatible structure
    out_dir = CHECKPOINT_BASE / 'anomaly_composite'
    out_dir.mkdir(exist_ok=True)
    
    # Part file name like Spark
    part_file = out_dir / 'part-00000-abc123de-f456-7890-abcd-ef1234567890-c000.snappy.parquet'
    merged.to_parquet(part_file, index=False)
    
    # Success marker
    (out_dir / '_SUCCESS').write_text('')
    
    print(f"\n✅ Composite checkpoint created: {out_dir}")
    print("\nTest command:")
    print(f"python -c \"import pandas as pd; df=pd.read_parquet('{out_dir}/part-*.parquet'); print('Shape:', df.shape); print('dtypes:', df.dtypes.to_dict()); print('head:', df.head(3).to_dict())\"")

if __name__ == "__main__":
    main()

