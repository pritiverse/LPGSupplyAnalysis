#!/usr/bin/env python3
"""Create composite anomaly checkpoint from IF+LOF+OCSVM+CP"""

import pandas as pd
import numpy as np
import glob
import os
from pathlib import Path

CHECKPOINT_BASE = Path('pipeline_checkpoints')

def load_checkpoint(name):
    #"\"\"Load Spark parquet folder to pandas DF\"\"\"  
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
    
    print(f"Loaded {name}: {df.shape} rows={len(df['country'].unique())} countries")
    print(f"Columns: {df.columns.tolist()}")
    return df

def standardize_cols(df, method):
    #"\"\"Standardize flag and score columns\"\"\"  
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
    
    if flag_col:
        # Ensure flag is 0/1, 1=anomaly
        if df[flag_col].dtype in ['int64', 'float64']:
            df[f'{method}_flag'] = (df[flag_col] != 0).astype(int)  # Handle -1==anomaly for sklearn
        else:
            df[f'{method}_flag'] = (df[flag_col].fillna(0) > 0).astype(int)
    
    if score_col:
        df[f'{method}_score'] = pd.to_numeric(df[score_col], errors='coerce').fillna(0)
    
    print(f"{method}: flag='{flag_col}'→'{method}_flag', score='{score_col}'→'{method}_score'")
    return df

def main():
    print("Loading 4 anomaly checkpoints...")
    
    # Load raw data
    if_raw = load_checkpoint('anomaly_isolation_forest')
    lof_raw = load_checkpoint('anomaly_lof') 
    ocsvm_raw = load_checkpoint('anomaly_ocsvm')
    cp_raw = load_checkpoint('change_points')
    
    if not all([if_raw, lof_raw, ocsvm_raw, cp_raw]):
        print("Missing one or more checkpoints!")
        return
    
    # Standardize
    if_df = standardize_cols(if_raw, 'if')
    lof_df = standardize_cols(lof_raw, 'lof')
    ocsvm_df = standardize_cols(ocsvm_raw, 'ocsvm')
    cp_df = standardize_cols(cp_raw, 'cp')
    
    # Merge on country (outer join)
    dfs = [if_df, lof_df, ocsvm_df, cp_df]
    merged = dfs[0][['country']].copy()
    
    for df in dfs:
        merged = merged.merge(df[['country', f'{df.name}_flag', f'{df.name}_score']], 
                             on='country', how='outer')
    
    # Fill missing flags/scores with 0
    flag_cols = ['if_flag', 'lof_flag', 'ocsvm_flag', 'cp_flag']
    score_cols = ['if_score', 'lof_score', 'ocsvm_score', 'cp_score']
    
    for col in flag_cols + score_cols:
        if col not in merged.columns:
            merged[col] = 0
        else:
            merged[col] = merged[col].fillna(0)
    
    # Composite calculations
    merged['composite_flag'] = (merged[flag_cols].sum(axis=1) > 0).astype(int)
    merged['composite_score'] = merged[flag_cols].sum(axis=1)  # 0-4 severity
    merged['composite_score_avg'] = merged[score_cols].mean(axis=1)
    
    print(f"\nMerged: {merged.shape}")
    print(f"Countries: {len(merged)}")
    print("Columns:", merged.columns.tolist())
    print("\nFlag summary:")
    print(merged[flag_cols + ['composite_flag']].sum().to_frame('Count'))
    print("\nScore ranges:")
    for col in score_cols:
        print(f"{col}: {merged[col].min():.3f} - {merged[col].max():.3f}")
    
    # Save Spark-like structure
    out_dir = CHECKPOINT_BASE / 'anomaly_composite'
    out_dir.mkdir(exist_ok=True)
    
    # Save single part (86 rows small)
    part_file = out_dir / 'part-00000-12345678-1234-1234-1234-123456789abc-c000.snappy.parquet'
    merged.to_parquet(part_file, index=False)
    
    # Create _SUCCESS
    (out_dir / '_SUCCESS').write_text('')
    
    print(f"\n✅ Saved to {out_dir}")
    print("Test: python -c \"import pandas as pd; df=pd.read_parquet('pipeline_checkpoints/anomaly_composite/part-*.parquet'); print('Shape:', df.shape); print('Columns:', df.columns.tolist()); print(df.head(3))\"")

if __name__ == '__main__':
    main()

