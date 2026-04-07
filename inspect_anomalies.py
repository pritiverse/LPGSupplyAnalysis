import pandas as pd
import glob
import os

CHECKPOINT_BASE = r'pipeline_checkpoints'

def inspect_checkpoint(name):
    folder = os.path.join(CHECKPOINT_BASE, name)
    if not os.path.exists(folder):
        print(f"❌ Missing: {folder}")
        return
    
    parts = [f for f in glob.glob(os.path.join(folder, "part-*.parquet")) if not f.endswith('.crc')]
    if not parts:
        print(f"❌ No parquet parts in {folder}")
        return
    
    print(f"\n🔍 {name.upper()}")
    print(f"📁 Folder: {folder}")
    print(f"📄 Parts: {len(parts)}")
    
    dfs = []
    for part in parts:
        df_part = pd.read_parquet(part)
        dfs.append(df_part)
    
    df = pd.concat(dfs, ignore_index=True)
    
    print(f"📊 Shape: {df.shape}")
    print(f"🏷️ Columns: {list(df.columns)}")
    print(f"🌍 Countries: {sorted(df['country'].unique())[:5]}... ({len(df['country'].unique())} total)")
    
    # Check for y and common anomaly cols
    anomaly_cols = [col for col in df.columns if any(k in col.lower() for k in ['anomaly', 'flag', 'score', 'y', 'outlier'])]
    print(f"🚩 Anomaly cols: {anomaly_cols}")
    
    if not df.empty:
        print("📋 Sample (head 3):")
        print(df.head(3).to_string())
        print("\nFlag value counts:")
        for col in anomaly_cols:
            if col in df.columns:
                print(f"  {col}: {df[col].value_counts().to_dict()}")
    
    print("-" * 80)

# Inspect IF, LOF, OCSVM, dashboard
for name in ['anomaly_isolation_forest', 'anomaly_lof', 'anomaly_ocsvm', 'anomaly_dashboard']:
    inspect_checkpoint(name)

