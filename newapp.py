#!/usr/bin/env python3
"""
Structural Anomaly Detection App
Loads country_pca_features (172 countries x structural features), applies:
🌲 Isolation Forest | 📍 Local Outlier Factor | 🧲 One-Class SVM
Saves Spark-compatible checkpoint + CSV + visualizations.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import glob
import os
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

print("🚀 Structural Anomaly Detection - Starting...")

# Paths
CHECKPOINT_BASE = Path.cwd() / "pipeline_checkpoints"
OUTPUT_BASE = Path.cwd() / "outputs" / "anomalies"
CHECKPOINT_BASE.mkdir(exist_ok=True)
OUTPUT_BASE.mkdir(parents=True, exist_ok=True)

def load_spark_parquet(name: str) -> pd.DataFrame:
    """Load Spark parquet folder → pandas"""
    folder = CHECKPOINT_BASE / name
    if not folder.exists():
        raise FileNotFoundError(f"❌ Missing: {folder}")
    
    files = sorted(glob.glob(str(folder / "part-*.parquet")) + 
                   glob.glob(str(folder / "*.parquet")))
    if not files:
        raise FileNotFoundError(f"❌ No .parquet files in {folder}")
    
    dfs = []
    for f in files:
        try:
            df = pd.read_parquet(f)
            dfs.append(df)
            print(f"✅ Loaded {f} ({len(df)} rows)")
        except Exception as e:
            print(f"⚠️ Skip {f}: {e}")
    
    df = pd.concat(dfs, ignore_index=True)
    print(f"📊 {name}: {df.shape} | Countries: {df['country'].nunique() if 'country' in df else 0}")
    return df

# === STEP 1: LOAD FEATURES ===
print("\n1️⃣ Loading country_pca_features...")
df = load_spark_parquet("country_pca_features")

# Normalize country names
if 'country' in df.columns:
    df['country'] = df['country'].astype(str).str.strip().str.title()

# Select numeric features (exclude country)
feature_cols = [c for c in df.columns if df[c].dtype in ['float64', 'int64'] and c != 'country']
X = df[feature_cols].copy()
X = X.fillna(X.mean())  # Impute missing
print(f"Features: {len(feature_cols)} | Shape: {X.shape}")

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print("✅ Features scaled")

# === STEP 2: FIT MODELS ===
print("\n2️⃣ Fitting Structural Anomaly Detectors...")

# 🌲 Isolation Forest
print("   🌲 Isolation Forest...")
if_model = IsolationForest(contamination=0.1, random_state=42, n_jobs=-1)
if_pred = if_model.fit_predict(X_scaled)
if_flag = (if_pred == -1).astype(int)
if_score = -if_model.score_samples(X_scaled)  # Higher = more anomalous

# 📍 Local Outlier Factor
print("   📍 Local Outlier Factor...")
lof_model = LocalOutlierFactor(n_neighbors=20, contamination=0.1, n_jobs=-1)
lof_pred = lof_model.fit_predict(X_scaled)
lof_flag = (lof_pred == -1).astype(int)
lof_score = -lof_model.negative_outlier_factor_  # Higher = more anomalous

# 🧲 One-Class SVM
print("   🧲 One-Class SVM...")
ocsvm_model = OneClassSVM(nu=0.1, kernel='rbf', gamma='scale')
ocsvm_pred = ocsvm_model.fit_predict(X_scaled)
ocsvm_flag = (ocsvm_pred == -1).astype(int)
ocsvm_score = -ocsvm_model.decision_function(X_scaled)  # Higher = more anomalous

print("✅ All models fitted")

# === STEP 3: CREATE OUTPUT DF ===
print("\n3️⃣ Computing composite scores...")

# Merge predictions back
results = df[['country']].copy() if 'country' in df else pd.DataFrame()
results['structural_if_flag'] = if_flag
results['structural_if_score'] = if_score
results['structural_lof_flag'] = lof_flag
results['structural_lof_score'] = lof_score
results['structural_ocsvm_flag'] = ocsvm_flag
results['structural_ocsvm_score'] = ocsvm_score

# Composite
flag_cols = ['structural_if_flag', 'structural_lof_flag', 'structural_ocsvm_flag']
results['structural_composite_flag'] = (results[flag_cols].sum(axis=1) > 0).astype(int)
results['structural_composite_score'] = results[flag_cols].sum(axis=1)  # 0-3 severity
results['structural_composite_score_avg'] = np.mean([results['structural_if_score'], 
                                                   results['structural_lof_score'], 
                                                   results['structural_ocsvm_score']], axis=0)

print(f"Results: {results.shape}")
print("Anomalies by model:")
print(results[flag_cols].sum().to_dict())

# === STEP 4: SAVE CHECKPOINT (Spark format) ===
print("\n4️⃣ Saving checkpoint...")
out_dir = CHECKPOINT_BASE / "structural_anomaly_detection"
out_dir.mkdir(exist_ok=True)

# Spark-like part file
part_file = out_dir / "part-00000-struct-12345678-9abc-def0-1234-56789abcdef0-c000.snappy.parquet"
results.to_parquet(part_file, index=False)

# Success marker
(out_dir / "_SUCCESS").write_text("")
print(f"✅ Checkpoint: {out_dir}")

# === STEP 5: SAVE CSV FALLBACK ===
csv_file = OUTPUT_BASE / "structural_anomaly_detection.csv"
results.to_csv(csv_file, index=False)
print(f"✅ CSV: {csv_file}")

# === STEP 6: SUMMARY STATS ===
print("\n5️⃣ 📈 SUMMARY")
print(f"Total countries: {len(results)}")
print(f"Structural anomalies: {results['structural_composite_flag'].sum()} ({results['structural_composite_flag'].mean():.1%})")
top_outliers = results.nlargest(10, 'structural_composite_score_avg')['country'].tolist()
print(f"Top 10 outliers: {', '.join(top_outliers[:5])}...")

# === STEP 7: VISUALIZATIONS ===
print("\n6️⃣ 📊 Plotting...")

# Reduce to first 2 principal features for scatter
if len(feature_cols) >= 2:
    feat2d = X_scaled[:, :2]
    anomaly_level = results['structural_composite_score']
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    scatter = plt.scatter(feat2d[:, 0], feat2d[:, 1], c=anomaly_level, cmap='Reds', s=50)
    plt.colorbar(scatter, label='Composite Score')
    plt.xlabel('PCA/Feature 1')
    plt.ylabel('PCA/Feature 2')
    plt.title('Structural Anomalies (Red=Highest Risk)')
    
    # Top outliers bar
    plt.subplot(1, 2, 2)
    top20 = results.nlargest(20, 'structural_composite_score_avg')
    sns.barplot(data=top20, x='structural_composite_score_avg', y='country', palette='Reds_r')
    plt.title('Top 20 Structurally Anomalous Countries')
    plt.xlabel('Avg Anomaly Score')
    
    plt.tight_layout()
    plot_file = Path.cwd() / "structural_anomalies.png"
    plt.savefig(plot_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    print(f"✅ Plot saved: {plot_file}")
else:
    print("⚠️ Need ≥2 features for scatter plot")

print("\n🎉 Structural Anomaly Detection COMPLETE!")
print("\n✅ Test load: python -c \"import pandas as pd; df=pd.read_parquet('pipeline_checkpoints/structural_anomaly_detection/part-*.parquet'); print(df.head(3).to_dict(orient='records')[:2])\"")
print("\n📋 TODO.md Step 3: Run `python newapp.py`")
