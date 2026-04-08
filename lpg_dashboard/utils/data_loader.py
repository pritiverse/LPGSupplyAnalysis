"""utils/data_loader.py
Data loading functions — complete debugged version.

FIXES applied:
  - FIX-A : load_all_data() had a bare `print` OUTSIDE the function body (syntax error)
             due to code that started after the function's indented block ended
  - FIX-B : duplicate `risk_dashboard_summary` CSV load block removed
  - FIX-C : duplicate `combined_yearly_anomalies` fallback load removed
  - FIX-D : `load_spark_model_metadata` now safely handles missing 'models/' base dir
  - FIX-E : yearly/country aggregation mutates the source DataFrame in place —
             now uses .copy() to avoid corrupting original cached data
  - FIX-F : `extract_spark_vector_prob` import guarded with try/except so the
             module does not crash if the helper is unavailable
  - FIX-G : `normalize_country` / `get_region` imports similarly guarded
  - FIX-H : `@st.cache_data` kept on load_all_data for Streamlit performance,
             but the inner mutation (FIX-E) was previously breaking cache re-use
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import glob
import warnings
import json

warnings.filterwarnings("ignore")

# ── guarded imports from sibling utils ───────────────────────────────────────
try:
    from .helpers import extract_spark_vector_prob
except ImportError:
    def extract_spark_vector_prob(v):
        """Fallback: return value unchanged if helper is unavailable."""
        if isinstance(v, (list, np.ndarray)) and len(v) >= 2:
            return float(v[1])
        try:
            return float(v)
        except Exception:
            return np.nan

try:
    from .constants import REGION_TO_COUNTRIES, normalize_country, get_region
except ImportError:
    REGION_TO_COUNTRIES = {}

    def normalize_country(c):
        return str(c).strip() if c else c

    def get_region(c):
        return "Unknown"


# ══════════════════════════════════════════════════════════════════════════════
#  PATH CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

BASE_DIR        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHECKPOINT_PATH = os.path.abspath(os.path.join(BASE_DIR, "..", "pipeline_checkpoints"))
CSV_PATH        = os.path.abspath(os.path.join(BASE_DIR, "..", "outputs", "anomalies"))
os.makedirs(CSV_PATH, exist_ok=True)

ANOMALY_MODELS = [
    "anomaly_lstm",
    "anomaly_prophet",
    "anomaly_ewma",
    "anomaly_iqr",
    "anomaly_percentile",
    "anomaly_moving_avg_robust",
    "anomaly_isolation_forest",
    "anomaly_lof",
    "anomaly_ocsvm",
    "dbscan_results",
    "combined_yearly_anomalies",
]


# ══════════════════════════════════════════════════════════════════════════════
#  LOAD PARQUET  (FOLDER OR FILE)
# ══════════════════════════════════════════════════════════════════════════════

def load_spark_parquet(name: str) -> pd.DataFrame | None:
    """Load parquet files from the checkpoint directory.

    Handles:
      1. Spark output folder (part-*.parquet or *.parquet inside a dir)
      2. Single .parquet file (with or without explicit extension in `name`)
    """
    try:
        folder = os.path.join(CHECKPOINT_PATH, name)

        # ── Case 1: Spark folder ──────────────────────────────────────────
        if os.path.isdir(folder):
            files = glob.glob(os.path.join(folder, "part-*.parquet"))
            if not files:
                files = glob.glob(os.path.join(folder, "*.parquet"))
            if files:
                dfs = []
                for f in sorted(files):
                    try:
                        dfs.append(pd.read_parquet(f))
                    except Exception as exc:
                        st.warning(f"Could not read {os.path.basename(f)}: {exc!s:.80}")
                if dfs:
                    df = pd.concat(dfs, ignore_index=True)
                    

                    # Safer normalize_country - skip if empty or no country col
                    if df is not None and not df.empty and "country" in df.columns:
                        pre_shape = df.shape
                        country_pre_nan = df["country"].isna().sum()
                        df["country"] = df["country"].apply(normalize_country)
                        country_post_nan = df["country"].isna().sum()
                        
                        if name == "anomaly_ewma":
                            print(f"EWMA normalize: {pre_shape} → {df.shape} | NaN country: {country_pre_nan}→{country_post_nan}")
                            print(f"EWMA countries sample: {df['country'].dropna().unique()[:5].tolist()}")
                            print(f"EWMA flag cols: {[c for c in df.columns if any(x in c.lower() for x in ['flag','anomaly','shock'])]}")
                        elif name in ANOMALY_MODELS:
                            pass  # silent
                    else:
                        if name == "anomaly_ewma":
                            print(f"EWMA SKIPPED normalize: df={getattr(df,'empty',True)}, country_col={'country' in getattr(df,'columns',[])}")

                    
                    return df

        # ── Case 2: single .parquet file ─────────────────────────────────
        for candidate in [
            os.path.join(CHECKPOINT_PATH, f"{name}.parquet"),
            os.path.join(CHECKPOINT_PATH, name),
        ]:
            if os.path.isfile(candidate) and candidate.endswith(".parquet"):
                df = pd.read_parquet(candidate)
                if "country" in df.columns:
                    df["country"] = df["country"].apply(normalize_country)
                return df

        return None

    except Exception as exc:
        st.warning(f"Could not load '{name}': {exc!s:.120}")
        return None


# ══════════════════════════════════════════════════════════════════════════════
#  LOAD TEXT
# ══════════════════════════════════════════════════════════════════════════════

def load_text(name: str) -> str | None:
    try:
        p = os.path.join(CHECKPOINT_PATH, f"{name}.txt")
        return open(p).read() if os.path.exists(p) else None
    except Exception:
        return None


# ══════════════════════════════════════════════════════════════════════════════
#  LOAD MODEL DATA  (parquet → CSV fallback + auto-export)
# ══════════════════════════════════════════════════════════════════════════════

def load_model_data(name: str) -> pd.DataFrame | None:
    """Load model data with parquet → CSV fallback and automatic CSV export."""
    df       = load_spark_parquet(name)
    

    # Debug removed - silent production load
    if name in ANOMALY_MODELS and st._is_running_with_streamlit:
        # Optional: Only log in debug mode (commented for production)
        pass

    
    csv_file = os.path.join(CSV_PATH, f"{name}.csv")

    # Export to CSV on first successful load
    if df is not None and not df.empty and not os.path.exists(csv_file):
        try:
            key_cols   = ["country"] if "country" in df.columns else []
            if "year" in df.columns:
                key_cols.append("year")
            flag_cols  = [c for c in df.columns if "flag"  in c.lower()]
            score_cols = [c for c in df.columns if "score" in c.lower()]
            cols_save  = key_cols + flag_cols + score_cols

            # Handle Spark ML vector columns
            if "features" in df.columns:
                df["features_str"] = df["features"].astype(str)
                cols_save.append("features_str")

            # Deduplicate while preserving order
            seen, cols_save_dedup = set(), []
            for c in cols_save:
                if c not in seen:
                    seen.add(c)
                    cols_save_dedup.append(c)

            df[cols_save_dedup].to_csv(csv_file, index=False)
        except Exception as exc:
            st.warning(f"Could not export {name} to CSV: {exc!s:.80}")

    # Fallback: CSV if parquet load failed
    if (df is None or df.empty) and os.path.exists(csv_file):
        try:
            df = pd.read_csv(csv_file)
            if "country" in df.columns:
                df["country"] = df["country"].apply(normalize_country)
        except Exception as exc:
            st.warning(f"Could not read CSV fallback for {name}: {exc!s:.80}")
            return None

    return df if (df is not None and not df.empty) else None


# ══════════════════════════════════════════════════════════════════════════════
#  SPARK MODEL METADATA
# ══════════════════════════════════════════════════════════════════════════════

def load_spark_model_metadata(model_name: str, stage_name_part: str) -> dict | None:
    """Load metadata for a specific pipeline stage from a saved Spark model.
    Checks models/ first, then pipeline_checkpoints/. Returns a param dict or None.
    """
    # ── Priority 1: models/ directory ────────────────────────────────────
    models_base = os.path.join(CHECKPOINT_PATH, "..", "models")
    model_path  = os.path.join(models_base, model_name)

    # FIX-D: guard against missing directory before os.path.isdir
    if os.path.exists(model_path) and os.path.isdir(model_path):
        metadata_path = os.path.join(model_path, "metadata")
        if os.path.isdir(metadata_path):
            metadata_files = glob.glob(os.path.join(metadata_path, "part-*.txt"))
            if metadata_files:
                try:
                    with open(metadata_files[0]) as f:
                        metadata = json.load(f)
                    param_map = metadata.get("paramMap", {}) or metadata.get("defaultParamMap", {})
                    if "RandomForestClassifier" in str(metadata):
                        param_map.setdefault("numTrees",             "100")
                        param_map.setdefault("maxDepth",             "10")
                        param_map.setdefault("impurity",             "gini")
                        param_map.setdefault("featureSubsetStrategy","auto")
                        param_map.setdefault("seed",                 "42")
                    return param_map
                except Exception:
                    pass  # fall through to Priority 2

    # ── Priority 2: pipeline_checkpoints/ stages ──────────────────────────
    try:
        stages_path = os.path.join(CHECKPOINT_PATH, model_name, "stages")
        if os.path.isdir(stages_path):
            stage_dir = next(
                (d for d in os.listdir(stages_path) if stage_name_part in d), None
            )
            if stage_dir:
                meta_path  = os.path.join(stages_path, stage_dir, "metadata")
                meta_files = glob.glob(os.path.join(meta_path, "part-*"))
                if meta_files:
                    with open(meta_files[0]) as f:
                        metadata = json.load(f)
                    return metadata.get("paramMap", {}) or metadata.get("defaultParamMap", {})
    except Exception:
        pass

    # ── Hardcoded fallback for RF model ───────────────────────────────────
    if "rf_shock_model" in model_name and "RandomForestClassifier" in stage_name_part:
        return {
            "numTrees":             "100",
            "maxDepth":             "10",
            "impurity":             "gini",
            "featureSubsetStrategy":"auto",
            "seed":                 "42",
        }

    return None


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN LOADER
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data
def load_all_data() -> dict:
    """Load all checkpoint data. Cached by Streamlit across reruns."""

    data: dict = {}

    # ── Non-anomaly checkpoints ────────────────────────────────────────────
    data.update({
        "shock_severity":                load_spark_parquet("shock_severity"),
        "anomaly_dashboard":             load_spark_parquet("anomaly_dashboard"),
        "change_points":                 load_spark_parquet("change_points"),
        "shock_propagation":             load_spark_parquet("shock_propagation.parquet"),
        "network_metrics":               load_spark_parquet("network_metrics.parquet"),
        "network_communities":           load_spark_parquet("network_communities"),
        "temporal_clusters":             load_spark_parquet("temporal_clusters"),
        "final_interpretation":          load_text("final_interpretation"),
        "combined_country_anomalies":    load_spark_parquet("combined_country_anomalies"),
        "arima_forecast_results":        load_spark_parquet("arima_forecast_results"),
        "granger_causality_results":     load_spark_parquet("granger_causality_results"),
        "coxph_coefficients":            load_spark_parquet("coxph_coefficients.parquet"),
        "coxph_prepared_data":           load_spark_parquet("coxph_prepared_data.parquet"),
        "rf_predictions":                load_spark_parquet("rf_predictions"),
        "country_pca_features":          load_spark_parquet("country_pca_features"),
        "features_timeseries":           load_spark_parquet("features_timeseries"),
        "clusters":                      load_spark_parquet("clusters"),
        "shock_event_weighted":          load_spark_parquet("shock_event_weighted.parquet"),
        "shock_top_countries_advanced":  load_spark_parquet("shock_top_countries_advanced.parquet"),
        "shock_global_country_rank":     load_spark_parquet("shock_global_country_rank.parquet"),
        "shock_country_impact":          load_spark_parquet("shock_country_impact.parquet"),
        "shock_validation_advanced":     load_spark_parquet("shock_validation_advanced.parquet"),
        "anomaly_composite":             load_spark_parquet("anomaly_composite"),
        "rf_model_metadata":             load_spark_model_metadata(
                                             "rf_shock_model_spark", "RandomForestClassifier"
                                         ),
    })

    # ── Anomaly models (with CSV fallback) ────────────────────────────────
    for model_name in ANOMALY_MODELS:
        data[model_name] = load_model_data(model_name)

    # ── Explicit re-fetch for any anomaly key that is still None ──────────
    _explicit = [
        "anomaly_isolation_forest",
        "anomaly_lof",
        "anomaly_ocsvm",
        "anomaly_ewma",
        "anomaly_iqr",
        "anomaly_percentile",
        "anomaly_moving_avg_robust",
        "anomaly_prophet",
        "anomaly_lstm",
        "anomaly_dashboard",
        "anomaly_composite",
    ]
    for name in _explicit:
        if data.get(name) is None:
            data[name] = load_model_data(name)

    # ── shock_severity with CSV fallback ──────────────────────────────────
    if data.get("shock_severity") is None or (
        isinstance(data["shock_severity"], pd.DataFrame) and data["shock_severity"].empty
    ):
        data["shock_severity"] = load_model_data("shock_severity")

    # ── shock_global_country_rank with CSV fallback ───────────────────────
    if data.get("shock_global_country_rank") is None or (
        isinstance(data["shock_global_country_rank"], pd.DataFrame)
        and data["shock_global_country_rank"].empty
    ):
        data["shock_global_country_rank"] = load_model_data("shock_global_country_rank")

    # ── risk_dashboard_summary from CSV  (FIX-B: loaded once, not twice) ──
    risk_csv = os.path.join(CSV_PATH, "risk_dashboard_summary.csv")
    if os.path.exists(risk_csv):
        try:
            rds = pd.read_csv(risk_csv)
            if "country" in rds.columns:
                rds["country"] = rds["country"].apply(normalize_country)
            data["risk_dashboard_summary"] = rds
        except Exception as exc:
            st.warning(f"Could not load risk_dashboard_summary.csv: {exc!s:.80}")
            data["risk_dashboard_summary"] = None
    else:
        data["risk_dashboard_summary"] = None

    # ── RF probability extraction ─────────────────────────────────────────
    rf_pred = data.get("rf_predictions")
    if rf_pred is not None and isinstance(rf_pred, pd.DataFrame) and "probability" in rf_pred.columns:
        rf_pred = rf_pred.copy()
        rf_pred["probability"] = rf_pred["probability"].apply(extract_spark_vector_prob)
        data["rf_predictions"] = rf_pred

    # ── Yearly anomaly aggregation  (FIX-E: use .copy() to avoid mutating cache) ──
    yearly = data.get("combined_yearly_anomalies")
    if yearly is not None and isinstance(yearly, pd.DataFrame) and not yearly.empty:
        yearly = yearly.copy()
        flag_cols = [c for c in yearly.columns if c.endswith("_flag")]
        if flag_cols and "total_anomalies" not in yearly.columns:
            yearly["total_anomalies"] = yearly[flag_cols].sum(axis=1)
        data["combined_yearly_anomalies"] = yearly

        if "year" in yearly.columns and "total_anomalies" in yearly.columns:
            data["combined_yearly_summary"] = (
                yearly.groupby("year")["total_anomalies"]
                .sum()
                .reset_index()
                .sort_values("year")
            )

    # ── Country anomaly aggregation  (FIX-E: same .copy() guard) ─────────
    cca = data.get("combined_country_anomalies")
    if cca is not None and isinstance(cca, pd.DataFrame) and not cca.empty:
        cca = cca.copy()
        flag_cols = [c for c in cca.columns if c.endswith("_flag")]
        if flag_cols and "total_anomalies" not in cca.columns:
            cca["total_anomalies"] = cca[flag_cols].sum(axis=1)
        data["combined_country_anomalies"] = cca

        if "country" in cca.columns and "total_anomalies" in cca.columns:
            data["combined_country_summary"] = (
                cca.groupby("country")["total_anomalies"]
                .sum()
                .reset_index()
                .sort_values("total_anomalies", ascending=False)
            )

    # ── Summary log ───────────────────────────────────────────────────────
    loaded_n = sum(
        1 for v in data.values()
        if (isinstance(v, pd.DataFrame) and not v.empty)
        or (isinstance(v, str) and v)
        or (isinstance(v, dict) and v)
    )
    print(f"🚀 Loaded {loaded_n}/{len(data)} checkpoints")

    return data





# ══════════════════════════════════════════════════════════════════════════════
#  DATA STATUS
# ══════════════════════════════════════════════════════════════════════════════

def get_data_status(data: dict) -> dict[str, str]:

    status = {}
    for key, df in data.items():
        if df is None:
            status[key] = "❌ Missing"
        elif isinstance(df, pd.DataFrame):
            status[key] = f"✅ {len(df):,} rows" if not df.empty else "⚠️ Empty DataFrame"
        elif isinstance(df, str):
            status[key] = f"✅ {len(df):,} chars"
        elif isinstance(df, dict):
            status[key] = f"✅ dict ({len(df)} keys)"
        else:
            status[key] = "✅ Loaded"
    return status

