#!/usr/bin/env python3
"""create_composite_anomaly.py
Create composite anomaly checkpoint from IF + LOF + OCSVM + CP.

FIXES applied:
  - FIX-I  : original doc3 version used df.name (DataFrame has no .name attr) in the
              merge loop — replaced with explicit zip(methods, data_list)
  - FIX-II : flag convention: sklearn uses -1 for anomaly; converted to 0/1 correctly
             (original used `!= 0` which wrongly treats 1 as anomaly too for some models)
  - FIX-III: .map() alignment used when building composite from Series indexed by country,
             matching the same fix applied in tab4_ml.py (FIX-5/FIX-10)
  - FIX-IV : output part file name uses a stable, short name (no UUID needed for pandas)
  - FIX-V  : missing column guard — if a method produced no flag/score col the script
              no longer crashes; it logs a warning and continues with zeros
"""

from __future__ import annotations

import glob
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

CHECKPOINT_BASE = Path.cwd() / "pipeline_checkpoints"


# ══════════════════════════════════════════════════════════════════════════════
#  LOAD
# ══════════════════════════════════════════════════════════════════════════════

def load_checkpoint(name: str) -> pd.DataFrame | None:
    """Load a Spark-style parquet folder to a pandas DataFrame."""
    folder = CHECKPOINT_BASE / name
    if not folder.exists():
        print(f"  ❌ Missing: {folder}")
        return None

    parts = glob.glob(str(folder / "part-*.parquet"))
    if not parts:
        print(f"  ❌ No part-*.parquet files in {folder}")
        return None

    dfs = []
    for part in sorted(parts):
        try:
            dfs.append(pd.read_parquet(part))
        except Exception as exc:
            print(f"  ⚠️  Error reading {os.path.basename(part)}: {exc}")

    if not dfs:
        return None

    df = pd.concat(dfs, ignore_index=True)
    if "country" in df.columns:
        df["country"] = df["country"].astype(str).str.strip()

    print(f"  ✅ {name}: {df.shape}  ({df['country'].nunique() if 'country' in df.columns else '?'} countries)")
    print(f"     Columns: {list(df.columns)}")
    return df


# ══════════════════════════════════════════════════════════════════════════════
#  STANDARDISE
# ══════════════════════════════════════════════════════════════════════════════

_FLAG_CANDIDATES = [
    "{m}_flag", "{m}_anomaly_flag", "anomaly", "flag", "is_anomaly",
]
_SCORE_CANDIDATES = [
    "{m}_score", "{m}_anomaly_score", "anomaly_score", "score", "decision_function",
]


def _find_col(df: pd.DataFrame, templates: list[str], method: str) -> str | None:
    for tmpl in templates:
        cand = tmpl.replace("{m}", method)
        if cand in df.columns:
            return cand
    return None


def standardize_cols(df: pd.DataFrame, method: str) -> pd.DataFrame:
    """Return a new DataFrame with normalised `{method}_flag` and `{method}_score` columns.

    Flag convention (FIX-II):
      sklearn  : anomaly = -1, normal = 1  → convert to 1/0
      binary   : anomaly = 1,  normal = 0  → keep as-is
    """
    df_out = df.copy()
    flag_src  = _find_col(df, _FLAG_CANDIDATES,  method)
    score_src = _find_col(df, _SCORE_CANDIDATES, method)

    if flag_src:
        raw = pd.to_numeric(df[flag_src], errors="coerce").fillna(0)
        # sklearn uses -1 for anomaly; binary uses 1 for anomaly
        if raw.min() < 0:
            df_out[f"{method}_flag"] = (raw == -1).astype(int)
        else:
            df_out[f"{method}_flag"] = (raw != 0).astype(int)
        print(f"  {method}: flag  '{flag_src}' → '{method}_flag'  "
              f"(anomalies: {df_out[f'{method}_flag'].sum()})")
    else:
        df_out[f"{method}_flag"] = 0
        print(f"  {method}: ⚠️  no flag column found — using zeros")  # FIX-V

    if score_src:
        df_out[f"{method}_score"] = pd.to_numeric(df[score_src], errors="coerce").fillna(0)
        print(f"  {method}: score '{score_src}' → '{method}_score'")
    else:
        df_out[f"{method}_score"] = 0.0
        print(f"  {method}: ⚠️  no score column found — using zeros")  # FIX-V

    return df_out


# ══════════════════════════════════════════════════════════════════════════════
#  NORMALISE SCORES  (0-1 per method)
# ══════════════════════════════════════════════════════════════════════════════

def _norm(s: pd.Series) -> pd.Series:
    lo, hi = s.min(), s.max()
    return (s - lo) / (hi - lo) if hi > lo else s * 0.0


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    print("=" * 60)
    print("  Creating Composite Anomaly Checkpoint")
    print("=" * 60)

    # ── 1. Load ───────────────────────────────────────────────────────────
    print("\n[1] Loading checkpoints…")
    if_raw   = load_checkpoint("anomaly_isolation_forest")
    lof_raw  = load_checkpoint("anomaly_lof")
    ocsvm_raw= load_checkpoint("anomaly_ocsvm")
    cp_raw   = load_checkpoint("change_points")

    missing = [
        n for n, d in [
            ("anomaly_isolation_forest", if_raw),
            ("anomaly_lof",              lof_raw),
            ("anomaly_ocsvm",            ocsvm_raw),
            ("change_points",            cp_raw),
        ] if d is None
    ]
    if missing:
        print(f"\n❌ Cannot continue — missing checkpoints: {missing}")
        sys.exit(1)

    # ── 2. Standardise ───────────────────────────────────────────────────
    print("\n[2] Standardising columns…")
    methods   = ["if", "lof", "ocsvm", "cp"]
    raw_dfs   = [if_raw, lof_raw, ocsvm_raw, cp_raw]
    std_dfs   = [standardize_cols(df, m) for df, m in zip(raw_dfs, methods)]

    # ── 3. Merge on country (outer join) ─────────────────────────────────
    print("\n[3] Merging on country…")
    # Start from union of all countries
    all_countries = set()
    for df in std_dfs:
        if "country" in df.columns:
            all_countries |= set(df["country"].dropna().unique())

    merged = pd.DataFrame({"country": sorted(all_countries)})

    for df, method in zip(std_dfs, methods):   # FIX-I: explicit zip, no df.name
        flag_col  = f"{method}_flag"
        score_col = f"{method}_score"

        if "country" not in df.columns:
            print(f"  ⚠️  {method}: no 'country' column — filling with zeros")
            merged[flag_col]  = 0
            merged[score_col] = 0.0
            continue

        # Aggregate to one row per country (take max flag, mean score)
        agg = (
            df.groupby("country")
            .agg({flag_col: "max", score_col: "mean"})
            .reset_index()
        )
        merged = merged.merge(agg, on="country", how="left")

    # Fill any remaining NaN
    flag_cols  = [f"{m}_flag"  for m in methods]
    score_cols = [f"{m}_score" for m in methods]
    for col in flag_cols + score_cols:
        merged[col] = merged[col].fillna(0)

    # ── 4. Composite calculations  (FIX-III: use .map() for alignment) ───
    print("\n[4] Computing composite scores…")

    for method, score_col in zip(methods, score_cols):
        norm_col = f"{method}_norm"
        score_series = merged.set_index("country")[score_col]
        merged[norm_col] = merged["country"].map(_norm(score_series))

    norm_cols = [f"{m}_norm" for m in methods]
    merged["composite_flag"]      = (merged[flag_cols].sum(axis=1) > 0).astype(int)
    merged["composite_score"]     = merged[flag_cols].sum(axis=1)          # 0–4 count
    merged["composite_score_avg"] = merged[norm_cols].mean(axis=1)         # 0–1 avg

    print(f"\n  Merged shape  : {merged.shape}")
    print(f"  Countries     : {len(merged)}")
    print(f"  Columns       : {list(merged.columns)}")
    print("\n  Flag counts:")
    print(merged[flag_cols + ["composite_flag"]].sum().to_string())
    print("\n  High-risk (composite_score >= 2):",
          (merged["composite_score"] >= 2).sum())

    # ── 5. Save  (FIX-IV: stable file name, no random UUID) ──────────────
    print("\n[5] Saving checkpoint…")
    out_dir  = CHECKPOINT_BASE / "anomaly_composite"
    out_dir.mkdir(parents=True, exist_ok=True)

    part_file = out_dir / "part-00000-composite.snappy.parquet"
    merged.to_parquet(part_file, index=False)
    (out_dir / "_SUCCESS").write_text("")

    print(f"\n✅ Saved → {out_dir}")
    print(f"   File  : {part_file.name}")
    print(f"   Rows  : {len(merged)}")
    print(f"   Cols  : {list(merged.columns)}")
    print("\nVerify with:")
    print(f"  python -c \"import pandas as pd; df=pd.read_parquet('{part_file}'); "
          f"print(df.shape); print(df.head(3))\"")


if __name__ == "__main__":
    main()