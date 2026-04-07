import pandas as pd
import glob
import os
from pathlib import Path

CHECKPOINT_BASE = r'pipeline_checkpoints'

ANOMALY_KEYWORDS = ['anomaly', 'flag', 'score', 'outlier', 'shock', 'alert']


def load_parquet_folder(folder: str) -> pd.DataFrame | None:
    """
    Load a Spark-style parquet folder (part-*.parquet) or a single .parquet file.
    Returns None if nothing could be loaded.
    """
    # ── Case 1: Spark folder with part files ──────────────────────────────
    if os.path.isdir(folder):
        parts = [
            f for f in glob.glob(os.path.join(folder, "part-*.parquet"))
            if not f.endswith('.crc')
        ]
        if not parts:
            # Some pipelines write plain *.parquet without 'part-' prefix
            parts = [
                f for f in glob.glob(os.path.join(folder, "*.parquet"))
                if not f.endswith('.crc')
            ]
        if not parts:
            return None

        dfs = []
        for part in sorted(parts):
            try:
                dfs.append(pd.read_parquet(part))
            except Exception as exc:
                print(f"    ⚠️  Could not read {os.path.basename(part)}: {exc}")

        return pd.concat(dfs, ignore_index=True) if dfs else None

    # ── Case 2: Single .parquet file ─────────────────────────────────────
    if os.path.isfile(folder) and folder.endswith('.parquet'):
        try:
            return pd.read_parquet(folder)
        except Exception as exc:
            print(f"    ⚠️  Could not read {folder}: {exc}")
            return None

    return None


def get_anomaly_cols(df: pd.DataFrame) -> list[str]:
    return [
        col for col in df.columns
        if any(k in col.lower() for k in ANOMALY_KEYWORDS)
    ]


def print_flag_distribution(df: pd.DataFrame, col: str, indent: str = "    ") -> None:
    """Print value counts + percentage for a column."""
    counts = df[col].value_counts(dropna=False)
    total  = len(df)
    dtype  = df[col].dtype
    null_n = df[col].isna().sum()

    print(f"{indent}{col!r}  [dtype={dtype}, nulls={null_n}]")
    for val, cnt in counts.items():
        bar = "█" * min(int(cnt / total * 30), 30)
        print(f"{indent}    {str(val):>8}  →  {cnt:>6,}  ({cnt/total:5.1%})  {bar}")


def inspect_checkpoint(name: str) -> dict | None:
    """
    Inspect a single checkpoint folder or .parquet file.
    Returns a summary dict for the final report, or None on failure.
    """
    # Resolve path: try as folder first, then with .parquet extension
    candidates = [
        os.path.join(CHECKPOINT_BASE, name),
        os.path.join(CHECKPOINT_BASE, f"{name}.parquet"),
        name,  # absolute path support
    ]
    resolved = next((c for c in candidates if os.path.exists(c)), None)

    print(f"\n{'═'*80}")
    print(f"🔍  {name.upper()}")

    if resolved is None:
        print(f"❌  Not found — searched:")
        for c in candidates:
            print(f"    {c}")
        return None

    print(f"📁  Path : {resolved}")

    df = load_parquet_folder(resolved)

    if df is None or df.empty:
        print("❌  No data loaded (all part files failed or folder is empty).")
        return None

    rows, cols = df.shape
    print(f"📊  Shape: {rows:,} rows  ×  {cols} columns")

    # ── Country info ──────────────────────────────────────────────────────
    if 'country' in df.columns:
        countries = df['country'].dropna().unique()
        preview   = sorted(countries)[:5]
        print(f"🌍  Countries: {preview}{'...' if len(countries) > 5 else ''}  ({len(countries):,} total)")
    else:
        print("🌍  Countries: ⚠️  no 'country' column found")

    # ── Year info ─────────────────────────────────────────────────────────
    for yr_col in ('year', 'Year', 'yr'):
        if yr_col in df.columns:
            yr_min = df[yr_col].min()
            yr_max = df[yr_col].max()
            print(f"📅  Years   : {yr_min} – {yr_max}  ({df[yr_col].nunique()} unique)")
            break

    # ── Column overview ───────────────────────────────────────────────────
    print(f"\n🏷️  All columns ({cols}):")
    for chunk_start in range(0, len(df.columns), 6):
        chunk = list(df.columns)[chunk_start:chunk_start + 6]
        print("    " + ",  ".join(chunk))

    # ── Anomaly columns ───────────────────────────────────────────────────
    anomaly_cols = get_anomaly_cols(df)
    if anomaly_cols:
        print(f"\n🚩  Anomaly / score columns ({len(anomaly_cols)}):")
        for col in anomaly_cols:
            print_flag_distribution(df, col)
    else:
        print("\n🚩  No anomaly/flag/score columns detected.")

    # ── Null summary (only columns with nulls) ────────────────────────────
    null_counts = df.isnull().sum()
    null_counts = null_counts[null_counts > 0]
    if not null_counts.empty:
        print(f"\n⚠️  Columns with NULLs:")
        for col, n in null_counts.items():
            print(f"    {col!r}: {n:,} nulls  ({n/rows:.1%})")

    # ── Sample rows ───────────────────────────────────────────────────────
    print(f"\n📋  Sample (first 3 rows):")
    with pd.option_context('display.max_columns', None, 'display.width', 200):
        print(df.head(3).to_string(index=False))

    print(f"{'─'*80}")

    return {
        "checkpoint": name,
        "rows":       rows,
        "cols":       cols,
        "countries":  len(df['country'].unique()) if 'country' in df.columns else "n/a",
        "anomaly_cols": len(anomaly_cols),
        "nulls_cols": len(null_counts),
        "status":     "✅",
    }


def print_summary(results: list[dict]) -> None:
    """Print a compact summary table of all inspected checkpoints."""
    print(f"\n{'═'*80}")
    print("📑  INSPECTION SUMMARY")
    print(f"{'═'*80}")
    header = f"{'Checkpoint':<36} {'Status':^6} {'Rows':>8} {'Cols':>5} {'Countries':>10} {'AnomCols':>9} {'NullCols':>9}"
    print(header)
    print("─" * 80)
    for r in results:
        if r is None:
            continue
        print(
            f"{r['checkpoint']:<36} "
            f"{r['status']:^6} "
            f"{r['rows']:>8,} "
            f"{r['cols']:>5} "
            f"{str(r['countries']):>10} "
            f"{r['anomaly_cols']:>9} "
            f"{r['nulls_cols']:>9}"
        )
    print("─" * 80)
    ok  = sum(1 for r in results if r and r['status'] == '✅')
    bad = len(results) - ok
    print(f"  Loaded: {ok}   Failed: {bad}   Total: {len(results)}")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    CHECKPOINTS = [
        'anomaly_isolation_forest',
        'anomaly_lof',
        'anomaly_ocsvm',
        'anomaly_dashboard',
    ]

    results = [inspect_checkpoint(name) for name in CHECKPOINTS]
    print_summary([r for r in results])  # None entries skipped inside