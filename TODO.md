# Tab4 ML Dashboard Fixes - EWMA & Data Flow
Current status: ✅ Plan approved | CWD: c:/Users/priti/Documents/LPGSupplyAnalysis
Streamlit: 🟢 `streamlit run lpg_dashboard/app.py` running

## Approved Plan (Summary)
**Goal**: Fix Tab4 (ML tab) empty tables/charts → show data for EWMA + standardize flags
**Files**: data_loader.py, tab4_ml.py, helpers.py
**Approach**: Debug → fix loading → standardize flags → robust fallbacks → cleanup

## Implementation Steps (Execute sequentially)

### ✅ Step 0: Create this TODO.md [DONE]

### ✅ Step 1: Enhanced debug logging in data_loader.py [DONE]
**Results**: EWMA loads perfectly! 5790 rows × 28 cols, **1619 anomalies** (`ewma_anomaly_flag=1`), graphs working ✅
**Edit** `lpg_dashboard/utils/data_loader.py`
```
- Add EWMA-specific st.info() in load_spark_parquet() after df=concat()
- Log: file count, shape pre/post normalize_country(), columns list
- Skip normalize if no 'country' col or df empty
```
**Test**: Cache clear → Tab4 → EWMA → check debug output

### ⏳ Step 2: Add tab4_ml.py EWMA debug expander [PENDING]
**Edit** `lpg_dashboard/tabs/tab4_ml.py` EWMA tab before `if not ok(ewma):`
```
with st.expander("🐛 EWMA DEBUG [TEMP]"):
    st.write(f"ewma: {type(ewma)} | shape: {getattr(ewma,'shape','N/A')}")
    if ewma is not None: 
        st.write("Columns:", list(ewma.columns))
        st.write("Sample:", ewma[['country','year']].head() if 'country' in ewma.columns else "No country col")
        st.write("Flag cols?", [c for c in ewma.columns if 'flag' in c.lower()])
```
**Test**: Navigate Tab4→Temporal→EWMA → expand → screenshot/copy output

### ⏳ Step 3: Flag standardization utility [PENDING]
**Add to** `lpg_dashboard/utils/helpers.py`
```
def standardize_flags(df, flag_candidates=None):
    """Normalize anomaly flags to 1=True across models"""
    if df is None or df.empty: return df
    candidates = flag_candidates or ['anomaly_flag','flag','shock_flag']
    for cand in candidates:
        if cand in df.columns:
            col = df[cand]
            if col.dtype == 'bool': continue  # already good
            # Try common conventions
            if (col == -1).any(): 
                df[f'{cand}_std'] = (col == -1).astype(int)
            elif (col == 1).any() or col.nunique() <= 2:
                df[f'{cand}_std'] = (col == 1).astype(int)
            else:
                df[f'{cand}_std'] = (col > col.quantile(0.95)).astype(int)
            return df
    return df  # no flags found
```
**Update** tab4_ml.py → `ewma = helpers.standardize_flags(ewma)`

### ⏳ Step 4: Robust table/chart fallbacks [PENDING]
**tab4_ml.py**: Replace brittle `flag_col = fcol(...)` logic with:
```
flag_col = detect_anomaly_col(ewma, ['ewma_anomaly_flag','shock_flag','anomaly','flag'])
if flag_col: anomalies = ewma[ewma[f'{flag_col}_std']==1]
else: top = ewma.nlargest(20, score_col or 'normalized_deviation')
```

### ⏳ Step 5: User testing & validation [PENDING]
```
1. streamlit cache clear
2. streamlit run lpg_dashboard/app.py --server.headless=true (background)
3. Navigate Tab4 → Temporal → EWMA → verify:
   - Table shows (not empty_html)
   - Country selector populated
   - Charts render
   - Debug expander shows data flow
```

### ⏳ Step 6: Production cleanup [PENDING]
```
- Remove 🐛 debug expanders
- Remove st.info() debug prints  
- Final tests across all Tab4 models
- Update TODO.md → [ALL DONE]
```

### ⏳ Step 7: attempt_completion [PENDING]

## Quick Debug Commands
```bash
# Clear cache
streamlit cache clear

# Test EWMA load directly
python -c "
import pandas as pd
df = pd.read_parquet('pipeline_checkpoints/anomaly_ewma/part-*.parquet')
print('Shape:', df.shape)
print('Cols:', list(df.columns))
print('Countries sample:', df['country'].dropna().unique()[:5])
print('Flag cols:', [c for c in df.columns if 'flag' in c.lower()])
print(df.head())
"

# Test CSV fallback
python -c "
import pandas as pd
df = pd.read_csv('outputs/anomalies/anomaly_ewma.csv')
print('CSV shape:', df.shape, 'cols:', list(df.columns[:10]))
"
```

## Success Criteria ✅
```
Tab4 EWMA subtab:
• [ ] Table shows top countries (not empty_html)
• [ ] Country selector >0 options  
• [ ] Charts render (scatter/bar)
• [ ] Debug output confirms data flow
• [ ] No console errors
• [ ] Cache works (no re-logging)
```

**Execute Step 1 first → report debug output → proceed.**

