# LPGSupplyAnalysis Debug: Fix tab4_ml.py Pylance Errors
*Status: 4/6 complete* | Target: Zero Pylance errors

✅ **Phase 1**: Add missing imports **✓ COMPLETE**  
✅ **Phase 2**: Create TODO.md **✓ COMPLETE**  
✅ **Phase 3**: Hardcode ML params (`scaler_type=standard` etc.) **✓ COMPLETE**  
✅ **Phase 4**: Move standalone ML demo inside `tab_comb:` **✓ COMPLETE**  
⏳ **Phase 5**: Clean up unused imports/vars  
⏳ **Phase 6**: Test run + completion

✅ **Phase 1**: Add missing imports (Path, sklearn) **✓ COMPLETE**  
✅ **Phase 2**: Create this TODO.md **✓ COMPLETE**  
🔄 **Phase 3**: Test imports → check remaining errors **RUNNING**  
⏳ **Phase 4**: Fix data scoping (data, rf_pred, clusters vars)  
⏳ **Phase 5**: Organize standalone ML code into render()  
⏳ **Phase 6**: Full test + attempt_completion  

**Next**: Phase 3 verification + hardcoded params (per user: skip sidebar)


## Approved Plan (Modified: Skip sidebar widgets)
✅ **Phase 1**: Add missing imports (Path, sklearn)  
✅ **Phase 2**: Create this TODO.md  
🔄 **Phase 3**: Test imports → check remaining errors  
⏳ **Phase 4**: Fix data scoping (data, rf_pred, clusters vars)  
⏳ **Phase 5**: Organize standalone ML code into render()  
⏳ **Phase 6**: Full test + attempt_completion  

---

## Next Action
**Apply Phase 1 imports** to resolve 12 critical sklearn/Path errors.

**Runtime check**: `streamlit run lpg_dashboard/app.py` → Tab 4 (no crashes expected post-imports)

**Estimated**: 90 seconds to Phase 3

