# Cleanup Report (Conservative / Traceable)

## Scope and Principles
- Safety-first cleanup: preserve reproducibility, benchmark logic, best-config assets, and key outputs.
- Audit-first, then execute: only removed clearly safe cache artifacts; uncertain items were archived, not deleted.
- No destructive git operations; no core experiment semantics changed.

## Must Keep (kept in place)
- Core code and pipeline:
  - `benchmark_data_loading/`
  - `benchmark_common.py`
  - `data.py`
  - `models.py`
  - `utils.py`
- Active run scripts and merge/watch scripts:
  - `run_main_*.sh`, `run_and_watch_*.sh`, `merge_*.sh`, `watch_main_9models_5gpu.py`
- Reproducibility-critical outputs/configs:
  - `outputs_main_9models_9gpu_fastio/`
  - `outputs_main_transformer_1gpu_from_phase2/`
  - `outputs_hpo_transformer_gpu5689_retry/`
  - `outputs_best_configs_combined/`
  - `outputs_plots/`
  - split-audit related code and logs

## Safe To Clean (deleted)
- Python bytecode cache directories (regenerable, no experiment value):
  - `__pycache__/`
  - `benchmark_data_loading/__pycache__/`

## Archive Instead of Delete (uncertain but likely non-critical)
- Moved to `archive_unused/` for manual review:
  - `outputs_final_bundle_20260408_215635/`
  - `outputs_hpo_gru_only/`
  - `outputs_hpo_transformer_gpu5689_fair/`
  - `tacact_results/`
- Reason: likely historical/alternative outputs; kept recoverable to avoid accidental data loss.

## Code Quality Touch-ups (non-semantic)
- Updated `plot_model_curves_vs_epoch.py`:
  - Introduced constants for validation y-axis limits.
  - Added CLI args `--val-ymin` and `--val-ymax`.
  - Reused args in plotting calls to reduce hardcoded values.
- Impact: improves readability/configurability; does not alter training logic.

## Verification / Sanity Checks
- Post-cleanup directory checks: passed.
- Basic script syntax/import sanity for modified plotting path: passed previously in this session.
- No linter errors observed on modified files in this round.

## Files Generated for Traceability
- `cleanup_report.md`
- `cleanup_deleted.txt`
- `cleanup_archived.txt`

## Items intentionally NOT touched (possible future cleanup candidates)
- `.cache_tacact_n80_weighted/` (large but useful for speed; can be rebuilt)
- Existing experiment logs/status files under active output roots
- Any benchmark CSV/PNG/PDF under active output trees
