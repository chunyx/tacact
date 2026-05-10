# Results Index

This directory groups generated experiment outputs so the project root stays focused on code and runnable scripts.

## Current main result

- `main/outputs_main_experiment_rerun_20260506_185627/`
  - Recommended result set for the latest main experiment.
  - Contains 10-seed merged metrics, summary CSVs, per-class metrics, confusion matrix, runtime summaries, and benchmark plots.

## Main experiment history

- `main/outputs_main_experiment_final_20260501_023752/`
  - Earlier main experiment result set.
- `archive/outputs_main_experiment_parallel_20260429_150413/`
  - Older parallel main experiment outputs.
- `archive/outputs_main_experiment_rerun_20260506_184642/`
  - Earlier rerun attempt.
- `archive/OUTPUT_DIR/`
  - Temporary/default output directory kept for traceability.

## HPO outputs

- `hpo/outputs_hpo_phase12_full_20260430_183024/`
- `hpo/outputs_hpo_tcn_diagnostic_20260430_084706/`
- `hpo/outputs_hpo_tcn_phase12_20260430_022523/`
- `hpo/outputs_hpo_three_stage_20260428_061937/`
- `hpo/outputs_hpo_transformer_phase12_20260501_012648/`
- `hpo/outputs_hpo_transformer_tuned_20260501_015818/`

## Phase 2 benchmark runs

- `phase2_benchmarks/bench_phase2_a_baseline/`
- `phase2_benchmarks/bench_phase2_b_loader8/`
- `phase2_benchmarks/bench_phase2_c_loader12/`

## Diagnostics

- `diagnostics/outputs/`
  - Targeted LeNet/LSTM diagnostics and comparison outputs.
- `diagnostics/outputs_debug_transformer_single_batch_20260501_123419/`
  - Transformer single-batch debug run.

## Paper bundle and artifacts

- `paper_bundle/outputs_main_paper_bundle_20260421_005345/`
  - Earlier paper-oriented bundle.
- `artifacts/`
  - Logs, cleanup reports, and standalone archives kept outside the project root.

## Not moved

The `.cache_tacact*` directories were left in the project root to avoid breaking cached data paths used by scripts.
