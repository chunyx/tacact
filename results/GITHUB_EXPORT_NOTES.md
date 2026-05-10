# GitHub Export Notes

This repository export intentionally keeps source code, experiment summaries, plots, tables, protocols, and logs needed for paper writing and reproducibility.

The following generated artifacts are not tracked in Git because they are large or reproducible:

- `.npy` cache files from `.cache_tacact_n80_weighted/`
- model checkpoint files (`*.pt`)
- `results/artifacts/main_parallel_5gpu.log` because it is larger than GitHub's normal file limit
- `results/main/outputs_main_experiment_rerun_20260506_185627/predictions_merged.csv` because it is larger than GitHub's normal file limit

The original TacAct `.xlsx` dataset is also not included here. Set `DATA_ROOT` to the local dataset path when reproducing experiments on another machine.
