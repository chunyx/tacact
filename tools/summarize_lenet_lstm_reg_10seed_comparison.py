#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

from summarize_motioninput_reg_10seed_comparison import (
    _build_aggregate_table,
    _build_confusion_pair_table,
    _build_per_class_table,
    _build_subject_wise_table,
    _extract_run_summary,
    _find_subject_run_dir,
    _load_original_runs,
    _save_figures,
    _make_seed_level_comparison,
)

import pandas as pd


def _load_reg_runs(root: Path) -> Dict[int, Path]:
    mapping: Dict[int, Path] = {}
    for sub in sorted(root.glob("reg_seed*")):
        run_dir = _find_subject_run_dir(sub)
        if run_dir is None:
            continue
        try:
            seed = int(run_dir.name.replace("subject_seed", ""))
        except Exception:
            continue
        mapping[seed] = run_dir
    return mapping


def _write_reg_report(
    seed_df: pd.DataFrame,
    agg_df: pd.DataFrame,
    per_class_df: pd.DataFrame,
    conf_df: pd.DataFrame,
    subject_df: pd.DataFrame,
    out_path: Path,
    seeds: list[int],
) -> None:
    agg = agg_df.set_index("metric")
    improved_better = float(agg.loc["test_macro_f1", "improved_mean"]) > float(agg.loc["test_macro_f1", "original_mean"])
    with out_path.open("w", encoding="utf-8") as f:
        f.write("# Regularization-only LeNet_LSTM 10-seed Report\n\n")
        f.write("## A. Purpose\n")
        f.write("This is a post-hoc targeted improvement validation for regularization only, not part of the original fair multi-model comparison.\n\n")
        f.write("## B. Compared models\n")
        f.write("- Original LeNet_LSTM\n")
        f.write("- LeNet_LSTM + label_smoothing=0.05 + weight_decay=3e-4\n\n")
        f.write("## C. Seed protocol\n")
        f.write(f"- Seeds used: {seeds}\n")
        f.write("- Same subject-independent split seeds were used for both models.\n\n")
        f.write("## D. Overall mean ± std comparison\n")
        for metric, label in [("test_accuracy", "Accuracy"), ("test_macro_f1", "Macro-F1"), ("inference_ms", "Inference time (ms)"), ("params_m", "Params (M)")]:
            f.write(
                f"- {label}: original {agg.loc[metric, 'original_mean']:.6f} ± {agg.loc[metric, 'original_std']:.6f}, "
                f"regularized {agg.loc[metric, 'improved_mean']:.6f} ± {agg.loc[metric, 'improved_std']:.6f}\n"
            )
        f.write("\n## E. Hold / Static Drag comparison\n")
        for metric, label in [("class_3_f1", "class_3 F1 (Hold)"), ("class_6_f1", "class_6 F1 (Static Drag)"), ("class_3_to_6_errors", "class_3->6 errors"), ("class_6_to_3_errors", "class_6->3 errors")]:
            f.write(
                f"- {label}: original {agg.loc[metric, 'original_mean']:.6f}, regularized {agg.loc[metric, 'improved_mean']:.6f}, "
                f"delta {agg.loc[metric, 'delta_mean']:.6f}\n"
            )
        f.write("\n## F. Per-class changes\n")
        top_improve = per_class_df.sort_values("delta_mean_f1", ascending=False).head(5)
        top_drop = per_class_df.sort_values("delta_mean_f1", ascending=True).head(5)
        f.write("- Most improved classes:\n")
        for _, row in top_improve.iterrows():
            f.write(f"  - {row['class_name']}: delta_mean_f1={row['delta_mean_f1']:.6f}\n")
        f.write("- Most degraded classes:\n")
        for _, row in top_drop.iterrows():
            f.write(f"  - {row['class_name']}: delta_mean_f1={row['delta_mean_f1']:.6f}\n")
        f.write("\n## G. Subject-wise comparison\n")
        if subject_df.empty:
            f.write("- Subject-wise metrics were not available.\n")
        else:
            s26 = subject_df[subject_df["subject_id"] == 26]
            if not s26.empty:
                row = s26.iloc[0]
                f.write(
                    f"- subject 26 macro-F1: original {row['original_macro_f1_mean']:.6f}, "
                    f"regularized {row['improved_macro_f1_mean']:.6f}\n"
                )
        f.write("\n## H. Interpretation\n")
        f.write("This comparison tests whether moderate regularization alone improves subject-independent tactile recognition without changing the original LeNet_LSTM architecture.\n")
        f.write("\n## I. Recommendation\n")
        if improved_better:
            f.write("The regularized model shows a higher mean Macro-F1 with negligible parameter overhead, so it is a useful ablation and a viable lightweight improvement path.\n")
        else:
            f.write("The regularized model does not show a stable mean Macro-F1 gain, so it should be reported as an ablation rather than the main improved variant.\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize original vs regularization-only LeNet_LSTM 10-seed comparison.")
    parser.add_argument("--original_runs_csv", required=True)
    parser.add_argument("--improved_runs_root", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    output_root = Path(args.output_dir)
    summary_dir = output_root / "summary"
    fig_dir = output_root / "figures"
    summary_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    original_df, original_map = _load_original_runs(Path(args.original_runs_csv))
    improved_map = _load_reg_runs(Path(args.improved_runs_root))

    original_rows = {}
    improved_rows = {}

    for seed, run_dir in original_map.items():
        row = _extract_run_summary(run_dir, "Original LeNet_LSTM")
        if row is not None:
            original_rows[seed] = row

    for seed, run_dir in improved_map.items():
        row = _extract_run_summary(run_dir, "LeNet_LSTM + LS0.05 + WD3e-4")
        if row is not None:
            improved_rows[seed] = row

    common_seeds = sorted(set(original_rows) & set(improved_rows))
    original_rows = {k: v for k, v in original_rows.items() if k in common_seeds}
    improved_rows = {k: v for k, v in improved_rows.items() if k in common_seeds}

    seed_df = _make_seed_level_comparison(original_rows, improved_rows)
    agg_df = _build_aggregate_table(original_rows, improved_rows)
    per_class_df = _build_per_class_table(original_rows, improved_rows)
    conf_df = _build_confusion_pair_table(original_rows, improved_rows)
    subj_df = _build_subject_wise_table(original_rows, improved_rows)

    seed_df.to_csv(summary_dir / "seed_level_comparison.csv", index=False)
    agg_df.to_csv(summary_dir / "aggregate_mean_std.csv", index=False)
    per_class_df.to_csv(summary_dir / "per_class_f1_mean_std_comparison.csv", index=False)
    conf_df.to_csv(summary_dir / "confusion_pair_comparison.csv", index=False)
    subj_df.to_csv(summary_dir / "subject_wise_mean_std_comparison.csv", index=False)

    _save_figures(seed_df, agg_df, per_class_df, conf_df, fig_dir)
    _write_reg_report(
        seed_df=seed_df,
        agg_df=agg_df,
        per_class_df=per_class_df,
        conf_df=conf_df,
        subject_df=subj_df,
        out_path=output_root / "lenet_lstm_reg_10seed_report.md",
        seeds=common_seeds,
    )

    out_df = original_df.copy()
    out_df.to_csv(output_root / "original_lenet_lstm_runs_found.csv", index=False)

    print(f"Regularization-only 10-seed summary saved under: {output_root}")


if __name__ == "__main__":
    main()
