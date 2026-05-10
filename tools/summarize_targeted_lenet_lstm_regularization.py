#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


def _safe_read_csv(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception as exc:
        print(f"[WARN] Failed to read CSV: {path} ({exc})")
        return None


def _macro_f1_from_predictions(df: pd.DataFrame) -> float:
    if df.empty:
        return float("nan")
    labels = sorted(set(df["true_label"].tolist()) | set(df["pred_label"].tolist()))
    f1s: List[float] = []
    for label in labels:
        tp = int(((df["true_label"] == label) & (df["pred_label"] == label)).sum())
        fp = int(((df["true_label"] != label) & (df["pred_label"] == label)).sum())
        fn = int(((df["true_label"] == label) & (df["pred_label"] != label)).sum())
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2.0 * precision * recall / (precision + recall)
        f1s.append(float(f1))
    return float(np.mean(f1s)) if f1s else float("nan")


def _find_subject_run_dir(variant_dir: Path) -> Optional[Path]:
    candidates = sorted(variant_dir.glob("subject_seed*"))
    if candidates:
        return candidates[0]
    if (variant_dir / "metrics.csv").exists():
        return variant_dir
    nested = sorted(variant_dir.glob("**/subject_seed*"))
    return nested[0] if nested else None


def _variant_name_from_dir(variant_dir: Path) -> str:
    return variant_dir.name


def _extract_variant_summary(variant_dir: Path) -> Optional[Dict[str, Any]]:
    run_dir = _find_subject_run_dir(variant_dir)
    if run_dir is None:
        print(f"[WARN] No subject_seed* directory found under {variant_dir}")
        return None

    metrics_df = _safe_read_csv(run_dir / "metrics.csv")
    per_class_df = _safe_read_csv(run_dir / "per_class_metrics.csv")
    preds_df = _safe_read_csv(run_dir / "predictions.csv")
    final_df = _safe_read_csv(run_dir / "final_split_metrics.csv")
    run_cfg_path = run_dir / "run_config.json"

    if metrics_df is None or metrics_df.empty or preds_df is None or preds_df.empty:
        print(f"[WARN] Missing core result files under {run_dir}")
        return None

    metrics_row = metrics_df.iloc[0].to_dict()
    final_row = final_df.iloc[0].to_dict() if final_df is not None and not final_df.empty else {}

    subject26 = preds_df[preds_df["subject_id"] == 26].copy() if "subject_id" in preds_df.columns else pd.DataFrame()
    subject26_acc = float(subject26["correct"].mean()) if not subject26.empty and "correct" in subject26.columns else float("nan")
    subject26_f1 = _macro_f1_from_predictions(subject26) if not subject26.empty else float("nan")

    class3_f1 = float("nan")
    class6_f1 = float("nan")
    if per_class_df is not None and not per_class_df.empty:
        for cls_id, key in [(3, "class3"), (6, "class6")]:
            match = per_class_df[per_class_df["class_id"] == cls_id]
            if not match.empty:
                if key == "class3":
                    class3_f1 = float(match.iloc[0]["f1"])
                else:
                    class6_f1 = float(match.iloc[0]["f1"])

    class_3_to_6 = int(((preds_df["true_label"] == 3) & (preds_df["pred_label"] == 6)).sum())
    class_6_to_3 = int(((preds_df["true_label"] == 6) & (preds_df["pred_label"] == 3)).sum())

    run_cfg: Dict[str, Any] = {}
    if run_cfg_path.exists():
        try:
            run_cfg = json.loads(run_cfg_path.read_text(encoding="utf-8"))
        except Exception as exc:
            print(f"[WARN] Failed to read run config: {run_cfg_path} ({exc})")

    return {
        "variant_name": _variant_name_from_dir(variant_dir),
        "seed": int(final_row.get("seed", preds_df["seed"].iloc[0] if "seed" in preds_df.columns else -1)),
        "label_smoothing": float(run_cfg.get("label_smoothing", 0.0)) if run_cfg else 0.0,
        "weight_decay_override": run_cfg.get("weight_decay_override", None),
        "test_accuracy": float(metrics_row.get("accuracy", np.nan)),
        "test_macro_f1": float(metrics_row.get("macro_f1", np.nan)),
        "test_macro_precision": float(metrics_row.get("macro_precision", np.nan)),
        "test_macro_recall": float(metrics_row.get("macro_recall", np.nan)),
        "class_3_f1": class3_f1,
        "class_6_f1": class6_f1,
        "class_3_to_6_errors": class_3_to_6,
        "class_6_to_3_errors": class_6_to_3,
        "subject_26_accuracy": subject26_acc,
        "subject_26_macro_f1": subject26_f1,
        "inference_ms": float(metrics_row.get("inference_ms", final_row.get("inference_ms", np.nan))),
        "params_m": float(metrics_row.get("params_m", final_row.get("params_m", np.nan))),
        "training_seconds": float(metrics_row.get("training_seconds", final_row.get("train_time_sec", np.nan))),
        "best_epoch": float(metrics_row.get("best_epoch", np.nan)),
        "best_val_loss": float(metrics_row.get("best_val_loss", np.nan)),
        "best_val_f1": float(metrics_row.get("best_val_f1", np.nan)),
        "run_dir": str(run_dir),
    }


def _write_markdown_table(df: pd.DataFrame, out_path: Path, title: str) -> None:
    with out_path.open("w", encoding="utf-8") as f:
        f.write(f"# {title}\n\n")
        if df.empty:
            f.write("No valid runs were found.\n")
            return
        cols = list(df.columns)
        f.write("| " + " | ".join(cols) + " |\n")
        f.write("| " + " | ".join(["---"] * len(cols)) + " |\n")
        for _, row in df.iterrows():
            vals = []
            for col in cols:
                val = row[col]
                if pd.isna(val):
                    vals.append("")
                else:
                    vals.append(str(val))
            f.write("| " + " | ".join(vals) + " |\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs_root", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--summary_prefix", type=str, default="regularization_quick")
    parser.add_argument("--title", type=str, default="Regularization Quick Comparison")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_roots = [p for p in sorted(args.runs_root.iterdir()) if p.is_dir()]
    rows: List[Dict[str, Any]] = []
    for variant_dir in run_roots:
        row = _extract_variant_summary(variant_dir)
        if row is not None:
            rows.append(row)

    summary_df = pd.DataFrame(rows)
    if not summary_df.empty:
        summary_df = summary_df.sort_values(["seed", "variant_name"]).reset_index(drop=True)
    summary_csv = args.output_dir / f"{args.summary_prefix}_summary.csv"
    summary_df.to_csv(summary_csv, index=False)
    md_path = args.output_dir / f"{args.summary_prefix}_summary.md"
    _write_markdown_table(summary_df, md_path, args.title)

    confusion_cols = [
        "variant_name",
        "seed",
        "class_3_to_6_errors",
        "class_6_to_3_errors",
        "best_val_f1",
        "test_macro_f1",
    ]
    per_class_cols = [
        "variant_name",
        "seed",
        "class_3_f1",
        "class_6_f1",
        "test_macro_f1",
    ]
    subject26_cols = [
        "variant_name",
        "seed",
        "subject_26_accuracy",
        "subject_26_macro_f1",
        "test_macro_f1",
    ]
    for cols, fname in [
        (confusion_cols, "class3_class6_confusion_comparison.csv"),
        (per_class_cols, "per_class_f1_comparison.csv"),
        (subject26_cols, "subject26_comparison.csv"),
    ]:
        out_df = summary_df[cols].copy() if not summary_df.empty else pd.DataFrame(columns=cols)
        out_df.to_csv(args.output_dir / fname, index=False)

    print(f"Saved summary to: {summary_csv}")
    print(f"Saved markdown to: {md_path}")


if __name__ == "__main__":
    main()
