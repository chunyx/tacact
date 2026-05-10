#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
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
    if df is None or df.empty:
        return float("nan")
    labels = sorted(set(df["true_label"].tolist()) | set(df["pred_label"].tolist()))
    f1s: List[float] = []
    for label in labels:
        tp = int(((df["true_label"] == label) & (df["pred_label"] == label)).sum())
        fp = int(((df["true_label"] != label) & (df["pred_label"] == label)).sum())
        fn = int(((df["true_label"] == label) & (df["pred_label"] != label)).sum())
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        f1 = 0.0 if precision + recall == 0 else 2.0 * precision * recall / (precision + recall)
        f1s.append(float(f1))
    return float(np.mean(f1s)) if f1s else float("nan")


def _find_subject_run_dir(root: Path) -> Optional[Path]:
    if (root / "metrics.csv").exists():
        return root
    direct = sorted(root.glob("subject_seed*"))
    if direct:
        return direct[0]
    nested = sorted(root.glob("**/subject_seed*"))
    return nested[0] if nested else None


def _collect_subject_metrics(preds_df: pd.DataFrame) -> pd.DataFrame:
    if preds_df is None or preds_df.empty or "subject_id" not in preds_df.columns:
        return pd.DataFrame()
    rows: List[Dict[str, Any]] = []
    for subject_id, sub in preds_df.groupby("subject_id"):
        rows.append(
            {
                "subject_id": int(subject_id),
                "accuracy": float(sub["correct"].mean()) if "correct" in sub.columns else float("nan"),
                "macro_f1": _macro_f1_from_predictions(sub),
            }
        )
    return pd.DataFrame(rows).sort_values("subject_id").reset_index(drop=True)


def _extract_run_summary(run_dir: Path, variant_name: str) -> Optional[Dict[str, Any]]:
    metrics_df = _safe_read_csv(run_dir / "metrics.csv")
    preds_df = _safe_read_csv(run_dir / "predictions.csv")
    per_class_df = _safe_read_csv(run_dir / "per_class_metrics.csv")
    run_cfg_path = run_dir / "run_config.json"
    if metrics_df is None or metrics_df.empty or preds_df is None or preds_df.empty:
        print(f"[WARN] Missing required files under {run_dir}")
        return None
    metrics_row = metrics_df.iloc[0].to_dict()
    subject_df = _collect_subject_metrics(preds_df)
    subject26 = subject_df[subject_df["subject_id"] == 26]
    class3_f1 = float("nan")
    class6_f1 = float("nan")
    if per_class_df is not None and not per_class_df.empty:
        m3 = per_class_df[per_class_df["class_id"] == 3]
        m6 = per_class_df[per_class_df["class_id"] == 6]
        if not m3.empty:
            class3_f1 = float(m3.iloc[0]["f1"])
        if not m6.empty:
            class6_f1 = float(m6.iloc[0]["f1"])
    run_cfg: Dict[str, Any] = {}
    if run_cfg_path.exists():
        try:
            run_cfg = json.loads(run_cfg_path.read_text(encoding="utf-8"))
        except Exception as exc:
            print(f"[WARN] Failed to read run config: {run_cfg_path} ({exc})")
    seed_val = int(run_cfg.get("seed", preds_df["seed"].iloc[0] if "seed" in preds_df.columns else -1))
    return {
        "variant_name": variant_name,
        "seed": seed_val,
        "test_accuracy": float(metrics_row.get("accuracy", np.nan)),
        "test_macro_f1": float(metrics_row.get("macro_f1", np.nan)),
        "test_macro_precision": float(metrics_row.get("macro_precision", np.nan)),
        "test_macro_recall": float(metrics_row.get("macro_recall", np.nan)),
        "class_3_f1": class3_f1,
        "class_6_f1": class6_f1,
        "class_3_to_6_errors": int(((preds_df["true_label"] == 3) & (preds_df["pred_label"] == 6)).sum()),
        "class_6_to_3_errors": int(((preds_df["true_label"] == 6) & (preds_df["pred_label"] == 3)).sum()),
        "subject_26_accuracy": float(subject26.iloc[0]["accuracy"]) if not subject26.empty else float("nan"),
        "subject_26_macro_f1": float(subject26.iloc[0]["macro_f1"]) if not subject26.empty else float("nan"),
        "inference_ms": float(metrics_row.get("inference_ms", np.nan)),
        "params_m": float(metrics_row.get("params_m", np.nan)),
        "training_seconds": float(metrics_row.get("training_seconds", np.nan)),
        "best_epoch": float(metrics_row.get("best_epoch", np.nan)),
        "best_val_loss": float(metrics_row.get("best_val_loss", np.nan)),
        "best_val_f1": float(metrics_row.get("best_val_f1", np.nan)),
        "run_dir": str(run_dir),
        "_preds_df": preds_df,
        "_per_class_df": per_class_df if per_class_df is not None else pd.DataFrame(),
        "_subject_df": subject_df,
    }


def _load_original_runs(csv_path: Path) -> Tuple[pd.DataFrame, Dict[int, Path]]:
    df = pd.read_csv(csv_path)
    mapping = {int(row["seed"]): Path(row["run_dir"]) for _, row in df.iterrows() if str(row["run_dir"]).strip()}
    return df, mapping


def _load_improved_runs(root: Path) -> Dict[int, Path]:
    mapping: Dict[int, Path] = {}
    for sub in sorted(root.glob("motioninput_reg_seed*")):
        run_dir = _find_subject_run_dir(sub)
        if run_dir is None:
            continue
        try:
            seed = int(run_dir.name.replace("subject_seed", ""))
        except Exception:
            continue
        mapping[seed] = run_dir
    return mapping


def _make_seed_level_comparison(orig_rows: Dict[int, Dict[str, Any]], imp_rows: Dict[int, Dict[str, Any]]) -> pd.DataFrame:
    seeds = sorted(set(orig_rows) & set(imp_rows))
    rows: List[Dict[str, Any]] = []
    for seed in seeds:
        o = orig_rows[seed]
        n = imp_rows[seed]
        rows.append(
            {
                "seed": seed,
                "original_test_accuracy": o["test_accuracy"],
                "improved_test_accuracy": n["test_accuracy"],
                "delta_test_accuracy": n["test_accuracy"] - o["test_accuracy"],
                "original_test_macro_f1": o["test_macro_f1"],
                "improved_test_macro_f1": n["test_macro_f1"],
                "delta_test_macro_f1": n["test_macro_f1"] - o["test_macro_f1"],
                "original_class_3_f1": o["class_3_f1"],
                "improved_class_3_f1": n["class_3_f1"],
                "delta_class_3_f1": n["class_3_f1"] - o["class_3_f1"],
                "original_class_6_f1": o["class_6_f1"],
                "improved_class_6_f1": n["class_6_f1"],
                "delta_class_6_f1": n["class_6_f1"] - o["class_6_f1"],
                "original_class_3_to_6_errors": o["class_3_to_6_errors"],
                "improved_class_3_to_6_errors": n["class_3_to_6_errors"],
                "delta_class_3_to_6_errors": n["class_3_to_6_errors"] - o["class_3_to_6_errors"],
                "original_class_6_to_3_errors": o["class_6_to_3_errors"],
                "improved_class_6_to_3_errors": n["class_6_to_3_errors"],
                "delta_class_6_to_3_errors": n["class_6_to_3_errors"] - o["class_6_to_3_errors"],
                "original_inference_ms": o["inference_ms"],
                "improved_inference_ms": n["inference_ms"],
                "original_params_m": o["params_m"],
                "improved_params_m": n["params_m"],
            }
        )
    return pd.DataFrame(rows)


def _aggregate_mean_std(rows: Dict[int, Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    metrics = [
        "test_accuracy",
        "test_macro_f1",
        "class_3_f1",
        "class_6_f1",
        "class_3_to_6_errors",
        "class_6_to_3_errors",
        "inference_ms",
        "params_m",
    ]
    out: Dict[str, Dict[str, float]] = {}
    df = pd.DataFrame([{k: v for k, v in row.items() if not k.startswith("_")} for row in rows.values()])
    for m in metrics:
        out[m] = {"mean": float(df[m].mean()), "std": float(df[m].std(ddof=1))}
    return out


def _build_aggregate_table(orig_rows: Dict[int, Dict[str, Any]], imp_rows: Dict[int, Dict[str, Any]]) -> pd.DataFrame:
    o = _aggregate_mean_std(orig_rows)
    n = _aggregate_mean_std(imp_rows)
    rows = []
    for metric in o.keys():
        rows.append(
            {
                "metric": metric,
                "original_mean": o[metric]["mean"],
                "original_std": o[metric]["std"],
                "improved_mean": n[metric]["mean"],
                "improved_std": n[metric]["std"],
                "delta_mean": n[metric]["mean"] - o[metric]["mean"],
            }
        )
    return pd.DataFrame(rows)


def _build_per_class_table(orig_rows: Dict[int, Dict[str, Any]], imp_rows: Dict[int, Dict[str, Any]]) -> pd.DataFrame:
    per_class_rows = []
    for cls_id in range(12):
        orig_vals = []
        imp_vals = []
        for row in orig_rows.values():
            df = row["_per_class_df"]
            m = df[df["class_id"] == cls_id]
            if not m.empty:
                orig_vals.append(float(m.iloc[0]["f1"]))
        for row in imp_rows.values():
            df = row["_per_class_df"]
            m = df[df["class_id"] == cls_id]
            if not m.empty:
                imp_vals.append(float(m.iloc[0]["f1"]))
        per_class_rows.append(
            {
                "class_id": cls_id,
                "class_name": f"class_{cls_id}",
                "original_mean_f1": float(np.mean(orig_vals)) if orig_vals else np.nan,
                "original_std_f1": float(np.std(orig_vals, ddof=1)) if len(orig_vals) > 1 else np.nan,
                "improved_mean_f1": float(np.mean(imp_vals)) if imp_vals else np.nan,
                "improved_std_f1": float(np.std(imp_vals, ddof=1)) if len(imp_vals) > 1 else np.nan,
                "delta_mean_f1": (float(np.mean(imp_vals)) - float(np.mean(orig_vals))) if orig_vals and imp_vals else np.nan,
            }
        )
    return pd.DataFrame(per_class_rows)


def _build_confusion_pair_table(orig_rows: Dict[int, Dict[str, Any]], imp_rows: Dict[int, Dict[str, Any]]) -> pd.DataFrame:
    metrics = [("class_3_to_6", "class_3_to_6_errors"), ("class_6_to_3", "class_6_to_3_errors")]
    rows = []
    for label, key in metrics:
        orig_vals = [float(r[key]) for r in orig_rows.values()]
        imp_vals = [float(r[key]) for r in imp_rows.values()]
        rows.append(
            {
                "pair": label,
                "original_mean": float(np.mean(orig_vals)),
                "original_std": float(np.std(orig_vals, ddof=1)) if len(orig_vals) > 1 else np.nan,
                "improved_mean": float(np.mean(imp_vals)),
                "improved_std": float(np.std(imp_vals, ddof=1)) if len(imp_vals) > 1 else np.nan,
                "delta_mean": float(np.mean(imp_vals) - np.mean(orig_vals)),
            }
        )
    return pd.DataFrame(rows)


def _build_subject_wise_table(orig_rows: Dict[int, Dict[str, Any]], imp_rows: Dict[int, Dict[str, Any]]) -> pd.DataFrame:
    rows = []
    subject_ids = sorted(
        set().union(*[set(r["_subject_df"]["subject_id"].tolist()) for r in orig_rows.values() if not r["_subject_df"].empty])
        | set().union(*[set(r["_subject_df"]["subject_id"].tolist()) for r in imp_rows.values() if not r["_subject_df"].empty])
    )
    for subject_id in subject_ids:
        orig_accs, orig_f1s, imp_accs, imp_f1s = [], [], [], []
        for r in orig_rows.values():
            df = r["_subject_df"]
            m = df[df["subject_id"] == subject_id]
            if not m.empty:
                orig_accs.append(float(m.iloc[0]["accuracy"]))
                orig_f1s.append(float(m.iloc[0]["macro_f1"]))
        for r in imp_rows.values():
            df = r["_subject_df"]
            m = df[df["subject_id"] == subject_id]
            if not m.empty:
                imp_accs.append(float(m.iloc[0]["accuracy"]))
                imp_f1s.append(float(m.iloc[0]["macro_f1"]))
        rows.append(
            {
                "subject_id": subject_id,
                "original_accuracy_mean": float(np.mean(orig_accs)) if orig_accs else np.nan,
                "original_accuracy_std": float(np.std(orig_accs, ddof=1)) if len(orig_accs) > 1 else np.nan,
                "improved_accuracy_mean": float(np.mean(imp_accs)) if imp_accs else np.nan,
                "improved_accuracy_std": float(np.std(imp_accs, ddof=1)) if len(imp_accs) > 1 else np.nan,
                "original_macro_f1_mean": float(np.mean(orig_f1s)) if orig_f1s else np.nan,
                "original_macro_f1_std": float(np.std(orig_f1s, ddof=1)) if len(orig_f1s) > 1 else np.nan,
                "improved_macro_f1_mean": float(np.mean(imp_f1s)) if imp_f1s else np.nan,
                "improved_macro_f1_std": float(np.std(imp_f1s, ddof=1)) if len(imp_f1s) > 1 else np.nan,
            }
        )
    return pd.DataFrame(rows)


def _save_figures(seed_df: pd.DataFrame, agg_df: pd.DataFrame, per_class_df: pd.DataFrame, conf_df: pd.DataFrame, fig_dir: Path) -> None:
    fig_dir.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")

    # 1 Macro-F1 by seed
    fig, ax = plt.subplots(figsize=(8, 5), dpi=160)
    ax.plot(seed_df["seed"], seed_df["original_test_macro_f1"] * 100, marker="o", label="Original LeNet_LSTM")
    ax.plot(seed_df["seed"], seed_df["improved_test_macro_f1"] * 100, marker="o", label="MotionInput + LS0.05 + WD3e-4")
    ax.set_xlabel("Seed")
    ax.set_ylabel("Macro-F1 (%)")
    ax.set_title("Macro-F1 by Seed")
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "macro_f1_by_seed_line_or_scatter.png", bbox_inches="tight")
    plt.close(fig)

    # 2 Accuracy by seed
    fig, ax = plt.subplots(figsize=(8, 5), dpi=160)
    ax.plot(seed_df["seed"], seed_df["original_test_accuracy"] * 100, marker="o", label="Original LeNet_LSTM")
    ax.plot(seed_df["seed"], seed_df["improved_test_accuracy"] * 100, marker="o", label="MotionInput + LS0.05 + WD3e-4")
    ax.set_xlabel("Seed")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Accuracy by Seed")
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "accuracy_by_seed_line_or_scatter.png", bbox_inches="tight")
    plt.close(fig)

    # 3 Macro-F1 boxplot
    fig, ax = plt.subplots(figsize=(7, 5), dpi=160)
    ax.boxplot(
        [seed_df["original_test_macro_f1"] * 100, seed_df["improved_test_macro_f1"] * 100],
        tick_labels=["Original", "Improved"],
    )
    ax.set_ylabel("Macro-F1 (%)")
    ax.set_title("Macro-F1 Distribution over 10 Seeds")
    fig.tight_layout()
    fig.savefig(fig_dir / "macro_f1_boxplot.png", bbox_inches="tight")
    plt.close(fig)

    # 4 class3 class6 F1 mean std
    fig, ax = plt.subplots(figsize=(8, 5), dpi=160)
    subset = agg_df.set_index("metric")
    labels = ["class_3_f1", "class_6_f1"]
    x = np.arange(len(labels))
    width = 0.34
    orig_means = [subset.loc[l, "original_mean"] * 100 for l in labels]
    orig_stds = [subset.loc[l, "original_std"] * 100 for l in labels]
    imp_means = [subset.loc[l, "improved_mean"] * 100 for l in labels]
    imp_stds = [subset.loc[l, "improved_std"] * 100 for l in labels]
    ax.bar(x - width/2, orig_means, width, yerr=orig_stds, label="Original")
    ax.bar(x + width/2, imp_means, width, yerr=imp_stds, label="Improved")
    ax.set_xticks(x)
    ax.set_xticklabels(["class_3 (Hold)", "class_6 (Static Drag)"])
    ax.set_ylabel("F1 (%)")
    ax.set_title("class_3 / class_6 F1 Mean ± Std")
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "class3_class6_f1_bar_mean_std.png", bbox_inches="tight")
    plt.close(fig)

    # 5 confusion bars
    fig, ax = plt.subplots(figsize=(8, 5), dpi=160)
    conf_map = conf_df.set_index("pair")
    labels = ["class_3_to_6", "class_6_to_3"]
    x = np.arange(len(labels))
    orig_means = [conf_map.loc[l, "original_mean"] for l in labels]
    orig_stds = [conf_map.loc[l, "original_std"] for l in labels]
    imp_means = [conf_map.loc[l, "improved_mean"] for l in labels]
    imp_stds = [conf_map.loc[l, "improved_std"] for l in labels]
    ax.bar(x - width/2, orig_means, width, yerr=orig_stds, label="Original")
    ax.bar(x + width/2, imp_means, width, yerr=imp_stds, label="Improved")
    ax.set_xticks(x)
    ax.set_xticklabels(["class_3→6", "class_6→3"])
    ax.set_ylabel("Error count")
    ax.set_title("Hold / Static Drag Confusion Mean ± Std")
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "hold_staticdrag_confusion_bar.png", bbox_inches="tight")
    plt.close(fig)

    # 6 per-class delta
    fig, ax = plt.subplots(figsize=(10, 5), dpi=160)
    colors = ["tab:red" if c in (3, 6) else "tab:blue" for c in per_class_df["class_id"]]
    ax.bar(per_class_df["class_id"].astype(str), per_class_df["delta_mean_f1"] * 100, color=colors)
    ax.axhline(0, color="black", linewidth=1)
    ax.set_xlabel("Class ID")
    ax.set_ylabel("Improved - Original Mean F1 (%)")
    ax.set_title("Per-class F1 Delta")
    fig.tight_layout()
    fig.savefig(fig_dir / "per_class_f1_delta.png", bbox_inches="tight")
    plt.close(fig)

    # 7 inference/params tradeoff
    fig, ax = plt.subplots(figsize=(8, 5), dpi=160)
    orig = seed_df[["original_inference_ms", "original_test_macro_f1", "original_params_m"]].copy()
    imp = seed_df[["improved_inference_ms", "improved_test_macro_f1", "improved_params_m"]].copy()
    ax.scatter(orig["original_inference_ms"], orig["original_test_macro_f1"] * 100, s=orig["original_params_m"] * 800, alpha=0.65, label="Original")
    ax.scatter(imp["improved_inference_ms"], imp["improved_test_macro_f1"] * 100, s=imp["improved_params_m"] * 800, alpha=0.65, label="Improved")
    ax.set_xlabel("Inference Time (ms)")
    ax.set_ylabel("Macro-F1 (%)")
    ax.set_title("Inference / Params / Macro-F1 Trade-off")
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "inference_params_tradeoff.png", bbox_inches="tight")
    plt.close(fig)


def _write_report(
    seed_df: pd.DataFrame,
    agg_df: pd.DataFrame,
    per_class_df: pd.DataFrame,
    conf_df: pd.DataFrame,
    subject_df: pd.DataFrame,
    out_path: Path,
    seeds: List[int],
) -> None:
    agg = agg_df.set_index("metric")
    improved_better = float(agg.loc["test_macro_f1", "improved_mean"]) > float(agg.loc["test_macro_f1", "original_mean"])
    with out_path.open("w", encoding="utf-8") as f:
        f.write("# MotionInput + Regularization 10-seed Report\n\n")
        f.write("## A. Purpose\n")
        f.write("This is a post-hoc targeted improvement validation, not part of the original fair multi-model comparison.\n\n")
        f.write("## B. Compared models\n")
        f.write("- Original LeNet_LSTM\n")
        f.write("- LeNet_LSTM_MotionInput + LS0.05 + WD3e-4\n\n")
        f.write("## C. Seed protocol\n")
        f.write(f"- Seeds used: {seeds}\n")
        f.write("- Same subject-independent split seeds were used for both models.\n\n")
        f.write("## D. Overall mean ± std comparison\n")
        for metric, label in [("test_accuracy", "Accuracy"), ("test_macro_f1", "Macro-F1"), ("inference_ms", "Inference time (ms)"), ("params_m", "Params (M)")]:
            f.write(
                f"- {label}: original {agg.loc[metric, 'original_mean']:.6f} ± {agg.loc[metric, 'original_std']:.6f}, "
                f"improved {agg.loc[metric, 'improved_mean']:.6f} ± {agg.loc[metric, 'improved_std']:.6f}\n"
            )
        f.write("\n## E. Hold / Static Drag comparison\n")
        for metric, label in [("class_3_f1", "class_3 F1 (Hold)"), ("class_6_f1", "class_6 F1 (Static Drag)"), ("class_3_to_6_errors", "class_3->6 errors"), ("class_6_to_3_errors", "class_6->3 errors")]:
            f.write(
                f"- {label}: original {agg.loc[metric, 'original_mean']:.6f}, improved {agg.loc[metric, 'improved_mean']:.6f}, "
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
                    f"improved {row['improved_macro_f1_mean']:.6f}\n"
                )
        f.write("\n## H. Interpretation\n")
        f.write("This comparison tests whether explicit frame-difference motion input plus moderate regularization consistently improves dynamic tactile recognition under the same subject-independent split seeds.\n")
        f.write("\n## I. Recommendation\n")
        if improved_better:
            f.write("The improved model shows a higher mean Macro-F1 with only a small overhead in inference time and parameter count, so it is the current recommended improved LeNet_LSTM variant.\n")
        else:
            f.write("The improved model does not show a stable mean Macro-F1 gain, so it should be treated as exploratory and needs further validation.\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--original_runs_csv", type=Path, required=True)
    parser.add_argument("--improved_runs_root", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_dir = args.output_dir / "summary"
    figures_dir = args.output_dir / "figures"
    summary_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    original_csv_df, original_map = _load_original_runs(args.original_runs_csv)
    improved_map = _load_improved_runs(args.improved_runs_root)

    orig_rows: Dict[int, Dict[str, Any]] = {}
    imp_rows: Dict[int, Dict[str, Any]] = {}
    for seed, run_dir in sorted(original_map.items()):
        row = _extract_run_summary(run_dir, "original")
        if row is not None:
            orig_rows[seed] = row
    for seed, run_dir in sorted(improved_map.items()):
        row = _extract_run_summary(run_dir, "improved")
        if row is not None:
            imp_rows[seed] = row

    common_seeds = sorted(set(orig_rows) & set(imp_rows))
    if not common_seeds:
        raise SystemExit("No overlapping seeds between original and improved results.")
    orig_rows = {k: orig_rows[k] for k in common_seeds}
    imp_rows = {k: imp_rows[k] for k in common_seeds}

    seed_df = _make_seed_level_comparison(orig_rows, imp_rows)
    agg_df = _build_aggregate_table(orig_rows, imp_rows)
    per_class_df = _build_per_class_table(orig_rows, imp_rows)
    conf_df = _build_confusion_pair_table(orig_rows, imp_rows)
    subject_df = _build_subject_wise_table(orig_rows, imp_rows)

    seed_df.to_csv(summary_dir / "seed_level_comparison.csv", index=False)
    agg_df.to_csv(summary_dir / "aggregate_mean_std.csv", index=False)
    per_class_df.to_csv(summary_dir / "per_class_f1_mean_std_comparison.csv", index=False)
    conf_df.to_csv(summary_dir / "confusion_pair_comparison.csv", index=False)
    if not subject_df.empty:
        subject_df.to_csv(summary_dir / "subject_wise_mean_std_comparison.csv", index=False)

    _save_figures(seed_df, agg_df, per_class_df, conf_df, figures_dir)
    _write_report(
        seed_df,
        agg_df,
        per_class_df,
        conf_df,
        subject_df,
        args.output_dir / "motioninput_reg_10seed_report.md",
        common_seeds,
    )

    print(f"Saved seed-level comparison to: {summary_dir / 'seed_level_comparison.csv'}")
    print(f"Saved report to: {args.output_dir / 'motioninput_reg_10seed_report.md'}")


if __name__ == "__main__":
    main()
