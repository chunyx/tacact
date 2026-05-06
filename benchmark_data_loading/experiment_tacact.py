#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from torch.utils.data import Subset

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tacact.models import ModelFactory
from tacact.benchmark_common import (
    build_split_audit,
    create_optimized_dataset,
    get_device,
    make_three_loaders,
    save_split_audit,
    split_indices_3way,
    warmup_cache,
)

# 统一从新建的 tacact.utils 导入所有工具和绘图函数
from tacact.utils import (
    benchmark_sklearn,
    benchmark_torch_gpu_deploy,
    benchmark_torch_model_only,
    confusion_matrix_np,
    count_parameters,
    count_sklearn_params,
    dataframe_to_results_dict,
    merge_metrics_csvs,
    parse_model_list,
    per_class_prf,
    set_seed,
    subset_to_numpy,
    train_torch_model,
    save_confusion_comparison,
    save_confusion_matrix,
    save_per_class_f1_bars,
    save_radar_top3,
    save_accuracy_vs_inference_bubble,
    save_accuracy_vs_params_scatter,
    save_macrof1_vs_inference_bubble,
    save_efficiency_score_bar,
    save_pareto_accuracy_inference,
    save_accuracy_vs_training_time,
    save_macrof1_vs_params_scatter,
    save_dl_pareto_macrof1_vs_inference,
    save_dl_family_tradeoff,
    save_dl_macrof1_vs_training_time,
    save_dl_params_vs_inference,
    save_params_inference_all_seeds_from_df,
    save_pareto_all_seeds_from_df,
    save_dl_performance_vs_sequence_length,
    save_scatter,
    save_summary_bar_with_error,
    save_training_curves,
    save_training_curves_with_std,
    save_per_model_accuracy_loss_curves,
    save_all_models_loss_overlay,
    save_all_models_loss_overlay_with_std,
    save_convergence_diagnostics,
    save_convergence_diagnostics_with_std,
    model_display_name,
)


def _write_status(path: Path | None, payload: Dict[str, Any]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.json")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def _split_results_for_paper(
    results: Dict[str, Dict[str, float]],
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, Dict[str, float]]]:
    deep = {k: v for k, v in results.items() if str(v.get("category", "")).lower() != "traditional"}
    traditional = {k: v for k, v in results.items() if str(v.get("category", "")).lower() == "traditional"}
    return deep, traditional


def _save_traditional_baseline_table_and_plot(
    traditional_results: Dict[str, Dict[str, float]],
    save_dir: Path,
    suffix: str = "",
) -> None:
    if not traditional_results:
        return
    rows: List[Dict[str, float | str]] = []
    for model_name, m in traditional_results.items():
        rows.append(
            {
                "model": model_name,
                "accuracy": float(m.get("accuracy", 0.0)),
                "macro_f1": float(m.get("macro_f1", 0.0)),
                "category": "traditional",
            }
        )
    base_df = pd.DataFrame(rows).sort_values(["macro_f1", "accuracy"], ascending=[False, False]).reset_index(drop=True)
    base_df.to_csv(save_dir / f"traditional_baseline_metrics{suffix}.csv", index=False)

    display_names = {
        "RandomForest": "RF",
        "XGBoost": "XGB",
        "SVM": "SVM",
    }
    x_names = [display_names.get(str(x), str(x)) for x in base_df["model"].tolist()]
    x = np.arange(len(base_df))
    width = 0.35
    plt.figure(figsize=(max(6.0, len(x) * 1.2), 4.2))
    plt.bar(x - width / 2, base_df["accuracy"].to_numpy(dtype=np.float64) * 100.0, width=width, label="Accuracy")
    plt.bar(x + width / 2, base_df["macro_f1"].to_numpy(dtype=np.float64) * 100.0, width=width, label="Macro-F1")
    plt.xticks(x, x_names, fontsize=10)
    plt.yticks(fontsize=10)
    plt.ylabel("Score (%)", fontsize=11)
    plt.title("Traditional ML Baselines (Reference Only)", fontsize=12)
    plt.legend(fontsize=10, frameon=False)
    plt.grid(True, axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(save_dir / f"traditional_baseline_reference{suffix}.png", dpi=180, bbox_inches="tight")
    plt.close()


def _scheduler_name_for_model(model_name: str) -> str:
    return "CosineAnnealingWarmRestarts" if "transformer" in str(model_name).lower() else "ReduceLROnPlateau"


def _class_names(n_classes: int = 12) -> List[str]:
    return [f"class_{i}" for i in range(n_classes)]


def _runtime_info() -> Dict[str, Any]:
    git_commit = ""
    try:
        git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        git_commit = ""
    return {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "python_version": sys.version.replace("\n", " "),
        "torch_version": torch.__version__,
        "platform": platform.platform(),
        "hostname": platform.node(),
        "git_commit": git_commit,
    }


def _save_loss_curves_per_model_seed(
    training_df: pd.DataFrame,
    selection_df: pd.DataFrame,
    save_dir: Path,
) -> None:
    if training_df.empty:
        return
    save_dir.mkdir(parents=True, exist_ok=True)
    has_multi_seed = training_df["seed"].nunique() > 1 if "seed" in training_df.columns else False

    for (model, seed), g in training_df.groupby(["model", "seed"], sort=False):
        gg = g.sort_values("epoch")
        epochs = gg["epoch"].to_numpy(dtype=np.int32)
        train_loss = gg["train_loss"].to_numpy(dtype=np.float64)
        val_loss = gg["val_loss"].to_numpy(dtype=np.float64)
        selected_epoch = None
        if not selection_df.empty:
            s = selection_df[(selection_df["model"] == model) & (selection_df["seed"] == seed)]
            if not s.empty and pd.notna(s.iloc[0].get("selected_epoch", np.nan)):
                try:
                    selected_epoch = int(s.iloc[0]["selected_epoch"])
                except Exception:
                    selected_epoch = None

        fig, ax = plt.subplots(figsize=(8.5, 5.2))
        ax.plot(epochs, train_loss, label="train_loss", color="#2a9d8f", linewidth=2.0)
        ax.plot(epochs, val_loss, label="val_loss", color="#e76f51", linewidth=2.0)
        if selected_epoch is not None and selected_epoch > 0:
            ax.axvline(selected_epoch, linestyle="--", color="#444444", linewidth=1.3, label=f"selected_epoch={selected_epoch}")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title(f"{model_display_name(str(model))} | model={model} | seed={int(seed)}")
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=False)
        plt.tight_layout()
        plt.savefig(save_dir / f"loss_curve_{model}_seed{int(seed)}.png", dpi=220, bbox_inches="tight")
        if not has_multi_seed:
            plt.savefig(save_dir / f"loss_curve_{model}.png", dpi=220, bbox_inches="tight")
        plt.close(fig)


def _save_val_loss_all_models(training_df: pd.DataFrame, save_path: Path) -> None:
    if training_df.empty:
        return
    fig, ax = plt.subplots(figsize=(10.5, 6.0))
    for model, g in training_df.groupby("model", sort=False):
        gg = g.sort_values("epoch")
        ax.plot(
            gg["epoch"].to_numpy(dtype=np.int32),
            gg["val_loss"].to_numpy(dtype=np.float64),
            linewidth=2.0,
            label=model_display_name(str(model)),
        )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation Loss")
    ax.set_title("Validation Loss Across Models")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, ncol=2)
    plt.tight_layout()
    plt.savefig(save_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _save_loss_curves_grid(training_df: pd.DataFrame, save_path: Path) -> None:
    if training_df.empty:
        return
    groups = list(training_df.groupby("model", sort=False))
    n = len(groups)
    ncols = 2 if n > 1 else 1
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(7.5 * ncols, 3.9 * nrows), squeeze=False)
    axes_flat = axes.flatten()

    for idx, (model, g) in enumerate(groups):
        ax = axes_flat[idx]
        gg = g.sort_values("epoch")
        ax.plot(gg["epoch"].to_numpy(dtype=np.int32), gg["train_loss"].to_numpy(dtype=np.float64), label="train_loss", linewidth=1.8, color="#2a9d8f")
        ax.plot(gg["epoch"].to_numpy(dtype=np.int32), gg["val_loss"].to_numpy(dtype=np.float64), label="val_loss", linewidth=1.8, color="#e76f51")
        ax.set_title(model_display_name(str(model)))
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=False, fontsize=8)

    for idx in range(n, len(axes_flat)):
        axes_flat[idx].axis("off")

    fig.suptitle("Loss Curves Grid (Train vs Validation)", fontsize=13, y=0.995)
    plt.tight_layout()
    plt.savefig(save_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _save_loss_mean_std_across_seeds(training_df: pd.DataFrame, save_path: Path) -> None:
    if training_df.empty or "seed" not in training_df.columns or training_df["seed"].nunique() <= 1:
        return
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    for model, g in training_df.groupby("model", sort=False):
        pivot_train = g.pivot_table(index="epoch", columns="seed", values="train_loss", aggfunc="mean").sort_index()
        pivot_val = g.pivot_table(index="epoch", columns="seed", values="val_loss", aggfunc="mean").sort_index()
        epochs_t = pivot_train.index.to_numpy(dtype=np.int32)
        epochs_v = pivot_val.index.to_numpy(dtype=np.int32)
        train_mean = pivot_train.mean(axis=1).to_numpy(dtype=np.float64)
        train_std = pivot_train.std(axis=1, ddof=0).to_numpy(dtype=np.float64)
        val_mean = pivot_val.mean(axis=1).to_numpy(dtype=np.float64)
        val_std = pivot_val.std(axis=1, ddof=0).to_numpy(dtype=np.float64)
        label = model_display_name(str(model))
        axes[0].plot(epochs_t, train_mean, linewidth=2.0, label=label)
        axes[0].fill_between(epochs_t, train_mean - train_std, train_mean + train_std, alpha=0.16)
        axes[1].plot(epochs_v, val_mean, linewidth=2.0, label=label)
        axes[1].fill_between(epochs_v, val_mean - val_std, val_mean + val_std, alpha=0.16)

    axes[0].set_title("Train Loss Mean ± Std Across Seeds")
    axes[1].set_title("Validation Loss Mean ± Std Across Seeds")
    axes[0].set_ylabel("Loss")
    axes[1].set_ylabel("Loss")
    axes[1].set_xlabel("Epoch")
    axes[0].grid(True, alpha=0.25)
    axes[1].grid(True, alpha=0.25)
    axes[0].legend(frameon=False, ncol=2)
    axes[1].legend(frameon=False, ncol=2)
    plt.tight_layout()
    plt.savefig(save_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _safe_concat_csv(paths: List[Path]) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for path in paths:
        if path.exists():
            try:
                frames.append(pd.read_csv(path))
            except Exception as e:
                print(f"[WARN] Could not read CSV {path}: {e}")
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _history_mean_std_df(training_df: pd.DataFrame) -> pd.DataFrame:
    if training_df.empty:
        return pd.DataFrame()
    metric_cols = [
        "train_loss", "val_loss", "test_loss",
        "train_acc", "val_acc", "test_acc",
        "train_f1", "val_f1", "test_f1",
        "lr", "epoch_time_sec",
    ]
    agg_spec: Dict[str, Tuple[str, str]] = {}
    for col in metric_cols:
        if col in training_df.columns:
            agg_spec[f"{col}_mean"] = (col, "mean")
            agg_spec[f"{col}_std"] = (col, "std")
    return (
        training_df
        .groupby(["model", "epoch"], as_index=False)
        .agg(**agg_spec)
        .sort_values(["model", "epoch"])
        .reset_index(drop=True)
    )


def _save_metric_overlay_with_std_from_training_df(
    training_df: pd.DataFrame,
    save_path: Path,
    metric_col: str,
    title: str,
    ylabel: str,
) -> None:
    if training_df.empty or metric_col not in training_df.columns:
        return
    fig, ax = plt.subplots(figsize=(11.0, 6.5))
    cmap = plt.get_cmap("tab10")
    for idx, (model, g) in enumerate(training_df.groupby("model", sort=False)):
        pivot = g.pivot_table(index="epoch", columns="seed", values=metric_col, aggfunc="mean").sort_index()
        if pivot.empty:
            continue
        epochs = pivot.index.to_numpy(dtype=np.int32)
        mean_vals = pivot.mean(axis=1).to_numpy(dtype=np.float64)
        std_vals = pivot.std(axis=1, ddof=0).fillna(0.0).to_numpy(dtype=np.float64)
        color = cmap(idx % 10)
        ax.plot(epochs, mean_vals, linewidth=2.0, label=model_display_name(str(model)), color=color)
        if pivot.shape[1] > 1:
            ax.fill_between(epochs, mean_vals - std_vals, mean_vals + std_vals, color=color, alpha=0.16)
    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, ncol=2)
    plt.tight_layout()
    plt.savefig(save_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _save_all_seed_points_plot(
    metrics_df: pd.DataFrame,
    save_path: Path,
    x_col: str,
    y_col: str,
    x_label: str,
    y_label: str,
    title: str,
    x_transform: str = "",
    y_limits: Tuple[float, float] | None = None,
) -> None:
    if metrics_df.empty or x_col not in metrics_df.columns or y_col not in metrics_df.columns:
        return
    df = metrics_df.copy()
    df = df[df["category"].astype(str).str.lower() != "traditional"].copy() if "category" in df.columns else df
    if df.empty:
        return
    if x_transform == "minutes":
        df["_x_plot"] = df[x_col].astype(float) / 60.0
    else:
        df["_x_plot"] = df[x_col].astype(float)

    fig, ax = plt.subplots(figsize=(10.8, 6.4))
    cmap = plt.get_cmap("tab10")
    for idx, (model, g) in enumerate(df.groupby("model", sort=False)):
        color = cmap(idx % 10)
        jitter = np.linspace(-0.06, 0.06, num=len(g)) if len(g) > 1 else np.array([0.0])
        x_vals = g["_x_plot"].to_numpy(dtype=np.float64) + jitter
        y_vals = g[y_col].to_numpy(dtype=np.float64)
        if float(np.nanmax(y_vals)) <= 1.5:
            y_vals = y_vals * 100.0
        ax.scatter(
            x_vals,
            y_vals,
            s=58,
            alpha=0.72,
            color=color,
            edgecolors="none",
            label=model_display_name(str(model)),
        )
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    if y_limits is not None:
        ax.set_ylim(*y_limits)
    ax.grid(True, alpha=0.22)
    ax.legend(frameon=False, ncol=2)
    plt.tight_layout()
    plt.savefig(save_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _build_parallel_merged_outputs(
    output_dir: Path,
    deep_models: List[str],
    seed_list: List[int],
) -> None:
    worker_root = output_dir / "parallel_workers"
    if not worker_root.exists():
        print(f"[WARN] parallel_workers not found under {output_dir}, skip auto-merge.")
        return

    metrics_paths: List[Path] = []
    training_paths: List[Path] = []
    selection_paths: List[Path] = []
    final_split_paths: List[Path] = []
    model_cfg_paths: List[Path] = []
    predictions_paths: List[Path] = []
    confusion_paths: List[Path] = []
    per_class_paths: List[Path] = []
    hpo_trace_paths: List[Path] = []
    runtime_paths: List[Path] = []

    for seed in seed_list:
        run_dir_name = f"subject_seed{int(seed)}"
        for model_name in deep_models:
            base = worker_root / f"{model_name}_seed{int(seed)}" / run_dir_name
            metrics_paths.append(base / "metrics.csv")
            training_paths.append(base / "training_history.csv")
            selection_paths.append(base / "selection_summary.csv")
            final_split_paths.append(base / "final_split_metrics.csv")
            model_cfg_paths.append(base / "model_config_summary.csv")
            predictions_paths.append(base / "predictions.csv")
            confusion_paths.append(base / "confusion_matrix.csv")
            per_class_paths.append(base / "per_class_metrics.csv")
            hpo_trace_paths.append(base / "hpo_to_main_trace.csv")
            runtime_paths.append(base / "runtime_summary.csv")

    metrics_df = _safe_concat_csv(metrics_paths)
    training_df = _safe_concat_csv(training_paths)
    selection_df = _safe_concat_csv(selection_paths)
    final_split_df = _safe_concat_csv(final_split_paths)
    model_cfg_df = _safe_concat_csv(model_cfg_paths)
    predictions_df = _safe_concat_csv(predictions_paths)
    confusion_df = _safe_concat_csv(confusion_paths)
    per_class_df = _safe_concat_csv(per_class_paths)
    hpo_trace_df = _safe_concat_csv(hpo_trace_paths)
    runtime_df = _safe_concat_csv(runtime_paths)

    if not metrics_df.empty:
        metrics_df.to_csv(output_dir / "metrics_merged.csv", index=False)
        metrics_summary_df = (
            metrics_df
            .groupby(["model", "category"], as_index=False)
            .agg(
                accuracy_mean=("accuracy", "mean"),
                accuracy_std=("accuracy", "std"),
                macro_f1_mean=("macro_f1", "mean"),
                macro_f1_std=("macro_f1", "std"),
                macro_precision_mean=("macro_precision", "mean"),
                macro_precision_std=("macro_precision", "std"),
                macro_recall_mean=("macro_recall", "mean"),
                macro_recall_std=("macro_recall", "std"),
                training_seconds_mean=("training_seconds", "mean"),
                training_seconds_std=("training_seconds", "std"),
                inference_ms_mean=("inference_ms", "mean"),
                inference_ms_std=("inference_ms", "std"),
                params_m=("params_m", "first"),
            )
            .sort_values(["category", "macro_f1_mean", "accuracy_mean"], ascending=[True, False, False])
            .reset_index(drop=True)
        )
        metrics_summary_df.to_csv(output_dir / "metrics_summary.csv", index=False)
    if not training_df.empty:
        training_df.to_csv(output_dir / "training_history_merged.csv", index=False)
    if not selection_df.empty:
        selection_df.to_csv(output_dir / "selection_summary_merged.csv", index=False)
    if not final_split_df.empty:
        final_split_df.to_csv(output_dir / "final_split_metrics_merged.csv", index=False)
        final_split_summary_df = (
            final_split_df
            .groupby(["model", "display_name"], as_index=False)
            .agg(
                train_loss_mean=("train_loss_at_selected", "mean"),
                train_loss_std=("train_loss_at_selected", "std"),
                train_acc_mean=("train_acc_at_selected", "mean"),
                train_acc_std=("train_acc_at_selected", "std"),
                train_f1_mean=("train_f1_at_selected", "mean"),
                train_f1_std=("train_f1_at_selected", "std"),
                val_loss_mean=("val_loss_at_selected", "mean"),
                val_loss_std=("val_loss_at_selected", "std"),
                val_acc_mean=("val_acc_at_selected", "mean"),
                val_acc_std=("val_acc_at_selected", "std"),
                val_f1_mean=("val_f1_at_selected", "mean"),
                val_f1_std=("val_f1_at_selected", "std"),
                test_acc_mean=("test_accuracy", "mean"),
                test_acc_std=("test_accuracy", "std"),
                test_macro_f1_mean=("test_macro_f1", "mean"),
                test_macro_f1_std=("test_macro_f1", "std"),
                test_macro_precision_mean=("test_macro_precision", "mean"),
                test_macro_precision_std=("test_macro_precision", "std"),
                test_macro_recall_mean=("test_macro_recall", "mean"),
                test_macro_recall_std=("test_macro_recall", "std"),
                params_m=("params_m", "first"),
                train_time_sec_mean=("train_time_sec", "mean"),
                train_time_sec_std=("train_time_sec", "std"),
                inference_ms_mean=("inference_ms", "mean"),
                inference_ms_std=("inference_ms", "std"),
            )
            .sort_values(["test_macro_f1_mean", "test_acc_mean"], ascending=[False, False])
            .reset_index(drop=True)
        )
        final_split_summary_df.to_csv(output_dir / "final_split_metrics_summary.csv", index=False)
    if not model_cfg_df.empty:
        model_cfg_df.to_csv(output_dir / "model_config_summary_merged.csv", index=False)
    if not predictions_df.empty:
        predictions_df.to_csv(output_dir / "predictions_merged.csv", index=False)
    if not confusion_df.empty:
        confusion_df.to_csv(output_dir / "confusion_matrix_merged.csv", index=False)
    if not per_class_df.empty:
        per_class_df.to_csv(output_dir / "per_class_metrics_merged.csv", index=False)
    if not hpo_trace_df.empty:
        hpo_trace_df.to_csv(output_dir / "hpo_to_main_trace_merged.csv", index=False)
    if not runtime_df.empty:
        runtime_df.to_csv(output_dir / "runtime_summary_merged.csv", index=False)

    if not training_df.empty:
        hist_mean_std_df = _history_mean_std_df(training_df)
        hist_mean_std_df.to_csv(output_dir / "training_history_mean_std_merged.csv", index=False)
        training_df[
            ["model", "seed", "epoch", "train_acc", "val_acc", "test_acc"]
        ].to_csv(output_dir / "accuracy_vs_epoch_all_seeds.csv", index=False)
        training_df.to_csv(output_dir / "epoch_history_all_splits_all_seeds.csv", index=False)

        _save_metric_overlay_with_std_from_training_df(
            training_df, output_dir / "all_models_train_loss_vs_epoch_mean_std.png",
            "train_loss", "All Models: Training Loss vs Epoch", "Training Loss"
        )
        _save_metric_overlay_with_std_from_training_df(
            training_df, output_dir / "all_models_val_loss_vs_epoch_mean_std.png",
            "val_loss", "All Models: Validation Loss vs Epoch", "Validation Loss"
        )
        _save_metric_overlay_with_std_from_training_df(
            training_df, output_dir / "all_models_test_loss_vs_epoch_mean_std.png",
            "test_loss", "All Models: Test Loss vs Epoch", "Test Loss"
        )
        _save_metric_overlay_with_std_from_training_df(
            training_df, output_dir / "all_models_train_accuracy_vs_epoch_mean_std.png",
            "train_acc", "All Models: Training Accuracy vs Epoch", "Training Accuracy"
        )
        _save_metric_overlay_with_std_from_training_df(
            training_df, output_dir / "all_models_val_accuracy_vs_epoch_mean_std.png",
            "val_acc", "All Models: Validation Accuracy vs Epoch", "Validation Accuracy"
        )
        _save_metric_overlay_with_std_from_training_df(
            training_df, output_dir / "all_models_test_accuracy_vs_epoch_mean_std.png",
            "test_acc", "All Models: Test Accuracy vs Epoch", "Test Accuracy"
        )
        _save_metric_overlay_with_std_from_training_df(
            training_df, output_dir / "all_models_train_f1_vs_epoch_mean_std.png",
            "train_f1", "All Models: Training Macro-F1 vs Epoch", "Training Macro-F1"
        )
        _save_metric_overlay_with_std_from_training_df(
            training_df, output_dir / "all_models_val_f1_vs_epoch_mean_std.png",
            "val_f1", "All Models: Validation Macro-F1 vs Epoch", "Validation Macro-F1"
        )
        _save_metric_overlay_with_std_from_training_df(
            training_df, output_dir / "all_models_test_f1_vs_epoch_mean_std.png",
            "test_f1", "All Models: Test Macro-F1 vs Epoch", "Test Macro-F1"
        )

    if not metrics_df.empty:
        metrics_df[
            ["model", "seed", "accuracy", "inference_ms", "training_seconds", "macro_f1", "params_m"]
        ].to_csv(output_dir / "accuracy_vs_inference_time_all_seeds.csv", index=False)
        _save_all_seed_points_plot(
            metrics_df=metrics_df,
            save_path=output_dir / "accuracy_training_time_all_seeds_points_only_linear_minutes.png",
            x_col="training_seconds",
            y_col="accuracy",
            x_label="Training Time (minutes)",
            y_label="Test Accuracy (%)",
            title="Test Accuracy vs Training Time (All Seeds, Points Only)",
            x_transform="minutes",
            y_limits=(70.0, 95.0),
        )
        _save_all_seed_points_plot(
            metrics_df=metrics_df,
            save_path=output_dir / "accuracy_inference_time_all_seeds_points_only_linear.png",
            x_col="inference_ms",
            y_col="accuracy",
            x_label="Inference Time (ms)",
            y_label="Test Accuracy (%)",
            title="Test Accuracy vs Inference Time (All Seeds, Points Only)",
            y_limits=(70.0, 95.0),
        )

    workbook_path = output_dir / "main_experiment_all_seeds_summary.xlsx"
    try:
        with pd.ExcelWriter(workbook_path) as writer:
            if not training_df.empty:
                training_df.to_excel(writer, sheet_name="epoch_history_all_seeds", index=False)
                training_df[["model", "seed", "epoch", "train_acc", "val_acc", "test_acc"]].to_excel(
                    writer, sheet_name="accuracy_vs_epoch", index=False
                )
            if not metrics_df.empty:
                metrics_df.to_excel(writer, sheet_name="metrics_all_seeds", index=False)
                metrics_df[["model", "seed", "accuracy", "inference_ms", "training_seconds", "macro_f1", "params_m"]].to_excel(
                    writer, sheet_name="accuracy_vs_inference", index=False
                )
                if 'metrics_summary_df' in locals():
                    metrics_summary_df.to_excel(writer, sheet_name="metrics_summary", index=False)
            if not final_split_df.empty:
                final_split_df.to_excel(writer, sheet_name="final_split_metrics", index=False)
                if 'final_split_summary_df' in locals():
                    final_split_summary_df.to_excel(writer, sheet_name="final_split_summary", index=False)
            if not selection_df.empty:
                selection_df.to_excel(writer, sheet_name="selection_summary", index=False)
            if not model_cfg_df.empty:
                model_cfg_df.to_excel(writer, sheet_name="model_config_summary", index=False)
        print(f"[Main Parallel] Saved merged workbook: {workbook_path}")
    except Exception as e:
        print(f"[WARN] Could not export merged workbook {workbook_path}: {e}")


def _parse_int_list_csv(text: str) -> List[int]:
    return [int(x.strip()) for x in str(text).split(",") if x.strip()]


def _run_parallel_main_experiment(args: argparse.Namespace, deep_models: List[str], seed_list: List[int]) -> None:
    if args.run_mode != "deep":
        raise ValueError("Parallel main scheduler currently supports --run_mode deep only.")
    gpu_ids = _parse_int_list_csv(args.gpu_ids)
    if not gpu_ids:
        raise ValueError("No gpu_ids parsed.")
    max_workers = min(int(args.max_workers), len(gpu_ids))
    if max_workers <= 0:
        raise ValueError("--max_workers must be >=1 and <= number of gpu_ids.")

    tasks: List[Dict[str, Any]] = []
    for seed in seed_list:
        for model_name in deep_models:
            task_name = f"{model_name}_seed{seed}"
            task_out = args.output_dir / "parallel_workers" / task_name
            status_path = args.output_dir / "parallel_status" / f"{task_name}.json"
            tasks.append(
                {
                    "model": model_name,
                    "seed": int(seed),
                    "name": task_name,
                    "out": task_out,
                    "status_path": status_path,
                }
            )

    py_exec = sys.executable
    script_path = Path(__file__).resolve()
    pending: List[Dict[str, Any]] = list(tasks)

    running: List[Dict[str, Any]] = []
    free_gpus = list(gpu_ids[:max_workers])
    done = 0
    total = len(tasks)

    def _render_dashboard() -> None:
        lines: List[str] = []
        lines.append(f"[Main Parallel Dashboard] done={done}/{total} pending={len(pending)} running={len(running)}")
        gpu_rows: Dict[int, str] = {g: f"GPU {g}: idle" for g in gpu_ids[:max_workers]}
        for item in running:
            gpu_id = int(item["gpu_id"])
            st_path: Path = item["task"]["status_path"]
            if st_path.exists():
                try:
                    prog = json.loads(st_path.read_text(encoding="utf-8"))
                    ep = int(prog.get("current_epoch", 0))
                    te = int(prog.get("total_epochs", 0))
                    model = str(prog.get("current_model") or item["task"]["model"])
                    status = str(prog.get("status", "running"))
                    f1 = prog.get("latest_val_f1", None)
                    acc = prog.get("latest_val_acc", None)
                    f1_txt = "NA" if f1 is None else f"{float(f1):.4f}"
                    acc_txt = "NA" if acc is None else f"{float(acc):.4f}"
                    gpu_rows[gpu_id] = (
                        f"GPU {gpu_id}: {status:8s} {model:18s} epoch {ep:>2}/{te:<2} "
                        f"| val_f1={f1_txt} val_acc={acc_txt}"
                    )
                except Exception:
                    gpu_rows[gpu_id] = f"GPU {gpu_id}: running  {item['task']['name']}"
            else:
                gpu_rows[gpu_id] = f"GPU {gpu_id}: running  {item['task']['name']}"
        lines.extend([gpu_rows[g] for g in sorted(gpu_rows.keys())])
        if pending:
            wait_preview = [f"{t['model']}_s{t['seed']}" for t in pending[:12]]
            lines.append("Waiting: " + ", ".join(wait_preview) + (" ..." if len(pending) > 12 else ""))
        print("\033[2J\033[H" + "\n".join(lines), flush=True)

    while done < total:
        while free_gpus and pending:
            task = pending.pop(0)
            gpu_id = free_gpus.pop(0)
            task["out"].mkdir(parents=True, exist_ok=True)
            task["status_path"].parent.mkdir(parents=True, exist_ok=True)
            cmd = [
                py_exec,
                str(script_path),
                "--worker_mode",
                "--worker_model",
                str(task["model"]),
                "--worker_seed",
                str(task["seed"]),
                "--data_root",
                str(args.data_root),
                "--output_dir",
                str(task["out"]),
                "--run_mode",
                "deep",
                "--deep_models",
                str(task["model"]),
                "--epochs",
                str(args.epochs),
                "--num_workers",
                str(args.num_workers),
                "--prefetch_factor",
                str(args.prefetch_factor),
                "--split_mode",
                str(args.split_mode),
                "--clip_mode",
                str(args.clip_mode),
                "--train_ratio",
                str(args.train_ratio),
                "--val_ratio",
                str(args.val_ratio),
                "--batch_size",
                str(args.batch_size),
                "--cache_dir",
                str(args.cache_dir),
            ]
            if args.disable_persistent_workers:
                cmd.append("--disable_persistent_workers")
            if args.best_config_path is not None:
                cmd.extend(["--best_config_path", str(args.best_config_path)])
            if args.skip_cache_warmup:
                cmd.append("--skip_cache_warmup")
            if args.overfit_single_batch_debug:
                cmd.append("--overfit_single_batch_debug")
                cmd.extend(["--overfit_debug_epochs", str(args.overfit_debug_epochs)])
                cmd.extend(["--overfit_debug_lr", str(args.overfit_debug_lr)])
            # Match HPO parallel data loading behavior: avoid huge preload per worker.
            cmd.append("--no_preload")

            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            env["TACACT_STATUS_FILE"] = str(task["status_path"])
            env["TACACT_GPU_ID"] = str(gpu_id)
            proc = subprocess.Popen(cmd, env=env, cwd=str(Path(__file__).resolve().parents[2]))
            running.append({"proc": proc, "gpu_id": gpu_id, "task": task})

        still_running: List[Dict[str, Any]] = []
        for item in running:
            ret = item["proc"].poll()
            if ret is None:
                still_running.append(item)
            else:
                free_gpus.append(int(item["gpu_id"]))
                done += 1
                if ret != 0:
                    print(f"[WARN] Task failed: {item['task']['name']} (code={ret})", flush=True)
                    st_path = Path(item["task"]["status_path"])
                    try:
                        st_path.write_text(
                            json.dumps(
                                {
                                    "status": "failed",
                                    "gpu_id": str(item["gpu_id"]),
                                    "task": item["task"]["name"],
                                    "return_code": int(ret),
                                    "last_update_ts": time.time(),
                                },
                                ensure_ascii=False,
                                indent=2,
                            ),
                            encoding="utf-8",
                        )
                    except Exception:
                        pass
        running = still_running
        _render_dashboard()
        if done < total:
            time.sleep(max(0.5, float(args.dashboard_interval)))

    print("[Main Parallel] All tasks completed. Building merged outputs...", flush=True)
    _build_parallel_merged_outputs(
        output_dir=args.output_dir,
        deep_models=deep_models,
        seed_list=seed_list,
    )
    print("[Main Parallel] Merged outputs completed.", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--train_ratio", type=float, default=0.7)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--clip_mode", choices=["weighted_center"], default="weighted_center")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--repeat_seeds", type=str, default="",
                        help="留空则单次运行(更快); 如 42,43,44 则多种子重复")
    parser.add_argument("--split_mode", choices=["subject", "random"], default="subject")
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--prefetch_factor", type=int, default=2,
                        help="DataLoader prefetch_factor when num_workers>0")
    parser.add_argument("--disable_persistent_workers", action="store_true",
                        help="Disable DataLoader persistent_workers even when num_workers>0")
    parser.add_argument("--output_dir", type=Path, default=Path("outputs"))
    parser.add_argument("--cache_dir", type=Path, default=Path(".cache_tacact_n80_weighted"))
    parser.add_argument("--run_mode", choices=["all", "traditional", "deep"], default="all")
    parser.add_argument("--traditional_models", type=str, default="SVM,RandomForest,XGBoost")
    parser.add_argument(
        "--deep_models",
        type=str,
        default="LeNet,ResNet18,EfficientNet_B0,LSTM,GRU,CNN_LSTM,LeNet_LSTM,TCN,Transformer",
    )
    parser.add_argument("--amp_infer", action="store_true")
    parser.add_argument("--bench_batch_sizes", type=str, default="1,32")
    parser.add_argument("--bench_iters", type=int, default=100)
    parser.add_argument("--merge_metrics_csvs", type=str, default="")
    parser.add_argument("--best_config_path", type=Path, default=None,
                        help="可选：读取自动搜索生成的 best_model_configs.json")
    parser.add_argument("--skip_cache_warmup", action="store_true",
                        help="跳过缓存预热(确信 .npy 已存在时使用)")
    parser.add_argument("--no_preload", action="store_true",
                        help="不预加载 24k 样本到内存，按需从磁盘读 .npy(省内存、省启动时间)")
    parser.add_argument("--parallel", action="store_true", help="Enable multi-process multi-GPU scheduler for main deep experiment.")
    parser.add_argument("--gpu_ids", type=str, default="0,1,2,3,4", help="Comma-separated GPU ids for --parallel.")
    parser.add_argument("--max_workers", type=int, default=5, help="Max concurrent workers for --parallel.")
    parser.add_argument("--dashboard_interval", type=float, default=1.0, help="Dashboard refresh interval (sec).")
    parser.add_argument("--overfit_single_batch_debug", action="store_true",
                        help="Run single-batch overfit debugging for one deep model only.")
    parser.add_argument("--overfit_debug_lr", type=float, default=1e-3,
                        help="Learning rate used by single-batch overfit debug mode.")
    parser.add_argument("--overfit_debug_epochs", type=int, default=300,
                        help="Epoch override used by single-batch overfit debug mode.")
    parser.add_argument("--worker_mode", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--worker_model", type=str, default="", help=argparse.SUPPRESS)
    parser.add_argument("--worker_seed", type=int, default=42, help=argparse.SUPPRESS)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.merge_metrics_csvs.strip():
        csv_paths = [Path(x.strip()) for x in args.merge_metrics_csvs.split(",") if x.strip()]
        merged_df = merge_metrics_csvs(csv_paths)
        merged_df.to_csv(args.output_dir / "metrics_merged.csv", index=False)
        merged_results = dataframe_to_results_dict(merged_df)
        deep_results, traditional_results = _split_results_for_paper(merged_results)

        # Main paper comparison: deep models only.
        if deep_results:
            # A) Legacy/original plots (kept for backward compatibility).
            save_scatter(deep_results, args.output_dir / "dl_main_accuracy_time_params_3d_merged.png")
            save_accuracy_vs_inference_bubble(
                deep_results, args.output_dir / "dl_main_accuracy_vs_inference_bubble_merged.png"
            )
            save_accuracy_vs_params_scatter(
                deep_results, args.output_dir / "dl_main_accuracy_vs_params_scatter_merged.png"
            )
            save_macrof1_vs_inference_bubble(
                deep_results, args.output_dir / "dl_main_macrof1_vs_inference_bubble_merged.png"
            )
            save_efficiency_score_bar(
                deep_results, args.output_dir / "dl_main_efficiency_score_bar_merged.png"
            )
            save_pareto_accuracy_inference(
                deep_results, args.output_dir / "dl_main_pareto_accuracy_inference_merged.png"
            )
            save_accuracy_vs_training_time(
                deep_results, args.output_dir / "dl_main_accuracy_vs_training_time_merged.png"
            )
            save_macrof1_vs_params_scatter(
                deep_results, args.output_dir / "dl_main_macrof1_vs_params_scatter_merged.png"
            )
            # B) New trade-off analysis plots (CVPR-style additions).
            save_dl_pareto_macrof1_vs_inference(
                deep_results, args.output_dir / "dl_pareto_macroF1_vs_inference_merged.png"
            )
            save_dl_family_tradeoff(
                deep_results, args.output_dir / "dl_family_tradeoff_merged.png"
            )
            save_dl_macrof1_vs_training_time(
                deep_results, args.output_dir / "dl_macroF1_vs_training_time_merged.png"
            )
            save_dl_params_vs_inference(
                deep_results, args.output_dir / "dl_params_vs_inference_merged.png"
            )
            save_pareto_all_seeds_from_df(
                merged_df[merged_df["category"].astype(str).str.lower() != "traditional"].copy()
                if "category" in merged_df.columns else merged_df.copy(),
                args.output_dir / "pareto_accuracy_all_seeds.png",
                y_col="accuracy",
                y_label="Test Accuracy (%)" if float(merged_df["accuracy"].max()) > 1.5 else "Test Accuracy",
                title_prefix="Pareto Frontier: Test Accuracy vs Inference Time",
            )
            save_pareto_all_seeds_from_df(
                merged_df[merged_df["category"].astype(str).str.lower() != "traditional"].copy()
                if "category" in merged_df.columns else merged_df.copy(),
                args.output_dir / "pareto_macroF1_all_seeds.png",
                y_col="macro_f1",
                y_label="Macro-F1 (%)" if float(merged_df["macro_f1"].max()) > 1.5 else "Macro-F1",
                title_prefix="Pareto Frontier: Macro-F1 vs Inference Time",
            )
            save_params_inference_all_seeds_from_df(
                merged_df[merged_df["category"].astype(str).str.lower() != "traditional"].copy()
                if "category" in merged_df.columns else merged_df.copy(),
                args.output_dir / "params_inference_all_seeds.png",
            )
            merged_seq_ok = save_dl_performance_vs_sequence_length(
                merged_df,
                args.output_dir / "dl_performance_vs_sequence_length_merged.png",
                metric_col="macro_f1",
            )
            if not merged_seq_ok:
                print("[WARN] Skip dl_performance_vs_sequence_length_merged.png: sequence-length field not found.")

        # Traditional baselines: separate compact reference outputs.
        _save_traditional_baseline_table_and_plot(traditional_results, args.output_dir, suffix="_merged")

        print(f"Saved merged metrics: {args.output_dir / 'metrics_merged.csv'}")
        print(f"Saved merged DL-main plot: {args.output_dir / 'dl_main_accuracy_time_params_3d_merged.png'}")
        return

    best_config_map: Dict[str, Dict[str, Dict[str, Any]]] = {"traditional": {}, "deep": {}}
    if args.best_config_path is not None:
        payload = json.loads(args.best_config_path.read_text(encoding="utf-8"))
        best_config_map["traditional"] = payload.get("traditional", {})
        best_config_map["deep"] = payload.get("deep", {})
        print(f"Loaded best configs from: {args.best_config_path}")

    dataset = create_optimized_dataset(
        args.data_root,
        n_frames=80,
        clip_mode=args.clip_mode,
        cache_dir=args.cache_dir,
        preload_cache=not args.no_preload,
    )
    print(f"Found {len(dataset)} samples")
    if not args.skip_cache_warmup:
        print("正在使用多进程预构建/检查数据缓存，首次运行耗时较长，请耐心等待...")
        warmup_cache(
            dataset,
            batch_size=128,
            num_workers=args.num_workers,
            prefetch_factor=args.prefetch_factor,
            shuffle=False,
            pin_memory=False,
            use_tqdm=True,
            tqdm_desc="Building Cache",
        )
        print("缓存就绪！后续数据加载将以光速进行。")
    else:
        print("已跳过缓存预热（假定 .npy 已存在）")
    traditional_models = parse_model_list(args.traditional_models, ["SVM", "RandomForest", "XGBoost"])
    deep_models = parse_model_list(
        args.deep_models,
        [
            "LeNet",
            "ResNet18",
            "MobileNet_V2",
            "EfficientNet_B0",
            "LSTM",
            "GRU",
            "CNN_LSTM",
            "LeNet_LSTM",
            "TCN",
            "Transformer",
        ],
    )
    if args.overfit_single_batch_debug:
        if args.run_mode != "deep":
            raise ValueError("--overfit_single_batch_debug requires --run_mode deep.")
        if len(deep_models) != 1:
            raise ValueError("--overfit_single_batch_debug requires exactly one deep model in --deep_models.")
        args.epochs = int(args.overfit_debug_epochs)
    print(f"Run mode: {args.run_mode} | traditional={traditional_models} | deep={deep_models}")

    if args.worker_mode:
        args.run_mode = "deep"
        args.deep_models = args.worker_model
        args.seed = int(args.worker_seed)
        args.repeat_seeds = ""

    if args.repeat_seeds.strip():
        seed_list = [int(x.strip()) for x in args.repeat_seeds.split(",") if x.strip()]
    else:
        seed_list = [args.seed]

    if args.parallel and not args.worker_mode:
        parallel_run_cfg = {
            "mode": "parallel_main_experiment",
            "model_list": {"traditional": traditional_models, "deep": deep_models},
            "seed_list": [int(s) for s in seed_list],
            "data_root": str(args.data_root),
            "output_dir": str(args.output_dir),
            "split_mode": str(args.split_mode),
            "n_frames": 80,
            "clip_mode": str(args.clip_mode),
            "epochs": int(args.epochs),
            "batch_size": int(args.batch_size),
            "num_workers": int(args.num_workers),
            "prefetch_factor": int(args.prefetch_factor),
            "gpu_ids": str(args.gpu_ids),
            "max_workers": int(args.max_workers),
            "best_config_path": str(args.best_config_path) if args.best_config_path is not None else "",
            "overfit_single_batch_debug": bool(args.overfit_single_batch_debug),
            "overfit_debug_lr": float(args.overfit_debug_lr),
            "overfit_debug_epochs": int(args.overfit_debug_epochs),
            "runtime": _runtime_info(),
        }
        (args.output_dir / "run_config.json").write_text(
            json.dumps(parallel_run_cfg, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        _run_parallel_main_experiment(args, deep_models, seed_list)
        return

    # Optional per-process status reporting for external live dashboard.
    status_file_env = os.environ.get("TACACT_STATUS_FILE", "").strip()
    status_path = Path(status_file_env) if status_file_env else None
    gpu_slot = os.environ.get("TACACT_GPU_ID", "").strip()
    queue_models_env = os.environ.get("TACACT_QUEUE_MODELS", "").strip()
    queue_models = [x.strip() for x in queue_models_env.split(",") if x.strip()]
    queue_total = int(os.environ.get("TACACT_QUEUE_TOTAL", str(len(queue_models) if queue_models else 0)))
    if queue_total <= 0:
        queue_total = len(queue_models)
    _write_status(
        status_path,
        {
            "status": "starting",
            "gpu_id": gpu_slot,
            "pid": int(os.getpid()),
            "queue_models": queue_models,
            "queue_total": int(queue_total),
            "queue_completed": 0,
            "current_model": None,
            "current_model_index": 0,
            "current_epoch": 0,
            "total_epochs": int(args.epochs),
            "latest_val_f1": None,
            "latest_val_acc": None,
            "seed": int(args.seed),
            "run_mode": str(args.run_mode),
            "last_update_ts": time.time(),
        },
    )

    bench_batch_sizes = [int(x.strip()) for x in args.bench_batch_sizes.split(",") if x.strip()]

    aggregated_rows: List[Dict[str, float]] = []
    aggregated_runtime_rows: List[Dict[str, float]] = []
    aggregated_final_split_rows: List[Dict[str, float | int | str]] = []
    aggregated_histories: Dict[str, List[Dict[str, List[float]]]] = {}

    for run_seed in seed_list:
        set_seed(run_seed)
        run_out = args.output_dir / f"{args.split_mode}_seed{run_seed}"
        run_out.mkdir(parents=True, exist_ok=True)

        train_indices, val_indices, test_indices = split_indices_3way(
            dataset,
            split_mode=args.split_mode,
            seed=run_seed,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
        )
        split_audit = build_split_audit(
            dataset,
            split_mode=args.split_mode,
            seed=run_seed,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            train_idx=train_indices,
            val_idx=val_indices,
            test_idx=test_indices,
        )
        save_split_audit(run_out / "split_audit.json", split_audit)
        print(f"Saved split audit: {run_out / 'split_audit.json'}")
        train_set = Subset(dataset, train_indices)
        val_set = Subset(dataset, val_indices)
        test_set = Subset(dataset, test_indices)
        class_names = _class_names(12)
        run_cfg = {
            "model_list": {"traditional": traditional_models, "deep": deep_models},
            "seed": int(run_seed),
            "data_root": str(args.data_root),
            "output_dir": str(run_out),
            "split_mode": str(args.split_mode),
            "split_seed": int(run_seed),
            "n_frames": 80,
            "threshold_method": "mean_std",
            "threshold_k": 3.0,
            "background_frames": 5,
            "clip_mode": str(args.clip_mode),
            "batch_size": int(args.batch_size),
            "epochs": int(args.epochs),
            "optimizer": "AdamW",
            "scheduler_default": "ReduceLROnPlateau",
            "scheduler_transformer": "CosineAnnealingWarmRestarts",
            "best_config_path": str(args.best_config_path) if args.best_config_path is not None else "",
            "model_kwargs": best_config_map.get("deep", {}),
            "runtime": _runtime_info(),
        }
        (run_out / "run_config.json").write_text(json.dumps(run_cfg, ensure_ascii=False, indent=2), encoding="utf-8")

        data_protocol = {
            "dataset_name": "TacAct",
            "num_samples_total": int(len(dataset)),
            "num_classes": 12,
            "class_names": class_names,
            "input_shape": [80, 32, 32],
            "train_count": int(len(train_indices)),
            "val_count": int(len(val_indices)),
            "test_count": int(len(test_indices)),
            "train_subjects": split_audit["splits"]["train"]["subjects"],
            "val_subjects": split_audit["splits"]["val"]["subjects"],
            "test_subjects": split_audit["splits"]["test"]["subjects"],
            "split_seed": int(run_seed),
            "preprocessing": {
                "n_frames": 80,
                "threshold_method": "mean_std",
                "threshold_k": 3.0,
                "background_frames": 5,
                "clip_mode": str(args.clip_mode),
            },
            "cache_dir": str(args.cache_dir),
        }
        (run_out / "data_protocol.json").write_text(
            json.dumps(data_protocol, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        test_meta_ordered = [dataset.samples[i] for i in test_indices]

        device = get_device()

        results: Dict[str, Dict[str, float]] = {}
        confusion_mats: Dict[str, np.ndarray] = {}
        histories: Dict[str, Dict[str, List[float]]] = {}
        per_class_f1: Dict[str, np.ndarray] = {}
        detail_rows: List[Dict[str, float | int | str]] = []
        training_history_rows: List[Dict[str, float | int | str]] = []
        selection_rows: List[Dict[str, float | int | str]] = []
        confusion_rows: List[Dict[str, float | int | str]] = []
        model_cfg_rows: List[Dict[str, float | int | str]] = []
        pred_rows: List[Dict[str, float | int | str]] = []
        hpo_trace_rows: List[Dict[str, float | int | str]] = []
        final_split_rows: List[Dict[str, float | int | str]] = []
        traditional_runtime_s, deep_runtime_s = 0.0, 0.0

        if args.run_mode in ("all", "traditional"):
            x_train, y_train = subset_to_numpy(dataset, train_set)
            x_val, y_val = subset_to_numpy(dataset, val_set)
            x_test, y_test = subset_to_numpy(dataset, test_set)
            x_train_full = x_train
            y_train_full = y_train
        else:
            x_train_full = y_train_full = x_test = y_test = None

        t0_traditional = time.perf_counter()
        for model_name in traditional_models:
            if args.run_mode not in ("all", "traditional"):
                break
            try:
                traditional_cfg = best_config_map["traditional"].get(model_name, {}).get("params", {})
                if not traditional_cfg:
                    traditional_cfg = best_config_map["traditional"].get(model_name.lower(), {}).get("params", {})
                traditional_entry = best_config_map["traditional"].get(model_name, {})
                if not traditional_entry:
                    traditional_entry = best_config_map["traditional"].get(model_name.lower(), {})
                clf = ModelFactory.build_traditional(model_name, **traditional_cfg)
                fit_st = time.perf_counter()
                clf.fit(x_train_full, y_train_full)
                train_seconds = time.perf_counter() - fit_st
                pred = clf.predict(x_test)
                acc = float((pred == y_test).mean())
                p_cls, r_cls, f1_cls = per_class_prf(y_test, pred, n_classes=12)
                support = np.bincount(y_test.astype(np.int64), minlength=12)
                inf_ms = benchmark_sklearn(clf, x_test)
                params = float(count_sklearn_params(clf))
                results[model_name] = {
                    "category": "traditional",
                    "accuracy": acc,
                    "macro_f1": float(np.nanmean(f1_cls)),
                    "macro_precision": float(np.nanmean(p_cls)),
                    "macro_recall": float(np.nanmean(r_cls)),
                    "training_seconds": float(train_seconds),
                    "inference_ms": inf_ms,
                    "params": params,
                    "params_m": params / 1e6,
                    "best_epoch": np.nan,
                    "best_val_loss": np.nan,
                    "best_val_acc": np.nan,
                    "best_val_f1": np.nan,
                }
                cm = confusion_matrix_np(y_test, pred, n_classes=12)
                confusion_mats[model_name] = cm
                per_class_f1[model_name] = f1_cls
                for i, (yt, yp) in enumerate(zip(y_test, pred)):
                    sample_meta = test_meta_ordered[i]
                    pred_rows.append(
                        {
                            "model": model_name,
                            "seed": int(run_seed),
                            "sample_id": int(test_indices[i]),
                            "subject_id": int(sample_meta.subject),
                            "true_label": int(yt),
                            "true_class": class_names[int(yt)],
                            "pred_label": int(yp),
                            "pred_class": class_names[int(yp)],
                            "correct": int(int(yt) == int(yp)),
                            "top1_confidence": np.nan,
                            **{f"prob_{k}": np.nan for k in range(12)},
                        }
                    )
                for c in range(12):
                    detail_rows.append(
                        {
                            "model": model_name,
                            "seed": int(run_seed),
                            "class_id": int(c),
                            "class_name": f"class_{c}",
                            "precision": float(p_cls[c]),
                            "recall": float(r_cls[c]),
                            "f1": float(f1_cls[c]),
                            "support": int(support[c]),
                        }
                    )
                for t in range(12):
                    for p in range(12):
                        confusion_rows.append(
                            {
                                "model": model_name,
                                "seed": int(run_seed),
                                "true_label": int(t),
                                "pred_label": int(p),
                                "count": int(cm[t, p]),
                            }
                        )
                selection_rows.append(
                    {
                        "model": model_name,
                        "seed": int(run_seed),
                        "selected_epoch": np.nan,
                        "selection_metric": "n/a_traditional",
                        "best_val_f1": np.nan,
                        "best_val_acc": np.nan,
                        "best_val_loss": np.nan,
                        "test_accuracy": float(acc),
                        "test_macro_f1": float(np.nanmean(f1_cls)),
                        "checkpoint_path": "",
                    }
                )
                final_split_rows.append(
                    {
                        "model": model_name,
                        "display_name": model_display_name(model_name),
                        "seed": int(run_seed),
                        "selected_epoch": np.nan,
                        "train_loss_at_selected": np.nan,
                        "train_acc_at_selected": np.nan,
                        "train_f1_at_selected": np.nan,
                        "val_loss_at_selected": np.nan,
                        "val_acc_at_selected": np.nan,
                        "val_f1_at_selected": np.nan,
                        "test_accuracy": float(acc),
                        "test_macro_f1": float(np.nanmean(f1_cls)),
                        "test_macro_precision": float(np.nanmean(p_cls)),
                        "test_macro_recall": float(np.nanmean(r_cls)),
                        "params_m": float(params / 1e6),
                        "train_time_sec": float(train_seconds),
                        "inference_ms": float(inf_ms),
                    }
                )
                model_cfg_rows.append(
                    {
                        "model": model_name,
                        "display_name": model_name,
                        "seed": int(run_seed),
                        "lr": np.nan,
                        "weight_decay": np.nan,
                        "batch_size": np.nan,
                        "epochs": int(args.epochs),
                        "optimizer": "n/a_traditional",
                        "scheduler": "n/a_traditional",
                        "model_kwargs_json": json.dumps(traditional_cfg, ensure_ascii=False),
                        "param_count": float(params),
                        "best_config_source": "best_config_path" if args.best_config_path is not None else "default",
                    }
                )
                hpo_trace_rows.append(
                    {
                        "model": model_name,
                        "seed": int(run_seed),
                        "best_config_path": str(args.best_config_path) if args.best_config_path is not None else "",
                        "hpo_trial_id": traditional_entry.get("trial_id", np.nan),
                        "hpo_best_val_f1": traditional_entry.get("best_val_f1", np.nan),
                        "hpo_best_val_acc": traditional_entry.get("best_val_acc", np.nan),
                        "main_best_val_f1": np.nan,
                        "main_test_macro_f1": float(np.nanmean(f1_cls)),
                        "config_json": json.dumps(traditional_cfg, ensure_ascii=False),
                    }
                )
                print(f"{model_name}: acc={acc * 100:.2f}%")
            except Exception as e:
                print(f"[WARN] Skip {model_name}: {e}")
        if args.run_mode in ("all", "traditional"):
            traditional_runtime_s = time.perf_counter() - t0_traditional

        t0_deep = time.perf_counter()
        deep_model_total = len(deep_models)
        deep_completed = 0
        for model_idx, model_name in enumerate(deep_models, start=1):
            if args.run_mode not in ("all", "deep"):
                break
            _write_status(
                status_path,
                {
                    "status": "running",
                    "gpu_id": gpu_slot,
                    "pid": int(os.getpid()),
                    "queue_models": queue_models,
                    "queue_total": int(queue_total if queue_total > 0 else deep_model_total),
                    "queue_completed": int(deep_completed),
                    "current_model": str(model_name),
                    "current_model_index": int(model_idx),
                    "current_epoch": 1,
                    "total_epochs": int(args.epochs),
                    "latest_val_f1": None,
                    "latest_val_acc": None,
                    "last_update_ts": time.time(),
                },
            )
            deep_cfg = best_config_map["deep"].get(model_name, {}).get("params", {})
            if not deep_cfg:
                deep_cfg = best_config_map["deep"].get(model_name.lower(), {}).get("params", {})
            deep_entry = best_config_map["deep"].get(model_name, {})
            if not deep_entry:
                deep_entry = best_config_map["deep"].get(model_name.lower(), {})
            if "batch_size" in deep_cfg and deep_cfg.get("batch_size") is not None:
                try:
                    model_batch_size = int(deep_cfg.get("batch_size"))
                except Exception:
                    model_batch_size = int(args.batch_size)
                if model_batch_size <= 0:
                    model_batch_size = int(args.batch_size)
                print(f"[MODEL={model_name}] Using batch_size={model_batch_size} (from best config)")
            else:
                model_batch_size = int(args.batch_size)
                print(f"[MODEL={model_name}] Using batch_size={model_batch_size} (default)")

            train_loader, val_loader, test_loader = make_three_loaders(
                train_set=train_set,
                val_set=val_set,
                test_set=test_set,
                batch_size=model_batch_size,
                num_workers=args.num_workers,
                pin_memory=True,
                prefetch_factor=args.prefetch_factor,
                persistent_workers=(False if args.disable_persistent_workers else None),
            )
            model_kwargs = {
                k: v for k, v in deep_cfg.items()
                if k in {"d_model", "nhead", "dim_feedforward", "pooling", "norm_first",
                         "dropout", "num_channels", "lstm_hidden", "hidden_size", "num_layers",
                         "input_proj_dim", "use_last_only", "feature_dim", "encoder_hidden_dim", "bidirectional"}
            }
            train_kwargs = {
                "lr_override": deep_cfg.get("lr"),
                "weight_decay_override": deep_cfg.get("weight_decay"),
            }
            model, cat = ModelFactory.build_torch(model_name, **model_kwargs)
            if args.overfit_single_batch_debug:
                print(f"[Single-Batch Debug] Running single-batch overfit test for {model_name}")
                train_kwargs["overfit_single_batch_debug"] = True
                train_kwargs["overfit_debug_lr"] = float(args.overfit_debug_lr)
                train_kwargs["weight_decay_override"] = 0.0
                test_loader_for_training = None
            else:
                test_loader_for_training = test_loader
            def _epoch_progress_cb(p: Dict[str, float], _model_name: str = str(model_name), _model_idx: int = int(model_idx)) -> None:
                _write_status(
                    status_path,
                    {
                        "status": "running",
                        "gpu_id": gpu_slot,
                        "pid": int(os.getpid()),
                        "queue_models": queue_models,
                        "queue_total": int(queue_total if queue_total > 0 else deep_model_total),
                        "queue_completed": int(deep_completed),
                        "current_model": _model_name,
                        "current_model_index": _model_idx,
                        "current_epoch": int(p.get("epoch", 0.0)),
                        "total_epochs": int(p.get("total_epochs", float(args.epochs))),
                        "latest_val_f1": float(p.get("val_f1", float("nan"))),
                        "latest_val_acc": float(p.get("val_acc", float("nan"))),
                        "last_update_ts": time.time(),
                    },
                )
            histories[model_name] = train_torch_model(model, train_loader, val_loader, test_loader_for_training, epochs=args.epochs,
                                                     device=device, progress_callback=_epoch_progress_cb, **train_kwargs)
            aggregated_histories.setdefault(model_name, []).append(histories[model_name])
            train_seconds = float(histories[model_name].get("cum_time_s", [0.0])[-1]) if histories[model_name].get("cum_time_s") else 0.0
            hist = histories[model_name]
            n_ep = int(len(hist.get("train_loss", [])))
            for ep_idx in range(n_ep):
                training_history_rows.append(
                    {
                        "model": model_name,
                        "seed": int(run_seed),
                        "epoch": int(ep_idx + 1),
                        "train_loss": float(hist["train_loss"][ep_idx]) if ep_idx < len(hist.get("train_loss", [])) else np.nan,
                        "train_acc": float(hist["train_acc"][ep_idx]) if ep_idx < len(hist.get("train_acc", [])) else np.nan,
                        "train_f1": float(hist["train_f1"][ep_idx]) if ep_idx < len(hist.get("train_f1", [])) else np.nan,
                        "val_loss": float(hist["val_loss"][ep_idx]) if ep_idx < len(hist.get("val_loss", [])) else np.nan,
                        "val_acc": float(hist["val_acc"][ep_idx]) if ep_idx < len(hist.get("val_acc", [])) else np.nan,
                        "val_f1": float(hist["val_f1"][ep_idx]) if ep_idx < len(hist.get("val_f1", [])) else np.nan,
                        "test_loss": float(hist["test_loss"][ep_idx]) if ep_idx < len(hist.get("test_loss", [])) else np.nan,
                        "test_acc": float(hist["test_acc"][ep_idx]) if ep_idx < len(hist.get("test_acc", [])) else np.nan,
                        "test_f1": float(hist["test_f1"][ep_idx]) if ep_idx < len(hist.get("test_f1", [])) else np.nan,
                        "lr": float(hist["lr"][ep_idx]) if ep_idx < len(hist.get("lr", [])) else np.nan,
                        "epoch_time_sec": float(hist["epoch_time_s"][ep_idx]) if ep_idx < len(hist.get("epoch_time_s", [])) else np.nan,
                    }
                )

            best_epoch = int(hist.get("best_epoch", [0])[-1]) if hist.get("best_epoch") else -1
            best_idx = max(0, best_epoch - 1)
            best_val_loss = (
                float(hist.get("val_loss", [np.nan])[best_idx]) if 0 <= best_idx < len(hist.get("val_loss", [])) else float("nan")
            )
            best_val_acc = (
                float(hist.get("val_acc", [np.nan])[best_idx]) if 0 <= best_idx < len(hist.get("val_acc", [])) else float("nan")
            )
            best_val_f1 = (
                float(hist.get("val_f1", [np.nan])[best_idx]) if 0 <= best_idx < len(hist.get("val_f1", [])) else float("nan")
            )
            best_train_loss = (
                float(hist.get("train_loss", [np.nan])[best_idx]) if 0 <= best_idx < len(hist.get("train_loss", [])) else float("nan")
            )
            best_train_acc = (
                float(hist.get("train_acc", [np.nan])[best_idx]) if 0 <= best_idx < len(hist.get("train_acc", [])) else float("nan")
            )
            best_train_f1 = (
                float(hist.get("train_f1", [np.nan])[best_idx]) if 0 <= best_idx < len(hist.get("train_f1", [])) else float("nan")
            )

            ckpt_dir = run_out / "checkpoints"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            ckpt_path = ckpt_dir / f"{model_name}_seed{run_seed}_best.pt"
            torch.save(model.state_dict(), ckpt_path)

            model.eval()
            ys = []
            ps = []
            confs = []
            probs_rows: List[np.ndarray] = []
            with torch.no_grad():
                for x, y in test_loader:
                    logits = model(x.to(device))
                    prob = torch.softmax(logits, dim=1).cpu().numpy()
                    pred = prob.argmax(axis=1)
                    conf = prob.max(axis=1)
                    ys.append(y.numpy())
                    ps.append(pred)
                    confs.append(conf)
                    probs_rows.append(prob)
            y_true, y_pred = np.concatenate(ys), np.concatenate(ps)
            y_conf = np.concatenate(confs) if confs else np.array([], dtype=np.float64)
            y_prob = np.concatenate(probs_rows, axis=0) if probs_rows else np.empty((0, 12), dtype=np.float64)
            p_cls, r_cls, f1_cls = per_class_prf(y_true, y_pred, n_classes=12)
            support = np.bincount(y_true.astype(np.int64), minlength=12)
            params = float(count_parameters(model))
            deploy_bench: Dict[str, float] = {}
            try:
                sample_batch = next(iter(test_loader))[0]
                deploy_bench = benchmark_torch_gpu_deploy(
                    model=model,
                    sample_batch=sample_batch,
                    device=device,
                    batch_sizes=bench_batch_sizes,
                    iters=args.bench_iters,
                    warmup=max(10, args.bench_iters // 5),
                    amp=bool(args.amp_infer),
                )
            except Exception as e:
                print(f"[WARN] GPU deploy benchmark skipped for {model_name}: {e}")
            results[model_name] = {
                "category": cat,
                "accuracy": float((y_true == y_pred).mean()),
                "macro_f1": float(np.nanmean(f1_cls)),
                "macro_precision": float(np.nanmean(p_cls)),
                "macro_recall": float(np.nanmean(r_cls)),
                "training_seconds": train_seconds,
                "inference_ms": benchmark_torch_model_only(model, test_loader, device),
                "params": params,
                "params_m": params / 1e6,
                "best_epoch": float(best_epoch),
                "best_val_loss": float(best_val_loss),
                "best_val_acc": float(best_val_acc),
                "best_val_f1": float(best_val_f1),
                **deploy_bench,
            }
            cm = confusion_matrix_np(y_true, y_pred, n_classes=12)
            confusion_mats[model_name] = cm
            per_class_f1[model_name] = f1_cls
            for i, (yt, yp) in enumerate(zip(y_true, y_pred)):
                sample_meta = test_meta_ordered[i]
                prob_vec = y_prob[i] if i < len(y_prob) else np.full((12,), np.nan, dtype=np.float64)
                pred_rows.append(
                    {
                        "model": model_name,
                        "seed": int(run_seed),
                        "sample_id": int(test_indices[i]),
                        "subject_id": int(sample_meta.subject),
                        "true_label": int(yt),
                        "true_class": class_names[int(yt)],
                        "pred_label": int(yp),
                        "pred_class": class_names[int(yp)],
                        "correct": int(int(yt) == int(yp)),
                        "top1_confidence": float(y_conf[i]) if i < len(y_conf) else np.nan,
                        **{f"prob_{k}": float(prob_vec[k]) for k in range(12)},
                    }
                )
            for c in range(12):
                detail_rows.append(
                    {
                        "model": model_name,
                        "seed": int(run_seed),
                        "class_id": int(c),
                        "class_name": f"class_{c}",
                        "precision": float(p_cls[c]),
                        "recall": float(r_cls[c]),
                        "f1": float(f1_cls[c]),
                        "support": int(support[c]),
                    }
                )
            for t in range(12):
                for p in range(12):
                    confusion_rows.append(
                        {
                            "model": model_name,
                            "seed": int(run_seed),
                            "true_label": int(t),
                            "pred_label": int(p),
                            "count": int(cm[t, p]),
                        }
                    )
            selection_rows.append(
                {
                    "model": model_name,
                    "seed": int(run_seed),
                    "selected_epoch": int(best_epoch),
                    "selection_metric": "best_val_f1_max",
                    "best_val_f1": float(best_val_f1),
                    "best_val_acc": float(best_val_acc),
                    "best_val_loss": float(best_val_loss),
                    "test_accuracy": float((y_true == y_pred).mean()),
                    "test_macro_f1": float(np.nanmean(f1_cls)),
                    "checkpoint_path": str(ckpt_path),
                }
            )
            final_split_rows.append(
                {
                    "model": model_name,
                    "display_name": model_display_name(model_name),
                    "seed": int(run_seed),
                    "selected_epoch": int(best_epoch),
                    "train_loss_at_selected": float(best_train_loss),
                    "train_acc_at_selected": float(best_train_acc),
                    "train_f1_at_selected": float(best_train_f1),
                    "val_loss_at_selected": float(best_val_loss),
                    "val_acc_at_selected": float(best_val_acc),
                    "val_f1_at_selected": float(best_val_f1),
                    "test_accuracy": float((y_true == y_pred).mean()),
                    "test_macro_f1": float(np.nanmean(f1_cls)),
                    "test_macro_precision": float(np.nanmean(p_cls)),
                    "test_macro_recall": float(np.nanmean(r_cls)),
                    "params_m": float(params / 1e6),
                    "train_time_sec": float(train_seconds),
                    "inference_ms": float(results[model_name]["inference_ms"]),
                }
            )
            model_cfg_rows.append(
                {
                    "model": model_name,
                    "display_name": model_name,
                    "seed": int(run_seed),
                    "lr": float(deep_cfg.get("lr", np.nan)) if deep_cfg else np.nan,
                    "weight_decay": float(deep_cfg.get("weight_decay", np.nan)) if deep_cfg else np.nan,
                    "batch_size": int(model_batch_size),
                    "epochs": int(args.epochs),
                    "optimizer": "AdamW",
                    "scheduler": _scheduler_name_for_model(model_name),
                    "model_kwargs_json": json.dumps(model_kwargs, ensure_ascii=False),
                    "param_count": float(params),
                    "best_config_source": "best_config_path" if args.best_config_path is not None else "default",
                }
            )
            hpo_trace_rows.append(
                {
                    "model": model_name,
                    "seed": int(run_seed),
                    "best_config_path": str(args.best_config_path) if args.best_config_path is not None else "",
                    "hpo_trial_id": deep_entry.get("trial_id", np.nan),
                    "hpo_best_val_f1": deep_entry.get("best_val_f1", np.nan),
                    "hpo_best_val_acc": deep_entry.get("best_val_acc", np.nan),
                    "main_best_val_f1": float(best_val_f1),
                    "main_test_macro_f1": float(np.nanmean(f1_cls)),
                    "config_json": json.dumps(deep_cfg, ensure_ascii=False),
                }
            )
            print(f"{model_name}: acc={results[model_name]['accuracy'] * 100:.2f}%")
            deep_completed += 1
            _write_status(
                status_path,
                {
                    "status": "running",
                    "gpu_id": gpu_slot,
                    "pid": int(os.getpid()),
                    "queue_models": queue_models,
                    "queue_total": int(queue_total if queue_total > 0 else deep_model_total),
                    "queue_completed": int(deep_completed),
                    "current_model": str(model_name),
                    "current_model_index": int(model_idx),
                    "current_epoch": int(len(histories[model_name].get("train_loss", []))),
                    "total_epochs": int(args.epochs),
                    "latest_val_f1": float(np.nanmean(f1_cls)),
                    "latest_val_acc": float(results[model_name]["accuracy"]),
                    "last_update_ts": time.time(),
                },
            )

        if args.run_mode in ("all", "deep"):
            deep_runtime_s = time.perf_counter() - t0_deep

        if not results:
            raise RuntimeError("No model produced results.")

        df = pd.DataFrame.from_dict(results, orient="index")
        df.to_csv(run_out / "metrics.csv", index_label="model")
        pd.DataFrame([
            {"segment": "traditional", "seconds": traditional_runtime_s},
            {"segment": "deep", "seconds": deep_runtime_s},
            {"segment": "total", "seconds": traditional_runtime_s + deep_runtime_s},
        ]).to_csv(run_out / "runtime_summary.csv", index=False)

        detail_df = pd.DataFrame(detail_rows)
        detail_df.to_csv(run_out / "per_class_metrics.csv", index=False)
        try:
            detail_df.to_excel(run_out / "per_class_metrics.xlsx", index=False)
        except Exception as e:
            print(f"[WARN] Could not export Excel metrics: {e}")
        pd.DataFrame(confusion_rows).to_csv(run_out / "confusion_matrix.csv", index=False)
        pd.DataFrame(training_history_rows).to_csv(run_out / "training_history.csv", index=False)
        pd.DataFrame(selection_rows).to_csv(run_out / "selection_summary.csv", index=False)
        pd.DataFrame(model_cfg_rows).to_csv(run_out / "model_config_summary.csv", index=False)
        pd.DataFrame(pred_rows).to_csv(run_out / "predictions.csv", index=False)
        pd.DataFrame(hpo_trace_rows).to_csv(run_out / "hpo_to_main_trace.csv", index=False)
        pd.DataFrame(final_split_rows).to_csv(run_out / "final_split_metrics.csv", index=False)
        training_df = pd.DataFrame(training_history_rows)
        selection_df = pd.DataFrame(selection_rows)
        _save_loss_curves_per_model_seed(training_df, selection_df, run_out)
        _save_val_loss_all_models(training_df, run_out / "val_loss_all_models.png")
        _save_loss_curves_grid(training_df, run_out / "loss_curves_grid.png")

        deep_results, traditional_results = _split_results_for_paper(results)
        if deep_results:
            # A) Legacy/original plots (kept).
            save_scatter(deep_results, run_out / "dl_main_accuracy_time_params_3d.png")
        for name, cm in confusion_mats.items():
            save_confusion_matrix(cm, name, run_out / f"confusion_{name}.png")
        save_confusion_comparison(confusion_mats, run_out / "confusion_comparative.png")
        save_training_curves(histories, run_out / "training_history_overlay.png")
        save_per_model_accuracy_loss_curves(histories, run_out, expected_models=deep_models)
        save_all_models_loss_overlay(
            histories,
            run_out / "all_models_train_loss_vs_epoch.png",
            loss_key="train_loss",
            title="Deep Models: Training Loss vs Epoch",
        )
        save_all_models_loss_overlay(
            histories,
            run_out / "all_models_val_loss_vs_epoch.png",
            loss_key="val_loss",
            title="Deep Models: Validation Loss vs Epoch",
        )
        save_convergence_diagnostics(histories, run_out / "training_convergence_diagnostics.png")
        save_per_class_f1_bars(per_class_f1, run_out / "per_class_f1_grouped.png")
        if deep_results:
            save_radar_top3(deep_results, run_out / "dl_main_radar_top3.png")
            # Main paper legacy figures: deep-only.
            save_accuracy_vs_inference_bubble(deep_results, run_out / "dl_main_accuracy_vs_inference_bubble.png")
            save_accuracy_vs_params_scatter(deep_results, run_out / "dl_main_accuracy_vs_params_scatter.png")
            save_macrof1_vs_inference_bubble(deep_results, run_out / "dl_main_macrof1_vs_inference_bubble.png")
            save_efficiency_score_bar(deep_results, run_out / "dl_main_efficiency_score_bar.png")
            save_pareto_accuracy_inference(deep_results, run_out / "dl_main_pareto_accuracy_inference.png")
            save_accuracy_vs_training_time(deep_results, run_out / "dl_main_accuracy_vs_training_time.png")
            save_macrof1_vs_params_scatter(deep_results, run_out / "dl_main_macrof1_vs_params_scatter.png")

            # B) New trade-off analysis figures.
            save_dl_pareto_macrof1_vs_inference(
                deep_results, run_out / "dl_pareto_macroF1_vs_inference.png"
            )
            save_dl_family_tradeoff(
                deep_results, run_out / "dl_family_tradeoff.png"
            )
            save_dl_macrof1_vs_training_time(
                deep_results, run_out / "dl_macroF1_vs_training_time.png"
            )
            save_dl_params_vs_inference(
                deep_results, run_out / "dl_params_vs_inference.png"
            )
            run_seq_ok = save_dl_performance_vs_sequence_length(
                df.reset_index().rename(columns={"index": "model"}),
                run_out / "dl_performance_vs_sequence_length.png",
                metric_col="macro_f1",
            )
            if not run_seq_ok:
                print("[WARN] Skip dl_performance_vs_sequence_length.png: sequence-length field not found.")

        # Separate compact baseline reference for traditional ML.
        _save_traditional_baseline_table_and_plot(traditional_results, run_out)

        for model_name, m in results.items():
            aggregated_rows.append({
                "split_mode": args.split_mode,
                "seed": float(run_seed),
                "model": model_name,
                "category": m["category"],
                "accuracy": float(m["accuracy"]),
                "macro_f1": float(m["macro_f1"]),
                "macro_precision": float(m["macro_precision"]),
                "macro_recall": float(m["macro_recall"]),
                "training_seconds": float(m.get("training_seconds", 0.0)),
                "inference_ms": float(m["inference_ms"]),
                "params": float(m["params"]),
                "params_m": float(m["params_m"]),
            })
        aggregated_final_split_rows.extend(final_split_rows)
        aggregated_runtime_rows.append({
            "split_mode": args.split_mode,
            "seed": float(run_seed),
            "traditional_seconds": float(traditional_runtime_s),
            "deep_seconds": float(deep_runtime_s),
            "total_seconds": float(traditional_runtime_s + deep_runtime_s),
        })

        print(f"Saved metrics: {run_out / 'metrics.csv'}")
        print(f"Saved runtime summary: {run_out / 'runtime_summary.csv'}")

    agg_df = pd.DataFrame(aggregated_rows)
    agg_df.to_csv(args.output_dir / "metrics_repeated.csv", index=False)
    summary = agg_df.groupby(["split_mode", "model", "category"], as_index=False).agg(
        accuracy_mean=("accuracy", "mean"),
        accuracy_std=("accuracy", "std"),
        macro_f1_mean=("macro_f1", "mean"),
        macro_f1_std=("macro_f1", "std"),
        macro_precision_mean=("macro_precision", "mean"),
        macro_precision_std=("macro_precision", "std"),
        macro_recall_mean=("macro_recall", "mean"),
        macro_recall_std=("macro_recall", "std"),
        training_seconds_mean=("training_seconds", "mean"),
        training_seconds_std=("training_seconds", "std"),
        inference_ms_mean=("inference_ms", "mean"),
        inference_ms_std=("inference_ms", "std"),
        params_m=("params_m", "first"),
    )
    summary.to_csv(args.output_dir / "metrics_summary.csv", index=False)

    deep_summary = summary[summary["category"] != "traditional"].copy()
    traditional_summary = summary[summary["category"] == "traditional"].copy()
    deep_summary.to_csv(args.output_dir / "metrics_summary_deep_main.csv", index=False)
    traditional_summary.to_csv(args.output_dir / "metrics_summary_traditional_baseline.csv", index=False)

    if not deep_summary.empty:
        # Legacy aggregated summary figures.
        save_summary_bar_with_error(
            summary_df=deep_summary,
            save_path=args.output_dir / "dl_main_accuracy_summary_bar.png",
            metric_col="accuracy_mean",
            error_col="accuracy_std",
            title="Deep Learning Models: Accuracy Across Seeds",
            ylabel="Accuracy",
        )
        save_summary_bar_with_error(
            summary_df=deep_summary,
            save_path=args.output_dir / "dl_main_macro_f1_summary_bar.png",
            metric_col="macro_f1_mean",
            error_col="macro_f1_std",
            title="Deep Learning Models: Macro-F1 Across Seeds",
            ylabel="Macro-F1",
        )
        # New aggregated trade-off figures (mean across seeds).
        deep_tradeoff_df = deep_summary.rename(
            columns={
                "accuracy_mean": "accuracy",
                "macro_f1_mean": "macro_f1",
                "training_seconds_mean": "training_seconds",
                "inference_ms_mean": "inference_ms",
            }
        )
        deep_tradeoff_results: Dict[str, Dict[str, float]] = {}
        for _, row in deep_tradeoff_df.iterrows():
            deep_tradeoff_results[str(row["model"])] = {
                "category": str(row.get("category", "unknown")),
                "accuracy": float(row.get("accuracy", np.nan)),
                "macro_f1": float(row.get("macro_f1", np.nan)),
                "training_seconds": float(row.get("training_seconds", np.nan)),
                "inference_ms": float(row.get("inference_ms", np.nan)),
                "params_m": float(row.get("params_m", np.nan)),
            }
        save_dl_pareto_macrof1_vs_inference(
            deep_tradeoff_results, args.output_dir / "dl_pareto_macroF1_vs_inference.png"
        )
        save_dl_family_tradeoff(
            deep_tradeoff_results, args.output_dir / "dl_family_tradeoff.png"
        )
        save_dl_macrof1_vs_training_time(
            deep_tradeoff_results, args.output_dir / "dl_macroF1_vs_training_time.png"
        )
        save_dl_params_vs_inference(
            deep_tradeoff_results, args.output_dir / "dl_params_vs_inference.png"
        )
        save_pareto_all_seeds_from_df(
            agg_df[agg_df["category"].astype(str).str.lower() != "traditional"].copy(),
            args.output_dir / "pareto_accuracy_all_seeds.png",
            y_col="accuracy",
            y_label="Test Accuracy",
            title_prefix="Pareto Frontier: Test Accuracy vs Inference Time",
        )
        save_pareto_all_seeds_from_df(
            agg_df[agg_df["category"].astype(str).str.lower() != "traditional"].copy(),
            args.output_dir / "pareto_macroF1_all_seeds.png",
            y_col="macro_f1",
            y_label="Macro-F1",
            title_prefix="Pareto Frontier: Macro-F1 vs Inference Time",
        )
        save_params_inference_all_seeds_from_df(
            agg_df[agg_df["category"].astype(str).str.lower() != "traditional"].copy(),
            args.output_dir / "params_inference_all_seeds.png",
        )
        summary_seq_ok = save_dl_performance_vs_sequence_length(
            deep_summary,
            args.output_dir / "dl_performance_vs_sequence_length.png",
            metric_col="macro_f1_mean",
        )
        if not summary_seq_ok:
            print("[WARN] Skip dl_performance_vs_sequence_length.png: sequence-length field not found in summary.")
    if not traditional_summary.empty:
        save_summary_bar_with_error(
            summary_df=traditional_summary,
            save_path=args.output_dir / "traditional_baseline_accuracy_summary_bar.png",
            metric_col="accuracy_mean",
            error_col="accuracy_std",
            title="Traditional ML Baselines (Reference): Accuracy",
            ylabel="Accuracy",
        )
        save_summary_bar_with_error(
            summary_df=traditional_summary,
            save_path=args.output_dir / "traditional_baseline_macro_f1_summary_bar.png",
            metric_col="macro_f1_mean",
            error_col="macro_f1_std",
            title="Traditional ML Baselines (Reference): Macro-F1",
            ylabel="Macro-F1",
        )
    if aggregated_histories:
        save_training_curves_with_std(
            histories_by_model=aggregated_histories,
            save_path=args.output_dir / "training_history_mean_std.png",
        )
        save_all_models_loss_overlay_with_std(
            histories_by_model=aggregated_histories,
            save_path=args.output_dir / "all_models_train_loss_vs_epoch_mean_std.png",
            loss_key="train_loss",
            title="Deep Models: Training Loss vs Epoch Across Seeds",
        )
        save_all_models_loss_overlay_with_std(
            histories_by_model=aggregated_histories,
            save_path=args.output_dir / "all_models_val_loss_vs_epoch_mean_std.png",
            loss_key="val_loss",
            title="Deep Models: Validation Loss vs Epoch Across Seeds",
        )
        save_convergence_diagnostics_with_std(
            histories_by_model=aggregated_histories,
            save_path=args.output_dir / "training_convergence_mean_std.png",
        )

    rt_df = pd.DataFrame(aggregated_runtime_rows)
    rt_df.to_csv(args.output_dir / "runtime_repeated.csv", index=False)
    if aggregated_final_split_rows:
        final_df = pd.DataFrame(aggregated_final_split_rows)
        final_df.to_csv(args.output_dir / "final_split_metrics_repeated.csv", index=False)
        summary_final = final_df.groupby(["model", "display_name"], as_index=False).agg(
            train_acc_mean=("train_acc_at_selected", "mean"),
            train_acc_std=("train_acc_at_selected", "std"),
            train_f1_mean=("train_f1_at_selected", "mean"),
            train_f1_std=("train_f1_at_selected", "std"),
            val_acc_mean=("val_acc_at_selected", "mean"),
            val_acc_std=("val_acc_at_selected", "std"),
            val_f1_mean=("val_f1_at_selected", "mean"),
            val_f1_std=("val_f1_at_selected", "std"),
            test_acc_mean=("test_accuracy", "mean"),
            test_acc_std=("test_accuracy", "std"),
            test_macro_f1_mean=("test_macro_f1", "mean"),
            test_macro_f1_std=("test_macro_f1", "std"),
            params_m=("params_m", "first"),
            train_time_sec_mean=("train_time_sec", "mean"),
            train_time_sec_std=("train_time_sec", "std"),
            inference_ms_mean=("inference_ms", "mean"),
            inference_ms_std=("inference_ms", "std"),
        )
        summary_final.to_csv(args.output_dir / "final_split_metrics_summary.csv", index=False)
    if aggregated_histories:
        agg_training_rows: List[Dict[str, float | int | str]] = []
        for model_name, runs in aggregated_histories.items():
            for run_idx, hist in enumerate(runs):
                seed_val = int(seed_list[run_idx]) if run_idx < len(seed_list) else int(seed_list[0])
                n_ep = int(len(hist.get("train_loss", [])))
                for ep_idx in range(n_ep):
                    agg_training_rows.append(
                        {
                            "model": str(model_name),
                            "seed": int(seed_val),
                            "epoch": int(ep_idx + 1),
                            "train_loss": float(hist["train_loss"][ep_idx]) if ep_idx < len(hist.get("train_loss", [])) else np.nan,
                            "val_loss": float(hist["val_loss"][ep_idx]) if ep_idx < len(hist.get("val_loss", [])) else np.nan,
                        }
                    )
        _save_loss_mean_std_across_seeds(
            pd.DataFrame(agg_training_rows),
            args.output_dir / "loss_mean_std_across_seeds.png",
        )
    _write_status(
        status_path,
        {
            "status": "done",
            "gpu_id": gpu_slot,
            "pid": int(os.getpid()),
            "queue_models": queue_models,
            "queue_total": int(queue_total),
            "queue_completed": int(queue_total),
            "current_model": None,
            "current_model_index": int(queue_total),
            "current_epoch": int(args.epochs),
            "total_epochs": int(args.epochs),
            "latest_val_f1": None,
            "latest_val_acc": None,
            "last_update_ts": time.time(),
        },
    )


if __name__ == "__main__":
    main()
