#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def _load_mean_df(root: Path) -> pd.DataFrame:
    df = pd.read_csv(root / "final_split_metrics_merged.csv").copy()
    df["training_time_sec"] = df["train_time_sec"]
    grouped = (
        df.groupby(["model", "display_name"], as_index=False)
        .agg(
            test_accuracy=("test_accuracy", "mean"),
            test_macro_f1=("test_macro_f1", "mean"),
            training_time_sec=("training_time_sec", "mean"),
            inference_ms=("inference_ms", "mean"),
            params_m=("params_m", "mean"),
        )
        .reset_index(drop=True)
    )
    grouped["test_accuracy_pct"] = grouped["test_accuracy"] * 100.0
    grouped["test_macro_f1_pct"] = grouped["test_macro_f1"] * 100.0
    return grouped


def _category_color(model: str) -> str:
    if model == "Transformer":
        return "#5CB85C"  # green
    if model in {"LSTM", "GRU", "LeNet_LSTM", "TCN"}:
        return "#E15759"  # red
    return "#5DA5DA"  # blue


def _label_points(ax, df: pd.DataFrame, x_col: str, y_col: str) -> None:
    default_dx, default_dy = 8, 6
    custom = {
        "Transformer": (2, 4),
        "LeNet": (2, 6),
        "GRU": (2, 2),
        "LSTM": (2, 6),
        "TCN": (2, 4),
        "ResNet18": (2, 6),
        "CNN_LSTM": (2, 2),
        "EfficientNet_B0": (2, 6),
        "MobileNet_V2": (2, 6),
        "LeNet_LSTM": (2, 6),
    }
    for _, row in df.iterrows():
        dx, dy = custom.get(str(row["model"]), (default_dx, default_dy))
        ax.annotate(
            str(row["model"]),
            xy=(row[x_col], row[y_col]),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=10,
            color="black",
        )


def _plot_mean_scatter(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    x_label: str,
    y_label: str,
    title: str,
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 7), dpi=220)
    colors = [_category_color(str(m)) for m in df["model"]]
    ax.scatter(df[x_col], df[y_col], s=120, c=colors, edgecolors="black", linewidths=0.5)
    _label_points(ax, df, x_col, y_col)
    ax.set_title(title, fontsize=16, pad=6)
    ax.set_xlabel(x_label, fontsize=12)
    ax.set_ylabel(y_label, fontsize=12)
    ax.set_ylim(60, 100)
    ax.grid(True, alpha=0.25)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate benchmark-style mean-point figures for the 10-seed main experiment.")
    parser.add_argument("--root", required=True, help="Main experiment root directory containing final_split_metrics_merged.csv")
    args = parser.parse_args()

    root = Path(args.root)
    df = _load_mean_df(root)

    _plot_mean_scatter(
        df,
        x_col="training_time_sec",
        y_col="test_accuracy_pct",
        x_label="Training Time (s)",
        y_label="Accuracy (%)",
        title="Benchmark: Accuracy vs Training Time",
        out_path=root / "benchmark_accuracy_vs_training_time_mean.png",
    )
    _plot_mean_scatter(
        df,
        x_col="inference_ms",
        y_col="test_accuracy_pct",
        x_label="Inference Time (ms)",
        y_label="Accuracy (%)",
        title="Benchmark: Accuracy vs Inference Time",
        out_path=root / "benchmark_accuracy_vs_inference_time_mean.png",
    )
    _plot_mean_scatter(
        df,
        x_col="training_time_sec",
        y_col="test_macro_f1_pct",
        x_label="Training Time (s)",
        y_label="Macro-F1 (%)",
        title="Benchmark: Macro-F1 vs Training Time",
        out_path=root / "benchmark_macrof1_vs_training_time_mean.png",
    )
    _plot_mean_scatter(
        df,
        x_col="inference_ms",
        y_col="test_macro_f1_pct",
        x_label="Inference Time (ms)",
        y_label="Macro-F1 (%)",
        title="Benchmark: Macro-F1 vs Inference Time",
        out_path=root / "benchmark_macrof1_vs_inference_time_mean.png",
    )

    print(f"Benchmark mean-point figures saved under: {root}")


if __name__ == "__main__":
    main()
