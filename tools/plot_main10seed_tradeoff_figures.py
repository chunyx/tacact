#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


BASE_PALETTE = [
    "#5DA5DA",  # blue
    "#FAA43A",  # orange
    "#60BD68",  # green
    "#F15854",  # red
    "#B291CF",  # purple
    "#B276B2",  # plum
    "#DECF3F",  # yellow
    "#4D4D4D",  # dark gray
    "#F17CB0",  # pink
    "#2CA8C2",  # teal
]


def _load_df(root: Path) -> pd.DataFrame:
    df = pd.read_csv(root / "final_split_metrics_merged.csv").copy()
    df["training_time_min"] = df["train_time_sec"] / 60.0
    df["test_accuracy_pct"] = df["test_accuracy"] * 100.0
    df["test_macro_f1_pct"] = df["test_macro_f1"] * 100.0
    return df


def _color_map(df: pd.DataFrame) -> dict[str, str]:
    model_keys = list(dict.fromkeys(df["model"].tolist()))
    return {model: BASE_PALETTE[idx % len(BASE_PALETTE)] for idx, model in enumerate(model_keys)}


def _plot_points_only(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    x_label: str,
    y_label: str,
    title: str,
    out_path: Path,
) -> None:
    colors = _color_map(df)
    fig, ax = plt.subplots(figsize=(13.5, 8.5), dpi=220)
    for model, sub in df.groupby("model", sort=False):
        ax.scatter(
            sub[x_col],
            sub[y_col],
            s=130,
            c=colors[model],
            alpha=0.78,
            edgecolors="#444444",
            linewidths=0.8,
            label=sub["display_name"].iloc[0],
        )
    ax.set_title(title, fontsize=22, pad=10)
    ax.set_xlabel(x_label, fontsize=16)
    ax.set_ylabel(y_label, fontsize=16)
    ax.set_ylim(60, 100)
    ax.tick_params(axis="both", labelsize=13)
    ax.grid(True, alpha=0.28, linewidth=1.0)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False, fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _plot_bubble(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    x_label: str,
    y_label: str,
    title: str,
    out_path: Path,
) -> None:
    colors = _color_map(df)
    fig, ax = plt.subplots(figsize=(13.5, 8.5), dpi=220)
    for model, sub in df.groupby("model", sort=False):
        ax.scatter(
            sub[x_col],
            sub[y_col],
            s=sub["params_m"] * 180.0 + 60.0,
            c=colors[model],
            alpha=0.55,
            edgecolors="#333333",
            linewidths=0.7,
            label=sub["display_name"].iloc[0],
        )
    ax.set_title(title, fontsize=22, pad=10)
    ax.set_xlabel(x_label, fontsize=16)
    ax.set_ylabel(y_label, fontsize=16)
    ax.set_ylim(60, 100)
    ax.tick_params(axis="both", labelsize=13)
    ax.grid(True, alpha=0.28, linewidth=1.0)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False, fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _non_dominated_mask(points: pd.DataFrame, x_col: str, y_col: str) -> pd.Series:
    mask = []
    for idx, row in points.iterrows():
        dominated = False
        for jdx, other in points.iterrows():
            if idx == jdx:
                continue
            no_worse = (other[x_col] <= row[x_col]) and (other[y_col] >= row[y_col])
            strictly_better = (other[x_col] < row[x_col]) or (other[y_col] > row[y_col])
            if no_worse and strictly_better:
                dominated = True
                break
        mask.append(not dominated)
    return pd.Series(mask, index=points.index)


def _plot_pareto(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    x_label: str,
    y_label: str,
    title: str,
    out_path: Path,
) -> None:
    colors = _color_map(df)
    frontier_mask = _non_dominated_mask(df[[x_col, y_col]], x_col, y_col)
    fig, ax = plt.subplots(figsize=(13.5, 8.5), dpi=220)
    for model, sub in df.groupby("model", sort=False):
        sub_mask = frontier_mask.loc[sub.index]
        ax.scatter(
            sub[x_col],
            sub[y_col],
            s=120,
            c=colors[model],
            alpha=0.72,
            edgecolors="#444444",
            linewidths=0.7,
            label=sub["display_name"].iloc[0],
        )
        front = sub[sub_mask]
        if not front.empty:
            ax.scatter(
                front[x_col],
                front[y_col],
                s=190,
                facecolors="none",
                edgecolors="black",
                linewidths=1.4,
                zorder=5,
            )
    ax.set_title(title, fontsize=22, pad=10)
    ax.set_xlabel(x_label, fontsize=16)
    ax.set_ylabel(y_label, fontsize=16)
    ax.set_ylim(60, 100)
    ax.tick_params(axis="both", labelsize=13)
    ax.grid(True, alpha=0.28, linewidth=1.0)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False, fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate 10-seed trade-off figures for the main experiment.")
    parser.add_argument("--root", required=True, help="Main experiment root directory containing final_split_metrics_merged.csv")
    args = parser.parse_args()

    root = Path(args.root)
    df = _load_df(root)

    _plot_points_only(
        df, "training_time_min", "test_accuracy_pct",
        "Training Time (minutes)", "Test Accuracy (%)",
        "Accuracy vs Training Time\nEach point represents one seed/run.",
        root / "accuracy_training_time_10seeds_points_only_linear_minutes.png",
    )
    _plot_points_only(
        df, "inference_ms", "test_accuracy_pct",
        "Inference Time (ms)", "Test Accuracy (%)",
        "Accuracy vs Inference Time\nEach point represents one seed/run.",
        root / "accuracy_inference_time_10seeds_points_only_linear.png",
    )
    _plot_points_only(
        df, "training_time_min", "test_macro_f1_pct",
        "Training Time (minutes)", "Test Macro-F1 (%)",
        "Macro-F1 vs Training Time\nEach point represents one seed/run.",
        root / "macrof1_training_time_10seeds_points_only_linear_minutes.png",
    )

    _plot_bubble(
        df, "inference_ms", "test_accuracy_pct",
        "Inference Time (ms)", "Test Accuracy (%)",
        "Accuracy vs Inference Time (Bubble)\nBubble size represents parameter count.",
        root / "accuracy_inference_time_10seeds_bubble_nodash.png",
    )
    _plot_bubble(
        df, "inference_ms", "test_macro_f1_pct",
        "Inference Time (ms)", "Test Macro-F1 (%)",
        "Macro-F1 vs Inference Time (Bubble)\nBubble size represents parameter count.",
        root / "macrof1_inference_time_10seeds_bubble_nodash.png",
    )
    _plot_pareto(
        df, "inference_ms", "test_accuracy_pct",
        "Inference Time (ms)", "Test Accuracy (%)",
        "Pareto-style Accuracy vs Inference Time\nNon-dominated points are circled; no frontier line.",
        root / "pareto_accuracy_inference_10seeds_nodash.png",
    )
    _plot_pareto(
        df, "inference_ms", "test_macro_f1_pct",
        "Inference Time (ms)", "Test Macro-F1 (%)",
        "Pareto-style Macro-F1 vs Inference Time\nNon-dominated points are circled; no frontier line.",
        root / "pareto_macrof1_inference_10seeds_nodash.png",
    )

    print(f"Trade-off figures saved under: {root}")


if __name__ == "__main__":
    main()
