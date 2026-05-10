from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


IMPROVED_MODEL_KEY = "LeNet_LSTM_MotionInput"
IMPROVED_DISPLAY_NAME = "LeNet_LSTM_MotionInput + LS0.05 + WD3e-4"
REG_MODEL_KEY = "LeNet_LSTM_RegOnly"
REG_DISPLAY_NAME = "LeNet_LSTM + LS0.05 + WD3e-4"


def _build_combined_df(main_root: Path, improved_root: Path | None = None, reg_root: Path | None = None) -> pd.DataFrame:
    main_df = pd.read_csv(main_root / "final_split_metrics_merged.csv").copy()
    frames = [main_df]
    if improved_root is not None:
        improved_paths = sorted(improved_root.glob("runs/motioninput_reg_seed*/subject_seed*/final_split_metrics.csv"))
        if not improved_paths:
            raise FileNotFoundError(f"No improved run files found under {improved_root}")
        improved_df = pd.concat([pd.read_csv(path) for path in improved_paths], ignore_index=True)
        improved_df = improved_df.copy()
        improved_df["model"] = IMPROVED_MODEL_KEY
        improved_df["display_name"] = IMPROVED_DISPLAY_NAME
        frames.append(improved_df)
    if reg_root is not None:
        reg_paths = sorted(reg_root.glob("runs/reg_seed*/subject_seed*/final_split_metrics.csv"))
        if not reg_paths:
            raise FileNotFoundError(f"No regularization-only run files found under {reg_root}")
        reg_df = pd.concat([pd.read_csv(path) for path in reg_paths], ignore_index=True).copy()
        reg_df["model"] = REG_MODEL_KEY
        reg_df["display_name"] = REG_DISPLAY_NAME
        frames.append(reg_df)

    combined = pd.concat(frames, ignore_index=True)
    combined["training_time_min"] = combined["train_time_sec"] / 60.0
    combined["test_accuracy_pct"] = combined["test_accuracy"] * 100.0
    combined["test_macro_f1_pct"] = combined["test_macro_f1"] * 100.0
    return combined


def _build_color_map(model_keys: list[str]) -> dict[str, str]:
    base_palette = [
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
    colors: dict[str, str] = {}
    non_improved = [m for m in model_keys if m != IMPROVED_MODEL_KEY]
    for idx, model_key in enumerate(non_improved):
        colors[model_key] = base_palette[idx % len(base_palette)]
    colors[IMPROVED_MODEL_KEY] = "#111111"
    colors[REG_MODEL_KEY] = "#8C2D04"
    return colors


def _plot_scatter(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    x_label: str,
    y_label: str,
    title: str,
    out_path: Path,
    y_lim: tuple[float, float] | None = None,
) -> None:
    model_keys = list(dict.fromkeys(df["model"].tolist()))
    colors = _build_color_map(model_keys)

    fig, ax = plt.subplots(figsize=(13.5, 8.5))
    for model_key in model_keys:
        sub = df[df["model"] == model_key].sort_values("seed")
        color = colors[model_key]
        is_variant = model_key in {IMPROVED_MODEL_KEY, REG_MODEL_KEY}
        label = sub["display_name"].iloc[0]
        ax.scatter(
            sub[x_col],
            sub[y_col],
            s=180 if is_variant else 130,
            c=color,
            alpha=0.92 if is_variant else 0.78,
            edgecolors="black" if is_variant else "#444444",
            linewidths=1.4 if is_variant else 0.8,
            label=label,
            zorder=4 if is_variant else 2,
        )

    ax.set_title(title, fontsize=22, pad=10)
    ax.set_xlabel(x_label, fontsize=16)
    ax.set_ylabel(y_label, fontsize=16)
    ax.tick_params(axis="both", labelsize=13)
    ax.grid(True, alpha=0.28, linewidth=1.0)
    ax.set_axisbelow(True)
    if y_lim is not None:
        ax.set_ylim(*y_lim)

    handles, labels = ax.get_legend_handles_labels()
    legend = ax.legend(handles, labels, loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False, fontsize=12)
    for text in legend.get_texts():
        if "MotionInput" in text.get_text() or "LS0.05" in text.get_text():
            text.set_fontweight("bold")

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _save_plot_data(df: pd.DataFrame, cols: list[str], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df[cols].to_csv(out_path, index=False)


def _build_summary(df: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        df.groupby(["model", "display_name"], as_index=False)
        .agg(
            mean_accuracy=("test_accuracy", "mean"),
            std_accuracy=("test_accuracy", "std"),
            mean_macro_f1=("test_macro_f1", "mean"),
            std_macro_f1=("test_macro_f1", "std"),
            mean_training_min=("training_time_min", "mean"),
            mean_inference_ms=("inference_ms", "mean"),
            mean_params_m=("params_m", "mean"),
        )
        .sort_values(["mean_macro_f1", "mean_accuracy"], ascending=False)
        .reset_index(drop=True)
    )
    grouped["rank_macro_f1"] = grouped["mean_macro_f1"].rank(ascending=False, method="min").astype(int)
    grouped["rank_accuracy"] = grouped["mean_accuracy"].rank(ascending=False, method="min").astype(int)
    grouped["mean_accuracy_pct"] = grouped["mean_accuracy"] * 100.0
    grouped["mean_macro_f1_pct"] = grouped["mean_macro_f1"] * 100.0
    return grouped


def _save_summary_md(summary_df: pd.DataFrame, out_path: Path) -> None:
    improved_rows = summary_df[summary_df["model"] == IMPROVED_MODEL_KEY]
    reg_rows = summary_df[summary_df["model"] == REG_MODEL_KEY]
    lines = [
        "# LeNet_LSTM Variant Position Summary",
        "",
        "- Compared against the original 10 deep-learning models using the existing 10-seed main experiment results.",
        "",
    ]
    if not improved_rows.empty:
        improved_row = improved_rows.iloc[0]
        lines.extend(
            [
                f"## {IMPROVED_DISPLAY_NAME}",
                f"- Macro-F1 rank: **{int(improved_row['rank_macro_f1'])} / {len(summary_df)}**",
                f"- Accuracy rank: **{int(improved_row['rank_accuracy'])} / {len(summary_df)}**",
                f"- Mean test Macro-F1: **{improved_row['mean_macro_f1_pct']:.3f}%**",
                f"- Mean test Accuracy: **{improved_row['mean_accuracy_pct']:.3f}%**",
                f"- Mean training time: **{improved_row['mean_training_min']:.2f} min**",
                f"- Mean inference time: **{improved_row['mean_inference_ms']:.3f} ms**",
                f"- Mean params: **{improved_row['mean_params_m']:.6f} M**",
                "",
            ]
        )
    if not reg_rows.empty:
        reg_row = reg_rows.iloc[0]
        lines.extend(
            [
                f"## {REG_DISPLAY_NAME}",
                f"- Macro-F1 rank: **{int(reg_row['rank_macro_f1'])} / {len(summary_df)}**",
                f"- Accuracy rank: **{int(reg_row['rank_accuracy'])} / {len(summary_df)}**",
                f"- Mean test Macro-F1: **{reg_row['mean_macro_f1_pct']:.3f}%**",
                f"- Mean test Accuracy: **{reg_row['mean_accuracy_pct']:.3f}%**",
                f"- Mean training time: **{reg_row['mean_training_min']:.2f} min**",
                f"- Mean inference time: **{reg_row['mean_inference_ms']:.3f} ms**",
                f"- Mean params: **{reg_row['mean_params_m']:.6f} M**",
                "",
            ]
        )
    lines.append("These files are new overlay comparison outputs only; the original 10-model figures were left unchanged.")
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Overlay improved LeNet_LSTM onto the existing 10-model 10-seed plots.")
    parser.add_argument("--main_root", required=True, help="Existing 10-model 10-seed main experiment root.")
    parser.add_argument("--improved_root", default="", help="Optional improved MotionInput+regularization 10-seed root.")
    parser.add_argument("--reg_root", default="", help="Optional regularization-only 10-seed root.")
    parser.add_argument("--output_dir", required=True, help="Directory to save overlay figures and CSVs.")
    args = parser.parse_args()

    main_root = Path(args.main_root)
    improved_root = Path(args.improved_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    improved_root = Path(args.improved_root) if str(args.improved_root).strip() else None
    reg_root = Path(args.reg_root) if str(args.reg_root).strip() else None
    df = _build_combined_df(main_root, improved_root=improved_root, reg_root=reg_root)

    acc_train_csv = output_dir / "accuracy_training_time_11models_10seeds_plot_data.csv"
    acc_inf_csv = output_dir / "accuracy_inference_time_11models_10seeds_plot_data.csv"
    f1_train_csv = output_dir / "macrof1_training_time_11models_10seeds_plot_data.csv"

    _save_plot_data(df, ["model", "display_name", "seed", "training_time_min", "test_accuracy_pct"], acc_train_csv)
    _save_plot_data(df, ["model", "display_name", "seed", "inference_ms", "test_accuracy_pct"], acc_inf_csv)
    _save_plot_data(df, ["model", "display_name", "seed", "training_time_min", "test_macro_f1_pct"], f1_train_csv)

    _plot_scatter(
        df=df,
        x_col="training_time_min",
        y_col="test_accuracy_pct",
        x_label="Training Time (minutes)",
        y_label="Test Accuracy (%)",
        title="Accuracy vs Training Time\n10 Original Models + LeNet_LSTM Variants",
        out_path=output_dir / "accuracy_training_time_11models_10seeds_with_improved.png",
        y_lim=(60, 100),
    )
    _plot_scatter(
        df=df,
        x_col="inference_ms",
        y_col="test_accuracy_pct",
        x_label="Inference Time (ms)",
        y_label="Test Accuracy (%)",
        title="Accuracy vs Inference Time\n10 Original Models + LeNet_LSTM Variants",
        out_path=output_dir / "accuracy_inference_time_11models_10seeds_with_improved.png",
        y_lim=(60, 100),
    )
    _plot_scatter(
        df=df,
        x_col="training_time_min",
        y_col="test_macro_f1_pct",
        x_label="Training Time (minutes)",
        y_label="Test Macro-F1 (%)",
        title="Macro-F1 vs Training Time\n10 Original Models + LeNet_LSTM Variants",
        out_path=output_dir / "macrof1_training_time_11models_10seeds_with_improved.png",
        y_lim=(60, 100),
    )

    summary_df = _build_summary(df)
    summary_df.to_csv(output_dir / "model_mean_summary_with_improved.csv", index=False)
    _save_summary_md(summary_df, output_dir / "improved_model_position_summary.md")

    print(f"Overlay comparison saved under: {output_dir}")


if __name__ == "__main__":
    main()
