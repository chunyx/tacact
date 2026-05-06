#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd


DEFAULT_ROOT_GLOBS = [
    "outputs_main*",
]

PRIORITY_MODELS = [
    "LeNet",
    "ResNet18",
    "MobileNet_V2",
    "EfficientNet_B0",
    "LSTM",
    "GRU",
    "CNN_LSTM",
    "TCN",
    "Transformer",
]

DEFAULT_VAL_YMIN = 0.6
DEFAULT_VAL_YMAX = 1.0

DISPLAY_NAME = {
    "LeNet": "LeNet-style CNN",
    "LSTM": "LSTM baseline",
    "GRU": "GRU baseline",
    "CNN_LSTM": "CNN-LSTM baseline",
    "LeNet_LSTM": "LeNet-LSTM baseline",
    "TCN": "TCN baseline",
    "Transformer": "Transformer baseline",
}


def display_name(model: str) -> str:
    return DISPLAY_NAME.get(model, model)


def discover_log_files(workspace: Path, root_globs: List[str]) -> List[Path]:
    roots: List[Path] = []
    for pattern in root_globs:
        roots.extend([p for p in workspace.glob(pattern) if p.is_dir()])

    log_files: List[Path] = []
    for root in sorted(set(roots)):
        logs_dir = root / "logs"
        if logs_dir.is_dir():
            log_files.extend(sorted(logs_dir.glob("gpu*.log")))
    return log_files


def infer_model_name(text: str, log_path: Path) -> Optional[str]:
    m = re.search(r"\[MODEL=([^\]]+)\]", text)
    if m:
        return m.group(1).strip()

    m = re.search(r"deep=\['([^']+)'\]", text)
    if m:
        return m.group(1).strip()

    m = re.search(r"--deep_models\s+([A-Za-z0-9_]+)", text)
    if m:
        return m.group(1).strip()

    # Fallback: infer from filename if obviously a single model name is present
    stem = log_path.stem
    for model in PRIORITY_MODELS:
        if model.lower() in stem.lower():
            return model
    return None


def parse_epoch_metrics(log_path: Path) -> List[Dict]:
    text = log_path.read_text(errors="ignore")
    model = infer_model_name(text, log_path)
    if not model:
        return []
    rows: List[Dict] = []

    # Typical line contains epoch summary with TrainLoss/ValLoss/ValAcc/ValF1.
    full_pattern = re.compile(
        r"Epoch\s+(\d+)/(\d+)\s*\|\s*TrainLoss:\s*([0-9]*\.?[0-9]+)\s*\|\s*ValLoss:\s*([0-9]*\.?[0-9]+).*?"
        r"ValAcc:\s*([0-9]+(?:\.[0-9]+)?)%\s*\|\s*ValF1:\s*([0-9]+(?:\.[0-9]+)?)%",
        re.IGNORECASE,
    )

    full_matches = list(full_pattern.finditer(text))
    if full_matches:
        seen_epochs = set()
        for mm in full_matches:
            ep = int(mm.group(1))
            if ep in seen_epochs:
                continue
            seen_epochs.add(ep)
            rows.append(
                {
                    "model": model,
                    "epoch": ep,
                    "val_accuracy": float(mm.group(5)) / 100.0,
                    "val_macro_f1": float(mm.group(6)) / 100.0,
                    "train_loss": float(mm.group(3)),
                    "val_loss": float(mm.group(4)),
                    "source_log": str(log_path),
                }
            )
        return rows

    # Fallback: line contains both ValAcc and ValF1 in percentage.
    pair_pattern = re.compile(
        r"ValAcc:\s*([0-9]+(?:\.[0-9]+)?)%\s*\|\s*ValF1:\s*([0-9]+(?:\.[0-9]+)?)%",
        re.IGNORECASE,
    )
    acc_pattern = re.compile(r"ValAcc:\s*([0-9]+(?:\.[0-9]+)?)%", re.IGNORECASE)
    f1_pattern = re.compile(r"ValF1:\s*([0-9]+(?:\.[0-9]+)?)%", re.IGNORECASE)

    pair_matches = list(pair_pattern.finditer(text))
    if pair_matches:
        for i, mm in enumerate(pair_matches, start=1):
            rows.append(
                {
                    "model": model,
                    "epoch": i,
                    "val_accuracy": float(mm.group(1)) / 100.0,
                    "val_macro_f1": float(mm.group(2)) / 100.0,
                    "train_loss": None,
                    "val_loss": None,
                    "source_log": str(log_path),
                }
            )
        return rows

    acc_matches = [float(m.group(1)) / 100.0 for m in acc_pattern.finditer(text)]
    f1_matches = [float(m.group(1)) / 100.0 for m in f1_pattern.finditer(text)]
    length = max(len(acc_matches), len(f1_matches))
    for i in range(length):
        rows.append(
            {
                "model": model,
                "epoch": i + 1,
                "val_accuracy": acc_matches[i] if i < len(acc_matches) else None,
                "val_macro_f1": f1_matches[i] if i < len(f1_matches) else None,
                "train_loss": None,
                "val_loss": None,
                "source_log": str(log_path),
            }
        )
    return rows


def pick_best_run_per_model(df: pd.DataFrame, log_files: List[Path]) -> pd.DataFrame:
    if df.empty:
        return df

    mtime_map = {str(p): p.stat().st_mtime for p in log_files}
    log_summary = (
        df.groupby(["model", "source_log"], as_index=False)["epoch"]
        .max()
        .rename(columns={"epoch": "num_epochs"})
    )
    log_summary["mtime"] = log_summary["source_log"].map(mtime_map).fillna(0.0)
    # Keep run with most epochs; tie-break by newer log.
    log_summary = log_summary.sort_values(
        by=["model", "num_epochs", "mtime"], ascending=[True, False, False]
    )
    best_logs = log_summary.drop_duplicates(subset=["model"], keep="first")["source_log"]
    return df[df["source_log"].isin(set(best_logs))].copy()


def choose_metric_and_finalize(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    final_rows = []
    for _, row in df.iterrows():
        if pd.notna(row["val_accuracy"]):
            metric_used = "val_accuracy"
            metric_value = row["val_accuracy"]
        elif pd.notna(row["val_macro_f1"]):
            metric_used = "val_macro_f1"
            metric_value = row["val_macro_f1"]
        else:
            metric_used = "missing"
            metric_value = None
        out = row.to_dict()
        out["metric_used"] = metric_used
        out["metric_value"] = metric_value
        final_rows.append(out)

    result = pd.DataFrame(final_rows)
    result = result.sort_values(["model", "epoch"]).reset_index(drop=True)
    return result


def make_plot(
    df: pd.DataFrame,
    out_path: Path,
    title: str,
    y_min: float = DEFAULT_VAL_YMIN,
    y_max: float = DEFAULT_VAL_YMAX,
) -> None:
    if df.empty:
        return

    plt.style.use("default")
    fig, ax = plt.subplots(figsize=(12, 7), dpi=200)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    models = sorted(df["model"].unique())
    cmap = plt.get_cmap("tab20")
    for i, model in enumerate(models):
        part = df[df["model"] == model].sort_values("epoch")
        ax.plot(
            part["epoch"],
            part["metric_value"],
            label=display_name(model),
            linewidth=2.0,
            color=cmap(i % 20),
            alpha=0.95,
        )

    ylabel = "Validation Accuracy / Macro-F1"
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, pad=12)
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.35)
    ax.set_ylim(float(y_min), float(y_max))

    legend = ax.legend(
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=True,
        framealpha=0.95,
        fontsize=9,
        title="Model",
    )
    legend.get_frame().set_edgecolor("#CCCCCC")

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def clip_epoch(df: pd.DataFrame, max_epoch: Optional[int]) -> pd.DataFrame:
    if max_epoch is None:
        return df
    return df[df["epoch"] <= int(max_epoch)].copy()


def make_loss_plot(
    df: pd.DataFrame,
    out_path: Path,
    title: str,
    loss_col: str,
    ylabel: str,
) -> None:
    df = df[df[loss_col].notna()].copy()
    if df.empty:
        return

    plt.style.use("default")
    fig, ax = plt.subplots(figsize=(12, 7), dpi=200)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    models = sorted(df["model"].unique())
    cmap = plt.get_cmap("tab20")
    for i, model in enumerate(models):
        part = df[df["model"] == model].sort_values("epoch")
        ax.plot(
            part["epoch"],
            part[loss_col],
            label=model,
            linewidth=2.0,
            color=cmap(i % 20),
            alpha=0.95,
        )

    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, pad=12)
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.35)

    legend = ax.legend(
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=True,
        framealpha=0.95,
        fontsize=9,
        title="Model",
    )
    legend.get_frame().set_edgecolor("#CCCCCC")

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def make_dual_loss_plot(df: pd.DataFrame, out_path: Path, title: str) -> None:
    df_train = df[df["train_loss"].notna()].copy()
    df_val = df[df["val_loss"].notna()].copy()
    if df_train.empty and df_val.empty:
        return

    plt.style.use("default")
    fig, ax = plt.subplots(figsize=(13, 7), dpi=200)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    models = sorted(df["model"].unique())
    cmap = plt.get_cmap("tab20")
    for i, model in enumerate(models):
        color = cmap(i % 20)
        t_part = df_train[df_train["model"] == model].sort_values("epoch")
        v_part = df_val[df_val["model"] == model].sort_values("epoch")
        if not t_part.empty:
            ax.plot(
                t_part["epoch"],
                t_part["train_loss"],
                linewidth=1.8,
                color=color,
                alpha=0.9,
                linestyle="-",
                label=f"{model} (train)",
            )
        if not v_part.empty:
            ax.plot(
                v_part["epoch"],
                v_part["val_loss"],
                linewidth=1.8,
                color=color,
                alpha=0.9,
                linestyle="--",
                label=f"{model} (val)",
            )

    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Loss", fontsize=12)
    ax.set_title(title, fontsize=14, pad=12)
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.35)

    legend = ax.legend(
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=True,
        framealpha=0.95,
        fontsize=8,
        title="Curve",
    )
    legend.get_frame().set_edgecolor("#CCCCCC")

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def make_final_loss_bar(df: pd.DataFrame, out_path: Path, title: str, loss_col: str) -> None:
    part = df[df[loss_col].notna()].copy()
    if part.empty:
        return
    final = part.sort_values("epoch").groupby("model", as_index=False).tail(1)
    final = final.sort_values(loss_col, ascending=True)

    plt.style.use("default")
    fig, ax = plt.subplots(figsize=(11, 6), dpi=200)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.bar(final["model"], final[loss_col], color="#4C78A8", alpha=0.9)
    ax.set_xlabel("Model", fontsize=12)
    ax.set_ylabel(loss_col.replace("_", " ").title(), fontsize=12)
    ax.set_title(title, fontsize=14, pad=12)
    ax.grid(True, axis="y", linestyle="--", linewidth=0.6, alpha=0.35)
    ax.tick_params(axis="x", rotation=30)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def get_representative_models(
    df: pd.DataFrame,
    representative_count: int,
) -> Tuple[List[str], List[str]]:
    final_metric = df.sort_values("epoch").groupby("model", as_index=False).tail(1)
    final_metric = final_metric.sort_values("metric_value", ascending=False)
    available_models = sorted(df["model"].unique())
    available_set = set(available_models)
    preferred = [m for m in PRIORITY_MODELS if m in available_set]
    rep_models = preferred[: representative_count]
    if len(rep_models) < representative_count:
        for m in final_metric["model"]:
            if m not in rep_models:
                rep_models.append(m)
            if len(rep_models) >= representative_count:
                break
    return rep_models, available_models


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot per-epoch validation curves for existing model logs."
    )
    parser.add_argument(
        "--workspace",
        type=Path,
        default=Path("."),
        help="Workspace root path.",
    )
    parser.add_argument(
        "--root-glob",
        action="append",
        default=[],
        help="Root directory glob to scan (can repeat). Default: outputs_main*",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs_plots"),
        help="Directory to save csv and figures.",
    )
    parser.add_argument(
        "--representative-count",
        type=int,
        default=6,
        help="Number of models in representative figure.",
    )
    parser.add_argument(
        "--max-epoch",
        type=int,
        default=30,
        help="Max epoch shown in validation metric plots (default: 30).",
    )
    parser.add_argument(
        "--val-ymin",
        type=float,
        default=DEFAULT_VAL_YMIN,
        help="Y-axis min for validation metric plots.",
    )
    parser.add_argument(
        "--val-ymax",
        type=float,
        default=DEFAULT_VAL_YMAX,
        help="Y-axis max for validation metric plots.",
    )
    args = parser.parse_args()

    workspace = args.workspace.resolve()
    root_globs = args.root_glob if args.root_glob else DEFAULT_ROOT_GLOBS
    out_dir = args.output_dir
    if not out_dir.is_absolute():
        out_dir = workspace / out_dir

    log_files = discover_log_files(workspace, root_globs)
    all_rows: List[Dict] = []
    for lf in log_files:
        all_rows.extend(parse_epoch_metrics(lf))

    df = pd.DataFrame(all_rows)
    if df.empty:
        print("No per-epoch ValAcc/ValF1 records found in scanned logs.")
        return

    df = pick_best_run_per_model(df, log_files)
    df = choose_metric_and_finalize(df)
    df = df[df["metric_used"] != "missing"].copy()

    csv_path = out_dir / "val_metric_by_epoch_all_models.csv"
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)

    # Validation metric plots (optionally clipped to early epochs).
    df_metric_plot = clip_epoch(df, args.max_epoch)

    # All models plot
    make_plot(
        df_metric_plot,
        out_dir / "val_accuracy_vs_epoch_all_models.png",
        "Validation Metric vs Epoch (All Models)",
        y_min=args.val_ymin,
        y_max=args.val_ymax,
    )

    rep_models, available_models = get_representative_models(df, args.representative_count)

    df_rep = df_metric_plot[df_metric_plot["model"].isin(rep_models)].copy()
    make_plot(
        df_rep,
        out_dir / "val_accuracy_vs_epoch_representative_models.png",
        "Validation Metric vs Epoch (Representative Models)",
        y_min=args.val_ymin,
        y_max=args.val_ymax,
    )

    # Loss figures
    make_dual_loss_plot(
        df,
        out_dir / "loss_vs_epoch_all_models.png",
        "Train/Validation Loss vs Epoch (All Models)",
    )
    make_loss_plot(
        df,
        out_dir / "train_loss_vs_epoch_all_models.png",
        "Training Loss vs Epoch (All Models)",
        "train_loss",
        "Training Loss",
    )
    make_loss_plot(
        df,
        out_dir / "val_loss_vs_epoch_all_models.png",
        "Validation Loss vs Epoch (All Models)",
        "val_loss",
        "Validation Loss",
    )
    df_loss_rep = df[df["model"].isin(rep_models)].copy()
    make_dual_loss_plot(
        df_loss_rep,
        out_dir / "loss_vs_epoch_representative_models.png",
        "Train/Validation Loss vs Epoch (Representative Models)",
    )
    make_final_loss_bar(
        df,
        out_dir / "final_val_loss_by_model.png",
        "Final Validation Loss by Model",
        "val_loss",
    )

    loss_csv_path = out_dir / "loss_by_epoch_all_models.csv"
    df[["model", "epoch", "train_loss", "val_loss", "source_log"]].to_csv(
        loss_csv_path, index=False
    )

    missing_priority = [m for m in PRIORITY_MODELS if m not in set(available_models)]
    print(f"Scanned logs: {len(log_files)}")
    print(f"Detected models: {available_models}")
    print(f"Missing priority models: {missing_priority}")
    print(f"Saved CSV: {csv_path}")
    print(f"Saved loss CSV: {loss_csv_path}")
    print(f"Saved figures in: {out_dir}")


if __name__ == "__main__":
    main()
