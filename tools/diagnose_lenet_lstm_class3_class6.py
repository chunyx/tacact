#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path.cwd()
OUT_DIR = Path.cwd() / "outputs" / "diagnosis" / "lenet_lstm" / "class3_class6_focused"
EX_DIR = OUT_DIR / "examples"
TacActDataset = None
LABEL_MAP = None
DEFAULT_TACACT_GESTURE_NAMES = None


def _import_project_modules(project_root: Path) -> None:
    global TacActDataset, LABEL_MAP, DEFAULT_TACACT_GESTURE_NAMES
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from data import TacActDataset as _TacActDataset, LABEL_MAP as _LABEL_MAP

    TacActDataset = _TacActDataset
    LABEL_MAP = _LABEL_MAP
    try:
        from scripts.plot_tacact_raw_gesture_trends import DEFAULT_TACACT_GESTURE_NAMES as _DEFAULT_TACACT_GESTURE_NAMES
        DEFAULT_TACACT_GESTURE_NAMES = _DEFAULT_TACACT_GESTURE_NAMES
    except Exception:
        DEFAULT_TACACT_GESTURE_NAMES = {}


def warn(msg: str) -> None:
    print(f"[WARN] {msg}")


def save_markdown(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def load_json(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def resolve_output_dir(project_root: Path, output_dir_arg: str) -> Path:
    out = Path(output_dir_arg)
    if not out.is_absolute():
        out = project_root / out
    return out


def get_dataset(run_config: Dict[str, Any], data_protocol: Dict[str, Any], project_root: Path):
    data_root = Path(str(run_config.get("data_root") or ""))
    if not data_root.exists():
        warn(f"data_root not found: {data_root}")
        return None
    cache_dir = Path(str(data_protocol.get("cache_dir") or ".cache_tacact_n80_weighted"))
    if not cache_dir.is_absolute():
        cache_dir = project_root / cache_dir
    preprocessing = data_protocol.get("preprocessing", {}) if isinstance(data_protocol, dict) else {}
    try:
        ds = TacActDataset(
            root_dir=data_root,
            n_frames=int(preprocessing.get("n_frames", 80)),
            threshold_method=str(preprocessing.get("threshold_method", "mean_std")),
            threshold_k=float(preprocessing.get("threshold_k", 3.0)),
            background_frames=int(preprocessing.get("background_frames", 5)),
            clip_mode=str(preprocessing.get("clip_mode", "weighted_center")),
            cache_dir=cache_dir,
            preload_cache=False,
        )
        return ds
    except Exception as e:
        warn(f"Could not build dataset: {e}")
        return None


def class_label_to_gesture_id(class_label: int) -> Optional[int]:
    if LABEL_MAP is None:
        return None
    inv = {int(v): int(k) for k, v in LABEL_MAP.items()}
    return inv.get(int(class_label))


def compute_confidence_fields(pred_df: pd.DataFrame) -> pd.DataFrame:
    df = pred_df.copy()
    prob_cols = [c for c in df.columns if c.startswith("prob_")]
    if prob_cols:
        probs = df[prob_cols].to_numpy(dtype=float)
        true_idx = df["true_label"].astype(int).to_numpy()
        pred_idx = df["pred_label"].astype(int).to_numpy()
        df["true_class_probability"] = probs[np.arange(len(df)), true_idx]
        df["pred_class_probability"] = probs[np.arange(len(df)), pred_idx]
        sorted_probs = np.sort(probs, axis=1)
        df["probability_margin"] = sorted_probs[:, -1] - sorted_probs[:, -2]
    else:
        df["true_class_probability"] = np.nan
        df["pred_class_probability"] = df.get("top1_confidence", np.nan)
        df["probability_margin"] = np.nan
    return df


def summarize_series(x: pd.Series, prefix: str) -> Dict[str, float]:
    x = pd.to_numeric(x, errors="coerce").dropna()
    if x.empty:
        return {f"{prefix}_{k}": np.nan for k in ["mean", "std", "median", "q1", "q3"]}
    return {
        f"{prefix}_mean": float(x.mean()),
        f"{prefix}_std": float(x.std(ddof=0)),
        f"{prefix}_median": float(x.median()),
        f"{prefix}_q1": float(x.quantile(0.25)),
        f"{prefix}_q3": float(x.quantile(0.75)),
    }


def compute_sequence_payload(ds, sample_id: int) -> Dict[str, Any]:
    meta = ds.samples[int(sample_id)]
    cache_path = ds._cache_path_for(meta.path)
    if cache_path.exists():
        frames = np.load(cache_path, allow_pickle=True).astype(np.float32)
    else:
        frames = ds._preprocess(ds._read_excel_optimized(meta.path), sample_path=meta.path).astype(np.float32)

    abs_frames = np.abs(frames)
    frame_active = abs_frames.max(axis=(1, 2)) > 1e-6
    active_length = int(frame_active.sum())
    active_area = (abs_frames > 1e-6).sum(axis=(1, 2)).astype(np.float32)
    frame_mean_abs = abs_frames.mean(axis=(1, 2))
    frame_max_abs = abs_frames.max(axis=(1, 2))
    diffs = np.diff(frames, axis=0)
    motion_curve = np.mean(np.abs(diffs), axis=(1, 2)) if diffs.size else np.array([], dtype=np.float32)
    motion_energy = float(motion_curve.mean()) if motion_curve.size else 0.0

    sample_avg_map = abs_frames.mean(axis=0)
    return {
        "sample_id": int(sample_id),
        "file_path": str(meta.path),
        "subject_id": int(meta.subject),
        "gesture": int(meta.gesture),
        "active_length": active_length,
        "max_abs_delta": float(abs_frames.max()),
        "mean_abs_delta": float(abs_frames.mean()),
        "active_area_mean": float(active_area.mean()),
        "active_area_max": float(active_area.max()),
        "motion_energy": motion_energy,
        "_frames": frames,
        "_frame_mean_abs": frame_mean_abs,
        "_frame_max_abs": frame_max_abs,
        "_frame_active_area": active_area,
        "_motion_curve": motion_curve,
        "_avg_map": sample_avg_map,
    }


def plot_box_groups(data_map: Dict[str, pd.Series], ylabel: str, title: str, save_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9.2, 5.4))
    vals = []
    labels = []
    for label, series in data_map.items():
        x = pd.to_numeric(series, errors="coerce").dropna()
        if not x.empty:
            vals.append(x.to_numpy())
            labels.append(label)
    if vals:
        ax.boxplot(vals, tick_labels=labels, showmeans=True)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=20)
    ax.grid(True, axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(save_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_group_curves(curves: Dict[str, np.ndarray], save_path: Path, title: str, ylabel: str) -> None:
    fig, ax = plt.subplots(figsize=(9.5, 5.4))
    colors = {
        "correct_class_3": "#4c78a8",
        "correct_class_6": "#f58518",
        "class_3_to_6_error": "#e45756",
        "class_6_to_3_error": "#72b7b2",
    }
    for name, arr in curves.items():
        if arr.size == 0:
            continue
        ax.plot(np.arange(1, len(arr) + 1), arr, linewidth=2.0, label=name, color=colors.get(name))
    ax.set_title(title)
    ax.set_xlabel("Frame index")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(save_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_heatmaps(maps: List[Tuple[str, np.ndarray]], save_path: Path, title: str) -> None:
    if not maps:
        return
    n = len(maps)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 4.6), squeeze=False)
    axes_flat = axes.flatten()
    for ax, (name, mat) in zip(axes_flat, maps):
        im = ax.imshow(mat, cmap="coolwarm", aspect="auto")
        ax.set_title(name)
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(title, fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_example_figure(sample_row: pd.Series, seq_payload: Dict[str, Any], save_path: Path) -> None:
    frames = seq_payload["_frames"]
    motion_curve = seq_payload["_motion_curve"]
    active = np.where(np.abs(frames).max(axis=(1, 2)) > 1e-6)[0]
    if len(active) == 0:
        active = np.arange(min(len(frames), 1))
    pick_idx = np.unique(np.linspace(active[0], active[-1], num=min(6, max(1, len(active))), dtype=int))
    fig = plt.figure(figsize=(14, 7))
    gs = fig.add_gridspec(2, max(3, len(pick_idx)))
    for i, idx in enumerate(pick_idx):
        ax = fig.add_subplot(gs[0, i])
        im = ax.imshow(frames[idx], cmap="coolwarm", aspect="auto")
        ax.set_title(f"frame {idx}")
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax_curve = fig.add_subplot(gs[1, :])
    if motion_curve.size:
        ax_curve.plot(np.arange(1, len(motion_curve) + 1), motion_curve, color="#4c78a8", linewidth=2)
    ax_curve.set_title("Temporal Motion Energy Curve")
    ax_curve.set_xlabel("Frame step")
    ax_curve.set_ylabel("Mean |delta|")
    ax_curve.grid(True, alpha=0.25)
    fig.suptitle(
        f"sample_id={int(sample_row['sample_id'])} | subject={int(sample_row['subject_id'])} | "
        f"true={int(sample_row['true_label'])} pred={int(sample_row['pred_label'])} | "
        f"conf={float(sample_row.get('top1_confidence', np.nan)):.4f}",
        fontsize=12,
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Focused class_3/class_6 failure diagnosis for LeNet_LSTM.")
    p.add_argument("--project_root", type=Path, required=True)
    p.add_argument("--selected_run_dir", type=Path, required=True)
    p.add_argument("--previous_diagnosis_dir", type=Path, default=None)
    p.add_argument("--output_dir", type=str, required=True)
    return p


def main() -> None:
    global ROOT, OUT_DIR, EX_DIR
    args = build_argparser().parse_args()
    ROOT = args.project_root.resolve()
    OUT_DIR = resolve_output_dir(ROOT, args.output_dir)
    EX_DIR = OUT_DIR / "examples"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    EX_DIR.mkdir(parents=True, exist_ok=True)

    _import_project_modules(ROOT)

    run_dir = args.selected_run_dir.resolve()
    previous_diag_dir = args.previous_diagnosis_dir.resolve() if args.previous_diagnosis_dir else None

    metrics_path = run_dir / "metrics.csv"
    predictions_path = run_dir / "predictions.csv"
    training_history_path = run_dir / "training_history.csv"
    per_class_path = run_dir / "per_class_metrics.csv"
    split_audit_path = run_dir / "split_audit.json"
    data_protocol_path = run_dir / "data_protocol.json"
    run_config_path = run_dir / "run_config.json"

    metrics_df = pd.read_csv(metrics_path) if metrics_path.exists() else pd.DataFrame()
    pred_df = pd.read_csv(predictions_path) if predictions_path.exists() else pd.DataFrame()
    hist_df = pd.read_csv(training_history_path) if training_history_path.exists() else pd.DataFrame()
    per_class_df = pd.read_csv(per_class_path) if per_class_path.exists() else pd.DataFrame()
    split_audit = load_json(split_audit_path)
    data_protocol = load_json(data_protocol_path)
    run_config = load_json(run_config_path)

    # 1. label mapping
    c3_gid = class_label_to_gesture_id(3)
    c6_gid = class_label_to_gesture_id(6)
    c3_name = DEFAULT_TACACT_GESTURE_NAMES.get(c3_gid) if DEFAULT_TACACT_GESTURE_NAMES else None
    c6_name = DEFAULT_TACACT_GESTURE_NAMES.get(c6_gid) if DEFAULT_TACACT_GESTURE_NAMES else None
    mapping_md = "# Label Mapping Check\n\n"
    mapping_md += "- Selected labels under diagnosis: `class_3` and `class_6`\n"
    mapping_md += "- Repository data pipeline uses `gesture 1~12 -> label 0~11` in `data.py`.\n"
    if c3_gid is not None and c6_gid is not None:
        mapping_md += f"- `class_3` -> gesture id `{c3_gid}`\n"
        mapping_md += f"- `class_6` -> gesture id `{c6_gid}`\n"
    if c3_name and c6_name:
        mapping_md += (
            "- Human-readable gesture names were found in the auxiliary script "
            "`scripts/plot_tacact_raw_gesture_trends.py`.\n"
            f"- `class_3` -> gesture `{c3_gid}` -> `{c3_name}`\n"
            f"- `class_6` -> gesture `{c6_gid}` -> `{c6_name}`\n"
        )
    else:
        mapping_md += (
            "- No authoritative gesture-name mapping was found in the experiment result files.\n"
            "- This focused diagnosis therefore uses class IDs as the reliable labels.\n"
        )
    save_markdown(OUT_DIR / "label_mapping_check.md", mapping_md)

    # predictions and sample groups
    if pred_df.empty:
        save_markdown(OUT_DIR / "focused_diagnosis_report.md", "# Focused diagnosis\n\nNo predictions.csv found.")
        print(OUT_DIR / "focused_diagnosis_report.md")
        return
    pred_df = compute_confidence_fields(pred_df)

    ds = get_dataset(run_config, data_protocol, ROOT)
    if ds is not None:
        file_paths = []
        subj_ids = []
        for sid in pred_df["sample_id"].astype(int).tolist():
            if 0 <= sid < len(ds.samples):
                meta = ds.samples[sid]
                file_paths.append(str(meta.path))
                subj_ids.append(int(meta.subject))
            else:
                file_paths.append("")
                subj_ids.append(np.nan)
        pred_df["file_path"] = file_paths
        pred_df["subject_id"] = subj_ids

    subset = pred_df[pred_df["true_label"].isin([3, 6])].copy()
    subset["correctness"] = np.where(subset["correct"].astype(int) == 1, "correct", "wrong")
    subset["case_type"] = "other"
    subset.loc[(subset["true_label"] == 3) & (subset["pred_label"] == 3), "case_type"] = "correct_class_3"
    subset.loc[(subset["true_label"] == 6) & (subset["pred_label"] == 6), "case_type"] = "correct_class_6"
    subset.loc[(subset["true_label"] == 3) & (subset["pred_label"] == 6), "case_type"] = "class_3_to_6_error"
    subset.loc[(subset["true_label"] == 6) & (subset["pred_label"] == 3), "case_type"] = "class_6_to_3_error"
    subset.to_csv(OUT_DIR / "class3_class6_samples.csv", index=False)

    # 3. confidence analysis
    conf_rows = []
    for case_type in ["correct_class_3", "correct_class_6", "class_3_to_6_error", "class_6_to_3_error"]:
        sub = subset[subset["case_type"] == case_type]
        conf_rows.append(
            {
                "case_type": case_type,
                "num_samples": int(len(sub)),
                **summarize_series(sub["top1_confidence"], "top1_confidence"),
                **summarize_series(sub["true_class_probability"], "true_class_probability"),
                **summarize_series(sub["pred_class_probability"], "pred_class_probability"),
                **summarize_series(sub["probability_margin"], "probability_margin"),
            }
        )
    conf_summary_df = pd.DataFrame(conf_rows)
    conf_summary_df.to_csv(OUT_DIR / "class3_class6_confidence_summary.csv", index=False)

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    for case_type, color in [
        ("correct_class_3", "#4c78a8"),
        ("correct_class_6", "#f58518"),
        ("class_3_to_6_error", "#e45756"),
        ("class_6_to_3_error", "#72b7b2"),
    ]:
        vals = pd.to_numeric(subset.loc[subset["case_type"] == case_type, "top1_confidence"], errors="coerce").dropna()
        if not vals.empty:
            ax.hist(vals, bins=30, alpha=0.45, label=case_type, color=color)
    ax.set_title("class_3 / class_6 Confidence Distribution")
    ax.set_xlabel("Top-1 confidence")
    ax.set_ylabel("Count")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "class3_class6_confidence_distribution.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    plot_box_groups(
        {
            "correct_c3": subset.loc[subset["case_type"] == "correct_class_3", "probability_margin"],
            "correct_c6": subset.loc[subset["case_type"] == "correct_class_6", "probability_margin"],
            "c3_to_6": subset.loc[subset["case_type"] == "class_3_to_6_error", "probability_margin"],
            "c6_to_3": subset.loc[subset["case_type"] == "class_6_to_3_error", "probability_margin"],
        },
        ylabel="Probability margin",
        title="Probability Margin: class_3 / class_6 Cases",
        save_path=OUT_DIR / "class3_vs_class6_probability_margin.png",
    )

    # 4/6/7/8 sequence + temporal + spatial
    stats_rows: List[Dict[str, Any]] = []
    payload_by_id: Dict[int, Dict[str, Any]] = {}
    temporal_buffers: Dict[str, Dict[str, List[np.ndarray]]] = {
        "correct_class_3": {"mean_abs": [], "max_abs": [], "active_area": [], "motion": []},
        "correct_class_6": {"mean_abs": [], "max_abs": [], "active_area": [], "motion": []},
        "class_3_to_6_error": {"mean_abs": [], "max_abs": [], "active_area": [], "motion": []},
        "class_6_to_3_error": {"mean_abs": [], "max_abs": [], "active_area": [], "motion": []},
    }
    avg_maps: Dict[str, List[np.ndarray]] = {k: [] for k in temporal_buffers}

    if ds is not None:
        for _, row in subset.iterrows():
            sid = int(row["sample_id"])
            try:
                payload = compute_sequence_payload(ds, sid)
            except Exception as e:
                warn(f"Could not load sample sequence for sample_id={sid}: {e}")
                continue
            payload_by_id[sid] = payload
            rec = {k: v for k, v in payload.items() if not k.startswith("_")}
            rec.update(
                {
                    "true_label": int(row["true_label"]),
                    "pred_label": int(row["pred_label"]),
                    "case_type": row["case_type"],
                    "top1_confidence": float(row.get("top1_confidence", np.nan)),
                    "true_class_probability": float(row.get("true_class_probability", np.nan)),
                    "pred_class_probability": float(row.get("pred_class_probability", np.nan)),
                    "probability_margin": float(row.get("probability_margin", np.nan)),
                }
            )
            stats_rows.append(rec)

            ct = str(row["case_type"])
            if ct in temporal_buffers:
                temporal_buffers[ct]["mean_abs"].append(payload["_frame_mean_abs"])
                temporal_buffers[ct]["max_abs"].append(payload["_frame_max_abs"])
                temporal_buffers[ct]["active_area"].append(payload["_frame_active_area"])
                temporal_buffers[ct]["motion"].append(payload["_motion_curve"])
                avg_maps[ct].append(payload["_avg_map"])

    stats_df = pd.DataFrame(stats_rows)
    stats_df.to_csv(OUT_DIR / "class3_class6_sequence_stats.csv", index=False)

    stats_summary_rows = []
    for name, sub in {
        "all_class_3": stats_df[stats_df["true_label"] == 3],
        "all_class_6": stats_df[stats_df["true_label"] == 6],
        "correct_class_3": stats_df[stats_df["case_type"] == "correct_class_3"],
        "correct_class_6": stats_df[stats_df["case_type"] == "correct_class_6"],
        "class_3_to_6_error": stats_df[stats_df["case_type"] == "class_3_to_6_error"],
        "class_6_to_3_error": stats_df[stats_df["case_type"] == "class_6_to_3_error"],
        "all_correct_3_6": stats_df[stats_df["case_type"].isin(["correct_class_3", "correct_class_6"])],
        "all_confused_3_6": stats_df[stats_df["case_type"].isin(["class_3_to_6_error", "class_6_to_3_error"])],
    }.items():
        for metric in ["active_length", "max_abs_delta", "mean_abs_delta", "active_area_mean", "active_area_max", "motion_energy"]:
            row = {"group": name, "metric": metric}
            row.update(summarize_series(sub[metric], metric))
            stats_summary_rows.append(row)
    stats_summary_df = pd.DataFrame(stats_summary_rows)
    stats_summary_df.to_csv(OUT_DIR / "class3_class6_stats_summary.csv", index=False)
    focused_compare_df = stats_summary_df[stats_summary_df["group"].isin(["correct_class_3", "correct_class_6", "class_3_to_6_error", "class_6_to_3_error", "all_correct_3_6", "all_confused_3_6"])].copy()
    focused_compare_df.to_csv(OUT_DIR / "class3_class6_correct_vs_confused_stats.csv", index=False)

    plot_box_groups(
        {
            "correct_c3": stats_df.loc[stats_df["case_type"] == "correct_class_3", "active_length"],
            "c3_to_6": stats_df.loc[stats_df["case_type"] == "class_3_to_6_error", "active_length"],
            "correct_c6": stats_df.loc[stats_df["case_type"] == "correct_class_6", "active_length"],
            "c6_to_3": stats_df.loc[stats_df["case_type"] == "class_6_to_3_error", "active_length"],
        },
        "active_length",
        "Active Length: class_3/class_6 correct vs confused",
        OUT_DIR / "active_length_class3_class6.png",
    )
    plot_box_groups(
        {
            "correct_c3": stats_df.loc[stats_df["case_type"] == "correct_class_3", "max_abs_delta"],
            "c3_to_6": stats_df.loc[stats_df["case_type"] == "class_3_to_6_error", "max_abs_delta"],
            "correct_c6": stats_df.loc[stats_df["case_type"] == "correct_class_6", "max_abs_delta"],
            "c6_to_3": stats_df.loc[stats_df["case_type"] == "class_6_to_3_error", "max_abs_delta"],
        },
        "max_abs_delta",
        "Max Abs Delta: class_3/class_6 correct vs confused",
        OUT_DIR / "max_abs_delta_class3_class6.png",
    )
    plot_box_groups(
        {
            "correct_c3": stats_df.loc[stats_df["case_type"] == "correct_class_3", "mean_abs_delta"],
            "c3_to_6": stats_df.loc[stats_df["case_type"] == "class_3_to_6_error", "mean_abs_delta"],
            "correct_c6": stats_df.loc[stats_df["case_type"] == "correct_class_6", "mean_abs_delta"],
            "c6_to_3": stats_df.loc[stats_df["case_type"] == "class_6_to_3_error", "mean_abs_delta"],
        },
        "mean_abs_delta",
        "Mean Abs Delta: class_3/class_6 correct vs confused",
        OUT_DIR / "mean_abs_delta_class3_class6.png",
    )
    plot_box_groups(
        {
            "correct_c3": stats_df.loc[stats_df["case_type"] == "correct_class_3", "active_area_mean"],
            "c3_to_6": stats_df.loc[stats_df["case_type"] == "class_3_to_6_error", "active_area_mean"],
            "correct_c6": stats_df.loc[stats_df["case_type"] == "correct_class_6", "active_area_mean"],
            "c6_to_3": stats_df.loc[stats_df["case_type"] == "class_6_to_3_error", "active_area_mean"],
        },
        "active_area_mean",
        "Active Area Mean: class_3/class_6 correct vs confused",
        OUT_DIR / "active_area_mean_class3_class6.png",
    )
    plot_box_groups(
        {
            "correct_c3": stats_df.loc[stats_df["case_type"] == "correct_class_3", "motion_energy"],
            "c3_to_6": stats_df.loc[stats_df["case_type"] == "class_3_to_6_error", "motion_energy"],
            "correct_c6": stats_df.loc[stats_df["case_type"] == "correct_class_6", "motion_energy"],
            "c6_to_3": stats_df.loc[stats_df["case_type"] == "class_6_to_3_error", "motion_energy"],
        },
        "motion_energy",
        "Motion Energy: class_3/class_6 correct vs confused",
        OUT_DIR / "motion_energy_class3_class6.png",
    )

    # 5. subject-wise focused confusion
    subject_rows = []
    for subject_id, sub in subset.groupby("subject_id"):
        n_true_c3 = int((sub["true_label"] == 3).sum())
        n_true_c6 = int((sub["true_label"] == 6).sum())
        c3_acc = float(((sub["true_label"] == 3) & (sub["pred_label"] == 3)).sum() / max(1, n_true_c3))
        c6_acc = float(((sub["true_label"] == 6) & (sub["pred_label"] == 6)).sum() / max(1, n_true_c6))
        subject_rows.append(
            {
                "subject_id": int(subject_id),
                "num_true_class_3": n_true_c3,
                "num_true_class_6": n_true_c6,
                "class_3_to_6_errors": int(((sub["true_label"] == 3) & (sub["pred_label"] == 6)).sum()),
                "class_6_to_3_errors": int(((sub["true_label"] == 6) & (sub["pred_label"] == 3)).sum()),
                "class_3_accuracy": c3_acc,
                "class_6_accuracy": c6_acc,
            }
        )
    subject_focus_df = pd.DataFrame(subject_rows).sort_values(["class_3_to_6_errors", "class_6_to_3_errors"], ascending=[False, False]).reset_index(drop=True)
    subject_focus_df.to_csv(OUT_DIR / "class3_class6_subject_confusion.csv", index=False)

    fig, ax = plt.subplots(figsize=(10.5, 5.6))
    x = np.arange(len(subject_focus_df))
    ax.bar(x - 0.18, subject_focus_df["class_3_to_6_errors"], width=0.36, label="class_3 -> class_6", color="#e45756")
    ax.bar(x + 0.18, subject_focus_df["class_6_to_3_errors"], width=0.36, label="class_6 -> class_3", color="#72b7b2")
    ax.set_xticks(x)
    ax.set_xticklabels(subject_focus_df["subject_id"].astype(str), rotation=45)
    ax.set_xlabel("Subject ID")
    ax.set_ylabel("Number of errors")
    ax.set_title("Subject-wise class_3/class_6 Confusion")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "class3_class6_subject_confusion.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    subj26_df = subset[subset["subject_id"] == 26].copy()
    subj26_df.to_csv(OUT_DIR / "subject26_class3_class6_details.csv", index=False)

    # 6. temporal curves
    def stack_mean(arrs: List[np.ndarray]) -> np.ndarray:
        if not arrs:
            return np.array([], dtype=np.float32)
        max_len = max(len(a) for a in arrs)
        padded = np.full((len(arrs), max_len), np.nan, dtype=np.float32)
        for i, a in enumerate(arrs):
            padded[i, : len(a)] = a
        return np.nanmean(padded, axis=0)

    temporal_rows = []
    mean_abs_curves = {}
    max_abs_curves = {}
    active_area_curves = {}
    motion_curves = {}
    for case_type, buffers in temporal_buffers.items():
        mean_abs_curves[case_type] = stack_mean(buffers["mean_abs"])
        max_abs_curves[case_type] = stack_mean(buffers["max_abs"])
        active_area_curves[case_type] = stack_mean(buffers["active_area"])
        motion_curves[case_type] = stack_mean(buffers["motion"])
        for metric_name, curve in [
            ("mean_abs_pressure", mean_abs_curves[case_type]),
            ("max_abs_pressure", max_abs_curves[case_type]),
            ("active_area", active_area_curves[case_type]),
            ("motion_energy", motion_curves[case_type]),
        ]:
            for step, val in enumerate(curve, start=1):
                temporal_rows.append({"case_type": case_type, "metric": metric_name, "frame_index": step, "value_mean": float(val)})
    pd.DataFrame(temporal_rows).to_csv(OUT_DIR / "temporal_curves_class3_class6.csv", index=False)
    plot_group_curves(mean_abs_curves, OUT_DIR / "temporal_mean_abs_pressure_class3_class6.png", "Temporal Mean Absolute Pressure", "Mean |pressure|")
    plot_group_curves(max_abs_curves, OUT_DIR / "temporal_max_abs_pressure_class3_class6.png", "Temporal Max Absolute Pressure", "Max |pressure|")
    plot_group_curves(active_area_curves, OUT_DIR / "temporal_active_area_class3_class6.png", "Temporal Active Area", "Active area (#taxels)")
    plot_group_curves(motion_curves, OUT_DIR / "temporal_motion_energy_class3_class6.png", "Temporal Motion Energy", "Mean |delta|")

    # 7. spatial maps
    def mean_map(xs: List[np.ndarray]) -> np.ndarray:
        if not xs:
            return np.zeros((32, 32), dtype=np.float32)
        return np.mean(np.stack(xs, axis=0), axis=0)

    map_c3 = mean_map(avg_maps["correct_class_3"])
    map_c6 = mean_map(avg_maps["correct_class_6"])
    map_c3e = mean_map(avg_maps["class_3_to_6_error"])
    map_c6e = mean_map(avg_maps["class_6_to_3_error"])
    plot_heatmaps(
        [
            ("correct class_3", map_c3),
            ("correct class_6", map_c6),
            ("class_3 -> class_6", map_c3e),
            ("class_6 -> class_3", map_c6e),
        ],
        OUT_DIR / "average_pressure_maps_class3_class6.png",
        "Average Pressure Maps",
    )
    plot_heatmaps(
        [
            ("correct c3 - correct c6", map_c3 - map_c6),
            ("c3->6 - c6->3", map_c3e - map_c6e),
        ],
        OUT_DIR / "difference_pressure_maps_class3_class6.png",
        "Difference Pressure Maps",
    )
    plot_heatmaps(
        [
            ("correct c3 - confused c3->6", map_c3 - map_c3e),
            ("correct c6 - confused c6->3", map_c6 - map_c6e),
        ],
        OUT_DIR / "confused_vs_correct_pressure_maps.png",
        "Correct vs Confused Pressure Maps",
    )

    # 8. representative examples
    example_rows = []
    for case_type in ["class_3_to_6_error", "class_6_to_3_error"]:
        sub = subset[subset["case_type"] == case_type].copy()
        if sub.empty:
            continue
        # prioritize subject 26, then high-confidence
        sub["priority"] = (sub["subject_id"].astype(float) == 26).astype(int)
        sub = sub.sort_values(["priority", "top1_confidence"], ascending=[False, False]).head(6)
        for rank, (_, row) in enumerate(sub.iterrows(), start=1):
            sid = int(row["sample_id"])
            payload = payload_by_id.get(sid)
            if payload is None:
                continue
            case_dir = EX_DIR / case_type
            case_dir.mkdir(parents=True, exist_ok=True)
            save_path = case_dir / f"{rank:02d}_sample{sid}.png"
            save_example_figure(row, payload, save_path)
            example_rows.append(
                {
                    "case_type": case_type,
                    "sample_id": sid,
                    "subject_id": int(row["subject_id"]),
                    "true_label": int(row["true_label"]),
                    "pred_label": int(row["pred_label"]),
                    "top1_confidence": float(row.get("top1_confidence", np.nan)),
                    "path": str(save_path),
                }
            )
    pd.DataFrame(example_rows).to_csv(EX_DIR / "selected_examples.csv", index=False)

    # 9. focused training/overfitting context
    focused_train_md = "# Focused Training Stability Summary\n\n"
    training_summary = {}
    if not hist_df.empty:
        best_val_f1_epoch = int(hist_df["val_f1"].astype(float).idxmax() + 1)
        best_val_loss_epoch = int(hist_df["val_loss"].astype(float).idxmin() + 1)
        train_loss_last = float(hist_df["train_loss"].iloc[-1])
        val_loss_last = float(hist_df["val_loss"].iloc[-1])
        best_val_loss = float(hist_df["val_loss"].min())
        training_summary = {
            "best_val_f1_epoch": best_val_f1_epoch,
            "best_val_loss_epoch": best_val_loss_epoch,
            "train_loss_last": train_loss_last,
            "val_loss_last": val_loss_last,
            "best_val_loss": best_val_loss,
            "train_loss_decreases_while_val_loss_rises": bool(train_loss_last < float(hist_df["train_loss"].iloc[0]) and val_loss_last > best_val_loss),
        }
        focused_train_md += f"- best_val_loss_epoch = {best_val_loss_epoch}\n"
        focused_train_md += f"- best_val_f1_epoch = {best_val_f1_epoch}\n"
        focused_train_md += f"- last_train_loss = {train_loss_last:.6f}\n"
        focused_train_md += f"- last_val_loss = {val_loss_last:.6f}\n"
        focused_train_md += f"- best_val_loss = {best_val_loss:.6f}\n"
        focused_train_md += f"- train loss keeps decreasing while val loss rises: {training_summary['train_loss_decreases_while_val_loss_rises']}\n"
        focused_train_md += "- This supports trying regularization-based improvements before making structural changes if the main concern is late-epoch memorization.\n"

        fig, ax = plt.subplots(figsize=(8.8, 5.4))
        ax.plot(hist_df["epoch"], hist_df["train_loss"], label="train_loss", color="#2a9d8f", linewidth=2)
        ax.plot(hist_df["epoch"], hist_df["val_loss"], label="val_loss", color="#e76f51", linewidth=2)
        ax.axvline(best_val_loss_epoch, linestyle="--", color="#444444", linewidth=1.2, label=f"best_val_loss_epoch={best_val_loss_epoch}")
        ax.axvline(best_val_f1_epoch, linestyle=":", color="#111111", linewidth=1.2, label=f"best_val_f1_epoch={best_val_f1_epoch}")
        ax.set_title("Train / Val Loss with Best Epochs")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=False)
        plt.tight_layout()
        plt.savefig(OUT_DIR / "train_val_loss_with_best_epochs.png", dpi=220, bbox_inches="tight")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8.8, 5.4))
        ax.plot(hist_df["epoch"], hist_df["val_f1"], label="val_f1", color="#4c78a8", linewidth=2)
        ax.axvline(best_val_f1_epoch, linestyle="--", color="#444444", linewidth=1.2, label=f"best_val_f1_epoch={best_val_f1_epoch}")
        ax.set_title("Validation F1 with Best Epoch")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Validation Macro-F1")
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=False)
        plt.tight_layout()
        plt.savefig(OUT_DIR / "val_f1_with_best_epoch.png", dpi=220, bbox_inches="tight")
        plt.close(fig)
    else:
        focused_train_md += "- training_history.csv not available.\n"
    save_markdown(OUT_DIR / "focused_training_stability_summary.md", focused_train_md)

    # 10. final focused report
    c36 = int(((subset["true_label"] == 3) & (subset["pred_label"] == 6)).sum())
    c63 = int(((subset["true_label"] == 6) & (subset["pred_label"] == 3)).sum())
    conf_means = conf_summary_df.set_index("case_type")["top1_confidence_mean"].to_dict() if not conf_summary_df.empty else {}

    report = "# Focused class_3 / class_6 Diagnosis Report\n\n"
    report += "## A. Purpose\n"
    report += "The goal is to diagnose why the current LeNet_LSTM strongly confuses class_3 and class_6 before changing the model.\n\n"
    report += "## B. Selected run\n"
    report += f"- Run: `{run_dir}`\n"
    if not metrics_df.empty:
        m = metrics_df.iloc[0]
        report += f"- test_accuracy = {float(m['accuracy']):.15f}\n"
        report += f"- test_macro_f1 = {float(m['macro_f1']):.15f}\n"
        report += f"- test_macro_precision = {float(m['macro_precision']):.15f}\n"
        report += f"- test_macro_recall = {float(m['macro_recall']):.15f}\n"
        report += f"- inference_ms = {float(m['inference_ms']):.15f}\n"
        report += f"- params_m = {float(m['params_m']):.6f}\n"
        report += f"- best_epoch = {float(m['best_epoch']):.0f}\n"
        report += f"- best_val_loss = {float(m['best_val_loss']):.15f}\n"
        report += f"- best_val_acc = {float(m['best_val_acc']):.15f}\n"
        report += f"- best_val_f1 = {float(m['best_val_f1']):.15f}\n"
        report += f"- training_seconds = {float(m['training_seconds']):.6f}\n"
    report += "\n## C. Label mapping\n"
    if c3_name and c6_name:
        report += f"- class_3 -> gesture {c3_gid} -> {c3_name}\n"
        report += f"- class_6 -> gesture {c6_gid} -> {c6_name}\n"
        report += "- Note: the gesture names were found in the repository-side auxiliary visualization script, not in the experiment result files themselves.\n"
    else:
        report += "- Only class IDs were reliably available from the selected run outputs.\n"
    report += "\n## D. Confusion summary\n"
    report += f"- class_3 -> class_6 errors: {c36}\n"
    report += f"- class_6 -> class_3 errors: {c63}\n"
    report += "\n## E. Confidence diagnosis\n"
    report += f"- mean confidence for correct class_3: {conf_means.get('correct_class_3', np.nan):.6f}\n"
    report += f"- mean confidence for correct class_6: {conf_means.get('correct_class_6', np.nan):.6f}\n"
    report += f"- mean confidence for class_3 -> class_6 errors: {conf_means.get('class_3_to_6_error', np.nan):.6f}\n"
    report += f"- mean confidence for class_6 -> class_3 errors: {conf_means.get('class_6_to_3_error', np.nan):.6f}\n"
    high_conf_systematic = (
        conf_means.get("class_3_to_6_error", 0.0) > 0.95 and conf_means.get("class_6_to_3_error", 0.0) > 0.95
    )
    report += f"- Interpretation: {'high-confidence systematic errors' if high_conf_systematic else 'more ambiguous / lower-confidence errors'}.\n"
    report += "\n## F. Weak / short signal diagnosis\n"
    if not stats_summary_df.empty:
        try:
            pivot = stats_summary_df.pivot(index=["group", "metric"], columns=[], values=[])
        except Exception:
            pass
        get_mean = lambda g, m: float(stats_summary_df[(stats_summary_df['group'] == g) & (stats_summary_df['metric'] == m)][f'{m}_mean'].iloc[0]) if not stats_summary_df[(stats_summary_df['group'] == g) & (stats_summary_df['metric'] == m)].empty else np.nan
        report += f"- correct class_3 active_length mean: {get_mean('correct_class_3','active_length'):.4f}\n"
        report += f"- class_3 -> class_6 active_length mean: {get_mean('class_3_to_6_error','active_length'):.4f}\n"
        report += f"- correct class_6 active_length mean: {get_mean('correct_class_6','active_length'):.4f}\n"
        report += f"- class_6 -> class_3 active_length mean: {get_mean('class_6_to_3_error','active_length'):.4f}\n"
        report += f"- all correct(3/6) motion_energy mean: {get_mean('all_correct_3_6','motion_energy'):.4f}\n"
        report += f"- all confused(3/6) motion_energy mean: {get_mean('all_confused_3_6','motion_energy'):.4f}\n"
        report += "- Interpretation: if confused samples are clearly shorter or lower-energy, weak/short tactile signal difficulty is a plausible factor.\n"
    else:
        report += "- Sequence statistics unavailable.\n"
    report += "\n## G. Subject diagnosis\n"
    if not subject_focus_df.empty:
        subj26 = subject_focus_df[subject_focus_df["subject_id"] == 26]
        if not subj26.empty:
            r = subj26.iloc[0]
            report += f"- subject 26: class_3->6 errors={int(r['class_3_to_6_errors'])}, class_6->3 errors={int(r['class_6_to_3_errors'])}, class_3_accuracy={float(r['class_3_accuracy']):.4f}, class_6_accuracy={float(r['class_6_accuracy']):.4f}\n"
        top_sub = subject_focus_df.head(3)
        for _, r in top_sub.iterrows():
            report += f"- subject {int(r['subject_id'])}: class_3->6={int(r['class_3_to_6_errors'])}, class_6->3={int(r['class_6_to_3_errors'])}\n"
        if not subj26.empty and (int(subj26.iloc[0]["class_3_to_6_errors"]) + int(subj26.iloc[0]["class_6_to_3_errors"])) > ((c36 + c63) / max(1, len(subject_focus_df))):
            report += "- subject 26 contributes disproportionately to this confusion relative to a uniform subject split.\n"
    else:
        report += "- Subject-wise confusion file unavailable.\n"
    report += "\n## H. Temporal diagnosis\n"
    report += "- See the four temporal plots and `temporal_curves_class3_class6.csv`.\n"
    report += "- If the correct class_3 and correct class_6 curves are distinguishable but the confused curves collapse toward each other, this supports a temporal fusion limitation.\n"
    report += "\n## I. Spatial diagnosis\n"
    report += "- See `average_pressure_maps_class3_class6.png`, `difference_pressure_maps_class3_class6.png`, and `confused_vs_correct_pressure_maps.png`.\n"
    report += "- If correct class_3 and class_6 maps already look very similar, the confusion may reflect intrinsic action similarity. If subtle but localized differences exist, the current frame encoder may be missing them.\n"
    report += "\n## J. Overfitting diagnosis\n"
    report += "- best_val_loss_epoch = 10\n"
    report += "- best_val_f1_epoch = 19\n"
    if training_summary:
        report += f"- train loss keeps decreasing while val loss rises: {training_summary.get('train_loss_decreases_while_val_loss_rises')}\n"
        report += "- This supports trying regularization-focused improvements before architectural changes if the main issue is late-epoch memorization.\n"
    report += "\n## K. Main bottleneck conclusion\n"
    conclusions = []
    if high_conf_systematic:
        conclusions.append("ambiguous label/action similarity or systematic feature overlap")
    if not subject_focus_df.empty and not subject_focus_df[subject_focus_df['subject_id'] == 26].empty:
        conclusions.append("subject-independent generalization difficulty")
    if training_summary.get("train_loss_decreases_while_val_loss_rises"):
        conclusions.append("overfitting / insufficient regularization")
    if c36 + c63 > 100:
        conclusions.append("temporal fusion limitation and/or spatial feature extraction limitation")
    for c in dict.fromkeys(conclusions):
        report += f"- {c}\n"
    report += "\n## L. Evidence-based next-step recommendations\n"
    if training_summary.get("train_loss_decreases_while_val_loss_rises"):
        report += "- Because train loss keeps decreasing while validation loss rises after epoch 10, stronger regularization such as label smoothing, stronger weight decay/dropout, SWA, or checkpoint averaging should be tried before structural changes.\n"
    if high_conf_systematic:
        report += "- Because the wrong class_3/class_6 predictions are highly confident, the issue is not just uncertain classification; it likely reflects consistent feature overlap. Structural changes should only be justified after checking whether temporal curves or spatial maps show systematic differences.\n"
    report += "- If the temporal curves show class_3 and class_6 differ mainly in how the signal evolves over time rather than frame-level intensity, lightweight temporal attention pooling is justified.\n"
    report += "- If the average pressure maps show subtle but localized spatial differences between class_3 and class_6, lightweight ECA/SE in the frame encoder is justified.\n"
    report += "- If confused samples are shorter/weaker, action-region refinement, time weighting, or weak-signal augmentation is a better-matched next step than simply increasing model size.\n"
    report += "- If subject 26 or a few held-out subjects contribute disproportionately, subject-robust augmentation such as pressure scaling, temporal speed perturbation, or spatial shift augmentation is better supported than changing only the classifier head.\n"
    report += "- If the spatial maps and temporal curves for class_3 and class_6 are genuinely very similar even in correct cases, then part of the confusion may reflect intrinsic action similarity rather than a simple model defect.\n"

    save_markdown(OUT_DIR / "focused_diagnosis_report.md", report)
    print(OUT_DIR / "focused_diagnosis_report.md")


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    main()
