#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tacact.data import FILENAME_RE, TacActDataset


DEFAULT_TACACT_GESTURE_NAMES: Dict[int, str] = {
    1: "Pull",
    2: "Squeeze",
    3: "Push",
    4: "Hold",
    5: "Grasp",
    6: "Poke",
    7: "Static Drag",
    8: "Strongly Hit (Once)",
    9: "Soft Slide",
    10: "Scratch",
    11: "Soft Hit (Twice)",
    12: "Sliding Drag",
}


@dataclass
class RawSample:
    path: Path
    gesture_id: int
    subject: int
    variant: str
    repeat: int


@dataclass
class LoadedCurve:
    sample: RawSample
    frames_shape: Tuple[int, int, int]
    curve: np.ndarray
    peak: float
    roughness: float


def _infer_gesture_from_path_fallback(path: Path) -> Optional[int]:
    """
    Best-effort gesture parser when filename does not follow the canonical pattern.
    """
    stem = path.stem
    tokens = stem.split("_")

    # Fallback A: second token in "<subject>_<gesture>_..."
    if len(tokens) >= 2:
        try:
            g = int(tokens[1])
            if 1 <= g <= 999:
                return g
        except Exception:
            pass

    # Fallback B: use first integer in parent directory name
    parent_text = path.parent.name
    m = re.search(r"(\d+)", parent_text)
    if m is not None:
        g = int(m.group(1))
        if 1 <= g <= 999:
            return g
    return None


def discover_raw_samples(data_root: Path) -> Tuple[List[RawSample], Dict[str, int]]:
    """
    Discover TacAct-style raw files.
    Returns samples and discovery stats used for startup diagnostics.
    """
    xlsx_paths = sorted(data_root.rglob("*.xlsx"))
    samples: List[RawSample] = []
    regex_hits = 0
    fallback_hits = 0
    skipped = 0

    for path in xlsx_paths:
        m = FILENAME_RE.match(path.name)
        if m is None:
            g = _infer_gesture_from_path_fallback(path)
            if g is None:
                skipped += 1
                continue
            fallback_hits += 1
            samples.append(
                RawSample(
                    path=path,
                    gesture_id=int(g),
                    subject=-1,
                    variant="UNK",
                    repeat=-1,
                )
            )
            continue

        regex_hits += 1
        samples.append(
            RawSample(
                path=path,
                gesture_id=int(m.group("gesture")),
                subject=int(m.group("subject")),
                variant=str(m.group("variant")),
                repeat=int(m.group("repeat")),
            )
        )

    stats = {
        "xlsx_total": len(xlsx_paths),
        "regex_hits": int(regex_hits),
        "fallback_hits": int(fallback_hits),
        "skipped_unrecognized": int(skipped),
        "usable_samples": len(samples),
    }
    return samples, stats


def compute_response_curve(frames: np.ndarray, reduction: str = "sum") -> np.ndarray:
    """
    Convert one sample sequence into a 1D pressure curve while preserving raw length.

    frames: [T, H, W]
    baseline subtraction:
        X' = X - X[0]
    scalar response per frame:
        sum:  s_t = sum(abs(X'_t))
        mean: s_t = mean(abs(X'_t))
    """
    if frames.ndim != 3:
        raise ValueError(f"Expected raw frames with ndim=3, got shape={tuple(frames.shape)}")
    if frames.shape[0] == 0:
        return np.zeros((0,), dtype=np.float64)
    delta = frames.astype(np.float64, copy=False) - frames[0:1].astype(np.float64, copy=False)
    abs_delta = np.abs(delta)
    if reduction == "mean":
        return np.mean(abs_delta, axis=(1, 2))
    return np.sum(abs_delta, axis=(1, 2))


def smooth_curve(curve: np.ndarray, window: int) -> np.ndarray:
    """
    Apply weak moving-average smoothing to avoid distorting raw trend.
    """
    if window <= 1 or len(curve) <= 2:
        return curve.copy()
    if window % 2 == 0:
        window += 1
    window = max(1, min(window, len(curve)))
    if window == 1:
        return curve.copy()
    kernel = np.ones(window, dtype=np.float64) / float(window)
    return np.convolve(curve, kernel, mode="same")


def load_class_name_map(gesture_ids: Sequence[int], class_names_json: Path | None) -> Dict[int, str]:
    if class_names_json is None:
        return {gid: DEFAULT_TACACT_GESTURE_NAMES.get(gid, f"Gesture {gid}") for gid in gesture_ids}

    payload = json.loads(class_names_json.read_text(encoding="utf-8"))
    out: Dict[int, str] = {}
    for gid in gesture_ids:
        if str(gid) in payload:
            out[gid] = str(payload[str(gid)])
        elif gid in payload:
            out[gid] = str(payload[gid])
        else:
            out[gid] = f"Gesture {gid}"
    return out


def load_curves_grouped(
    samples: Sequence[RawSample],
    reduction: str,
    curve_cache_dir: Optional[Path],
    disable_curve_cache: bool,
) -> Tuple[Dict[int, List[LoadedCurve]], List[Tuple[Path, str]]]:
    """
    Robust loader:
    - keeps running when one file fails
    - records failed files for diagnostics
    """
    grouped: Dict[int, List[LoadedCurve]] = {}
    failures: List[Tuple[Path, str]] = []

    if curve_cache_dir is not None and not disable_curve_cache:
        curve_cache_dir.mkdir(parents=True, exist_ok=True)

    for s in samples:
        try:
            cache_hit = False
            cache_path = None
            curve: np.ndarray
            frames_shape: Tuple[int, int, int]

            if curve_cache_dir is not None and not disable_curve_cache:
                key_raw = f"{s.path.resolve()}|{s.path.stat().st_mtime_ns}|{reduction}"
                key = hashlib.sha1(key_raw.encode("utf-8")).hexdigest()[:20]
                cache_path = curve_cache_dir / f"{s.path.stem}_{key}.npz"
                if cache_path.exists():
                    payload = np.load(cache_path, allow_pickle=False)
                    curve = payload["curve"].astype(np.float64)
                    shape_arr = payload["shape"].astype(np.int64)
                    frames_shape = (int(shape_arr[0]), int(shape_arr[1]), int(shape_arr[2]))
                    cache_hit = True
                else:
                    frames = TacActDataset._read_excel_optimized(s.path)
                    if frames.ndim != 3:
                        raise ValueError(f"raw frame ndim != 3: {frames.ndim}")
                    curve = compute_response_curve(frames, reduction=reduction)
                    frames_shape = (int(frames.shape[0]), int(frames.shape[1]), int(frames.shape[2]))
            else:
                frames = TacActDataset._read_excel_optimized(s.path)
                if frames.ndim != 3:
                    raise ValueError(f"raw frame ndim != 3: {frames.ndim}")
                curve = compute_response_curve(frames, reduction=reduction)
                frames_shape = (int(frames.shape[0]), int(frames.shape[1]), int(frames.shape[2]))

            if len(curve) == 0:
                raise ValueError("empty curve after conversion")
            peak = float(np.max(curve))
            roughness = float(np.mean(np.abs(np.diff(curve)))) if len(curve) > 1 else 0.0

            if (not cache_hit) and (cache_path is not None):
                tmp_path = cache_path.with_suffix(".tmp.npz")
                np.savez_compressed(
                    tmp_path,
                    curve=curve.astype(np.float64, copy=False),
                    shape=np.asarray(frames_shape, dtype=np.int64),
                )
                tmp_path.replace(cache_path)

            grouped.setdefault(int(s.gesture_id), []).append(
                LoadedCurve(
                    sample=s,
                    frames_shape=frames_shape,
                    curve=curve.astype(np.float64, copy=False),
                    peak=peak,
                    roughness=roughness,
                )
            )
        except Exception as e:
            failures.append((s.path, str(e)))
    return grouped, failures


def pick_representative(curves: Sequence[LoadedCurve]) -> Tuple[int, Dict[str, float]]:
    """
    Pick the most representative sample in one gesture class by joint score:
    - length close to class median
    - peak close to class median
    - roughness close to class median (small weight, encourages typical smoothness)
    """
    if not curves:
        raise ValueError("Cannot select representative from empty class.")
    if len(curves) == 1:
        return 0, {"score": 0.0, "len_score": 0.0, "peak_score": 0.0, "rough_score": 0.0}

    lengths = np.asarray([len(x.curve) for x in curves], dtype=np.float64)
    peaks = np.asarray([x.peak for x in curves], dtype=np.float64)
    rough = np.asarray([x.roughness for x in curves], dtype=np.float64)

    med_len = float(np.median(lengths))
    med_peak = float(np.median(peaks))
    med_rough = float(np.median(rough))

    eps = 1e-12
    best_idx = 0
    best_score = float("inf")
    best_details = {"score": float("inf"), "len_score": 0.0, "peak_score": 0.0, "rough_score": 0.0}
    for i, x in enumerate(curves):
        len_score = abs(float(len(x.curve)) - med_len) / max(med_len, eps)
        peak_score = abs(float(x.peak) - med_peak) / max(abs(med_peak), eps)
        rough_score = abs(float(x.roughness) - med_rough) / max(abs(med_rough), eps)
        score = 0.45 * len_score + 0.45 * peak_score + 0.10 * rough_score
        if score < best_score:
            best_score = score
            best_idx = i
            best_details = {
                "score": float(score),
                "len_score": float(len_score),
                "peak_score": float(peak_score),
                "rough_score": float(rough_score),
            }
    return best_idx, best_details


def subplot_tag(i: int) -> str:
    """
    0 -> (a), 1 -> (b), ... 25 -> (z), 26 -> (aa)
    """
    n = i
    letters: List[str] = []
    while True:
        n, r = divmod(n, 26)
        letters.append(chr(ord("a") + r))
        if n == 0:
            break
        n -= 1
    return f"({''.join(reversed(letters))})"


def print_startup_report(
    data_root: Path,
    discover_stats: Dict[str, int],
    grouped: Dict[int, List[LoadedCurve]],
    failures: Sequence[Tuple[Path, str]],
) -> None:
    print("[RawTrend] Data organization report")
    print(f"  data_root: {data_root}")
    print(f"  xlsx_total: {discover_stats['xlsx_total']}")
    print(f"  regex_hits: {discover_stats['regex_hits']}")
    print(f"  fallback_hits: {discover_stats['fallback_hits']}")
    print(f"  skipped_unrecognized: {discover_stats['skipped_unrecognized']}")
    print(f"  usable_samples: {discover_stats['usable_samples']}")
    print(f"  load_failures: {len(failures)}")
    print(f"  detected_classes: {len(grouped)}")

    for gid in sorted(grouped.keys()):
        items = grouped[gid]
        lengths = np.asarray([len(x.curve) for x in items], dtype=np.int64)
        shape_examples = sorted(set(x.frames_shape for x in items))
        shape_text = ", ".join(str(s) for s in shape_examples[:3])
        if len(shape_examples) > 3:
            shape_text += ", ..."
        print(
            f"  - Gesture {gid}: n={len(items)} "
            f"len[min/med/max]={int(lengths.min())}/{float(np.median(lengths)):.1f}/{int(lengths.max())} "
            f"format examples={shape_text}"
        )

    if failures:
        print("[RawTrend] Failed files (showing first 10):")
        for p, msg in failures[:10]:
            print(f"  - {p}: {msg}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Plot one representative raw pressure trend per gesture class for TacAct."
    )
    p.add_argument("--data_root", type=Path, required=True, help="TacAct raw data root (contains *.xlsx files).")
    p.add_argument(
        "--output_png",
        type=Path,
        default=Path("outputs/visualizations/tacact_raw_gesture_trends.png"),
        help="Output figure path (PNG).",
    )
    p.add_argument(
        "--output_csv",
        type=Path,
        default=Path("outputs/visualizations/tacact_raw_gesture_trends_summary.csv"),
        help="Output summary CSV path.",
    )
    p.add_argument(
        "--reduction",
        choices=["sum", "mean"],
        default="sum",
        help="Frame-wise pressure scalar aggregation.",
    )
    p.add_argument(
        "--smooth_window",
        type=int,
        default=3,
        help="Weak moving-average smoothing window. Use 1 to disable.",
    )
    p.add_argument(
        "--class_names_json",
        type=Path,
        default=None,
        help="Optional JSON mapping gesture_id -> class_name.",
    )
    p.add_argument(
        "--overlay_all_samples",
        action="store_true",
        help="Overlay all curves (light blue) and highlight representative curve (dark blue).",
    )
    p.add_argument("--dpi", type=int, default=300, help="Figure DPI.")
    p.add_argument("--sharey", action="store_true", help="Share Y-axis among subplots.")
    p.add_argument(
        "--curve_cache_dir",
        type=Path,
        default=Path(".cache_tacact_raw_curve"),
        help="Cache directory for raw per-sample 1D curves (keeps raw sequence length).",
    )
    p.add_argument(
        "--disable_curve_cache",
        action="store_true",
        help="Disable raw-curve cache and always read .xlsx.",
    )
    return p


def main() -> None:
    args = build_parser().parse_args()
    if args.smooth_window <= 0:
        raise ValueError("--smooth_window must be >= 1")
    if not args.data_root.exists():
        raise RuntimeError(
            f"Data root does not exist: {args.data_root}\n"
            "Expected: a directory containing TacAct .xlsx files."
        )

    samples, discover_stats = discover_raw_samples(args.data_root)
    if discover_stats["xlsx_total"] == 0:
        raise RuntimeError(
            f"No .xlsx files found under: {args.data_root}\n"
            "Expected format: TacAct raw files such as <subject>_<gesture>_<variant>_<repeat>.xlsx"
        )
    if not samples:
        raise RuntimeError(
            "No usable samples after discovery.\n"
            f"Found .xlsx files: {discover_stats['xlsx_total']}, but none could be mapped to gesture IDs.\n"
            "Expected format: <subject>_<gesture>_<variant>_<repeat>.xlsx\n"
            "Suggestion: rename files to canonical pattern or place each gesture in folder names containing gesture id."
        )

    grouped, failures = load_curves_grouped(
        samples,
        reduction=str(args.reduction),
        curve_cache_dir=args.curve_cache_dir,
        disable_curve_cache=bool(args.disable_curve_cache),
    )
    if not grouped:
        err_preview = "\n".join([f"- {p}: {msg}" for p, msg in failures[:5]])
        raise RuntimeError(
            "All samples failed to parse as raw tactile sequence.\n"
            "Expected each sample to be convertible to frames [T,32,32] from Excel matrix.\n"
            f"Examples of failures:\n{err_preview if err_preview else '- <none>'}"
        )

    gesture_ids = sorted(grouped.keys())
    class_name_map = load_class_name_map(gesture_ids, args.class_names_json)
    print_startup_report(args.data_root, discover_stats, grouped, failures)
    if args.disable_curve_cache:
        print("[RawTrend] Raw-curve cache: disabled")
    else:
        print(f"[RawTrend] Raw-curve cache dir: {args.curve_cache_dir}")

    n_classes = len(gesture_ids)
    n_cols = min(4, max(1, n_classes))
    n_rows = int(math.ceil(n_classes / n_cols))

    fig_w = 4.2 * n_cols
    fig_h = 2.9 * n_rows
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_w, fig_h), sharey=bool(args.sharey))
    if isinstance(axes, np.ndarray):
        axes_flat = axes.ravel().tolist()
    else:
        axes_flat = [axes]

    summary_rows = []
    for i, gid in enumerate(gesture_ids):
        ax = axes_flat[i]
        items = grouped.get(gid, [])
        if not items:
            ax.set_visible(False)
            continue

        rep_idx, score_detail = pick_representative(items)
        rep = items[rep_idx]
        rep_curve = smooth_curve(rep.curve, window=int(args.smooth_window))

        if args.overlay_all_samples:
            for x in items:
                xf = np.arange(len(x.curve), dtype=np.int64)
                ax.plot(xf, x.curve, color="#9ecae1", linewidth=0.8, alpha=0.26)

        x = np.arange(len(rep_curve), dtype=np.int64)
        ax.plot(x, rep_curve, color="#08519c", linewidth=2.0, alpha=0.98)

        lengths = np.asarray([len(x.curve) for x in items], dtype=np.int64)
        title = class_name_map.get(gid, f"Gesture {gid}")
        ax.set_title(title, fontsize=10, color="black")
        ax.text(
            0.01,
            0.02,
            subplot_tag(i),
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=9,
            color="black",
        )
        ax.text(
            0.99,
            0.02,
            f"n={len(items)}",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=8,
            color="black",
        )
        ax.grid(True, alpha=0.15, linewidth=0.6)
        ax.tick_params(axis="both", labelsize=8, colors="black")
        ax.set_facecolor("white")
        ax.set_xlabel("Frame index", fontsize=8, color="black")
        ax.set_ylabel("Total pressure" if args.reduction == "sum" else "Mean pressure", fontsize=8, color="black")

        summary_rows.append(
            {
                "gesture_id": int(gid),
                "class_name": title,
                "n_samples": int(len(items)),
                "mean_length": float(np.mean(lengths)),
                "median_length": float(np.median(lengths)),
                "max_length": int(np.max(lengths)),
                "min_length": int(np.min(lengths)),
                "representative_path": str(rep.sample.path),
                "representative_length": int(len(rep.curve)),
                "representative_peak": float(rep.peak),
                "representative_score": float(score_detail["score"]),
                "representative_len_score": float(score_detail["len_score"]),
                "representative_peak_score": float(score_detail["peak_score"]),
                "representative_rough_score": float(score_detail["rough_score"]),
            }
        )
        print(
            f"[RawTrend] Gesture {gid} representative -> "
            f"path={rep.sample.path.name}, len={len(rep.curve)}, peak={rep.peak:.3f}, score={score_detail['score']:.6f}"
        )

    for j in range(n_classes, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.patch.set_facecolor("white")
    fig.supxlabel("Frame index", fontsize=11, color="black")
    fig.supylabel("Total pressure" if args.reduction == "sum" else "Mean pressure", fontsize=11, color="black")
    fig.suptitle(
        "TacAct Representative Raw Pressure Trends",
        fontsize=12,
        color="black",
    )
    fig.tight_layout(rect=[0.02, 0.03, 1.0, 0.96])

    args.output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_png, dpi=int(args.dpi), bbox_inches="tight", facecolor="white")
    plt.close(fig)

    import pandas as pd

    df = pd.DataFrame(summary_rows).sort_values("gesture_id").reset_index(drop=True)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output_csv, index=False)

    print(f"[RawTrend] Saved figure: {args.output_png}")
    print(f"[RawTrend] Saved summary: {args.output_csv}")
    print(
        "[RawTrend] Class labels source: gesture IDs parsed from TacAct filenames "
        "(optional --class_names_json can provide human-readable names)."
    )


if __name__ == "__main__":
    main()
