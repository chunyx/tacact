from __future__ import annotations

import hashlib
import os
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, get_worker_info

FILENAME_RE = re.compile(r"^(?P<subject>\d+)_(?P<gesture>\d+)_(?P<variant>[A-Za-z]+)_(?P<repeat>\d+)\.xlsx$")
LABEL_MAP: Dict[int, int] = {i: i - 1 for i in range(1, 13)}


@dataclass
class SampleMeta:
    path: Path
    subject: int
    gesture: int
    variant: str
    repeat: int


class TacActDataset(Dataset):
    """TacAct dataset with disk cache, optional memory preload, and robust preprocessing."""

    def __init__(
        self,
        root_dir: Path,
        n_frames: int = 80,
        threshold: Optional[float] = None,
        threshold_method: str = "mean_std",
        threshold_k: float = 3.0,
        background_frames: int = 5,
        center_weight_sigma: Optional[float] = None,
        clip_mode: str = "weighted_center",
        cache_dir: Optional[Path] = None,
        preload_cache: bool = True,
        num_workers: int = 4,
        segmentation_log: bool = False,
        cache_trace: bool = False,
    ) -> None:
        self.root_dir = root_dir
        self.n_frames = n_frames
        self.threshold = threshold
        self.threshold_method = threshold_method
        self.threshold_k = float(threshold_k)
        self.background_frames = int(background_frames)
        self.center_weight_sigma = None if center_weight_sigma is None else float(center_weight_sigma)
        self.segmentation_log = bool(segmentation_log)
        self.clip_mode = clip_mode
        self.cache_dir = cache_dir
        self.preload_cache = preload_cache
        self.enable_memory_cache = bool(preload_cache)
        self.num_workers = num_workers
        env_trace = os.environ.get("TACACT_CACHE_TRACE", "").strip().lower() in {"1", "true", "yes", "y"}
        self.cache_trace = bool(cache_trace or env_trace)
        self.cache_trace_limit = int(os.environ.get("TACACT_CACHE_TRACE_LIMIT", "200"))
        self.cache_trace_every = int(os.environ.get("TACACT_CACHE_TRACE_EVERY", "200"))
        self.last_segmentation_info: Dict[str, float | int | str] = {}

        if self.threshold_method not in {"mean_std", "fixed"}:
            raise ValueError(f"Unsupported threshold_method={self.threshold_method}. Use 'mean_std' or 'fixed'.")
        if self.threshold_method == "fixed" and self.threshold is None:
            raise ValueError("threshold must be provided when threshold_method='fixed'.")
        if self.background_frames < 1:
            raise ValueError("background_frames must be >= 1.")

        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.samples = self._discover_samples(root_dir)
        if not self.samples:
            raise RuntimeError(f"No matching .xlsx files found under {root_dir}")

        self._memory_cache: Dict[str, np.ndarray] = {}
        self._cache_lock = threading.Lock()
        self._cache_stats = {
            "total": 0,
            "memory_hit": 0,
            "disk_hit": 0,
            "disk_miss_xlsx": 0,
            "raw_xlsx_read": 0,
        }
        self._trace_printed = 0
        if self.cache_dir is not None:
            existing = 0
            for meta in self.samples:
                if self._cache_path_for(meta.path).exists():
                    existing += 1
            missing = len(self.samples) - existing
            ratio = existing / max(1, len(self.samples))
            print(
                f"[Dataset Cache Scan] dir={self.cache_dir} total={len(self.samples)} "
                f"existing={existing} missing={missing} hit_ratio={ratio:.3f}"
            )

        if self.preload_cache and self.cache_dir is not None:
            print("[Dataset] Preload ENABLED (single-process mode)")
            self._preload_cache()
        else:
            print("[Dataset] Preload DISABLED (parallel mode, lazy loading)")

    @staticmethod
    def _discover_samples(root_dir: Path) -> List[SampleMeta]:
        out: List[SampleMeta] = []
        for path in sorted(root_dir.rglob("*.xlsx")):
            match = FILENAME_RE.match(path.name)
            if not match:
                continue
            gesture = int(match.group("gesture"))
            if gesture not in LABEL_MAP:
                continue
            out.append(
                SampleMeta(
                    path=path,
                    subject=int(match.group("subject")),
                    gesture=gesture,
                    variant=match.group("variant"),
                    repeat=int(match.group("repeat")),
                )
            )
        return out

    @staticmethod
    def _uniform_sample_indices(length: int, target_len: int) -> np.ndarray:
        """Uniformly sample target_len indices from the full sequence [0, length-1]."""
        if length <= target_len:
            return np.arange(length, dtype=np.int64)
        idx = np.linspace(0, length - 1, num=target_len, dtype=np.float64)
        return np.clip(np.round(idx).astype(np.int64), 0, length - 1)

    @staticmethod
    def _gaussian_center_weights(length: int, sigma: float) -> np.ndarray:
        """
        Build normalized Gaussian center weights.

        Weights are normalized by max value, so center weight is 1.0 and edges are attenuated.
        """
        if length <= 1:
            return np.ones((length,), dtype=np.float32)
        center = 0.5 * float(length - 1)
        sigma = max(float(sigma), 1e-6)
        x = np.arange(length, dtype=np.float32)
        w = np.exp(-((x - center) ** 2) / (2.0 * sigma * sigma))
        w = w / max(float(w.max()), 1e-12)
        return w.astype(np.float32)

    def _weighted_center_resample(self, frames: np.ndarray, target_len: int) -> np.ndarray:
        """
        Uniformly cover the full active segment, then apply Gaussian center weighting.

        This keeps global temporal coverage (start->end) while emphasizing the middle phase.
        """
        idx = self._uniform_sample_indices(frames.shape[0], target_len)
        sampled = frames[idx]
        sigma = self.center_weight_sigma if self.center_weight_sigma is not None else (target_len / 6.0)
        weights = self._gaussian_center_weights(target_len, sigma=sigma)
        return sampled * weights[:, None, None]

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_cache_lock"] = None
        state["_memory_cache"] = {}
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._cache_lock = threading.Lock()
        if not hasattr(self, "_memory_cache") or self._memory_cache is None:
            self._memory_cache = {}

    @staticmethod
    def _read_excel_optimized(path: Path) -> np.ndarray:
        """Read an Excel sample and robustly convert to [T, 32, 32] float32 frames."""
        try:
            df = pd.read_excel(path, header=None, engine="openpyxl", dtype=np.float32)
            arr = df.to_numpy(dtype=np.float32)
        except Exception:
            df = pd.read_excel(path, header=None, dtype=np.float32)
            arr = df.to_numpy(dtype=np.float32)

        if arr.ndim != 2:
            raise ValueError(f"Unexpected excel dimensionality in {path}")

        if np.any(np.isnan(arr)):
            row_mask = ~np.all(np.isnan(arr), axis=1)
            col_mask = ~np.all(np.isnan(arr), axis=0)
            arr = arr[row_mask][:, col_mask]
            arr = np.where(np.isnan(arr), 0.0, arr)

        if arr.size == 0:
            raise ValueError(f"Empty numeric matrix in {path}")

        if arr.shape[1] == 32:
            n_rows = arr.shape[0]
            if n_rows % 32 != 0:
                n_frames = n_rows // 32
                arr = arr[: n_frames * 32]
            frames = arr.reshape(-1, 32, 32)
        elif arr.shape[0] == 32:
            n_cols = arr.shape[1]
            if n_cols % 32 != 0:
                n_frames = n_cols // 32
                arr = arr[:, : n_frames * 32]
            frames = arr.T.reshape(-1, 32, 32)
        elif arr.shape[1] == 1024:
            frames = arr.reshape(-1, 32, 32)
        elif arr.shape[0] == 1024:
            frames = arr.T.reshape(-1, 32, 32)
        else:
            total_elements = arr.size
            n_frames = total_elements // (32 * 32)
            if n_frames == 0:
                raise ValueError(f"Cannot infer 32x32 frames from {path}, got {arr.shape}")
            usable = n_frames * 32 * 32
            flat = arr.ravel()[:usable]
            frames = flat.reshape(-1, 32, 32)

        return frames.astype(np.float32)

    def _compute_segmentation_threshold(self, baseline_diff_frames: np.ndarray) -> Tuple[float, float, float]:
        """
        Compute segmentation threshold from background statistics.

        Background is defined as the first N frames of each sequence after baseline subtraction.
        """
        bg_n = min(self.background_frames, max(1, baseline_diff_frames.shape[0]))
        background = np.abs(baseline_diff_frames[:bg_n])
        bg_mean = float(background.mean())
        bg_std = float(background.std())

        if self.threshold_method == "mean_std":
            threshold = bg_mean + self.threshold_k * bg_std
        else:
            threshold = float(self.threshold)
        return float(threshold), bg_mean, bg_std

    def _preprocess(self, frames: np.ndarray, sample_path: Optional[Path] = None) -> np.ndarray:
        frames = frames.copy()
        frames -= frames[0:1]
        bg_n = min(self.background_frames, max(1, frames.shape[0]))
        threshold, bg_mean, bg_std = self._compute_segmentation_threshold(frames)
        active = np.any(np.abs(frames) > threshold, axis=(1, 2))

        if np.any(active):
            active_indices = np.where(active)[0]
            start = int(active_indices[0])
            end = int(active_indices[-1]) + 1
            frames = frames[start:end]
        else:
            start = 0
            end = 1
            frames = frames[:1]

        self.last_segmentation_info = {
            "threshold_method": self.threshold_method,
            "background_mean": float(bg_mean),
            "background_std": float(bg_std),
            "computed_threshold": float(threshold),
            "threshold_k": float(self.threshold_k),
            "background_frames": int(bg_n),
            "signal_rule": "active if any taxel satisfies |delta| > threshold",
            "segment_start": int(start),
            "segment_end": int(end),
            "sample_path": str(sample_path) if sample_path is not None else "",
        }
        if self.segmentation_log:
            print(
                "[Segmentation] "
                f"method={self.threshold_method} "
                f"bg_mean={bg_mean:.6f} bg_std={bg_std:.6f} "
                f"threshold={threshold:.6f} "
                f"sample={self.last_segmentation_info['sample_path']}"
            )

        t = frames.shape[0]
        if t < self.n_frames:
            pad = np.zeros((self.n_frames - t, 32, 32), dtype=np.float32)
            frames = np.concatenate([frames, pad], axis=0)
        elif t > self.n_frames:
            # Long-sequence strategy (single path): uniform full-sequence sampling + center weighting.
            # clip_mode is kept only for backward compatibility and no longer affects long-sequence behavior.
            frames = self._weighted_center_resample(frames, self.n_frames)

        return frames.astype(np.float32)

    def _cache_path_for(self, path: Path) -> Path:
        key = hashlib.sha1(str(path).encode("utf-8")).hexdigest()[:16]
        return self.cache_dir / f"{path.stem}_{key}.npy"

    def _preload_cache(self) -> None:
        print(f"Preloading cached samples into memory ({len(self.samples)} total)...")

        def load_sample(meta: SampleMeta):
            cache_path = self._cache_path_for(meta.path)
            if cache_path.exists():
                try:
                    frames = np.load(cache_path, allow_pickle=True)
                    return str(meta.path), frames
                except Exception as exc:
                    print(f"[WARN] Failed to load cache {cache_path}: {exc}")
            return None

        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = [executor.submit(load_sample, meta) for meta in self.samples]
            for future in as_completed(futures):
                result = future.result()
                if result is not None:
                    path_str, frames = result
                    with self._cache_lock:
                        self._memory_cache[path_str] = frames

        print(f"Preload complete: {len(self._memory_cache)} samples in memory cache.")

    @staticmethod
    def _safe_standardize(frames: np.ndarray, clip_abs: float = 1e6) -> np.ndarray:
        """Clip then z-score in float64 to avoid overflow in std computation."""
        frames_64 = frames.astype(np.float64, copy=False)
        frames_64 = np.nan_to_num(frames_64, nan=0.0, posinf=0.0, neginf=0.0)
        frames_64 = np.clip(frames_64, -clip_abs, clip_abs)
        mean = float(np.mean(frames_64))
        std = float(np.std(frames_64))
        if std > 1e-6:
            frames_64 = (frames_64 - mean) / std
        else:
            frames_64 = frames_64 - mean
        frames = frames_64.astype(np.float32)
        return np.nan_to_num(frames, nan=0.0, posinf=0.0, neginf=0.0)

    def clear_memory_cache(self) -> None:
        with self._cache_lock:
            self._memory_cache.clear()
        print("Memory cache cleared.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        meta = self.samples[idx]
        path_str = str(meta.path)
        use_memory_cache = self.enable_memory_cache and (get_worker_info() is None)
        source = "unknown"

        if use_memory_cache and path_str in self._memory_cache:
            frames = self._memory_cache[path_str].copy()
            source = "memory_cache_hit"
            self._cache_stats["memory_hit"] += 1
        elif self.cache_dir is not None:
            cache_path = self._cache_path_for(meta.path)
            if cache_path.exists():
                frames = np.load(cache_path, allow_pickle=True)
                source = "disk_cache_hit"
                self._cache_stats["disk_hit"] += 1
                if use_memory_cache:
                    with self._cache_lock:
                        if path_str not in self._memory_cache:
                            self._memory_cache[path_str] = frames.copy()
            else:
                source = "disk_cache_miss_read_xlsx"
                self._cache_stats["disk_miss_xlsx"] += 1
                self._cache_stats["raw_xlsx_read"] += 1
                frames = self._preprocess(self._read_excel_optimized(meta.path), sample_path=meta.path)
                tmp = cache_path.with_suffix(".tmp.npy")
                np.save(tmp, frames)
                tmp.replace(cache_path)
                if use_memory_cache:
                    with self._cache_lock:
                        self._memory_cache[path_str] = frames.copy()
        else:
            source = "no_cache_dir_read_xlsx"
            self._cache_stats["raw_xlsx_read"] += 1
            frames = self._preprocess(self._read_excel_optimized(meta.path), sample_path=meta.path)

        self._cache_stats["total"] += 1
        if self.cache_trace:
            if self._trace_printed < self.cache_trace_limit:
                print(f"[Dataset Trace] idx={idx} source={source} sample={meta.path}")
                self._trace_printed += 1
            if self._cache_stats["total"] % max(1, self.cache_trace_every) == 0:
                print(
                    "[Dataset Trace Summary] "
                    f"total={self._cache_stats['total']} "
                    f"memory_hit={self._cache_stats['memory_hit']} "
                    f"disk_hit={self._cache_stats['disk_hit']} "
                    f"disk_miss_xlsx={self._cache_stats['disk_miss_xlsx']} "
                    f"raw_xlsx_read={self._cache_stats['raw_xlsx_read']}"
                )

        frames = self._safe_standardize(frames)
        label = LABEL_MAP[meta.gesture]
        return torch.from_numpy(frames).float(), label


# Backward-compatible alias.
OptimizedTacActDataset = TacActDataset


__all__ = [
    "FILENAME_RE",
    "LABEL_MAP",
    "SampleMeta",
    "TacActDataset",
    "OptimizedTacActDataset",
]
