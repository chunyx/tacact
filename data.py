from __future__ import annotations

import hashlib
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
        threshold: float = 20.0,
        clip_mode: str = "front",
        cache_dir: Optional[Path] = None,
        preload_cache: bool = True,
        num_workers: int = 4,
    ) -> None:
        self.root_dir = root_dir
        self.n_frames = n_frames
        self.threshold = threshold
        self.clip_mode = clip_mode
        self.cache_dir = cache_dir
        self.preload_cache = preload_cache
        self.num_workers = num_workers

        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.samples = self._discover_samples(root_dir)
        if not self.samples:
            raise RuntimeError(f"No matching .xlsx files found under {root_dir}")

        self._memory_cache: Dict[str, np.ndarray] = {}
        self._cache_lock = threading.Lock()

        if self.preload_cache and self.cache_dir is not None:
            self._preload_cache()

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
    def _center_weighted_indices(length: int, target_len: int, gamma: float = 0.65) -> np.ndarray:
        if length <= target_len:
            return np.arange(length, dtype=np.int64)
        u = np.linspace(0.0, 1.0, target_len)
        left = 0.5 * np.power(2.0 * u, gamma)
        right = 1.0 - 0.5 * np.power(2.0 * (1.0 - u), gamma)
        warped = np.where(u <= 0.5, left, right)
        idx = np.clip(np.round(warped * (length - 1)).astype(np.int64), 0, length - 1)
        idx[0] = 0
        idx[-1] = length - 1
        return idx

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

    def _preprocess(self, frames: np.ndarray) -> np.ndarray:
        frames = frames.copy()
        frames -= frames[0:1]
        active = np.any(np.abs(frames) > self.threshold, axis=(1, 2))

        if np.any(active):
            active_indices = np.where(active)[0]
            start = int(active_indices[0])
            end = int(active_indices[-1]) + 1
            frames = frames[start:end]
        else:
            frames = frames[:1]

        t = frames.shape[0]
        if t < self.n_frames:
            pad = np.zeros((self.n_frames - t, 32, 32), dtype=np.float32)
            frames = np.concatenate([frames, pad], axis=0)
        elif t > self.n_frames:
            if self.clip_mode == "front":
                frames = frames[: self.n_frames]
            elif self.clip_mode in {"weighted", "weighted_center", "center_weighted"}:
                sample_idx = self._center_weighted_indices(t, self.n_frames, gamma=0.65)
                frames = frames[sample_idx]
            else:
                st = (t - self.n_frames) // 2
                frames = frames[st : st + self.n_frames]

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
        use_memory_cache = get_worker_info() is None

        if use_memory_cache and path_str in self._memory_cache:
            frames = self._memory_cache[path_str].copy()
        elif self.cache_dir is not None:
            cache_path = self._cache_path_for(meta.path)
            if cache_path.exists():
                frames = np.load(cache_path, allow_pickle=True)
                if use_memory_cache:
                    with self._cache_lock:
                        if path_str not in self._memory_cache:
                            self._memory_cache[path_str] = frames.copy()
            else:
                frames = self._preprocess(self._read_excel_optimized(meta.path))
                tmp = cache_path.with_suffix(".tmp.npy")
                np.save(tmp, frames)
                tmp.replace(cache_path)
                if use_memory_cache:
                    with self._cache_lock:
                        self._memory_cache[path_str] = frames.copy()
        else:
            frames = self._preprocess(self._read_excel_optimized(meta.path))

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
