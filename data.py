from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

FILENAME_RE = re.compile(r"^(?P<subject>\d+)_(?P<gesture>\d+)_(?P<variant>[A-Za-z]+)_(?P<repeat>\d+)\.xlsx$")


@dataclass
class SampleMeta:
    path: Path
    subject: int
    gesture: int
    variant: str
    repeat: int


class TacActDataset(Dataset):
    def __init__(
        self,
        root_dir: Path,
        n_frames: int = 80,
        threshold: float = 20.0,
        clip_mode: str = "center",
        cache_dir: Optional[Path] = None,
    ) -> None:
        self.root_dir = root_dir
        self.n_frames = n_frames
        self.threshold = threshold
        self.clip_mode = clip_mode
        self.cache_dir = cache_dir
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.samples = self._discover_samples(root_dir)
        if not self.samples:
            raise RuntimeError(f"No matching .xlsx files found under {root_dir}")

    @staticmethod
    def _discover_samples(root_dir: Path) -> List[SampleMeta]:
        out: List[SampleMeta] = []
        for p in sorted(root_dir.rglob("*.xlsx")):
            m = FILENAME_RE.match(p.name)
            if not m:
                continue
            gesture = int(m.group("gesture"))
            if not (1 <= gesture <= 12):
                continue
            out.append(
                SampleMeta(
                    path=p,
                    subject=int(m.group("subject")),
                    gesture=gesture,
                    variant=m.group("variant"),
                    repeat=int(m.group("repeat")),
                )
            )
        return out

    @staticmethod
    def _read_excel_as_frames(path: Path) -> np.ndarray:
        df = pd.read_excel(path, header=None)
        arr = df.to_numpy(dtype=np.float32)
        if arr.ndim != 2:
            raise ValueError(f"Unexpected excel dimensionality in {path}")

        valid_rows = ~np.all(np.isnan(arr), axis=1)
        valid_cols = ~np.all(np.isnan(arr), axis=0)
        arr = arr[valid_rows][:, valid_cols]
        arr = np.nan_to_num(arr, nan=0.0)
        if arr.size == 0:
            raise ValueError(f"Empty numeric matrix in {path}")

        if arr.shape[1] == 1024:
            frames_flat = arr
        elif arr.shape[0] == 1024:
            frames_flat = arr.T
        else:
            flat = arr.reshape(-1)
            usable = (flat.size // 1024) * 1024
            if usable == 0:
                raise ValueError(f"Cannot infer 32x32 frames from {path}, got {arr.shape}")
            frames_flat = flat[:usable].reshape(-1, 1024)

        return frames_flat.reshape(-1, 32, 32).astype(np.float32)

    def _preprocess(self, frames: np.ndarray) -> np.ndarray:
        frames = frames - frames[0:1]
        # 加上 np.abs()，确保负向剧烈变化也能被捕捉
        active = np.any(np.abs(frames) > self.threshold, axis=(1, 2))
        if np.any(active):
            start = int(np.argmax(active))
            end = len(active) - int(np.argmax(active[::-1]))
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
            else:
                st = (t - self.n_frames) // 2
                frames = frames[st : st + self.n_frames]
        return frames.astype(np.float32)

    def _cache_path_for(self, path: Path) -> Path:
        key = hashlib.sha1(str(path).encode("utf-8")).hexdigest()[:16]
        return self.cache_dir / f"{path.stem}_{key}.npy"

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        meta = self.samples[idx]
        if self.cache_dir is not None:
            cp = self._cache_path_for(meta.path)
            if cp.exists():
                frames = np.load(cp)
            else:
                frames = self._preprocess(self._read_excel_as_frames(meta.path))
                tmp = cp.with_suffix(".tmp.npy")
                np.save(tmp, frames)
                tmp.replace(cp)
        else:
            frames = self._preprocess(self._read_excel_as_frames(meta.path))

        return torch.from_numpy(frames), meta.gesture - 1