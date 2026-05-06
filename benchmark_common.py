from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from tacact.data import OptimizedTacActDataset
from tacact.utils import choose_device_or_exit


DEFAULT_N_FRAMES = 80
DEFAULT_THRESHOLD_METHOD = "mean_std"
DEFAULT_THRESHOLD_K = 3.0
DEFAULT_BACKGROUND_FRAMES = 5


def get_device() -> torch.device:
    return choose_device_or_exit()


def create_optimized_dataset(
    data_root: Path,
    *,
    n_frames: int = DEFAULT_N_FRAMES,
    threshold: Optional[float] = None,
    threshold_method: str = DEFAULT_THRESHOLD_METHOD,
    threshold_k: float = DEFAULT_THRESHOLD_K,
    background_frames: int = DEFAULT_BACKGROUND_FRAMES,
    center_weight_sigma: Optional[float] = None,
    clip_mode: str = "weighted_center",
    cache_dir: Optional[Path] = None,
    preload_cache: bool = True,
    segmentation_log: bool = False,
    cache_trace: bool = False,
) -> OptimizedTacActDataset:
    return OptimizedTacActDataset(
        data_root,
        n_frames=n_frames,
        threshold=threshold,
        threshold_method=threshold_method,
        threshold_k=threshold_k,
        background_frames=background_frames,
        center_weight_sigma=center_weight_sigma,
        clip_mode=clip_mode,
        cache_dir=cache_dir,
        preload_cache=preload_cache,
        segmentation_log=segmentation_log,
        cache_trace=cache_trace,
    )


def warmup_cache(
    dataset: Dataset,
    *,
    batch_size: int = 128,
    num_workers: int = 0,
    prefetch_factor: int = 2,
    max_batches: Optional[int] = None,
    shuffle: bool = False,
    pin_memory: bool = False,
    use_tqdm: bool = False,
    tqdm_desc: str = "Building Cache",
) -> None:
    """
    触发 dataset 的 __getitem__ 以构建/验证缓存。

    max_batches=None 表示遍历完整个 dataset；
    max_batches=1 表示只跑一个 batch（用于轻量预热）。
    """
    loader_kwargs = dict(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = max(1, int(prefetch_factor))
    loader = DataLoader(**loader_kwargs)

    it: Optional[object] = None
    if use_tqdm:
        from tqdm import tqdm  # lazy import

        it = tqdm(loader, desc=tqdm_desc)
    else:
        it = loader

    batches_done = 0
    for _ in it:
        batches_done += 1
        if max_batches is not None and batches_done >= max_batches:
            break


def _subjects_from_dataset(dataset: Dataset) -> np.ndarray:
    # 两个 tacact dataset 都暴露 samples，并且 meta 有 subject 字段
    return np.array([meta.subject for meta in getattr(dataset, "samples")])


def split_indices_3way(
    dataset: Dataset,
    *,
    split_mode: str,
    seed: int,
    train_ratio: float,
    val_ratio: float,
) -> Tuple[List[int], List[int], List[int]]:
    """
    返回 train/val/test 的索引列表。
    split_mode: "random" | "subject"
    """
    if split_mode == "random":
        rng = np.random.default_rng(seed)
        idxs = rng.permutation(len(dataset)).tolist()
        n = len(idxs)
        n_train = max(1, int(n * train_ratio))
        n_val = max(1, int(n * val_ratio))
        n_test = n - n_train - n_val
        if n_test <= 0:
            raise ValueError(
                "Not enough samples to split into train/val/test. Need more samples or adjust ratios."
            )
        train_idx = idxs[:n_train]
        val_idx = idxs[n_train : n_train + n_val]
        test_idx = idxs[n_train + n_val :]
        return train_idx, val_idx, test_idx

    subjects = _subjects_from_dataset(dataset)
    unique_subjects = np.unique(subjects)
    rng = np.random.default_rng(seed)
    rng.shuffle(unique_subjects)

    n_sub = len(unique_subjects)
    train_sub_n = max(1, int(n_sub * train_ratio))
    val_sub_n = max(1, int(n_sub * val_ratio))
    train_subs = unique_subjects[:train_sub_n]
    val_subs = unique_subjects[train_sub_n : train_sub_n + val_sub_n]
    test_subs = unique_subjects[train_sub_n + val_sub_n :]
    if len(test_subs) == 0:
        raise ValueError("Not enough subjects to split into train/val/test.")

    train_set = set(train_subs.tolist())
    val_set = set(val_subs.tolist())
    test_set = set(test_subs.tolist())
    train_idx = [i for i, s in enumerate(subjects) if s in train_set]
    val_idx = [i for i, s in enumerate(subjects) if s in val_set]
    test_idx = [i for i, s in enumerate(subjects) if s in test_set]
    return train_idx, val_idx, test_idx


def split_indices_train_val(
    dataset: Dataset,
    *,
    split_mode: str,
    seed: int,
    train_ratio: float,
) -> Tuple[List[int], List[int]]:
    """
    返回 train/val 的索引列表（用于只需要 train+val 的实验，如 CNNLSTM 搜索脚本）。
    split_mode: "random" | "subject"
    """
    if split_mode == "random":
        rng = np.random.default_rng(seed)
        idxs = rng.permutation(len(dataset)).tolist()
        n_train = max(1, int(len(idxs) * train_ratio))
        return idxs[:n_train], idxs[n_train:]

    subjects = _subjects_from_dataset(dataset)
    uniq = np.unique(subjects)
    rng = np.random.default_rng(seed)
    rng.shuffle(uniq)
    n = len(uniq)
    n_train = max(1, int(n * train_ratio))

    train_set = set(uniq[:n_train].tolist())
    val_set = set(uniq[n_train:].tolist())
    train_idx = [i for i, s in enumerate(subjects) if s in train_set]
    val_idx = [i for i, s in enumerate(subjects) if s in val_set]
    return train_idx, val_idx


def build_split_audit(
    dataset: Dataset,
    *,
    split_mode: str,
    seed: int,
    train_ratio: float,
    val_ratio: float,
    train_idx: List[int],
    val_idx: List[int],
    test_idx: List[int],
) -> Dict[str, Any]:
    samples = getattr(dataset, "samples", None)
    if samples is None:
        raise ValueError("Dataset does not expose `samples`; cannot build split audit.")

    def _stats(idxs: List[int]) -> Dict[str, Any]:
        subjects = [int(samples[i].subject) for i in idxs]
        gestures = [int(samples[i].gesture) for i in idxs]
        unique_keys = {
            (
                int(samples[i].subject),
                int(samples[i].gesture),
                str(samples[i].variant),
                int(samples[i].repeat),
                str(samples[i].path),
            )
            for i in idxs
        }
        return {
            "num_samples": int(len(idxs)),
            "subjects": sorted(set(subjects)),
            "num_subjects": int(len(set(subjects))),
            "subject_counts": {str(k): int(v) for k, v in sorted(Counter(subjects).items())},
            "class_counts": {str(k): int(v) for k, v in sorted(Counter(gestures).items())},
            "missing_gestures": [int(g) for g in range(1, 13) if g not in set(gestures)],
            "_index_set": set(idxs),
            "_key_set": unique_keys,
        }

    train_stats = _stats(train_idx)
    val_stats = _stats(val_idx)
    test_stats = _stats(test_idx)

    def _overlap(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "subject_overlap": sorted(set(a["subjects"]) & set(b["subjects"])),
            "index_overlap_count": int(len(a["_index_set"] & b["_index_set"])),
            "sample_key_overlap_count": int(len(a["_key_set"] & b["_key_set"])),
        }

    overlap = {
        "train_val": _overlap(train_stats, val_stats),
        "train_test": _overlap(train_stats, test_stats),
        "val_test": _overlap(val_stats, test_stats),
    }

    # Internal helper sets are not JSON serializable.
    for block in (train_stats, val_stats, test_stats):
        block.pop("_index_set", None)
        block.pop("_key_set", None)

    return {
        "split_mode": str(split_mode),
        "seed": int(seed),
        "train_ratio": float(train_ratio),
        "val_ratio": float(val_ratio),
        "dataset_size": int(len(dataset)),
        "splits": {
            "train": train_stats,
            "val": val_stats,
            "test": test_stats,
        },
        "overlap_check": overlap,
    }


def save_split_audit(save_path: Path, payload: Dict[str, Any]) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def make_three_loaders(
    *,
    train_set: Dataset,
    val_set: Dataset,
    test_set: Dataset,
    batch_size: int,
    num_workers: int,
    pin_memory: bool = True,
    prefetch_factor: int = 2,
    persistent_workers: Optional[bool] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    if persistent_workers is None:
        persistent_workers = num_workers > 0

    shared_kwargs = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    if num_workers > 0:
        shared_kwargs["persistent_workers"] = bool(persistent_workers)
        shared_kwargs["prefetch_factor"] = max(1, int(prefetch_factor))

    train_loader = DataLoader(
        train_set,
        shuffle=True,
        **shared_kwargs,
    )
    val_loader = DataLoader(
        val_set,
        shuffle=False,
        **shared_kwargs,
    )
    test_loader = DataLoader(
        test_set,
        shuffle=False,
        **shared_kwargs,
    )
    return train_loader, val_loader, test_loader
