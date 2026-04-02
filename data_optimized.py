from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, Dict
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import torch
from torch.utils.data import get_worker_info

from .data import SampleMeta, TacActDataset


class OptimizedTacActDataset(TacActDataset):
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
        super().__init__(
            root_dir=root_dir,
            n_frames=n_frames,
            threshold=threshold,
            clip_mode=clip_mode,
            cache_dir=cache_dir,
        )
        self.preload_cache = preload_cache
        self.num_workers = num_workers

        # 内存缓存
        self._memory_cache: Dict[str, np.ndarray] = {}
        self._cache_lock = threading.Lock()
        
        # 预加载缓存到内存
        if self.preload_cache and self.cache_dir is not None:
            self._preload_cache()

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
        """优化的Excel读取，直接处理32列格式"""
        try:
            # 使用更快的Excel读取参数
            df = pd.read_excel(
                path, 
                header=None, 
                engine='openpyxl',  # 通常比xlrd更快
                dtype=np.float32   # 直接指定类型避免转换
            )
            arr = df.to_numpy(dtype=np.float32)
        except Exception as e:
            # 如果openpyxl失败，尝试其他引擎
            df = pd.read_excel(path, header=None, dtype=np.float32)
            arr = df.to_numpy(dtype=np.float32)
        
        if arr.ndim != 2:
            raise ValueError(f"Unexpected excel dimensionality in {path}")

        # 快速处理NaN值
        if np.any(np.isnan(arr)):
            # 使用更快的NaN处理
            row_mask = ~np.all(np.isnan(arr), axis=1)
            col_mask = ~np.all(np.isnan(arr), axis=0)
            arr = arr[row_mask][:, col_mask]
            arr = np.where(np.isnan(arr), 0.0, arr)
        
        if arr.size == 0:
            raise ValueError(f"Empty numeric matrix in {path}")

        # 智能reshape：优先假设是32列格式
        if arr.shape[1] == 32:
            # 直接是32列，行数应该是32的倍数
            n_rows = arr.shape[0]
            if n_rows % 32 != 0:
                # 截断到32的倍数
                n_frames = n_rows // 32
                arr = arr[:n_frames * 32]
            frames = arr.reshape(-1, 32, 32)
        elif arr.shape[0] == 32:
            # 32行，需要转置
            n_cols = arr.shape[1]
            if n_cols % 32 != 0:
                n_frames = n_cols // 32
                arr = arr[:, :n_frames * 32]
            frames = arr.T.reshape(-1, 32, 32)
        elif arr.shape[1] == 1024:
            # 展平的格式
            frames = arr.reshape(-1, 32, 32)
        elif arr.shape[0] == 1024:
            frames = arr.T.reshape(-1, 32, 32)
        else:
            # 兜底逻辑：尝试找到最合适的reshape方式
            total_elements = arr.size
            n_frames = total_elements // (32 * 32)
            if n_frames == 0:
                raise ValueError(f"Cannot infer 32x32 frames from {path}, got {arr.shape}")
            
            usable = n_frames * 32 * 32
            if arr.shape[1] * arr.shape[0] > usable:
                # 截断到合适大小
                if arr.shape[1] >= 1024:
                    frames_flat = arr[:n_frames, :1024].reshape(-1, 32, 32)
                else:
                    flat = arr.ravel()[:usable]
                    frames = flat.reshape(-1, 32, 32)
            else:
                frames = arr.reshape(-1, 32, 32)

        return frames.astype(np.float32)

    def _preprocess_optimized(self, frames: np.ndarray) -> np.ndarray:
        """优化的预处理，减少内存拷贝"""
        # 原地操作减少内存分配
        frames -= frames[0:1]
        
        # 使用向量化操作
        frame_abs = np.abs(frames)
        active = np.any(frame_abs > self.threshold, axis=(1, 2))
        
        if np.any(active):
            active_indices = np.where(active)[0]
            start = active_indices[0]
            end = active_indices[-1] + 1
            frames = frames[start:end]
        else:
            frames = frames[:1]

        t = frames.shape[0]
        if t < self.n_frames:
            # 预分配填充数组
            pad = np.zeros((self.n_frames - t, 32, 32), dtype=np.float32)
            frames = np.concatenate([frames, pad], axis=0)
        elif t > self.n_frames:
            if self.clip_mode == "front":
                frames = frames[:self.n_frames]
            elif self.clip_mode in {"weighted", "weighted_center", "center_weighted"}:
                sample_idx = self._center_weighted_indices(t, self.n_frames, gamma=0.65)
                frames = frames[sample_idx]
            else:
                st = (t - self.n_frames) // 2
                frames = frames[st:st + self.n_frames]
        
        return frames

    def _preload_cache(self):
        """预加载所有缓存文件到内存"""
        print(f"预加载缓存到内存，共 {len(self.samples)} 个样本...")
        
        def load_sample(meta: SampleMeta):
            cache_path = self._cache_path_for(meta.path)
            if cache_path.exists():
                try:
                    frames = np.load(cache_path, allow_pickle=True)
                    return str(meta.path), frames
                except Exception as e:
                    print(f"警告：无法加载缓存 {cache_path}: {e}")
            return None
        
        # 并行加载
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = [executor.submit(load_sample, meta) for meta in self.samples]
            
            for future in as_completed(futures):
                result = future.result()
                if result is not None:
                    path_str, frames = result
                    with self._cache_lock:
                        self._memory_cache[path_str] = frames
        
        print(f"预加载完成，缓存了 {len(self._memory_cache)} 个样本")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        meta = self.samples[idx]
        path_str = str(meta.path)

        use_memory_cache = get_worker_info() is None

        # 首先检查内存缓存
        if use_memory_cache and path_str in self._memory_cache:
            frames = self._memory_cache[path_str].copy()  # 返回副本避免修改缓存
        elif self.cache_dir is not None:
            cp = self._cache_path_for(meta.path)
            if cp.exists():
                frames = np.load(cp, allow_pickle=True)
                if use_memory_cache:
                    with self._cache_lock:
                        if path_str not in self._memory_cache:
                            self._memory_cache[path_str] = frames.copy()
            else:
                # 处理并缓存
                frames = self._preprocess_optimized(self._read_excel_optimized(meta.path))
                tmp = cp.with_suffix(".tmp.npy")
                np.save(tmp, frames)
                tmp.replace(cp)

                if use_memory_cache:
                    with self._cache_lock:
                        self._memory_cache[path_str] = frames.copy()
        else:
            frames = self._preprocess_optimized(self._read_excel_optimized(meta.path))

        # ==================== 新增：数据标准化与安全兜底 (双精度防溢出) ====================
        # 1. 临时转为 float64 (双精度)，防止计算标准差平方时 float32 溢出
        frames_64 = frames.astype(np.float64)
        frames_64 = np.nan_to_num(frames_64, nan=0.0, posinf=0.0, neginf=0.0)

        # 2. Z-score 标准化 (减去均值，除以标准差)，把数值强行拉回到 0 附近
        f_mean = np.mean(frames_64)
        f_std = np.std(frames_64)
        if f_std > 1e-6:
            frames_64 = (frames_64 - f_mean) / f_std
        else:
            frames_64 = frames_64 - f_mean  # 如果全是一模一样的值，只减均值防除零错误

        # 3. 转回 float32 并做最后一次防 NaN 兜底
        frames = frames_64.astype(np.float32)
        frames = np.nan_to_num(frames, nan=0.0, posinf=0.0, neginf=0.0)
        # ====================================================================

        return torch.from_numpy(frames).float(), meta.gesture - 1

    def clear_memory_cache(self):
        """清空内存缓存"""
        with self._cache_lock:
            self._memory_cache.clear()
        print("内存缓存已清空")


# 为了向后兼容，保留原始类名的别名
TacActDataset = OptimizedTacActDataset
