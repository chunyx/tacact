#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import re
import sys
from pathlib import Path
from typing import Dict, Optional

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tacact.data import FILENAME_RE


def _build_cache_gesture_map(data_root: Path) -> Dict[str, int]:
    """
    Build a reliable map: cache filename -> gesture id, based on dataset metadata and cache key rule.

    Cache key follows data.py logic:
    key = sha1(str(sample_path).encode("utf-8")).hexdigest()[:16]
    cache_name = f"{sample_path.stem}_{key}.npy"
    """
    mapping: Dict[str, int] = {}
    for sample_path in sorted(data_root.rglob("*.xlsx")):
        match = FILENAME_RE.match(sample_path.name)
        if not match:
            continue
        gesture = int(match.group("gesture"))
        key = hashlib.sha1(str(sample_path).encode("utf-8")).hexdigest()[:16]
        cache_name = f"{sample_path.stem}_{key}.npy"
        mapping[cache_name] = gesture
    return mapping


def _infer_gesture_from_filename(cache_filename: str) -> Optional[int]:
    # Fallback parser for names like: subject_gesture_variant_repeat_<hash>.npy
    # Example: 12_3_A_1_ab12cd34ef56aa78.npy
    pattern = re.compile(r"^(?P<subject>\d+)_(?P<gesture>\d+)_(?P<variant>[A-Za-z]+)_(?P<repeat>\d+)_[0-9a-f]{16}\.npy$")
    match = pattern.match(cache_filename)
    if not match:
        return None
    try:
        gesture = int(match.group("gesture"))
    except ValueError:
        return None
    return gesture if 1 <= gesture <= 12 else None


def _init_stats() -> Dict[int, Dict[str, int]]:
    stats: Dict[int, Dict[str, int]] = {}
    for g in range(1, 13):
        stats[g] = {"valid": 0, "bad": 0, "total": 0}
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate TacAct .npy cache files.")
    parser.add_argument("--cache-dir", type=Path, default=Path(".cache_tacact_n80_weighted"))
    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help="Dataset root used for reliable cache->gesture mapping via source metadata.",
    )
    parser.add_argument(
        "--allow-filename-fallback",
        action="store_true",
        help="Allow filename-based gesture parsing only for cache files not mappable from dataset metadata.",
    )
    parser.add_argument(
        "--delete-bad",
        action="store_true",
        help="Delete corrupted cache files after reporting them.",
    )
    args = parser.parse_args()

    cache_dir = args.cache_dir
    print("=" * 80)
    print("TacAct Cache Checker")
    print(f"Cache dir: {cache_dir.resolve()}")
    print(f"Delete bad: {args.delete_bad}")
    print("=" * 80)

    if not cache_dir.exists():
        print(f"[ERROR] Cache directory does not exist: {cache_dir}")
        return

    cache_files = sorted(cache_dir.glob("*.npy"))
    if not cache_files:
        print("[INFO] No .npy cache files found.")
        return

    if not args.data_root.exists():
        print(f"[ERROR] data_root does not exist: {args.data_root}")
        return
    gesture_map = _build_cache_gesture_map(args.data_root)
    print(f"[INFO] Built metadata mapping from data_root: {args.data_root} (entries={len(gesture_map)})")

    stats = _init_stats()
    unknown = {"valid": 0, "bad": 0, "total": 0}
    overall = {"valid": 0, "bad": 0, "total": 0}

    print(f"[INFO] Checking {len(cache_files)} cache files...")
    for cp in cache_files:
        gesture = gesture_map.get(cp.name)
        if gesture is None and args.allow_filename_fallback:
            gesture = _infer_gesture_from_filename(cp.name)

        is_known = gesture is not None and 1 <= gesture <= 12
        bucket = stats[gesture] if is_known else unknown

        bucket["total"] += 1
        overall["total"] += 1

        try:
            _ = np.load(cp, allow_pickle=True)
            bucket["valid"] += 1
            overall["valid"] += 1
        except Exception as exc:
            bucket["bad"] += 1
            overall["bad"] += 1
            print(f"[BAD] {cp.resolve()} | error={repr(exc)}")
            if args.delete_bad:
                try:
                    cp.unlink()
                    print(f"      -> deleted")
                except Exception as del_exc:
                    print(f"      -> delete failed: {repr(del_exc)}")

    print("\n" + "=" * 80)
    print("Per-gesture summary")
    print("=" * 80)
    for g in range(1, 13):
        s = stats[g]
        print(f"Gesture {g:2d}: valid={s['valid']:4d}, bad={s['bad']:4d}, total={s['total']:4d}")
    print(f"Gesture ?? : valid={unknown['valid']:4d}, bad={unknown['bad']:4d}, total={unknown['total']:4d} (unmapped)")

    print("\n" + "=" * 80)
    print(
        f"Overall: valid={overall['valid']}, bad={overall['bad']}, total={overall['total']}"
    )
    print("=" * 80)


if __name__ == "__main__":
    main()
