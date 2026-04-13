#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import os
import random
import socket
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tacact.benchmark_common import create_optimized_dataset, get_device, split_indices_3way, warmup_cache
from tacact.models import ModelFactory
from tacact.utils import count_parameters, parse_model_list, per_class_prf, set_seed


DEEP_SPACE: Dict[str, Dict[str, List[Any]]] = {
    "lenet": {
        "lr": [3e-4, 1e-3, 3e-3],
        "weight_decay": [1e-5, 1e-4],
        "batch_size": [16, 32],
    },
    "alexnet": {
        "lr": [1e-4, 3e-4, 1e-3],
        "weight_decay": [1e-5, 1e-4],
        "batch_size": [8, 16],
    },
    "resnet18": {
        "lr": [3e-4, 1e-3],
        "weight_decay": [1e-5, 1e-4],
        "batch_size": [8, 16],
    },
    "mobilenet_v2": {
        "lr": [3e-4, 1e-3],
        "weight_decay": [1e-5, 1e-4, 1e-3],
        "batch_size": [8, 16],
    },
    "efficientnet_b0": {
        "lr": [3e-4, 1e-3],
        "weight_decay": [1e-5, 1e-4, 5e-4],
        "batch_size": [8, 16],
    },
    "lstm": {
        "input_proj_dim": [256, 512],
        "hidden_size": [128, 256],
        "num_layers": [1, 2],
        "dropout": [0.3, 0.5],
        "use_last_only": [False],
        "lr": [3e-4, 1e-3],
        "weight_decay": [1e-5, 1e-4],
        "batch_size": [8, 16],
    },
    "gru": {
        "input_proj_dim": [256, 512],
        "hidden_size": [128, 256],
        "num_layers": [1, 2],
        "dropout": [0.3, 0.5],
        "use_last_only": [False],
        "lr": [3e-4, 1e-3],
        "weight_decay": [1e-5, 1e-4],
        "batch_size": [8, 16],
    },
    "cnn_lstm": {
        "lstm_hidden": [128, 256],
        "lr": [1e-4, 3e-4, 1e-3],
        "weight_decay": [1e-5, 1e-4],
        "batch_size": [4, 8],
    },
    "tcn": {
        "num_channels": [256, 512],
        "dropout": [0.0, 0.1],
        "lr": [1e-4, 3e-4, 5e-4],
        "weight_decay": [1e-5, 1e-4],
        "batch_size": [4, 8],
    },
    "vit": {
        "dim": [192, 256],
        "depth": [3, 4],
        "heads": [4, 8],
        "patch_size": [16],
        "dropout": [0.1, 0.25],
        "lr": [5e-5, 1e-4],
        "weight_decay": [0.01, 0.05],
        "batch_size": [2, 4],
    },
}

# Prefer refined phase-0 informed search spaces when available.
try:
    from refined_search_space import DEEP_SPACE_REFINED  # type: ignore

    DEEP_SPACE = DEEP_SPACE_REFINED
except Exception:
    try:
        from benchmark_data_loading.refined_search_space import DEEP_SPACE_REFINED  # type: ignore

        DEEP_SPACE = DEEP_SPACE_REFINED
    except Exception:
        pass


MODEL_KWARGS_KEYS = {
    "dim",
    "depth",
    "heads",
    "patch_size",
    "dropout",
    "num_channels",
    "lstm_hidden",
    "hidden_size",
    "num_layers",
    "input_proj_dim",
    "use_last_only",
}


PHASE_DEFAULTS = {
    1: {"epochs": 8, "use_early_stopping": False},
    2: {"epochs": 20, "use_early_stopping": False},
    3: {"epochs": 30, "use_early_stopping": False},
}


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _model_progress_path(output_root: Path, model_name: str) -> Path:
    return output_root / "model_progress" / f"{model_name}.json"


def _save_model_progress(output_root: Path, model_name: str, payload: Dict[str, Any]) -> None:
    path = _model_progress_path(output_root, model_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _load_model_progress(output_root: Path, model_name: str) -> Optional[Dict[str, Any]]:
    path = _model_progress_path(output_root, model_name)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _is_trial_csv_complete(
    csv_path: Path,
    *,
    expected_rows: int,
    trial_id_col: str = "trial_id",
    require_ok_status: bool = False,
) -> bool:
    if expected_rows <= 0:
        return True
    if not csv_path.exists():
        return False
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return False
    if len(df) < expected_rows:
        return False
    if trial_id_col in df.columns:
        if int(df[trial_id_col].nunique()) < expected_rows:
            return False
    if require_ok_status and "status" in df.columns:
        ok = int((df["status"].astype(str) == "ok").sum())
        if ok < expected_rows:
            return False
    return True


def _is_phase1_complete(output_root: Path, model_name: str, expected_trials: int) -> bool:
    return _is_trial_csv_complete(
        output_root / "phase1" / model_name / "phase1_results.csv",
        expected_rows=int(expected_trials),
        trial_id_col="trial_id",
        require_ok_status=False,
    )


def _is_phase2_complete(output_root: Path, model_name: str, expected_trials: int) -> bool:
    return _is_trial_csv_complete(
        output_root / "phase2" / model_name / "phase2_results.csv",
        expected_rows=int(expected_trials),
        trial_id_col="trial_id",
        require_ok_status=False,
    )


def _is_phase3_complete(output_root: Path, model_name: str, expected_runs: int) -> bool:
    csv_path = output_root / "phase3" / model_name / "final_results.csv"
    if expected_runs <= 0:
        return False
    if not csv_path.exists():
        return False
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return False
    if len(df) < expected_runs:
        return False
    if "seed" in df.columns and int(df["seed"].nunique()) < expected_runs:
        return False
    return True


def _parse_gpu_ids(gpu_ids_arg: Optional[str], visible_gpu_count: int) -> List[int]:
    if visible_gpu_count <= 0:
        return []
    if gpu_ids_arg is None or not gpu_ids_arg.strip():
        return list(range(visible_gpu_count))
    ids: List[int] = []
    for token in gpu_ids_arg.split(","):
        token = token.strip()
        if not token:
            continue
        idx = int(token)
        if idx < 0 or idx >= visible_gpu_count:
            raise ValueError(
                f"--gpu_ids contains invalid local GPU id {idx}; visible local range is [0, {visible_gpu_count - 1}]"
            )
        ids.append(idx)
    if not ids:
        raise ValueError("--gpu_ids parsed to an empty list.")
    if len(set(ids)) != len(ids):
        raise ValueError("--gpu_ids contains duplicate local GPU ids.")
    return ids


def _get_visible_device_tokens(visible_gpu_count: int) -> List[str]:
    raw = os.environ.get("CUDA_VISIBLE_DEVICES")
    if raw is None or raw.strip() == "":
        return [str(i) for i in range(visible_gpu_count)]
    tokens = [x.strip() for x in raw.split(",") if x.strip()]
    if len(tokens) < visible_gpu_count:
        # fallback guard; should not happen in normal CUDA runtime behavior
        return [str(i) for i in range(visible_gpu_count)]
    return tokens[:visible_gpu_count]


def _build_worker_task(
    *,
    task_id: str,
    phase: int,
    model_name: str,
    trial_id: int,
    seed: int,
    config: Dict[str, Any],
    result_json_path: Path,
    history_csv_path: Path,
    progress_json_path: Optional[Path],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    payload = {
        "task_id": task_id,
        "phase": int(phase),
        "model_name": model_name,
        "trial_id": int(trial_id),
        "seed": int(seed),
        "config": sanitize_config(config),
        "result_json_path": str(result_json_path),
        "history_csv_path": str(history_csv_path),
        "data_cfg": {
            "data_root": str(args.data_root),
            "cache_dir": str(args.cache_dir),
            "split_mode": args.split_mode,
            "clip_mode": args.clip_mode,
            "split_seed": int(args.seed),
            "train_ratio": float(args.train_ratio),
            "val_ratio": float(args.val_ratio),
            "no_preload": bool(args.no_preload),
            "parallel_mode": bool(args.parallel),
            "cache_trace": bool(args.cache_trace),
            "num_workers": int(args.num_workers),
        },
    }
    if progress_json_path is not None:
        payload["progress_json_path"] = str(progress_json_path)
    return payload


def _run_tasks_with_gpu_scheduler(
    tasks: Sequence[Dict[str, Any]],
    *,
    gpu_ids: Sequence[int],
    max_workers: int,
    heartbeat_sec: int = 10,
    progress_desc: str = "HPO Trials",
    show_heartbeat: bool = True,
    gpu_dashboard: bool = False,
) -> None:
    if not tasks:
        return
    if not gpu_ids:
        raise RuntimeError("No GPU ids available for parallel scheduling.")
    n_workers = min(int(max_workers), len(gpu_ids))
    if n_workers <= 0:
        raise RuntimeError("max_workers resolved to 0.")

    visible_gpu_count = torch.cuda.device_count()
    visible_tokens = _get_visible_device_tokens(visible_gpu_count)
    local_to_token = {local_id: visible_tokens[local_id] for local_id in gpu_ids}
    script_path = Path(__file__).resolve()

    pending = list(tasks)
    free_gpus = list(gpu_ids[:n_workers])
    running: List[Dict[str, Any]] = []
    total = len(tasks)
    completed = 0
    last_heartbeat = time.time()
    pbar = None
    use_tty = bool(sys.stdout.isatty())
    use_dashboard = bool(gpu_dashboard and use_tty)
    use_tqdm = bool(use_tty and not use_dashboard)
    if use_tqdm:
        try:
            from tqdm import tqdm  # lazy import

            pbar = tqdm(total=total, desc=progress_desc, unit="trial")
        except Exception:
            pbar = None

    def _running_desc(items: Sequence[Dict[str, Any]]) -> str:
        if not items:
            return "idle"
        return ", ".join(
            [f"g{x['gpu_local_id']}:{x['task']['model_name']}/t{x['task']['trial_id']}" for x in items]
        )

    def _render_dashboard() -> None:
        if not use_dashboard:
            return
        running_map = {int(x["gpu_local_id"]): x for x in running}
        pending_preview = ", ".join([f"{t['model_name']}/t{t['trial_id']}" for t in pending[:8]])
        if len(pending) > 8:
            pending_preview += ", ..."
        lines: List[str] = []
        lines.append(
            f"Phase2 Global Progress: {completed}/{total} trials completed | "
            f"{len(running)} running | {len(pending)} pending"
        )
        lines.append("")
        for gid in gpu_ids[:n_workers]:
            item = running_map.get(int(gid))
            if item is None:
                lines.append(f"GPU {gid:<2} | IDLE")
                continue
            task = item["task"]
            prog = item.get("live_progress", {}) or {}
            epoch = int(prog.get("epoch", 0) or 0)
            total_epochs = int(
                prog.get("total_epochs", task.get("config", {}).get("epochs_budget", 0)) or 0
            )
            val_f1 = prog.get("val_f1", None)
            if isinstance(val_f1, (int, float)):
                f1_txt = f"{float(val_f1):.4f}"
            else:
                f1_txt = "-"
            lines.append(
                f"GPU {gid:<2} | {str(task['model_name']):<14} | trial {int(task['trial_id']):<2} | "
                f"epoch {epoch:>2}/{max(total_epochs, 0):<2} | val_f1={f1_txt}"
            )
        lines.append("")
        lines.append(f"Pending queue: {pending_preview if pending_preview else '-'}")
        sys.stdout.write("\x1b[H\x1b[J" + "\n".join(lines) + "\n")
        sys.stdout.flush()

    try:
        while pending or running:
            while pending and free_gpus:
                gpu_local_id = free_gpus.pop(0)
                task = pending.pop(0)
                task_file = Path(task["task_file"])

                env = os.environ.copy()
                env["CUDA_VISIBLE_DEVICES"] = str(local_to_token[gpu_local_id])
                env["HPO_ASSIGNED_GPU_LOCAL_ID"] = str(gpu_local_id)
                env["PYTHONUNBUFFERED"] = "1"

                cmd = [
                    sys.executable,
                    "-u",
                    str(script_path),
                    "--worker_mode",
                    "--task_file",
                    str(task_file),
                ]
                proc = subprocess.Popen(cmd, env=env)
                print(
                    f"[Scheduler] launch phase={task['phase']} model={task['model_name']} trial={task['trial_id']} "
                    f"seed={task['seed']} gpu_local_id={gpu_local_id} pid={proc.pid}",
                    flush=True,
                )
                running.append(
                    {
                        "proc": proc,
                        "gpu_local_id": gpu_local_id,
                        "task": task,
                        "start_time": time.time(),
                        "progress_path": Path(task["progress_json_path"]) if task.get("progress_json_path") else None,
                        "live_progress": {},
                    }
                )

            still_running: List[Dict[str, Any]] = []
            for item in running:
                progress_path = item.get("progress_path")
                if isinstance(progress_path, Path) and progress_path.exists():
                    try:
                        item["live_progress"] = json.loads(progress_path.read_text(encoding="utf-8"))
                    except Exception:
                        pass
                proc = item["proc"]
                ret = proc.poll()
                if ret is None:
                    still_running.append(item)
                    continue
                gpu_local_id = int(item["gpu_local_id"])
                task = item["task"]
                if ret != 0:
                    raise RuntimeError(
                        f"Worker failed (code={ret}) phase={task['phase']} model={task['model_name']} "
                        f"trial={task['trial_id']} seed={task['seed']} gpu_local_id={gpu_local_id}"
                    )
                completed += 1
                elapsed = time.time() - float(item.get("start_time", time.time()))
                print(
                    f"[Scheduler] done {completed}/{total} | phase={task['phase']} "
                    f"model={task['model_name']} trial={task['trial_id']} seed={task['seed']} "
                    f"gpu_local_id={gpu_local_id} elapsed={elapsed:.1f}s",
                    flush=True,
                )
                if pbar is not None:
                    pbar.update(1)
                free_gpus.append(gpu_local_id)
            running = still_running
            _render_dashboard()
            if pbar is not None:
                pbar.set_postfix_str(
                    f"pending={len(pending)} running={len(running)} [{_running_desc(running)}]"
                )
            now = time.time()
            if show_heartbeat and now - last_heartbeat >= max(1, int(heartbeat_sec)):
                print(
                    f"[Scheduler] heartbeat completed={completed}/{total} pending={len(pending)} "
                    f"running={len(running)} [{_running_desc(running)}]",
                    flush=True,
                )
                last_heartbeat = now
            if pending or running:
                time.sleep(1.0)
    finally:
        if use_dashboard:
            _render_dashboard()
        if pbar is not None:
            pbar.close()


def _run_single_worker_task(task_file: Path) -> None:
    task = json.loads(task_file.read_text(encoding="utf-8"))
    phase = int(task["phase"])
    model_name = str(task["model_name"])
    trial_id = int(task["trial_id"])
    seed = int(task["seed"])
    config = dict(task["config"])
    data_cfg = dict(task["data_cfg"])
    parallel_mode = bool(data_cfg.get("parallel_mode", False))
    progress_json_path = Path(task["progress_json_path"]) if task.get("progress_json_path") else None

    result_json_path = Path(task["result_json_path"])
    history_csv_path = Path(task["history_csv_path"])
    result_json_path.parent.mkdir(parents=True, exist_ok=True)
    history_csv_path.parent.mkdir(parents=True, exist_ok=True)
    if progress_json_path is not None:
        progress_json_path.parent.mkdir(parents=True, exist_ok=True)

    host = socket.gethostname()
    pid = os.getpid()
    gpu_local_id = int(os.environ.get("HPO_ASSIGNED_GPU_LOCAL_ID", "-1"))
    cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    print(
        f"[Worker] phase={phase} model={model_name} trial={trial_id} seed={seed} "
        f"gpu_local_id={gpu_local_id} pid={pid} host={host} CVD={cuda_visible_devices}"
    , flush=True)

    def _write_progress(payload: Dict[str, Any]) -> None:
        if progress_json_path is None:
            return
        tmp = progress_json_path.with_suffix(".tmp.json")
        tmp.write_text(json.dumps(sanitize_config(payload), ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(progress_json_path)

    _write_progress(
        {
            "status": "running",
            "phase": int(phase),
            "model_name": model_name,
            "trial_id": int(trial_id),
            "seed": int(seed),
            "gpu_local_id": int(gpu_local_id),
            "pid": int(pid),
            "epoch": 0,
            "total_epochs": int(config.get("epochs_budget", PHASE_DEFAULTS.get(phase, {}).get("epochs", 0))),
            "val_f1": None,
            "val_acc": None,
            "val_loss": None,
        }
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    try:
        use_preload = (not parallel_mode) and (not bool(data_cfg["no_preload"]))
        dataset = create_optimized_dataset(
            Path(data_cfg["data_root"]),
            n_frames=80,
            clip_mode=str(data_cfg["clip_mode"]),
            cache_dir=Path(data_cfg["cache_dir"]),
            preload_cache=use_preload,
            cache_trace=bool(data_cfg.get("cache_trace", False)),
        )
        train_idx, val_idx, test_idx = split_indices_3way(
            dataset,
            split_mode=str(data_cfg["split_mode"]),
            seed=int(data_cfg["split_seed"]),
            train_ratio=float(data_cfg["train_ratio"]),
            val_ratio=float(data_cfg["val_ratio"]),
        )
        train_set = Subset(dataset, train_idx)
        val_set = Subset(dataset, val_idx)
        test_set = Subset(dataset, test_idx)

        batch_size = int(config.get("batch_size", 8))
        train_loader, val_loader, test_loader = make_loaders(
            train_set,
            val_set,
            test_set if phase == 3 else None,
            batch_size=batch_size,
            num_workers=int(data_cfg["num_workers"]),
            seed=seed,
            parallel_mode=parallel_mode,
        )

        result, history = run_single_trial(
            model_name=model_name,
            config=config,
            phase=phase,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader if phase == 3 else None,
            device=device,
            seed=seed,
            progress_callback=(
                (lambda p: _write_progress({**p, "status": "running"}))
                if (parallel_mode and progress_json_path is not None)
                else None
            ),
            verbose_epoch_log=not (parallel_mode and progress_json_path is not None),
        )
    except Exception as e:
        result = {
            "model_name": model_name,
            "phase": int(phase),
            "trial_id": int(trial_id),
            "seed": int(seed),
            "lr": float(config.get("lr", np.nan)),
            "batch_size": int(config.get("batch_size", 8)),
            "weight_decay": float(config.get("weight_decay", np.nan)),
            "epochs_budget": int(config.get("epochs_budget", PHASE_DEFAULTS[phase]["epochs"])),
            "epochs_ran": 0,
            "best_epoch": -1,
            "best_val_f1": np.nan,
            "best_val_acc": np.nan,
            "best_val_loss": np.nan,
            "train_time_sec": np.nan,
            "num_params": np.nan,
            "test_f1": np.nan,
            "test_acc": np.nan,
            "status": "failed",
            "status_reason": str(e),
        }
        history = {
            "epoch": [],
            "train_loss": [],
            "val_loss": [],
            "val_acc": [],
            "val_f1": [],
            "lr": [],
            "epoch_time_s": [],
            "cum_time_s": [],
        }
        _write_progress(
            {
                "status": "failed",
                "phase": int(phase),
                "model_name": model_name,
                "trial_id": int(trial_id),
                "seed": int(seed),
                "gpu_local_id": int(gpu_local_id),
                "pid": int(pid),
                "error": str(e),
            }
        )

    result["host"] = host
    result["gpu_local_id"] = gpu_local_id
    result["cuda_visible_devices"] = cuda_visible_devices
    result["pid"] = int(pid)
    _write_history_csv(history, history_csv_path)
    result_json_path.write_text(json.dumps(sanitize_config(result), ensure_ascii=False, indent=2), encoding="utf-8")
    _write_progress(
        {
            "status": "done" if str(result.get("status", "")) == "ok" else str(result.get("status", "done")),
            "phase": int(phase),
            "model_name": model_name,
            "trial_id": int(trial_id),
            "seed": int(seed),
            "gpu_local_id": int(gpu_local_id),
            "pid": int(pid),
            "epoch": int(result.get("epochs_ran", 0)),
            "total_epochs": int(result.get("epochs_budget", 0)),
            "val_f1": result.get("best_val_f1", None),
            "val_acc": result.get("best_val_acc", None),
            "val_loss": result.get("best_val_loss", None),
        }
    )


class GentleEarlyStopping:
    def __init__(
        self,
        monitor: str = "val_f1",
        mode: str = "max",
        min_epochs: int = 15,
        patience: int = 8,
        min_delta: float = 1e-3,
    ) -> None:
        if mode not in {"max", "min"}:
            raise ValueError(f"Unsupported mode: {mode}")
        self.monitor = monitor
        self.mode = mode
        self.min_epochs = int(min_epochs)
        self.patience = int(patience)
        self.min_delta = float(min_delta)

        self.best_score: Optional[float] = None
        self.best_epoch: int = -1
        self.bad_epochs: int = 0
        self.best_state_dict: Optional[Dict[str, torch.Tensor]] = None

    def _is_improvement(self, value: float) -> bool:
        if self.best_score is None:
            return True
        if self.mode == "max":
            return value > self.best_score + self.min_delta
        return value < self.best_score - self.min_delta

    def step(self, value: float, epoch: int, model: nn.Module) -> bool:
        if self._is_improvement(value):
            self.best_score = float(value)
            self.best_epoch = int(epoch)
            self.bad_epochs = 0
            self.best_state_dict = copy.deepcopy(model.state_dict())
        else:
            self.bad_epochs += 1

        if epoch < self.min_epochs:
            return False
        return self.bad_epochs >= self.patience


def sample_cfg(space: Dict[str, List[Any]], rng: random.Random) -> Dict[str, Any]:
    cfg = {k: rng.choice(v) for k, v in space.items()}
    if "batch_size" not in cfg:
        cfg["batch_size"] = 8
    return cfg


def sanitize_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in cfg.items():
        if isinstance(v, (np.integer,)):
            out[k] = int(v)
        elif isinstance(v, (np.floating,)):
            out[k] = float(v)
        else:
            out[k] = v
    return out


def is_better_by_f1_then_acc(cand_f1: float, cand_acc: float, best_f1: float, best_acc: float, eps: float = 1e-12) -> bool:
    if cand_f1 > best_f1 + eps:
        return True
    if abs(cand_f1 - best_f1) <= eps and cand_acc > best_acc + eps:
        return True
    return False


def seed_everything(seed: int) -> None:
    set_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _worker_init_fn(seed: int):
    def _seed_worker(worker_id: int) -> None:
        worker_seed = seed + worker_id
        random.seed(worker_seed)
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)

    return _seed_worker


def make_loaders(
    train_set: Subset,
    val_set: Subset,
    test_set: Optional[Subset],
    batch_size: int,
    num_workers: int,
    seed: int,
    parallel_mode: bool = False,
) -> Tuple[DataLoader, DataLoader, Optional[DataLoader]]:
    effective_num_workers = min(int(num_workers), 2) if parallel_mode else int(num_workers)
    generator = torch.Generator()
    generator.manual_seed(seed)
    persistent_workers = effective_num_workers > 0
    pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=effective_num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        worker_init_fn=_worker_init_fn(seed),
        generator=generator,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=effective_num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        worker_init_fn=_worker_init_fn(seed + 97),
    )
    test_loader = None
    if test_set is not None:
        test_loader = DataLoader(
            test_set,
            batch_size=batch_size,
            shuffle=False,
            num_workers=effective_num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            worker_init_fn=_worker_init_fn(seed + 193),
        )
    return train_loader, val_loader, test_loader


def _evaluate_with_loss(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
) -> Tuple[float, float, float]:
    model.eval()
    total_loss = 0.0
    total_samples = 0
    ys: List[np.ndarray] = []
    ps: List[np.ndarray] = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            batch_size = y.size(0)
            total_loss += float(loss.item()) * batch_size
            total_samples += batch_size

            pred = logits.argmax(dim=1).cpu().numpy()
            ys.append(y.cpu().numpy())
            ps.append(pred)

    y_true = np.concatenate(ys)
    y_pred = np.concatenate(ps)
    _, _, f1_cls = per_class_prf(y_true, y_pred, n_classes=12)
    val_acc = float((y_true == y_pred).mean())
    val_f1 = float(np.nanmean(f1_cls))
    val_loss = total_loss / max(1, total_samples)
    return val_loss, val_acc, val_f1


def run_single_trial(
    model_name: str,
    config: Dict[str, Any],
    phase: int,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: Optional[DataLoader] = None,
    device: Optional[torch.device] = None,
    seed: int = 42,
    progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    verbose_epoch_log: bool = True,
) -> Tuple[Dict[str, Any], Dict[str, List[float]]]:
    if phase not in PHASE_DEFAULTS:
        raise ValueError(f"Unsupported phase: {phase}")
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    seed_everything(seed)
    cfg = sanitize_config(config)
    phase_cfg = PHASE_DEFAULTS[phase]

    epochs_budget = int(cfg.get("epochs_budget", phase_cfg["epochs"]))
    # Early stopping removed: all phases run fixed epoch budgets.
    use_early_stopping = False
    trial_id = int(cfg.get("trial_id", -1))

    lr = float(cfg.get("lr", 1e-3))
    weight_decay = float(cfg.get("weight_decay", 1e-5))
    batch_size = int(cfg.get("batch_size", train_loader.batch_size or 8))

    model_kwargs = {k: v for k, v in cfg.items() if k in MODEL_KWARGS_KEYS}
    if model_name.lower() == "vit" and int(model_kwargs.get("dim", 256)) % int(model_kwargs.get("heads", 8)) != 0:
        raise ValueError(f"Invalid ViT config: dim={model_kwargs.get('dim')} must be divisible by heads={model_kwargs.get('heads')}")

    model, _ = ModelFactory.build_torch(model_name, **model_kwargs)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    if "vit" in model_name.lower():
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=lr * 0.1)
    else:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

    scaler = torch.amp.GradScaler(device.type, enabled=(device.type == "cuda"))
    stopper = None
    if use_early_stopping:
        stopper = GentleEarlyStopping(
            monitor=str(cfg.get("es_monitor", "val_f1")),
            mode=str(cfg.get("es_mode", "max")),
            min_epochs=int(cfg.get("es_min_epochs", 15)),
            patience=int(cfg.get("es_patience", 8)),
            min_delta=float(cfg.get("es_min_delta", 1e-3)),
        )

    history: Dict[str, List[float]] = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "val_acc": [],
        "val_f1": [],
        "lr": [],
        "epoch_time_s": [],
        "cum_time_s": [],
    }

    best_weights = copy.deepcopy(model.state_dict())
    best_val_f1 = -1.0
    best_val_acc = -1.0
    best_val_loss = float("inf")
    best_epoch = -1

    cum_time = 0.0
    train_start = time.perf_counter()
    for epoch in range(1, epochs_budget + 1):
        ep_start = time.perf_counter()
        model.train()
        loss_sum = 0.0
        n_batches = 0

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=(device.type == "cuda")):
                logits = model(x)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            if "vit" in model_name.lower():
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            loss_sum += float(loss.item())
            n_batches += 1

        train_loss = loss_sum / max(1, n_batches)
        val_loss, val_acc, val_f1 = _evaluate_with_loss(model, val_loader, device, criterion)
        if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(val_loss)
        else:
            scheduler.step()

        epoch_time = time.perf_counter() - ep_start
        cum_time += epoch_time
        current_lr = float(optimizer.param_groups[0]["lr"])

        history["epoch"].append(float(epoch))
        history["train_loss"].append(float(train_loss))
        history["val_loss"].append(float(val_loss))
        history["val_acc"].append(float(val_acc))
        history["val_f1"].append(float(val_f1))
        history["lr"].append(float(current_lr))
        history["epoch_time_s"].append(float(epoch_time))
        history["cum_time_s"].append(float(cum_time))

        if is_better_by_f1_then_acc(val_f1, val_acc, best_val_f1, best_val_acc):
            best_val_f1 = float(val_f1)
            best_val_acc = float(val_acc)
            best_val_loss = float(val_loss)
            best_epoch = int(epoch)
            best_weights = copy.deepcopy(model.state_dict())

        if progress_callback is not None:
            progress_callback(
                {
                    "phase": int(phase),
                    "model_name": model_name,
                    "trial_id": int(trial_id),
                    "seed": int(seed),
                    "gpu_local_id": int(os.environ.get("HPO_ASSIGNED_GPU_LOCAL_ID", "-1")),
                    "pid": int(os.getpid()),
                    "epoch": int(epoch),
                    "total_epochs": int(epochs_budget),
                    "val_f1": float(val_f1),
                    "val_acc": float(val_acc),
                    "val_loss": float(val_loss),
                }
            )
        if verbose_epoch_log:
            gpu_local_id = os.environ.get("HPO_ASSIGNED_GPU_LOCAL_ID", "-1")
            print(
                f"[Phase {phase}] [Model {model_name}] [Trial {trial_id}] [Seed {seed}] "
                f"[GPU {gpu_local_id}] [PID {os.getpid()}] "
                f"Epoch {epoch}/{epochs_budget} | "
                f"val_f1={val_f1:.4f} val_acc={val_acc:.4f} val_loss={val_loss:.6f}"
            , flush=True)

        if stopper is not None:
            should_stop = stopper.step(value=val_f1, epoch=epoch, model=model)
            if should_stop:
                print(
                    f"[Phase {phase}] [Model {model_name}] [Trial {trial_id}] "
                    f"GentleEarlyStopping triggered at epoch {epoch}."
                )
                break

    if stopper is not None and stopper.best_state_dict is not None:
        model.load_state_dict(stopper.best_state_dict)
        if stopper.best_score is not None and stopper.best_epoch > 0:
            best_epoch = int(stopper.best_epoch)
            best_val_f1 = float(stopper.best_score)
            idx = best_epoch - 1
            if 0 <= idx < len(history["val_acc"]):
                best_val_acc = float(history["val_acc"][idx])
                best_val_loss = float(history["val_loss"][idx])
    else:
        model.load_state_dict(best_weights)

    epochs_ran = len(history["epoch"])
    train_time_sec = time.perf_counter() - train_start

    test_f1 = np.nan
    test_acc = np.nan
    if test_loader is not None:
        _, test_acc, test_f1 = _evaluate_with_loss(model, test_loader, device, criterion)

    result = {
        "model_name": model_name,
        "phase": int(phase),
        "trial_id": int(trial_id),
        "seed": int(seed),
        "lr": float(lr),
        "batch_size": int(batch_size),
        "weight_decay": float(weight_decay),
        "epochs_budget": int(epochs_budget),
        "epochs_ran": int(epochs_ran),
        "best_epoch": int(best_epoch),
        "best_val_f1": float(best_val_f1),
        "best_val_acc": float(best_val_acc),
        "best_val_loss": float(best_val_loss),
        "train_time_sec": float(train_time_sec),
        "num_params": int(count_parameters(model)),
        "test_f1": float(test_f1),
        "test_acc": float(test_acc),
        "status": "ok",
        "status_reason": "",
    }

    for k, v in model_kwargs.items():
        result[k] = v

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result, history


def select_top_k_configs(results: pd.DataFrame | Path | str, top_k: int = 3) -> List[Dict[str, Any]]:
    df = pd.read_csv(results) if isinstance(results, (Path, str)) else results.copy()
    if df.empty:
        return []

    sortable = df[df["status"] == "ok"].copy() if "status" in df.columns else df.copy()
    if sortable.empty:
        return []

    ranked = sortable.sort_values(["best_val_f1", "best_val_acc"], ascending=[False, False]).head(top_k)
    drop_cols = {
        "model_name",
        "phase",
        "trial_id",
        "seed",
        "epochs_budget",
        "epochs_ran",
        "best_epoch",
        "best_val_f1",
        "best_val_acc",
        "best_val_loss",
        "train_time_sec",
        "num_params",
        "test_f1",
        "test_acc",
        "status",
        "status_reason",
    }

    configs: List[Dict[str, Any]] = []
    for row in ranked.to_dict(orient="records"):
        cfg = {k: v for k, v in row.items() if k not in drop_cols and pd.notna(v)}
        configs.append(sanitize_config(cfg))
    return configs


def _write_history_csv(history: Dict[str, List[float]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(history).to_csv(out_path, index=False)


def run_phase1_for_model(
    model_name: str,
    search_space: Dict[str, List[Any]],
    n_trials: int,
    train_set: Optional[Subset],
    val_set: Optional[Subset],
    output_root: Path,
    seed: int,
    args: argparse.Namespace,
    device: torch.device,
    parallel: bool = False,
    gpu_ids: Optional[Sequence[int]] = None,
    max_workers: int = 1,
    epochs: int = 8,
) -> pd.DataFrame:
    phase_dir = output_root / "phase1" / model_name
    phase_dir.mkdir(parents=True, exist_ok=True)

    tasks_to_run: List[Dict[str, Any]] = []
    resumed_rows_by_trial: Dict[int, Dict[str, Any]] = {}
    rng = random.Random(seed)

    for trial in range(1, n_trials + 1):
        result_json = phase_dir / "trial_results" / f"trial_{trial:03d}.json"
        if args.resume and result_json.exists():
            try:
                old_row = json.loads(result_json.read_text(encoding="utf-8"))
                if str(old_row.get("status", "")).lower() == "ok":
                    resumed_rows_by_trial[int(trial)] = old_row
                    continue
            except Exception:
                pass

        cfg = sample_cfg(search_space, rng)
        cfg["trial_id"] = trial
        cfg["epochs_budget"] = int(epochs)
        cfg["use_early_stopping"] = False
        trial_seed = seed + trial
        history_csv = phase_dir / "histories" / f"trial_{trial:03d}_history.csv"
        task = _build_worker_task(
            task_id=f"phase1_{model_name}_{trial:03d}",
            phase=1,
            model_name=model_name,
            trial_id=trial,
            seed=trial_seed,
            config=cfg,
            result_json_path=result_json,
            history_csv_path=history_csv,
            progress_json_path=phase_dir / "progress" / f"trial_{trial:03d}.json",
            args=args,
        )
        task_file = phase_dir / "tasks" / f"trial_{trial:03d}.json"
        task_file.parent.mkdir(parents=True, exist_ok=True)
        task_file.write_text(json.dumps(task, ensure_ascii=False, indent=2), encoding="utf-8")
        task["task_file"] = str(task_file)
        tasks_to_run.append(task)

    if resumed_rows_by_trial:
        print(
            f"[Resume][Phase 1][{model_name}] Reusing {len(resumed_rows_by_trial)}/{n_trials} completed trials, "
            f"running remaining {len(tasks_to_run)}."
        )

    result_rows: List[Dict[str, Any]] = []
    if parallel and device.type == "cuda":
        if tasks_to_run:
            _run_tasks_with_gpu_scheduler(
                tasks_to_run,
                gpu_ids=gpu_ids or [],
                max_workers=max_workers,
                progress_desc=f"Phase1 {model_name} Trials",
                show_heartbeat=False,
                gpu_dashboard=True,
            )
        for task in tasks_to_run:
            result_json = Path(task["result_json_path"])
            result_rows.append(json.loads(result_json.read_text(encoding="utf-8")))
    else:
        if train_set is None or val_set is None:
            raise ValueError("train_set/val_set are required in non-parallel mode.")
        for task in tasks_to_run:
            cfg = dict(task["config"])
            trial_seed = int(task["seed"])
            batch_size = int(cfg.get("batch_size", 8))
            train_loader, val_loader, _ = make_loaders(
                train_set,
                val_set,
                None,
                batch_size=batch_size,
                num_workers=args.num_workers,
                seed=trial_seed,
                parallel_mode=False,
            )
            result, history = run_single_trial(
                model_name=model_name,
                config=cfg,
                phase=1,
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=None,
                device=device,
                seed=trial_seed,
            )
            result["host"] = socket.gethostname()
            result["gpu_local_id"] = -1
            result["cuda_visible_devices"] = os.environ.get("CUDA_VISIBLE_DEVICES", "")
            result["pid"] = int(os.getpid())
            _write_history_csv(history, Path(task["history_csv_path"]))
            Path(task["result_json_path"]).write_text(
                json.dumps(sanitize_config(result), ensure_ascii=False, indent=2), encoding="utf-8"
            )
            result_rows.append(result)

    all_rows = list(resumed_rows_by_trial.values()) + result_rows
    df = pd.DataFrame(all_rows)
    if not df.empty and "trial_id" in df.columns:
        df = df.sort_values("trial_id").reset_index(drop=True)
    df.to_csv(phase_dir / "phase1_results.csv", index=False)
    return df


def build_phase2_tasks_for_model(
    model_name: str,
    top_configs: Sequence[Dict[str, Any]],
    output_root: Path,
    seed: int,
    args: argparse.Namespace,
    epochs: int = 20,
) -> List[Dict[str, Any]]:
    phase_dir = output_root / "phase2" / model_name
    phase_dir.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, Any]] = []
    for idx, base_cfg in enumerate(top_configs, start=1):
        cfg = sanitize_config(dict(base_cfg))
        cfg["trial_id"] = idx
        cfg["epochs_budget"] = int(epochs)
        cfg["use_early_stopping"] = False
        trial_seed = seed + 200 + idx
        result_json = phase_dir / "trial_results" / f"trial_{idx:03d}.json"
        history_csv = phase_dir / "histories" / f"trial_{idx:03d}_history.csv"
        task = _build_worker_task(
            task_id=f"phase2_{model_name}_{idx:03d}",
            phase=2,
            model_name=model_name,
            trial_id=idx,
            seed=trial_seed,
            config=cfg,
            result_json_path=result_json,
            history_csv_path=history_csv,
            progress_json_path=phase_dir / "progress" / f"trial_{idx:03d}.json",
            args=args,
        )
        task_file = phase_dir / "tasks" / f"trial_{idx:03d}.json"
        task_file.parent.mkdir(parents=True, exist_ok=True)
        task_file.write_text(json.dumps(task, ensure_ascii=False, indent=2), encoding="utf-8")
        task["task_file"] = str(task_file)
        rows.append(task)
    return rows


def finalize_phase2_results_for_model(
    model_name: str,
    tasks: Sequence[Dict[str, Any]],
    output_root: Path,
) -> pd.DataFrame:
    phase_dir = output_root / "phase2" / model_name
    result_rows: List[Dict[str, Any]] = []
    for task in tasks:
        result_json = Path(task["result_json_path"])
        result_rows.append(json.loads(result_json.read_text(encoding="utf-8")))
    df = pd.DataFrame(result_rows)
    df.to_csv(phase_dir / "phase2_results.csv", index=False)
    return df


def run_phase2_for_model(
    model_name: str,
    top_configs: Sequence[Dict[str, Any]],
    train_set: Optional[Subset],
    val_set: Optional[Subset],
    output_root: Path,
    seed: int,
    args: argparse.Namespace,
    device: torch.device,
    parallel: bool = False,
    gpu_ids: Optional[Sequence[int]] = None,
    max_workers: int = 1,
    epochs: int = 20,
) -> pd.DataFrame:
    phase_dir = output_root / "phase2" / model_name
    phase_dir.mkdir(parents=True, exist_ok=True)
    rows = build_phase2_tasks_for_model(
        model_name=model_name,
        top_configs=top_configs,
        output_root=output_root,
        seed=seed,
        args=args,
        epochs=epochs,
    )

    result_rows: List[Dict[str, Any]] = []
    if parallel and device.type == "cuda":
        _run_tasks_with_gpu_scheduler(rows, gpu_ids=gpu_ids or [], max_workers=max_workers, progress_desc="Phase2 Trials")
        return finalize_phase2_results_for_model(model_name=model_name, tasks=rows, output_root=output_root)
    else:
        if train_set is None or val_set is None:
            raise ValueError("train_set/val_set are required in non-parallel mode.")
        for idx, task in enumerate(rows, start=1):
            cfg = dict(task["config"])
            trial_seed = int(task["seed"])
            batch_size = int(cfg.get("batch_size", 8))
            train_loader, val_loader, _ = make_loaders(
                train_set,
                val_set,
                None,
                batch_size=batch_size,
                num_workers=args.num_workers,
                seed=trial_seed,
                parallel_mode=False,
            )
            result, history = run_single_trial(
                model_name=model_name,
                config=cfg,
                phase=2,
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=None,
                device=device,
                seed=trial_seed,
            )
            result["host"] = socket.gethostname()
            result["gpu_local_id"] = -1
            result["cuda_visible_devices"] = os.environ.get("CUDA_VISIBLE_DEVICES", "")
            result["pid"] = int(os.getpid())
            _write_history_csv(history, Path(task["history_csv_path"]))
            Path(task["result_json_path"]).write_text(
                json.dumps(sanitize_config(result), ensure_ascii=False, indent=2), encoding="utf-8"
            )
            result_rows.append(result)

    df = pd.DataFrame(result_rows)
    df.to_csv(phase_dir / "phase2_results.csv", index=False)
    return df


def run_phase3_final(
    model_name: str,
    best_config: Dict[str, Any],
    train_set: Optional[Subset],
    val_set: Optional[Subset],
    test_set: Optional[Subset],
    output_root: Path,
    seeds: Sequence[int],
    args: argparse.Namespace,
    device: torch.device,
    parallel: bool = False,
    gpu_ids: Optional[Sequence[int]] = None,
    max_workers: int = 1,
    epochs: int = 40,
    min_epochs: int = 15,
    patience: int = 8,
    min_delta: float = 1e-3,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    phase_dir = output_root / "phase3" / model_name
    phase_dir.mkdir(parents=True, exist_ok=True)

    rows = build_phase3_tasks_for_model(
        model_name=model_name,
        best_config=best_config,
        seeds=seeds,
        output_root=output_root,
        args=args,
        epochs=epochs,
        min_epochs=min_epochs,
        patience=patience,
        min_delta=min_delta,
    )

    result_rows: List[Dict[str, Any]] = []
    if parallel and device.type == "cuda":
        _run_tasks_with_gpu_scheduler(rows, gpu_ids=gpu_ids or [], max_workers=max_workers, progress_desc="Phase3 Trials")
        return finalize_phase3_results_for_model(model_name=model_name, tasks=rows, output_root=output_root)
    else:
        if train_set is None or val_set is None or test_set is None:
            raise ValueError("train_set/val_set/test_set are required in non-parallel mode.")
        for task in rows:
            cfg = dict(task["config"])
            run_seed = int(task["seed"])
            batch_size = int(cfg.get("batch_size", 8))
            train_loader, val_loader, test_loader = make_loaders(
                train_set,
                val_set,
                test_set,
                batch_size=batch_size,
                num_workers=args.num_workers,
                seed=run_seed,
                parallel_mode=False,
            )
            result, history = run_single_trial(
                model_name=model_name,
                config=cfg,
                phase=3,
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=test_loader,
                device=device,
                seed=run_seed,
            )
            result["host"] = socket.gethostname()
            result["gpu_local_id"] = -1
            result["cuda_visible_devices"] = os.environ.get("CUDA_VISIBLE_DEVICES", "")
            result["pid"] = int(os.getpid())
            _write_history_csv(history, Path(task["history_csv_path"]))
            Path(task["result_json_path"]).write_text(
                json.dumps(sanitize_config(result), ensure_ascii=False, indent=2), encoding="utf-8"
            )
            result_rows.append(result)
    final_df = pd.DataFrame(result_rows)
    final_df.to_csv(phase_dir / "final_results.csv", index=False)
    summary_df = summarize_phase3_results(model_name=model_name, final_df=final_df)
    summary_df.to_csv(phase_dir / "final_results_summary.csv", index=False)
    return final_df, summary_df


def build_phase3_tasks_for_model(
    model_name: str,
    best_config: Dict[str, Any],
    seeds: Sequence[int],
    output_root: Path,
    args: argparse.Namespace,
    epochs: int = 30,
    min_epochs: int = 15,
    patience: int = 8,
    min_delta: float = 1e-3,
) -> List[Dict[str, Any]]:
    phase_dir = output_root / "phase3" / model_name
    phase_dir.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, Any]] = []
    for run_id, run_seed in enumerate(seeds, start=1):
        cfg = sanitize_config(dict(best_config))
        cfg["trial_id"] = run_id
        cfg["epochs_budget"] = int(epochs)
        cfg["use_early_stopping"] = False
        result_json = phase_dir / "trial_results" / f"seed_{run_seed}.json"
        history_csv = phase_dir / "histories" / f"seed_{run_seed}_history.csv"
        task = _build_worker_task(
            task_id=f"phase3_{model_name}_{run_seed}",
            phase=3,
            model_name=model_name,
            trial_id=run_id,
            seed=run_seed,
            config=cfg,
            result_json_path=result_json,
            history_csv_path=history_csv,
            progress_json_path=None,
            args=args,
        )
        task_file = phase_dir / "tasks" / f"seed_{run_seed}.json"
        task_file.parent.mkdir(parents=True, exist_ok=True)
        task_file.write_text(json.dumps(task, ensure_ascii=False, indent=2), encoding="utf-8")
        task["task_file"] = str(task_file)
        rows.append(task)
    return rows


def summarize_phase3_results(model_name: str, final_df: pd.DataFrame) -> pd.DataFrame:
    ok = final_df[final_df["status"] == "ok"].copy() if "status" in final_df.columns else final_df.copy()
    if ok.empty:
        return pd.DataFrame([{"model_name": model_name, "n_runs": 0}])
    return pd.DataFrame(
        [
            {
                "model_name": model_name,
                "n_runs": int(len(ok)),
                "best_val_f1_mean": float(ok["best_val_f1"].mean()),
                "best_val_f1_std": float(ok["best_val_f1"].std(ddof=0)),
                "best_val_acc_mean": float(ok["best_val_acc"].mean()),
                "best_val_acc_std": float(ok["best_val_acc"].std(ddof=0)),
                "test_f1_mean": float(ok["test_f1"].mean()),
                "test_f1_std": float(ok["test_f1"].std(ddof=0)),
                "test_acc_mean": float(ok["test_acc"].mean()),
                "test_acc_std": float(ok["test_acc"].std(ddof=0)),
                "train_time_sec_mean": float(ok["train_time_sec"].mean()),
                "train_time_sec_std": float(ok["train_time_sec"].std(ddof=0)),
            }
        ]
    )


def finalize_phase3_results_for_model(
    model_name: str,
    tasks: Sequence[Dict[str, Any]],
    output_root: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    phase_dir = output_root / "phase3" / model_name
    result_rows: List[Dict[str, Any]] = []
    for task in tasks:
        result_json = Path(task["result_json_path"])
        result_rows.append(json.loads(result_json.read_text(encoding="utf-8")))
    final_df = pd.DataFrame(result_rows)
    final_df.to_csv(phase_dir / "final_results.csv", index=False)
    summary_df = summarize_phase3_results(model_name=model_name, final_df=final_df)
    summary_df.to_csv(phase_dir / "final_results_summary.csv", index=False)
    return final_df, summary_df


def _prepare_data(args: argparse.Namespace) -> Tuple[Subset, Subset, Subset]:
    use_preload = (not bool(args.parallel)) and (not bool(args.no_preload))
    dataset = create_optimized_dataset(
        args.data_root,
        n_frames=80,
        clip_mode=args.clip_mode,
        cache_dir=args.cache_dir,
        preload_cache=use_preload,
        cache_trace=bool(args.cache_trace),
    )
    print(f"Dataset: {len(dataset)} samples")

    if not args.skip_cache_warmup:
        if args.parallel:
            print("[Warmup Note] Parallel mode + max_batches=1 warmup gives limited benefit for global cache coverage.")
        print("[Stage] Cache warmup started...")
        warmup_max_batches = None if bool(args.warmup_full_cache) else 1
        warmup_cache(
            dataset,
            batch_size=128,
            num_workers=0,
            shuffle=False,
            pin_memory=False,
            max_batches=warmup_max_batches,
            use_tqdm=not args.no_warmup_tqdm,
            tqdm_desc="Cache Warmup",
        )
        print("[Stage] Cache warmup done.")

    train_idx, val_idx, test_idx = split_indices_3way(
        dataset,
        split_mode=args.split_mode,
        seed=args.seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
    )
    train_set = Subset(dataset, train_idx)
    val_set = Subset(dataset, val_idx)
    test_set = Subset(dataset, test_idx)
    print(f"Split sizes: train={len(train_set)} val={len(val_set)} test={len(test_set)}")
    return train_set, val_set, test_set


def run_pipeline(args: argparse.Namespace) -> None:
    output_root = args.output_dir
    output_root.mkdir(parents=True, exist_ok=True)

    seed_everything(args.seed)
    device = get_device()
    print(f"Device: {device}")

    train_set, val_set, test_set = _prepare_data(args)
    model_names_raw = parse_model_list(
        args.models,
        ["LeNet", "AlexNet", "ResNet18", "MobileNet_V2", "EfficientNet_B0", "LSTM", "GRU", "CNN_LSTM", "TCN", "ViT"],
    )
    deep_first = [m for m in model_names_raw if m.lower() in DEEP_SPACE]
    non_deep_later = [m for m in model_names_raw if m.lower() not in DEEP_SPACE]
    model_names = deep_first + non_deep_later
    if non_deep_later:
        print(
            f"[Queue] Deep/GPU models first: {deep_first} | Non-deep models postponed: {non_deep_later}"
        )
    else:
        print(f"[Queue] Deep/GPU model order: {deep_first}")

    all_phase_meta: Dict[str, Any] = {
        "meta": {
            "seed": args.seed,
            "phase1_epochs": args.phase1_epochs,
            "phase2_epochs": args.phase2_epochs,
            "phase3_epochs": args.phase3_epochs,
            "phase1_trials": args.phase1_trials,
            "phase2_topk": args.phase2_topk,
            "phase3_seeds": args.phase3_seeds,
            "selection_rule": "primary=best_val_f1, tie_breaker=best_val_acc",
            "parallel": bool(args.parallel),
            "gpu_ids": list(args.gpu_ids_list),
            "max_workers": int(args.max_workers),
            "resume": bool(args.resume),
        },
        "models": {},
    }

    total_models = len(model_names)

    # Phase-2 only: use a single global cross-model trial queue when parallel GPU mode is enabled.
    if args.phase == "phase2" and bool(args.parallel) and device.type == "cuda":
        phase2_global_tasks: List[Dict[str, Any]] = []
        phase2_tasks_by_model: Dict[str, List[Dict[str, Any]]] = {}
        phase1_df_by_model: Dict[str, pd.DataFrame] = {}
        phase2_df_by_model: Dict[str, pd.DataFrame] = {}

        for model_idx, model_name in enumerate(model_names, start=1):
            print(f"\n[Model Progress] {model_idx}/{total_models} -> {model_name}")
            key = model_name.lower()
            if key not in DEEP_SPACE:
                print(f"[WARN] No search space for model {model_name}, skipped.")
                _save_model_progress(
                    output_root,
                    model_name,
                    {
                        "model_name": model_name,
                        "status": "skipped",
                        "reason": "no_search_space",
                        "finished_at": _now_iso(),
                        "phase": "phase2",
                    },
                )
                continue

            model_meta: Dict[str, Any] = {}
            phase1_csv = output_root / "phase1" / model_name / "phase1_results.csv"
            phase2_csv = output_root / "phase2" / model_name / "phase2_results.csv"

            if phase1_csv.exists():
                phase1_df = pd.read_csv(phase1_csv)
            else:
                print(f"[Phase 1] Missing for {model_name}; generating prerequisite phase1 results first.")
                phase1_df = run_phase1_for_model(
                    model_name=model_name,
                    search_space=DEEP_SPACE[key],
                    n_trials=args.phase1_trials,
                    train_set=train_set,
                    val_set=val_set,
                    output_root=output_root,
                    seed=args.seed,
                    args=args,
                    device=device,
                    parallel=bool(args.parallel),
                    gpu_ids=args.gpu_ids_list,
                    max_workers=args.max_workers,
                    epochs=args.phase1_epochs,
                )

            phase1_df_by_model[model_name] = phase1_df
            top_from_phase1 = select_top_k_configs(phase1_df, top_k=args.phase2_topk)
            model_meta["phase1_top_configs"] = top_from_phase1
            all_phase_meta["models"][model_name] = model_meta

            expected_trials = len(top_from_phase1)
            if expected_trials <= 0:
                print(f"[Phase 2] No top configs available for {model_name}; skip.")
                phase2_df_by_model[model_name] = pd.DataFrame()
                continue

            if args.resume and _is_phase2_complete(output_root, model_name, expected_trials=expected_trials):
                print(f"[Resume] Skip completed phase2 for {model_name} (phase2 results already complete).")
                phase2_df_by_model[model_name] = pd.read_csv(phase2_csv)
                continue

            tasks = build_phase2_tasks_for_model(
                model_name=model_name,
                top_configs=top_from_phase1,
                output_root=output_root,
                seed=args.seed,
                args=args,
                epochs=args.phase2_epochs,
            )
            phase2_tasks_by_model[model_name] = tasks
            phase2_global_tasks.extend(tasks)
            _save_model_progress(
                output_root,
                model_name,
                {
                    "model_name": model_name,
                    "status": "running",
                    "started_at": _now_iso(),
                    "phase": "phase2",
                    "pending_trials": int(len(tasks)),
                },
            )

        queued_models = sorted([m for m, ts in phase2_tasks_by_model.items() if ts])
        print(
            f"[Phase2 Global Queue] total_tasks={len(phase2_global_tasks)} "
            f"total_models={len(queued_models)} models={queued_models}"
        )
        if phase2_global_tasks:
            _run_tasks_with_gpu_scheduler(
                phase2_global_tasks,
                gpu_ids=args.gpu_ids_list,
                max_workers=args.max_workers,
                progress_desc="Phase2 Global Trials",
                show_heartbeat=False,
                gpu_dashboard=True,
            )

        for model_name in model_names:
            if model_name in phase2_tasks_by_model:
                phase2_df_by_model[model_name] = finalize_phase2_results_for_model(
                    model_name=model_name,
                    tasks=phase2_tasks_by_model[model_name],
                    output_root=output_root,
                )

            if model_name not in all_phase_meta["models"]:
                continue

            model_meta = all_phase_meta["models"][model_name]
            phase1_df = phase1_df_by_model.get(model_name, pd.DataFrame())
            phase2_df = phase2_df_by_model.get(model_name, pd.DataFrame())
            best_list = select_top_k_configs(phase2_df if not phase2_df.empty else phase1_df, top_k=1)
            model_meta["best_config"] = best_list[0] if best_list else None

            _save_model_progress(
                output_root,
                model_name,
                {
                    "model_name": model_name,
                    "status": "done",
                    "finished_at": _now_iso(),
                    "phase": "phase2",
                    "paths": {
                        "phase1": str(output_root / "phase1" / model_name / "phase1_results.csv"),
                        "phase2": str(output_root / "phase2" / model_name / "phase2_results.csv"),
                        "phase3": str(output_root / "phase3" / model_name / "final_results.csv"),
                        "phase3_summary": str(output_root / "phase3" / model_name / "final_results_summary.csv"),
                    },
                },
            )
            print(f"[Model Done] {model_name} phase2 finalized.")

        meta_path = output_root / "hpo_pipeline_meta.json"
        meta_path.write_text(json.dumps(all_phase_meta, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Saved pipeline meta: {meta_path}")
        return

    # Phase-3 only: use a single global cross-model seed queue when parallel GPU mode is enabled.
    if args.phase == "phase3" and bool(args.parallel) and device.type == "cuda":
        phase3_global_tasks: List[Dict[str, Any]] = []
        phase3_tasks_by_model: Dict[str, List[Dict[str, Any]]] = {}
        phase1_df_by_model: Dict[str, pd.DataFrame] = {}
        phase2_df_by_model: Dict[str, pd.DataFrame] = {}
        phase3_df_by_model: Dict[str, pd.DataFrame] = {}
        phase3_summary_by_model: Dict[str, pd.DataFrame] = {}

        for model_idx, model_name in enumerate(model_names, start=1):
            print(f"\n[Model Progress] {model_idx}/{total_models} -> {model_name}")
            key = model_name.lower()
            if key not in DEEP_SPACE:
                print(f"[WARN] No search space for model {model_name}, skipped.")
                _save_model_progress(
                    output_root,
                    model_name,
                    {
                        "model_name": model_name,
                        "status": "skipped",
                        "reason": "no_search_space",
                        "finished_at": _now_iso(),
                        "phase": "phase3",
                    },
                )
                continue

            model_meta: Dict[str, Any] = {}
            phase1_csv = output_root / "phase1" / model_name / "phase1_results.csv"
            phase2_csv = output_root / "phase2" / model_name / "phase2_results.csv"

            if phase1_csv.exists():
                phase1_df = pd.read_csv(phase1_csv)
            else:
                print(f"[Phase 1] Missing for {model_name}; generating prerequisite phase1 results first.")
                phase1_df = run_phase1_for_model(
                    model_name=model_name,
                    search_space=DEEP_SPACE[key],
                    n_trials=args.phase1_trials,
                    train_set=train_set,
                    val_set=val_set,
                    output_root=output_root,
                    seed=args.seed,
                    args=args,
                    device=device,
                    parallel=bool(args.parallel),
                    gpu_ids=args.gpu_ids_list,
                    max_workers=args.max_workers,
                    epochs=args.phase1_epochs,
                )
            phase1_df_by_model[model_name] = phase1_df
            top_from_phase1 = select_top_k_configs(phase1_df, top_k=args.phase2_topk)
            model_meta["phase1_top_configs"] = top_from_phase1

            if phase2_csv.exists():
                phase2_df = pd.read_csv(phase2_csv)
            elif top_from_phase1:
                print(
                    f"[Phase 2] Missing for {model_name}; generating prerequisite phase2 results "
                    f"(top-{len(top_from_phase1)}, epochs={args.phase2_epochs})."
                )
                phase2_df = run_phase2_for_model(
                    model_name=model_name,
                    top_configs=top_from_phase1,
                    train_set=train_set,
                    val_set=val_set,
                    output_root=output_root,
                    seed=args.seed,
                    args=args,
                    device=device,
                    parallel=bool(args.parallel),
                    gpu_ids=args.gpu_ids_list,
                    max_workers=args.max_workers,
                    epochs=args.phase2_epochs,
                )
            else:
                phase2_df = pd.DataFrame()
            phase2_df_by_model[model_name] = phase2_df

            best_list = select_top_k_configs(phase2_df if not phase2_df.empty else phase1_df, top_k=1)
            model_meta["best_config"] = best_list[0] if best_list else None
            all_phase_meta["models"][model_name] = model_meta

            if not best_list:
                print(f"[Phase 3] No valid best config for {model_name}; skip.")
                continue

            seeds = [args.seed + 1000 + i for i in range(args.phase3_seeds)]
            if args.resume and _is_phase3_complete(output_root, model_name, expected_runs=len(seeds)):
                print(f"[Resume] Skip completed phase3 for {model_name} (phase3 results already complete).")
                final_df = pd.read_csv(output_root / "phase3" / model_name / "final_results.csv")
                phase3_df_by_model[model_name] = final_df
                phase3_summary_by_model[model_name] = summarize_phase3_results(model_name=model_name, final_df=final_df)
                continue

            tasks = build_phase3_tasks_for_model(
                model_name=model_name,
                best_config=best_list[0],
                seeds=seeds,
                output_root=output_root,
                args=args,
                epochs=args.phase3_epochs,
                min_epochs=args.es_min_epochs,
                patience=args.es_patience,
                min_delta=args.es_min_delta,
            )
            phase3_tasks_by_model[model_name] = tasks
            phase3_global_tasks.extend(tasks)
            _save_model_progress(
                output_root,
                model_name,
                {
                    "model_name": model_name,
                    "status": "running",
                    "started_at": _now_iso(),
                    "phase": "phase3",
                    "pending_runs": int(len(tasks)),
                },
            )

        queued_models = sorted([m for m, ts in phase3_tasks_by_model.items() if ts])
        print(
            f"[Phase3 Global Queue] total_tasks={len(phase3_global_tasks)} "
            f"total_models={len(queued_models)} models={queued_models}"
        )
        if phase3_global_tasks:
            _run_tasks_with_gpu_scheduler(
                phase3_global_tasks,
                gpu_ids=args.gpu_ids_list,
                max_workers=args.max_workers,
                progress_desc="Phase3 Global Trials",
                show_heartbeat=False,
                gpu_dashboard=False,
            )

        for model_name in model_names:
            if model_name in phase3_tasks_by_model:
                final_df, summary_df = finalize_phase3_results_for_model(
                    model_name=model_name,
                    tasks=phase3_tasks_by_model[model_name],
                    output_root=output_root,
                )
                phase3_df_by_model[model_name] = final_df
                phase3_summary_by_model[model_name] = summary_df

            model_meta = all_phase_meta["models"].get(model_name)
            if model_meta is None:
                continue
            if model_name in phase3_df_by_model:
                model_meta["phase3_n_runs"] = int(len(phase3_df_by_model[model_name]))
                model_meta["phase3_summary"] = phase3_summary_by_model[model_name].to_dict(orient="records")

            _save_model_progress(
                output_root,
                model_name,
                {
                    "model_name": model_name,
                    "status": "done",
                    "finished_at": _now_iso(),
                    "phase": "phase3",
                    "paths": {
                        "phase1": str(output_root / "phase1" / model_name / "phase1_results.csv"),
                        "phase2": str(output_root / "phase2" / model_name / "phase2_results.csv"),
                        "phase3": str(output_root / "phase3" / model_name / "final_results.csv"),
                        "phase3_summary": str(output_root / "phase3" / model_name / "final_results_summary.csv"),
                    },
                },
            )
            print(f"[Model Done] {model_name} phase3 finalized.")

        meta_path = output_root / "hpo_pipeline_meta.json"
        meta_path.write_text(json.dumps(all_phase_meta, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Saved pipeline meta: {meta_path}")
        return

    for model_idx, model_name in enumerate(model_names, start=1):
        print(f"\n[Model Progress] {model_idx}/{total_models} -> {model_name}")
        key = model_name.lower()
        if key not in DEEP_SPACE:
            print(f"[WARN] No search space for model {model_name}, skipped.")
            _save_model_progress(
                output_root,
                model_name,
                {
                    "model_name": model_name,
                    "status": "skipped",
                    "reason": "no_search_space",
                    "finished_at": _now_iso(),
                },
            )
            continue

        print(f"\n=== Model: {model_name} ===")
        model_meta: Dict[str, Any] = {}
        phase1_csv = output_root / "phase1" / model_name / "phase1_results.csv"
        phase2_csv = output_root / "phase2" / model_name / "phase2_results.csv"

        if args.phase in {"phase1", "all"} and args.resume and _is_phase1_complete(
            output_root, model_name, expected_trials=args.phase1_trials
        ):
            print(f"[Resume] Skip completed phase1 for {model_name}.")
            phase1_df = pd.read_csv(phase1_csv)
        elif phase1_csv.exists() and args.phase in {"phase2", "phase3"}:
            phase1_df = pd.read_csv(phase1_csv)
        else:
            print(f"[Phase 1] Running {args.phase1_trials} trials, fixed {args.phase1_epochs} epochs, no early stopping.")
            phase1_df = run_phase1_for_model(
                model_name=model_name,
                search_space=DEEP_SPACE[key],
                n_trials=args.phase1_trials,
                train_set=train_set,
                val_set=val_set,
                output_root=output_root,
                seed=args.seed,
                args=args,
                device=device,
                parallel=bool(args.parallel),
                gpu_ids=args.gpu_ids_list,
                max_workers=args.max_workers,
                epochs=args.phase1_epochs,
            )

        top_from_phase1 = select_top_k_configs(phase1_df, top_k=args.phase2_topk)
        model_meta["phase1_top_configs"] = top_from_phase1

        if args.phase in {"phase2", "all", "phase3"} and top_from_phase1:
            expected_phase2_trials = len(top_from_phase1)
            if args.phase in {"phase2", "all"} and args.resume and _is_phase2_complete(
                output_root, model_name, expected_trials=expected_phase2_trials
            ):
                print(f"[Resume] Skip completed phase2 for {model_name}.")
                phase2_df = pd.read_csv(phase2_csv)
            elif args.phase == "phase3" and phase2_csv.exists():
                phase2_df = pd.read_csv(phase2_csv)
            else:
                print(
                    f"[Phase 2] Running top-{len(top_from_phase1)} configs, fixed {args.phase2_epochs} epochs, no early stopping."
                )
                phase2_df = run_phase2_for_model(
                    model_name=model_name,
                    top_configs=top_from_phase1,
                    train_set=train_set,
                    val_set=val_set,
                    output_root=output_root,
                    seed=args.seed,
                    args=args,
                    device=device,
                    parallel=bool(args.parallel),
                    gpu_ids=args.gpu_ids_list,
                    max_workers=args.max_workers,
                    epochs=args.phase2_epochs,
                )
        elif phase2_csv.exists():
            phase2_df = pd.read_csv(phase2_csv)
        else:
            phase2_df = pd.DataFrame()

        best_list = select_top_k_configs(phase2_df if not phase2_df.empty else phase1_df, top_k=1)
        model_meta["best_config"] = best_list[0] if best_list else None

        if args.phase in {"phase3", "all"} and best_list:
            if args.resume and _is_phase3_complete(output_root, model_name, expected_runs=args.phase3_seeds):
                print(f"[Resume] Skip completed phase3 for {model_name}.")
            else:
                print(
                    f"[Phase 3] Running final training over {args.phase3_seeds} seeds "
                    f"(fixed epochs={args.phase3_epochs}, no early stopping)."
                )
                seeds = [args.seed + 1000 + i for i in range(args.phase3_seeds)]
                final_df, summary_df = run_phase3_final(
                    model_name=model_name,
                    best_config=best_list[0],
                    train_set=train_set,
                    val_set=val_set,
                    test_set=test_set,
                    output_root=output_root,
                    seeds=seeds,
                    args=args,
                    device=device,
                    parallel=bool(args.parallel),
                    gpu_ids=args.gpu_ids_list,
                    max_workers=args.max_workers,
                    epochs=args.phase3_epochs,
                    min_epochs=args.es_min_epochs,
                    patience=args.es_patience,
                    min_delta=args.es_min_delta,
                )
                model_meta["phase3_n_runs"] = int(len(final_df))
                model_meta["phase3_summary"] = summary_df.to_dict(orient="records")

        all_phase_meta["models"][model_name] = model_meta
        _save_model_progress(
            output_root,
            model_name,
            {
                "model_name": model_name,
                "status": "done",
                "finished_at": _now_iso(),
                "phase": args.phase,
                "paths": {
                    "phase1": str(output_root / "phase1" / model_name / "phase1_results.csv"),
                    "phase2": str(output_root / "phase2" / model_name / "phase2_results.csv"),
                    "phase3": str(output_root / "phase3" / model_name / "final_results.csv"),
                    "phase3_summary": str(output_root / "phase3" / model_name / "final_results_summary.csv"),
                },
            },
        )
        meta_path = output_root / "hpo_pipeline_meta.json"
        meta_path.write_text(json.dumps(all_phase_meta, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[Model Done] {model_name} saved progress + meta.")

    meta_path = output_root / "hpo_pipeline_meta.json"
    meta_path.write_text(json.dumps(all_phase_meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved pipeline meta: {meta_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Three-phase HPO pipeline (fixed budget + staged selection + gentle ES)")
    parser.add_argument("--data_root", type=Path, default=None)
    parser.add_argument("--cache_dir", type=Path, default=Path(".cache_tacact_n80_weighted"))
    parser.add_argument("--output_dir", type=Path, default=Path("outputs"))
    parser.add_argument("--split_mode", choices=["subject", "random"], default="subject")
    parser.add_argument("--clip_mode", choices=["weighted_center"], default="weighted_center")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_ratio", type=float, default=0.7)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument(
        "--models",
        type=str,
        default="LeNet,AlexNet,ResNet18,MobileNet_V2,EfficientNet_B0,LSTM,GRU,CNN_LSTM,TCN,ViT",
    )
    parser.add_argument("--phase", choices=["phase1", "phase2", "phase3", "all"], default="all")

    parser.add_argument("--phase1_trials", type=int, default=12)
    parser.add_argument("--phase1_epochs", type=int, default=8)
    parser.add_argument("--phase2_topk", type=int, default=2)
    parser.add_argument("--phase2_epochs", type=int, default=18)

    parser.add_argument("--phase3_epochs", type=int, default=30)
    parser.add_argument("--phase3_seeds", type=int, default=2)
    parser.add_argument("--es_min_epochs", type=int, default=15)
    parser.add_argument("--es_patience", type=int, default=8)
    parser.add_argument("--es_min_delta", type=float, default=1e-3)

    parser.add_argument("--skip_cache_warmup", action="store_true")
    parser.add_argument(
        "--warmup_full_cache",
        action="store_true",
        help="Warm up full dataset cache before training (max_batches=None).",
    )
    parser.add_argument("--no_warmup_tqdm", action="store_true")
    parser.add_argument("--no_preload", action="store_true")
    parser.add_argument("--cache_trace", action="store_true", help="Print dataset cache hit/miss traces and summaries.")
    parser.add_argument("--parallel", action="store_true", help="Enable multi-process single-GPU trial scheduling.")
    parser.add_argument(
        "--gpu_ids",
        type=str,
        default=None,
        help="Comma-separated local GPU ids from current visible pool (e.g., 0,1,2,3,4).",
    )
    parser.add_argument("--max_workers", type=int, default=5, help="Max concurrent trial workers.")
    parser.add_argument("--worker_mode", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--task_file", type=Path, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--resume", action="store_true", default=True, help="Resume by skipping models with done progress files.")
    parser.add_argument("--no_resume", action="store_true", help="Disable resume behavior and rerun all models.")
    return parser


def main() -> None:
    mp.set_start_method("spawn", force=True)
    parser = build_parser()
    args = parser.parse_args()
    if args.no_resume:
        args.resume = False
    if args.worker_mode:
        if args.task_file is None:
            raise ValueError("--worker_mode requires --task_file")
        _run_single_worker_task(args.task_file)
        return
    if args.data_root is None:
        raise ValueError("--data_root is required unless --worker_mode is set")

    visible_gpu_count = torch.cuda.device_count()
    args.gpu_ids_list = _parse_gpu_ids(args.gpu_ids, visible_gpu_count)
    if args.max_workers <= 0:
        raise ValueError("--max_workers must be >= 1")
    if args.max_workers > len(args.gpu_ids_list):
        raise ValueError(
            f"--max_workers ({args.max_workers}) cannot exceed selected gpu_ids size ({len(args.gpu_ids_list)})."
        )
    if args.parallel and not args.gpu_ids_list:
        raise ValueError("Parallel mode requires at least one visible GPU.")

    print(
        f"[GPU Pool] CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')} | "
        f"visible={visible_gpu_count} | selected_local_gpu_ids={args.gpu_ids_list} | "
        f"max_workers={args.max_workers} | parallel={args.parallel}"
    )
    run_pipeline(args)


if __name__ == "__main__":
    main()
