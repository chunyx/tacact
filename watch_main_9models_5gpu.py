#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List


def _load_status(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {
            "status": "missing",
            "gpu_id": path.stem.replace("gpu", ""),
            "queue_total": 0,
            "queue_completed": 0,
            "current_model": None,
            "current_model_index": 0,
            "current_epoch": 0,
            "total_epochs": 0,
            "latest_val_f1": None,
            "queue_models": [],
        }
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {
            "status": "unreadable",
            "gpu_id": path.stem.replace("gpu", ""),
            "queue_total": 0,
            "queue_completed": 0,
            "current_model": None,
            "current_model_index": 0,
            "current_epoch": 0,
            "total_epochs": 0,
            "latest_val_f1": None,
            "queue_models": [],
        }


def _fmt_f1(x: Any) -> str:
    try:
        v = float(x)
        if v != v:
            return "-"
        return f"{v:.4f}"
    except Exception:
        return "-"


def _pid_alive(pid: Any) -> bool:
    try:
        pid_int = int(pid)
    except Exception:
        return False
    if pid_int <= 0:
        return False
    return os.path.exists(f"/proc/{pid_int}")


def _normalize_runtime_status(d: Dict[str, Any], stale_sec: float) -> Dict[str, Any]:
    out = dict(d)
    status = str(out.get("status", "unknown"))
    pid = out.get("pid")
    last_update_ts = float(out.get("last_update_ts", 0) or 0)
    now_ts = time.time()
    alive = _pid_alive(pid)

    if status == "running" and not alive:
        if last_update_ts > 0 and now_ts - last_update_ts >= stale_sec:
            out["status"] = "stale_dead"
        else:
            out["status"] = "dead"
    elif status == "starting" and not alive:
        out["status"] = "failed"
    return out


def _gpu_line(d: Dict[str, Any]) -> str:
    gid = str(d.get("gpu_id", "?"))
    status = str(d.get("status", "unknown"))
    model = d.get("current_model")
    q_done = int(d.get("queue_completed", 0) or 0)
    q_total = int(d.get("queue_total", 0) or 0)
    m_idx = int(d.get("current_model_index", 0) or 0)
    ep = int(d.get("current_epoch", 0) or 0)
    ep_total = int(d.get("total_epochs", 0) or 0)
    f1 = _fmt_f1(d.get("latest_val_f1"))

    if status in {"done", "failed", "terminated", "stale_dead", "dead", "queued", "missing", "unreadable"} and not model:
        return (
            f"GPU {gid:<2} | {'-':<16} | model -/-   | epoch -/- | val_f1=-      | "
            f"status={status}"
        )

    model_name = str(model) if model else "IDLE"
    model_pos = f"{m_idx}/{max(q_total,1)}" if m_idx > 0 else f"{q_done}/{max(q_total,1)}"
    return (
        f"GPU {gid:<2} | {model_name:<16} | model {model_pos:<5} | "
        f"epoch {ep:>2}/{ep_total:<2} | val_f1={f1:<6} | status={status}"
    )


def _global_counts(statuses: List[Dict[str, Any]]) -> Dict[str, int]:
    total = 0
    completed = 0
    running = 0
    for d in statuses:
        qt = int(d.get("queue_total", 0) or 0)
        qc = int(d.get("queue_completed", 0) or 0)
        total += qt
        completed += min(qc, qt)
        st = str(d.get("status", ""))
        if st == "running":
            running += 1
    pending = max(total - completed - running, 0)
    return {"total": total, "completed": completed, "running": running, "pending": pending}


def _pending_text(statuses: List[Dict[str, Any]]) -> str:
    parts: List[str] = []
    for d in statuses:
        gid = str(d.get("gpu_id", "?"))
        q = [str(x) for x in d.get("queue_models", [])]
        done = int(d.get("queue_completed", 0) or 0)
        current = d.get("current_model")
        rem = q[done:]
        if rem and current and rem and rem[0] == str(current):
            rem = rem[1:]
        if rem:
            parts.append(f"GPU {gid} -> {', '.join(rem)}")
    return " | ".join(parts) if parts else "-"


def main() -> None:
    parser = argparse.ArgumentParser(description="Live watcher for multi-GPU main experiment status")
    parser.add_argument("--run_root", type=Path, required=True)
    parser.add_argument("--gpu_count", type=int, default=5)
    parser.add_argument("--refresh_sec", type=float, default=1.0)
    parser.add_argument("--stale_sec", type=float, default=120.0)
    args = parser.parse_args()

    status_dir = args.run_root / "status"
    gpu_files = [status_dir / f"gpu{i}.json" for i in range(max(1, args.gpu_count))]

    while True:
        statuses = []
        for i, fp in enumerate(gpu_files):
            d = _load_status(fp)
            d.setdefault("gpu_id", i)
            statuses.append(_normalize_runtime_status(d, stale_sec=float(args.stale_sec)))

        g = _global_counts(statuses)
        header = (
            f"Global progress: {g['completed']}/{g['total']} models completed | "
            f"{g['running']} running | {g['pending']} pending"
        )
        lines = [header, ""]
        for d in statuses:
            lines.append(_gpu_line(d))
        lines.append("")
        lines.append(f"Pending queue: {_pending_text(statuses)}")

        print("\x1b[H\x1b[J" + "\n".join(lines), flush=True)

        all_done = all(
            str(d.get("status", "")) in {"done", "failed", "terminated", "stale_dead", "dead", "missing", "unreadable"}
            for d in statuses
        )
        if all_done and g["running"] == 0:
            break
        time.sleep(max(0.2, args.refresh_sec))


if __name__ == "__main__":
    main()
