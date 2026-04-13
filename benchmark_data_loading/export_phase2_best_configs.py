#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd


HP_KEYS = {
    "lr",
    "batch_size",
    "weight_decay",
    "input_proj_dim",
    "hidden_size",
    "num_layers",
    "dropout",
    "use_last_only",
    "lstm_hidden",
    "num_channels",
    "dim",
    "depth",
    "heads",
    "patch_size",
}

RUNTIME_KEYS = {
    "host",
    "gpu_local_id",
    "cuda_visible_devices",
    "pid",
    "phase",
    "trial_id",
    "seed",
    "task_id",
    "status",
    "status_reason",
}


def _pick_cfg_from_df(df: pd.DataFrame) -> Dict[str, Any]:
    if df.empty:
        return {}
    work = df.copy()
    if "status" in work.columns:
        ok = work[work["status"].astype(str) == "ok"]
        if not ok.empty:
            work = ok
    sort_cols: List[str] = []
    ascending: List[bool] = []
    if "best_val_f1" in work.columns:
        sort_cols.append("best_val_f1")
        ascending.append(False)
    if "best_val_acc" in work.columns:
        sort_cols.append("best_val_acc")
        ascending.append(False)
    if sort_cols:
        work = work.sort_values(sort_cols, ascending=ascending)
    row = work.iloc[0].to_dict()
    cfg: Dict[str, Any] = {}
    for k, v in row.items():
        if k in RUNTIME_KEYS:
            continue
        if k in HP_KEYS and pd.notna(v):
            cfg[k] = v.item() if hasattr(v, "item") else v
    return cfg


def from_hpo_root(hpo_root: Path) -> Dict[str, Any]:
    phase2_root = hpo_root / "phase2"
    if not phase2_root.exists():
        raise FileNotFoundError(f"phase2 directory not found under: {hpo_root}")
    deep: Dict[str, Dict[str, Any]] = {}
    for model_dir in sorted(p for p in phase2_root.iterdir() if p.is_dir()):
        csv_path = model_dir / "phase2_results.csv"
        if not csv_path.exists():
            continue
        df = pd.read_csv(csv_path)
        cfg = _pick_cfg_from_df(df)
        if cfg:
            deep[model_dir.name] = {"params": cfg}
    return {"traditional": {}, "deep": deep}


def from_meta_json(meta_json: Path) -> Dict[str, Any]:
    payload = json.loads(meta_json.read_text(encoding="utf-8"))
    models = payload.get("models", {})
    deep: Dict[str, Dict[str, Any]] = {}
    for model_name, item in models.items():
        best = item.get("best_config", {}) if isinstance(item, dict) else {}
        if not isinstance(best, dict):
            continue
        cfg: Dict[str, Any] = {}
        for k, v in best.items():
            if k in RUNTIME_KEYS:
                continue
            if k in HP_KEYS and v is not None:
                cfg[k] = v
        if cfg:
            deep[model_name] = {"params": cfg}
    return {"traditional": {}, "deep": deep}


def main() -> None:
    parser = argparse.ArgumentParser(description="Export phase2 best configs for experiment_tacact.py")
    parser.add_argument("--hpo_root", type=Path, default=None, help="HPO output root containing phase2/<model>/phase2_results.csv")
    parser.add_argument("--meta_json", type=Path, default=None, help="hpo_pipeline_meta.json path (fallback source)")
    parser.add_argument("--output", type=Path, required=True, help="Output JSON path")
    args = parser.parse_args()

    if args.hpo_root is None and args.meta_json is None:
        raise ValueError("Provide either --hpo_root or --meta_json.")
    if args.hpo_root is not None and args.meta_json is not None:
        raise ValueError("Provide only one source: --hpo_root OR --meta_json.")

    if args.hpo_root is not None:
        data = from_hpo_root(args.hpo_root)
        src = str(args.hpo_root)
    else:
        data = from_meta_json(args.meta_json)
        src = str(args.meta_json)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    n_deep = len(data.get("deep", {}))
    print(f"[BestConfig Export] source={src} deep_models={n_deep} -> {args.output}")


if __name__ == "__main__":
    main()

