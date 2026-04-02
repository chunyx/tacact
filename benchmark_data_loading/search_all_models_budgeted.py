#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tacact.benchmark_common import create_optimized_dataset, get_device, split_indices_3way, warmup_cache
from tacact.models import ModelFactory
from tacact.utils import (
    count_parameters,
    count_sklearn_params,
    evaluate_torch,
    parse_model_list,
    per_class_prf,
    set_seed,
    subset_to_numpy,
    train_torch_model,
)


TRADITIONAL_SPACE: Dict[str, Dict[str, List[Any]]] = {
    "svm": {
        "C": [3.0, 10.0, 30.0],
        "gamma": ["scale", "auto", 1e-3],
    },
    "randomforest": {
        "n_estimators": [200, 300, 500],
        "max_depth": [16, 24, 32],
        "min_samples_leaf": [1, 2, 4],
    },
    "xgboost": {
        "n_estimators": [150, 250, 350],
        "max_depth": [4, 6, 8],
        "learning_rate": [0.03, 0.05, 0.1],
        "subsample": [0.7, 0.9],
        "colsample_bytree": [0.7, 0.9],
    },
}


DEEP_SPACE: Dict[str, Dict[str, List[Any]]] = {
    "lenet": {
        "lr": [3e-4, 1e-3, 3e-3],
        "weight_decay": [1e-5, 1e-4],
    },
    "alexnet": {
        "lr": [1e-4, 3e-4, 1e-3],
        "weight_decay": [1e-5, 1e-4],
    },
    "resnet18": {
        "lr": [3e-4, 1e-3],
        "weight_decay": [1e-5, 1e-4],
    },
    "mobilenet_v2": {
        "lr": [3e-4, 1e-3],
        "weight_decay": [1e-5, 1e-4, 1e-3],
    },
    "efficientnet_b0": {
        "lr": [3e-4, 1e-3],
        "weight_decay": [1e-5, 1e-4, 5e-4],
    },
    "lstm": {
        "input_proj_dim": [256, 512],
        "hidden_size": [128, 256],
        "num_layers": [1, 2],
        "dropout": [0.3, 0.5],
        "use_last_only": [False],
        "lr": [3e-4, 1e-3],
        "weight_decay": [1e-5, 1e-4],
    },
    "cnn_lstm": {
        "lstm_hidden": [128, 256],
        "lr": [1e-4, 3e-4, 1e-3],
        "weight_decay": [1e-5, 1e-4],
    },
    "tcn": {
        "num_channels": [256, 512],
        "dropout": [0.0, 0.1],
        "lr": [1e-4, 3e-4, 5e-4],
        "weight_decay": [1e-5, 1e-4],
    },
    "vit": {
        "dim": [192, 256],
        "depth": [3, 4],
        "heads": [4, 8],
        "patch_size": [16],
        "dropout": [0.1, 0.25],
        "lr": [5e-5, 1e-4],
        "weight_decay": [0.01, 0.05],
    },
}


SAFE_BATCH_SIZE = {
    "lenet": 32,
    "alexnet": 16,
    "resnet18": 16,
    "mobilenet_v2": 16,
    "efficientnet_b0": 16,
    "lstm": 16,
    "cnn_lstm": 8,
    "tcn": 8,
    "vit": 4,
}


def sample_cfg(space: Dict[str, List[Any]], rng: random.Random) -> Dict[str, Any]:
    return {k: rng.choice(v) for k, v in space.items()}


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


def build_traditional_with_cfg(name: str, cfg: Dict[str, Any], seed: int):
    n = name.lower()
    if n == "svm":
        from sklearn.svm import SVC

        return SVC(kernel="rbf", C=float(cfg["C"]), gamma=cfg["gamma"])
    if n == "randomforest":
        from sklearn.ensemble import RandomForestClassifier

        return RandomForestClassifier(
            n_estimators=int(cfg["n_estimators"]),
            max_depth=int(cfg["max_depth"]),
            min_samples_leaf=int(cfg["min_samples_leaf"]),
            random_state=seed,
            n_jobs=-1,
        )
    if n == "xgboost":
        import xgboost as xgb

        return xgb.XGBClassifier(
            n_estimators=int(cfg["n_estimators"]),
            max_depth=int(cfg["max_depth"]),
            learning_rate=float(cfg["learning_rate"]),
            subsample=float(cfg["subsample"]),
            colsample_bytree=float(cfg["colsample_bytree"]),
            objective="multi:softmax",
            num_class=12,
            n_jobs=-1,
            random_state=seed,
        )
    raise ValueError(f"Unknown traditional model: {name}")


def profile_deep_trial(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    lr: float,
    weight_decay: float,
    batches: int = 2,
) -> float:
    model = model.to(device)
    model.train()
    criterion = nn.CrossEntropyLoss()
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    used = 0
    start = time.perf_counter()
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        opt.zero_grad(set_to_none=True)
        loss = criterion(model(x), y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt.step()
        used += 1
        if used >= batches:
            break
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    if used == 0:
        return 0.0
    return elapsed / used * len(loader)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=Path, required=True)
    parser.add_argument("--cache_dir", type=Path, default=Path(".cache_tacact_n80_front"))
    parser.add_argument("--output_dir", type=Path, default=Path("outputs_tuning_budgeted"))
    parser.add_argument("--split_mode", choices=["subject", "random"], default="subject")
    parser.add_argument("--clip_mode", choices=["front", "center", "weighted_center"], default="weighted_center")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_ratio", type=float, default=0.7)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--traditional_models", type=str, default="SVM,RandomForest,XGBoost")
    parser.add_argument(
        "--deep_models",
        type=str,
        default="LeNet,AlexNet,ResNet18,MobileNet_V2,EfficientNet_B0,LSTM,CNN_LSTM,TCN,ViT",
    )
    parser.add_argument("--trials_traditional", type=int, default=10)
    parser.add_argument("--trials_deep", type=int, default=6)
    parser.add_argument("--max_params_m", type=float, default=20.0)
    parser.add_argument("--max_trial_minutes", type=float, default=12.0)
    parser.add_argument("--max_model_minutes", type=float, default=45.0)
    parser.add_argument("--profile_batches", type=int, default=2)
    parser.add_argument("--skip_cache_warmup", action="store_true")
    parser.add_argument("--no_preload", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    rng = random.Random(args.seed)
    device = get_device()
    print(f"Device: {device}")

    dataset = create_optimized_dataset(
        args.data_root,
        n_frames=80,
        threshold=20.0,
        clip_mode=args.clip_mode,
        cache_dir=args.cache_dir,
        preload_cache=not args.no_preload,
    )
    print(f"Dataset: {len(dataset)} samples")
    if not args.skip_cache_warmup:
        warmup_cache(
            dataset,
            batch_size=128,
            num_workers=0,
            shuffle=False,
            pin_memory=False,
            max_batches=1,
            use_tqdm=False,
        )

    train_idx, val_idx, _ = split_indices_3way(
        dataset,
        split_mode=args.split_mode,
        seed=args.seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
    )
    train_set = Subset(dataset, train_idx)
    val_set = Subset(dataset, val_idx)
    print(f"Split sizes: train={len(train_set)} val={len(val_set)}")

    traditional_models = parse_model_list(args.traditional_models, ["SVM", "RandomForest", "XGBoost"])
    deep_models = parse_model_list(
        args.deep_models,
        ["LeNet", "AlexNet", "ResNet18", "MobileNet_V2", "EfficientNet_B0", "LSTM", "CNN_LSTM", "TCN", "ViT"],
    )

    x_train, y_train = subset_to_numpy(dataset, train_set)
    x_val, y_val = subset_to_numpy(dataset, val_set)
    x_train_flat = x_train.reshape(x_train.shape[0], -1)
    x_val_flat = x_val.reshape(x_val.shape[0], -1)

    trial_rows: List[Dict[str, Any]] = []
    best_rows: List[Dict[str, Any]] = []

    for model_name in traditional_models:
        n = model_name.lower()
        space = TRADITIONAL_SPACE.get(n)
        if not space:
            print(f"[WARN] No tuning space for {model_name}, skipped.")
            continue

        print(f"\n[TUNE] Traditional: {model_name}")
        model_start = time.perf_counter()
        best_acc = -1.0
        best_f1 = -1.0
        best_cfg: Dict[str, Any] | None = None

        for i in range(args.trials_traditional):
            if time.perf_counter() - model_start > args.max_model_minutes * 60.0:
                print(f"  [STOP] {model_name} 超出模型总预算，提前结束。")
                break
            cfg = sample_cfg(space, rng)
            try:
                y_train_fit = y_train
                y_val_fit = y_val
                if n == "xgboost" and int(np.min(y_train)) > 0:
                    offset = int(np.min(y_train))
                    y_train_fit = y_train - offset
                    y_val_fit = y_val - offset
                else:
                    offset = 0

                st = time.perf_counter()
                clf = build_traditional_with_cfg(model_name, cfg, args.seed + i)
                clf.fit(x_train_flat, y_train_fit)
                fit_seconds = time.perf_counter() - st
                if fit_seconds > args.max_trial_minutes * 60.0:
                    print(f"  [SKIP] {model_name} trial {i + 1} 超时 {fit_seconds:.1f}s")
                    continue

                pred = clf.predict(x_val_flat)
                if offset:
                    pred = pred + offset
                acc = float((pred == y_val).mean())
                _, _, f1c = per_class_prf(y_val, pred, n_classes=12)
                f1 = float(np.nanmean(f1c))
                params = float(count_sklearn_params(clf))
            except Exception as e:
                print(f"  [WARN] {model_name} trial {i + 1} failed: {e}")
                continue

            trial_rows.append(
                {
                    "family": "traditional",
                    "model": model_name,
                    "trial": i + 1,
                    "val_acc": acc,
                    "val_f1": f1,
                    "fit_seconds": fit_seconds,
                    "params": params,
                    "params_m": params / 1e6,
                    "status": "ok",
                    **cfg,
                }
            )
            if acc > best_acc:
                best_acc, best_f1, best_cfg = acc, f1, cfg
            print(f"  trial {i + 1}/{args.trials_traditional}: acc={acc * 100:.2f}% time={fit_seconds:.1f}s")

        if best_cfg is not None:
            best_rows.append(
                {
                    "family": "traditional",
                    "model": model_name,
                    "best_val_acc": best_acc,
                    "best_val_f1": best_f1,
                    **best_cfg,
                }
            )

    for model_name in deep_models:
        n = model_name.lower()
        space = DEEP_SPACE.get(n)
        if not space:
            print(f"[WARN] No tuning space for {model_name}, skipped.")
            continue

        print(f"\n[TUNE] Deep: {model_name}")
        model_start = time.perf_counter()
        safe_batch_size = SAFE_BATCH_SIZE.get(n, 8)
        train_loader = DataLoader(
            train_set,
            batch_size=safe_batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
            persistent_workers=args.num_workers > 0,
        )
        val_loader = DataLoader(
            val_set,
            batch_size=safe_batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
            persistent_workers=args.num_workers > 0,
        )
        print(f"  使用安全 batch_size={safe_batch_size}")

        best_acc = -1.0
        best_f1 = -1.0
        best_cfg: Dict[str, Any] | None = None

        for i in range(args.trials_deep):
            if time.perf_counter() - model_start > args.max_model_minutes * 60.0:
                print(f"  [STOP] {model_name} 超出模型总预算，提前结束。")
                break

            while True:
                cfg = sample_cfg(space, rng)
                if n != "vit" or int(cfg["dim"]) % int(cfg["heads"]) == 0:
                    break

            model_kwargs = {
                k: v
                for k, v in cfg.items()
                if k
                in {
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
            }
            lr = float(cfg.get("lr", 1e-3))
            weight_decay = float(cfg.get("weight_decay", 1e-5))

            try:
                set_seed(args.seed + i)
                model, _ = ModelFactory.build_torch(model_name, **model_kwargs)
                params = float(count_parameters(model))
                if params > args.max_params_m * 1e6:
                    print(f"  [SKIP] trial {i + 1} 参数量 {params / 1e6:.2f}M 超过上限")
                    trial_rows.append(
                        {
                            "family": "deep",
                            "model": model_name,
                            "trial": i + 1,
                            "status": "skip_params",
                            "params": params,
                            "params_m": params / 1e6,
                            **cfg,
                        }
                    )
                    continue

                est_epoch_seconds = profile_deep_trial(
                    model=model,
                    loader=train_loader,
                    device=device,
                    lr=lr,
                    weight_decay=weight_decay,
                    batches=max(1, args.profile_batches),
                )
                est_trial_seconds = est_epoch_seconds * min(args.epochs, args.patience + 3)
                if est_trial_seconds > args.max_trial_minutes * 60.0:
                    print(f"  [SKIP] trial {i + 1} 预计训练 {est_trial_seconds / 60.0:.1f} 分钟，超过上限")
                    trial_rows.append(
                        {
                            "family": "deep",
                            "model": model_name,
                            "trial": i + 1,
                            "status": "skip_time",
                            "params": params,
                            "params_m": params / 1e6,
                            "est_epoch_seconds": est_epoch_seconds,
                            "est_trial_seconds": est_trial_seconds,
                            **cfg,
                        }
                    )
                    del model
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                    continue

                model, _ = ModelFactory.build_torch(model_name, **model_kwargs)
                history = train_torch_model(
                    model=model,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    epochs=args.epochs,
                    device=device,
                    patience=args.patience,
                    lr_override=lr,
                    weight_decay_override=weight_decay,
                )
                acc, f1 = evaluate_torch(model, val_loader, device)
                train_seconds = float(history.get("cum_time_s", [0.0])[-1]) if history.get("cum_time_s") else 0.0
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"  [WARN] {model_name} trial {i + 1} OOM，跳过。")
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                    continue
                print(f"  [WARN] {model_name} trial {i + 1} RuntimeError: {e}")
                continue
            except Exception as e:
                print(f"  [WARN] {model_name} trial {i + 1} failed: {e}")
                continue

            trial_rows.append(
                {
                    "family": "deep",
                    "model": model_name,
                    "trial": i + 1,
                    "val_acc": acc,
                    "val_f1": f1,
                    "train_seconds": train_seconds,
                    "est_epoch_seconds": est_epoch_seconds,
                    "est_trial_seconds": est_trial_seconds,
                    "params": params,
                    "params_m": params / 1e6,
                    "status": "ok",
                    **cfg,
                }
            )
            if acc > best_acc:
                best_acc, best_f1, best_cfg = acc, f1, cfg
            print(
                f"  trial {i + 1}/{args.trials_deep}: "
                f"acc={acc * 100:.2f}% params={params / 1e6:.2f}M time={train_seconds / 60.0:.1f}min"
            )

            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

        if best_cfg is not None:
            best_rows.append(
                {
                    "family": "deep",
                    "model": model_name,
                    "best_val_acc": best_acc,
                    "best_val_f1": best_f1,
                    **best_cfg,
                }
            )

    if not trial_rows:
        raise RuntimeError("No successful tuning trial. Try relaxing the search budget.")

    trial_df = pd.DataFrame(trial_rows)
    best_df = pd.DataFrame(best_rows)

    trial_path = args.output_dir / "budgeted_tuning_trials.csv"
    best_path = args.output_dir / "budgeted_best_configs.csv"
    config_path = args.output_dir / "best_model_configs.json"
    trial_df.to_csv(trial_path, index=False)
    best_df.to_csv(best_path, index=False)

    config_payload = {
        "meta": {
            "generated_by": "search_all_models_budgeted.py",
            "seed": args.seed,
            "split_mode": args.split_mode,
            "epochs": args.epochs,
            "patience": args.patience,
            "max_params_m": args.max_params_m,
            "max_trial_minutes": args.max_trial_minutes,
            "max_model_minutes": args.max_model_minutes,
        },
        "traditional": {},
        "deep": {},
    }
    for row in best_rows:
        family = str(row["family"])
        model_name = str(row["model"])
        payload = {k: v for k, v in row.items() if k not in {"family", "model", "best_val_acc", "best_val_f1"}}
        config_payload[family][model_name] = {
            "best_val_acc": float(row["best_val_acc"]),
            "best_val_f1": float(row["best_val_f1"]),
            "params": sanitize_config(payload),
        }
    config_path.write_text(json.dumps(config_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved trials: {trial_path}")
    print(f"Saved best configs: {best_path}")
    print(f"Saved reusable config: {config_path}")
    if not best_df.empty:
        print("\nBest configs preview:")
        print(best_df.sort_values(["family", "best_val_acc"], ascending=[True, False]).to_string(index=False))


if __name__ == "__main__":
    main()
