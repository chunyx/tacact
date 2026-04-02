#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tacact.models import ModelFactory
from tacact.benchmark_common import (
    create_optimized_dataset,
    get_device,
    split_indices_3way,
    warmup_cache,
)
from tacact.utils import evaluate_torch, parse_model_list, per_class_prf, set_seed, subset_to_numpy, train_torch_model

TRADITIONAL_SPACE: Dict[str, Dict[str, List[Any]]] = {
    "svm": {
        "C": [1.0, 3.0, 10.0, 30.0, 100.0],
        "gamma": ["scale", 1e-3, 1e-2, 5e-2],
    },
    "randomforest": {
        "n_estimators": [200, 300, 500, 700],
        "max_depth": [None, 20, 30, 40],
        "min_samples_leaf": [1, 2, 4],
    },
    "xgboost": {
        "n_estimators": [200, 300, 500],
        "max_depth": [6, 8, 10],
        "learning_rate": [0.03, 0.05, 0.1],
        "subsample": [0.7, 0.8, 1.0],
        "colsample_bytree": [0.7, 0.8, 1.0],
    },
}

DEEP_SPACE: Dict[str, Dict[str, List[Any]]] = {
    "lenet": {
        "lr": [3e-4, 1e-3, 3e-3],
        "weight_decay": [1e-5, 1e-4, 1e-3],
    },
    "alexnet": {
        "lr": [1e-4, 3e-4, 1e-3],
        "weight_decay": [1e-5, 1e-4, 1e-3],
    },
    "resnet18": {
        "lr": [1e-4, 3e-4, 1e-3],
        "weight_decay": [1e-5, 1e-4, 1e-3],
    },
    "mobilenet_v2": {
        "lr": [1e-4, 3e-4, 1e-3],
        "weight_decay": [1e-5, 1e-4, 1e-3],
    },
    "efficientnet_b0": {
        "lr": [1e-4, 3e-4, 1e-3],
        "weight_decay": [1e-5, 1e-4, 1e-3],
    },
    "cnn_lstm": {
        "lstm_hidden": [128, 256, 384],
        "lr": [1e-4, 3e-4, 1e-3],
        "weight_decay": [1e-5, 1e-4, 1e-3],
    },
    "lstm": {
        "hidden_size": [128, 256, 384],
        "num_layers": [1, 2, 3],
        "dropout": [0.1, 0.3, 0.5],
        "lr": [1e-4, 3e-4, 1e-3],
        "weight_decay": [1e-5, 1e-4, 1e-3],
    },
    "tcn": {
        "num_channels": [128, 256, 384],
        "dropout": [0.1, 0.2, 0.3],
        "lr": [1e-4, 3e-4, 5e-4],
        "weight_decay": [1e-5, 1e-4, 1e-3],
    },
    "vit": {
        "dim": [192, 256, 384],
        "depth": [3, 4, 5],
        "heads": [4, 6, 8],
        "patch_size": [8, 16],
        "dropout": [0.1, 0.15, 0.2, 0.25],
        "lr": [3e-5, 5e-5, 1e-4, 2e-4],
        "weight_decay": [0.01, 0.05, 0.1],
    },
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
            max_depth=cfg["max_depth"],
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=Path, required=True)
    parser.add_argument("--cache_dir", type=Path, default=Path(".cache_tacact_n80_front"))
    parser.add_argument("--output_dir", type=Path, default=Path("outputs_tuning"))
    parser.add_argument("--split_mode", choices=["subject", "random"], default="subject")
    parser.add_argument("--clip_mode", choices=["front", "center", "weighted_center"], default="weighted_center")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_ratio", type=float, default=0.7)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--traditional_models", type=str, default="SVM,RandomForest,XGBoost")
    parser.add_argument("--deep_models", type=str,
                        default="LeNet,AlexNet,ResNet18,MobileNet_V2,EfficientNet_B0,LSTM,CNN_LSTM,TCN,ViT")
    parser.add_argument("--trials_traditional", type=int, default=16)
    parser.add_argument("--trials_deep", type=int, default=8)
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

    traditional_models = parse_model_list(args.traditional_models, ["SVM", "RandomForest", "XGBoost"])
    deep_models = parse_model_list(args.deep_models,
                                   ["LeNet", "AlexNet", "ResNet18", "MobileNet_V2", "EfficientNet_B0", "LSTM",
                                    "CNN_LSTM", "TCN", "ViT"])

    x_train, y_train = subset_to_numpy(dataset, train_set)
    x_val, y_val = subset_to_numpy(dataset, val_set)

    trial_rows: List[Dict[str, Any]] = []
    best_rows: List[Dict[str, Any]] = []

    # 🔧 修复1：将多维矩阵展平提到循环外部，全局只执行一次！极其省时省内存！
    x_train_flat = x_train.reshape(x_train.shape[0], -1)
    x_val_flat = x_val.reshape(x_val.shape[0], -1)

    for model_name in traditional_models:
        n = model_name.lower()
        space = TRADITIONAL_SPACE.get(n)
        if not space:
            print(f"[WARN] No tuning space for traditional model {model_name}, skipped.")
            continue

        # ⚠️ 友情提示：SVM 在高维特征下训练极慢
        if n == "svm":
            print("  [WARN] SVM 在 8万维特征下计算极慢，请耐心等待...")

        print(f"\n[TUNE] Traditional: {model_name}")
        best_acc = -1.0
        best_cfg: Dict[str, Any] | None = None
        best_f1 = -1.0

        for i in range(args.trials_traditional):
            cfg = sample_cfg(space, rng)
            try:
                # 🔧 修复2：XGBoost 标签防抖，确保从 0 开始
                y_train_fit = y_train
                y_val_fit = y_val
                if n == "xgboost" and np.min(y_train) > 0:
                    y_train_fit = y_train - np.min(y_train)
                    y_val_fit = y_val - np.min(y_val)

                clf = build_traditional_with_cfg(model_name, cfg, args.seed + i)
                clf.fit(x_train_flat, y_train_fit)
                pred = clf.predict(x_val_flat)

                # 预测完还原标签偏移
                if n == "xgboost" and np.min(y_train) > 0:
                    pred = pred + np.min(y_train)

                acc = float((pred == y_val).mean())
                _, _, f1c = per_class_prf(y_val, pred, n_classes=12)
                f1 = float(np.nanmean(f1c))
            except Exception as e:
                print(f"  [WARN] {model_name} trial {i + 1} failed: {e}")
                continue

            trial_rows.append({
                "family": "traditional",
                "model": model_name,
                "trial": i + 1,
                "val_acc": acc,
                "val_f1": f1,
                **cfg,
            })
            if acc > best_acc:
                best_acc, best_f1, best_cfg = acc, f1, cfg
            print(f"  trial {i + 1}/{args.trials_traditional}: acc={acc * 100:.2f}%")

        if best_cfg is not None:
            best_rows.append({
                "family": "traditional",
                "model": model_name,
                "best_val_acc": best_acc,
                "best_val_f1": best_f1,
                **best_cfg,
            })

    for model_name in deep_models:
        n = model_name.lower()
        space = DEEP_SPACE.get(n)
        if not space:
            print(f"[WARN] No tuning space for deep model {model_name}, skipped.")
            continue
        print(f"\n[TUNE] Deep: {model_name}")

        # 🚨 智能Batch Size分配：防止大模型OOM
        model_batch_sizes = {
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
        safe_batch_size = model_batch_sizes.get(n, 8)

        model_train_loader = DataLoader(
            train_set, batch_size=safe_batch_size, shuffle=True,
            num_workers=args.num_workers, pin_memory=True,
            persistent_workers=args.num_workers > 0
        )
        model_val_loader = DataLoader(
            val_set, batch_size=safe_batch_size, shuffle=False,
            num_workers=args.num_workers, pin_memory=True,
            persistent_workers=args.num_workers > 0
        )
        print(f"  使用安全batch_size={safe_batch_size} for {model_name}")

        best_acc = -1.0
        best_cfg: Dict[str, Any] | None = None
        best_f1 = -1.0

        for i in range(args.trials_deep):
            # 🔧 修复3：用 while 循环代替 continue，保证抽满配置不浪费额度
            while True:
                cfg = sample_cfg(space, rng)
                if n != "vit" or int(cfg["dim"]) % int(cfg["heads"]) == 0:
                    break

            set_seed(args.seed + i)
            model_kwargs = {k: v for k, v in cfg.items() if
                            k in {"dim", "depth", "heads", "patch_size", "dropout", "num_channels", "lstm_hidden",
                                  "hidden_size", "num_layers"}}
            lr = float(cfg.get("lr", 1e-3))
            weight_decay = float(cfg.get("weight_decay", 1e-5))

            try:
                model, _ = ModelFactory.build_torch(model_name, **model_kwargs)

                # 🛡️ OOM保护机制：捕获显存溢出
                _ = train_torch_model(
                    model=model,
                    train_loader=model_train_loader,
                    val_loader=model_val_loader,
                    epochs=args.epochs,
                    device=device,
                    patience=args.patience,
                    lr_override=lr,
                    weight_decay_override=weight_decay,
                )
                acc, f1 = evaluate_torch(model, model_val_loader, device)

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"  [WARN] {model_name} trial {i + 1} OOM 显存溢出，跳过此配置")
                    torch.cuda.empty_cache()  # 清理显存碎片
                    continue
                else:
                    print(f"  [WARN] {model_name} trial {i + 1} RuntimeError: {e}")
                    continue
            except Exception as e:
                print(f"  [WARN] {model_name} trial {i + 1} failed: {e}")
                continue

            trial_rows.append({
                "family": "deep",
                "model": model_name,
                "trial": i + 1,
                "val_acc": acc,
                "val_f1": f1,
                **cfg,
            })
            if acc > best_acc:
                best_acc, best_f1, best_cfg = acc, f1, cfg
            print(f"  trial {i + 1}/{args.trials_deep}: acc={acc * 100:.2f}%")

        if best_cfg is not None:
            best_rows.append({
                "family": "deep",
                "model": model_name,
                "best_val_acc": best_acc,
                "best_val_f1": best_f1,
                **best_cfg,
            })

    if not trial_rows:
        raise RuntimeError("No successful tuning trial. Check dependencies and dataset path.")

    trial_df = pd.DataFrame(trial_rows).sort_values(["family", "model", "val_acc"], ascending=[True, True, False])
    best_df = pd.DataFrame(best_rows).sort_values(["family", "best_val_acc"], ascending=[True, False])

    trial_path = args.output_dir / "all_models_tuning_trials.csv"
    best_path = args.output_dir / "all_models_best_configs.csv"
    config_path = args.output_dir / "best_model_configs.json"
    trial_df.to_csv(trial_path, index=False)
    best_df.to_csv(best_path, index=False)
    config_payload = {
        "meta": {
            "generated_by": "tune_all_models.py",
            "seed": args.seed,
            "split_mode": args.split_mode,
            "epochs": args.epochs,
            "patience": args.patience,
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
    print("\nBest configs preview:")
    print(best_df.to_string(index=False))


if __name__ == "__main__":
    main()
