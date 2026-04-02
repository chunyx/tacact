#!/usr/bin/env python3
"""
CNNLSTM 超参数搜索脚本：在 TacAct 数据集上搜索 CNNLSTM 最佳配置。
用法：python search_cnnlstm_params.py --data_root "你的数据路径" [--mode grid|random] [--n_trials 20]
"""
from __future__ import annotations

import argparse
import itertools
import random
from pathlib import Path

import numpy as np
from tqdm import tqdm
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import models

from tacact.benchmark_common import (
    create_optimized_dataset,
    get_device,
    split_indices_train_val,
    warmup_cache,
)
from tacact.utils import evaluate_torch, set_seed

# ==========================================
# 1. 搜索空间定义
# ==========================================
PARAM_GRID = {
    "lstm_hidden": [128, 256, 512],
    "lstm_layers": [1, 2],  # 减少一层，防止过拟合和降低计算量
    "lstm_dropout": [0.0, 0.2, 0.4],

    # 移除了 ResNet34，因为 32x32 实在不需要那么深的网络，反而容易崩溃
    "backbone": ["resnet18", "mobilenet_v2", "efficientnet_b0"],

    "lr": [1e-4, 3e-4, 5e-4],  # 调参更为保守
    "weight_decay": [1e-4, 1e-3, 5e-3],
    "dropout": [0.0, 0.2, 0.5],
    "use_last_only": [True, False],
}


def sample_random_configs(n: int, seed: int = 42) -> list[dict]:
    rng = random.Random(seed)
    configs = []
    for _ in range(n * 2):
        c = {k: rng.choice(v) for k, v in PARAM_GRID.items()}
        if c not in configs:
            configs.append(c)
            if len(configs) >= n:
                break
    return configs


# ==========================================
# 2. 定制版 CNN-LSTM (修复了特征塌缩问题)
# ==========================================
class CustomCNNLSTM(torch.nn.Module):
    def __init__(self, config: dict, num_classes: int = 12):
        super().__init__()

        # 构建CNN骨干网络
        if config["backbone"] == "resnet18":
            backbone = models.resnet18(weights=None)
            feature_dim = 512
            # 🌟 关键修复：针对 32x32 输入，改用 3x3 卷积，stride=1 不降维！
            backbone.conv1 = torch.nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
            # 移除原来的 MaxPool2d，防止早期特征丢失
            backbone.maxpool = torch.nn.Identity()
            self.frame_extractor = torch.nn.Sequential(*list(backbone.children())[:-1])

        elif config["backbone"] == "mobilenet_v2":
            backbone = models.mobilenet_v2(weights=None)
            feature_dim = 1280
            # 🌟 关键修复：改用 stride=1
            backbone.features[0][0] = torch.nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1, bias=False)
            self.frame_extractor = backbone.features

        elif config["backbone"] == "efficientnet_b0":
            backbone = models.efficientnet_b0(weights=None)
            feature_dim = 1280
            # 🌟 关键修复：改用 stride=1
            backbone.features[0][0] = torch.nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1, bias=False)
            self.frame_extractor = backbone.features
        else:
            raise ValueError(f"Unsupported backbone: {config['backbone']}")

        self.lstm = torch.nn.LSTM(
            input_size=feature_dim,
            hidden_size=config["lstm_hidden"],
            num_layers=config["lstm_layers"],
            dropout=config["lstm_dropout"] if config["lstm_layers"] > 1 else 0.0,
            batch_first=True
        )

        self.use_last_only = config["use_last_only"]
        self.dropout = torch.nn.Dropout(config["dropout"])
        self.head = torch.nn.Linear(config["lstm_hidden"], num_classes)

        if config["backbone"] in ["mobilenet_v2", "efficientnet_b0"]:
            self.global_pool = torch.nn.AdaptiveAvgPool2d((1, 1))
        else:
            self.global_pool = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, h, w = x.shape
        x = x.view(b * t, 1, h, w)

        features = self.frame_extractor(x)

        if self.global_pool is not None:
            features = self.global_pool(features).flatten(1)
        else:
            features = features.flatten(1)

        features = features.view(b, t, -1)
        lstm_out, _ = self.lstm(features)

        if self.use_last_only:
            aggregated = lstm_out[:, -1, :]
        else:
            aggregated = torch.mean(lstm_out, dim=1)

        return self.head(self.dropout(aggregated))


def create_cnnlstm_from_config(config: dict, num_classes: int = 12):
    return CustomCNNLSTM(config, num_classes)


# ==========================================
# 3. 训练与评估逻辑 (AMP 混合精度 + 数据类型防错)
# ==========================================
def train_eval_cnnlstm(
        config: dict,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
        epochs: int = 40,
        patience: int = 8,
        gl_alpha: float = 2.0,
        show_progress: bool = True,
) -> tuple[float, float]:
    model = create_cnnlstm_from_config(config)
    model.to(device)

    opt = torch.optim.Adam(
        model.parameters(),
        lr=config["lr"],
        weight_decay=config["weight_decay"],
    )

    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode='min', factor=0.5, patience=3, verbose=False
    )
    criterion = torch.nn.CrossEntropyLoss()

    # 动态匹配设备类型，兼容 CPU/CUDA 测试
    scaler = torch.amp.GradScaler(device.type)

    report_acc_at_best_val_loss = 0.0
    best_f1 = 0.0
    best_val_loss = float("inf")
    best_epoch = -1
    best_weights = None
    pbar_epoch = tqdm(range(epochs), desc="Epoch", leave=False, disable=not show_progress)

    for ep in pbar_epoch:
        model.train()
        for x, y in train_loader:
            # 强制转换为 float32，彻底杜绝数据类型冲突报错
            x = x.to(device, dtype=torch.float32)
            y = y.to(device)
            opt.zero_grad()

            with torch.amp.autocast(device.type):
                outputs = model(x)
                loss = criterion(outputs, y)

            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(opt)
            scaler.update()

        model.eval()
        val_loss_sum = 0.0
        val_count = 0
        with torch.no_grad():
            for xv, yv in val_loader:
                xv = xv.to(device, dtype=torch.float32)
                yv = yv.to(device)
                logits = model(xv)
                bs = yv.size(0)
                batch_loss = criterion(logits, yv)
                val_loss_sum += float(batch_loss.item()) * bs
                val_count += bs
        val_loss = val_loss_sum / max(1, val_count)
        acc, f1 = evaluate_torch(model, val_loader, device)
        sched.step(val_loss)

        current_best = min(best_val_loss, val_loss)
        safe_best = max(current_best, 1e-12)
        current_gl = 100.0 * (val_loss / safe_best - 1.0)
        stop_triggered = current_gl > float(gl_alpha)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = ep + 1
            best_weights = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            report_acc_at_best_val_loss = acc
            best_f1 = f1

        pbar_epoch.set_postfix(
            val_loss=f"{val_loss:.5f}",
            best_val_loss=f"{best_val_loss:.5f}",
            gl=f"{current_gl:.3f}",
            stop=str(stop_triggered),
        )
        print(
            f"Epoch {ep + 1}/{epochs} | "
            f"ValLoss: {val_loss:.6f} | "
            f"BestValLoss: {best_val_loss:.6f} (epoch={best_epoch}) | "
            f"GL: {current_gl:.4f} | "
            f"Alpha: {float(gl_alpha):.2f} | "
            f"Stop: {stop_triggered}"
        )

        if stop_triggered:
            pbar_epoch.set_description(f"Epoch (GL_alpha stop @{ep + 1})")
            break

    if best_weights is not None:
        model.load_state_dict(best_weights)

    return report_acc_at_best_val_loss, best_f1


# ==========================================
# 4. 主函数
# ==========================================
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=Path, required=True)
    parser.add_argument("--cache_dir", type=Path, default=Path(".cache_tacact_n80_front"))
    parser.add_argument("--output", type=Path, default=Path("cnnlstm_search_results.csv"))
    parser.add_argument("--mode", choices=["grid", "random"], default="random")
    parser.add_argument("--n_trials", type=int, default=25, help="random 模式下的尝试次数")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split_mode", choices=["random", "subject"], default="subject")
    parser.add_argument("--skip_cache_warmup", action="store_true")
    parser.add_argument("--no_preload", action="store_true")
    parser.add_argument("--no_progress", action="store_true")
    args = parser.parse_args()

    # 开启 CuDNN 性能基准测试，进一步加速 CNN 运算
    torch.backends.cudnn.benchmark = True

    set_seed(args.seed)
    device = get_device()
    print(f"Device: {device}")

    dataset = create_optimized_dataset(
        args.data_root,
        n_frames=80,
        threshold=20.0,
        clip_mode="front",
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
        print("Cache ready.")

    train_idx, val_idx = split_indices_train_val(
        dataset,
        split_mode=args.split_mode,
        seed=args.seed,
        train_ratio=0.8,
    )

    train_set = Subset(dataset, train_idx)
    val_set = Subset(dataset, val_idx)

    # 针对 CNNLSTM 显存消耗大，降低 DataLoader 的 batch_size 保证安全
    train_loader = DataLoader(train_set, batch_size=8, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=8, shuffle=False, num_workers=0, pin_memory=True)

    if args.mode == "grid":
        keys = list(PARAM_GRID.keys())
        values = [PARAM_GRID[k] for k in keys]
        configs = [dict(zip(keys, v)) for v in itertools.product(*values)]
        if len(configs) > 40:
            configs = configs[:40]
        print(f"Grid search: {len(configs)} configs")
    else:
        configs = sample_random_configs(args.n_trials, args.seed)
        print(f"Random search: {len(configs)} configs")

    results = []
    best_acc, best_cfg = 0.0, None
    show_progress = not args.no_progress

    trial_iter = tqdm(enumerate(configs), total=len(configs), desc="CNNLSTM超参搜索", unit="trial")
    for i, cfg in trial_iter:
        trial_iter.set_postfix_str(
            f"backbone={cfg['backbone']} hidden={cfg['lstm_hidden']} | best={best_acc * 100:.1f}%"
        )
        set_seed(args.seed + i)

        try:
            acc, f1 = train_eval_cnnlstm(
                cfg, train_loader, val_loader, device,
                epochs=args.epochs,
                patience=args.patience,
                show_progress=show_progress
            )

            results.append({**cfg, "val_acc": acc, "val_f1": f1})
            trial_iter.write(
                f"  [{i + 1}/{len(configs)}] backbone={cfg['backbone']} hidden={cfg['lstm_hidden']} "
                f"lr={cfg['lr']:.0e} -> Val Acc: {acc * 100:.2f}%  F1: {f1 * 100:.2f}%"
            )
            if acc > best_acc:
                best_acc = acc
                best_cfg = cfg

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                trial_iter.write(f"  [{i + 1}/{len(configs)}] ⚠️ 发生 OOM 显存溢出，跳过此配置: {cfg['backbone']}")
                torch.cuda.empty_cache()
                continue
            else:
                raise e

    df = pd.DataFrame(results)
    df = df.sort_values("val_acc", ascending=False).reset_index(drop=True)
    df.to_csv(args.output, index=False)

    print(f"\nResults saved to {args.output}")
    print("\n=== Best CNNLSTM Config (by Val Acc) ===")
    for k, v in best_cfg.items():
        print(f"  {k}: {v}")
    print(f"  val_acc: {best_acc * 100:.2f}%")
    print(f"\nTop 5 configs:\n{df.head().to_string()}")


if __name__ == "__main__":
    main()
