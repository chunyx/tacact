from __future__ import annotations
from tqdm import tqdm
import random
import time
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
# 导入绘图库，放在函数内部或顶部均可，这里为了整洁放在顶部，但注意环境依赖
import matplotlib.pyplot as plt
import copy


# ===========================
# 原 tacacteval_utils.py 内容
# ===========================

def parse_model_list(arg: str | None, default_models: List[str]) -> List[str]:
    if arg is None or not arg.strip():
        return list(default_models)
    return [x.strip() for x in arg.split(',') if x.strip()]


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def confusion_matrix_np(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int = 12) -> np.ndarray:
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    return cm


def per_class_prf(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int = 12) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray]:
    eps = 1e-12
    precision = np.full(n_classes, np.nan, dtype=np.float64)
    recall = np.full(n_classes, np.nan, dtype=np.float64)
    f1 = np.full(n_classes, np.nan, dtype=np.float64)
    for c in range(n_classes):
        tp = np.sum((y_true == c) & (y_pred == c))
        fp = np.sum((y_true != c) & (y_pred == c))
        fn = np.sum((y_true == c) & (y_pred != c))
        denom_p = tp + fp
        denom_r = tp + fn
        if denom_p > 0:
            precision[c] = tp / (denom_p + eps)
        if denom_r > 0:
            recall[c] = tp / (denom_r + eps)
        if not np.isnan(precision[c]) and not np.isnan(recall[c]):
            denom_f = precision[c] + recall[c]
            if denom_f > 0:
                f1[c] = 2 * precision[c] * recall[c] / (denom_f + eps)
    return precision, recall, f1


def evaluate_torch(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[float, float]:
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            pred = model(x).argmax(dim=1).cpu().numpy()
            ys.append(y.numpy())
            ps.append(pred)
    y_true = np.concatenate(ys)
    y_pred = np.concatenate(ps)
    _, _, f1c = per_class_prf(y_true, y_pred, 12)
    return float((y_true == y_pred).mean()), float(np.nanmean(f1c))


def benchmark_torch(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    # 简单的预热 (Warm-up) 防止首次推理慢
    if device.type == "cuda":
        dummy = torch.randn(1, 80, 32, 32).to(device)
        model(dummy)
        torch.cuda.synchronize()

    model.eval()
    n = 0
    start = time.perf_counter()
    with torch.no_grad():
        for x, _ in loader:
            _ = model(x.to(device))
            n += x.size(0)
    if device.type == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1000.0 / max(1, n)


def train_torch_model(model: nn.Module, train_loader: torch.utils.data.DataLoader,
                      val_loader: torch.utils.data.DataLoader, epochs: int,
                      device: torch.device, patience: int = 7) -> Dict[str, List[float]]:
    """
    改进的训练函数，针对ViT优化超参数
    patience: 早停忍耐度。
    """
    criterion = nn.CrossEntropyLoss()
    
    # 根据模型类型调整学习率
    model_name = model.__class__.__name__
    if "ViT" in model_name:
        # ViT需要更小的学习率和更长的训练
        lr = 1e-4
        weight_decay = 1e-4
        patience = 15  # ViT需要更多耐心
    else:
        # 其他模型使用原来的参数
        lr = 1e-3
        weight_decay = 1e-5
    
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # 使用cosine annealing scheduler，对ViT更友好
    if "ViT" in model_name:
        sched = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            opt, T_0=10, T_mult=2, eta_min=lr * 0.1
        )
    else:
        sched = optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=2)

    model.to(device)
    history = {"train_loss": [], "val_acc": []}

    best_acc = 0.0
    best_weights = copy.deepcopy(model.state_dict())
    early_stop_counter = 0

    for epoch in range(epochs):
        model.train()
        loss_sum = 0.0
        n_batches = 0

        # 这里现在可以正常使用 tqdm 了
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}", leave=False)
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            
            # 梯度裁剪，对ViT很重要
            if "ViT" in model_name:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            opt.step()

            current_loss = float(loss.item())
            loss_sum += current_loss
            n_batches += 1
            pbar.set_postfix({"loss": f"{current_loss:.4f}"})

        # 【重点修改】直接调用，不需要再写 import
        val_acc, _ = evaluate_torch(model, val_loader, device)

        # 根据scheduler类型调整step
        if "ViT" in model_name:
            sched.step()
        else:
            sched.step(val_acc)
            
        epoch_loss = loss_sum / max(1, n_batches)
        history["train_loss"].append(epoch_loss)
        history["val_acc"].append(val_acc)

        if val_acc > best_acc:
            best_acc = val_acc
            best_weights = copy.deepcopy(model.state_dict())
            early_stop_counter = 0
        else:
            early_stop_counter += 1

        # 显示当前学习率
        current_lr = opt.param_groups[0]['lr']
        print(f"Epoch {epoch + 1}/{epochs} | Loss: {epoch_loss:.4f} | Val Acc: {val_acc * 100:.2f}% | LR: {current_lr:.6f} | Stop: {early_stop_counter}/{patience}")

        if early_stop_counter >= patience:
            print(f"\n[Early Stopping] 触发早停。")
            break

    model.load_state_dict(best_weights)
    return history

def subset_to_numpy(dataset, subset: Subset) -> Tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    for i in subset.indices:
        x, y = dataset[i]
        a = x.numpy().astype(np.float32, copy=False)
        if a.ndim != 3:
            a = a.reshape(a.shape[0], 32, 32)
        t, h, w = a.shape
        spatial = 8
        bins = 10
        ph = h // spatial
        pw = w // spatial
        a = a[:, : spatial * ph, : spatial * pw]
        a = a.reshape(t, spatial, ph, spatial, pw).mean(axis=(2, 4))
        step = max(1, t // bins)
        tt = step * bins
        if t < tt:
            pad = np.zeros((tt - t, spatial, spatial), dtype=np.float32)
            a = np.concatenate([a, pad], axis=0)
        else:
            a = a[:tt]
        a = a.reshape(bins, step, spatial, spatial).mean(axis=1)
        xs.append(a.reshape(-1))
        ys.append(y)
    return np.stack(xs).astype(np.float32), np.array(ys, dtype=np.int64)


def benchmark_sklearn(model, x_test: np.ndarray) -> float:
    start = time.perf_counter()
    _ = model.predict(x_test)
    return (time.perf_counter() - start) * 1000.0 / max(1, len(x_test))


def merge_metrics_csvs(csv_paths: List[Path]) -> pd.DataFrame:
    frames = []
    for cp in csv_paths:
        if not cp.exists():
            raise FileNotFoundError(f"Metrics CSV not found: {cp}")
        df = pd.read_csv(cp)
        if "model" not in df.columns:
            if len(df.columns) > 0 and str(df.columns[0]).startswith("Unnamed"):
                df = df.rename(columns={df.columns[0]: "model"})
            else:
                raise ValueError(f"CSV missing model column: {cp}")
        frames.append(df)
    merged = pd.concat(frames, axis=0, ignore_index=True)
    return merged.drop_duplicates(subset=["model"], keep="last")


def dataframe_to_results_dict(df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    req = ["model", "category", "accuracy", "macro_f1", "macro_precision", "macro_recall", "inference_ms", "params",
           "params_m"]
    # 简单兼容检查，不强制报错
    out: Dict[str, Dict[str, float]] = {}
    for _, r in df.iterrows():
        out[str(r["model"])] = {
            "category": str(r["category"]),
            "accuracy": float(r["accuracy"]),
            "macro_f1": float(r["macro_f1"]),
            "macro_precision": float(r["macro_precision"]),
            "macro_recall": float(r["macro_recall"]),
            "inference_ms": float(r["inference_ms"]),
            "params": float(r["params"]),
            "params_m": float(r["params_m"]),
        }
    return out


# ===========================
# 绘图函数
# ===========================

def save_confusion_matrix(cm: np.ndarray, model_name: str, save_path: Path) -> None:
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f'Confusion Matrix - {model_name}')
    plt.colorbar()
    tick_marks = np.arange(12)
    plt.xticks(tick_marks, tick_marks)
    plt.yticks(tick_marks, tick_marks)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def save_confusion_comparison(confusion_mats: Dict[str, np.ndarray], save_path: Path) -> None:
    n_models = len(confusion_mats)
    fig, axes = plt.subplots(2, (n_models + 1) // 2, figsize=(15, 10))
    if n_models == 1:
        axes = [axes]
    elif n_models <= 2:
        axes = axes.flatten()
    else:
        axes = axes.flatten()
    
    for idx, (name, cm) in enumerate(confusion_mats.items()):
        ax = axes[idx]
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax.set_title(f'{name}')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        
        # 添加数字标签
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="black" if cm[i, j] < cm.max() / 2 else "white")
    
    # 隐藏多余的子图
    for idx in range(n_models, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def save_training_curves(histories: Dict[str, Dict[str, List[float]]], save_path: Path) -> None:
    plt.figure(figsize=(12, 8))
    
    # 绘制训练损失
    plt.subplot(2, 1, 1)
    for name, history in histories.items():
        plt.plot(history["train_loss"], label=f'{name} - Train Loss')
    plt.title('Training Loss Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 绘制验证准确率
    plt.subplot(2, 1, 2)
    for name, history in histories.items():
        plt.plot(history["val_acc"], label=f'{name} - Val Acc')
    plt.title('Validation Accuracy Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def save_per_class_f1_bars(per_class_f1: Dict[str, np.ndarray], save_path: Path) -> None:
    plt.figure(figsize=(15, 8))
    models = list(per_class_f1.keys())
    classes = list(range(12))
    
    x = np.arange(len(classes))
    width = 0.8 / len(models)
    
    for i, model in enumerate(models):
        plt.bar(x + i * width, per_class_f1[model], width, label=model)
    
    plt.xlabel('Class')
    plt.ylabel('F1 Score')
    plt.title('Per-Class F1 Scores Comparison')
    plt.xticks(x + width * (len(models) - 1) / 2, classes)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def save_scatter(results: Dict[str, Dict[str, float]], save_path: Path) -> None:
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    
    categories = {}
    for name, result in results.items():
        cat = result["category"]
        if cat not in categories:
            categories[cat] = {"x": [], "y": [], "z": [], "names": []}
        categories[cat]["x"].append(result["accuracy"] * 100)
        categories[cat]["y"].append(result["inference_ms"])
        categories[cat]["z"].append(result["params_m"])
        categories[cat]["names"].append(name)
    
    colors = {'cnn': 'blue', 'temporal': 'red', 'attention': 'green', 'traditional': 'orange'}
    markers = {'cnn': 'o', 'temporal': 's', 'attention': '^', 'traditional': 'D'}
    
    for cat, data in categories.items():
        ax.scatter(data["x"], data["y"], data["z"], 
                  c=colors.get(cat, 'black'), 
                  marker=markers.get(cat, 'o'),
                  label=cat, s=100, alpha=0.7)
        
        # 添加模型名称标签
        for i, name in enumerate(data["names"]):
            ax.text(data["x"][i], data["y"][i], data["z"][i], name, fontsize=8)
    
    ax.set_xlabel('Accuracy (%)')
    ax.set_ylabel('Inference Time (ms)')
    ax.set_zlabel('Parameters (M)')
    ax.set_title('Model Comparison: Accuracy vs Inference Time vs Parameters')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def save_radar_top3(results: Dict[str, Dict[str, float]], save_path: Path) -> None:
    # 选择准确率最高的前3个模型
    sorted_results = sorted(results.items(), key=lambda x: x[1]["accuracy"], reverse=True)[:3]
    
    metrics = ['accuracy', 'macro_f1', 'macro_precision', 'macro_recall']
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]  # 闭合图形
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    colors = ['blue', 'red', 'green']
    for i, (name, result) in enumerate(sorted_results):
        values = [result[m] for m in metrics]
        values += values[:1]  # 闭合图形
        ax.plot(angles, values, 'o-', linewidth=2, label=name, color=colors[i])
        ax.fill(angles, values, alpha=0.25, color=colors[i])
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1)
    ax.set_title('Top 3 Models Performance Comparison')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


from . import utils as _utils

_EXPORTS = [
    "parse_model_list",
    "set_seed",
    "count_parameters",
    "confusion_matrix_np",
    "per_class_prf",
    "evaluate_torch",
    "benchmark_torch",
    "benchmark_torch_model_only",
    "benchmark_torch_gpu_deploy",
    "train_torch_model",
    "subset_to_numpy",
    "count_sklearn_params",
    "benchmark_sklearn",
    "merge_metrics_csvs",
    "dataframe_to_results_dict",
    "save_confusion_matrix",
    "save_confusion_comparison",
    "save_training_curves",
    "save_per_class_f1_bars",
    "save_scatter",
    "save_radar_top3",
]

for _name in _EXPORTS:
    if hasattr(_utils, _name):
        globals()[_name] = getattr(_utils, _name)

__all__ = list(_EXPORTS)
