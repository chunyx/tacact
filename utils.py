from __future__ import annotations
from contextlib import nullcontext
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


def _is_cuda_device(device: torch.device) -> bool:
    return device.type == "cuda"


def _sync_if_cuda(device: torch.device) -> None:
    if _is_cuda_device(device):
        torch.cuda.synchronize()


def _autocast_if_needed(device: torch.device, amp: bool):
    if amp and _is_cuda_device(device):
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def _get_model_name(model: nn.Module) -> str:
    return model.__class__.__name__


def _is_vit_model(model_name: str) -> bool:
    return "ViT" in model_name


def _uses_gradient_clipping(model_name: str) -> bool:
    return any(name in model_name for name in ("ViT", "TCN", "LSTM"))


def _resolve_training_hyperparams(
    model_name: str,
    lr_override,
    weight_decay_override,
) -> Tuple[float, float]:
    if _is_vit_model(model_name):
        default_lr, default_weight_decay = 5e-5, 0.05
    elif "TCN" in model_name:
        default_lr, default_weight_decay = 5e-4, 1e-4
    else:
        default_lr, default_weight_decay = 1e-3, 1e-5
    lr = default_lr if lr_override is None else lr_override
    weight_decay = default_weight_decay if weight_decay_override is None else weight_decay_override
    return float(lr), float(weight_decay)


def benchmark_torch(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    # 简单的预热 (Warm-up) 防止首次推理慢
    if _is_cuda_device(device):
        dummy = torch.randn(1, 80, 32, 32).to(device)
        model(dummy)
        _sync_if_cuda(device)

    model.eval()
    n = 0
    start = time.perf_counter()
    with torch.no_grad():
        for x, _ in loader:
            _ = model(x.to(device))
            n += x.size(0)
    _sync_if_cuda(device)
    return (time.perf_counter() - start) * 1000.0 / max(1, n)


def benchmark_torch_gpu_deploy(
        model: nn.Module,
        sample_batch: torch.Tensor,
        device: torch.device,
        batch_sizes: List[int],
        iters: int = 100,
        warmup: int = 20,
        amp: bool = False,
) -> Dict[str, float]:
    model.eval()
    if not _is_cuda_device(device):
        amp = False

    x1 = sample_batch[:1].to(device, non_blocking=True)
    _sync_if_cuda(device)

    with torch.no_grad():
        for _ in range(max(1, warmup)):
            with _autocast_if_needed(device, amp):
                _ = model(x1)
    _sync_if_cuda(device)

    out: Dict[str, float] = {}

    def _bench(x: torch.Tensor) -> float:
        with torch.no_grad():
            _sync_if_cuda(device)
            st = time.perf_counter()
            for _ in range(max(1, iters)):
                with _autocast_if_needed(device, amp):
                    _ = model(x)
            _sync_if_cuda(device)
            el = time.perf_counter() - st
        return el * 1000.0 / max(1, iters) / float(x.size(0))

    for bs in batch_sizes:
        xb = sample_batch[:bs]
        if xb.size(0) < bs:
            rep = (bs + xb.size(0) - 1) // max(1, xb.size(0))
            xb = xb.repeat(rep, 1, 1, 1)[:bs]
        xb = xb.to(device, non_blocking=True)
        ms_per_sample = _bench(xb)
        out[f"latency_ms_bs{bs}"] = float(ms_per_sample)
        out[f"throughput_sps_bs{bs}"] = float(1000.0 / max(1e-9, ms_per_sample))

    return out


def benchmark_torch_model_only(model: nn.Module, loader: DataLoader, device: torch.device,
                              max_batches: int = 20, warmup_batches: int = 2) -> float:
    batches: List[torch.Tensor] = []
    for i, (x, _) in enumerate(loader):
        if i >= max_batches:
            break
        batches.append(x)
    if not batches:
        return 0.0

    model.eval()
    xs: List[torch.Tensor] = [b.to(device, non_blocking=True) for b in batches]
    _sync_if_cuda(device)

    with torch.no_grad():
        for i in range(min(warmup_batches, len(xs))):
            _ = model(xs[i])
    _sync_if_cuda(device)

    n = 0
    start = time.perf_counter()
    with torch.no_grad():
        for x in xs:
            _ = model(x)
            n += x.size(0)
    _sync_if_cuda(device)
    return (time.perf_counter() - start) * 1000.0 / max(1, n)


def train_torch_model(model: nn.Module, train_loader: torch.utils.data.DataLoader,
                      val_loader: torch.utils.data.DataLoader, epochs: int,
                      device: torch.device, patience: int = 10,
                      lr_override=None, weight_decay_override=None) -> Dict[str, List[float]]:
    """
    改进的训练函数，针对ViT/TCN优化超参数
    patience: 早停忍耐度
    lr_override, weight_decay_override: 可选覆盖（用于超参搜索）
    """
    criterion = nn.CrossEntropyLoss()
    
    # 根据模型类型调整学习率与调度器
    model_name = _get_model_name(model)
    lr, weight_decay = _resolve_training_hyperparams(model_name, lr_override, weight_decay_override)

    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    if _is_vit_model(model_name):
        sched = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            opt, T_0=15, T_mult=2, eta_min=lr * 0.01
        )
    elif "TCN" in model_name:
        sched = optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=3)
    else:
        sched = optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=2)

    model.to(device)
    history = {"train_loss": [], "val_acc": [], "epoch_time_s": [], "cum_time_s": []}

    best_acc = 0.0
    best_weights = copy.deepcopy(model.state_dict())
    early_stop_counter = 0

    for epoch in range(epochs):
        epoch_start = time.perf_counter()
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
            
            if _uses_gradient_clipping(model_name):
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            opt.step()

            current_loss = float(loss.item())
            loss_sum += current_loss
            n_batches += 1
            pbar.set_postfix({"loss": f"{current_loss:.4f}"})

        # 【重点修改】直接调用，不需要再写 import
        val_acc, _ = evaluate_torch(model, val_loader, device)

        if _is_vit_model(model_name):
            sched.step()
        else:
            sched.step(val_acc)
            
        epoch_loss = loss_sum / max(1, n_batches)
        epoch_time_s = time.perf_counter() - epoch_start
        cum_time_s = (history["cum_time_s"][-1] if history["cum_time_s"] else 0.0) + epoch_time_s
        history["train_loss"].append(epoch_loss)
        history["val_acc"].append(val_acc)
        history["epoch_time_s"].append(float(epoch_time_s))
        history["cum_time_s"].append(float(cum_time_s))

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


def count_sklearn_params(model) -> int:
    """估算传统ML模型的"等效参数量"（避免在对比图中被硬编码为0）"""
    model_name = model.__class__.__name__.lower()
    
    if "svc" in model_name or "svm" in model_name:
        # SVM: 支持向量数 × 特征维度 + 偏置
        if hasattr(model, 'support_vectors_'):
            n_support = model.support_vectors_.shape[0]
            n_features = model.support_vectors_.shape[1]
            return n_support * n_features + model.n_support_.sum()
        return 0
    
    elif "randomforest" in model_name or "forest" in model_name:
        # RandomForest: 估算所有树的节点数
        # 每个节点存储: feature_index(1) + threshold(1) + left/right child(2) + value(1)
        if hasattr(model, 'estimators_'):
            total_nodes = 0
            for tree in model.estimators_:
                if hasattr(tree, 'tree_'):
                    total_nodes += tree.tree_.node_count
            return total_nodes * 5  # 每个节点约5个数值参数
        return 0
    
    elif "xgb" in model_name or "xgboost" in model_name or "gradientboosting" in model_name:
        # XGBoost: 类似RandomForest，但通常更深
        if hasattr(model, 'get_booster'):
            try:
                # 获取树的结构信息
                trees = model.get_booster().get_dump()
                total_nodes = sum(len(tree.strip().split('\n')) for tree in trees if tree.strip())
                return total_nodes * 5
            except:
                pass
        if hasattr(model, 'n_estimators'):
            # 保守估计：每棵树平均100个节点
            return model.n_estimators * 100 * 5
        return 0
    
    return 0


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

def _apply_pub_style() -> None:
    plt.rcParams.update({
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "legend.fontsize": 9,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.2,
        "grid.linestyle": "--",
    })


def _normalize_confusion_matrix(cm: np.ndarray) -> np.ndarray:
    row_sums = cm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    return cm.astype(np.float64) / row_sums


def _safe_inference_ms(result: Dict[str, float]) -> float:
    return max(float(result["inference_ms"]), 1e-6)


def save_confusion_matrix(cm: np.ndarray, model_name: str, save_path: Path) -> None:
    _apply_pub_style()
    cm_norm = _normalize_confusion_matrix(cm)
    plt.figure(figsize=(8.4, 6.8))
    plt.imshow(cm_norm, interpolation='nearest', cmap='Blues', vmin=0.0, vmax=1.0)
    plt.title(f'Normalized Confusion Matrix: {model_name}')
    cbar = plt.colorbar()
    cbar.set_label('Proportion')
    tick_marks = np.arange(12)
    plt.xticks(tick_marks, tick_marks)
    plt.yticks(tick_marks, tick_marks)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            value = cm_norm[i, j] * 100.0
            color = "white" if cm_norm[i, j] > 0.5 else "black"
            plt.text(j, i, f"{value:.1f}", ha="center", va="center", color=color, fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()


def save_confusion_comparison(confusion_mats: Dict[str, np.ndarray], save_path: Path) -> None:
    _apply_pub_style()
    n_models = len(confusion_mats)
    n_cols = min(3, max(1, n_models))
    n_rows = int(np.ceil(n_models / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.0 * n_cols, 4.6 * n_rows))
    axes = np.atleast_1d(axes).flatten()
    last_im = None

    for idx, (name, cm) in enumerate(confusion_mats.items()):
        ax = axes[idx]
        cm_norm = _normalize_confusion_matrix(cm)
        im = ax.imshow(cm_norm, interpolation='nearest', cmap='Blues', vmin=0.0, vmax=1.0)
        last_im = im
        ax.set_title(name)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                value = cm_norm[i, j] * 100.0
                color = "white" if cm_norm[i, j] > 0.5 else "black"
                ax.text(j, i, f"{value:.0f}", ha="center", va="center", color=color, fontsize=7)

    for idx in range(n_models, len(axes)):
        axes[idx].axis('off')

    if last_im is not None:
        fig.colorbar(last_im, ax=axes[:n_models], shrink=0.85, label='Proportion')
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()


def save_training_curves(histories: Dict[str, Dict[str, List[float]]], save_path: Path) -> None:
    _apply_pub_style()
    fig, axes = plt.subplots(2, 1, figsize=(10, 7))
    palette = plt.cm.tab10(np.linspace(0, 1, max(1, len(histories))))

    for idx, (name, history) in enumerate(histories.items()):
        color = palette[idx]
        x_loss = list(range(1, len(history["train_loss"]) + 1))
        x_acc = list(range(1, len(history["val_acc"]) + 1))
        axes[0].plot(x_loss, history["train_loss"], label=name, color=color, linewidth=2.0)
        axes[1].plot(x_acc, history["val_acc"], label=name, color=color, linewidth=2.0)

    axes[0].set_title('Training Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend(frameon=False, ncol=2)

    axes[1].set_title('Validation Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend(frameon=False, ncol=2)
    axes[1].set_ylim(0.0, 1.0)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()


def save_training_curves_with_std(
    histories_by_model: Dict[str, List[Dict[str, List[float]]]],
    save_path: Path,
) -> None:
    _apply_pub_style()
    fig, axes = plt.subplots(2, 1, figsize=(10, 7))
    model_names = list(histories_by_model.keys())
    palette = plt.cm.tab10(np.linspace(0, 1, max(1, len(model_names))))

    for idx, model_name in enumerate(model_names):
        runs = histories_by_model[model_name]
        if not runs:
            continue
        color = palette[idx]

        max_loss_len = max(len(run.get("train_loss", [])) for run in runs)
        max_acc_len = max(len(run.get("val_acc", [])) for run in runs)

        loss_mat = np.full((len(runs), max_loss_len), np.nan, dtype=np.float64)
        acc_mat = np.full((len(runs), max_acc_len), np.nan, dtype=np.float64)
        for ridx, run in enumerate(runs):
            train_loss = np.asarray(run.get("train_loss", []), dtype=np.float64)
            val_acc = np.asarray(run.get("val_acc", []), dtype=np.float64)
            loss_mat[ridx, :len(train_loss)] = train_loss
            acc_mat[ridx, :len(val_acc)] = val_acc

        loss_mean = np.nanmean(loss_mat, axis=0)
        loss_std = np.nanstd(loss_mat, axis=0)
        acc_mean = np.nanmean(acc_mat, axis=0)
        acc_std = np.nanstd(acc_mat, axis=0)

        x_loss = np.arange(1, len(loss_mean) + 1)
        x_acc = np.arange(1, len(acc_mean) + 1)

        axes[0].plot(x_loss, loss_mean, label=model_name, color=color, linewidth=2.0)
        axes[0].fill_between(x_loss, loss_mean - loss_std, loss_mean + loss_std, color=color, alpha=0.18)

        axes[1].plot(x_acc, acc_mean, label=model_name, color=color, linewidth=2.0)
        axes[1].fill_between(
            x_acc,
            np.clip(acc_mean - acc_std, 0.0, 1.0),
            np.clip(acc_mean + acc_std, 0.0, 1.0),
            color=color,
            alpha=0.18,
        )

    axes[0].set_title('Training Loss Across Seeds')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend(frameon=False, ncol=2)

    axes[1].set_title('Validation Accuracy Across Seeds')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_ylim(0.0, 1.0)
    axes[1].legend(frameon=False, ncol=2)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()


def save_per_class_f1_bars(per_class_f1: Dict[str, np.ndarray], save_path: Path) -> None:
    _apply_pub_style()
    plt.figure(figsize=(14, 6.8))
    models = list(per_class_f1.keys())
    classes = list(range(12))
    x = np.arange(len(classes))
    width = 0.84 / max(1, len(models))
    palette = plt.cm.Set2(np.linspace(0, 1, max(1, len(models))))

    for i, model in enumerate(models):
        plt.bar(x + i * width, per_class_f1[model], width, label=model, color=palette[i], edgecolor='black', linewidth=0.4)

    plt.xlabel('Class')
    plt.ylabel('F1 Score')
    plt.title('Per-Class F1 Score Comparison')
    plt.xticks(x + width * (len(models) - 1) / 2, classes)
    plt.ylim(0.0, 1.0)
    plt.legend(frameon=False, ncol=2)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()


def save_summary_bar_with_error(
    summary_df: pd.DataFrame,
    save_path: Path,
    metric_col: str = "accuracy_mean",
    error_col: str = "accuracy_std",
    title: str = "Model Accuracy Across Seeds",
    ylabel: str = "Accuracy",
) -> None:
    if summary_df.empty:
        return

    _apply_pub_style()
    plot_df = summary_df.sort_values(metric_col, ascending=False).copy()
    x = np.arange(len(plot_df))
    colors = plt.cm.tab20(np.linspace(0, 1, max(1, len(plot_df))))

    plt.figure(figsize=(max(10, len(plot_df) * 0.7), 6.5))
    plt.bar(
        x,
        plot_df[metric_col].to_numpy(dtype=np.float64),
        yerr=plot_df[error_col].fillna(0.0).to_numpy(dtype=np.float64),
        color=colors[:len(plot_df)],
        edgecolor='black',
        linewidth=0.5,
        capsize=4,
    )
    plt.xticks(x, plot_df["model"].tolist(), rotation=30, ha='right')
    plt.ylabel(ylabel)
    plt.title(title)
    if "accuracy" in metric_col or "f1" in metric_col or "precision" in metric_col or "recall" in metric_col:
        plt.ylim(0.0, 1.0)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()


def save_scatter(results: Dict[str, Dict[str, float]], save_path: Path) -> None:
    _apply_pub_style()
    fig, ax = plt.subplots(figsize=(10, 7))
    categories = {}
    for name, result in results.items():
        cat = result["category"]
        if cat not in categories:
            categories[cat] = {"x": [], "y": [], "s": [], "names": []}
        categories[cat]["x"].append(result["accuracy"] * 100.0)
        categories[cat]["y"].append(_safe_inference_ms(result))
        categories[cat]["s"].append(max(40.0, float(result["params_m"]) * 55.0 + 40.0))
        categories[cat]["names"].append(name)

    colors = {'cnn': '#1f77b4', 'temporal': '#d62728', 'attention': '#2ca02c', 'traditional': '#ff7f0e'}
    markers = {'cnn': 'o', 'temporal': 's', 'attention': '^', 'traditional': 'D'}

    for cat, data in categories.items():
        ax.scatter(
            data["x"],
            data["y"],
            s=data["s"],
            c=colors.get(cat, '#444444'),
            marker=markers.get(cat, 'o'),
            label=cat,
            alpha=0.8,
            edgecolors='black',
            linewidths=0.5,
        )
        for i, name in enumerate(data["names"]):
            ax.annotate(name, (data["x"][i], data["y"][i]), textcoords="offset points", xytext=(5, 5), fontsize=8)

    ax.set_xlabel('Accuracy (%)')
    ax.set_ylabel('Inference Time per Sample (ms)')
    ax.set_title('Accuracy-Latency Trade-off with Parameter Size')
    ax.set_yscale('log')
    ax.legend(frameon=False, title='Family')

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()


def save_radar_top3(results: Dict[str, Dict[str, float]], save_path: Path) -> None:
    _apply_pub_style()
    sorted_results = sorted(results.items(), key=lambda x: x[1]["accuracy"], reverse=True)[:3]
    metrics = ['accuracy', 'macro_f1', 'macro_precision', 'macro_recall']
    fig, ax = plt.subplots(figsize=(9, 5.8))
    x = np.arange(len(metrics))
    width = 0.24
    colors = plt.cm.Dark2(np.linspace(0, 1, max(1, len(sorted_results))))

    for i, (name, result) in enumerate(sorted_results):
        values = [result[m] for m in metrics]
        ax.bar(x + (i - (len(sorted_results) - 1) / 2) * width, values, width=width, label=name,
               color=colors[i], edgecolor='black', linewidth=0.4)

    ax.set_xticks(x)
    ax.set_xticklabels(['Accuracy', 'Macro-F1', 'Precision', 'Recall'])
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel('Score')
    ax.set_title('Top-3 Model Performance Comparison')
    ax.legend(frameon=False)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
