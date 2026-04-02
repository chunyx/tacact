from __future__ import annotations

import copy
import random
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm


# ===========================
# Hardware Setup
# ===========================

def choose_device_or_exit() -> torch.device:
    """Return CUDA device if available, otherwise ask whether to continue on CPU."""
    if torch.cuda.is_available():
        current_index = torch.cuda.current_device()
        gpu_count = torch.cuda.device_count()
        gpu_name = torch.cuda.get_device_name(current_index)
        print(f"[Hardware] Detected CUDA GPU ({current_index + 1}/{gpu_count}): {gpu_name}")
        return torch.device("cuda")

    print("[Hardware] No CUDA GPU detected. The program will run on CPU, which may be much slower.")
    try:
        answer = input("No GPU was detected. Continue with CPU? [y/N]: ").strip().lower()
    except EOFError:
        answer = ""

    if answer not in {"y", "yes"}:
        print("Program aborted because no GPU was available.")
        raise SystemExit(0)

    print("[Hardware] Continuing with CPU.")
    return torch.device("cpu")


def _is_cuda_device(device: torch.device) -> bool:
    return device.type == "cuda"


def _sync_if_cuda(device: torch.device) -> None:
    if _is_cuda_device(device):
        torch.cuda.synchronize()


def _autocast_if_needed(device: torch.device, amp: bool):
    if amp and _is_cuda_device(device):
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


# ===========================
# Data Utilities
# ===========================

def parse_model_list(arg: str | None, default_models: List[str]) -> List[str]:
    return list(default_models) if arg is None or not arg.strip() else [x.strip() for x in arg.split(",") if x.strip()]


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


def per_class_prf(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_classes: int = 12,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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
    model_name = model.__class__.__name__.lower()
    if "svc" in model_name or "svm" in model_name:
        if hasattr(model, "support_vectors_"):
            n_support = model.support_vectors_.shape[0]
            n_features = model.support_vectors_.shape[1]
            return n_support * n_features + model.n_support_.sum()
        return 0
    if "randomforest" in model_name or "forest" in model_name:
        if hasattr(model, "estimators_"):
            total_nodes = 0
            for tree in model.estimators_:
                if hasattr(tree, "tree_"):
                    total_nodes += tree.tree_.node_count
            return total_nodes * 5
        return 0
    if "xgb" in model_name or "xgboost" in model_name or "gradientboosting" in model_name:
        if hasattr(model, "get_booster"):
            try:
                trees = model.get_booster().get_dump()
                total_nodes = sum(len(tree.strip().split("\n")) for tree in trees if tree.strip())
                return total_nodes * 5
            except Exception:
                pass
        if hasattr(model, "n_estimators"):
            return model.n_estimators * 100 * 5
        return 0
    return 0


# ===========================
# Training Core
# ===========================

def _resolve_training_hyperparams(
    model_name: str,
    lr_override,
    weight_decay_override,
) -> Tuple[float, float]:
    if "ViT" in model_name:
        default_lr = 1e-4
        default_weight_decay = 1e-4
    else:
        default_lr = 1e-3
        default_weight_decay = 1e-5
    lr = default_lr if lr_override is None else float(lr_override)
    weight_decay = default_weight_decay if weight_decay_override is None else float(weight_decay_override)
    return lr, weight_decay


def evaluate_torch_val_loss(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
) -> float:
    """Compute average validation loss over all samples."""
    model.eval()
    total_loss = 0.0
    total_samples = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            batch_size = y.size(0)
            loss = criterion(logits, y)
            total_loss += float(loss.item()) * batch_size
            total_samples += batch_size
    return total_loss / max(1, total_samples)


def train_torch_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    device: torch.device,
    patience: int = 7,
    lr_override=None,
    weight_decay_override=None,
    gl_alpha: float = 2.0,
) -> Dict[str, List[float]]:
    """Train a torch model with strict Prechelt GL_alpha early stopping on validation loss."""
    _ = patience  # kept for backward-compatible signature; not used in stopping logic
    criterion = nn.CrossEntropyLoss()
    model_name = model.__class__.__name__
    lr, weight_decay = _resolve_training_hyperparams(model_name, lr_override, weight_decay_override)

    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    if "ViT" in model_name:
        sched = optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=10, T_mult=2, eta_min=lr * 0.1)
    else:
        sched = optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=2)

    model.to(device)
    scaler = torch.amp.GradScaler(device.type)
    amp_enabled = _is_cuda_device(device)
    history = {
        "train_loss": [],
        "val_loss": [],
        "val_acc": [],
        "epoch_time_s": [],
        "cum_time_s": [],
        "best_val_loss": [],
        "best_epoch": [],
        "current_GL": [],
        "stop_triggered": [],
        "alpha": [],
    }
    best_val_loss = float("inf")
    best_epoch = -1
    best_weights = copy.deepcopy(model.state_dict())

    for epoch in range(epochs):
        epoch_start = time.perf_counter()
        model.train()
        loss_sum = 0.0
        n_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}", leave=False)
        for x, y in pbar:
            try:
                x, y = x.to(device), y.to(device)
                opt.zero_grad()
                with torch.amp.autocast(device.type, enabled=amp_enabled):
                    loss = criterion(model(x), y)
                scaler.scale(loss).backward()
                if "ViT" in model_name:
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(opt)
                scaler.update()
            except RuntimeError as exc:
                if "out of memory" in str(exc).lower():
                    print("[WARN] CUDA OOM detected during training step; skipping current batch.")
                    if _is_cuda_device(device):
                        torch.cuda.empty_cache()
                    continue
                raise

            current_loss = float(loss.item())
            loss_sum += current_loss
            n_batches += 1
            pbar.set_postfix({"loss": f"{current_loss:.4f}"})

        val_loss = evaluate_torch_val_loss(model, val_loader, device, criterion)
        val_acc, _ = evaluate_torch(model, val_loader, device)
        if "ViT" in model_name:
            sched.step()
        else:
            sched.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = float(val_loss)
            best_epoch = epoch + 1
            best_weights = copy.deepcopy(model.state_dict())

        best_reference = max(best_val_loss, 1e-12)  # epsilon only prevents division-by-zero
        current_gl = 100.0 * (float(val_loss) / best_reference - 1.0)
        stop_triggered = current_gl > float(gl_alpha)

        epoch_loss = loss_sum / max(1, n_batches)
        epoch_time_s = time.perf_counter() - epoch_start
        cum_time_s = (history["cum_time_s"][-1] if history["cum_time_s"] else 0.0) + epoch_time_s
        history["train_loss"].append(epoch_loss)
        history["val_loss"].append(float(val_loss))
        history["val_acc"].append(val_acc)
        history["epoch_time_s"].append(float(epoch_time_s))
        history["cum_time_s"].append(float(cum_time_s))
        history["best_val_loss"].append(float(best_val_loss))
        history["best_epoch"].append(int(best_epoch))
        history["current_GL"].append(float(current_gl))
        history["stop_triggered"].append(bool(stop_triggered))
        history["alpha"].append(float(gl_alpha))

        current_lr = opt.param_groups[0]["lr"]
        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"TrainLoss: {epoch_loss:.4f} | "
            f"ValLoss: {float(val_loss):.6f} | "
            f"BestValLoss: {float(best_val_loss):.6f} | "
            f"ValAcc: {val_acc * 100:.2f}% | "
            f"GL: {float(current_gl):.4f} | "
            f"Alpha: {float(gl_alpha):.2f} | "
            f"Stop: {stop_triggered} | "
            f"LR: {current_lr:.6f}"
        )

        if stop_triggered:
            print("\n[Early Stopping] Prechelt GL_alpha triggered.")
            break

    model.load_state_dict(best_weights)
    return history


# ===========================
# Evaluation & Logging
# ===========================

def evaluate_torch(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[float, float]:
    """Evaluate model and return (accuracy, macro_f1)."""
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
    """Benchmark average inference latency in milliseconds per sample."""
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


def benchmark_torch_model_only(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    max_batches: int = 20,
    warmup_batches: int = 2,
) -> float:
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
            elapsed = time.perf_counter() - st
        return elapsed * 1000.0 / max(1, iters) / float(x.size(0))

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


def results_to_csv(results: Dict[str, Dict[str, float]], save_path: Path) -> pd.DataFrame:
    """Serialize experiment results to a CSV file and return the DataFrame."""
    df = pd.DataFrame.from_dict(results, orient="index").reset_index().rename(columns={"index": "model"})
    save_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(save_path, index=False)
    return df


def save_confusion_matrix(cm: np.ndarray, model_name: str, save_path: Path) -> None:
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title(f"Confusion Matrix - {model_name}")
    plt.colorbar()
    tick_marks = np.arange(12)
    plt.xticks(tick_marks, tick_marks)
    plt.yticks(tick_marks, tick_marks)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def save_confusion_comparison(confusion_mats: Dict[str, np.ndarray], save_path: Path) -> None:
    n_models = len(confusion_mats)
    fig, axes = plt.subplots(2, (n_models + 1) // 2, figsize=(15, 10))
    axes = np.atleast_1d(axes).flatten()

    for idx, (name, cm) in enumerate(confusion_mats.items()):
        ax = axes[idx]
        ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
        ax.set_title(name)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        max_val = cm.max() if cm.size else 0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                color = "black" if cm[i, j] < max_val / 2 else "white"
                ax.text(j, i, str(cm[i, j]), ha="center", va="center", color=color)

    for idx in range(n_models, len(axes)):
        axes[idx].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def save_training_curves(histories: Dict[str, Dict[str, List[float]]], save_path: Path) -> None:
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 1, 1)
    for name, history in histories.items():
        plt.plot(history["train_loss"], label=f"{name} - Train Loss")
    plt.title("Training Loss Curves")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 1, 2)
    for name, history in histories.items():
        if "val_loss" in history and len(history["val_loss"]) > 0:
            plt.plot(history["val_loss"], label=f"{name} - Val Loss")
        elif "val_acc" in history and len(history["val_acc"]) > 0:
            plt.plot(history["val_acc"], label=f"{name} - Val Acc (fallback)")
    plt.title("Validation Loss Curves")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def save_training_curves_with_std(
    histories_by_model: Dict[str, List[Dict[str, List[float]]]],
    save_path: Path,
) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(10, 7))
    model_names = list(histories_by_model.keys())
    palette = plt.cm.tab10(np.linspace(0, 1, max(1, len(model_names))))

    for idx, model_name in enumerate(model_names):
        runs = histories_by_model[model_name]
        if not runs:
            continue
        color = palette[idx]
        max_loss_len = max(len(run.get("train_loss", [])) for run in runs)
        max_val_len = max(
            len(run.get("val_loss", run.get("val_acc", [])))
            for run in runs
        )

        loss_mat = np.full((len(runs), max_loss_len), np.nan, dtype=np.float64)
        val_mat = np.full((len(runs), max_val_len), np.nan, dtype=np.float64)
        for ridx, run in enumerate(runs):
            train_loss = np.asarray(run.get("train_loss", []), dtype=np.float64)
            val_curve = np.asarray(run.get("val_loss", run.get("val_acc", [])), dtype=np.float64)
            loss_mat[ridx, :len(train_loss)] = train_loss
            val_mat[ridx, :len(val_curve)] = val_curve

        loss_mean = np.nanmean(loss_mat, axis=0)
        loss_std = np.nanstd(loss_mat, axis=0)
        val_mean = np.nanmean(val_mat, axis=0)
        val_std = np.nanstd(val_mat, axis=0)
        x_loss = np.arange(1, len(loss_mean) + 1)
        x_val = np.arange(1, len(val_mean) + 1)

        axes[0].plot(x_loss, loss_mean, label=model_name, color=color, linewidth=2.0)
        axes[0].fill_between(x_loss, loss_mean - loss_std, loss_mean + loss_std, color=color, alpha=0.18)
        axes[1].plot(x_val, val_mean, label=model_name, color=color, linewidth=2.0)
        axes[1].fill_between(
            x_val,
            val_mean - val_std,
            val_mean + val_std,
            color=color,
            alpha=0.18,
        )

    axes[0].set_title("Training Loss Across Seeds")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend(frameon=False, ncol=2)
    axes[1].set_title("Validation Loss Across Seeds")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend(frameon=False, ncol=2)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def save_per_class_f1_bars(per_class_f1: Dict[str, np.ndarray], save_path: Path) -> None:
    plt.figure(figsize=(15, 8))
    models = list(per_class_f1.keys())
    classes = list(range(12))
    x = np.arange(len(classes))
    width = 0.8 / max(1, len(models))

    for i, model in enumerate(models):
        plt.bar(x + i * width, per_class_f1[model], width, label=model)

    plt.xlabel("Class")
    plt.ylabel("F1 Score")
    plt.title("Per-Class F1 Scores Comparison")
    plt.xticks(x + width * (len(models) - 1) / 2, classes)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
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

    plot_df = summary_df.sort_values(metric_col, ascending=False).copy()
    x = np.arange(len(plot_df))
    colors = plt.cm.tab20(np.linspace(0, 1, max(1, len(plot_df))))

    plt.figure(figsize=(max(10, len(plot_df) * 0.7), 6.5))
    plt.bar(
        x,
        plot_df[metric_col].to_numpy(dtype=np.float64),
        yerr=plot_df[error_col].fillna(0.0).to_numpy(dtype=np.float64),
        color=colors[: len(plot_df)],
        edgecolor="black",
        linewidth=0.5,
        capsize=4,
    )
    plt.xticks(x, plot_df["model"].tolist(), rotation=30, ha="right")
    plt.ylabel(ylabel)
    plt.title(title)
    if "accuracy" in metric_col or "f1" in metric_col or "precision" in metric_col or "recall" in metric_col:
        plt.ylim(0.0, 1.0)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def save_scatter(results: Dict[str, Dict[str, float]], save_path: Path) -> None:
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection="3d")

    categories: Dict[str, Dict[str, List[float]]] = {}
    for name, result in results.items():
        cat = result["category"]
        if cat not in categories:
            categories[cat] = {"x": [], "y": [], "z": [], "names": []}
        categories[cat]["x"].append(result["accuracy"] * 100.0)
        categories[cat]["y"].append(result["inference_ms"])
        categories[cat]["z"].append(result["params_m"])
        categories[cat]["names"].append(name)

    colors = {"cnn": "blue", "temporal": "red", "attention": "green", "traditional": "orange"}
    markers = {"cnn": "o", "temporal": "s", "attention": "^", "traditional": "D"}

    for cat, data in categories.items():
        ax.scatter(
            data["x"],
            data["y"],
            data["z"],
            c=colors.get(cat, "black"),
            marker=markers.get(cat, "o"),
            label=cat,
            s=100,
            alpha=0.7,
        )
        for i, name in enumerate(data["names"]):
            ax.text(data["x"][i], data["y"][i], data["z"][i], name, fontsize=8)

    ax.set_xlabel("Accuracy (%)")
    ax.set_ylabel("Inference Time (ms)")
    ax.set_zlabel("Parameters (M)")
    ax.set_title("Model Comparison: Accuracy vs Inference Time vs Parameters")
    ax.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def save_radar_top3(results: Dict[str, Dict[str, float]], save_path: Path) -> None:
    sorted_results = sorted(results.items(), key=lambda x: x[1]["accuracy"], reverse=True)[:3]
    metrics = ["accuracy", "macro_f1", "macro_precision", "macro_recall"]
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection="polar"))
    colors = ["blue", "red", "green"]
    for i, (name, result) in enumerate(sorted_results):
        values = [result[m] for m in metrics]
        values += values[:1]
        ax.plot(angles, values, "o-", linewidth=2, label=name, color=colors[i % len(colors)])
        ax.fill(angles, values, alpha=0.25, color=colors[i % len(colors)])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1)
    ax.set_title("Top 3 Models Performance Comparison")
    ax.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


__all__ = [
    "benchmark_sklearn",
    "benchmark_torch",
    "benchmark_torch_gpu_deploy",
    "benchmark_torch_model_only",
    "choose_device_or_exit",
    "confusion_matrix_np",
    "count_parameters",
    "count_sklearn_params",
    "dataframe_to_results_dict",
    "evaluate_torch",
    "evaluate_torch_val_loss",
    "merge_metrics_csvs",
    "parse_model_list",
    "per_class_prf",
    "results_to_csv",
    "save_confusion_comparison",
    "save_confusion_matrix",
    "save_per_class_f1_bars",
    "save_radar_top3",
    "save_scatter",
    "save_summary_bar_with_error",
    "save_training_curves",
    "save_training_curves_with_std",
    "set_seed",
    "subset_to_numpy",
    "train_torch_model",
]
