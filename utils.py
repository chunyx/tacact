from __future__ import annotations

import copy
import random
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

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
    progress_callback: Optional[Callable[[Dict[str, float]], None]] = None,
) -> Dict[str, List[float]]:
    """Train a torch model for fixed epochs (early stopping disabled)."""
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
        "val_f1": [],
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
        val_acc, val_f1 = evaluate_torch(model, val_loader, device)
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
        # Early stopping is intentionally disabled: always run the full epoch budget.
        stop_triggered = False

        epoch_loss = loss_sum / max(1, n_batches)
        epoch_time_s = time.perf_counter() - epoch_start
        cum_time_s = (history["cum_time_s"][-1] if history["cum_time_s"] else 0.0) + epoch_time_s
        history["train_loss"].append(epoch_loss)
        history["val_loss"].append(float(val_loss))
        history["val_acc"].append(val_acc)
        history["val_f1"].append(val_f1)
        history["epoch_time_s"].append(float(epoch_time_s))
        history["cum_time_s"].append(float(cum_time_s))
        history["best_val_loss"].append(float(best_val_loss))
        history["best_epoch"].append(int(best_epoch))
        history["current_GL"].append(float(current_gl))
        history["stop_triggered"].append(bool(stop_triggered))
        history["alpha"].append(float(gl_alpha))

        current_lr = opt.param_groups[0]["lr"]
        if progress_callback is not None:
            try:
                progress_callback(
                    {
                        "epoch": float(epoch + 1),
                        "total_epochs": float(epochs),
                        "train_loss": float(epoch_loss),
                        "val_loss": float(val_loss),
                        "val_acc": float(val_acc),
                        "val_f1": float(val_f1),
                        "best_val_loss": float(best_val_loss),
                        "best_epoch": float(best_epoch),
                        "gl": float(current_gl),
                        "lr": float(current_lr),
                        "stop_triggered": float(1.0 if stop_triggered else 0.0),
                    }
                )
            except Exception:
                # Progress reporting should never break training.
                pass
        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"TrainLoss: {epoch_loss:.4f} | "
            f"ValLoss: {float(val_loss):.6f} | "
            f"BestValLoss: {float(best_val_loss):.6f} | "
            f"ValAcc: {val_acc * 100:.2f}% | "
            f"ValF1: {val_f1 * 100:.2f}% | "
            f"GL: {float(current_gl):.4f} | "
            f"Alpha: {float(gl_alpha):.2f} | "
            f"Stop: {stop_triggered} | "
            f"LR: {current_lr:.6f}"
        )

        # No early-stop break here by design.

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


_DEEP_FAMILY_BY_MODEL = {
    "LeNet": "CNN",
    "AlexNet": "CNN",
    "ResNet18": "CNN",
    "MobileNet_V2": "CNN",
    "EfficientNet_B0": "CNN",
    "LSTM": "Temporal",
    "GRU": "Temporal",
    "CNN_LSTM": "Temporal",
    "TCN": "Temporal",
    "ViT": "Transformer",
}

_DEEP_FAMILY_ORDER = {
    "CNN": ["LeNet", "AlexNet", "ResNet18", "MobileNet_V2", "EfficientNet_B0"],
    "Temporal": ["LSTM", "GRU", "CNN_LSTM", "TCN"],
    "Transformer": ["ViT"],
}

_DEEP_FAMILY_CMAP = {
    "CNN": plt.cm.Blues,
    "Temporal": plt.cm.Oranges,
    "Transformer": plt.cm.Greens,
}


def _deep_model_names_from_histories(histories: Dict[str, Any]) -> List[str]:
    return [name for name in histories.keys() if name in _DEEP_FAMILY_BY_MODEL]


def _deep_model_color(model_name: str) -> tuple:
    family = _DEEP_FAMILY_BY_MODEL.get(model_name, "CNN")
    ordered = _DEEP_FAMILY_ORDER.get(family, [model_name])
    try:
        idx = ordered.index(model_name)
    except ValueError:
        idx = 0
    cmap = _DEEP_FAMILY_CMAP.get(family, plt.cm.tab20)
    if len(ordered) == 1:
        return cmap(0.65)
    pos = np.linspace(0.45, 0.9, len(ordered))[idx]
    return cmap(float(pos))


def _loss_overlay_base(
    curves_by_model: Dict[str, np.ndarray],
    save_path: Path,
    *,
    title: str,
    ylabel: str = "Loss",
    with_std: Dict[str, np.ndarray] | None = None,
) -> None:
    model_names = [name for name in curves_by_model.keys() if name in _DEEP_FAMILY_BY_MODEL]
    if not model_names:
        return

    ordered_names: List[str] = []
    for family in ["CNN", "Temporal", "Transformer"]:
        ordered_names.extend([n for n in _DEEP_FAMILY_ORDER[family] if n in model_names])
    ordered_names.extend([n for n in model_names if n not in ordered_names])

    plt.figure(figsize=(14, 8))
    for model_name in ordered_names:
        curve = np.asarray(curves_by_model.get(model_name, []), dtype=np.float64)
        if curve.size == 0:
            continue
        epochs = np.arange(1, len(curve) + 1, dtype=np.int32)
        color = _deep_model_color(model_name)
        plt.plot(epochs, curve, label=model_name, color=color, linewidth=2.2, alpha=0.95)
        if with_std is not None and model_name in with_std:
            std_curve = np.asarray(with_std[model_name], dtype=np.float64)
            if std_curve.size == curve.size:
                plt.fill_between(epochs, curve - std_curve, curve + std_curve, color=color, alpha=0.14)

    family_handles = []
    for family in ["CNN", "Temporal", "Transformer"]:
        family_models = [m for m in ordered_names if _DEEP_FAMILY_BY_MODEL.get(m) == family]
        if not family_models:
            continue
        family_handles.append(plt.Line2D([0], [0], color=_deep_model_color(family_models[0]), linewidth=3, label=family))

    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.25)
    legend1 = plt.legend(frameon=False, ncol=2, fontsize=10, loc="upper right", title="Models")
    plt.gca().add_artist(legend1)
    if family_handles:
        plt.legend(handles=family_handles, frameon=False, loc="upper center", ncol=len(family_handles), title="Families")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def save_all_models_loss_overlay(
    histories: Dict[str, Dict[str, List[float]]],
    save_path: Path,
    loss_key: str = "train_loss",
    title: str = "All Models Loss vs Epoch",
) -> None:
    curves_by_model: Dict[str, np.ndarray] = {}
    for model_name in _deep_model_names_from_histories(histories):
        history = histories.get(model_name, {})
        curve = np.asarray(history.get(loss_key, []), dtype=np.float64)
        if curve.size > 0:
            curves_by_model[model_name] = curve
    _loss_overlay_base(curves_by_model, save_path, title=title, ylabel="Loss")


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


def save_all_models_loss_overlay_with_std(
    histories_by_model: Dict[str, List[Dict[str, List[float]]]],
    save_path: Path,
    loss_key: str = "train_loss",
    title: str = "All Models Loss vs Epoch Across Seeds",
) -> None:
    if not histories_by_model:
        return

    mean_curves: Dict[str, np.ndarray] = {}
    std_curves: Dict[str, np.ndarray] = {}
    for model_name in _deep_model_names_from_histories(histories_by_model):
        runs = histories_by_model.get(model_name, [])
        if not runs:
            continue
        max_len = max(len(run.get(loss_key, [])) for run in runs)
        if max_len == 0:
            continue
        mat = np.full((len(runs), max_len), np.nan, dtype=np.float64)
        for ridx, run in enumerate(runs):
            curve = np.asarray(run.get(loss_key, []), dtype=np.float64)
            mat[ridx, :len(curve)] = curve
        mean_curves[model_name] = np.nanmean(mat, axis=0)
        std_curves[model_name] = np.nanstd(mat, axis=0)

    _loss_overlay_base(mean_curves, save_path, title=title, ylabel="Loss", with_std=std_curves)


def save_convergence_diagnostics(
    histories: Dict[str, Dict[str, List[float]]],
    save_path: Path,
) -> None:
    if not histories:
        return

    model_names = list(histories.keys())
    n_models = len(model_names)
    ncols = 2 if n_models > 1 else 1
    nrows = int(np.ceil(n_models / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(7.4 * ncols, 3.8 * nrows), squeeze=False)
    axes_flat = axes.flatten()

    for idx, model_name in enumerate(model_names):
        history = histories[model_name]
        ax = axes_flat[idx]
        epochs = np.arange(1, len(history.get("val_f1", [])) + 1, dtype=np.int32)
        if len(epochs) == 0:
            ax.set_title(f"{model_name} (no history)")
            ax.axis("off")
            continue

        val_f1 = np.asarray(history.get("val_f1", []), dtype=np.float64)
        val_acc = np.asarray(history.get("val_acc", []), dtype=np.float64)
        val_loss = np.asarray(history.get("val_loss", []), dtype=np.float64)
        best_epoch_hist = history.get("best_epoch", [])
        best_epoch = int(best_epoch_hist[-1]) if best_epoch_hist else int(np.nanargmax(val_f1) + 1)

        ax.plot(epochs, val_f1 * 100.0, color="#1f77b4", linewidth=2.2, marker="o", markersize=3.5, label="Val F1")
        if len(val_acc) == len(epochs):
            ax.plot(
                epochs,
                val_acc * 100.0,
                color="#2ca02c",
                linewidth=1.6,
                linestyle="--",
                alpha=0.9,
                label="Val Acc",
            )
        ax.axvline(best_epoch, color="#444444", linestyle=":", linewidth=1.4, alpha=0.8, label="Best Epoch")
        ax.set_title(f"{model_name} | best epoch={best_epoch}", fontsize=11)
        ax.set_xlabel("Epoch", fontsize=10)
        ax.set_ylabel("Score (%)", fontsize=10)
        ax.grid(True, alpha=0.25)
        ax.tick_params(labelsize=9)
        ax.set_xlim(1, len(epochs))

        if len(val_loss) == len(epochs):
            ax2 = ax.twinx()
            ax2.plot(
                epochs,
                val_loss,
                color="#ff7f0e",
                linewidth=1.3,
                alpha=0.75,
                label="Val Loss",
            )
            ax2.set_ylabel("Val Loss", fontsize=10, color="#ff7f0e")
            ax2.tick_params(axis="y", labelsize=8, colors="#ff7f0e")

            lines_1, labels_1 = ax.get_legend_handles_labels()
            lines_2, labels_2 = ax2.get_legend_handles_labels()
            ax.legend(lines_1 + lines_2, labels_1 + labels_2, loc="best", fontsize=8, frameon=False)
        else:
            ax.legend(loc="best", fontsize=8, frameon=False)

    for idx in range(n_models, len(axes_flat)):
        axes_flat[idx].axis("off")

    fig.suptitle("Model Convergence Diagnostics", fontsize=14, y=0.995)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def save_convergence_diagnostics_with_std(
    histories_by_model: Dict[str, List[Dict[str, List[float]]]],
    save_path: Path,
) -> None:
    if not histories_by_model:
        return

    model_names = list(histories_by_model.keys())
    n_models = len(model_names)
    ncols = 2 if n_models > 1 else 1
    nrows = int(np.ceil(n_models / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(7.4 * ncols, 3.8 * nrows), squeeze=False)
    axes_flat = axes.flatten()

    for idx, model_name in enumerate(model_names):
        runs = histories_by_model[model_name]
        ax = axes_flat[idx]
        if not runs:
            ax.set_title(f"{model_name} (no history)")
            ax.axis("off")
            continue

        max_len = max(len(run.get("val_f1", [])) for run in runs)
        if max_len == 0:
            ax.set_title(f"{model_name} (no val_f1)")
            ax.axis("off")
            continue

        f1_mat = np.full((len(runs), max_len), np.nan, dtype=np.float64)
        acc_mat = np.full((len(runs), max_len), np.nan, dtype=np.float64)
        for ridx, run in enumerate(runs):
            val_f1 = np.asarray(run.get("val_f1", []), dtype=np.float64)
            val_acc = np.asarray(run.get("val_acc", []), dtype=np.float64)
            f1_mat[ridx, :len(val_f1)] = val_f1
            acc_mat[ridx, :len(val_acc)] = val_acc

        f1_mean = np.nanmean(f1_mat, axis=0)
        f1_std = np.nanstd(f1_mat, axis=0)
        acc_mean = np.nanmean(acc_mat, axis=0)
        epochs = np.arange(1, len(f1_mean) + 1)
        best_epoch = int(np.nanargmax(f1_mean) + 1)

        ax.plot(epochs, f1_mean * 100.0, color="#1f77b4", linewidth=2.2, label="Val F1 mean")
        ax.fill_between(
            epochs,
            (f1_mean - f1_std) * 100.0,
            (f1_mean + f1_std) * 100.0,
            color="#1f77b4",
            alpha=0.18,
            label="Val F1 ± std",
        )
        if not np.isnan(acc_mean).all():
            ax.plot(
                epochs,
                acc_mean * 100.0,
                color="#2ca02c",
                linewidth=1.6,
                linestyle="--",
                alpha=0.9,
                label="Val Acc mean",
            )
        ax.axvline(best_epoch, color="#444444", linestyle=":", linewidth=1.4, alpha=0.8, label="Best Mean Epoch")
        ax.set_title(f"{model_name} | best mean epoch={best_epoch}", fontsize=11)
        ax.set_xlabel("Epoch", fontsize=10)
        ax.set_ylabel("Score (%)", fontsize=10)
        ax.grid(True, alpha=0.25)
        ax.tick_params(labelsize=9)
        ax.legend(loc="best", fontsize=8, frameon=False)

    for idx in range(n_models, len(axes_flat)):
        axes_flat[idx].axis("off")

    fig.suptitle("Model Convergence Diagnostics Across Seeds", fontsize=14, y=0.995)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
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


def _benchmark_plot_arrays(results: Dict[str, Dict[str, float]]):
    names = list(results.keys())
    categories = [str(results[n].get("category", "unknown")) for n in names]
    acc = np.array([float(results[n].get("accuracy", 0.0)) for n in names], dtype=np.float64) * 100.0
    f1 = np.array([float(results[n].get("macro_f1", 0.0)) for n in names], dtype=np.float64) * 100.0
    inf_ms = np.array([float(results[n].get("inference_ms", 0.0)) for n in names], dtype=np.float64)
    params_m = np.array([float(results[n].get("params_m", 0.0)) for n in names], dtype=np.float64)
    train_s = np.array([float(results[n].get("training_seconds", 0.0)) for n in names], dtype=np.float64)
    return names, categories, acc, f1, inf_ms, params_m, train_s


def _category_color(cat: str) -> str:
    palette = {
        "cnn": "#1f77b4",
        "temporal": "#d62728",
        "attention": "#2ca02c",
        "traditional": "#ff7f0e",
    }
    return palette.get(cat, "#7f7f7f")


def _bubble_sizes(params_m: np.ndarray) -> np.ndarray:
    return 120.0 + 280.0 * np.sqrt(np.clip(params_m, 0.0, None))


def save_accuracy_vs_inference_bubble(results: Dict[str, Dict[str, float]], save_path: Path) -> None:
    names, categories, acc, _, inf_ms, params_m, _ = _benchmark_plot_arrays(results)
    plt.figure(figsize=(10, 7))
    for i, name in enumerate(names):
        c = _category_color(categories[i])
        s = _bubble_sizes(params_m[i : i + 1])[0]
        plt.scatter(inf_ms[i], acc[i], s=s, color=c, alpha=0.75, edgecolors="black", linewidths=0.5)
        plt.text(inf_ms[i], acc[i], name, fontsize=8, ha="left", va="bottom")
    plt.xlabel("Inference Time (ms/sample)")
    plt.ylabel("Accuracy (%)")
    plt.title("Benchmark: Accuracy vs Inference Time (Bubble size = Params)")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def save_accuracy_vs_params_scatter(results: Dict[str, Dict[str, float]], save_path: Path) -> None:
    names, categories, acc, _, _, params_m, _ = _benchmark_plot_arrays(results)
    plt.figure(figsize=(10, 7))
    for i, name in enumerate(names):
        c = _category_color(categories[i])
        plt.scatter(params_m[i], acc[i], color=c, s=90, alpha=0.8, edgecolors="black", linewidths=0.5)
        plt.text(params_m[i], acc[i], name, fontsize=8, ha="left", va="bottom")
    plt.xlabel("Parameters (M)")
    plt.ylabel("Accuracy (%)")
    plt.title("Benchmark: Accuracy vs Model Complexity")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def save_macrof1_vs_inference_bubble(results: Dict[str, Dict[str, float]], save_path: Path) -> None:
    names, categories, _, f1, inf_ms, params_m, _ = _benchmark_plot_arrays(results)
    plt.figure(figsize=(10, 7))
    for i, name in enumerate(names):
        c = _category_color(categories[i])
        s = _bubble_sizes(params_m[i : i + 1])[0]
        plt.scatter(inf_ms[i], f1[i], s=s, color=c, alpha=0.75, edgecolors="black", linewidths=0.5)
        plt.text(inf_ms[i], f1[i], name, fontsize=8, ha="left", va="bottom")
    plt.xlabel("Inference Time (ms/sample)")
    plt.ylabel("Macro-F1 (%)")
    plt.title("Benchmark: Macro-F1 vs Inference Time (Bubble size = Params)")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def save_efficiency_score_bar(results: Dict[str, Dict[str, float]], save_path: Path) -> None:
    names, categories, acc, _, inf_ms, _, _ = _benchmark_plot_arrays(results)
    # Explicit formula: efficiency = accuracy(%) / inference_time(ms)
    eff = acc / np.maximum(inf_ms, 1e-9)
    order = np.argsort(-eff)
    plt.figure(figsize=(max(10, len(names) * 0.7), 6.5))
    xs = np.arange(len(names))
    colors = [_category_color(categories[i]) for i in order]
    plt.bar(xs, eff[order], color=colors, edgecolor="black", linewidth=0.5)
    plt.xticks(xs, [names[i] for i in order], rotation=30, ha="right")
    plt.ylabel("Efficiency Score (accuracy% / ms)")
    plt.title("Benchmark Efficiency Score")
    plt.grid(True, axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def save_pareto_accuracy_inference(results: Dict[str, Dict[str, float]], save_path: Path) -> None:
    names, categories, acc, _, inf_ms, _, _ = _benchmark_plot_arrays(results)
    plt.figure(figsize=(10, 7))
    for i, name in enumerate(names):
        c = _category_color(categories[i])
        plt.scatter(inf_ms[i], acc[i], color=c, s=90, alpha=0.8, edgecolors="black", linewidths=0.5)
        plt.text(inf_ms[i], acc[i], name, fontsize=8, ha="left", va="bottom")

    # Pareto frontier: minimize inference, maximize accuracy
    points = sorted(zip(inf_ms.tolist(), acc.tolist()))
    frontier_x: List[float] = []
    frontier_y: List[float] = []
    best_acc = -np.inf
    for x, y in points:
        if y > best_acc:
            frontier_x.append(x)
            frontier_y.append(y)
            best_acc = y
    if frontier_x:
        plt.plot(frontier_x, frontier_y, color="black", linewidth=2.0, linestyle="--", label="Pareto frontier")
        plt.legend(frameon=False)

    plt.xlabel("Inference Time (ms/sample)")
    plt.ylabel("Accuracy (%)")
    plt.title("Benchmark: Pareto Frontier (Accuracy vs Inference)")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def save_accuracy_vs_training_time(results: Dict[str, Dict[str, float]], save_path: Path) -> None:
    names, categories, acc, _, _, _, train_s = _benchmark_plot_arrays(results)
    plt.figure(figsize=(10, 7))
    for i, name in enumerate(names):
        c = _category_color(categories[i])
        plt.scatter(train_s[i], acc[i], color=c, s=90, alpha=0.8, edgecolors="black", linewidths=0.5)
        plt.text(train_s[i], acc[i], name, fontsize=8, ha="left", va="bottom")
    plt.xlabel("Training Time (s)")
    plt.ylabel("Accuracy (%)")
    plt.title("Benchmark: Accuracy vs Training Time")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def save_macrof1_vs_params_scatter(results: Dict[str, Dict[str, float]], save_path: Path) -> None:
    names, categories, _, f1, _, params_m, _ = _benchmark_plot_arrays(results)
    plt.figure(figsize=(10, 7))
    for i, name in enumerate(names):
        c = _category_color(categories[i])
        plt.scatter(params_m[i], f1[i], color=c, s=90, alpha=0.8, edgecolors="black", linewidths=0.5)
        plt.text(params_m[i], f1[i], name, fontsize=8, ha="left", va="bottom")
    plt.xlabel("Parameters (M)")
    plt.ylabel("Macro-F1 (%)")
    plt.title("Benchmark: Macro-F1 vs Model Complexity")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def _family_label(category: str) -> str:
    c = str(category).lower()
    if c == "cnn":
        return "CNN"
    if c == "temporal":
        return "Temporal"
    if c == "attention":
        return "Transformer"
    if c == "traditional":
        return "Traditional"
    return "Other"


def _pareto_frontier_xy(xs: np.ndarray, ys: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    # Optimize for lower x (faster) and higher y (better score).
    order = np.argsort(xs)
    xs_sorted = xs[order]
    ys_sorted = ys[order]
    fx: List[float] = []
    fy: List[float] = []
    best_y = -np.inf
    for x, y in zip(xs_sorted, ys_sorted):
        if y > best_y:
            fx.append(float(x))
            fy.append(float(y))
            best_y = y
    return np.asarray(fx, dtype=np.float64), np.asarray(fy, dtype=np.float64)


def _plot_tradeoff_base(
    xs: np.ndarray,
    ys: np.ndarray,
    names: List[str],
    categories: List[str],
    save_path: Path,
    *,
    title: str,
    xlabel: str,
    ylabel: str,
    use_log_x: bool = False,
    show_pareto: bool = False,
    dpi: int = 300,
) -> None:
    plt.figure(figsize=(10, 7))
    for i, name in enumerate(names):
        color = _category_color(categories[i])
        marker = {"cnn": "o", "temporal": "s", "attention": "^", "traditional": "D"}.get(categories[i], "o")
        plt.scatter(xs[i], ys[i], color=color, marker=marker, s=95, alpha=0.85, edgecolors="black", linewidths=0.5)
        plt.text(xs[i], ys[i], name, fontsize=8, ha="left", va="bottom")

    if show_pareto:
        fx, fy = _pareto_frontier_xy(xs=xs, ys=ys)
        if len(fx) > 0:
            plt.plot(fx, fy, color="black", linestyle="--", linewidth=2.0, label="Pareto frontier")
            plt.legend(frameon=False)

    if use_log_x:
        plt.xscale("log")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close()


def save_dl_pareto_macrof1_vs_inference(results: Dict[str, Dict[str, float]], save_path: Path) -> None:
    names, categories, _, f1, inf_ms, _, _ = _benchmark_plot_arrays(results)
    if len(names) == 0:
        return
    use_log = bool(np.nanmax(inf_ms) / max(1e-9, np.nanmin(inf_ms)) > 8.0)
    _plot_tradeoff_base(
        xs=inf_ms,
        ys=f1,
        names=names,
        categories=categories,
        save_path=save_path,
        title="DL Trade-Off: Macro-F1 vs Inference Time",
        xlabel="Inference Time (ms/sample)",
        ylabel="Macro-F1 (%)",
        use_log_x=use_log,
        show_pareto=True,
        dpi=300,
    )


def save_dl_family_tradeoff(results: Dict[str, Dict[str, float]], save_path: Path) -> None:
    names, categories, _, f1, inf_ms, _, _ = _benchmark_plot_arrays(results)
    if len(names) == 0:
        return
    plt.figure(figsize=(10, 7))
    fams = [_family_label(c) for c in categories]
    fam_order = ["CNN", "Temporal", "Transformer", "Traditional", "Other"]
    fam_color = {
        "CNN": "#1f77b4",
        "Temporal": "#d62728",
        "Transformer": "#2ca02c",
        "Traditional": "#ff7f0e",
        "Other": "#7f7f7f",
    }
    fam_marker = {"CNN": "o", "Temporal": "s", "Transformer": "^", "Traditional": "D", "Other": "o"}
    for fam in fam_order:
        idx = [i for i, f in enumerate(fams) if f == fam]
        if not idx:
            continue
        plt.scatter(
            inf_ms[idx],
            f1[idx],
            s=95,
            c=fam_color[fam],
            marker=fam_marker[fam],
            edgecolors="black",
            linewidths=0.5,
            alpha=0.85,
            label=fam,
        )
        for i in idx:
            plt.text(inf_ms[i], f1[i], names[i], fontsize=8, ha="left", va="bottom")
    use_log = bool(np.nanmax(inf_ms) / max(1e-9, np.nanmin(inf_ms)) > 8.0)
    if use_log:
        plt.xscale("log")
    plt.xlabel("Inference Time (ms/sample)")
    plt.ylabel("Macro-F1 (%)")
    plt.title("DL Model Family Trade-Off")
    plt.legend(frameon=False)
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def save_dl_macrof1_vs_training_time(results: Dict[str, Dict[str, float]], save_path: Path) -> None:
    names, categories, _, f1, _, _, train_s = _benchmark_plot_arrays(results)
    if len(names) == 0:
        return
    use_log = bool(np.nanmax(train_s) / max(1e-9, np.nanmin(train_s)) > 8.0)
    _plot_tradeoff_base(
        xs=train_s,
        ys=f1,
        names=names,
        categories=categories,
        save_path=save_path,
        title="DL Training Budget Trade-Off: Macro-F1 vs Training Time",
        xlabel="Training Time (s)",
        ylabel="Macro-F1 (%)",
        use_log_x=use_log,
        show_pareto=False,
        dpi=300,
    )


def save_dl_params_vs_inference(results: Dict[str, Dict[str, float]], save_path: Path) -> None:
    names, categories, _, _, inf_ms, params_m, _ = _benchmark_plot_arrays(results)
    if len(names) == 0:
        return
    use_log_x = bool(np.nanmax(params_m) / max(1e-9, np.nanmin(params_m)) > 8.0)
    use_log_y = bool(np.nanmax(inf_ms) / max(1e-9, np.nanmin(inf_ms)) > 8.0)
    plt.figure(figsize=(10, 7))
    for i, name in enumerate(names):
        c = _category_color(categories[i])
        marker = {"cnn": "o", "temporal": "s", "attention": "^", "traditional": "D"}.get(categories[i], "o")
        plt.scatter(params_m[i], inf_ms[i], color=c, marker=marker, s=95, alpha=0.85, edgecolors="black", linewidths=0.5)
        plt.text(params_m[i], inf_ms[i], name, fontsize=8, ha="left", va="bottom")
    if use_log_x:
        plt.xscale("log")
    if use_log_y:
        plt.yscale("log")
    plt.xlabel("Parameters (M)")
    plt.ylabel("Inference Time (ms/sample)")
    plt.title("DL Runtime vs Parameter Count")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def save_dl_performance_vs_sequence_length(
    df: pd.DataFrame,
    save_path: Path,
    *,
    metric_col: str = "macro_f1",
    model_col: str = "model",
    category_col: str = "category",
) -> bool:
    seq_col: Optional[str] = None
    for cand in ["sequence_length", "n_frames", "seq_len", "window_size", "N"]:
        if cand in df.columns:
            seq_col = cand
            break
    if seq_col is None or metric_col not in df.columns or model_col not in df.columns:
        return False
    plot_df = df.copy()
    plot_df = plot_df[plot_df[metric_col].notna()]
    if category_col in plot_df.columns:
        plot_df = plot_df[plot_df[category_col].astype(str).str.lower() != "traditional"]
    if plot_df.empty:
        return False
    grouped = (
        plot_df.groupby([model_col, seq_col], as_index=False)[metric_col]
        .mean()
        .sort_values([model_col, seq_col])
    )
    plt.figure(figsize=(10, 7))
    for model_name, g in grouped.groupby(model_col):
        x = g[seq_col].to_numpy(dtype=np.float64)
        y = g[metric_col].to_numpy(dtype=np.float64)
        plt.plot(x, y * 100.0, marker="o", linewidth=2.0, label=str(model_name))
    plt.xlabel("Sequence Length (N)")
    plt.ylabel("Macro-F1 (%)")
    plt.title("DL Performance vs Sequence Length")
    plt.grid(True, alpha=0.25)
    plt.legend(frameon=False, ncol=2)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    return True


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
    "save_accuracy_vs_inference_bubble",
    "save_accuracy_vs_params_scatter",
    "save_macrof1_vs_inference_bubble",
    "save_efficiency_score_bar",
    "save_pareto_accuracy_inference",
    "save_accuracy_vs_training_time",
    "save_macrof1_vs_params_scatter",
    "save_dl_pareto_macrof1_vs_inference",
    "save_dl_family_tradeoff",
    "save_dl_macrof1_vs_training_time",
    "save_dl_params_vs_inference",
    "save_dl_performance_vs_sequence_length",
    "save_scatter",
    "save_summary_bar_with_error",
    "save_all_models_loss_overlay",
    "save_all_models_loss_overlay_with_std",
    "save_training_curves",
    "save_training_curves_with_std",
    "set_seed",
    "subset_to_numpy",
    "train_torch_model",
]
