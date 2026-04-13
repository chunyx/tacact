#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from torch.utils.data import Subset

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tacact.models import ModelFactory
from tacact.benchmark_common import (
    create_optimized_dataset,
    get_device,
    make_three_loaders,
    split_indices_3way,
    warmup_cache,
)

# 统一从新建的 tacact.utils 导入所有工具和绘图函数
from tacact.utils import (
    benchmark_sklearn,
    benchmark_torch_gpu_deploy,
    benchmark_torch_model_only,
    confusion_matrix_np,
    count_parameters,
    count_sklearn_params,
    dataframe_to_results_dict,
    merge_metrics_csvs,
    parse_model_list,
    per_class_prf,
    set_seed,
    subset_to_numpy,
    train_torch_model,
    save_confusion_comparison,
    save_confusion_matrix,
    save_per_class_f1_bars,
    save_radar_top3,
    save_accuracy_vs_inference_bubble,
    save_accuracy_vs_params_scatter,
    save_macrof1_vs_inference_bubble,
    save_efficiency_score_bar,
    save_pareto_accuracy_inference,
    save_accuracy_vs_training_time,
    save_macrof1_vs_params_scatter,
    save_dl_pareto_macrof1_vs_inference,
    save_dl_family_tradeoff,
    save_dl_macrof1_vs_training_time,
    save_dl_params_vs_inference,
    save_dl_performance_vs_sequence_length,
    save_scatter,
    save_summary_bar_with_error,
    save_training_curves,
    save_training_curves_with_std,
    save_all_models_loss_overlay,
    save_all_models_loss_overlay_with_std,
    save_convergence_diagnostics,
    save_convergence_diagnostics_with_std,
)


def _write_status(path: Path | None, payload: Dict[str, Any]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.json")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def _split_results_for_paper(
    results: Dict[str, Dict[str, float]],
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, Dict[str, float]]]:
    deep = {k: v for k, v in results.items() if str(v.get("category", "")).lower() != "traditional"}
    traditional = {k: v for k, v in results.items() if str(v.get("category", "")).lower() == "traditional"}
    return deep, traditional


def _save_traditional_baseline_table_and_plot(
    traditional_results: Dict[str, Dict[str, float]],
    save_dir: Path,
    suffix: str = "",
) -> None:
    if not traditional_results:
        return
    rows: List[Dict[str, float | str]] = []
    for model_name, m in traditional_results.items():
        rows.append(
            {
                "model": model_name,
                "accuracy": float(m.get("accuracy", 0.0)),
                "macro_f1": float(m.get("macro_f1", 0.0)),
                "category": "traditional",
            }
        )
    base_df = pd.DataFrame(rows).sort_values(["macro_f1", "accuracy"], ascending=[False, False]).reset_index(drop=True)
    base_df.to_csv(save_dir / f"traditional_baseline_metrics{suffix}.csv", index=False)

    display_names = {
        "RandomForest": "RF",
        "XGBoost": "XGB",
        "SVM": "SVM",
    }
    x_names = [display_names.get(str(x), str(x)) for x in base_df["model"].tolist()]
    x = np.arange(len(base_df))
    width = 0.35
    plt.figure(figsize=(max(6.0, len(x) * 1.2), 4.2))
    plt.bar(x - width / 2, base_df["accuracy"].to_numpy(dtype=np.float64) * 100.0, width=width, label="Accuracy")
    plt.bar(x + width / 2, base_df["macro_f1"].to_numpy(dtype=np.float64) * 100.0, width=width, label="Macro-F1")
    plt.xticks(x, x_names, fontsize=10)
    plt.yticks(fontsize=10)
    plt.ylabel("Score (%)", fontsize=11)
    plt.title("Traditional ML Baselines (Reference Only)", fontsize=12)
    plt.legend(fontsize=10, frameon=False)
    plt.grid(True, axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(save_dir / f"traditional_baseline_reference{suffix}.png", dpi=180, bbox_inches="tight")
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--train_ratio", type=float, default=0.7)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--clip_mode", choices=["weighted_center"], default="weighted_center")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--repeat_seeds", type=str, default="",
                        help="留空则单次运行(更快); 如 42,43,44 则多种子重复")
    parser.add_argument("--split_mode", choices=["subject", "random"], default="subject")
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--output_dir", type=Path, default=Path("outputs"))
    parser.add_argument("--cache_dir", type=Path, default=Path(".cache_tacact_n80_weighted"))
    parser.add_argument("--run_mode", choices=["all", "traditional", "deep"], default="all")
    parser.add_argument("--traditional_models", type=str, default="SVM,RandomForest,XGBoost")
    parser.add_argument("--deep_models", type=str, default="LeNet,ResNet18,EfficientNet_B0,LSTM,GRU,CNN_LSTM,TCN,ViT")
    parser.add_argument("--amp_infer", action="store_true")
    parser.add_argument("--bench_batch_sizes", type=str, default="1,32")
    parser.add_argument("--bench_iters", type=int, default=100)
    parser.add_argument("--merge_metrics_csvs", type=str, default="")
    parser.add_argument("--best_config_path", type=Path, default=None,
                        help="可选：读取自动搜索生成的 best_model_configs.json")
    parser.add_argument("--skip_cache_warmup", action="store_true",
                        help="跳过缓存预热(确信 .npy 已存在时使用)")
    parser.add_argument("--no_preload", action="store_true",
                        help="不预加载 24k 样本到内存，按需从磁盘读 .npy(省内存、省启动时间)")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.merge_metrics_csvs.strip():
        csv_paths = [Path(x.strip()) for x in args.merge_metrics_csvs.split(",") if x.strip()]
        merged_df = merge_metrics_csvs(csv_paths)
        merged_df.to_csv(args.output_dir / "metrics_merged.csv", index=False)
        merged_results = dataframe_to_results_dict(merged_df)
        deep_results, traditional_results = _split_results_for_paper(merged_results)

        # Main paper comparison: deep models only.
        if deep_results:
            # A) Legacy/original plots (kept for backward compatibility).
            save_scatter(deep_results, args.output_dir / "dl_main_accuracy_time_params_3d_merged.png")
            save_accuracy_vs_inference_bubble(
                deep_results, args.output_dir / "dl_main_accuracy_vs_inference_bubble_merged.png"
            )
            save_accuracy_vs_params_scatter(
                deep_results, args.output_dir / "dl_main_accuracy_vs_params_scatter_merged.png"
            )
            save_macrof1_vs_inference_bubble(
                deep_results, args.output_dir / "dl_main_macrof1_vs_inference_bubble_merged.png"
            )
            save_efficiency_score_bar(
                deep_results, args.output_dir / "dl_main_efficiency_score_bar_merged.png"
            )
            save_pareto_accuracy_inference(
                deep_results, args.output_dir / "dl_main_pareto_accuracy_inference_merged.png"
            )
            save_accuracy_vs_training_time(
                deep_results, args.output_dir / "dl_main_accuracy_vs_training_time_merged.png"
            )
            save_macrof1_vs_params_scatter(
                deep_results, args.output_dir / "dl_main_macrof1_vs_params_scatter_merged.png"
            )
            # B) New trade-off analysis plots (CVPR-style additions).
            save_dl_pareto_macrof1_vs_inference(
                deep_results, args.output_dir / "dl_pareto_macroF1_vs_inference_merged.png"
            )
            save_dl_family_tradeoff(
                deep_results, args.output_dir / "dl_family_tradeoff_merged.png"
            )
            save_dl_macrof1_vs_training_time(
                deep_results, args.output_dir / "dl_macroF1_vs_training_time_merged.png"
            )
            save_dl_params_vs_inference(
                deep_results, args.output_dir / "dl_params_vs_inference_merged.png"
            )
            merged_seq_ok = save_dl_performance_vs_sequence_length(
                merged_df,
                args.output_dir / "dl_performance_vs_sequence_length_merged.png",
                metric_col="macro_f1",
            )
            if not merged_seq_ok:
                print("[WARN] Skip dl_performance_vs_sequence_length_merged.png: sequence-length field not found.")

        # Traditional baselines: separate compact reference outputs.
        _save_traditional_baseline_table_and_plot(traditional_results, args.output_dir, suffix="_merged")

        print(f"Saved merged metrics: {args.output_dir / 'metrics_merged.csv'}")
        print(f"Saved merged DL-main plot: {args.output_dir / 'dl_main_accuracy_time_params_3d_merged.png'}")
        return

    best_config_map: Dict[str, Dict[str, Dict[str, Any]]] = {"traditional": {}, "deep": {}}
    if args.best_config_path is not None:
        payload = json.loads(args.best_config_path.read_text(encoding="utf-8"))
        best_config_map["traditional"] = payload.get("traditional", {})
        best_config_map["deep"] = payload.get("deep", {})
        print(f"Loaded best configs from: {args.best_config_path}")

    dataset = create_optimized_dataset(
        args.data_root,
        n_frames=80,
        clip_mode=args.clip_mode,
        cache_dir=args.cache_dir,
        preload_cache=not args.no_preload,
    )
    print(f"Found {len(dataset)} samples")
    if not args.skip_cache_warmup:
        print("正在使用多进程预构建/检查数据缓存，首次运行耗时较长，请耐心等待...")
        warmup_cache(
            dataset,
            batch_size=128,
            num_workers=args.num_workers,
            shuffle=False,
            pin_memory=False,
            use_tqdm=True,
            tqdm_desc="Building Cache",
        )
        print("缓存就绪！后续数据加载将以光速进行。")
    else:
        print("已跳过缓存预热（假定 .npy 已存在）")
    traditional_models = parse_model_list(args.traditional_models, ["SVM", "RandomForest", "XGBoost"])
    deep_models = parse_model_list(
        args.deep_models,
        ["LeNet", "AlexNet", "ResNet18", "MobileNet_V2", "EfficientNet_B0", "LSTM", "GRU", "CNN_LSTM", "TCN", "ViT"],
    )
    print(f"Run mode: {args.run_mode} | traditional={traditional_models} | deep={deep_models}")

    if args.repeat_seeds.strip():
        seed_list = [int(x.strip()) for x in args.repeat_seeds.split(",") if x.strip()]
    else:
        seed_list = [args.seed]

    # Optional per-process status reporting for external live dashboard.
    status_file_env = os.environ.get("TACACT_STATUS_FILE", "").strip()
    status_path = Path(status_file_env) if status_file_env else None
    gpu_slot = os.environ.get("TACACT_GPU_ID", "").strip()
    queue_models_env = os.environ.get("TACACT_QUEUE_MODELS", "").strip()
    queue_models = [x.strip() for x in queue_models_env.split(",") if x.strip()]
    queue_total = int(os.environ.get("TACACT_QUEUE_TOTAL", str(len(queue_models) if queue_models else 0)))
    if queue_total <= 0:
        queue_total = len(queue_models)
    _write_status(
        status_path,
        {
            "status": "starting",
            "gpu_id": gpu_slot,
            "pid": int(os.getpid()),
            "queue_models": queue_models,
            "queue_total": int(queue_total),
            "queue_completed": 0,
            "current_model": None,
            "current_model_index": 0,
            "current_epoch": 0,
            "total_epochs": int(args.epochs),
            "latest_val_f1": None,
            "seed": int(args.seed),
            "run_mode": str(args.run_mode),
            "last_update_ts": time.time(),
        },
    )

    bench_batch_sizes = [int(x.strip()) for x in args.bench_batch_sizes.split(",") if x.strip()]

    aggregated_rows: List[Dict[str, float]] = []
    aggregated_runtime_rows: List[Dict[str, float]] = []
    aggregated_histories: Dict[str, List[Dict[str, List[float]]]] = {}

    for run_seed in seed_list:
        set_seed(run_seed)
        run_out = args.output_dir / f"{args.split_mode}_seed{run_seed}"
        run_out.mkdir(parents=True, exist_ok=True)

        train_indices, val_indices, test_indices = split_indices_3way(
            dataset,
            split_mode=args.split_mode,
            seed=run_seed,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
        )
        train_set = Subset(dataset, train_indices)
        val_set = Subset(dataset, val_indices)
        test_set = Subset(dataset, test_indices)

        device = get_device()

        results: Dict[str, Dict[str, float]] = {}
        confusion_mats: Dict[str, np.ndarray] = {}
        histories: Dict[str, Dict[str, List[float]]] = {}
        per_class_f1: Dict[str, np.ndarray] = {}
        detail_rows: List[Dict[str, float]] = []
        traditional_runtime_s, deep_runtime_s = 0.0, 0.0

        if args.run_mode in ("all", "traditional"):
            x_train, y_train = subset_to_numpy(dataset, train_set)
            x_val, y_val = subset_to_numpy(dataset, val_set)
            x_test, y_test = subset_to_numpy(dataset, test_set)
            x_train_full = x_train
            y_train_full = y_train
        else:
            x_train_full = y_train_full = x_test = y_test = None

        t0_traditional = time.perf_counter()
        for model_name in traditional_models:
            if args.run_mode not in ("all", "traditional"):
                break
            try:
                traditional_cfg = best_config_map["traditional"].get(model_name, {}).get("params", {})
                if not traditional_cfg:
                    traditional_cfg = best_config_map["traditional"].get(model_name.lower(), {}).get("params", {})
                clf = ModelFactory.build_traditional(model_name, **traditional_cfg)
                fit_st = time.perf_counter()
                clf.fit(x_train_full, y_train_full)
                train_seconds = time.perf_counter() - fit_st
                pred = clf.predict(x_test)
                acc = float((pred == y_test).mean())
                p_cls, r_cls, f1_cls = per_class_prf(y_test, pred, n_classes=12)
                inf_ms = benchmark_sklearn(clf, x_test)
                params = float(count_sklearn_params(clf))
                results[model_name] = {
                    "category": "traditional",
                    "accuracy": acc,
                    "macro_f1": float(np.nanmean(f1_cls)),
                    "macro_precision": float(np.nanmean(p_cls)),
                    "macro_recall": float(np.nanmean(r_cls)),
                    "training_seconds": float(train_seconds),
                    "inference_ms": inf_ms,
                    "params": params,
                    "params_m": params / 1e6,
                }
                cm = confusion_matrix_np(y_test, pred, n_classes=12)
                confusion_mats[model_name] = cm
                per_class_f1[model_name] = f1_cls
                for c in range(12):
                    detail_rows.append(
                        {"model": model_name, "class": c, "precision": p_cls[c], "recall": r_cls[c],
                         "f1": f1_cls[c]})
                print(f"{model_name}: acc={acc * 100:.2f}%")
            except Exception as e:
                print(f"[WARN] Skip {model_name}: {e}")
        if args.run_mode in ("all", "traditional"):
            traditional_runtime_s = time.perf_counter() - t0_traditional

        t0_deep = time.perf_counter()
        deep_model_total = len(deep_models)
        deep_completed = 0
        for model_idx, model_name in enumerate(deep_models, start=1):
            if args.run_mode not in ("all", "deep"):
                break
            _write_status(
                status_path,
                {
                    "status": "running",
                    "gpu_id": gpu_slot,
                    "pid": int(os.getpid()),
                    "queue_models": queue_models,
                    "queue_total": int(queue_total if queue_total > 0 else deep_model_total),
                    "queue_completed": int(deep_completed),
                    "current_model": str(model_name),
                    "current_model_index": int(model_idx),
                    "current_epoch": 1,
                    "total_epochs": int(args.epochs),
                    "latest_val_f1": None,
                    "last_update_ts": time.time(),
                },
            )
            deep_cfg = best_config_map["deep"].get(model_name, {}).get("params", {})
            if not deep_cfg:
                deep_cfg = best_config_map["deep"].get(model_name.lower(), {}).get("params", {})
            if "batch_size" in deep_cfg and deep_cfg.get("batch_size") is not None:
                try:
                    model_batch_size = int(deep_cfg.get("batch_size"))
                except Exception:
                    model_batch_size = int(args.batch_size)
                if model_batch_size <= 0:
                    model_batch_size = int(args.batch_size)
                print(f"[MODEL={model_name}] Using batch_size={model_batch_size} (from best config)")
            else:
                model_batch_size = int(args.batch_size)
                print(f"[MODEL={model_name}] Using batch_size={model_batch_size} (default)")

            train_loader, val_loader, test_loader = make_three_loaders(
                train_set=train_set,
                val_set=val_set,
                test_set=test_set,
                batch_size=model_batch_size,
                num_workers=args.num_workers,
                pin_memory=True,
            )
            model_kwargs = {
                k: v for k, v in deep_cfg.items()
                if k in {"dim", "depth", "heads", "patch_size", "dropout", "num_channels", "lstm_hidden",
                         "hidden_size", "num_layers", "input_proj_dim", "use_last_only"}
            }
            train_kwargs = {
                "lr_override": deep_cfg.get("lr"),
                "weight_decay_override": deep_cfg.get("weight_decay"),
            }
            model, cat = ModelFactory.build_torch(model_name, **model_kwargs)
            def _epoch_progress_cb(p: Dict[str, float], _model_name: str = str(model_name), _model_idx: int = int(model_idx)) -> None:
                _write_status(
                    status_path,
                    {
                        "status": "running",
                        "gpu_id": gpu_slot,
                        "pid": int(os.getpid()),
                        "queue_models": queue_models,
                        "queue_total": int(queue_total if queue_total > 0 else deep_model_total),
                        "queue_completed": int(deep_completed),
                        "current_model": _model_name,
                        "current_model_index": _model_idx,
                        "current_epoch": int(p.get("epoch", 0.0)),
                        "total_epochs": int(p.get("total_epochs", float(args.epochs))),
                        "latest_val_f1": float(p.get("val_f1", float("nan"))),
                        "last_update_ts": time.time(),
                    },
                )
            histories[model_name] = train_torch_model(model, train_loader, val_loader, epochs=args.epochs,
                                                     device=device, progress_callback=_epoch_progress_cb, **train_kwargs)
            aggregated_histories.setdefault(model_name, []).append(histories[model_name])
            train_seconds = float(histories[model_name].get("cum_time_s", [0.0])[-1]) if histories[model_name].get("cum_time_s") else 0.0
            model.eval()
            ys = []
            ps = []
            with torch.no_grad():
                for x, y in test_loader:
                    pred = model(x.to(device)).argmax(dim=1).cpu().numpy()
                    ys.append(y.numpy())
                    ps.append(pred)
            y_true, y_pred = np.concatenate(ys), np.concatenate(ps)
            p_cls, r_cls, f1_cls = per_class_prf(y_true, y_pred, n_classes=12)
            params = float(count_parameters(model))
            deploy_bench: Dict[str, float] = {}
            try:
                sample_batch = next(iter(test_loader))[0]
                deploy_bench = benchmark_torch_gpu_deploy(
                    model=model,
                    sample_batch=sample_batch,
                    device=device,
                    batch_sizes=bench_batch_sizes,
                    iters=args.bench_iters,
                    warmup=max(10, args.bench_iters // 5),
                    amp=bool(args.amp_infer),
                )
            except Exception as e:
                print(f"[WARN] GPU deploy benchmark skipped for {model_name}: {e}")
            results[model_name] = {
                "category": cat,
                "accuracy": float((y_true == y_pred).mean()),
                "macro_f1": float(np.nanmean(f1_cls)),
                "macro_precision": float(np.nanmean(p_cls)),
                "macro_recall": float(np.nanmean(r_cls)),
                "training_seconds": train_seconds,
                "inference_ms": benchmark_torch_model_only(model, test_loader, device),
                "params": params,
                "params_m": params / 1e6,
                **deploy_bench,
            }
            cm = confusion_matrix_np(y_true, y_pred, n_classes=12)
            confusion_mats[model_name] = cm
            per_class_f1[model_name] = f1_cls
            for c in range(12):
                detail_rows.append(
                    {"model": model_name, "class": c, "precision": p_cls[c], "recall": r_cls[c], "f1": f1_cls[c]})
            print(f"{model_name}: acc={results[model_name]['accuracy'] * 100:.2f}%")
            deep_completed += 1
            _write_status(
                status_path,
                {
                    "status": "running",
                    "gpu_id": gpu_slot,
                    "pid": int(os.getpid()),
                    "queue_models": queue_models,
                    "queue_total": int(queue_total if queue_total > 0 else deep_model_total),
                    "queue_completed": int(deep_completed),
                    "current_model": str(model_name),
                    "current_model_index": int(model_idx),
                    "current_epoch": int(len(histories[model_name].get("train_loss", []))),
                    "total_epochs": int(args.epochs),
                    "latest_val_f1": float(np.nanmean(f1_cls)),
                    "last_update_ts": time.time(),
                },
            )

        if args.run_mode in ("all", "deep"):
            deep_runtime_s = time.perf_counter() - t0_deep

        if not results:
            raise RuntimeError("No model produced results.")

        df = pd.DataFrame.from_dict(results, orient="index")
        df.to_csv(run_out / "metrics.csv", index_label="model")
        pd.DataFrame([
            {"segment": "traditional", "seconds": traditional_runtime_s},
            {"segment": "deep", "seconds": deep_runtime_s},
            {"segment": "total", "seconds": traditional_runtime_s + deep_runtime_s},
        ]).to_csv(run_out / "runtime_summary.csv", index=False)

        detail_df = pd.DataFrame(detail_rows)
        detail_df.to_csv(run_out / "per_class_metrics.csv", index=False)
        try:
            detail_df.to_excel(run_out / "per_class_metrics.xlsx", index=False)
        except Exception as e:
            print(f"[WARN] Could not export Excel metrics: {e}")

        deep_results, traditional_results = _split_results_for_paper(results)
        if deep_results:
            # A) Legacy/original plots (kept).
            save_scatter(deep_results, run_out / "dl_main_accuracy_time_params_3d.png")
        for name, cm in confusion_mats.items():
            save_confusion_matrix(cm, name, run_out / f"confusion_{name}.png")
        save_confusion_comparison(confusion_mats, run_out / "confusion_comparative.png")
        save_training_curves(histories, run_out / "training_history_overlay.png")
        save_all_models_loss_overlay(
            histories,
            run_out / "all_models_train_loss_vs_epoch.png",
            loss_key="train_loss",
            title="Deep Models: Training Loss vs Epoch",
        )
        save_all_models_loss_overlay(
            histories,
            run_out / "all_models_val_loss_vs_epoch.png",
            loss_key="val_loss",
            title="Deep Models: Validation Loss vs Epoch",
        )
        save_convergence_diagnostics(histories, run_out / "training_convergence_diagnostics.png")
        save_per_class_f1_bars(per_class_f1, run_out / "per_class_f1_grouped.png")
        if deep_results:
            save_radar_top3(deep_results, run_out / "dl_main_radar_top3.png")
            # Main paper legacy figures: deep-only.
            save_accuracy_vs_inference_bubble(deep_results, run_out / "dl_main_accuracy_vs_inference_bubble.png")
            save_accuracy_vs_params_scatter(deep_results, run_out / "dl_main_accuracy_vs_params_scatter.png")
            save_macrof1_vs_inference_bubble(deep_results, run_out / "dl_main_macrof1_vs_inference_bubble.png")
            save_efficiency_score_bar(deep_results, run_out / "dl_main_efficiency_score_bar.png")
            save_pareto_accuracy_inference(deep_results, run_out / "dl_main_pareto_accuracy_inference.png")
            save_accuracy_vs_training_time(deep_results, run_out / "dl_main_accuracy_vs_training_time.png")
            save_macrof1_vs_params_scatter(deep_results, run_out / "dl_main_macrof1_vs_params_scatter.png")

            # B) New trade-off analysis figures.
            save_dl_pareto_macrof1_vs_inference(
                deep_results, run_out / "dl_pareto_macroF1_vs_inference.png"
            )
            save_dl_family_tradeoff(
                deep_results, run_out / "dl_family_tradeoff.png"
            )
            save_dl_macrof1_vs_training_time(
                deep_results, run_out / "dl_macroF1_vs_training_time.png"
            )
            save_dl_params_vs_inference(
                deep_results, run_out / "dl_params_vs_inference.png"
            )
            run_seq_ok = save_dl_performance_vs_sequence_length(
                df.reset_index().rename(columns={"index": "model"}),
                run_out / "dl_performance_vs_sequence_length.png",
                metric_col="macro_f1",
            )
            if not run_seq_ok:
                print("[WARN] Skip dl_performance_vs_sequence_length.png: sequence-length field not found.")

        # Separate compact baseline reference for traditional ML.
        _save_traditional_baseline_table_and_plot(traditional_results, run_out)

        for model_name, m in results.items():
            aggregated_rows.append({
                "split_mode": args.split_mode,
                "seed": float(run_seed),
                "model": model_name,
                "category": m["category"],
                "accuracy": float(m["accuracy"]),
                "macro_f1": float(m["macro_f1"]),
                "macro_precision": float(m["macro_precision"]),
                "macro_recall": float(m["macro_recall"]),
                "training_seconds": float(m.get("training_seconds", 0.0)),
                "inference_ms": float(m["inference_ms"]),
                "params": float(m["params"]),
                "params_m": float(m["params_m"]),
            })
        aggregated_runtime_rows.append({
            "split_mode": args.split_mode,
            "seed": float(run_seed),
            "traditional_seconds": float(traditional_runtime_s),
            "deep_seconds": float(deep_runtime_s),
            "total_seconds": float(traditional_runtime_s + deep_runtime_s),
        })

        print(f"Saved metrics: {run_out / 'metrics.csv'}")
        print(f"Saved runtime summary: {run_out / 'runtime_summary.csv'}")

    agg_df = pd.DataFrame(aggregated_rows)
    agg_df.to_csv(args.output_dir / "metrics_repeated.csv", index=False)
    summary = agg_df.groupby(["split_mode", "model", "category"], as_index=False).agg(
        accuracy_mean=("accuracy", "mean"),
        accuracy_std=("accuracy", "std"),
        macro_f1_mean=("macro_f1", "mean"),
        macro_f1_std=("macro_f1", "std"),
        macro_precision_mean=("macro_precision", "mean"),
        macro_precision_std=("macro_precision", "std"),
        macro_recall_mean=("macro_recall", "mean"),
        macro_recall_std=("macro_recall", "std"),
        training_seconds_mean=("training_seconds", "mean"),
        training_seconds_std=("training_seconds", "std"),
        inference_ms_mean=("inference_ms", "mean"),
        inference_ms_std=("inference_ms", "std"),
        params_m=("params_m", "first"),
    )
    summary.to_csv(args.output_dir / "metrics_summary.csv", index=False)

    deep_summary = summary[summary["category"] != "traditional"].copy()
    traditional_summary = summary[summary["category"] == "traditional"].copy()
    deep_summary.to_csv(args.output_dir / "metrics_summary_deep_main.csv", index=False)
    traditional_summary.to_csv(args.output_dir / "metrics_summary_traditional_baseline.csv", index=False)

    if not deep_summary.empty:
        # Legacy aggregated summary figures.
        save_summary_bar_with_error(
            summary_df=deep_summary,
            save_path=args.output_dir / "dl_main_accuracy_summary_bar.png",
            metric_col="accuracy_mean",
            error_col="accuracy_std",
            title="Deep Learning Models: Accuracy Across Seeds",
            ylabel="Accuracy",
        )
        save_summary_bar_with_error(
            summary_df=deep_summary,
            save_path=args.output_dir / "dl_main_macro_f1_summary_bar.png",
            metric_col="macro_f1_mean",
            error_col="macro_f1_std",
            title="Deep Learning Models: Macro-F1 Across Seeds",
            ylabel="Macro-F1",
        )
        # New aggregated trade-off figures (mean across seeds).
        deep_tradeoff_df = deep_summary.rename(
            columns={
                "accuracy_mean": "accuracy",
                "macro_f1_mean": "macro_f1",
                "training_seconds_mean": "training_seconds",
                "inference_ms_mean": "inference_ms",
            }
        )
        deep_tradeoff_results: Dict[str, Dict[str, float]] = {}
        for _, row in deep_tradeoff_df.iterrows():
            deep_tradeoff_results[str(row["model"])] = {
                "category": str(row.get("category", "unknown")),
                "accuracy": float(row.get("accuracy", np.nan)),
                "macro_f1": float(row.get("macro_f1", np.nan)),
                "training_seconds": float(row.get("training_seconds", np.nan)),
                "inference_ms": float(row.get("inference_ms", np.nan)),
                "params_m": float(row.get("params_m", np.nan)),
            }
        save_dl_pareto_macrof1_vs_inference(
            deep_tradeoff_results, args.output_dir / "dl_pareto_macroF1_vs_inference.png"
        )
        save_dl_family_tradeoff(
            deep_tradeoff_results, args.output_dir / "dl_family_tradeoff.png"
        )
        save_dl_macrof1_vs_training_time(
            deep_tradeoff_results, args.output_dir / "dl_macroF1_vs_training_time.png"
        )
        save_dl_params_vs_inference(
            deep_tradeoff_results, args.output_dir / "dl_params_vs_inference.png"
        )
        summary_seq_ok = save_dl_performance_vs_sequence_length(
            deep_summary,
            args.output_dir / "dl_performance_vs_sequence_length.png",
            metric_col="macro_f1_mean",
        )
        if not summary_seq_ok:
            print("[WARN] Skip dl_performance_vs_sequence_length.png: sequence-length field not found in summary.")
    if not traditional_summary.empty:
        save_summary_bar_with_error(
            summary_df=traditional_summary,
            save_path=args.output_dir / "traditional_baseline_accuracy_summary_bar.png",
            metric_col="accuracy_mean",
            error_col="accuracy_std",
            title="Traditional ML Baselines (Reference): Accuracy",
            ylabel="Accuracy",
        )
        save_summary_bar_with_error(
            summary_df=traditional_summary,
            save_path=args.output_dir / "traditional_baseline_macro_f1_summary_bar.png",
            metric_col="macro_f1_mean",
            error_col="macro_f1_std",
            title="Traditional ML Baselines (Reference): Macro-F1",
            ylabel="Macro-F1",
        )
    if aggregated_histories:
        save_training_curves_with_std(
            histories_by_model=aggregated_histories,
            save_path=args.output_dir / "training_history_mean_std.png",
        )
        save_all_models_loss_overlay_with_std(
            histories_by_model=aggregated_histories,
            save_path=args.output_dir / "all_models_train_loss_vs_epoch_mean_std.png",
            loss_key="train_loss",
            title="Deep Models: Training Loss vs Epoch Across Seeds",
        )
        save_all_models_loss_overlay_with_std(
            histories_by_model=aggregated_histories,
            save_path=args.output_dir / "all_models_val_loss_vs_epoch_mean_std.png",
            loss_key="val_loss",
            title="Deep Models: Validation Loss vs Epoch Across Seeds",
        )
        save_convergence_diagnostics_with_std(
            histories_by_model=aggregated_histories,
            save_path=args.output_dir / "training_convergence_mean_std.png",
        )

    rt_df = pd.DataFrame(aggregated_runtime_rows)
    rt_df.to_csv(args.output_dir / "runtime_repeated.csv", index=False)
    _write_status(
        status_path,
        {
            "status": "done",
            "gpu_id": gpu_slot,
            "pid": int(os.getpid()),
            "queue_models": queue_models,
            "queue_total": int(queue_total),
            "queue_completed": int(queue_total),
            "current_model": None,
            "current_model_index": int(queue_total),
            "current_epoch": int(args.epochs),
            "total_epochs": int(args.epochs),
            "latest_val_f1": None,
            "last_update_ts": time.time(),
        },
    )


if __name__ == "__main__":
    main()
