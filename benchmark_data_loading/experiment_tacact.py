#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
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
    save_scatter,
    save_summary_bar_with_error,
    save_training_curves,
    save_training_curves_with_std,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--train_ratio", type=float, default=0.7)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--clip_mode", choices=["center", "front", "weighted_center"], default="front")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--repeat_seeds", type=str, default="",
                        help="留空则单次运行(更快); 如 42,43,44 则多种子重复")
    parser.add_argument("--split_mode", choices=["subject", "random"], default="subject")
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--output_dir", type=Path, default=Path("outputs"))
    parser.add_argument("--cache_dir", type=Path, default=Path(".cache_tacact_n80"))
    parser.add_argument("--run_mode", choices=["all", "traditional", "deep"], default="all")
    parser.add_argument("--traditional_models", type=str, default="SVM,RandomForest,XGBoost")
    parser.add_argument("--deep_models", type=str, default="LeNet,ResNet18,EfficientNet_B0,LSTM,CNN_LSTM,TCN,ViT")
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
        save_scatter(dataframe_to_results_dict(merged_df), args.output_dir / "accuracy_time_params_3d_merged.png")
        print(f"Saved merged metrics: {args.output_dir / 'metrics_merged.csv'}")
        print(f"Saved merged plot: {args.output_dir / 'accuracy_time_params_3d_merged.png'}")
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
        threshold=20.0,
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
        ["LeNet", "AlexNet", "ResNet18", "MobileNet_V2", "EfficientNet_B0", "LSTM", "CNN_LSTM", "TCN", "ViT"],
    )
    print(f"Run mode: {args.run_mode} | traditional={traditional_models} | deep={deep_models}")

    if args.repeat_seeds.strip():
        seed_list = [int(x.strip()) for x in args.repeat_seeds.split(",") if x.strip()]
    else:
        seed_list = [args.seed]

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

        train_loader, val_loader, test_loader = make_three_loaders(
            train_set=train_set,
            val_set=val_set,
            test_set=test_set,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=True,
        )
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
        for model_name in deep_models:
            if args.run_mode not in ("all", "deep"):
                break
            deep_cfg = best_config_map["deep"].get(model_name, {}).get("params", {})
            if not deep_cfg:
                deep_cfg = best_config_map["deep"].get(model_name.lower(), {}).get("params", {})
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
            histories[model_name] = train_torch_model(model, train_loader, val_loader, epochs=args.epochs,
                                                     device=device, **train_kwargs)
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

        save_scatter(results, run_out / "accuracy_time_params_3d.png")
        for name, cm in confusion_mats.items():
            save_confusion_matrix(cm, name, run_out / f"confusion_{name}.png")
        save_confusion_comparison(confusion_mats, run_out / "confusion_comparative.png")
        save_training_curves(histories, run_out / "training_history_overlay.png")
        save_per_class_f1_bars(per_class_f1, run_out / "per_class_f1_grouped.png")
        save_radar_top3(results, run_out / "radar_top3.png")

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
    save_summary_bar_with_error(
        summary_df=summary,
        save_path=args.output_dir / "accuracy_summary_bar.png",
        metric_col="accuracy_mean",
        error_col="accuracy_std",
        title="Accuracy Across Seeds",
        ylabel="Accuracy",
    )
    save_summary_bar_with_error(
        summary_df=summary,
        save_path=args.output_dir / "macro_f1_summary_bar.png",
        metric_col="macro_f1_mean",
        error_col="macro_f1_std",
        title="Macro-F1 Across Seeds",
        ylabel="Macro-F1",
    )
    if aggregated_histories:
        save_training_curves_with_std(
            histories_by_model=aggregated_histories,
            save_path=args.output_dir / "training_history_mean_std.png",
        )

    rt_df = pd.DataFrame(aggregated_runtime_rows)
    rt_df.to_csv(args.output_dir / "runtime_repeated.csv", index=False)


if __name__ == "__main__":
    main()
