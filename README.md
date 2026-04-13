# TacAct Benchmark Toolkit

TacAct 手势识别实验代码库（HPO + 主实验 + 恢复合并 +论文图表）。

## 1. 当前仓库里你会用到的入口

- 三阶段 HPO 入口：`benchmark_data_loading/search_all_models_budgeted.py`
- 主实验入口：`benchmark_data_loading/experiment_tacact.py`
- 5GPU 主实验脚本：`run_main_9models_5gpu.sh`
- 一键运行+监控：`run_and_watch_9models_5gpu.sh`
- 失败补跑+自动合并：`recover_failed_and_merge.sh`
- 手动合并：`merge_main_9models_5gpu.sh`
- 监控面板：`watch_main_9models_5gpu.py`

## 2. 环境准备

```bash
module load cuda/11.8
source <your_venv>/bin/activate
pip install torch torchvision numpy pandas tqdm scikit-learn xgboost matplotlib openpyxl psutil
```

说明：
- 本仓库里脚本默认按 CUDA GPU 运行。
- `experiment_tacact.py` 在无 GPU 时会提示并默认中止。

## 3. 数据与缓存

- 输入：`xlsx`
- 文件命名：`<subject>_<gesture>_<variant>_<repeat>.xlsx`
- 类别：12 类（`gesture 1~12 -> label 0~11`）
- 默认缓存目录：`.cache_tacact_n80_weighted`

并行时的数据策略（已实现）：
- 单进程：可 preload
- 并行：禁用 preload，lazy loading，避免多进程内存放大

## 4. 三阶段 HPO（固定预算 + 分阶段筛选）

### 4.1 阶段定义

- Phase1：固定 epoch，无 early stopping（默认 `phase1_epochs=8`）
- Phase2：固定 epoch，无 early stopping（默认 `phase2_epochs=20`）
- Phase3：固定 epoch（当前已禁用 early stopping）

### 4.2 常用命令

5 GPU 并行（示例）：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4 \
python benchmark_data_loading/search_all_models_budgeted.py \
  --data_root /path/to/TacAct-original \
  --output_dir outputs_hpo_3phase \
  --phase all \
  --parallel \
  --gpu_ids 0,1,2,3,4 \
  --max_workers 5 \
  --num_workers 0
```

只跑 Phase2（示例）：

```bash
CUDA_VISIBLE_DEVICES=0,1 \
python benchmark_data_loading/search_all_models_budgeted.py \
  --data_root /path/to/TacAct-original \
  --output_dir outputs_hpo_3phase \
  --phase phase2 \
  --parallel \
  --gpu_ids 0,1 \
  --max_workers 2 \
  --phase2_topk 2 \
  --phase2_epochs 18 \
  --no_resume
```

### 4.3 输出结构（HPO）

```text
<hpo_output>/
├── model_progress/<Model>.json
├── phase1/<Model>/phase1_results.csv
├── phase2/<Model>/phase2_results.csv
├── phase3/<Model>/final_results.csv
├── phase3/<Model>/final_results_summary.csv
└── hpo_pipeline_meta.json
```

## 5. 主实验（9个深度模型）与合并

9 个深度模型：
- LeNet, AlexNet, ResNet18, MobileNet_V2, EfficientNet_B0, LSTM, CNN_LSTM, TCN, ViT

### 5.1 一键跑 5GPU 主实验

```bash
DATA_ROOT=/path/to/TacAct-original \
HPO_ROOT=outputs_hpo_3phase \
./run_main_9models_5gpu.sh
```

### 5.2 一键跑 + 自动监控面板

```bash
DATA_ROOT=/path/to/TacAct-original \
HPO_ROOT=outputs_hpo_3phase \
./run_and_watch_9models_5gpu.sh
```

### 5.3 失败后只补跑缺失模型 + 自动最终合并

```bash
GPU_A=0 GPU_B=1 GPU_C=2 \
./recover_failed_and_merge.sh
```

## 6. 监控与日志

- HPO：`search_all_models_budgeted.py` 会输出全局进度与 worker 状态
- 主实验：`watch_main_9models_5gpu.py --run_root <RUN_ROOT>`
- 每张 GPU 的日志在：`<RUN_ROOT>/logs/*.log`

## 7. 关于 early stopping（重要）

- HPO 的 Phase1/Phase2/Phase3 当前均为固定 epoch（不早停）。
- `experiment_tacact.py` 的训练当前也已禁用 early stopping，按设定 epoch 跑满。

你本次已导出的实际轮数表：
- `outputs_main_9models_5gpu_recovery_20260408_150432/actual_epochs_all_9models.csv`

## 8. 你当前最终结果目录（已合并）

当前你整理后的总目录：

```text
outputs_final_bundle_20260408_215635/
├── outputs_hpo_3phase_aborted_20260407_183249
├── outputs_tuning_budgeted_paper_deep_only
├── outputs_main_9models_5gpu
└── outputs_main_9models_5gpu_recovery_20260408_150432
    └── merged_final/
        ├── metrics_merged.csv
        └── *_merged.png
```

论文主用总表/总图位置：
- `.../merged_final/metrics_merged.csv`
- `.../merged_final/*_merged.png`

## 9. 项目结构（核心）

```text
tacact/
├── data.py
├── models.py
├── utils.py
├── benchmark_common.py
├── run_main_9models_5gpu.sh
├── run_and_watch_9models_5gpu.sh
├── recover_failed_and_merge.sh
├── merge_main_9models_5gpu.sh
├── watch_main_9models_5gpu.py
└── benchmark_data_loading/
    ├── hpo_pipeline.py
    ├── search_all_models_budgeted.py
    ├── experiment_tacact.py
    ├── benchmark_data_loading.py
    └── clean_cache.py
```
