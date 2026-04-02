# TacAct 多模型性能对比实验

## 🎯 项目概述
基于TacAct数据集的手势识别模型对比实验，支持深度学习和传统机器学习模型。

## 🚀 快速开始

### 环境安装
```bash
pip install torch torchvision numpy pandas tqdm scikit-learn xgboost matplotlib openpyxl
```

### 单模型调参
```bash
# 深度学习模型
python search_lenet_params.py --data_root "数据路径" --n_trials 20
python search_alexnet_params.py --data_root "数据路径" --n_trials 20
python search_resnet_params.py --data_root "数据路径" --n_trials 20
python search_mobilenet_params.py --data_root "数据路径" --n_trials 20
python search_cnnlstm_params.py --data_root "数据路径" --n_trials 20
python search_tcn_params.py --data_root "数据路径" --n_trials 20
python search_vit_params.py --data_root "数据路径" --n_trials 20

# 传统机器学习模型
python search_randomforest_params.py --data_root "数据路径" --n_trials 20
python search_xgboost_params.py --data_root "数据路径" --n_trials 20
python search_svm_params.py --data_root "数据路径" --n_trials 20
```

### 全模型对比
```bash
python experiment_tacact.py --data_root "数据路径" --run_mode deep --deep_models "LeNet,AlexNet,ResNet18,MobileNet_V2,CNNLSTM,TCN,ViT"
```

## 📊 模型性能排名

| 排名 | 模型 | 最佳准确率 | 状态 |
|--------|--------|------------|------|
| 🥇 1 | **ResNet18** | **82.44%** | ✅ 已优化 |
| 🥈 2 | **CNNLSTM** | **81.90%** | ✅ 已优化 |
| 🥉 3 | **ViT** | **78%** | ✅ 已优化 |
| 🏅 4 | **LeNet** | **~75%** | ✅ 已优化 |
| 🏅 5 | **SVM** | **72.33%** | ✅ 已优化 |
| 🏅 6 | **XGBoost** | **71.75%** | ✅ 已优化 |
| 🏅 7 | **RandomForest** | **65.35%** | ✅ 已优化 |
| 🔄 8 | **AlexNet** | 待搜索 | 🔄 脚本已准备 |
| 🔄 9 | **MobileNet_V2** | 待搜索 | 🔄 脚本已准备 |
| 🔄 10 | **TCN** | 待搜索 | 🔄 脚本已准备 |

## 📁 项目结构

```
D:\compare\
├── tacact/                    # 核心模块
│   ├── models.py             # 统一模型工厂
│   ├── data.py              # 数据加载（含优化能力）
│   └── utils.py             # 工具函数
├── search_*_params.py         # 各模型调参脚本
├── experiment_tacact.py       # 主实验脚本
└── *_search_results.csv       # 调参结果
```

## 🏆 关键发现

- **ResNet18**: 82.44% - 当前最佳模型
- **CNNLSTM**: 81.90% - 时序建模强于纯CNN
- **ViT**: 78% - 注意力机制有效
- **SVM**: 72.33% - 传统ML最佳
- **调参完成度**: 7/11 = 63.6%

## 💡 使用提示

- Windows用户用 `py` 代替 `python`
- 数据放在SSD，缓存放内存盘
- 修改预处理参数需更新缓存目录
- 支持GPU自动检测和混合精度训练
- 进行大规模重构（如合并/删除/重命名 `*_improved.py`）后，建议清理 `__pycache__` 和 `.pyc`，避免旧字节码缓存导致导入异常或行为混淆

## 📈 调参进度

✅ **已优化** (7个): ResNet18, CNNLSTM, ViT, LeNet, SVM, XGBoost, RandomForest  
🔄 **脚本已准备** (4个): AlexNet, MobileNet_V2, TCN, EfficientNet_B0
