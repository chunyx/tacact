#!/usr/bin/env python3
"""
修复 utils_improved.py：删除中间的 re-export 块，并在文件末尾追加，
使其成为 utils.py 的兼容层（单一权威来源）。
"""
from pathlib import Path

FILE = Path(__file__).parent / "tacact" / "utils_improved.py"

RE_EXPORT_BLOCK = '''\n\nfrom . import utils as _utils

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
'''

def main() -> None:
    text = FILE.read_text(encoding="utf-8")
    # 如果已经存在 _EXPORTS，先删除（避免重复）
    if "_EXPORTS" in text or "from . import utils as _utils" in text:
        # 简单正则：删除从 "from . import utils as _utils" 到文件尾之间的 re-export 块
        import re
        # 匹配 re-export 块（包括前面的空行）
        pattern = re.compile(r'\n\nfrom \. import utils as _utils\n\n_EXPORTS = \[.*?\n\]\n\nfor _name in _EXPORTS:.*?\n\n__all__ = list\(_EXPORTS\)', re.DOTALL)
        text = pattern.sub('', text)
        # 清理可能残留的多余空行
        text = text.rstrip('\n') + '\n'
    # 追加到文件末尾
    text = text.rstrip('\n') + RE_EXPORT_BLOCK
    FILE.write_text(text, encoding="utf-8")
    print("Fixed: re-export block moved to end of utils_improved.py")

if __name__ == "__main__":
    main()
