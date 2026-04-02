#!/usr/bin/env python3
"""
数据加载性能测试脚本
比较原始和优化版本的数据加载性能
"""

import argparse
import os
import time
from pathlib import Path

import psutil
from torch.utils.data import DataLoader

# 导入两个版本的数据集
from tacact.data import TacActDataset as OriginalDataset
from tacact.data_optimized import OptimizedTacActDataset


def get_memory_usage():
    """获取当前内存使用量(MB)"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024


def benchmark_dataset(dataset_class, dataset_name, data_root, batch_size=32, num_workers=0):
    """测试数据集性能"""
    print(f"\n{'='*50}")
    print(f"测试 {dataset_name}")
    print(f"{'='*50}")
    
    # 记录初始内存
    initial_memory = get_memory_usage()
    print(f"初始内存使用: {initial_memory:.1f} MB")
    
    # 创建数据集
    start_time = time.time()
    dataset = dataset_class(
        data_root, 
        n_frames=80, 
        threshold=20.0, 
        clip_mode="center",
        cache_dir=Path(f".cache_{dataset_name.lower()}"),
        preload_cache=(dataset_class == OptimizedTacActDataset)
    )
    dataset_creation_time = time.time() - start_time
    print(f"数据集创建时间: {dataset_creation_time:.2f} 秒")
    print(f"数据集大小: {len(dataset)} 个样本")
    
    # 创建后的内存使用
    after_creation_memory = get_memory_usage()
    print(f"创建后内存使用: {after_creation_memory:.1f} MB (+{after_creation_memory - initial_memory:.1f} MB)")
    
    # 测试单个样本加载时间
    print("\n测试单个样本加载时间...")
    single_sample_times = []
    for i in range(min(10, len(dataset))):
        start = time.time()
        x, y = dataset[i]
        single_sample_times.append(time.time() - start)
    
    avg_single_time = sum(single_sample_times) / len(single_sample_times)
    print(f"平均单样本加载时间: {avg_single_time*1000:.2f} 毫秒")
    
    # 测试DataLoader性能
    print("\n测试DataLoader性能...")
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    # 预热
    print("预热中...")
    for i, (x, y) in enumerate(dataloader):
        if i >= 2:  # 只预热2个batch
            break
    
    # 正式测试
    print("正式测试...")
    start_time = time.time()
    total_batches = 0
    total_samples = 0
    
    for i, (x, y) in enumerate(dataloader):
        total_batches += 1
        total_samples += x.size(0)
        if i >= 20:  # 测试20个batch
            break
    
    dataloader_time = time.time() - start_time
    avg_batch_time = dataloader_time / total_batches
    samples_per_second = total_samples / dataloader_time
    
    print(f"DataLoader测试结果:")
    print(f"  总时间: {dataloader_time:.2f} 秒")
    print(f"  平均每batch时间: {avg_batch_time*1000:.2f} 毫秒")
    print(f"  处理速度: {samples_per_second:.1f} 样本/秒")
    
    # 峰值内存
    peak_memory = get_memory_usage()
    print(f"峰值内存使用: {peak_memory:.1f} MB (+{peak_memory - initial_memory:.1f} MB)")
    
    return {
        'dataset_name': dataset_name,
        'dataset_creation_time': dataset_creation_time,
        'avg_single_sample_time_ms': avg_single_time * 1000,
        'avg_batch_time_ms': avg_batch_time * 1000,
        'samples_per_second': samples_per_second,
        'peak_memory_mb': peak_memory,
        'memory_increase_mb': peak_memory - initial_memory
    }


def main():
    parser = argparse.ArgumentParser(description="数据加载性能测试")
    parser.add_argument("--data_root", type=Path, required=True, help="数据根目录")
    parser.add_argument("--batch_size", type=int, default=32, help="批次大小")
    parser.add_argument("--num_workers", type=int, default=0, help="DataLoader工作进程数")
    args = parser.parse_args()
    
    print("数据加载性能对比测试")
    print(f"数据目录: {args.data_root}")
    print(f"批次大小: {args.batch_size}")
    print(f"工作进程数: {args.num_workers}")
    
    # 测试原始版本
    original_results = benchmark_dataset(
        OriginalDataset, "原始数据集", args.data_root, args.batch_size, args.num_workers
    )
    
    # 测试优化版本
    optimized_results = benchmark_dataset(
        OptimizedTacActDataset, "优化数据集", args.data_root, args.batch_size, args.num_workers
    )
    
    # 性能对比
    print(f"\n{'='*60}")
    print("性能对比总结")
    print(f"{'='*60}")
    
    metrics = [
        ('dataset_creation_time', '数据集创建时间', '秒'),
        ('avg_single_sample_time_ms', '单样本加载时间', '毫秒'),
        ('avg_batch_time_ms', '平均批次时间', '毫秒'),
        ('samples_per_second', '处理速度', '样本/秒'),
        ('memory_increase_mb', '内存增长', 'MB')
    ]
    
    for key, name, unit in metrics:
        original_val = original_results[key]
        optimized_val = optimized_results[key]
        
        if key == 'samples_per_second':
            # 越大越好
            improvement = (optimized_val - original_val) / original_val * 100
            comparison = f"{original_val:.1f} -> {optimized_val:.1f} ({improvement:+.1f}%)"
        else:
            # 越小越好
            improvement = (original_val - optimized_val) / original_val * 100
            comparison = f"{original_val:.2f} -> {optimized_val:.2f} ({improvement:+.1f}%)"
        
        print(f"{name:12} ({unit}): {comparison}")
    
    print(f"\n优化效果总结:")
    speed_improvement = (optimized_results['samples_per_second'] - original_results['samples_per_second']) / original_results['samples_per_second'] * 100
    memory_reduction = (original_results['memory_increase_mb'] - optimized_results['memory_increase_mb']) / original_results['memory_increase_mb'] * 100
    
    print(f"  处理速度提升: {speed_improvement:+.1f}%")
    print(f"  内存使用减少: {memory_reduction:+.1f}%")


if __name__ == "__main__":
    main()
