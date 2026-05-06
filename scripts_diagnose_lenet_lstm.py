from __future__ import annotations

import json
import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, f1_score, accuracy_score

from data import TacActDataset

ROOT = Path('/home/yaxin/tacact')
OUT_DIR = ROOT / 'outputs' / 'diagnosis' / 'lenet_lstm'
OUT_DIR.mkdir(parents=True, exist_ok=True)
(EX_DIR := OUT_DIR / 'confused_pair_examples').mkdir(parents=True, exist_ok=True)


def warn(msg: str) -> None:
    print(f'[WARN] {msg}')


def save_markdown(path: Path, text: str) -> None:
    path.write_text(text, encoding='utf-8')


def model_run_record_from_metrics(metrics_path: Path) -> Optional[Dict[str, Any]]:
    try:
        df = pd.read_csv(metrics_path)
    except Exception:
        return None
    record = None
    if 'model' in df.columns:
        sub = df[df['model'].astype(str) == 'LeNet_LSTM']
        if not sub.empty:
            record = sub.iloc[0].to_dict()
    else:
        try:
            idx = df.iloc[:, 0].astype(str)
            if 'LeNet_LSTM' in idx.values:
                record = df[idx == 'LeNet_LSTM'].iloc[0].to_dict()
        except Exception:
            pass
    if record is None:
        return None
    base = metrics_path.parent
    final_split_path = base / 'final_split_metrics.csv'
    if final_split_path.exists():
        try:
            fdf = pd.read_csv(final_split_path)
            sub = fdf[fdf['model'].astype(str) == 'LeNet_LSTM']
            if not sub.empty:
                for k, v in sub.iloc[0].to_dict().items():
                    record.setdefault(k, v)
        except Exception:
            pass
    run_config = base / 'run_config.json'
    split_audit = base / 'split_audit.json'
    data_protocol = base / 'data_protocol.json'
    predictions = base / 'predictions.csv'
    training_history = base / 'training_history.csv'
    per_class = base / 'per_class_metrics.csv'
    conf = base / 'confusion_matrix.csv'
    return {
        'run_dir': str(base),
        'metrics_path': str(metrics_path),
        'predictions_path': str(predictions) if predictions.exists() else '',
        'training_history_path': str(training_history) if training_history.exists() else '',
        'per_class_path': str(per_class) if per_class.exists() else '',
        'confusion_path': str(conf) if conf.exists() else '',
        'split_audit_path': str(split_audit) if split_audit.exists() else '',
        'data_protocol_path': str(data_protocol) if data_protocol.exists() else '',
        'run_config_path': str(run_config) if run_config.exists() else '',
        'mtime': metrics_path.stat().st_mtime,
        **record,
    }


def discover_runs() -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for root in ROOT.glob('outputs*'):
        if not root.is_dir():
            continue
        for metrics_path in root.rglob('metrics.csv'):
            rec = model_run_record_from_metrics(metrics_path)
            if rec is not None:
                rows.append(rec)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    for col in ['macro_f1', 'accuracy', 'training_seconds', 'inference_ms', 'best_epoch', 'best_val_f1', 'best_val_acc', 'best_val_loss', 'test_macro_f1', 'test_accuracy']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    if 'test_macro_f1' not in df.columns and 'macro_f1' in df.columns:
        df['test_macro_f1'] = df['macro_f1']
    if 'test_accuracy' not in df.columns and 'accuracy' in df.columns:
        df['test_accuracy'] = df['accuracy']
    return df


def select_run(candidates: pd.DataFrame) -> pd.Series:
    c = candidates.copy()
    c['test_macro_f1'] = pd.to_numeric(c.get('test_macro_f1'), errors='coerce')
    c['test_accuracy'] = pd.to_numeric(c.get('test_accuracy'), errors='coerce')
    c['mtime'] = pd.to_numeric(c.get('mtime'), errors='coerce')
    c = c.sort_values(['test_macro_f1', 'test_accuracy', 'mtime'], ascending=[False, False, False]).reset_index(drop=True)
    return c.iloc[0]


def load_json(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except Exception:
        return {}


def get_dataset(run_config: Dict[str, Any], data_protocol: Dict[str, Any]) -> Optional[TacActDataset]:
    data_root = Path(str(run_config.get('data_root') or ''))
    if not data_root.exists():
        warn(f'data_root not found: {data_root}')
        return None
    cache_dir = Path(str(data_protocol.get('cache_dir') or '.cache_tacact_n80_weighted'))
    if not cache_dir.is_absolute():
        cache_dir = ROOT / cache_dir
    preprocessing = data_protocol.get('preprocessing', {}) if isinstance(data_protocol, dict) else {}
    try:
        ds = TacActDataset(
            root_dir=data_root,
            n_frames=int(preprocessing.get('n_frames', 80)),
            threshold_method=str(preprocessing.get('threshold_method', 'mean_std')),
            threshold_k=float(preprocessing.get('threshold_k', 3.0)),
            background_frames=int(preprocessing.get('background_frames', 5)),
            clip_mode=str(preprocessing.get('clip_mode', 'weighted_center')),
            cache_dir=cache_dir,
            preload_cache=False,
        )
        return ds
    except Exception as e:
        warn(f'Could not build dataset: {e}')
        return None


def compute_confidence_fields(pred_df: pd.DataFrame) -> pd.DataFrame:
    df = pred_df.copy()
    prob_cols = [c for c in df.columns if c.startswith('prob_')]
    if prob_cols:
        probs = df[prob_cols].to_numpy(dtype=float)
        true_idx = df['true_label'].astype(int).to_numpy()
        pred_idx = df['pred_label'].astype(int).to_numpy()
        df['true_class_probability'] = probs[np.arange(len(df)), true_idx]
        df['pred_class_probability'] = probs[np.arange(len(df)), pred_idx]
        sorted_probs = np.sort(probs, axis=1)
        df['probability_margin'] = sorted_probs[:, -1] - sorted_probs[:, -2]
    else:
        df['true_class_probability'] = np.nan
        df['pred_class_probability'] = df.get('top1_confidence', np.nan)
        df['probability_margin'] = np.nan
    return df


def summarize_series(x: pd.Series, prefix: str) -> Dict[str, float]:
    x = pd.to_numeric(x, errors='coerce').dropna()
    if x.empty:
        return {f'{prefix}_{k}': np.nan for k in ['mean', 'std', 'median', 'q1', 'q3']}
    return {
        f'{prefix}_mean': float(x.mean()),
        f'{prefix}_std': float(x.std(ddof=0)),
        f'{prefix}_median': float(x.median()),
        f'{prefix}_q1': float(x.quantile(0.25)),
        f'{prefix}_q3': float(x.quantile(0.75)),
    }


def plot_confusion(cm: np.ndarray, labels: List[str], save_path: Path, normalize: bool = False) -> None:
    mat = cm.astype(float)
    title = 'Confusion Matrix'
    if normalize:
        row_sums = mat.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        mat = mat / row_sums
        title = 'Normalized Confusion Matrix'
    fig, ax = plt.subplots(figsize=(9.2, 7.5))
    im = ax.imshow(mat, cmap='Blues', aspect='auto')
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_yticklabels(labels)
    ax.set_xlabel('Predicted label')
    ax.set_ylabel('True label')
    ax.set_title(title)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(save_path, dpi=220, bbox_inches='tight')
    plt.close(fig)


def plot_bar(df: pd.DataFrame, x: str, y: str, save_path: Path, title: str, color: str = '#4c78a8') -> None:
    fig, ax = plt.subplots(figsize=(10.5, 5.6))
    ax.bar(df[x].astype(str), df[y].astype(float), color=color)
    ax.set_title(title)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True, axis='y', alpha=0.25)
    plt.tight_layout()
    plt.savefig(save_path, dpi=220, bbox_inches='tight')
    plt.close(fig)


def plot_box_compare(df: pd.DataFrame, metric: str, save_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    vals = []
    labels = []
    for flag, label in [(1, 'Correct'), (0, 'Wrong')]:
        sub = pd.to_numeric(df.loc[df['correct'] == flag, metric], errors='coerce').dropna()
        if not sub.empty:
            vals.append(sub.to_numpy())
            labels.append(label)
    if vals:
        ax.boxplot(vals, labels=labels, showmeans=True)
    ax.set_title(title)
    ax.set_ylabel(metric)
    ax.grid(True, axis='y', alpha=0.25)
    plt.tight_layout()
    plt.savefig(save_path, dpi=220, bbox_inches='tight')
    plt.close(fig)


def compute_sequence_stats_for_sample(ds: TacActDataset, sample_id: int) -> Dict[str, Any]:
    meta = ds.samples[int(sample_id)]
    cache_path = ds._cache_path_for(meta.path)
    if cache_path.exists():
        frames = np.load(cache_path, allow_pickle=True).astype(np.float32)
    else:
        frames = ds._preprocess(ds._read_excel_optimized(meta.path), sample_path=meta.path).astype(np.float32)
    abs_frames = np.abs(frames)
    frame_active = abs_frames.max(axis=(1, 2)) > 1e-6
    active_length = int(frame_active.sum())
    active_area = (abs_frames > 1e-6).sum(axis=(1, 2)).astype(np.float32)
    diffs = np.diff(frames, axis=0)
    motion_energy = float(np.mean(np.abs(diffs))) if diffs.size else 0.0
    temporal_energy_curve = np.mean(np.abs(diffs), axis=(1, 2)) if diffs.size else np.array([], dtype=np.float32)
    return {
        'sample_id': int(sample_id),
        'file_path': str(meta.path),
        'subject_id': int(meta.subject),
        'gesture': int(meta.gesture),
        'active_length': active_length,
        'max_abs_delta': float(abs_frames.max()),
        'mean_abs_delta': float(abs_frames.mean()),
        'active_area_mean': float(active_area.mean()),
        'active_area_max': float(active_area.max()),
        'motion_energy': motion_energy,
        '_frames': frames,
        '_temporal_energy_curve': temporal_energy_curve,
    }


def save_confused_example(sample_row: pd.Series, seq_stats: Dict[str, Any], save_path: Path) -> None:
    frames = seq_stats['_frames']
    energy = seq_stats['_temporal_energy_curve']
    active = np.where(np.abs(frames).max(axis=(1,2)) > 1e-6)[0]
    if len(active) == 0:
        active = np.arange(min(len(frames), 1))
    pick_idx = np.unique(np.linspace(active[0], active[-1], num=min(6, max(1, len(active))), dtype=int))
    fig = plt.figure(figsize=(14, 7))
    gs = fig.add_gridspec(2, max(3, len(pick_idx)))
    for i, idx in enumerate(pick_idx):
        ax = fig.add_subplot(gs[0, i])
        im = ax.imshow(frames[idx], cmap='coolwarm', aspect='auto')
        ax.set_title(f'frame {idx}')
        ax.axis('off')
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax_curve = fig.add_subplot(gs[1, :])
    if energy.size:
        ax_curve.plot(np.arange(1, len(energy)+1), energy, color='#4c78a8', linewidth=2)
    ax_curve.set_title('Temporal Energy Curve')
    ax_curve.set_xlabel('Frame step')
    ax_curve.set_ylabel('Mean |delta|')
    ax_curve.grid(True, alpha=0.25)
    title = (
        f"sample_id={int(sample_row['sample_id'])} | subject={int(sample_row['subject_id'])} | "
        f"true={sample_row['true_label']} pred={sample_row['pred_label']} | "
        f"conf={float(sample_row.get('top1_confidence', np.nan)):.4f}"
    )
    fig.suptitle(title, fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=220, bbox_inches='tight')
    plt.close(fig)


def main() -> None:
    candidates = discover_runs()
    if candidates.empty:
        save_markdown(OUT_DIR / 'diagnosis_report.md', '# LeNet_LSTM Diagnosis\n\nNo completed LeNet_LSTM runs were found.')
        return

    candidates = candidates.sort_values(['test_macro_f1', 'test_accuracy', 'mtime'], ascending=[False, False, False])
    candidates.to_csv(OUT_DIR / 'candidate_runs.csv', index=False)
    selected = select_run(candidates)

    selected_reason = {
        'selected_run_dir': selected['run_dir'],
        'selection_rule': 'highest test_macro_f1, then highest test_accuracy, then latest modified time',
        'selected_test_macro_f1': float(selected.get('test_macro_f1', np.nan)),
        'selected_test_accuracy': float(selected.get('test_accuracy', np.nan)),
        'selected_metrics_path': selected['metrics_path'],
        'note': 'Selected the best completed LeNet_LSTM run among all discovered outputs.'
    }
    (OUT_DIR / 'selected_run.json').write_text(json.dumps(selected_reason, ensure_ascii=False, indent=2), encoding='utf-8')
    save_markdown(
        OUT_DIR / 'selected_run.md',
        '# Selected LeNet_LSTM Run\n\n'
        f"- Selected run: `{selected['run_dir']}`\n"
        f"- Reason: highest available test macro-F1 ({float(selected.get('test_macro_f1', np.nan)):.6f}), "
        f"with test accuracy {float(selected.get('test_accuracy', np.nan)):.6f}.\n"
    )

    run_dir = Path(str(selected['run_dir']))
    metrics_path = run_dir / 'metrics.csv'
    predictions_path = run_dir / 'predictions.csv'
    training_history_path = run_dir / 'training_history.csv'
    per_class_path = run_dir / 'per_class_metrics.csv'
    confusion_path = run_dir / 'confusion_matrix.csv'
    split_audit_path = run_dir / 'split_audit.json'
    data_protocol_path = run_dir / 'data_protocol.json'
    run_config_path = run_dir / 'run_config.json'
    final_split_path = run_dir / 'final_split_metrics.csv'

    metrics_df = pd.read_csv(metrics_path) if metrics_path.exists() else pd.DataFrame()
    pred_df = pd.read_csv(predictions_path) if predictions_path.exists() else pd.DataFrame()
    hist_df = pd.read_csv(training_history_path) if training_history_path.exists() else pd.DataFrame()
    per_class_df = pd.read_csv(per_class_path) if per_class_path.exists() else pd.DataFrame()
    conf_df = pd.read_csv(confusion_path) if confusion_path.exists() else pd.DataFrame()
    split_audit = load_json(split_audit_path)
    data_protocol = load_json(data_protocol_path)
    run_config = load_json(run_config_path)
    final_split_df = pd.read_csv(final_split_path) if final_split_path.exists() else pd.DataFrame()

    # 1. Overall summary
    overall = {}
    if not metrics_df.empty:
        m = metrics_df.iloc[0].to_dict()
        overall = {
            'model': 'LeNet_LSTM',
            'test_accuracy': float(m.get('accuracy', np.nan)),
            'test_macro_f1': float(m.get('macro_f1', np.nan)),
            'test_macro_precision': float(m.get('macro_precision', np.nan)),
            'test_macro_recall': float(m.get('macro_recall', np.nan)),
            'inference_ms': float(m.get('inference_ms', np.nan)),
            'params': float(m.get('params', np.nan)),
            'params_m': float(m.get('params_m', np.nan)),
            'best_epoch': float(m.get('best_epoch', np.nan)),
            'best_val_loss': float(m.get('best_val_loss', np.nan)),
            'best_val_acc': float(m.get('best_val_acc', np.nan)),
            'best_val_f1': float(m.get('best_val_f1', np.nan)),
            'training_seconds': float(m.get('training_seconds', np.nan)),
        }
    elif not final_split_df.empty:
        r = final_split_df.iloc[0].to_dict()
        overall = {
            'model': 'LeNet_LSTM',
            'test_accuracy': float(r.get('test_accuracy', np.nan)),
            'test_macro_f1': float(r.get('test_macro_f1', np.nan)),
            'test_macro_precision': float(r.get('test_macro_precision', np.nan)),
            'test_macro_recall': float(r.get('test_macro_recall', np.nan)),
            'inference_ms': float(r.get('inference_ms', np.nan)),
            'params_m': float(r.get('params_m', np.nan)),
            'training_seconds': float(r.get('train_time_sec', np.nan)),
        }
    overall_df = pd.DataFrame([overall]) if overall else pd.DataFrame()
    overall_df.to_csv(OUT_DIR / 'overall_summary.csv', index=False)
    overall_md = '# Overall Summary\n\n'
    if not overall_df.empty:
        row = overall_df.iloc[0]
        for key in overall_df.columns:
            overall_md += f'- {key}: {row[key]}\n'
    else:
        overall_md += 'No overall metrics available.\n'
    save_markdown(OUT_DIR / 'overall_summary.md', overall_md)

    # 2/3/4 predictions-derived analyses
    class_names = data_protocol.get('class_names') or [f'class_{i}' for i in range(12)]
    if not pred_df.empty:
        pred_df = compute_confidence_fields(pred_df)
        if 'file_path' not in pred_df.columns:
            pred_df['file_path'] = ''
        if 'subject_id' not in pred_df.columns:
            pred_df['subject_id'] = np.nan

    # dataset for sample-level diagnostics
    ds = get_dataset(run_config, data_protocol)
    if ds is not None and not pred_df.empty:
        file_paths = []
        subj_ids = []
        for sid in pred_df['sample_id'].astype(int).tolist():
            if 0 <= sid < len(ds.samples):
                meta = ds.samples[sid]
                file_paths.append(str(meta.path))
                subj_ids.append(int(meta.subject))
            else:
                file_paths.append('')
                subj_ids.append(np.nan)
        pred_df['file_path'] = file_paths
        pred_df['subject_id'] = subj_ids

    # confusion matrix
    cm = None
    if not conf_df.empty and {'true_label', 'pred_label', 'count'}.issubset(conf_df.columns):
        cm = np.zeros((12, 12), dtype=int)
        for _, row in conf_df.iterrows():
            cm[int(row['true_label']), int(row['pred_label'])] = int(row['count'])
    elif not pred_df.empty:
        cm = confusion_matrix(pred_df['true_label'], pred_df['pred_label'], labels=list(range(12)))
    if cm is not None:
        cm_rows = []
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                cm_rows.append({'true_label': i, 'pred_label': j, 'count': int(cm[i, j])})
        pd.DataFrame(cm_rows).to_csv(OUT_DIR / 'confusion_matrix.csv', index=False)
        plot_confusion(cm, class_names, OUT_DIR / 'confusion_matrix.png', normalize=False)
        plot_confusion(cm, class_names, OUT_DIR / 'normalized_confusion_matrix.png', normalize=True)
        pair_rows = []
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                if i == j:
                    continue
                pair_rows.append({
                    'true_label': i,
                    'pred_label': j,
                    'true_class': class_names[i],
                    'pred_class': class_names[j],
                    'count': int(cm[i, j]),
                })
        top_pairs_df = pd.DataFrame(pair_rows).sort_values('count', ascending=False).reset_index(drop=True)
        top_pairs_df.to_csv(OUT_DIR / 'top_confused_pairs.csv', index=False)
    else:
        top_pairs_df = pd.DataFrame()

    # per-class metrics
    if per_class_df.empty and not pred_df.empty:
        p, r, f, s = precision_recall_fscore_support(
            pred_df['true_label'], pred_df['pred_label'], labels=list(range(12)), zero_division=0
        )
        per_class_df = pd.DataFrame({
            'class_id': list(range(12)),
            'class_name': class_names,
            'precision': p,
            'recall': r,
            'f1': f,
            'support': s,
        })
    if not per_class_df.empty:
        per_class_df = per_class_df.sort_values('f1', ascending=True).reset_index(drop=True)
        per_class_df.to_csv(OUT_DIR / 'per_class_metrics.csv', index=False)
        plot_bar(per_class_df, 'class_name', 'f1', OUT_DIR / 'per_class_f1_bar.png', 'Per-class F1 (low to high)', color='#e15759')
        weakest_df = per_class_df.head(5).copy()
        weakest_df.to_csv(OUT_DIR / 'weakest_classes.csv', index=False)
    else:
        weakest_df = pd.DataFrame()

    # wrong sample analysis
    if not pred_df.empty:
        wrong_df = pred_df[pred_df['correct'].astype(int) == 0].copy()
        wrong_df.to_csv(OUT_DIR / 'wrong_samples.csv', index=False)
        if 'top1_confidence' in pred_df.columns:
            fig, ax = plt.subplots(figsize=(8,5.2))
            wrong_conf = pd.to_numeric(wrong_df['top1_confidence'], errors='coerce').dropna()
            if not wrong_conf.empty:
                ax.hist(wrong_conf, bins=30, color='#e15759', alpha=0.8)
            ax.set_title('Wrong Prediction Confidence Distribution')
            ax.set_xlabel('Top-1 confidence')
            ax.set_ylabel('Count')
            ax.grid(True, alpha=0.25)
            plt.tight_layout(); plt.savefig(OUT_DIR / 'wrong_confidence_distribution.png', dpi=220, bbox_inches='tight'); plt.close(fig)

            fig, ax = plt.subplots(figsize=(8,5.2))
            corr_conf = pd.to_numeric(pred_df.loc[pred_df['correct'] == 1, 'top1_confidence'], errors='coerce').dropna()
            wrong_conf = pd.to_numeric(pred_df.loc[pred_df['correct'] == 0, 'top1_confidence'], errors='coerce').dropna()
            vals=[]; labels=[]
            if not corr_conf.empty:
                vals.append(corr_conf.to_numpy()); labels.append('Correct')
            if not wrong_conf.empty:
                vals.append(wrong_conf.to_numpy()); labels.append('Wrong')
            if vals:
                ax.boxplot(vals, labels=labels, showmeans=True)
            ax.set_title('Correct vs Wrong Confidence')
            ax.set_ylabel('Top-1 confidence')
            ax.grid(True, axis='y', alpha=0.25)
            plt.tight_layout(); plt.savefig(OUT_DIR / 'correct_vs_wrong_confidence.png', dpi=220, bbox_inches='tight'); plt.close(fig)
    else:
        wrong_df = pd.DataFrame()

    # tactile sequence statistics
    stats_rows: List[Dict[str, Any]] = []
    if ds is not None and not pred_df.empty:
        for _, row in pred_df.iterrows():
            sid = int(row['sample_id'])
            try:
                seq = compute_sequence_stats_for_sample(ds, sid)
            except Exception as e:
                warn(f'Could not compute sequence stats for sample_id={sid}: {e}')
                continue
            rec = {k: v for k, v in seq.items() if not k.startswith('_')}
            rec.update({
                'true_label': int(row['true_label']),
                'pred_label': int(row['pred_label']),
                'correct': int(row['correct']),
                'top1_confidence': float(row.get('top1_confidence', np.nan)),
                'true_class_probability': float(row.get('true_class_probability', np.nan)),
                'pred_class_probability': float(row.get('pred_class_probability', np.nan)),
                'probability_margin': float(row.get('probability_margin', np.nan)),
            })
            stats_rows.append(rec)
        stats_df = pd.DataFrame(stats_rows)
    else:
        stats_df = pd.DataFrame()
    if not stats_df.empty:
        stats_df.to_csv(OUT_DIR / 'sample_sequence_stats.csv', index=False)
        compare_rows = []
        for flag, label in [(1, 'correct'), (0, 'wrong')]:
            sub = stats_df[stats_df['correct'] == flag]
            for metric in ['active_length', 'max_abs_delta', 'mean_abs_delta', 'active_area_mean', 'active_area_max', 'motion_energy']:
                d = {'group': label, 'metric': metric}
                d.update(summarize_series(sub[metric], metric))
                compare_rows.append(d)
        compare_df = pd.DataFrame(compare_rows)
        compare_df.to_csv(OUT_DIR / 'correct_vs_wrong_stats.csv', index=False)
        plot_box_compare(stats_df, 'active_length', OUT_DIR / 'active_length_correct_vs_wrong.png', 'Active Length: Correct vs Wrong')
        plot_box_compare(stats_df, 'max_abs_delta', OUT_DIR / 'max_abs_delta_correct_vs_wrong.png', 'Max Abs Delta: Correct vs Wrong')
        plot_box_compare(stats_df, 'mean_abs_delta', OUT_DIR / 'mean_abs_delta_correct_vs_wrong.png', 'Mean Abs Delta: Correct vs Wrong')
        plot_box_compare(stats_df, 'active_area_mean', OUT_DIR / 'active_area_mean_correct_vs_wrong.png', 'Active Area Mean: Correct vs Wrong')
        plot_box_compare(stats_df, 'motion_energy', OUT_DIR / 'motion_energy_correct_vs_wrong.png', 'Motion Energy: Correct vs Wrong')
    else:
        compare_df = pd.DataFrame()

    # subject-wise
    if not pred_df.empty and 'subject_id' in pred_df.columns:
        rows = []
        for subject_id, sub in pred_df.groupby('subject_id'):
            y_true = sub['true_label'].astype(int).to_numpy()
            y_pred = sub['pred_label'].astype(int).to_numpy()
            rows.append({
                'subject_id': int(subject_id),
                'num_samples': int(len(sub)),
                'accuracy': float(accuracy_score(y_true, y_pred)),
                'macro_f1': float(f1_score(y_true, y_pred, average='macro', labels=list(range(12)), zero_division=0)),
            })
        subject_df = pd.DataFrame(rows).sort_values('macro_f1').reset_index(drop=True)
        subject_df.to_csv(OUT_DIR / 'subject_wise_metrics.csv', index=False)
        plot_bar(subject_df, 'subject_id', 'accuracy', OUT_DIR / 'subject_wise_accuracy.png', 'Subject-wise Accuracy', color='#59a14f')
        plot_bar(subject_df, 'subject_id', 'macro_f1', OUT_DIR / 'subject_wise_macro_f1.png', 'Subject-wise Macro-F1', color='#f28e2b')
    else:
        subject_df = pd.DataFrame()

    # confused pair examples
    if ds is not None and not wrong_df.empty and not top_pairs_df.empty:
        top_n_pairs = top_pairs_df[top_pairs_df['count'] > 0].head(3)
        selected_examples_rows = []
        for _, pair in top_n_pairs.iterrows():
            pair_sub = wrong_df[(wrong_df['true_label'] == pair['true_label']) & (wrong_df['pred_label'] == pair['pred_label'])].copy()
            pair_sub = pair_sub.sort_values('top1_confidence', ascending=False).head(3)
            pair_slug = f"true{int(pair['true_label'])}_pred{int(pair['pred_label'])}"
            pair_dir = EX_DIR / pair_slug
            pair_dir.mkdir(parents=True, exist_ok=True)
            for i, (_, row) in enumerate(pair_sub.iterrows(), start=1):
                sid = int(row['sample_id'])
                try:
                    seq = compute_sequence_stats_for_sample(ds, sid)
                    save_confused_example(row, seq, pair_dir / f'example_{i}_sample{sid}.png')
                    selected_examples_rows.append({
                        'pair': pair_slug,
                        'sample_id': sid,
                        'subject_id': int(row['subject_id']),
                        'true_label': int(row['true_label']),
                        'pred_label': int(row['pred_label']),
                        'top1_confidence': float(row.get('top1_confidence', np.nan)),
                    })
                except Exception as e:
                    warn(f'Could not save confused example for sample {sid}: {e}')
        pd.DataFrame(selected_examples_rows).to_csv(EX_DIR / 'selected_examples.csv', index=False)

    # training stability / overfitting
    training_summary = {}
    if not hist_df.empty:
        best_val_f1_epoch = int(hist_df['val_f1'].astype(float).idxmax() + 1) if 'val_f1' in hist_df.columns else -1
        best_val_loss_epoch = int(hist_df['val_loss'].astype(float).idxmin() + 1) if 'val_loss' in hist_df.columns else -1
        train_loss_last = float(hist_df['train_loss'].iloc[-1]) if 'train_loss' in hist_df.columns else np.nan
        val_loss_last = float(hist_df['val_loss'].iloc[-1]) if 'val_loss' in hist_df.columns else np.nan
        best_val_loss = float(hist_df['val_loss'].min()) if 'val_loss' in hist_df.columns else np.nan
        post_best = hist_df.loc[hist_df['epoch'] >= best_val_f1_epoch, 'val_f1'].astype(float) if 'val_f1' in hist_df.columns else pd.Series(dtype=float)
        val_f1_osc_std = float(post_best.std(ddof=0)) if not post_best.empty else np.nan
        overfit_flag = bool((val_loss_last > best_val_loss + 0.05) and (train_loss_last < float(hist_df['train_loss'].median()))) if 'train_loss' in hist_df.columns and 'val_loss' in hist_df.columns else False
        unstable_flag = bool(val_f1_osc_std > 0.02) if not math.isnan(val_f1_osc_std) else False
        training_summary = {
            'best_val_f1_epoch': best_val_f1_epoch,
            'best_val_loss_epoch': best_val_loss_epoch,
            'best_val_f1': float(hist_df['val_f1'].max()) if 'val_f1' in hist_df.columns else np.nan,
            'best_val_loss': best_val_loss,
            'last_train_loss': train_loss_last,
            'last_val_loss': val_loss_last,
            'val_loss_increases_while_train_loss_decreases': bool(train_loss_last < float(hist_df['train_loss'].iloc[0]) and val_loss_last > best_val_loss) if 'train_loss' in hist_df.columns and 'val_loss' in hist_df.columns else False,
            'val_f1_post_best_std': val_f1_osc_std,
            'clear_overfitting_pattern': overfit_flag,
            'optimization_looks_unstable': unstable_flag,
        }
        pd.DataFrame([training_summary]).to_csv(OUT_DIR / 'training_stability_summary.csv', index=False)

        fig, ax = plt.subplots(figsize=(8.5, 5.2))
        ax.plot(hist_df['epoch'], hist_df['train_loss'], label='train_loss', color='#2a9d8f', linewidth=2)
        ax.plot(hist_df['epoch'], hist_df['val_loss'], label='val_loss', color='#e76f51', linewidth=2)
        ax.axvline(best_val_loss_epoch, color='#444444', linestyle='--', linewidth=1.2, label=f'best_val_loss_epoch={best_val_loss_epoch}')
        ax.set_title('Train vs Validation Loss')
        ax.set_xlabel('Epoch'); ax.set_ylabel('Loss'); ax.grid(True, alpha=0.25); ax.legend(frameon=False)
        plt.tight_layout(); plt.savefig(OUT_DIR / 'train_val_loss_curve.png', dpi=220, bbox_inches='tight'); plt.close(fig)

        fig, ax = plt.subplots(figsize=(8.5, 5.2))
        ax.plot(hist_df['epoch'], hist_df['val_f1'], label='val_f1', color='#4c78a8', linewidth=2)
        ax.plot(hist_df['epoch'], hist_df['val_acc'], label='val_acc', color='#f28e2b', linewidth=2)
        ax.axvline(best_val_f1_epoch, color='#444444', linestyle='--', linewidth=1.2, label=f'best_val_f1_epoch={best_val_f1_epoch}')
        ax.set_title('Validation F1 / Accuracy')
        ax.set_xlabel('Epoch'); ax.set_ylabel('Score'); ax.grid(True, alpha=0.25); ax.legend(frameon=False)
        plt.tight_layout(); plt.savefig(OUT_DIR / 'val_f1_curve.png', dpi=220, bbox_inches='tight'); plt.close(fig)
    else:
        pd.DataFrame().to_csv(OUT_DIR / 'training_stability_summary.csv', index=False)

    # diagnosis report
    weakest_lines = ''
    if not weakest_df.empty:
        for _, row in weakest_df.iterrows():
            weakest_lines += f"- {row['class_name']} (class_id={row['class_id']}): F1={float(row['f1']):.4f}, precision={float(row['precision']):.4f}, recall={float(row['recall']):.4f}\n"
    pair_lines = ''
    if not top_pairs_df.empty:
        for _, row in top_pairs_df.head(5).iterrows():
            if int(row['count']) <= 0:
                continue
            pair_lines += f"- true {row['true_class']} -> pred {row['pred_class']}: {int(row['count'])} samples\n"
    subject_lines = ''
    if not subject_df.empty:
        low_sub = subject_df.head(min(5, len(subject_df)))
        for _, row in low_sub.iterrows():
            subject_lines += f"- subject {int(row['subject_id'])}: accuracy={float(row['accuracy']):.4f}, macro_f1={float(row['macro_f1']):.4f}\n"

    evidence = []
    suggestions = []
    bottlenecks = []

    if training_summary.get('clear_overfitting_pattern'):
        bottlenecks.append('5. optimization instability / overfitting')
        evidence.append('Validation loss rises relative to its minimum while training loss continues to stay very low, which is a classic overfitting pattern.')
        suggestions.append('Because validation loss rises after the best epoch while train loss becomes extremely small, stronger regularization such as label smoothing, SWA, or checkpoint averaging is justified.')
    if not weakest_df.empty and weakest_df['f1'].min() < 0.80:
        bottlenecks.append('1. spatial feature extraction limitation')
        evidence.append('A small subset of classes is much weaker than the rest, suggesting the frame encoder is not equally discriminative for all tactile patterns.')
        suggestions.append('Because only a few classes are consistently weak, a lightweight channel-attention module such as ECA/SE in the frame encoder is a targeted next step to strengthen spatial discrimination without redesigning the whole model.')
    if not compare_df.empty:
        pivot = compare_df.pivot(index='metric', columns='group', values='metric_mean') if {'metric','group','metric_mean'}.issubset(compare_df.columns) else pd.DataFrame()
        try:
            if pivot.loc['active_length','wrong'] < pivot.loc['active_length','correct'] or pivot.loc['motion_energy','wrong'] < pivot.loc['motion_energy','correct']:
                bottlenecks.append('3. weak-signal / short-action difficulty')
                evidence.append('Wrong samples tend to have shorter active duration and/or lower motion energy than correct samples.')
                suggestions.append('Because wrong samples are weaker/shorter, time weighting, action-region refinement, or signal-strength-aware augmentation is better motivated than simply adding model depth.')
        except Exception:
            pass
    if not subject_df.empty:
        if subject_df['macro_f1'].max() - subject_df['macro_f1'].min() > 0.15:
            bottlenecks.append('4. subject-independent generalization difficulty')
            evidence.append('Performance varies strongly across held-out subjects, indicating subject-specific transfer difficulty.')
            suggestions.append('Because errors concentrate on some held-out subjects, subject-level normalization or subject-robust augmentation is a better-matched next step than only tuning the classifier head.')
    if not top_pairs_df.empty and top_pairs_df['count'].head(3).sum() > 0:
        bottlenecks.append('2. temporal fusion limitation')
        evidence.append('The most common confusions are concentrated in a few class pairs rather than uniformly spread, which can indicate insufficient temporal discrimination for similar action patterns.')
        suggestions.append('Because the main confusions are concentrated in a few class pairs, lightweight temporal attention pooling is a targeted next step if the representative examples appear temporally similar.')

    if not bottlenecks:
        bottlenecks.append('No single dominant failure mode could be isolated from the available outputs; likely a mix of moderate overfitting and class-pair confusion.')

    report = f"""# LeNet_LSTM Diagnosis Report

## A. Selected run
- Selected run: `{selected['run_dir']}`
- Selection rule: highest test macro-F1, then highest test accuracy, then latest modified time.
- Selected test macro-F1: {float(selected.get('test_macro_f1', np.nan)):.6f}
- Selected test accuracy: {float(selected.get('test_accuracy', np.nan)):.6f}

## B. Overall result
- Test accuracy: {overall.get('test_accuracy', np.nan):.6f}
- Test macro-F1: {overall.get('test_macro_f1', np.nan):.6f}
- Test macro-precision: {overall.get('test_macro_precision', np.nan):.6f}
- Test macro-recall: {overall.get('test_macro_recall', np.nan):.6f}
- Inference time (ms): {overall.get('inference_ms', np.nan):.6f}
- Parameter count (M): {overall.get('params_m', np.nan):.6f}
- Best validation epoch: {overall.get('best_epoch', np.nan)}

## C. Weakest classes
{weakest_lines if weakest_lines else '- Per-class metrics unavailable.'}
Possible interpretation: the weakest classes likely correspond to tactile patterns that the current frame encoder or temporal pooling separates less reliably.

## D. Most confused class pairs
{pair_lines if pair_lines else '- Confusion pairs unavailable.'}
Interpretation note: class names are generic (`class_i`), so confusion semantics cannot be fully inferred from labels alone. The saved representative examples should be used to judge whether the confusion looks more spatial or more temporal.

## E. Correct vs wrong sample statistics
"""
    if not compare_df.empty:
        report += "- Statistical comparison files saved in `correct_vs_wrong_stats.csv`.\n"
        try:
            pivot = compare_df.pivot(index='metric', columns='group', values='metric_mean')
            for metric in ['active_length', 'max_abs_delta', 'mean_abs_delta', 'active_area_mean', 'motion_energy']:
                if metric in pivot.index:
                    report += f"- {metric}: correct_mean={pivot.loc[metric, 'correct']:.4f}, wrong_mean={pivot.loc[metric, 'wrong']:.4f}\n"
        except Exception:
            pass
    else:
        report += "- Sequence statistics unavailable.\n"
    report += "\n## F. Subject-wise generalization\n"
    report += subject_lines if subject_lines else '- Subject-wise metrics unavailable.\n'
    report += "\n## G. Training stability\n"
    if training_summary:
        report += f"- best_val_f1_epoch: {training_summary.get('best_val_f1_epoch')}\n"
        report += f"- best_val_loss_epoch: {training_summary.get('best_val_loss_epoch')}\n"
        report += f"- val loss increases while train loss decreases: {training_summary.get('val_loss_increases_while_train_loss_decreases')}\n"
        report += f"- val F1 oscillation std after best epoch: {training_summary.get('val_f1_post_best_std'):.6f}\n"
        report += f"- clear overfitting pattern: {training_summary.get('clear_overfitting_pattern')}\n"
        report += f"- optimization looks unstable: {training_summary.get('optimization_looks_unstable')}\n"
    else:
        report += '- Training history unavailable.\n'

    report += '\n## H. Diagnosis conclusion\n'
    for b in dict.fromkeys(bottlenecks):
        report += f'- {b}\n'
    report += '\n### Evidence\n'
    for ev in dict.fromkeys(evidence):
        report += f'- {ev}\n'

    report += '\n## I. Suggested next-step modifications\n'
    if suggestions:
        for s in dict.fromkeys(suggestions):
            report += f'- {s}\n'
    else:
        report += '- No evidence-backed recommendation could be made from the available files alone.\n'

    report += "\n## Notes on feature definitions\n"
    report += "- `active_length` = number of preprocessed frames with any non-zero absolute tactile delta (> 1e-6).\n"
    report += "- `active_area_mean/max` = mean/max number of active taxels per frame using the same > 1e-6 rule on preprocessed cached frames.\n"
    report += "- `motion_energy` = mean absolute difference between consecutive preprocessed frames.\n"
    report += "- All sequence statistics were computed from the cached preprocessed tactile sequences when available, before dataset standardization.\n"

    save_markdown(OUT_DIR / 'diagnosis_report.md', report)
    print(f'Diagnosis completed under: {OUT_DIR}')


if __name__ == '__main__':
    warnings.filterwarnings('ignore', category=RuntimeWarning)
    main()
