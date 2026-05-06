from __future__ import annotations

from typing import Any, Dict, List


# Phase-0 evidence tags from legacy budgeted tuning table.
MODEL_EVIDENCE_TAGS: Dict[str, str] = {
    "lenet": "reliable_for_space_reduction",
    "resnet18": "reliable_for_space_reduction",
    "mobilenet_v2": "reliable_for_space_reduction",
    "efficientnet_b0": "reliable_for_space_reduction",
    "lstm": "reliable_for_space_reduction",
    "gru": "weak_evidence_only",
    "cnn_lstm": "unusable_due_to_unfair_search",
    "lenet_lstm": "weak_evidence_only",
    "tcn": "reliable_for_space_reduction",
    "transformer": "weak_evidence_only",
}


# Priority layers for the two-stage random search + final multi-seed evaluation pipeline.
# A: core candidates, B: secondary baselines, C: exploratory / low-priority.
MODEL_PRIORITY_TIER: Dict[str, str] = {
    "resnet18": "A",
    "efficientnet_b0": "A",
    "mobilenet_v2": "A",
    "lenet": "A",
    "lstm": "B",
    "gru": "B",
    "tcn": "B",
    "transformer": "B",
    "cnn_lstm": "C",
    "lenet_lstm": "B",
}


# Refined search spaces generated from phase-0 evidence.
# Important: these are not single-winner configs; each keeps a small candidate neighborhood.
DEEP_SPACE_REFINED: Dict[str, Dict[str, List[Any]]] = {
    "lenet": {
        "lr": [1e-4, 2e-4, 3e-4, 5e-4],
        "weight_decay": [5e-5, 1e-4, 3e-4, 5e-4],
        "batch_size": [16, 32, 64],
    },
    "resnet18": {
        "lr": [1e-4, 3e-4, 7e-4],
        "weight_decay": [1e-4, 3e-4, 1e-3],
        "batch_size": [8, 16, 32],
    },
    "mobilenet_v2": {
        "lr": [1e-4, 3e-4, 6e-4],
        "weight_decay": [1e-4, 3e-4, 1e-3],
        "batch_size": [8, 16, 32],
    },
    "efficientnet_b0": {
        "lr": [1e-4, 3e-4, 7e-4],
        "weight_decay": [1e-4, 3e-4, 1e-3],
        "batch_size": [8, 16, 32],
    },
    "lstm": {
        "input_proj_dim": [64, 128, 256],
        "hidden_size": [64, 128, 192],
        "num_layers": [1, 2],
        "dropout": [0.3, 0.5, 0.6],
        "use_last_only": [False],
        "lr": [1e-4, 3e-4, 6e-4],
        "weight_decay": [1e-4, 3e-4, 1e-3],
        "batch_size": [16, 32, 64],
    },
    "gru": {
        "input_proj_dim": [64, 128, 256],
        "hidden_size": [64, 128, 192],
        "num_layers": [1, 2],
        "dropout": [0.3, 0.5, 0.6],
        "use_last_only": [False],
        "lr": [1e-4, 3e-4, 6e-4],
        "weight_decay": [1e-4, 3e-4, 1e-3],
        "batch_size": [16, 32, 64],
    },
    "cnn_lstm": {
        "lstm_hidden": [64, 128, 192],
        "dropout": [0.3, 0.5],
        "lr": [1e-4, 3e-4],
        "weight_decay": [1e-4, 3e-4, 1e-3],
        "batch_size": [8, 16, 32],
    },
    "lenet_lstm": {
        "feature_dim": [64, 96, 128],
        "encoder_hidden_dim": [96, 128, 160],
        "hidden_size": [64, 96, 128],
        "num_layers": [1, 2],
        "dropout": [0.3, 0.5, 0.6],
        "use_last_only": [False, True],
        "bidirectional": [False],
        "lr": [1e-4, 3e-4, 6e-4],
        "weight_decay": [1e-4, 3e-4, 1e-3],
        "batch_size": [16, 32, 64],
    },
    "tcn": {
        "num_channels": [64, 128, 192],
        "dropout": [0.3, 0.5],
        "lr": [5e-5, 1e-4],
        "weight_decay": [1e-3, 3e-3],
        "batch_size": [16, 32, 64],
    },
    "transformer": {
        "d_model": [32, 64, 96],
        "nhead": [2, 4],
        "num_layers": [1, 2],
        "dim_feedforward": [64, 128, 192],
        "dropout": [0.2, 0.3, 0.4],
        "pooling": ["mean"],
        "norm_first": [True],
        "lr": [5e-5, 1e-4, 1.5e-4],
        "weight_decay": [3e-4, 1e-3, 3e-3],
        "batch_size": [16, 32],
    },
}
