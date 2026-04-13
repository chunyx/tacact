from __future__ import annotations

from typing import Any, Dict, List


# Phase-0 evidence tags from legacy budgeted tuning table.
MODEL_EVIDENCE_TAGS: Dict[str, str] = {
    "lenet": "reliable_for_space_reduction",
    "alexnet": "unusable_due_to_unfair_search",
    "resnet18": "reliable_for_space_reduction",
    "mobilenet_v2": "reliable_for_space_reduction",
    "efficientnet_b0": "reliable_for_space_reduction",
    "lstm": "reliable_for_space_reduction",
    "gru": "weak_evidence_only",
    "cnn_lstm": "unusable_due_to_unfair_search",
    "tcn": "reliable_for_space_reduction",
    "vit": "weak_evidence_only",
}


# Priority layers for the new 3-phase pipeline.
# A: core candidates, B: secondary baselines, C: exploratory / low-priority.
MODEL_PRIORITY_TIER: Dict[str, str] = {
    "resnet18": "A",
    "efficientnet_b0": "A",
    "mobilenet_v2": "A",
    "lenet": "A",
    "lstm": "B",
    "gru": "B",
    "tcn": "B",
    "vit": "C",
    "alexnet": "C",
    "cnn_lstm": "C",
}


# Refined search spaces generated from phase-0 evidence.
# Important: these are not single-winner configs; each keeps a small candidate neighborhood.
DEEP_SPACE_REFINED: Dict[str, Dict[str, List[Any]]] = {
    "lenet": {
        "lr": [2e-4, 3e-4, 5e-4, 1e-3],
        "weight_decay": [1e-5, 5e-5, 1e-4],
        "batch_size": [16, 32],
    },
    "alexnet": {
        # Old trials were all skip_params under legacy constraints; keep moderately broad.
        "lr": [1e-4, 3e-4, 7e-4, 1e-3],
        "weight_decay": [1e-5, 1e-4, 5e-4],
        "batch_size": [4, 8, 16],
    },
    "resnet18": {
        "lr": [3e-4, 7e-4, 1e-3],
        "weight_decay": [1e-5, 1e-4, 3e-4],
        "batch_size": [8, 16],
    },
    "mobilenet_v2": {
        "lr": [3e-4, 6e-4, 1e-3],
        "weight_decay": [1e-5, 1e-4, 5e-4, 1e-3],
        "batch_size": [8, 16],
    },
    "efficientnet_b0": {
        "lr": [3e-4, 7e-4, 1e-3],
        "weight_decay": [1e-5, 1e-4, 5e-4],
        "batch_size": [8, 16],
    },
    "lstm": {
        "input_proj_dim": [256, 512],
        "hidden_size": [128, 256],
        "num_layers": [1, 2],
        "dropout": [0.3, 0.5],
        "use_last_only": [False],
        "lr": [3e-4, 6e-4, 1e-3],
        "weight_decay": [1e-5, 1e-4],
        "batch_size": [8, 16],
    },
    "gru": {
        # Start with LSTM-aligned space to keep fairness in recurrent-unit ablation.
        "input_proj_dim": [256, 512],
        "hidden_size": [128, 256],
        "num_layers": [1, 2],
        "dropout": [0.3, 0.5],
        "use_last_only": [False],
        "lr": [3e-4, 6e-4, 1e-3],
        "weight_decay": [1e-5, 1e-4],
        "batch_size": [8, 16],
    },
    "cnn_lstm": {
        # Old trials were all skip_time; preserve wider exploratory space.
        "lstm_hidden": [96, 128, 192, 256],
        "lr": [1e-4, 3e-4, 1e-3],
        "weight_decay": [1e-5, 1e-4],
        "batch_size": [2, 4, 8],
    },
    "tcn": {
        "num_channels": [256, 384, 512],
        "dropout": [0.0, 0.1],
        "lr": [1e-4, 2e-4, 3e-4],
        "weight_decay": [1e-5, 1e-4],
        "batch_size": [4, 8],
    },
    "vit": {
        # Weak evidence + many skip_time; bias toward cheaper/faster variants but keep diversity.
        "dim": [160, 192, 224, 256],
        "depth": [2, 3, 4],
        "heads": [4, 8],
        "patch_size": [16],
        "dropout": [0.1, 0.2, 0.25],
        "lr": [5e-5, 1e-4, 2e-4],
        "weight_decay": [0.01, 0.03, 0.05],
        "batch_size": [2, 4],
    },
}
