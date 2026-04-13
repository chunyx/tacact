#!/usr/bin/env python3
from __future__ import annotations
from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


def _flatten_batch(x: torch.Tensor) -> torch.Tensor:
    return x.view(x.size(0), -1)


def _pool_temporal_features(sequence: torch.Tensor, use_last_only: bool) -> torch.Tensor:
    return sequence[:, -1, :] if use_last_only else sequence.mean(dim=1)


def _make_small_image_backbone(builder, in_channels: int, *, width_mult: float | None = None) -> nn.Module:
    kwargs = {"weights": None}
    if width_mult is not None:
        kwargs["width_mult"] = width_mult
    model = builder(**kwargs)
    first_conv = model.features[0][0]
    model.features[0][0] = nn.Conv2d(
        in_channels,
        first_conv.out_channels,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False,
    )
    return model


def _replace_classifier_with_dropout(model: nn.Module, dropout: float) -> nn.Module:
    classifier = model.classifier
    model.classifier = nn.Sequential(nn.Dropout(dropout), classifier[1])
    return model


def _make_resnet18_32(in_channels: int) -> nn.Module:
    """Build a ResNet18 adapted for 32x32 inputs."""
    m = models.resnet18(weights=None)
    m.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
    m.maxpool = nn.Identity()
    return m


# --- 1. LeNet 适配版 (基于超参搜索优化) ---
class LeNet(nn.Module):
    def __init__(self, in_channels=80, num_classes=12):
        super(LeNet, self).__init__()
        # 最佳参数: conv1_out=16, conv2_out=48, fc1_out=160, fc2_out=84, kernel_size=5, dropout=0.1
        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=5, padding=2)  # 保持 32x32
        self.conv2 = nn.Conv2d(16, 48, kernel_size=5)  # 变为 14x14 -> 7x7 after pooling
        self.dropout = nn.Dropout(0.1)

        # 计算卷积输出尺寸: (32-5+1)/2 = 14, (14-5+1)/2 = 5, 但实际为 6x6
        self.fc1 = nn.Linear(48 * 6 * 6, 160)
        self.fc2 = nn.Linear(160, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)  # 16x16
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)  # 6x6
        x = _flatten_batch(x)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.fc3(x)
        return x


# --- 2. AlexNet 适配版 (针对 32x32 输入) ---
class AlexNet(nn.Module):
    def __init__(self, in_channels=80, num_classes=12):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 16x16
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 8x8
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 4x4
        )
        
        # 🏆 最佳参数: lr=0.0005, weight_decay=0.0001, optimizer=adam, scheduler=step, batch_size=8, dropout=0.3, adaptive_pool=max (val_acc=83.25%)
        self.adaptive_pool = nn.AdaptiveMaxPool2d((4, 4))  # 最佳配置使用max pooling
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),  # 最佳dropout配置
            nn.Linear(256 * 4 * 4, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),  # 最佳dropout配置
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.adaptive_pool(x)  # 使用自适应池化
        x = _flatten_batch(x)
        x = self.classifier(x)
        return x


# --- 3. 其他模型 (LSTM, CNNLSTM, TCN, ViT) ---
class LSTMClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int = 12,
        input_proj_dim: int = 512,
        hidden_size: int = 128,
        num_layers: int = 1,
        dropout: float = 0.5,
        use_last_only: bool = False,
    ) -> None:
        super().__init__()
        input_dim = 32 * 32
        self.frame_proj = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, input_proj_dim),
            nn.ReLU(inplace=True),
        )
        self.lstm = nn.LSTM(
            input_size=input_proj_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_size, num_classes)
        self.use_last_only = use_last_only

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, h, w = x.shape
        x = x.view(b, t, h * w)
        x = self.frame_proj(x)
        lstm_out, _ = self.lstm(x)
        return self.head(self.dropout(_pool_temporal_features(lstm_out, self.use_last_only)))


class GRUClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int = 12,
        input_proj_dim: int = 512,
        hidden_size: int = 128,
        num_layers: int = 1,
        dropout: float = 0.5,
        use_last_only: bool = False,
    ) -> None:
        super().__init__()
        input_dim = 32 * 32
        self.frame_proj = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, input_proj_dim),
            nn.ReLU(inplace=True),
        )
        self.gru = nn.GRU(
            input_size=input_proj_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_size, num_classes)
        self.use_last_only = use_last_only

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, h, w = x.shape
        x = x.view(b, t, h * w)
        x = self.frame_proj(x)
        gru_out, _ = self.gru(x)
        return self.head(self.dropout(_pool_temporal_features(gru_out, self.use_last_only)))


class CNNLSTM(nn.Module):
    def __init__(self, num_classes: int = 12, lstm_hidden: int = 128) -> None:
        super().__init__()
        # 🏆 最佳参数: backbone=resnet18, lstm_hidden=128, lstm_layers=1, lstm_dropout=0.4, dropout=0.5, use_last_only=False (val_acc=81.90%)
        backbone = _make_resnet18_32(1)
        self.frame_extractor = nn.Sequential(*list(backbone.children())[:-1])  # 去掉fc层

        # LSTM层 - 最佳配置
        lstm_layers = 1
        lstm_dropout = 0.4
        self.lstm = nn.LSTM(
            input_size=512,  # ResNet18输出维度 (512)
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,  # 最佳配置
            dropout=lstm_dropout if lstm_layers > 1 else 0.0,
            batch_first=True
        )

        # 分类头 - 使用最佳配置
        self.dropout = nn.Dropout(0.5)  # 最佳配置
        self.head = nn.Linear(lstm_hidden, num_classes)
        self.use_last_only = False  # 最佳配置：使用全时序平均

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, h, w = x.shape
        x = x.view(b * t, 1, h, w)

        # CNN特征提取 (ResNet18)
        features = self.frame_extractor(x)
        # ResNet18需要全局池化
        features = F.adaptive_avg_pool2d(features, (1, 1)).flatten(1)

        # 重塑为序列格式 (b, t, feature_dim)
        features = features.view(b, t, -1)

        # LSTM处理
        lstm_out, _ = self.lstm(features)

        return self.head(self.dropout(_pool_temporal_features(lstm_out, self.use_last_only)))


class TemporalBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, dilation: int, kernel_size: int = 3) -> None:
        super().__init__()
        pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, padding=pad, dilation=dilation)
        self.relu = nn.ReLU(inplace=True)
        self.proj = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv(x)
        crop = self.conv.padding[0]
        if crop > 0:
            y = y[:, :, :-crop]
        return self.relu(y + self.proj(x))


class TCNClassifier(nn.Module):
    def __init__(self, num_classes: int = 12, num_channels: int = 512, dropout: float = 0.0) -> None:
        super().__init__()
        in_dim = 32 * 32
        # 🏆 最佳参数: lr=0.001, weight_decay=1e-05, optimizer=adamw, scheduler=plateau, batch_size=8, num_channels=512, dropout=0.0, kernel_size=7, levels=5 (val_acc=70.77%)
        # 使用最佳配置：512通道，5层，kernel_size=7
        self.blocks = nn.Sequential(
            TemporalBlock(in_dim, num_channels, dilation=1, kernel_size=7),
            TemporalBlock(num_channels, num_channels, dilation=2, kernel_size=7),
            TemporalBlock(num_channels, num_channels, dilation=4, kernel_size=7),
            TemporalBlock(num_channels, num_channels, dilation=8, kernel_size=7),
            TemporalBlock(num_channels, num_channels, dilation=16, kernel_size=7),
        )
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(num_channels, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), x.size(1), -1).transpose(1, 2)
        y = self.blocks(x)
        return self.head(self.dropout(y[:, :, -1]))


class SmallViT(nn.Module):
    def __init__(self, num_classes: int = 12, dim: int = 256, depth: int = 3, heads: int = 8,
                 patch_size: int = 16, dropout: float = 0.25) -> None:
        super().__init__()
        # 🏆 最佳参数: dim=256, depth=3, heads=8, patch_size=16, dropout=0.25
        if patch_size not in {8, 16}:
            raise ValueError(f"Unsupported patch_size={patch_size}. Only 8 or 16 are supported.")
        self.patch_size = patch_size
        if 32 % self.patch_size != 0:
            raise ValueError(f"patch_size must divide 32, got {self.patch_size}")
        self.patches_per_frame = (32 // patch_size) ** 2
        self.total_patches = 80 * self.patches_per_frame

        self.patch_embed = nn.Linear(patch_size * patch_size, dim)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=depth)

        # CLS token和位置编码
        self.cls = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos = nn.Parameter(torch.randn(1, self.total_patches + 1, dim) * 0.02)

        # 分类头
        self.head = nn.Linear(dim, num_classes)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        nn.init.trunc_normal_(self.cls, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = x.size(0)
        x = x.unfold(2, self.patch_size, self.patch_size) \
            .unfold(3, self.patch_size, self.patch_size)
        x = x.contiguous().view(b, 80, self.patches_per_frame, -1)
        x = x.view(b, self.total_patches, -1)

        x = self.patch_embed(x)

        cls_tokens = self.cls.expand(b, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = x + self.pos[:, :self.total_patches + 1]

        x = self.encoder(x)
        return self.head(x[:, 0, :])


# --- 4. 统一的 ModelFactory ---
class ModelFactory:
    @staticmethod
    def build_torch(name: str, in_channels: int = 80, num_classes: int = 12, **kwargs) -> Tuple[nn.Module, str]:
        n = name.lower()
        if n == "lenet":
            return LeNet(in_channels=in_channels, num_classes=num_classes), "cnn"
        if n == "alexnet":
            return AlexNet(in_channels=in_channels, num_classes=num_classes), "cnn"

        if n == "resnet18":
            m = _make_resnet18_32(in_channels)
            
            # 🏆 最佳参数: lr=0.001, weight_decay=1e-05, optimizer=adam, scheduler=none, batch_size=16, dropout=0.2 (val_acc=82.44%)
            # 添加dropout到分类头
            original_fc = m.fc
            m.fc = nn.Sequential(
                nn.Dropout(0.2),
                original_fc
            )
            
            return m, "cnn"

        if n == "mobilenet_v2":
            # 🏆 最佳参数: lr=0.001, weight_decay=0.001, optimizer=adam, scheduler=cosine, batch_size=16, dropout=0.1, width_mult=0.5 (val_acc=82.10%)
            m = _make_small_image_backbone(models.mobilenet_v2, in_channels, width_mult=0.5)
            m = _replace_classifier_with_dropout(m, 0.1)
            return m, "cnn"

        if n == "efficientnet_b0":
            # 🏆 最佳参数: lr=0.005, weight_decay=0.0005, optimizer=adamw, scheduler=step, batch_size=32, dropout=0.2, stochastic_depth=0.0 (val_acc=81.90%)
            m = _make_small_image_backbone(models.efficientnet_b0, in_channels)
            m = _replace_classifier_with_dropout(m, 0.2)
            return m, "cnn"

        if n == "cnn_lstm":
            lstm_hidden = int(kwargs.get("lstm_hidden", 128))  # 更新为最佳配置
            return CNNLSTM(num_classes=num_classes, lstm_hidden=lstm_hidden), "temporal"

        if n in {"lstm", "lstm_ablation", "raw_lstm"}:
            input_proj_dim = int(kwargs.get("input_proj_dim", 512))
            hidden_size = int(kwargs.get("hidden_size", 128))
            num_layers = int(kwargs.get("num_layers", 1))
            dropout = float(kwargs.get("dropout", 0.5))
            use_last_only = bool(kwargs.get("use_last_only", False))
            return LSTMClassifier(
                num_classes=num_classes,
                input_proj_dim=input_proj_dim,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout,
                use_last_only=use_last_only,
            ), "temporal"

        if n in {"gru", "raw_gru"}:
            input_proj_dim = int(kwargs.get("input_proj_dim", 512))
            hidden_size = int(kwargs.get("hidden_size", 128))
            num_layers = int(kwargs.get("num_layers", 1))
            dropout = float(kwargs.get("dropout", 0.5))
            use_last_only = bool(kwargs.get("use_last_only", False))
            return GRUClassifier(
                num_classes=num_classes,
                input_proj_dim=input_proj_dim,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout,
                use_last_only=use_last_only,
            ), "temporal"

        if n == "tcn":
            # 🏆 最佳参数: lr=0.001, weight_decay=1e-05, optimizer=adamw, scheduler=plateau, batch_size=8, num_channels=512, dropout=0.0, kernel_size=7, levels=5 (val_acc=70.77%)
            num_channels = int(kwargs.get("num_channels", 512))  # 更新为最佳配置
            dropout = float(kwargs.get("dropout", 0.0))  # 更新为最佳配置
            return TCNClassifier(num_classes=num_classes, num_channels=num_channels, dropout=dropout), "temporal"

        if n == "vit":
            dim = int(kwargs.get("dim", 256))
            depth = int(kwargs.get("depth", 3))
            heads = int(kwargs.get("heads", 8))
            patch_size = int(kwargs.get("patch_size", 16))
            dropout = float(kwargs.get("dropout", 0.25))
            return SmallViT(
                num_classes=num_classes,
                dim=dim,
                depth=depth,
                heads=heads,
                patch_size=patch_size,
                dropout=dropout,
            ), "attention"

        raise ValueError(f"Unknown torch model: {name}")

    @staticmethod
    def build_traditional(name: str, **kwargs):
        n = name.lower()
        if n == "svm":
            from sklearn.svm import SVC
            c = float(kwargs.get("C", 30.0))
            gamma = kwargs.get("gamma", "auto")
            class_weight = kwargs.get("class_weight", "balanced")
            return SVC(kernel="rbf", C=c, gamma=gamma, class_weight=class_weight, random_state=42)

        if n == "randomforest":
            from sklearn.ensemble import RandomForestClassifier
            return RandomForestClassifier(
                n_estimators=int(kwargs.get("n_estimators", 100)),
                max_depth=kwargs.get("max_depth", 20),
                min_samples_split=int(kwargs.get("min_samples_split", 2)),
                min_samples_leaf=int(kwargs.get("min_samples_leaf", 2)),
                max_features=kwargs.get("max_features", 0.05),
                bootstrap=bool(kwargs.get("bootstrap", False)),
                criterion=kwargs.get("criterion", "gini"),
                random_state=int(kwargs.get("random_state", 42)),
                n_jobs=int(kwargs.get("n_jobs", -1))
            )

        if n == "xgboost":
            import xgboost as xgb
            max_depth = kwargs.get("max_depth", 10)
            if max_depth is not None:
                max_depth = int(max_depth)
            return xgb.XGBClassifier(
                n_estimators=int(kwargs.get("n_estimators", 300)),
                max_depth=max_depth,
                learning_rate=float(kwargs.get("learning_rate", 0.1)),
                subsample=float(kwargs.get("subsample", 0.6)),
                colsample_bytree=float(kwargs.get("colsample_bytree", 0.6)),
                objective="multi:softmax",
                num_class=12,
                n_jobs=int(kwargs.get("n_jobs", -1)),
                random_state=int(kwargs.get("random_state", 42))
            )

        raise ValueError(f"Unknown traditional model: {name}")
