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


def _reshape_temporal_frames(x: torch.Tensor) -> torch.Tensor:
    if x.dim() == 5:
        if x.size(2) != 1:
            raise ValueError(f"Expected single tactile channel for temporal input, got shape={tuple(x.shape)}")
        x = x.squeeze(2)
    if x.dim() != 4:
        raise ValueError(f"Expected temporal tactile input shaped (B, T, 32, 32) or (B, T, 1, 32, 32), got {tuple(x.shape)}")
    return x


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


def _replace_classifier_output(model: nn.Module, num_classes: int) -> nn.Module:
    """Keep torchvision classifier body unchanged and only adapt final logits dim."""
    classifier = model.classifier
    if not isinstance(classifier, nn.Sequential) or len(classifier) == 0:
        raise ValueError(f"Unsupported classifier type: {type(classifier)}")
    last = classifier[-1]
    if not isinstance(last, nn.Linear):
        raise ValueError(f"Unsupported classifier tail layer: {type(last)}")
    classifier[-1] = nn.Linear(last.in_features, num_classes)
    model.classifier = classifier
    return model


def _make_resnet18_32(in_channels: int, num_classes: int) -> nn.Module:
    """Build torchvision ResNet18 with only input/output task adaptation."""
    m = models.resnet18(weights=None)
    conv1 = m.conv1
    m.conv1 = nn.Conv2d(
        in_channels,
        conv1.out_channels,
        kernel_size=conv1.kernel_size,
        stride=conv1.stride,
        padding=conv1.padding,
        bias=False,
    )
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    return m


# --- 1. LeNet-style CNN (接近 LeNet-5 的任务适配版) ---
class LeNet(nn.Module):
    """Task-specific baseline: LeNet-style CNN (not strict original LeNet-5)."""

    def __init__(self, in_channels=80, num_classes=12):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 6, kernel_size=5, padding=0)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, padding=0)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = self.pool(torch.tanh(self.conv1(x)))
        x = self.pool(torch.tanh(self.conv2(x)))
        x = _flatten_batch(x)
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = self.fc3(x)
        return x


# --- 2. 其他模型 (LSTM, CNNLSTM, TCN) ---
class LSTMClassifier(nn.Module):
    """Task-specific temporal baseline using frame projection + LSTM."""

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
            nn.Dropout(dropout),
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
        x = _reshape_temporal_frames(x)
        b, t, h, w = x.shape
        x = x.view(b, t, h * w)
        x = self.frame_proj(x)
        lstm_out, _ = self.lstm(x)
        return self.head(self.dropout(_pool_temporal_features(lstm_out, self.use_last_only)))


class GRUClassifier(nn.Module):
    """Task-specific temporal baseline using frame projection + GRU."""

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
            nn.Dropout(dropout),
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
        x = _reshape_temporal_frames(x)
        b, t, h, w = x.shape
        x = x.view(b, t, h * w)
        x = self.frame_proj(x)
        gru_out, _ = self.gru(x)
        return self.head(self.dropout(_pool_temporal_features(gru_out, self.use_last_only)))


class TransformerClassifier(nn.Module):
    """Task-specific temporal Transformer baseline."""

    def __init__(
        self,
        num_classes: int = 12,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        pooling: str = "mean",
        norm_first: bool = False,
        max_seq_len: int = 80,
    ) -> None:
        super().__init__()
        if d_model % nhead != 0:
            raise ValueError(f"d_model must be divisible by nhead, got d_model={d_model}, nhead={nhead}")
        if pooling != "mean":
            raise ValueError(f"Unsupported pooling={pooling}. Currently only 'mean' is supported.")

        input_dim = 32 * 32
        self.frame_proj = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.pos_embed = nn.Parameter(torch.zeros(1, max_seq_len, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=norm_first,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, num_classes),
        )
        self.pooling = pooling
        self.max_seq_len = int(max_seq_len)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
        for layer in self.encoder.layers:
            for name, param in layer.named_parameters():
                if param.dim() > 1:
                    if "in_proj_weight" in name:
                        nn.init.xavier_uniform_(param)
                    else:
                        nn.init.xavier_uniform_(param)
                else:
                    nn.init.zeros_(param)
            nn.init.ones_(layer.norm1.weight)
            nn.init.zeros_(layer.norm1.bias)
            nn.init.ones_(layer.norm2.weight)
            nn.init.zeros_(layer.norm2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = _reshape_temporal_frames(x)
        b, t, h, w = x.shape
        if t > self.max_seq_len:
            raise ValueError(f"Sequence length {t} exceeds max_seq_len={self.max_seq_len}")
        x = x.view(b, t, h * w)
        x = self.frame_proj(x)
        x = x + self.pos_embed[:, :t, :]
        x = self.encoder(x)
        x = _pool_temporal_features(x, use_last_only=False)
        return self.head(self.dropout(x))


class LeNetFrameEncoder(nn.Module):
    """Task-specific frame encoder used by LeNet_LSTM baseline."""

    def __init__(
        self,
        in_channels: int = 1,
        conv1_out: int = 16,
        conv2_out: int = 48,
        encoder_hidden_dim: int = 160,
        feature_dim: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, conv1_out, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(conv1_out, conv2_out, kernel_size=5)
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(conv2_out * 6 * 6, encoder_hidden_dim)
        self.fc2 = nn.Linear(encoder_hidden_dim, feature_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = _flatten_batch(x)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        return x


class LeNetLSTMClassifier(nn.Module):
    """Task-specific temporal baseline: LeNet-style frame encoder + LSTM."""

    def __init__(
        self,
        num_classes: int = 12,
        feature_dim: int = 128,
        encoder_hidden_dim: int = 160,
        hidden_size: int = 128,
        num_layers: int = 1,
        dropout: float = 0.3,
        use_last_only: bool = False,
        bidirectional: bool = False,
    ) -> None:
        super().__init__()
        self.frame_encoder = LeNetFrameEncoder(
            in_channels=1,
            encoder_hidden_dim=encoder_hidden_dim,
            feature_dim=feature_dim,
            dropout=dropout,
        )
        self.lstm = nn.LSTM(
            input_size=feature_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=bidirectional,
        )
        out_dim = hidden_size * (2 if bidirectional else 1)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(out_dim, num_classes)
        self.use_last_only = use_last_only

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = _reshape_temporal_frames(x)
        b, t, h, w = x.shape
        frames = x.view(b * t, 1, h, w)
        features = self.frame_encoder(frames).view(b, t, -1)
        lstm_out, _ = self.lstm(features)
        pooled = _pool_temporal_features(lstm_out, self.use_last_only)
        return self.head(self.dropout(pooled))


class LeNetLSTMMeanMaxClassifier(LeNetLSTMClassifier):
    """Post-hoc targeted variant: LeNet-style frame encoder + LSTM + mean-max temporal pooling."""

    def __init__(
        self,
        num_classes: int = 12,
        feature_dim: int = 128,
        encoder_hidden_dim: int = 160,
        hidden_size: int = 128,
        num_layers: int = 1,
        dropout: float = 0.3,
        use_last_only: bool = False,
        bidirectional: bool = False,
    ) -> None:
        super().__init__(
            num_classes=num_classes,
            feature_dim=feature_dim,
            encoder_hidden_dim=encoder_hidden_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            use_last_only=use_last_only,
            bidirectional=bidirectional,
        )
        out_dim = hidden_size * (2 if bidirectional else 1)
        self.head = nn.Linear(out_dim * 2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = _reshape_temporal_frames(x)
        b, t, h, w = x.shape
        frames = x.view(b * t, 1, h, w)
        features = self.frame_encoder(frames).view(b, t, -1)
        lstm_out, _ = self.lstm(features)
        mean_pool = lstm_out.mean(dim=1)
        max_pool = lstm_out.max(dim=1).values
        pooled = torch.cat([mean_pool, max_pool], dim=1)
        return self.head(self.dropout(pooled))


class LeNetLSTMMotionInputClassifier(LeNetLSTMClassifier):
    """Post-hoc targeted variant: LeNet-style frame encoder + explicit motion-input channel + LSTM."""

    def __init__(
        self,
        num_classes: int = 12,
        feature_dim: int = 128,
        encoder_hidden_dim: int = 160,
        hidden_size: int = 128,
        num_layers: int = 1,
        dropout: float = 0.3,
        use_last_only: bool = False,
        bidirectional: bool = False,
    ) -> None:
        super().__init__(
            num_classes=num_classes,
            feature_dim=feature_dim,
            encoder_hidden_dim=encoder_hidden_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            use_last_only=use_last_only,
            bidirectional=bidirectional,
        )
        self.frame_encoder = LeNetFrameEncoder(
            in_channels=2,
            encoder_hidden_dim=encoder_hidden_dim,
            feature_dim=feature_dim,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = _reshape_temporal_frames(x)
        b, t, h, w = x.shape
        motion = torch.zeros_like(x)
        motion[:, 1:] = (x[:, 1:] - x[:, :-1]).abs()
        frames_2ch = torch.stack([x, motion], dim=2)
        features = self.frame_encoder(frames_2ch.view(b * t, 2, h, w)).view(b, t, -1)
        lstm_out, _ = self.lstm(features)
        pooled = _pool_temporal_features(lstm_out, self.use_last_only)
        return self.head(self.dropout(pooled))


class CNNLSTM(nn.Module):
    """Task-specific temporal baseline: per-frame CNN backbone + sequence LSTM."""

    def __init__(self, num_classes: int = 12, lstm_hidden: int = 128, dropout: float = 0.5) -> None:
        super().__init__()
        # 🏆 最佳参数: backbone=resnet18, lstm_hidden=128, lstm_layers=1, lstm_dropout=0.4, dropout=0.5, use_last_only=False (val_acc=81.90%)
        backbone = models.resnet18(weights=None)
        conv1 = backbone.conv1
        backbone.conv1 = nn.Conv2d(
            1,
            conv1.out_channels,
            kernel_size=conv1.kernel_size,
            stride=conv1.stride,
            padding=conv1.padding,
            bias=False,
        )
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
        self.feature_dropout = nn.Dropout(dropout)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(lstm_hidden, num_classes)
        self.use_last_only = False  # 最佳配置：使用全时序平均

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = _reshape_temporal_frames(x)
        b, t, h, w = x.shape
        x = x.view(b * t, 1, h, w)

        # CNN特征提取 (ResNet18)
        features = self.frame_extractor(x)
        # ResNet18需要全局池化
        features = F.adaptive_avg_pool2d(features, (1, 1)).flatten(1)

        # 重塑为序列格式 (b, t, feature_dim)
        features = self.feature_dropout(features.view(b, t, -1))

        # LSTM处理
        lstm_out, _ = self.lstm(features)

        return self.head(self.dropout(_pool_temporal_features(lstm_out, self.use_last_only)))


class TemporalBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, dilation: int, kernel_size: int = 3, dropout: float = 0.0) -> None:
        super().__init__()
        pad = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size, padding=pad, dilation=dilation)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, padding=pad, dilation=dilation)
        self.relu1 = nn.ReLU(inplace=True)
        self.relu2 = nn.ReLU(inplace=True)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.proj = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    @staticmethod
    def _crop_causal(y: torch.Tensor, conv: nn.Conv1d) -> torch.Tensor:
        crop = conv.padding[0]
        if crop > 0:
            y = y[:, :, :-crop]
        return y

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv1(x)
        y = self._crop_causal(y, self.conv1)
        y = self.relu1(y)
        y = self.dropout1(y)

        y = self.conv2(y)
        y = self._crop_causal(y, self.conv2)
        y = self.relu2(y)
        y = self.dropout2(y)

        return y + self.proj(x)


class TCNClassifier(nn.Module):
    """Task-specific temporal baseline with dilated causal 1D convolutions."""

    def __init__(self, num_classes: int = 12, num_channels: int = 512, dropout: float = 0.0) -> None:
        super().__init__()
        in_dim = 32 * 32
        self.blocks = nn.Sequential(
            TemporalBlock(in_dim, num_channels, dilation=1, kernel_size=7, dropout=dropout),
            TemporalBlock(num_channels, num_channels, dilation=2, kernel_size=7, dropout=dropout),
            TemporalBlock(num_channels, num_channels, dilation=4, kernel_size=7, dropout=dropout),
            TemporalBlock(num_channels, num_channels, dilation=8, kernel_size=7, dropout=dropout),
            TemporalBlock(num_channels, num_channels, dilation=16, kernel_size=7, dropout=dropout),
        )
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(num_channels, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = _reshape_temporal_frames(x)
        x = x.view(x.size(0), x.size(1), -1).transpose(1, 2)
        y = self.blocks(x)
        return self.head(self.dropout(y[:, :, -1]))


# --- 4. 统一的 ModelFactory ---
class ModelFactory:
    @staticmethod
    def build_torch(name: str, in_channels: int = 80, num_classes: int = 12, **kwargs) -> Tuple[nn.Module, str]:
        n = name.lower()
        if n == "lenet":
            return LeNet(in_channels=in_channels, num_classes=num_classes), "cnn"

        if n == "resnet18":
            m = _make_resnet18_32(in_channels, num_classes)
            return m, "cnn"

        if n == "mobilenet_v2":
            m = _make_small_image_backbone(models.mobilenet_v2, in_channels)
            m = _replace_classifier_output(m, num_classes)
            return m, "cnn"

        if n == "efficientnet_b0":
            m = _make_small_image_backbone(models.efficientnet_b0, in_channels)
            m = _replace_classifier_output(m, num_classes)
            return m, "cnn"

        if n == "cnn_lstm":
            lstm_hidden = int(kwargs.get("lstm_hidden", 128))
            dropout = float(kwargs.get("dropout", 0.5))
            return CNNLSTM(num_classes=num_classes, lstm_hidden=lstm_hidden, dropout=dropout), "temporal"

        if n in {"lenet_lstm", "lenetlstm"}:
            feature_dim = int(kwargs.get("feature_dim", 128))
            encoder_hidden_dim = int(kwargs.get("encoder_hidden_dim", 160))
            hidden_size = int(kwargs.get("hidden_size", 128))
            num_layers = int(kwargs.get("num_layers", 1))
            dropout = float(kwargs.get("dropout", 0.3))
            use_last_only = bool(kwargs.get("use_last_only", False))
            bidirectional = bool(kwargs.get("bidirectional", False))
            return LeNetLSTMClassifier(
                num_classes=num_classes,
                feature_dim=feature_dim,
                encoder_hidden_dim=encoder_hidden_dim,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout,
                use_last_only=use_last_only,
                bidirectional=bidirectional,
            ), "temporal"

        if n in {"lenet_lstm_meanmax", "lenetlstm_meanmax"}:
            feature_dim = int(kwargs.get("feature_dim", 128))
            encoder_hidden_dim = int(kwargs.get("encoder_hidden_dim", 160))
            hidden_size = int(kwargs.get("hidden_size", 128))
            num_layers = int(kwargs.get("num_layers", 1))
            dropout = float(kwargs.get("dropout", 0.3))
            use_last_only = bool(kwargs.get("use_last_only", False))
            bidirectional = bool(kwargs.get("bidirectional", False))
            return LeNetLSTMMeanMaxClassifier(
                num_classes=num_classes,
                feature_dim=feature_dim,
                encoder_hidden_dim=encoder_hidden_dim,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout,
                use_last_only=use_last_only,
                bidirectional=bidirectional,
            ), "temporal"

        if n in {"lenet_lstm_motioninput", "lenetlstm_motioninput", "lenet_lstm_motion"}:
            feature_dim = int(kwargs.get("feature_dim", 128))
            encoder_hidden_dim = int(kwargs.get("encoder_hidden_dim", 160))
            hidden_size = int(kwargs.get("hidden_size", 128))
            num_layers = int(kwargs.get("num_layers", 1))
            dropout = float(kwargs.get("dropout", 0.3))
            use_last_only = bool(kwargs.get("use_last_only", False))
            bidirectional = bool(kwargs.get("bidirectional", False))
            return LeNetLSTMMotionInputClassifier(
                num_classes=num_classes,
                feature_dim=feature_dim,
                encoder_hidden_dim=encoder_hidden_dim,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout,
                use_last_only=use_last_only,
                bidirectional=bidirectional,
            ), "temporal"

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

        if n == "transformer":
            d_model = int(kwargs.get("d_model", 128))
            nhead = int(kwargs.get("nhead", 4))
            num_layers = int(kwargs.get("num_layers", 2))
            dim_feedforward = int(kwargs.get("dim_feedforward", 256))
            dropout = float(kwargs.get("dropout", 0.1))
            pooling = str(kwargs.get("pooling", "mean"))
            norm_first = bool(kwargs.get("norm_first", False))
            return TransformerClassifier(
                num_classes=num_classes,
                d_model=d_model,
                nhead=nhead,
                num_layers=num_layers,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                pooling=pooling,
                norm_first=norm_first,
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
