"""PyTorch TC-ResNet Streaming Architecture for ESP32-P4 KWS Engine."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, dilation: int = 1):
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=1, dilation=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.shortcut(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = F.relu(out + res)
        return out


class TCResNet(nn.Module):
    def __init__(self, num_classes: int = 2, in_bins: int = 40):
        super().__init__()
        self.num_classes = num_classes
        self.in_bins = in_bins

        # Conv0: (40 -> 16)
        self.conv0 = nn.Conv1d(in_bins, 16, kernel_size=3, padding=1, bias=False)
        self.bn0 = nn.BatchNorm1d(16)

        # ResBlocks
        self.block1 = ConvBlock(16, 16, kernel_size=3, dilation=1)
        self.block2 = ConvBlock(16, 32, kernel_size=3, dilation=2)
        self.block3 = ConvBlock(32, 32, kernel_size=3, dilation=2)

        # Dense Classifier
        self.classifier = nn.Linear(32, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input shape: (Batch, Bins, Time)
        x = F.relu(self.bn0(self.conv0(x)))
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)

        # Global Average Pooling over Time dimension
        x = x.mean(dim=-1)
        logits = self.classifier(x)
        return logits


def fold_batchnorm_into_conv(conv: nn.Conv1d, bn: nn.BatchNorm1d) -> Tuple[torch.Tensor, torch.Tensor]:
    """Fold BatchNorm parameters into Conv1d weights and biases for native C inference."""
    with torch.no_grad():
        gamma = bn.weight
        beta = bn.bias
        mean = bn.running_mean
        var = bn.running_var
        eps = bn.eps

        std = torch.sqrt(var + eps)
        w_scaled = conv.weight * (gamma / std).view(-1, 1, 1)

        b_conv = conv.bias if conv.bias is not None else torch.zeros_like(mean)
        b_scaled = (b_conv - mean) * (gamma / std) + beta

    return w_scaled, b_scaled


def extract_folded_weights(model: TCResNet) -> Dict[str, torch.Tensor]:
    """Extracts all model layers with folded BatchNormalizations for ESPM export."""
    model.eval()
    folded = {}

    # conv0
    w0, b0 = fold_batchnorm_into_conv(model.conv0, model.bn0)
    folded["conv0.weight"] = w0
    folded["conv0.bias"] = b0

    # block1
    w, b = fold_batchnorm_into_conv(model.block1.conv1, model.block1.bn1)
    folded["block1.conv1.weight"] = w
    folded["block1.conv1.bias"] = b
    w, b = fold_batchnorm_into_conv(model.block1.conv2, model.block1.bn2)
    folded["block1.conv2.weight"] = w
    folded["block1.conv2.bias"] = b

    # block2
    w, b = fold_batchnorm_into_conv(model.block2.conv1, model.block2.bn1)
    folded["block2.conv1.weight"] = w
    folded["block2.conv1.bias"] = b
    w, b = fold_batchnorm_into_conv(model.block2.conv2, model.block2.bn2)
    folded["block2.conv2.weight"] = w
    folded["block2.conv2.bias"] = b
    w, b = fold_batchnorm_into_conv(model.block2.shortcut[0], model.block2.shortcut[1])
    folded["block2.skip.weight"] = w
    folded["block2.skip.bias"] = b

    # block3
    w, b = fold_batchnorm_into_conv(model.block3.conv1, model.block3.bn1)
    folded["block3.conv1.weight"] = w
    folded["block3.conv1.bias"] = b
    w, b = fold_batchnorm_into_conv(model.block3.conv2, model.block3.bn2)
    folded["block3.conv2.weight"] = w
    folded["block3.conv2.bias"] = b

    # classifier
    folded["classifier.weight"] = model.classifier.weight.detach().clone()
    folded["classifier.bias"] = model.classifier.bias.detach().clone()

    return folded
