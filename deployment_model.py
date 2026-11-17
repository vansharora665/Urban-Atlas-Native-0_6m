"""Exact joint five-class network; auxiliary heads are retained for strict loading."""

import torch
from torch import nn
from torch.nn import functional as F
import segmentation_models_pytorch as smp


def conv_block(inputs, outputs, dilation=1):
    return nn.Sequential(
        nn.Conv2d(inputs, outputs, 3, padding=dilation, dilation=dilation, bias=False),
        nn.GroupNorm(4, outputs), nn.SiLU())


class DeploymentModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = smp.UnetPlusPlus(
            encoder_name="resnet34", encoder_weights=None,
            decoder_attention_type="scse", in_channels=4, classes=5, activation=None)
        self.dropout = nn.Dropout2d(.10)
        self.road_head = nn.Conv2d(16, 1, 1)
        self.boundary_head = nn.Conv2d(16, 1, 1)
        self.native_detail = nn.Sequential(conv_block(4, 16), conv_block(16, 16, 2))
        self.native_fusion = nn.Sequential(conv_block(32, 32), conv_block(32, 16, 3))
        self.native_output = nn.Conv2d(16, 5, 1)

    def source_logits(self, image):
        height, width = image.shape[-2:]
        image = F.pad(image, (0, (32 - width % 32) % 32,
                              0, (32 - height % 32) % 32), mode="replicate")
        features = self.network.encoder(image)
        decoded = self.dropout(self.network.decoder(features))
        detail = self.native_detail(image)
        residual = self.native_output(self.native_fusion(torch.cat((decoded, detail), dim=1)))
        return (self.network.segmentation_head(decoded) + residual)[..., :height, :width]

    def forward(self, image):
        # Original training order: water, roads, built-up, barren, vegetation.
        return self.source_logits(image)[:, [0, 4, 3, 2, 1]]
