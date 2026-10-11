"""
model.py

Two-stream camouflaged object segmentation network:
  Stream A (appearance): takes the raw RGB frame.
  Stream B (motion):     takes the single-channel motion-magnitude map computed
                          from optical flow on the AMPLIFIED frame pair.

At each of 4 spatial scales, motion features gate (attention-weight) the
appearance features via a sigmoid gate, then a U-Net-style decoder upsamples
the fused features back to full resolution to produce the segmentation mask.

Backbones here are small custom CNNs (not pretrained ImageNet weights) because
this environment cannot reach the servers that host pretrained torchvision
weights. If you're running this somewhere with internet access, swap
AppearanceEncoder's stem/stages for a pretrained ResNet-18/PVTv2 backbone
(torchvision.models.resnet18(weights=...)) for a meaningful accuracy boost —
the rest of the architecture (fusion + decoder) does not need to change.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_block(in_ch: int, out_ch: int, stride: int = 1) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class AppearanceEncoder(nn.Module):
    """4-stage encoder over the RGB frame. Channels: 32 -> 64 -> 128 -> 256."""

    def __init__(self, in_ch: int = 3, base_ch: int = 32):
        super().__init__()
        self.stage1 = conv_block(in_ch, base_ch)                  # H,   W
        self.stage2 = conv_block(base_ch, base_ch * 2, stride=2)  # H/2, W/2
        self.stage3 = conv_block(base_ch * 2, base_ch * 4, stride=2)  # H/4, W/4
        self.stage4 = conv_block(base_ch * 4, base_ch * 8, stride=2)  # H/8, W/8

    def forward(self, x: torch.Tensor):
        f1 = self.stage1(x)
        f2 = self.stage2(f1)
        f3 = self.stage3(f2)
        f4 = self.stage4(f3)
        return [f1, f2, f3, f4]  # multi-scale appearance features


class MotionEncoder(nn.Module):
    """
    Lighter 4-stage encoder over the single-channel motion-magnitude map.
    Produces feature maps at the SAME 4 spatial resolutions as AppearanceEncoder
    so they can be fused scale-by-scale.
    """

    def __init__(self, in_ch: int = 1, base_ch: int = 16):
        super().__init__()
        self.stage1 = conv_block(in_ch, base_ch)
        self.stage2 = conv_block(base_ch, base_ch * 2, stride=2)
        self.stage3 = conv_block(base_ch * 2, base_ch * 4, stride=2)
        self.stage4 = conv_block(base_ch * 4, base_ch * 8, stride=2)

    def forward(self, x: torch.Tensor):
        f1 = self.stage1(x)
        f2 = self.stage2(f1)
        f3 = self.stage3(f2)
        f4 = self.stage4(f3)
        return [f1, f2, f3, f4]


class GatedFusion(nn.Module):
    """
    Fuses appearance features F_a and motion features F_m at one scale:
        gate = sigmoid(Conv(F_m))                 # projected to F_a's channel count
        F_fused = F_a * gate + F_a                 # residual gating
    Where there's no motion, the gate is ~0 and the network falls back to pure
    appearance features. Where there IS motion, that region gets boosted.
    """

    def __init__(self, appearance_ch: int, motion_ch: int):
        super().__init__()
        self.project = nn.Conv2d(motion_ch, appearance_ch, kernel_size=1)

    def forward(self, f_appearance: torch.Tensor, f_motion: torch.Tensor) -> torch.Tensor:
        gate = torch.sigmoid(self.project(f_motion))
        return f_appearance * gate + f_appearance


class DecoderBlock(nn.Module):
    """Upsample + concatenate skip connection + conv."""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.reduce = nn.Conv2d(in_ch, out_ch, kernel_size=1)
        self.conv = conv_block(out_ch + skip_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.reduce(x)
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class TwoStreamCOD(nn.Module):
    """
    Full model: AppearanceEncoder + MotionEncoder + per-scale GatedFusion +
    U-Net-style decoder -> single-channel logit mask at full input resolution.
    """

    def __init__(self, base_ch: int = 32):
        super().__init__()
        self.appearance_encoder = AppearanceEncoder(in_ch=3, base_ch=base_ch)
        self.motion_encoder = MotionEncoder(in_ch=1, base_ch=base_ch // 2)

        app_channels = [base_ch, base_ch * 2, base_ch * 4, base_ch * 8]
        mot_channels = [base_ch // 2, base_ch, base_ch * 2, base_ch * 4]

        self.fusions = nn.ModuleList(
            [GatedFusion(a, m) for a, m in zip(app_channels, mot_channels)]
        )

        # Decoder: start from the deepest fused scale and progressively upsample,
        # merging with shallower fused skip connections.
        self.dec3 = DecoderBlock(app_channels[3], app_channels[2], app_channels[2])
        self.dec2 = DecoderBlock(app_channels[2], app_channels[1], app_channels[1])
        self.dec1 = DecoderBlock(app_channels[1], app_channels[0], app_channels[0])

        self.head = nn.Conv2d(app_channels[0], 1, kernel_size=1)

    def forward(self, rgb: torch.Tensor, motion_map: torch.Tensor) -> torch.Tensor:
        """
        rgb:         (B, 3, H, W) float, normalized RGB frame.
        motion_map:  (B, 1, H, W) float, normalized optical-flow magnitude map
                     computed from the AMPLIFIED frame pair.
        Returns:     (B, 1, H, W) raw logits (apply sigmoid outside for probabilities).
        """
        app_feats = self.appearance_encoder(rgb)
        mot_feats = self.motion_encoder(motion_map)

        fused = [
            fusion(app_feats[i], mot_feats[i]) for i, fusion in enumerate(self.fusions)
        ]  # fused[0..3] at scales H, H/2, H/4, H/8

        x = self.dec3(fused[3], fused[2])
        x = self.dec2(x, fused[1])
        x = self.dec1(x, fused[0])

        logits = self.head(x)
        logits = F.interpolate(logits, size=rgb.shape[-2:], mode="bilinear", align_corners=False)
        return logits


if __name__ == "__main__":
    # Quick shape sanity check.
    model = TwoStreamCOD(base_ch=32)
    rgb = torch.randn(2, 3, 256, 256)
    motion = torch.randn(2, 1, 256, 256)
    out = model(rgb, motion)
    print("Output shape:", out.shape)  # expect (2, 1, 256, 256)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {n_params:,}")
