"""ViT-Base/16 encoder paired with a U-Net-style decoder.

The encoder is loaded from timm with ImageNet-21k pretrained weights. The
decoder is freshly initialised and receives skip connections from four
intermediate encoder blocks (layers 3, 6, 9, 12 by default; see
``configs/vit.yaml``).

Architectural details match Sections 4.4 and 4.6 of the report.
"""
from __future__ import annotations
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------------------------------- decoder blocks
class DecoderBlock(nn.Module):
    """Upsample + concatenate skip + two Conv-BN-ReLU layers."""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int) -> None:
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch + skip_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor | None) -> torch.Tensor:
        x = self.up(x)
        if skip is not None:
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = torch.cat([x, skip], dim=1)
        return self.conv(x)


# ----------------------------------------------------------- full model
class ViTUNet(nn.Module):
    """ViT-Base/16 encoder + U-Net decoder for 3-class segmentation."""

    def __init__(
        self,
        encoder_name: str = "vit_base_patch16_224",
        pretrained: bool = True,
        num_classes: int = 3,
        decoder_channels: Iterable[int] = (256, 128, 64, 32),
        skip_layers: Iterable[int] = (3, 6, 9, 12),
    ) -> None:
        super().__init__()
        try:
            import timm
        except ImportError as e:
            raise ImportError(
                "timm is required for the ViT encoder. Install via "
                "`pip install timm==0.9.16`."
            ) from e

        self.encoder = timm.create_model(
            encoder_name,
            pretrained=pretrained,
            num_classes=0,
            features_only=False,
        )
        self.skip_layers = tuple(int(i) for i in skip_layers)

        # The ViT hidden size for Base/16 is 768; project to decoder widths.
        hidden = self.encoder.embed_dim
        d1, d2, d3, d4 = decoder_channels

        self.dec1 = DecoderBlock(hidden, hidden, d1)
        self.dec2 = DecoderBlock(d1, hidden, d2)
        self.dec3 = DecoderBlock(d2, hidden, d3)
        self.dec4 = DecoderBlock(d3, hidden, d4)
        self.head = nn.Conv2d(d4, num_classes, kernel_size=1)

        # Cache for the intermediate token sequences captured during fwd.
        self._skips: list[torch.Tensor] = []
        self._install_hooks()

    # ------------------------------------------------ hooks
    def _install_hooks(self) -> None:
        # timm ViTs expose `.blocks` as a ModuleList of Transformer blocks.
        for idx, block in enumerate(self.encoder.blocks, start=1):
            if idx in self.skip_layers:
                block.register_forward_hook(self._capture)

    def _capture(self, module, inputs, output) -> None:
        # output shape: (B, N_tokens, C). Drop the CLS token and reshape to 2D.
        self._skips.append(output)

    # ------------------------------------------------ forward
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _, h, w = x.shape
        self._skips = []

        # Encoder forward (timm returns the final-layer token sequence).
        _ = self.encoder.forward_features(x)
        if len(self._skips) < len(self.skip_layers):
            raise RuntimeError(
                "Encoder skip hooks fired fewer times than expected; "
                "check that skip_layers indices are valid for this ViT."
            )

        # The deepest captured layer becomes the decoder bottleneck.
        skips_2d = [self._tokens_to_feature_map(t, h, w) for t in self._skips]
        bottleneck = skips_2d[-1]
        d = self.dec1(bottleneck, skips_2d[-1])
        d = self.dec2(d, skips_2d[-2])
        d = self.dec3(d, skips_2d[-3])
        d = self.dec4(d, skips_2d[-4])
        logits = self.head(d)
        if logits.shape[-2:] != (h, w):
            logits = F.interpolate(logits, size=(h, w), mode="bilinear", align_corners=False)
        return logits

    @staticmethod
    def _tokens_to_feature_map(tokens: torch.Tensor, h: int, w: int) -> torch.Tensor:
        """(B, 1+N, C) -> (B, C, H/16, W/16). Drops the CLS token."""
        b, n_plus_one, c = tokens.shape
        n = n_plus_one - 1
        side = int(round(n ** 0.5))
        if side * side != n:
            raise ValueError(f"Non-square ViT token grid: n={n}, expected square.")
        patch_tokens = tokens[:, 1:, :].transpose(1, 2)            # B, C, N
        return patch_tokens.reshape(b, c, side, side)


def build_model(model_cfg: dict, num_classes: int = 3) -> ViTUNet:
    """Convenience factory that takes the model section of the YAML config."""
    return ViTUNet(
        encoder_name=model_cfg.get("encoder", "vit_base_patch16_224"),
        pretrained=str(model_cfg.get("pretrained", "imagenet21k")) != "none",
        num_classes=int(model_cfg.get("num_classes", num_classes)),
        decoder_channels=model_cfg.get("decoder_channels", (256, 128, 64, 32)),
        skip_layers=model_cfg.get("skip_connections_from", (3, 6, 9, 12)),
    )
