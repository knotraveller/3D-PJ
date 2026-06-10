"""ZeroGS-UNet: feed-forward posed-image encoder for 3D Gaussian parameters."""

from __future__ import annotations

from typing import Dict, Optional

import torch
from torch import nn

try:
    from .gaussian_utils import raw_to_gaussians
    from .modules import Downsample, MaybeAttention, ResBlock, Upsample, make_group_norm
except ImportError:  # Allows `python code/models/zerogs_unet.py`.
    from gaussian_utils import raw_to_gaussians
    from modules import Downsample, MaybeAttention, ResBlock, Upsample, make_group_norm


def _valid_num_heads(channels: int, requested_heads: int) -> int:
    """Use requested_heads when possible, otherwise find a smaller divisor."""
    for heads in range(min(channels, requested_heads), 0, -1):
        if channels % heads == 0:
            return heads
    raise ValueError(f"No valid attention head count for channels={channels}.")


class ZeroGSUNet(nn.Module):
    """Simplified LGM-style multi-view U-Net for Gaussian Splatting parameters.

    Input:
        x: [B, V, 9, image_size, image_size]
           9 channels = RGB(3) + Pluecker ray embedding(6).

    Output with K/c2w:
        raw: [B, V, 16, splat_size, splat_size]
        gaussian_map: [B, V, 14, splat_size, splat_size]
        gaussians: [B, V*splat_size*splat_size, 14]
    """

    def __init__(
        self,
        in_channels: int = 9,
        num_views: int = 7,
        image_size: int = 256,
        splat_size: int = 64,
        base_channels: int = 64,
        out_raw_channels: int = 16,
        depth_min: float = 2.5,
        depth_max: float = 5.5,
        offset_scale: float = 0.05,
        opacity_bias: float = 4.0,
        attention_heads: int = 8,
        use_view_attention: bool = True,
        use_global_attention: bool = True,
    ) -> None:
        super().__init__()
        if image_size % 16 != 0:
            raise ValueError("image_size must be divisible by 16.")
        if splat_size != image_size // 4:
            raise ValueError("splat_size must equal image_size // 4 for this U-Net.")
        if out_raw_channels != 16:
            raise ValueError("out_raw_channels must be 16 for the Gaussian head layout.")

        self.in_channels = int(in_channels)
        self.num_views = int(num_views)
        self.image_size = int(image_size)
        self.splat_size = int(splat_size)
        self.out_raw_channels = int(out_raw_channels)
        self.depth_min = float(depth_min)
        self.depth_max = float(depth_max)
        self.offset_scale = float(offset_scale)
        self.opacity_bias = float(opacity_bias)

        c0 = int(base_channels)
        c1 = c0 * 2
        c2 = c0 * 4
        c3 = c0 * 8
        cb = c3

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, c0, kernel_size=3, padding=1),
            make_group_norm(c0),
            nn.SiLU(),
        )

        self.enc0 = nn.Sequential(
            ResBlock(c0, c0),
            ResBlock(c0, c0),
        )
        self.down0 = Downsample(c0, c1)

        self.enc1 = nn.Sequential(
            ResBlock(c1, c1),
            ResBlock(c1, c1),
        )
        self.down1 = Downsample(c1, c2)

        self.enc2_res0 = ResBlock(c2, c2)
        self.enc2_attn = MaybeAttention(
            channels=c2,
            num_views=num_views,
            num_heads=_valid_num_heads(c2, attention_heads),
            mode="view",
            enabled=use_view_attention,
        )
        self.enc2_res1 = ResBlock(c2, c2)
        self.down2 = Downsample(c2, c3)

        self.enc3_res0 = ResBlock(c3, c3)
        self.enc3_attn = MaybeAttention(
            channels=c3,
            num_views=num_views,
            num_heads=_valid_num_heads(c3, attention_heads),
            mode="view",
            enabled=use_view_attention,
        )
        self.enc3_res1 = ResBlock(c3, c3)
        self.down3 = Downsample(c3, cb)

        self.bottleneck_res0 = ResBlock(cb, cb)
        self.bottleneck_attn = MaybeAttention(
            channels=cb,
            num_views=num_views,
            num_heads=_valid_num_heads(cb, attention_heads),
            mode="global",
            enabled=use_global_attention,
        )
        self.bottleneck_res1 = ResBlock(cb, cb)

        self.up0 = Upsample(cb, c3)
        self.dec0_res0 = ResBlock(c3 + c3, c3)
        self.dec0_attn = MaybeAttention(
            channels=c3,
            num_views=num_views,
            num_heads=_valid_num_heads(c3, attention_heads),
            mode="view",
            enabled=use_view_attention,
        )
        self.dec0_res1 = ResBlock(c3, c3)

        self.up1 = Upsample(c3, c2)
        self.dec1_res0 = ResBlock(c2 + c2, c2)
        self.dec1_attn = MaybeAttention(
            channels=c2,
            num_views=num_views,
            num_heads=_valid_num_heads(c2, attention_heads),
            mode="view",
            enabled=use_view_attention,
        )
        self.dec1_res1 = ResBlock(c2, c2)

        self.head = nn.Sequential(
            make_group_norm(c2),
            nn.SiLU(),
            nn.Conv2d(c2, c1, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(c1, out_raw_channels, kernel_size=1),
        )

    def forward(
        self,
        x: torch.Tensor,
        K: Optional[torch.Tensor] = None,
        c2w: Optional[torch.Tensor] = None,
        return_raw: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """Run ZeroGS-UNet.

        Args:
            x: [B, 7, 9, H, W] RGB + ray embedding.
            K: [B, 7, 3, 3], optional camera intrinsics.
            c2w: [B, 7, 4, 4], optional camera-to-world matrices.
            return_raw: include raw map when Gaussian post-processing is run.
        """
        if x.ndim != 5:
            raise ValueError(f"x must be [B, V, C, H, W], got {tuple(x.shape)}.")
        batch_size, num_views, channels, height, width = x.shape
        if num_views != self.num_views:
            raise ValueError(f"Expected V={self.num_views}, got {num_views}.")
        if channels != self.in_channels:
            raise ValueError(f"Expected C={self.in_channels}, got {channels}.")
        if height != self.image_size or width != self.image_size:
            raise ValueError(
                f"Expected image size {self.image_size}x{self.image_size}, "
                f"got {height}x{width}."
            )

        # Merge batch and view dimensions for 2D convolutions:
        # [B, V, 9, H, W] -> [B*V, 9, H, W].
        x = x.contiguous().reshape(batch_size * num_views, channels, height, width)

        # Stem: [B*V, 9, 256, 256] -> [B*V, 64, 256, 256] by default.
        x = self.stem(x)

        # Encoder level 0 keeps 256x256; skip0 is not decoded because output is 64x64.
        # x0/skip0: [B*V, 64, H, W].
        x = self.enc0(x)
        skip0 = x
        x = self.down0(x)  # [B*V, 128, H/2, W/2].

        # Encoder level 1 keeps 128x128; skip1 is also intentionally unused.
        # x1/skip1: [B*V, 128, H/2, W/2].
        x = self.enc1(x)
        skip1 = x
        x = self.down1(x)  # [B*V, 256, H/4, W/4].

        # Encoder level 2 works at splat resolution 64x64 for image_size=256.
        # skip2 is used by decoder up1.
        x = self.enc2_res0(x)
        x = self.enc2_attn(x, batch_size=batch_size)
        x = self.enc2_res1(x)
        skip2 = x  # [B*V, 256, H/4, W/4].
        x = self.down2(x)  # [B*V, 512, H/8, W/8].

        # Encoder level 3 works at 32x32 for image_size=256.
        # skip3 is used by decoder up0.
        x = self.enc3_res0(x)
        x = self.enc3_attn(x, batch_size=batch_size)
        x = self.enc3_res1(x)
        skip3 = x  # [B*V, 512, H/8, W/8].
        x = self.down3(x)  # [B*V, 512, H/16, W/16].

        # Bottleneck: 16x16 for image_size=256, with optional global attention.
        x = self.bottleneck_res0(x)
        x = self.bottleneck_attn(x, batch_size=batch_size)
        x = self.bottleneck_res1(x)

        # Decoder up 0: 16x16 -> 32x32, concatenate skip3 on channel dim.
        # [B*V, 512, H/16, W/16] -> [B*V, 512, H/8, W/8]
        # cat skip3 -> [B*V, 1024, H/8, W/8].
        x = self.up0(x)
        x = torch.cat([x, skip3], dim=1)
        x = self.dec0_res0(x)
        x = self.dec0_attn(x, batch_size=batch_size)
        x = self.dec0_res1(x)

        # Decoder up 1: 32x32 -> 64x64, concatenate skip2 on channel dim.
        # [B*V, 512, H/8, W/8] -> [B*V, 256, H/4, W/4]
        # cat skip2 -> [B*V, 512, H/4, W/4].
        x = self.up1(x)
        x = torch.cat([x, skip2], dim=1)
        x = self.dec1_res0(x)
        x = self.dec1_attn(x, batch_size=batch_size)
        feat = self.dec1_res1(x)  # [B*V, 256, splat_size, splat_size].

        # Gaussian head:
        # [B*V, 256, S, S] -> [B*V, 16, S, S]
        # -> [B, V, 16, S, S].
        raw = self.head(feat)
        if raw.shape[-2:] != (self.splat_size, self.splat_size):
            raise RuntimeError(
                f"Expected raw spatial size {self.splat_size}, got {tuple(raw.shape[-2:])}."
            )
        raw = raw.reshape(
            batch_size,
            num_views,
            self.out_raw_channels,
            self.splat_size,
            self.splat_size,
        )

        # Keep these references alive only for readable dimension comments above.
        _ = skip0, skip1

        if K is None or c2w is None:
            return {"raw": raw}

        gaussian_dict = raw_to_gaussians(
            raw=raw,
            K=K,
            c2w=c2w,
            image_size=self.image_size,
            splat_size=self.splat_size,
            depth_min=self.depth_min,
            depth_max=self.depth_max,
            offset_scale=self.offset_scale,
            opacity_bias=self.opacity_bias,
        )
        if return_raw:
            gaussian_dict["raw"] = raw
        return gaussian_dict


if __name__ == "__main__":
    B, V = 1, 7
    x = torch.randn(B, V, 9, 128, 128)
    model = ZeroGSUNet(image_size=128, splat_size=32, base_channels=16)
    out = model(x)
    print(out["raw"].shape)
