"""Reusable neural network blocks for ZeroGS-UNet."""

from __future__ import annotations

import torch
from torch import nn


def make_group_norm(channels: int, max_groups: int = 32) -> nn.GroupNorm:
    """Create GroupNorm with the largest valid group count up to max_groups."""
    if channels <= 0:
        raise ValueError("channels must be positive.")
    for groups in range(min(max_groups, channels), 0, -1):
        if channels % groups == 0:
            return nn.GroupNorm(num_groups=groups, num_channels=channels)
    raise ValueError(f"Could not find valid GroupNorm groups for channels={channels}.")


class ResBlock(nn.Module):
    """Pre-activation residual block.

    Shape:
        [B*V, C_in, H, W] -> [B*V, C_out, H, W]
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.main = nn.Sequential(
            make_group_norm(in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            make_group_norm(out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        )
        if in_channels == out_channels:
            self.skip = nn.Identity()
        else:
            self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.main(x) + self.skip(x)


class Downsample(nn.Module):
    """Stride-2 convolutional downsample.

    Shape:
        [B*V, C_in, H, W] -> [B*V, C_out, H/2, W/2]
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=2,
            padding=1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    """Nearest-neighbor upsample followed by convolution.

    Shape:
        [B*V, C_in, H, W] -> [B*V, C_out, 2H, 2W]
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MultiViewAttention(nn.Module):
    """Cross-view attention block.

    mode="global":
        [B*V, C, H, W] -> [B, V*H*W, C] -> attention -> [B*V, C, H, W]

    mode="view":
        [B*V, C, H, W] -> [B*H*W, V, C] -> attention over views
        -> [B*V, C, H, W]

    The global mode is useful at 16x16 bottlenecks. At higher resolutions,
    view mode is much cheaper because each attention sequence has only V tokens.
    """

    def __init__(
        self,
        channels: int,
        num_views: int = 7,
        num_heads: int = 8,
        mode: str = "global",
    ) -> None:
        super().__init__()
        if mode not in {"global", "view"}:
            raise ValueError('mode must be "global" or "view".')
        if channels % num_heads != 0:
            raise ValueError(
                f"channels={channels} must be divisible by num_heads={num_heads}."
            )

        self.channels = int(channels)
        self.num_views = int(num_views)
        self.mode = mode

        self.norm1 = nn.LayerNorm(channels)
        self.attn = nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=num_heads,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(channels)
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels * 4),
            nn.SiLU(),
            nn.Linear(channels * 4, channels),
        )

    def forward(self, x: torch.Tensor, batch_size: int) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"x must be [B*V, C, H, W], got shape {tuple(x.shape)}.")

        bv, channels, height, width = x.shape
        if channels != self.channels:
            raise ValueError(f"Expected C={self.channels}, got C={channels}.")
        if bv != batch_size * self.num_views:
            raise ValueError(
                f"Expected B*V={batch_size * self.num_views}, got {bv}."
            )

        if self.mode == "global":
            return self._forward_global(x, batch_size, height, width)
        return self._forward_view(x, batch_size, height, width)

    def _apply_attention(self, tokens: torch.Tensor) -> torch.Tensor:
        # tokens: [N, T, C]. Residual attention + residual MLP.
        attn_in = self.norm1(tokens)
        attn_out, _ = self.attn(attn_in, attn_in, attn_in, need_weights=False)
        tokens = tokens + attn_out
        tokens = tokens + self.mlp(self.norm2(tokens))
        return tokens

    def _forward_global(
        self,
        x: torch.Tensor,
        batch_size: int,
        height: int,
        width: int,
    ) -> torch.Tensor:
        # [B*V, C, H, W] -> [B, V, C, H, W]
        x = x.reshape(batch_size, self.num_views, self.channels, height, width)
        # [B, V, C, H, W] -> [B, V*H*W, C]
        tokens = x.permute(0, 1, 3, 4, 2).reshape(
            batch_size,
            self.num_views * height * width,
            self.channels,
        )
        tokens = self._apply_attention(tokens)
        # [B, V*H*W, C] -> [B, V, C, H, W] -> [B*V, C, H, W]
        return (
            tokens.reshape(batch_size, self.num_views, height, width, self.channels)
            .permute(0, 1, 4, 2, 3)
            .contiguous()
            .reshape(batch_size * self.num_views, self.channels, height, width)
        )

    def _forward_view(
        self,
        x: torch.Tensor,
        batch_size: int,
        height: int,
        width: int,
    ) -> torch.Tensor:
        # [B*V, C, H, W] -> [B, V, C, H, W]
        x = x.reshape(batch_size, self.num_views, self.channels, height, width)
        # [B, V, C, H, W] -> [B*H*W, V, C]
        tokens = (
            x.permute(0, 3, 4, 1, 2)
            .contiguous()
            .reshape(batch_size * height * width, self.num_views, self.channels)
        )
        tokens = self._apply_attention(tokens)
        # [B*H*W, V, C] -> [B, V, C, H, W] -> [B*V, C, H, W]
        return (
            tokens.reshape(batch_size, height, width, self.num_views, self.channels)
            .permute(0, 3, 4, 1, 2)
            .contiguous()
            .reshape(batch_size * self.num_views, self.channels, height, width)
        )


class MaybeAttention(nn.Module):
    """Optional MultiViewAttention wrapper used by ZeroGSUNet switches."""

    def __init__(
        self,
        channels: int,
        num_views: int,
        num_heads: int,
        mode: str,
        enabled: bool,
    ) -> None:
        super().__init__()
        self.enabled = bool(enabled)
        if self.enabled:
            self.block = MultiViewAttention(
                channels=channels,
                num_views=num_views,
                num_heads=num_heads,
                mode=mode,
            )
        else:
            self.block = None

    def forward(self, x: torch.Tensor, batch_size: int) -> torch.Tensor:
        if self.block is None:
            return x
        return self.block(x, batch_size=batch_size)
