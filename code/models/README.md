# ZeroGS-UNet

`ZeroGSUNet` consumes posed Zero123++-style views:

```text
images:  [B, 7, 3, H, W]
rays:    [B, 7, 6, H, W]
input:   [B, 7, 9, H, W]
raw:     [B, 7, 16, H/4, W/4]
output:  [B, 7*(H/4)*(W/4), 14]
```

The 14 Gaussian channels are:

```text
[x, y, z, opacity, sx, sy, sz, qw, qx, qy, qz, r, g, b]
```

`c2w` means camera-to-world, `w2c` means world-to-camera, and Blender camera local `-Z` is the forward direction. Pluecker ray embedding uses `[direction, moment]` by default, with `moment = origin x direction`.

## Attention Switches

The default model enables view attention at 64x64/32x32 and global attention at the bottleneck. Global attention over 32x32 or higher resolutions can be expensive, so the implementation keeps global attention only at the bottleneck by default.

For quick CPU tests or low-memory runs:

```python
model = ZeroGSUNet(
    image_size=128,
    splat_size=32,
    base_channels=16,
    use_view_attention=False,
    use_global_attention=False,
)
```

The attention implementation uses `nn.MultiheadAttention`; for larger training runs it can later be replaced with xformers or flash attention without changing the model interface.
