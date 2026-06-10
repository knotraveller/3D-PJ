你现在接手一个课程项目中的 3DGS 前馈重建模型搭建任务。项目目标是：输入 Zero123++ 格式的 7 张 posed images，直接前向输出一组 3D Gaussian Splatting 参数。暂时不需要实现 3DGS 渲染器、loss 或完整训练循环，只需要完成模型结构、维度流、Gaussian 参数后处理和最小测试。

背景信息：

先阅读对话历史信息和项目仓库，明确我所做的项目目的。

我们已经使用 Blender 渲染 Objaverse 本地 `.glb` 文件，渲染参数固定为：

```text
--resolution 256
--ref_azimuths 0
--fov 30
--camera_radius 4.0
--target_radius 0.8
```

每个样本包含 7 张图：

```text
view 0: cond view
view 1-6: Zero123++ 风格 target views
```

7 个视角的相机内外参已经由前面的 `camera_utils.py` 和 `ray_utils.py` 计算并保存。训练时可以通过：

```python
from ray_utils import get_embedding

ray_emb = get_embedding(K, c2w, resolution=256)
```

得到 Plücker ray embedding：

```text
ray_emb: [B, V, 6, H, W]
```

图像输入为：

```text
images: [B, V, 3, H, W]
```

其中：

```text
B = batch size
V = 7
H = W = 256
```

最终模型输入为 RGB 与 ray embedding 在 channel 维拼接：

```text
model_input = concat([images, ray_emb], dim=2)
model_input: [B, 7, 9, 256, 256]
```

模型目标：

```text
[B, 7, 9, 256, 256]
    ↓
ZeroGS-UNet
    ↓
[B, 7 * 64 * 64, 14]
```

其中每个 Gaussian 的 14 维参数为：

```text
[x, y, z, opacity, sx, sy, sz, qw, qx, qy, qz, r, g, b]
```

暂时不需要输出 spherical harmonics，只输出 RGB 颜色即可。

---

# 一、需要实现的文件

请新增以下文件：

```text
models/
  __init__.py
  zerogs_unet.py
  modules.py
  gaussian_utils.py

tests/
  test_zerogs_unet.py
```

如果项目已有类似目录，可以合并进去，但请保持接口清晰。

---

# 二、总模型名称

实现主类：

```python
class ZeroGSUNet(nn.Module):
    ...
```

建议初始化接口：

```python
model = ZeroGSUNet(
    in_channels=9,
    num_views=7,
    image_size=256,
    splat_size=64,
    base_channels=64,
    out_raw_channels=16,
    depth_min=2.5,
    depth_max=5.5,
    offset_scale=0.05,
)
```

其中：

```text
in_channels = 9
  RGB 3 通道 + Plücker ray embedding 6 通道

num_views = 7

image_size = 256

splat_size = 64
  最终每个 view 输出 64×64 个 Gaussian candidate

out_raw_channels = 16
  depth_raw: 1
  offset_raw: 3
  scale_raw: 3
  quat_raw: 4
  opacity_raw: 1
  rgb_raw: 3
  confidence_raw: 1
  total = 16
```

forward 接口建议：

```python
def forward(
    self,
    x: torch.Tensor,
    K: torch.Tensor | None = None,
    c2w: torch.Tensor | None = None,
    return_raw: bool = True,
) -> dict:
    ...
```

输入：

```text
x:   [B, 7, 9, 256, 256]
K:   [B, 7, 3, 3]
c2w: [B, 7, 4, 4]
```

输出 dict：

```python
{
    "gaussians": gaussians,          # [B, 7*64*64, 14]
    "gaussian_map": gaussian_map,    # [B, 7, 14, 64, 64]
    "raw": raw,                      # [B, 7, 16, 64, 64]
    "confidence": confidence,        # [B, 7, 1, 64, 64]
}
```

如果 `K is None` 或 `c2w is None`，模型仍然应该能运行，但只返回 raw prediction，不做 3D unprojection：

```python
{
    "raw": raw
}
```

---

# 三、整体模型结构

模型采用一个 LGM 风格的简化多视角 U-Net：

```text
Input:
  [B, 7, 9, 256, 256]

reshape:
  [B*7, 9, 256, 256]

Stem:
  Conv2d(9 → 64, kernel=3, stride=1, padding=1)
  GroupNorm
  SiLU

Encoder:
  Level 0: 256×256, channels 64
  Level 1: 128×128, channels 128
  Level 2: 64×64, channels 256
  Level 3: 32×32, channels 512
  Bottleneck: 16×16, channels 512

Decoder:
  16×16 → 32×32
  32×32 → 64×64

Output:
  [B*7, 16, 64, 64]

reshape:
  [B, 7, 16, 64, 64]
```

注意：decoder 不需要上采样回 256×256。最终 Gaussian map 的空间分辨率是 64×64。这样每个样本 Gaussian 数量为：

```text
N = 7 * 64 * 64 = 28672
```

---

# 四、基础模块实现

请在 `models/modules.py` 中实现以下模块。

---

## 1. make_group_norm

实现：

```python
def make_group_norm(channels: int, max_groups: int = 32) -> nn.GroupNorm:
    ...
```

要求：

* `num_groups` 不超过 `max_groups`；
* `channels % num_groups == 0`；
* 如果 32 不整除，则自动往下找可整除的 groups；
* 例如：

  * C=64 → groups=32
  * C=128 → groups=32
  * C=256 → groups=32
  * C=512 → groups=32

---

## 2. ResBlock

实现：

```python
class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        ...
```

结构：

```text
GroupNorm(in_channels)
SiLU
Conv2d(in_channels → out_channels, kernel=3, padding=1)

GroupNorm(out_channels)
SiLU
Conv2d(out_channels → out_channels, kernel=3, padding=1)

skip:
  如果 in_channels != out_channels:
    Conv2d(in_channels → out_channels, kernel=1)
  否则 Identity
```

forward：

```text
out = main_branch(x) + skip(x)
```

输入输出 shape：

```text
[B*V, C_in, H, W] → [B*V, C_out, H, W]
```

---

## 3. Downsample

实现：

```python
class Downsample(nn.Module):
    ...
```

结构：

```text
Conv2d(in_channels → out_channels, kernel=3, stride=2, padding=1)
```

shape：

```text
[B*V, C_in, H, W] → [B*V, C_out, H/2, W/2]
```

---

## 4. Upsample

实现：

```python
class Upsample(nn.Module):
    ...
```

结构：

```text
Nearest upsample scale_factor=2
Conv2d(in_channels → out_channels, kernel=3, stride=1, padding=1)
```

shape：

```text
[B*V, C_in, H, W] → [B*V, C_out, 2H, 2W]
```

---

## 5. MultiViewAttention

实现：

```python
class MultiViewAttention(nn.Module):
    def __init__(
        self,
        channels: int,
        num_views: int = 7,
        num_heads: int = 8,
        mode: str = "global",
    ):
        ...
```

需要支持两种模式。

---

### mode="global"

用于低分辨率特征，例如 32×32 和 16×16。

输入：

```text
x: [B*V, C, H, W]
```

其中需要知道 `B` 和 `V`。forward 接口建议：

```python
def forward(self, x: torch.Tensor, batch_size: int) -> torch.Tensor:
    ...
```

reshape：

```text
[B*V, C, H, W]
→ [B, V, C, H, W]
→ [B, V*H*W, C]
```

然后：

```text
LayerNorm(C)
MultiheadAttention(embed_dim=C, num_heads=num_heads, batch_first=True)
residual
LayerNorm(C)
MLP(C → 4C → C)
residual
```

再 reshape 回：

```text
[B, V*H*W, C]
→ [B, V, C, H, W]
→ [B*V, C, H, W]
```

注意：

* 32×32 时 token 数为 `7*32*32=7168`，可能较大；
* 如果显存不足，可以先只在 16×16 用 global attention；
* 代码中请保留可配置开关。并写入README中，详细说明如何使用。

---

### mode="view"

用于 64×64 或更高分辨率的轻量 cross-view attention。

它不是对所有空间 token 做全局 attention，而是在每个空间位置只对 7 个 view 做 attention。

输入：

```text
x: [B*V, C, H, W]
```

reshape：

```text
[B*V, C, H, W]
→ [B, V, C, H, W]
→ [B, H, W, V, C]
→ [B*H*W, V, C]
```

然后对 view 维做 MultiheadAttention：

```text
LayerNorm(C)
MultiheadAttention over V tokens
residual
LayerNorm(C)
MLP(C → 4C → C)
residual
```

再 reshape 回：

```text
[B*H*W, V, C]
→ [B, H, W, V, C]
→ [B, V, C, H, W]
→ [B*V, C, H, W]
```

这种模式复杂度小很多，适合在 64×64 使用。

---

# 五、ZeroGS-UNet 具体层级与维度变化

请在 `models/zerogs_unet.py` 中实现 `ZeroGSUNet`。

输入：

```text
x: [B, 7, 9, 256, 256]
```

先检查：

```python
assert x.ndim == 5
assert x.shape[1] == 7
assert x.shape[2] == 9
```

reshape：

```text
x = x.reshape(B * 7, 9, 256, 256)
```

---

## Stem

```text
Conv2d(9 → 64, kernel=3, padding=1)
GroupNorm(64)
SiLU
```

输出：

```text
x0: [B*7, 64, 256, 256]
```

---

## Encoder Level 0

```text
ResBlock(64 → 64)
ResBlock(64 → 64)
```

保存 skip：

```text
skip0: [B*7, 64, 256, 256]
```

下采样：

```text
Downsample(64 → 128)
```

输出：

```text
x1_in: [B*7, 128, 128, 128]
```

---

## Encoder Level 1

```text
ResBlock(128 → 128)
ResBlock(128 → 128)
```

保存 skip：

```text
skip1: [B*7, 128, 128, 128]
```

下采样：

```text
Downsample(128 → 256)
```

输出：

```text
x2_in: [B*7, 256, 64, 64]
```

---

## Encoder Level 2

```text
ResBlock(256 → 256)
MultiViewAttention(256, mode="view", num_heads=8)
ResBlock(256 → 256)
```

保存 skip：

```text
skip2: [B*7, 256, 64, 64]
```

下采样：

```text
Downsample(256 → 512)
```

输出：

```text
x3_in: [B*7, 512, 32, 32]
```

---

## Encoder Level 3

```text
ResBlock(512 → 512)
MultiViewAttention(512, mode="view", num_heads=8)
ResBlock(512 → 512)
```

保存 skip：

```text
skip3: [B*7, 512, 32, 32]
```

下采样：

```text
Downsample(512 → 512)
```

输出：

```text
xb: [B*7, 512, 16, 16]
```

---

## Bottleneck

```text
ResBlock(512 → 512)
MultiViewAttention(512, mode="global", num_heads=8)
ResBlock(512 → 512)
```

输出：

```text
xb: [B*7, 512, 16, 16]
```

---

## Decoder Up 0: 16×16 → 32×32

```text
Upsample(512 → 512)
```

输出：

```text
[B*7, 512, 32, 32]
```

concat skip3：

```text
cat([x, skip3], dim=1)
→ [B*7, 1024, 32, 32]
```

然后：

```text
ResBlock(1024 → 512)
MultiViewAttention(512, mode="view", num_heads=8)
ResBlock(512 → 512)
```

输出：

```text
[B*7, 512, 32, 32]
```

---

## Decoder Up 1: 32×32 → 64×64

```text
Upsample(512 → 256)
```

输出：

```text
[B*7, 256, 64, 64]
```

concat skip2：

```text
cat([x, skip2], dim=1)
→ [B*7, 512, 64, 64]
```

然后：

```text
ResBlock(512 → 256)
MultiViewAttention(256, mode="view", num_heads=8)
ResBlock(256 → 256)
```

输出：

```text
feat: [B*7, 256, 64, 64]
```

---

## Gaussian Head

结构：

```text
GroupNorm(256)
SiLU
Conv2d(256 → 128, kernel=3, padding=1)
SiLU
Conv2d(128 → 16, kernel=1)
```

输出：

```text
raw: [B*7, 16, 64, 64]
```

reshape：

```text
raw: [B, 7, 16, 64, 64]
```

16 个通道拆分：

```text
raw[:, :, 0:1]   depth_raw
raw[:, :, 1:4]   offset_raw
raw[:, :, 4:7]   scale_raw
raw[:, :, 7:11]  quat_raw
raw[:, :, 11:12] opacity_raw
raw[:, :, 12:15] rgb_raw
raw[:, :, 15:16] confidence_raw
```

---

# 六、Gaussian 参数后处理

请在 `utils/gaussian_utils.py` 中实现。

主函数：

```python
def raw_to_gaussians(
    raw: torch.Tensor,
    K: torch.Tensor,
    c2w: torch.Tensor,
    image_size: int = 256,
    splat_size: int = 64,
    depth_min: float = 2.5,
    depth_max: float = 5.5,
    offset_scale: float = 0.05,
) -> dict:
    ...
```

输入：

```text
raw: [B, V, 16, 64, 64]
K:   [B, V, 3, 3]
c2w: [B, V, 4, 4]
```

输出：

```python
{
    "gaussians": gaussians,          # [B, V*64*64, 14]
    "gaussian_map": gaussian_map,    # [B, V, 14, 64, 64]
    "confidence": confidence,        # [B, V, 1, 64, 64]
}
```

---

## 1. depth

```python
depth = depth_min + (depth_max - depth_min) * sigmoid(depth_raw)
```

shape：

```text
depth: [B, V, 1, 64, 64]
```

---

## 2. feature cell 对应原图像素坐标

因为原图是 256×256，Gaussian map 是 64×64，所以每个 cell 对应原图 4×4 区域。

对于 Gaussian map 中坐标：

```text
i = 0,...,63
j = 0,...,63
```

对应原图像素中心为：

```python
u = (j + 0.5) * (image_size / splat_size)
v = (i + 0.5) * (image_size / splat_size)
```

当 `image_size=256, splat_size=64` 时：

```text
u = (j + 0.5) * 4
v = (i + 0.5) * 4
```

---

## 3. 根据 depth + K + c2w 反投影得到 Gaussian center

Blender camera convention：

```text
camera local +X: right
camera local +Y: up
camera local -Z: forward
```

所以 camera 坐标下：

```python
x_cam = (u - cx) / fx * depth
y_cam = -(v - cy) / fy * depth
z_cam = -depth
```

组成：

```text
p_cam: [B, V, 64, 64, 3]
```

然后转成世界坐标：

```python
p_world = R_c2w @ p_cam + t_c2w
```

其中：

```text
R_c2w = c2w[:, :, :3, :3]
t_c2w = c2w[:, :, :3, 3]
```

---

## 4. offset

```python
offset = offset_scale * tanh(offset_raw)
```

注意 `offset_raw` 当前 shape 是：

```text
[B, V, 3, 64, 64]
```

需要 permute 成：

```text
[B, V, 64, 64, 3]
```

然后：

```python
center = p_world + offset
```

最后 center shape：

```text
[B, V, 3, 64, 64]
```

---

## 5. scale

```python
scale = 0.005 + 0.05 * softplus(scale_raw)
scale = clamp(scale, min=0.002, max=0.15)
```

shape：

```text
[B, V, 3, 64, 64]
```

---

## 6. rotation quaternion

```python
quat = normalize(quat_raw, dim=2)
```

为了避免零向量，加入 eps：

```python
quat = quat_raw / (norm(quat_raw, dim=2, keepdim=True) + 1e-8)
```

shape：

```text
[B, V, 4, 64, 64]
```

---

## 7. opacity

```python
opacity = sigmoid(opacity_raw)
```

shape：

```text
[B, V, 1, 64, 64]
```

---

## 8. rgb

```python
rgb = sigmoid(rgb_raw)
```

shape：

```text
[B, V, 3, 64, 64]
```

---

## 9. confidence

```python
confidence = sigmoid(confidence_raw)
```

shape：

```text
[B, V, 1, 64, 64]
```

confidence 暂时不进入 14 维 Gaussian 参数，只作为后续 pruning 或 top-k 的辅助量。

---

## 10. 组织 gaussian_map

按照通道顺序：

```text
[x, y, z, opacity, sx, sy, sz, qw, qx, qy, qz, r, g, b]
```

组织成：

```text
gaussian_map: [B, V, 14, 64, 64]
```

然后 flatten：

```text
gaussians = gaussian_map.permute(0, 1, 3, 4, 2)
gaussians = gaussians.reshape(B, V * 64 * 64, 14)
```

最终：

```text
gaussians: [B, 28672, 14]
```

---

# 七、forward 逻辑

`ZeroGSUNet.forward` 中：

```python
def forward(self, x, K=None, c2w=None, return_raw=True):
    B, V, C, H, W = x.shape

    assert V == self.num_views
    assert C == self.in_channels
    assert H == W == self.image_size

    x = x.reshape(B * V, C, H, W)

    # stem
    # encoder
    # bottleneck
    # decoder
    # head

    raw = raw.reshape(B, V, 16, self.splat_size, self.splat_size)

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
    )

    if return_raw:
        gaussian_dict["raw"] = raw

    return gaussian_dict
```

---

# 八、最小测试

请在 `tests/test_zerogs_unet.py` 中写最小测试，不需要 pytest 也可以，但建议 pytest。

测试 1：模型 raw 输出 shape

```python
B = 2
V = 7
x = torch.randn(B, V, 9, 256, 256)

model = ZeroGSUNet()
out = model(x)

assert out["raw"].shape == (B, V, 16, 64, 64)
```

测试 2：带 K/c2w 时 Gaussian 输出 shape

构造 dummy K：

```python
fx = fy = 477.7
cx = cy = 128.0
K = torch.tensor([
    [fx, 0, cx],
    [0, fy, cy],
    [0, 0, 1],
], dtype=torch.float32)
K = K[None, None].repeat(B, V, 1, 1)
```

构造 dummy c2w：

```python
c2w = torch.eye(4)[None, None].repeat(B, V, 1, 1)
```

调用：

```python
out = model(x, K=K, c2w=c2w)
```

检查：

```python
assert out["raw"].shape == (B, V, 16, 64, 64)
assert out["gaussian_map"].shape == (B, V, 14, 64, 64)
assert out["gaussians"].shape == (B, V * 64 * 64, 14)
assert out["confidence"].shape == (B, V, 1, 64, 64)
```

测试 3：数值范围

检查：

```python
gaussians = out["gaussians"]

opacity = gaussians[..., 3]
scale = gaussians[..., 4:7]
quat = gaussians[..., 7:11]
rgb = gaussians[..., 11:14]

assert opacity.min() >= 0 and opacity.max() <= 1
assert scale.min() >= 0
assert rgb.min() >= 0 and rgb.max() <= 1

quat_norm = torch.linalg.norm(quat, dim=-1)
assert torch.allclose(quat_norm.mean(), torch.tensor(1.0), atol=1e-2)
```

测试 4：小尺寸快速测试

为了避免显存过大，请让模型支持：

```python
model = ZeroGSUNet(image_size=128, splat_size=32)
x = torch.randn(B, V, 9, 128, 128)
```

输出应为：

```text
raw: [B, 7, 16, 32, 32]
gaussians: [B, 7*32*32, 14]
```

注意：当 `image_size=128, splat_size=32` 时，网络 encoder/decoder 结构仍然应该能正常工作。下采样路径为：

```text
128 → 64 → 32 → 16 → 8
decoder:
8 → 16 → 32
```

因此代码不要写死 256/64 的中间尺寸。

---

# 九、实现注意事项

1. 不要实现 renderer。
2. 不要实现 loss。
3. 不要实现训练循环。
4. 不要直接输出 mesh。
5. 不要使用 BatchNorm，统一使用 GroupNorm。
6. 激活函数使用 SiLU。
7. 注意 `.reshape` 前后内存连续性，必要时用 `.contiguous()`。
8. `MultiViewAttention` 的 global 模式可能显存较大，因此在默认配置中只在 bottleneck 16×16 使用 global attention；32×32 和 64×64 使用 view attention。
9. 如果 `nn.MultiheadAttention` 对输入太慢，可以保留实现，但在注释中说明后续可替换为 xformers 或 flash attention。
10. 所有函数写清楚 docstring 和 shape 注释。
11. 代码应能在 CPU 上完成小尺寸测试，例如 `image_size=128, splat_size=32`。
12. 模型文件中提供一个最小示例：

```python
if __name__ == "__main__":
    B, V = 1, 7
    x = torch.randn(B, V, 9, 128, 128)
    model = ZeroGSUNet(image_size=128, splat_size=32)
    out = model(x)
    print(out["raw"].shape)
```

---

# 十、最终期望

实现完成后，应该可以这样使用：

```python
import XXXXX

B, V, H, W = 2, 7, 256, 256

images = torch.randn(B, V, 3, H, W)
K = torch.randn(B, V, 3, 3)
c2w = torch.eye(4).reshape(1, 1, 4, 4).repeat(B, V, 1, 1)

ray_emb = get_embedding(K, c2w, resolution=256)
x = torch.cat([images, ray_emb], dim=2)

model = ZeroGSUNet(image_size=256, splat_size=64)

out = model(x, K=K, c2w=c2w)

print(out["raw"].shape)
# [2, 7, 16, 64, 64]

print(out["gaussians"].shape)
# [2, 28672, 14]
```

本任务只要求完成模型搭建和 3DGS 参数输出，不要求渲染预测结果。

如果有任何库不能使用，不要“保留接口”，而是直接使用。并在最后告诉我有哪些库缺失，需要下载。