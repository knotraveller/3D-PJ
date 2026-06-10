你现在接手一个 3DGS 前馈重建项目的训练与可微渲染部分。当前项目已经具备：

1. Blender 渲染得到的 Objaverse 数据；
2. `camera_utils.py`：负责计算/保存/读取 `K/c2w/w2c`；
3. `ray_utils.py`：负责训练时调用 `get_embedding(K, c2w)` 得到 Plücker ray embedding；
4. `models/zerogs_unet.py`：包含 `ZeroGSUNet`，输入 `[B, 7, 9, H, W]`，输出 3D Gaussian 参数。

现在请完成以下内容：

```text
1. 可微 3DGS 渲染器封装
2. Dataset / DataLoader
3. loss 计算
4. 训练脚本
5. 验证脚本
6. 训练过程可视化
7. 结果可视化
8. 配置文件和 README
```

请注意：不要重新实现模型主体结构。只需要调用已有的 `ZeroGSUNet`。如果现有模型接口不完全一致，可以做最小适配，但不要大幅改动模型结构。

---

# 一、项目背景与数据格式

以下给出详细要求，可能有少数细节处和现有接口有一点点差距。

Blender 渲染参数固定为：

```text
--resolution 256
--ref_azimuths 0
--fov 30
--camera_radius 4.0
--target_radius 0.8
```

每个样本包含 7 个视角：

```text
view 0: cond view
view 1-6: Zero123++ target views
```

数据目录结构大致如下：

```text
renders_256/
  XXX/
    ref_000/
      meta.json
      cameras.json
      cond/
        rgb.png
        alpha.png
      targets/
        000_rgb.png
        000_alpha.png
        001_rgb.png
        001_alpha.png
        002_rgb.png
        002_alpha.png
        003_rgb.png
        003_alpha.png
        004_rgb.png
        004_alpha.png
        005_rgb.png
        005_alpha.png
```

每个样本读取后应得到：

```text
images: [7, 3, H, W]      # RGB 白底图，float32, range [0,1]
alphas: [7, 1, H, W]      # 前景 mask / alpha，float32, range [0,1]
K:      [7, 3, 3]         # camera intrinsics
c2w:    [7, 4, 4]         # camera-to-world
w2c:    [7, 4, 4]         # world-to-camera
```

训练时 batch 后：

```text
images: [B, 7, 3, H, W]
alphas: [B, 7, 1, H, W]
K:      [B, 7, 3, 3]
c2w:    [B, 7, 4, 4]
w2c:    [B, 7, 4, 4]
```

然后调用：

```python
from ray_utils import get_embedding

ray_emb = get_embedding(
    K=K,
    c2w=c2w,
    resolution=H,
    embedding_type="plucker",
    order="dm",
    channel_first=True,
)
# ray_emb: [B, 7, 6, H, W]

model_input = torch.cat([images, ray_emb], dim=2)
# model_input: [B, 7, 9, H, W]
```

模型输出：

```python
out = model(model_input, K=K, c2w=c2w)
gaussians = out["gaussians"]
# gaussians: [B, N, 14]
```

其中每个 Gaussian 14 维为：

```text
[x, y, z, opacity, sx, sy, sz, qw, qx, qy, qz, r, g, b]
```

注意 quaternion 顺序必须保持为 `wxyz`。

---

# 二、推荐依赖

请优先使用 `gsplat` 作为可微 Gaussian rasterizer。

请在 `requirements.txt` 或 `README_train.md` 中说明：

```text
torch
torchvision
tqdm
pyyaml
pillow
numpy
matplotlib
tensorboard
gsplat
lpips
```


不要强依赖 wandb。可以提供可选 wandb 支持，但默认使用 TensorBoard 和本地图片保存。

如有需要但本地环境还没有的库，指出并指导我安装。而不要为了避开该库而保留不需要的接口

---

# 三、需要新增文件

大致文件接口期望：

```text
datasets/
  __init__.py
  objaverse_rendered_dataset.py

renderers/
  __init__.py
  gsplat_renderer.py

training/
  __init__.py
  losses.py
  trainer.py
  train.py
  validate.py

utils/
  __init__.py
  image_utils.py
  checkpoint.py
  visualization.py
  metrics.py
  config.py

configs/
  zerogs_default.yaml

scripts/
  run_train.ps1
  run_train.sh
  run_validate.ps1
  run_validate.sh

README_train.md
```

如果项目已有类似目录，可合并或兼容。也可做出较小的文件移动或目录修改。但接口需要保持清晰。

---

# 四、实现可微 3DGS 渲染器

请在 `renderers/gsplat_renderer.py` 中实现：

```python
class GSplatRenderer(nn.Module):
    def __init__(
        self,
        image_size: int = 256,
        background: str = "white",
        near_plane: float = 0.01,
        far_plane: float = 100.0,
        radius_clip: float = 0.0,
        eps2d: float = 0.3,
        packed: bool = True,
        tile_size: int = 16,
        render_mode: str = "RGB",
        rasterize_mode: str = "classic",
    ):
        ...
```

核心接口：

```python
def forward(
    self,
    gaussians: torch.Tensor,
    K: torch.Tensor,
    w2c: torch.Tensor,
) -> dict:
    ...
```

输入：

```text
gaussians: [B, N, 14]
K:         [B, V, 3, 3]
w2c:       [B, V, 4, 4]
```

输出：

```python
{
    "rgb": pred_rgb,       # [B, V, 3, H, W], range [0,1]
    "alpha": pred_alpha,   # [B, V, 1, H, W], range [0,1]
    "meta": meta,
}
```

## 1. Gaussian 参数拆分

从 `gaussians` 中拆分：

```python
means = gaussians[..., 0:3]        # [B, N, 3]
opacities = gaussians[..., 3]      # [B, N]
scales = gaussians[..., 4:7]       # [B, N, 3]
quats = gaussians[..., 7:11]       # [B, N, 4], wxyz
colors = gaussians[..., 11:14]     # [B, N, 3]
```

请做必要的数值保护：

```python
colors = colors.clamp(0.0, 1.0)
opacities = opacities.clamp(0.0, 1.0)
scales = scales.clamp_min(1e-4)
quats = normalize(quats, dim=-1)
```

注意：模型端应该已经做过 sigmoid/softplus/normalize，但渲染前仍然做一次轻量保护，避免训练初期数值异常。

## 2. 背景颜色

如果 `background="white"`：

```python
backgrounds = torch.ones(B, V, 3, device=device, dtype=dtype)
```

如果 `background="black"`：

```python
backgrounds = torch.zeros(B, V, 3, device=device, dtype=dtype)
```

默认白底，因为 Blender GT 图像是白底合成的 RGB。

## 3. 调用 gsplat

优先使用：

```python
from gsplat.rendering import rasterization
```

调用时应使用：

```python
render_colors, render_alphas, meta = rasterization(
    means=means,
    quats=quats,
    scales=scales,
    opacities=opacities,
    colors=colors,
    viewmats=w2c,
    Ks=K,
    width=image_size,
    height=image_size,
    near_plane=near_plane,
    far_plane=far_plane,
    radius_clip=radius_clip,
    eps2d=eps2d,
    packed=packed,
    tile_size=tile_size,
    backgrounds=backgrounds,
    render_mode="RGB",
    rasterize_mode=rasterize_mode,
)
```

`gsplat` 可能返回：

```text
render_colors: [B, V, H, W, 3]
render_alphas: [B, V, H, W, 1]
```

请转换为 channel-first：

```python
pred_rgb = render_colors.permute(0, 1, 4, 2, 3).contiguous()
pred_alpha = render_alphas.permute(0, 1, 4, 2, 3).contiguous()
```

如果实际 gsplat 版本不支持 batched `B` 维，请实现 fallback：循环 batch 维逐个调用 `rasterization`，最后 stack。不要循环 view 维，优先一次渲染一个样本的 7 个 camera。

## 4. 异常处理

如果未安装 `gsplat`，请在初始化或第一次 forward 时抛出清晰错误：

```text
gsplat is required for differentiable 3DGS rendering. Please install it with pip install gsplat, or follow the official build instructions.
```

不要默默返回空结果。

---

# 五、Dataset 实现

请在 `datasets/objaverse_rendered_dataset.py` 中实现：

```python
class ObjaverseRenderedDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        root_dir: str,
        image_size: int = 256,
        split_file: str | None = None,
        max_samples: int | None = None,
        require_all_views: bool = True,
    ):
        ...
```

## 1. 样本发现

递归查找满足以下结构的目录：

```text
*/ref_000/cameras.json
*/ref_000/cond/rgb.png
*/ref_000/cond/alpha.png
*/ref_000/targets/000_rgb.png
...
*/ref_000/targets/005_rgb.png
```

如果 `split_file` 存在，则只读取其中列出的 sample path。

## 2. 读取图像

使用 PIL：

```python
Image.open(path).convert("RGB")
Image.open(alpha_path).convert("L")
```

转为 torch tensor：

```text
RGB:   [3, H, W], float32, range [0,1]
Alpha: [1, H, W], float32, range [0,1]
```

如果图像尺寸不是 `image_size`，请 resize 到 `image_size`，RGB 用 bilinear，alpha 用 bilinear 或 nearest 均可。默认 bilinear。

返回 7 views 顺序：

```text
0: cond
1: target_000
2: target_001
3: target_002
4: target_003
5: target_004
6: target_005
```

## 3. 读取 cameras.json

读取每个 view 的：

```text
K
c2w
w2c
azimuth
elevation
relative_azimuth
```

返回：

```python
sample = {
    "images": images,       # [7, 3, H, W]
    "alphas": alphas,       # [7, 1, H, W]
    "K": K,                 # [7, 3, 3]
    "c2w": c2w,             # [7, 4, 4]
    "w2c": w2c,             # [7, 4, 4]
    "sample_id": sample_id,
    "sample_dir": str(path),
}
```

## 4. Dataset 测试

写一个简单 `if __name__ == "__main__"` 测试：

```python
dataset = ObjaverseRenderedDataset(root_dir="renders_test", image_size=256, max_samples=3)
item = dataset[0]
print(item["images"].shape)
print(item["alphas"].shape)
print(item["K"].shape)
```

---

# 六、Loss 实现

请在 `training/losses.py` 中实现：

```python
class ReconstructionLoss(nn.Module):
    def __init__(
        self,
        lambda_rgb: float = 1.0,
        lambda_mask: float = 0.5,
        lambda_lpips: float = 0.1,
        use_lpips: bool = True,
        lpips_net: str = "vgg",
        mask_rgb_loss: bool = True,
        eps: float = 1e-6,
    ):
        ...
```

forward：

```python
def forward(
    self,
    pred_rgb: torch.Tensor,
    pred_alpha: torch.Tensor,
    gt_rgb: torch.Tensor,
    gt_alpha: torch.Tensor,
) -> dict:
    ...
```

输入形状：

```text
pred_rgb:   [B, V, 3, H, W]
pred_alpha: [B, V, 1, H, W]
gt_rgb:     [B, V, 3, H, W]
gt_alpha:   [B, V, 1, H, W]
```

## 1. RGB L1 loss

如果 `mask_rgb_loss=True`：

```python
rgb_loss = (gt_alpha * (pred_rgb - gt_rgb).abs()).sum() / (gt_alpha.sum() * 3 + eps)
```

如果 `mask_rgb_loss=False`：

```python
rgb_loss = (pred_rgb - gt_rgb).abs().mean()
```

## 2. Mask loss

使用 L1：

```python
mask_loss = (pred_alpha - gt_alpha).abs().mean()
```

第一版不要用 BCE，避免数值不稳定。

## 3. LPIPS loss

如果 `use_lpips=True` (默认使用) 且 `lpips` 包可用：

* 将 `[B, V, 3, H, W]` reshape 成 `[B*V, 3, H, W]`；
* 输入 LPIPS 前从 `[0,1]` 转到 `[-1,1]`：

```python
x = x * 2 - 1
```

* 对白底完整图计算 LPIPS，不需要 mask；
* 如果 `lpips` 包不可用，应打印 warning 并自动将 `use_lpips=False`。

## 4. 总 loss

```python
total = (
    lambda_rgb * rgb_loss
    + lambda_mask * mask_loss
    + lambda_lpips * lpips_loss
)
```

返回 dict：

```python
{
    "loss": total,
    "rgb_loss": rgb_loss.detach(),
    "mask_loss": mask_loss.detach(),
    "lpips_loss": lpips_loss.detach(),
}
```

---

# 七、训练框架

请在 `training/trainer.py` 中实现：

```python
class Trainer:
    def __init__(self, config: dict):
        ...
    def train(self):
        ...
    def train_one_epoch(self, epoch: int):
        ...
    @torch.no_grad()
    def validate(self, epoch: int):
        ...
    def save_checkpoint(self, name: str):
        ...
    def load_checkpoint(self, path: str):
        ...
```

训练流程：

```text
batch
  ↓
images, alphas, K, c2w, w2c
  ↓
ray_emb = get_embedding(K, c2w)
  ↓
model_input = concat(images, ray_emb)
  ↓
model(model_input, K, c2w)
  ↓
gaussians
  ↓
renderer(gaussians, K, w2c)
  ↓
pred_rgb, pred_alpha
  ↓
loss(pred, gt)
  ↓
backward
  ↓
optimizer step
  ↓
logging / visualization
```

## 1. 设备

支持：

```text
cuda
cpu
```

默认优先使用 cuda。

## 2. AMP 混合精度

支持 config：

```yaml
train:
  amp: true
```

使用：

```python
torch.cuda.amp.autocast
torch.cuda.amp.GradScaler
```

如果 device 是 CPU，自动关闭 AMP。

## 3. 优化器

默认 AdamW：

```python
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=config["train"]["lr"],
    weight_decay=config["train"]["weight_decay"],
)
```

默认：

```yaml
lr: 1.0e-4
weight_decay: 0.01
```

## 4. 学习率调度器

实现一个简单版本即可：

```text
CosineAnnealingLR
```

或可以先不用 scheduler，但 config 中保留选项。

## 5. 梯度裁剪

支持：

```yaml
grad_clip: 1.0
```

如果为 null 或 0，则不裁剪。

## 6. checkpoint

每隔 `save_every` epoch 保存：

```text
outputs/exp_name/checkpoints/epoch_0001.pt
```

同时维护：

```text
latest.pt
best.pt
```

checkpoint 内容：

```python
{
    "epoch": epoch,
    "global_step": global_step,
    "model": model.state_dict(),
    "optimizer": optimizer.state_dict(),
    "scaler": scaler.state_dict() if amp else None,
    "config": config,
    "best_val_loss": best_val_loss,
}
```

## 7. resume

支持：

```bash
python -m training.train --config configs/zerogs_default.yaml --resume outputs/exp/checkpoints/latest.pt
```

---

# 八、训练脚本

请在 `training/train.py` 中实现命令行入口：

```bash
python -m training.train --config configs/zerogs_default.yaml
```

支持参数：

```text
--config
--resume
--debug
--overfit_one_batch
```

`--debug` 时：

```text
max_train_samples = 8
max_val_samples = 4
epochs = 2
log_every = 1
vis_every = 1
```

`--overfit_one_batch` 时：

* 只取一个 batch；
* 重复训练这个 batch；
* 用于检查模型和 renderer 是否能把 loss 降下来。

---

# 九、验证脚本

请在 `training/validate.py` 中实现：

```bash
python -m training.validate --config configs/zerogs_default.yaml --checkpoint outputs/exp/checkpoints/best.pt
```

验证时：

* 加载 checkpoint；
* 遍历 val dataset；
* 计算平均 loss；
* 保存若干可视化结果到：

```text
outputs/exp/val_visuals/
```

---

# 十、可视化

请在 `utils/visualization.py` 中实现。

## 1. save_training_visualization

接口：

```python
def save_training_visualization(
    pred_rgb: torch.Tensor,
    pred_alpha: torch.Tensor,
    gt_rgb: torch.Tensor,
    gt_alpha: torch.Tensor,
    save_path: str,
    max_views: int = 7,
):
    ...
```

输入：

```text
pred_rgb:   [B, V, 3, H, W]
pred_alpha: [B, V, 1, H, W]
gt_rgb:     [B, V, 3, H, W]
gt_alpha:   [B, V, 1, H, W]
```

默认只可视化 batch 中第一个样本。

保存一张 grid，布局建议：

```text
row 1: GT RGB, views 0-6
row 2: Pred RGB, views 0-6
row 3: GT alpha, views 0-6
row 4: Pred alpha, views 0-6
row 5: abs error heatmap or abs RGB error, views 0-6
```

如果实现 heatmap 麻烦，可以先保存灰度 error。

保存到：

```text
outputs/exp/visuals/step_000001.png
```

## 2. save_loss_curves

训练时将 loss 记录到：

```text
outputs/exp/logs/train_log.jsonl
```

每行：

```json
{"step": 1, "epoch": 0, "loss": 1.23, "rgb_loss": 0.4, "mask_loss": 0.1, "lpips_loss": 0.2, "lr": 0.0001}
```

实现：

```python
def save_loss_curves(log_jsonl_path: str, save_path: str):
    ...
```

用 matplotlib 画：

```text
total loss
rgb loss
mask loss
lpips loss
```

保存：

```text
outputs/exp/plots/loss_curve.png
```

## 3. save_gaussian_stats

实现：

```python
def save_gaussian_stats(gaussians: torch.Tensor, save_path: str):
    ...
```

统计并保存：

```text
opacity min/mean/max
scale min/mean/max
xyz min/mean/max
rgb min/mean/max
number of gaussians with opacity > 0.01
number of gaussians with opacity > 0.05
```

保存为 JSON。

---

# 十一、指标

请在 `utils/metrics.py` 中实现：

```python
def psnr(pred: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor | None = None, eps: float = 1e-8):
    ...
```

输入：

```text
pred, gt: [B, V, 3, H, W]
mask:     [B, V, 1, H, W] or None
```

如果 mask 不为空，只在前景区域计算 MSE。

PSNR：

```python
psnr = -10 * torch.log10(mse + eps)
```

SSIM 可以暂时不实现，留 TODO。

LPIPS 已在 loss 中实现。

---

# 十二、配置文件

请创建 `configs/zerogs_default.yaml`：

```yaml
experiment:
  name: zerogs_default
  output_dir: outputs/zerogs_default
  seed: 42

data:
  train_root: renders_train
  val_root: renders_val
  image_size: 256
  num_workers: 4
  max_train_samples: null
  max_val_samples: null

model:
  in_channels: 9
  num_views: 7
  image_size: 256
  splat_size: 64
  base_channels: 64
  depth_min: 2.5
  depth_max: 5.5
  offset_scale: 0.05

renderer:
  image_size: 256
  background: white
  near_plane: 0.01
  far_plane: 100.0
  radius_clip: 0.0
  eps2d: 0.3
  packed: true
  tile_size: 16
  rasterize_mode: classic

loss:
  lambda_rgb: 1.0
  lambda_mask: 0.5
  lambda_lpips: 0.1
  use_lpips: true
  lpips_net: vgg
  mask_rgb_loss: true

train:
  epochs: 50
  batch_size: 2
  lr: 1.0e-4
  weight_decay: 0.01
  amp: true
  grad_clip: 1.0
  log_every: 10
  vis_every: 200
  save_every: 1
  val_every: 1
```

注意：默认 `batch_size=2`，因为 3DGS renderer + 28672 Gaussians + 7 views 显存压力较大。后续用户可以改大。

---

# 十三、训练可视化输出目录

训练过程中输出：

```text
outputs/zerogs_default/
  checkpoints/
    latest.pt
    best.pt
    epoch_0001.pt
  visuals/
    step_000000.png
    step_000200.png
  plots/
    loss_curve.png
  stats/
    gaussian_stats_step_000200.json
  logs/
    train_log.jsonl
  tensorboard/
```

TensorBoard 记录：

```text
train/loss
train/rgb_loss
train/mask_loss
train/lpips_loss
train/psnr
val/loss
val/psnr
lr
```

并且每隔 `vis_every` step 把可视化图片写入 TensorBoard。

---

# 十四、训练过程数值检查

在每个 step 后，检查：

```python
if not torch.isfinite(loss):
    print diagnostics
    save gaussian stats
    raise RuntimeError
```

需要输出：

```text
loss
rgb_loss
mask_loss
lpips_loss
gaussians min/max
opacity min/max
scale min/max
pred_rgb min/max
pred_alpha min/max
```

避免训练 silently 崩掉。

---

# 十五、overfit-one-batch 检查

实现 `--overfit_one_batch` 模式。

这个模式非常重要，用于确认：

```text
model → renderer → loss → backward
```

整个链路可微且能训练。

行为：

```text
只取一个 batch
重复训练 200~1000 steps
每 20 steps 保存一次可视化
观察 loss 是否下降，pred 是否逐渐接近 GT
```

如果 overfit 一个 batch 都不能下降，说明模型、renderer、坐标系、相机矩阵、loss 或数据读取中存在问题。

---

# 十六、README_train.md

请写一个简短但完整的 README，包含：

1. 安装依赖；
2. 如何安装 gsplat；
3. 数据目录结构；
4. 如何启动 debug 训练；
5. 如何 overfit 一个 batch；
6. 如何正式训练；
7. 如何验证；
8. 如何查看 TensorBoard；
9. 输出目录说明；
10. 常见问题。

命令示例：

```bash
python -m training.train --config configs/zerogs_default.yaml --debug
```

```bash
python -m training.train --config configs/zerogs_default.yaml --overfit_one_batch
```

```bash
python -m training.train --config configs/zerogs_default.yaml
```

```bash
tensorboard --logdir outputs/zerogs_default/tensorboard
```

Windows PowerShell 示例：

```powershell
python -m training.train --config .\configs\zerogs_default.yaml --debug
```

---

# 十七、实现注意事项

1. 不要重写 Blender 渲染脚本。
2. 不要重写 `camera_utils.py` 和 `ray_utils.py`，只调用其中的 `get_embedding`。
3. 不要重写 `ZeroGSUNet` 主体结构。
4. 不要实现 mesh 输出。
5. 不要实现动态增删 Gaussian。
6. 默认所有 28672 个 Gaussian 都参与渲染。
7. 训练后可以在可视化统计中记录 `opacity > 0.01` 的有效 Gaussian 数量，但不要在训练中 hard prune。
8. 渲染背景必须默认 white，和 Blender GT RGB 对齐。
9. RGB loss 默认只在 GT alpha 前景区域计算。
10. LPIPS 对白底整图计算。
11. Mask loss 使用 pred_alpha 与 gt_alpha 的 L1。
12. Camera renderer 使用 `w2c`，模型 unprojection 使用 `c2w`。
13. 确保 `K/w2c/c2w/images/alphas` 都在同一 device 和 dtype。
14. 如果使用 AMP，注意 gsplat 是否支持当前 dtype。如果遇到问题，可以在 renderer 内部临时转 float32，并在 README 中说明。
15. 所有图像保存前 clamp 到 `[0,1]`。
16. 所有 shape 都要写断言，错误信息要清楚。

---

# 十八、最小集成测试

请新增一个可以手动运行的测试脚本：

```text
tests/test_training_pipeline.py
```

它执行：

```python
dataset = ObjaverseRenderedDataset(root_dir="renders_test", image_size=128, max_samples=1)
batch = collate one sample

ray_emb = get_embedding(K, c2w, resolution=128)
x = torch.cat([images, ray_emb], dim=2)

model = ZeroGSUNet(image_size=128, splat_size=32)
renderer = GSplatRenderer(image_size=128)
criterion = ReconstructionLoss(use_lpips=False)

out = model(x, K=K, c2w=c2w)
render_out = renderer(out["gaussians"], K=K, w2c=w2c)

loss_dict = criterion(
    pred_rgb=render_out["rgb"],
    pred_alpha=render_out["alpha"],
    gt_rgb=images,
    gt_alpha=alphas,
)

loss = loss_dict["loss"]
loss.backward()

print("pipeline ok")
```

这个测试的目标不是训练出效果，而是确认：

```text
dataset → embedding → model → gsplat renderer → loss → backward
```

整条链路是通的。

---

# 十九、最终验收标准

完成后应当可以：

1. 用 `--debug` 跑通训练；
2. 用 `--overfit_one_batch` 观察 loss 下降；
3. 在 `outputs/.../visuals/` 中看到 GT vs Pred 的可视化图；
4. 在 TensorBoard 中看到 loss 曲线；
5. 在 `checkpoints/` 中看到 latest/best checkpoint；
6. 在验证脚本中加载 checkpoint 并输出验证 loss；
7. 训练阶段没有 shape mismatch；
8. 3DGS renderer 返回 RGB 和 alpha；
9. backward 可以通过 renderer 回传到 `ZeroGSUNet` 参数。
