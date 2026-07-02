# ZeroGS 3D Gaussian Reconstruction

这是一个基于多视角渲染图像的前馈式 3D Gaussian Splatting 重建项目。当前代码链路为：

```text
Blender/Objaverse 渲染数据
  -> ObjaverseRenderedDataset
  -> RGB + Plucker ray embedding
  -> ZeroGSUNet
  -> 3D Gaussian 参数
  -> gsplat 可微渲染
  -> RGB/alpha 重建损失
  -> 训练、验证、可视化、性能统计
```

项目默认使用 1 个条件视角和 6 个目标视角，共 7 个视角。模型输入为白底 RGB 图像和相机射线嵌入，输出一组 3D Gaussians，并通过 `gsplat` 渲染回目标视角。

## 环境准备

推荐在已经激活的 `3d` conda 环境中运行：

```powershell
conda activate 3d
```

CUDA 环境依赖示例：

```powershell
conda install -n 3d -c pytorch -c nvidia -c conda-forge pytorch torchvision pytorch-cuda=12.1 numpy pillow tqdm pyyaml matplotlib tensorboard "setuptools<81" psutil pytest -y
conda run -n 3d pip install gsplat lpips
```

也可以参考 `requirements.txt`：

```powershell
pip install -r requirements.txt
```

Windows 上 `gsplat` 首次运行可能会 JIT 编译 CUDA/C++ 扩展，需要 Visual Studio Build Tools C++ 工具链。PowerShell 脚本会调用：

```text
scripts/use_vsdevcmd.ps1
```

它通过 `vcvars64.bat` 加载 MSVC 环境，避免普通 PowerShell 找不到 `cl.exe`。

## 项目结构

```text
configs/
  zerogs_default.yaml      # 基线/调试配置
  zerogs_train.yaml        # 当前主要训练配置
  zerogs_finetune.yaml     # finetune 配置
  zerogs_finetune2.yaml    # finetune 配置

scripts/
  run_train.ps1            # Windows 训练入口
  run_validate.ps1         # Windows 验证入口
  generate_all.ps1         # 批量生成可视化
  use_vsdevcmd.ps1         # 加载 MSVC 编译环境
  run_train.sh             # Linux/macOS 训练入口
  run_validate.sh          # Linux/macOS 验证入口

code/
  datasets/                # 数据集读取
  models/                  # ZeroGSUNet 和 Gaussian 参数后处理
  renderers/               # gsplat renderer 封装
  training/                # train/validate/loss/trainer
  tools/                   # 曲线绘图、GPU 图、批量可视化、Blender 渲染工具
  utils/                   # 相机、ray embedding、checkpoint、可视化、性能统计等

tests/                     # 单元测试和 pipeline 测试
outputs/                   # 训练输出，默认不需要手动创建
README_train.md            # 更细的训练说明
README_models.md           # 模型接口说明
```

## 数据格式

默认数据根目录为：

```text
dataset/renders_256/
```

每个样本目录应类似：

```text
dataset/renders_256/
  asset_id/
    ref_000/
      cameras.json
      meta.json
      cond/
        rgb.png
        alpha.png
      targets/
        000_rgb.png
        000_alpha.png
        ...
        005_rgb.png
        005_alpha.png
```

读取后单个样本包含：

```text
images:     [7, 3, H, W]
alphas:     [7, 1, H, W]
K:          [7, 3, 3]
c2w:        [7, 4, 4]
w2c:        [7, 4, 4]
sample_id:  str
sample_dir: str
```

7 个视角顺序为：

```text
0: cond
1: target_000
2: target_001
3: target_002
4: target_003
5: target_004
6: target_005
```

## 快速开始

### Debug 训练

用于快速检查环境、数据、renderer、loss、backward 是否能跑通：

```powershell
conda activate 3d
.\scripts\run_train.ps1 --config .\configs\zerogs_train.yaml --debug
```

`--debug` 会覆盖部分配置：

```text
data.max_train_samples = 8
data.max_val_samples = 4
data.num_workers = 0
train.epochs = 2
train.batch_size = 1
train.gradient_accumulation_steps = 1
train.save_every = 1
train.val_every = 1
```

### 正式训练

当前主配置：

```powershell
conda activate 3d
.\scripts\run_train.ps1 --config .\configs\zerogs_train.yaml
```

等价 Python 入口：

```powershell
$env:PYTHONPATH="code"
python -m training.train --config .\configs\zerogs_train.yaml
```

`configs/zerogs_train.yaml` 当前要点：

```text
output_dir: outputs/zerogs_train
epochs: 1000
batch_size: 1
gradient_accumulation_steps: 2
lr: 1.0e-5
scheduler: plateau
save_every: 10
val_every: 2
```

### Overfit One Batch

用于确认一条样本上的 `model -> renderer -> loss -> backward` 链路是否可学习：

```powershell
.\scripts\run_train.ps1 --config .\configs\zerogs_train.yaml --overfit_one_batch
```

如果这个模式下 loss 也无法下降，优先检查数据、相机矩阵、renderer 坐标系、loss 和 Gaussian 参数范围。

## 训练恢复与微调

只加载模型权重并从新实验开始微调：

```powershell
.\scripts\run_train.ps1 `
  --config .\configs\zerogs_finetune.yaml `
  --checkpoint .\outputs\zerogs_train\checkpoints\latest.pt
```

从中断处完整恢复训练状态：

```powershell
.\scripts\run_train.ps1 `
  --config .\configs\zerogs_train.yaml `
  --checkpoint .\outputs\zerogs_train\checkpoints\latest.pt `
  --resume
```

区别：

```text
--checkpoint 不加 --resume:
  只加载 model 权重，optimizer/scheduler/scaler/global_step 不恢复。

--checkpoint 加 --resume:
  恢复 completed_epoch、global_step、optimizer、scheduler、AMP scaler、best_val_loss。
```

注意：`train.epochs` 表示目标总 epoch 数。比如 checkpoint 已完成 50 epoch，而配置写 `epochs: 1000`，则 resume 后从 epoch 51 继续直到 epoch 1000。

## 验证

验证某个 checkpoint：

```powershell
conda activate 3d
.\scripts\run_validate.ps1 `
  --config .\configs\zerogs_train.yaml `
  --checkpoint .\outputs\zerogs_train\checkpoints\best.pt
```

等价 Python 入口：

```powershell
$env:PYTHONPATH="code"
python -m training.validate `
  --config .\configs\zerogs_train.yaml `
  --checkpoint .\outputs\zerogs_train\checkpoints\best.pt
```

独立验证会为验证集每个样本保存五行对比图：

```text
outputs/zerogs_train/validate/all_visuals/
```

并输出：

```text
outputs/zerogs_train/validate/loss.yaml
outputs/zerogs_train/validate/logs/validate_log.jsonl
```

训练过程中的周期性验证较轻量，只保存每个验证 epoch 的首个样本：

```text
outputs/zerogs_train/validate/epoch_visuals/
```

## TensorBoard

```powershell
conda activate 3d
tensorboard --logdir .\outputs\zerogs_train\tensorboard
```

浏览器打开终端打印的地址，通常是：

```text
http://localhost:6006/
```

可查看：

```text
train/loss
train/rgb_loss
train/mask_loss
train/lpips_loss
train/psnr
val/loss
val/psnr
lr
perf/timing_ms/...
perf/system/...
perf/gpu_modules/...
```

如果 TensorBoard 报 `No module named 'pkg_resources'`：

```powershell
pip install "setuptools<81"
```

## 结果可视化

### 训练/验证可视化图

训练可视化保存到：

```text
outputs/zerogs_train/visuals/
```

默认五行含义：

```text
第 1 行: ground truth RGB
第 2 行: predicted RGB
第 3 行: ground truth alpha
第 4 行: predicted alpha
第 5 行: RGB absolute error
```

每一列对应一个视角，最多 7 列。

### 批量生成所有样本可视化

```powershell
conda activate 3d
.\scripts\generate_all.ps1
```

或手动指定 checkpoint 和数据根目录：

```powershell
$env:PYTHONPATH="code"
python -m tools.generate_all_visuals `
  --model .\outputs\zerogs_train\checkpoints\latest.pt `
  --image .\dataset\renders_256
```

输出到：

```text
outputs/all_visuals/<timestamp>/
```

### 绘制训练曲线

```powershell
$env:PYTHONPATH="code"
python -m tools.plot_loss_curves `
  --json .\outputs\zerogs_train\logs\train_log.jsonl `
  --name zerogs_train_epoch_metrics
```

输出到：

```text
outputs/plots/zerogs_train_epoch_metrics/
```

### 绘制 GPU/性能图

```powershell
$env:PYTHONPATH="code"
python -m tools.plot_GPU `
  --json .\outputs\zerogs_train\stats\performance_latest.json `
  --name zerogs_train_gpu
```

输出到：

```text
outputs/plots/zerogs_train_gpu/
```

## 性能统计

配置位于 YAML 的 `performance` 段：

```yaml
performance:
  enabled: true
  ema_momentum: 0.8
  sync_cuda: true
  write_every: 100
  system_sample_every: 100
  sample_system: false
  sample_gpu_utilization: false
  profile_gpu_modules: false
```

统计项包括：

```text
data_load
to_device
ray_embedding
model_forward
render
loss
metrics
backward
grad_clip
optimizer_step
validation
```

每个指标记录：

```text
latest
ema
min
max
mean
count
```

EMA 公式：

```text
ema = ema_momentum * previous_ema + (1 - ema_momentum) * current
```

性能文件：

```text
outputs/zerogs_train/stats/performance_latest.json
outputs/zerogs_train/stats/performance_log.jsonl
```

`sync_cuda: true` 的 GPU 阶段计时更准确，但会降低训练速度；如果只想低开销观察趋势，可以设为 `false`。

## 主要接口说明

### `ObjaverseRenderedDataset`

位置：

```text
code/datasets/objaverse_rendered_dataset.py
```

接口：

```python
ObjaverseRenderedDataset(
    root_dir: str,
    image_size: int = 256,
    split_file: str | None = None,
    max_samples: int | None = None,
    require_all_views: bool = True,
)
```

返回字段：

```text
images, alphas, K, c2w, w2c, sample_id, sample_dir
```

### `ZeroGSUNet`

位置：

```text
code/models/zerogs_unet.py
```

输入：

```text
x:   [B, 7, 9, H, W]
K:   [B, 7, 3, 3]
c2w: [B, 7, 4, 4]
```

输出：

```text
raw:          [B, 7, 16, H/4, W/4]
gaussian_map: [B, 7, 14, H/4, W/4]
gaussians:    [B, 7*(H/4)*(W/4), 14]
```

Gaussian 14 维定义：

```text
[x, y, z, opacity, sx, sy, sz, qw, qx, qy, qz, r, g, b]
```

### `GSplatRenderer`

位置：

```text
code/renderers/gsplat_renderer.py
```

接口：

```python
renderer(
    gaussians,  # [B, N, 14]
    K,          # [B, V, 3, 3]
    w2c,        # [B, V, 4, 4]
)
```

输出：

```text
rgb:   [B, V, 3, H, W]
alpha: [B, V, 1, H, W]
meta:  gsplat metadata
```

实现细节：

```text
数据和 Blender 相机一致: +X right, +Y up, -Z forward
gsplat 期望 OpenCV 相机: +X right, +Y down, +Z forward
renderer 内部会转换 view matrix。
```

renderer 内部强制用 `float32` 调用 `gsplat`，避免 AMP 下出现 `expected scalar type Float but found Half`。

### `ReconstructionLoss`

位置：

```text
code/training/losses.py
```

组成：

```text
RGB L1 loss
alpha/mask L1 loss
可选 LPIPS loss
```

配置：

```yaml
loss:
  lambda_rgb: 1.0
  lambda_mask: 0.5
  lambda_lpips: 0.1
  use_lpips: true
  lpips_net: vgg
  lpips_chunk_size: 1
  mask_rgb_loss: true
```

### `Trainer`

位置：

```text
code/training/trainer.py
```

职责：

```text
构建 dataset/dataloader
构建 model/renderer/loss
训练循环
验证循环
checkpoint 保存和恢复
TensorBoard 写入
epoch 级日志
可视化和 Gaussian stats
性能统计
```

### 命令行入口

训练：

```text
code/training/train.py
```

参数：

```text
--config
--checkpoint
--resume
--debug
--overfit_one_batch
```

验证：

```text
code/training/validate.py
```

参数：

```text
--config
--checkpoint
```

## 输出目录

典型输出：

```text
outputs/zerogs_train/
  checkpoints/
    latest.pt
    best.pt
    epoch_0010.pt
  visuals/
    epoch_0001.png
  validate/
    all_visuals/
    epoch_visuals/
    logs/
      validate_log.jsonl
    loss.yaml
  plots/
    loss_curve.png
  stats/
    gaussian_stats_epoch_0001.json
    performance_latest.json
    performance_log.jsonl
  logs/
    train_log.jsonl
    train_events.jsonl
  tensorboard/
```

`train_log.jsonl` 是 epoch 级日志，每行记录：

```text
epoch
num_batches
lr
loss/rgb_loss/mask_loss/lpips_loss/psnr 的 mean/max/min
```

`train_events.jsonl` 记录 resume 等训练事件。

## 生成数据集

如需从 `.glb` 生成渲染数据，可使用 Blender 后台运行：

```powershell
blender -b --python .\code\tools\render_objaverse.py -- `
  --input_dir .\PATH\TO\GLBS `
  --output_dir .\dataset\renders_256 `
  --resolution 256 `
  --ref_azimuths 0 90 180 270 `
  --fov 30 `
  --camera_radius 4.0 `
  --target_radius 0.8 `
  --skip_existing
```

常用参数：

```text
--input_dir       GLB 文件目录
--output_dir      输出渲染目录
--resolution      图像分辨率，默认 256
--ref_azimuths    条件视角方位角列表
--input_elevation 条件视角 elevation
--fov             相机 FOV
--camera_radius   相机半径
--target_radius   物体归一化半径
--engine          Blender 渲染引擎
--skip_existing   跳过完整样本
--max_files       只渲染前 N 个文件
```

## 测试

基础单元测试：

```powershell
conda activate 3d
$env:PYTHONPATH="code"
python -m pytest tests/test_performance.py tests/test_zerogs_unet.py code/utils/test_camera_rays.py
```

最小训练链路反向传播测试：

```powershell
conda activate 3d
$env:PYTHONPATH="code"
python tests/test_training_pipeline.py --root dataset/renders_256 --image_size 64
```

这个测试会检查：

```text
dataset -> ray embedding -> model -> gsplat renderer -> loss -> backward
```

## 常见问题

### `where.exe cl` 找不到

普通 PowerShell 默认不加载 MSVC 环境。推荐使用：

```powershell
.\scripts\run_train.ps1 --config .\configs\zerogs_train.yaml --debug
```

脚本会调用 `scripts/use_vsdevcmd.ps1` 自动加载 `vcvars64.bat`。

### `No module named training`

需要从项目根目录运行，并设置：

```powershell
$env:PYTHONPATH="code"
```

使用 `scripts/run_train.ps1` 时会自动设置。

### TensorBoard 缺少 `pkg_resources`

```powershell
pip install "setuptools<81"
```

### CUDA OOM

可尝试：

```text
降低 batch_size
提高 gradient_accumulation_steps 保持有效 batch
关闭 LPIPS: loss.use_lpips=false
保持 lpips_chunk_size=1
降低 model.base_channels
关闭 attention
使用 --debug 先检查链路
```

### 可视化中 predicted RGB/alpha 为空

优先检查：

```text
是否使用了修复后的 GSplatRenderer
是否从旧错误 renderer 训练出的 checkpoint resume
相机坐标系是否为 Blender -> OpenCV 转换路径
Gaussian stats 中 opacity/scale/xyz 是否异常
```

旧 renderer 下训练出的 checkpoint 不建议继续 resume，应重新训练或换新的输出目录。

### 看到 cuDNN / flash attention / torchvision warning

这些通常是 warning，不一定会中断训练。真正需要优先处理的是 traceback 最底部的 `RuntimeError`、`ValueError` 或 `CUDA out of memory`。

## 进一步文档

更细的训练说明见：

```text
README_train.md
```

模型接口简述见：

```text
README_models.md
```
