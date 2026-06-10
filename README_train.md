# ZeroGS Training

This training stack connects the existing rendered Objaverse data, ray embedding, `ZeroGSUNet`, and a differentiable 3DGS renderer.

## Install

Recommended conda install for the `3d` environment with CUDA:

```powershell
conda install -n 3d -c pytorch -c nvidia -c conda-forge pytorch torchvision pytorch-cuda=12.1 numpy pillow tqdm pyyaml matplotlib tensorboard "setuptools<81" psutil pytest -y
conda run -n 3d pip install gsplat lpips
```

CPU-only install:

```powershell
conda install -n 3d -c pytorch -c conda-forge pytorch torchvision cpuonly numpy pillow tqdm pyyaml matplotlib tensorboard "setuptools<81" psutil pytest -y
conda run -n 3d pip install gsplat lpips
```

`gsplat` is required for differentiable rendering. If `pip install gsplat` fails, follow the official gsplat build instructions for your CUDA/PyTorch version.

On Windows, `gsplat` may compile CUDA/C++ extensions on first use. Make sure Visual Studio Build Tools C++ workload is installed and `where.exe cl` can find `cl.exe`. If it cannot, open a Developer PowerShell or install Build Tools:

```powershell
winget install Microsoft.VisualStudio.2022.BuildTools --override "--wait --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended"
```

The PowerShell scripts assume the `3d` conda environment is already active and
use its current `python`. They load the MSVC environment through `vcvars64.bat`
instead of `Launch-VsDevShell.ps1`; this avoids localized `vswhere` JSON parsing
errors on Windows.

## Data Layout

The dataset expects rendered samples like:

```text
renders_256/
  asset_id/
    ref_000/
      cameras.json
      meta.json
      cond/rgb.png
      cond/alpha.png
      targets/000_rgb.png
      targets/000_alpha.png
      ...
      targets/005_rgb.png
      targets/005_alpha.png
```

Each sample loads as:

```text
images: [7, 3, H, W]
alphas: [7, 1, H, W]
K:      [7, 3, 3]
c2w:    [7, 4, 4]
w2c:    [7, 4, 4]
```

The default config points both train and validation roots at `dataset/renders_256` so the pipeline can be tested immediately. For real training, edit `configs/zerogs_default.yaml` to use separate train/val folders or split files.

## Debug Training

Debug mode is a quick smoke test for the training stack. It keeps the model,
image size, renderer, and loss settings from the config, but uses tiny dataset
and loop settings so setup problems surface quickly before a full run. Passing
`--debug` applies these overrides:

```text
data.max_train_samples = 8
data.max_val_samples = 4
data.num_workers = 0
train.epochs = 2
train.batch_size = 1
train.log_every = 1
train.vis_every = 1
train.save_every = 1
train.val_every = 1
```

Use it after installing dependencies, changing dataset paths, or touching the
renderer/training code. It is not a quality benchmark and it is different from
`--overfit_one_batch`: debug mode still trains and validates across several
samples, while overfit mode repeats one batch to check that gradients and losses
can move in the right direction.

PowerShell:

```powershell
$env:PYTHONPATH="code"
conda run -n 3d python -m training.train --config .\configs\zerogs_default.yaml --debug
```

or:

```powershell
conda activate 3d
.\scripts\run_train.ps1 --debug
```

Expected terminal feedback:

```text
train epoch 0: ... loss=...
val epoch 0: ...
train epoch 1: ... loss=...
val epoch 1: ...
```

During the first renderer call, `gsplat` may print CUDA extension compilation messages. This is normal if it finishes successfully.

Files created in debug mode:

```text
outputs/zerogs_default/
  checkpoints/latest.pt
  checkpoints/best.pt
  checkpoints/epoch_0001.pt
  visuals/step_000000.png
  val_visuals/epoch_0000.png
  plots/loss_curve.png
  stats/gaussian_stats_step_000000.json
  logs/train_log.jsonl
  tensorboard/
```

Linux/macOS:

```bash
PYTHONPATH=code conda run -n 3d python -m training.train --config ./configs/zerogs_default.yaml --debug
```

## Overfit One Batch

This mode repeatedly trains on one batch and is the first thing to try after installing `gsplat`:

```powershell
$env:PYTHONPATH="code"
conda run -n 3d python -m training.train --config .\configs\zerogs_default.yaml --overfit_one_batch
```

Expected terminal feedback:

```text
overfit one batch:  10%|...| loss=...
overfit one batch: 100%|...|
```

Files created are the same as debug training, but visualizations are saved more frequently according to `train.vis_every`. Watch `outputs/zerogs_default/visuals/` and `logs/train_log.jsonl` to confirm the loss trends downward.

If this cannot reduce loss, inspect data scale, camera matrices, renderer convention, and Gaussian statistics before starting full training.

## Full Training

```powershell
$env:PYTHONPATH="code"
conda run -n 3d python -m training.train --config .\configs\zerogs_default.yaml
```

Expected terminal feedback:

```text
train epoch 0: ... loss=...
val epoch 0: ...
train epoch 1: ... loss=...
val epoch 1: ...
```

Every `save_every` epochs, checkpoint files are written. Every `vis_every` steps, visualization PNGs, Gaussian stats JSON, and loss curves are updated.

Resume:

```powershell
$env:PYTHONPATH="code"
conda run -n 3d python -m training.train --config .\configs\zerogs_default.yaml --resume .\outputs\zerogs_default\checkpoints\latest.pt
```

## Validation

```powershell
$env:PYTHONPATH="code"
conda run -n 3d python -m training.validate --config .\configs\zerogs_default.yaml --checkpoint .\outputs\zerogs_default\checkpoints\best.pt
```

or:

```powershell
conda activate 3d
.\scripts\run_validate.ps1 --checkpoint .\outputs\zerogs_default\checkpoints\best.pt
```

Expected terminal feedback:

```text
val epoch <step_or_epoch>: ...
validation loss: 0.123456
```

Validation writes representative GT-vs-prediction grids to:

```text
outputs/zerogs_default/val_visuals/
```

## Performance Stats

The `performance` config section enables lightweight timing and resource
sampling during training:

```yaml
performance:
  enabled: true
  ema_momentum: 0.8
  sync_cuda: true
  write_every: 10
  system_sample_every: 10
  sample_system: true
  sample_gpu_utilization: true
```

Stage timings are recorded for data loading, device transfer, ray embedding,
model forward, rendering, loss, metrics, backward, gradient clipping, optimizer
step, logging, visualization, and validation. Each metric stores `latest`,
`ema`, `min`, `max`, `mean`, and `count`. The EMA formula is:

```text
ema = ema_momentum * previous_ema + (1 - ema_momentum) * current
```

With `ema_momentum: 0.8`, the curve is relatively smooth. Use a lower value such
as `0.5` or `0.2` if you want the current step to affect the EMA more strongly.

Performance snapshots are written under:

```text
outputs/zerogs_default/stats/performance_latest.json
outputs/zerogs_default/stats/performance_log.jsonl
```

TensorBoard also receives `perf/timing_ms/...` and `perf/system/...` scalars.
CUDA memory is sampled through PyTorch. GPU utilization and device memory are
sampled through `nvidia-smi` when it is available. CPU utilization and process
RSS are sampled when `psutil` is installed; training still works if it is not.

`sync_cuda: true` gives more accurate stage timings for GPU work because CUDA
kernels are asynchronous, but it can slow training. Set it to `false` when you
only want low-overhead CPU wall-clock measurements.

## TensorBoard

```powershell
conda run -n 3d tensorboard --logdir .\outputs\zerogs_default\tensorboard
```

If the environment is already activated:

```powershell
conda activate 3d
tensorboard --logdir .\outputs\zerogs_default\tensorboard
```

Expected terminal feedback:

```text
TensorBoard ... at http://localhost:6006/
```

Then open the printed URL in a browser. You should see scalar curves for `train/loss`, `train/rgb_loss`, `train/mask_loss`, `train/lpips_loss`, `train/psnr`, `val/loss`, `val/psnr`, and `lr`.

TensorBoard watches the event files while it is running, so you can keep this
terminal open and run training in another PowerShell window. The browser page at
`http://localhost:6006/` updates as new scalars and images are written, usually
with a short delay.

If TensorBoard fails with `No module named 'pkg_resources'`, install a setuptools
version that still ships it:

```powershell
conda run -n 3d python -m pip install "setuptools<81"
```

## Outputs

```text
outputs/zerogs_default/
  checkpoints/
    latest.pt
    best.pt
    epoch_0001.pt
  visuals/
    step_000000.png
  val_visuals/
    epoch_0000.png
  plots/
    loss_curve.png
  stats/
    gaussian_stats_step_000000.json
    performance_latest.json
    performance_log.jsonl
  logs/
    train_log.jsonl
  tensorboard/
```

## Notes

- Renderer camera matrices use `w2c`; model unprojection uses `c2w`.
- Blender camera local `-Z` is forward.
- Ground-truth RGB is white-background composited, so renderer background defaults to white.
- RGB L1 loss is masked by GT alpha by default.
- LPIPS is computed on white-background full images.
- `GSplatRenderer` casts renderer inputs to float32 internally; this avoids many AMP/rasterizer dtype issues while keeping gradients connected.
- No hard pruning is done during training; all `7*64*64` Gaussians are rendered.

## Common Problems

- `ImportError: gsplat is required`: install `gsplat` with pip or build it for your CUDA/PyTorch version.
- `No module named training`: run from the repository root and set `PYTHONPATH=code`, or use the scripts in `scripts/`.
- `No module named 'pkg_resources'` when starting TensorBoard: run `conda run -n 3d python -m pip install "setuptools<81"`.
- `ConvertFrom-Json` inside `Launch-VsDevShell.ps1`: use the scripts in `scripts/`, which load `vcvars64.bat` directly.
- CUDA OOM: reduce `batch_size`, use `--debug`, lower `base_channels`, reduce `loss.lpips_chunk_size`, disable LPIPS, or disable attention in the model config.
- CUDA OOM inside LPIPS/VGG: keep `loss.lpips_chunk_size: 1` so LPIPS processes one rendered view at a time.
- Loss is non-finite: inspect the JSON written under `outputs/.../stats/`, which records Gaussian ranges and opacity/scale statistics.
