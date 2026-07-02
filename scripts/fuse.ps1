$ErrorActionPreference = "Stop"
$python = (Get-Command python -ErrorAction Stop).Source
. "$PSScriptRoot\use_vsdevcmd.ps1"
$env:PYTHONPATH = "code"
$env:TORCH_CUDA_ARCH_LIST = "8.9"
# & $python fuse.py .\image.png --123_model .\latest.pt --gs_model .\outputs\zerogs_train\checkpoints\latest.pt
& $python fuse2.py --gs_model .\outputs\zerogs_train\checkpoints\latest.pt