$ErrorActionPreference = "Stop"
$python = (Get-Command python -ErrorAction Stop).Source
. "$PSScriptRoot\use_vsdevcmd.ps1"
$env:PYTHONPATH = "code"
$env:TORCH_CUDA_ARCH_LIST = "8.9"
# & $python -m training.train --config .\configs\zerogs_train.yaml @args
& $python -m training.train @args