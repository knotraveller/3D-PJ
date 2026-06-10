#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH=code
python -m training.validate --config ./configs/zerogs_default.yaml "$@"
