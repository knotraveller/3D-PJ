#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH=code
python -m training.train --config ./configs/zerogs_default.yaml "$@"
