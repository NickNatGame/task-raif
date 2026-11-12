#!/usr/bin/env bash
set -euo pipefail

task_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONDONTWRITEBYTECODE=1

# В текущей WSL-установке временные файлы и PyTorch вынесены на D:.
if [[ -d /mnt/d/raif-gru/tmp ]]; then
    export TMPDIR=/mnt/d/raif-gru/tmp
fi
if [[ -d /mnt/d/raif-gru/site-packages ]]; then
    export PYTHONPATH="/mnt/d/raif-gru/site-packages${PYTHONPATH:+:$PYTHONPATH}"
fi
exec "$task_dir/../.venv/bin/python" -B "$task_dir/predict_best.py" "$@"
