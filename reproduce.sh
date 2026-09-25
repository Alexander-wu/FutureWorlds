#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
export USE_TF=0
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
if [[ -n "${PYTHON:-}" ]]; then
  exec "$PYTHON" -m futureworlds reproduce "$@"
elif [[ -x .venv/bin/python ]]; then
  exec .venv/bin/python -m futureworlds reproduce "$@"
else
  exec python3 -m futureworlds reproduce "$@"
fi
