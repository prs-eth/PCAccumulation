#!/usr/bin/env bash
# Regenerate the ONNX models for the C++ project (pcacc_cpp/models/) and verify them in onnxruntime.
# Extra arguments are passed to export_onnx.py, e.g.:  bash export/export_models.sh --frame 100
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
source venv_pcaccumulation/bin/activate
python -W ignore export/export_onnx.py --out-dir pcacc_cpp/models "$@"
