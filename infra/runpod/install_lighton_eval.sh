#!/usr/bin/env bash
# EVALUATION-ONLY install of LightOnOCR-2-1B beside the running stack (own venv; never touches the Qwen venv).
# The model is registered as a candidate, not a champion: it may be benchmarked, not served to users.
# Usage: bash infra/runpod/install_lighton_eval.sh   (run from a pod that has /workspace/cdi/.venv)
set -euo pipefail
cd "$(dirname "$0")/../.."
VENV="${LIGHTON_VENV:-/workspace/lighton-venv}"
MODEL="lightonai/LightOnOCR-2-1B"
export HF_HOME="${HF_HOME:-/workspace/hf}"
[ -d "$VENV" ] || python3 -m venv "$VENV"
"$VENV/bin/pip" install -q -U pip
"$VENV/bin/pip" install -q "transformers>=5.0.0" accelerate pillow numpy huggingface_hub hf_transfer torch

# the exact revision and the per-file checksums come from the registry, never from "main"
REV=$(.venv/bin/python -c "from cdi_adapter.compliance import models as m; print(m.pinned_revision('$MODEL'))")
echo "pinned revision: $REV"
HF_HUB_ENABLE_HF_TRANSFER=1 "$VENV/bin/python" -c "
from huggingface_hub import snapshot_download
print('downloaded', snapshot_download('$MODEL', revision='$REV'))"

.venv/bin/python -c "
from cdi_adapter.compliance import models as m
snap = m.snapshot_dir('$MODEL', '$REV')
print('files checked against the registry:', m.verify_files('$MODEL', snap))"
echo "LightOnOCR-2-1B installed and checksum-verified at revision $REV (evaluation only)"
