#!/usr/bin/env bash
# CPU OCR host (recognition v2): RapidOCR reads the printed lines. Handwriting is read by Qwen2.5-VL (vLLM), not here.
set -euo pipefail
REPO=/workspace/cdi
cd "$REPO"
. .venv/bin/activate
set -a; . .env; set +a
export HF_HOME=/workspace/hf-cache
PORT="${CDI_OCRHOST_PORT:-8079}"
mkdir -p /workspace/logs
pkill -f "cdi_adapter.ocrhost" 2>/dev/null || true
sleep 1
setsid nohup python -m cdi_adapter.ocrhost > /workspace/logs/ocrhost.log 2>&1 < /dev/null &
echo "ocrhost pid $!  port $PORT  ->  /workspace/logs/ocrhost.log"
for _ in $(seq 1 30); do
  curl -sf "http://127.0.0.1:${PORT}/healthz" && echo && exit 0
  sleep 1
done
echo "ocrhost did not answer on :${PORT}; tail log:"; tail -40 /workspace/logs/ocrhost.log; exit 1
