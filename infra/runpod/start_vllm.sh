#!/usr/bin/env bash
# Opt-in: start a `vllm serve` process for the model gateway's `vllm` backend.
#
#   bash /workspace/cdi/infra/runpod/start_vllm.sh
#
# Runs in ITS OWN venv (/workspace/vllm-venv) so vLLM's torch/deps never touch
# the app venv. The mlserve `hf` backend must be stopped first - only one process
# can hold the model. Rollback = stop this, set CDI_MLSERVE_BACKEND=hf, restart
# mlserve.
set -uo pipefail

VENV=/workspace/vllm-venv
LOG=/workspace/logs/vllm.log
PORT="${CDI_VLLM_PORT:-8078}"
MODEL="${CDI_VLM_MODEL_ID:-Qwen/Qwen2.5-VL-7B-Instruct}"
QUANT="${CDI_VLLM_QUANT:-}"                   # "" (bf16) | fp8
UTIL="${CDI_VLLM_GPU_UTIL:-0.80}"             # share of the GPU memory vLLM may take; the rest is for the CUDA context and other processes
MAXSEQS="${CDI_VLLM_MAX_SEQS:-16}"
export HF_HOME=/workspace/hf-cache PYTHONUNBUFFERED=1 VLLM_LOGGING_LEVEL=INFO
# Blackwell (sm_120): vLLM 0.28's FlashInfer sampler misfires a stale CUDA-version
# check and aborts engine init - use the native sampler + FlashAttention.
CC_MAJOR="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | cut -d. -f1)"
if [ "${CC_MAJOR:-0}" -ge 12 ]; then
  export VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_ATTENTION_BACKEND=FLASH_ATTN
else
  # other GPUs (an L4 is 8.9): vLLM chooses its own backend. Forcing FLASH_ATTN breaks Qwen2.5-VL's vision layers there
  # ("flash attention build does not support headdim not being a multiple of 32").
  unset VLLM_ATTENTION_BACKEND
fi
export PATH="$VENV/bin:$PATH"          # ninja / nvcc live in the venv; without this the engine cannot start
mkdir -p /workspace/logs

echo "########## 0. free the GPU (stop the hf gateway) ##########"
pkill -f cdi_adapter.mlserve 2>/dev/null || true
sleep 3
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader || true

echo "########## 1. venv + vLLM ##########"
# The newest vLLM wheels carry a CUDA 13 torch, which needs a newer NVIDIA driver than many pods have (a 570 driver
# reports CUDA 12.8: "driver too old"). So: driver CUDA >= 13 -> the latest vLLM; older -> 0.11.0 (torch 2.8.0, CUDA 12.8).
# Override with CDI_VLLM_PIN="vllm==x.y.z".
DRV_CUDA="$(nvidia-smi 2>/dev/null | grep -o 'CUDA Version: [0-9]*' | grep -o '[0-9]*$' | head -1)"
if [ -n "${CDI_VLLM_PIN:-}" ]; then PIN="$CDI_VLLM_PIN"
elif [ -n "$DRV_CUDA" ] && [ "$DRV_CUDA" -lt 13 ]; then PIN="vllm==0.11.0"
else PIN="vllm"; fi
echo "driver CUDA: ${DRV_CUDA:-unknown}  ->  installing: $PIN"
if [ -x "$VENV/bin/vllm" ] && ! "$VENV/bin/python" -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
  echo "the vLLM venv's torch cannot use this GPU (driver too old for it): rebuilding the venv"
  rm -rf "$VENV"
fi
if [ ! -x "$VENV/bin/vllm" ]; then
  export TMPDIR=/workspace/tmp PIP_CACHE_DIR=/workspace/tmp/pipcache
  mkdir -p "$TMPDIR"
  python3 -m venv "$VENV"          # NOT --system-site-packages: vLLM brings its own torch
  "$VENV/bin/pip" install -U pip wheel
  "$VENV/bin/pip" install "$PIN"   # pulls its own pinned torch + CUDA libs
  # an older vLLM does not know transformers 5's config format ("rope_type=default conflicts with type=mrope"): keep 4.x
  case "$PIN" in vllm==0.1[0-9].*) "$VENV/bin/pip" install "transformers>=4.56,<5" ;; esac
fi
"$VENV/bin/vllm" --version || { echo "vLLM install failed"; exit 1; }

echo "########## 2. serve $MODEL on :$PORT ##########"
pkill -f "vllm serve" 2>/dev/null || true
sleep 2
setsid nohup "$VENV/bin/vllm" serve "$MODEL" \
  --host 127.0.0.1 --port "$PORT" \
  --served-model-name "$MODEL" \
  --dtype bfloat16 ${QUANT:+--quantization "$QUANT"} \
  --gpu-memory-utilization "$UTIL" \
  --max-model-len 16384 \
  --max-num-seqs "$MAXSEQS" \
  --limit-mm-per-prompt '{"image": 2}' \
  --mm-processor-kwargs '{"max_pixels": 2000000, "min_pixels": 3136}' \
  --enable-prefix-caching \
  > "$LOG" 2>&1 < /dev/null &   # xgrammar is the default structured-output backend
echo "vllm pid $!  (log: $LOG)"

echo "########## 3. wait for readiness (model load is slow) ##########"
for _ in $(seq 1 120); do
  curl -sf "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1 && { echo "vLLM up"; break; }
  sleep 5
done
curl -s "http://127.0.0.1:${PORT}/v1/models" || { echo; echo "NOT READY - tail $LOG:"; tail -n 40 "$LOG"; exit 1; }
echo
echo "next:  set CDI_MLSERVE_BACKEND=vllm in /workspace/cdi/.env  &&  restart mlserve + webapp"
