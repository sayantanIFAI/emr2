#!/usr/bin/env bash
# GPU NODE: the model only. vLLM (Qwen2.5-VL-7B) and the model gateway. No database, no web app, no OCR, no stored documents.
# The APP NODE (a separate machine) reaches the two ports below through an SSH tunnel (infra/runpod/model_tunnel.sh); they are bound to this
# machine's loopback and are NEVER published: neither has any sign-in.
#
#   setsid nohup bash -c 'curl -fsSL https://raw.githubusercontent.com/sayantanIFAI/emr2/main/infra/runpod/start_gpu_node.sh | bash' > /workspace/gpu_node_start.log 2>&1 < /dev/null &
#   (or, once the repo is on the pod:  bash /workspace/cdi/infra/runpod/start_gpu_node.sh)      safe to re-run: a healthy vLLM / gateway is left alone
#   bash /workspace/cdi/infra/runpod/start_gpu_node.sh keyonly      only (re-)install the app node's tunnel key: fast, touches nothing else
#
# Steps: 1 code from git -> 2 settings (.env) -> 3 python environment (the project's base dependencies only) -> 4 the app node's tunnel key
#        -> 5 model download (about 16 GB, once per volume) -> 6 vLLM -> 7 model gateway -> 8 summary (what the app node needs)
# Optional overrides (for tuning, e.g. CDI_VLLM_MAX_SEQS=32): put KEY=VALUE lines in /workspace/gpu_node.env
set -uo pipefail
ACTION="${1:-all}"
WS=/workspace
REPO=$WS/cdi
URL="${CDI_REPO_URL:-https://github.com/sayantanIFAI/emr2.git}"
PUB="$WS/secrets/app_node_key.pub"           # the app node's PUBLIC tunnel key (from: model_tunnel.sh keygen on the app node)
mkdir -p "$WS/logs" "$WS/secrets" "$WS/hf-cache"
chmod 700 "$WS/secrets"
export DEBIAN_FRONTEND=noninteractive HF_HOME="$WS/hf-cache" PIP_BREAK_SYSTEM_PACKAGES=1 PIP_ROOT_USER_ACTION=ignore
penv() { tr '\0' '\n' < /proc/1/environ 2>/dev/null | sed -n "s/^$1=//p"; }     # RunPod sets its variables for PID 1, not for ssh sessions

install_app_node_key() {
  # the tunnel's key may ONLY open the gateway and vLLM ports: no shell, no other forwarding (OpenSSH authorized_keys options)
  if [ ! -s "$PUB" ]; then
    echo "app node key: none yet ($PUB). On the app node run:  bash /workspace/cdi/infra/runpod/model_tunnel.sh keygen"
    echo "              put the printed public key into $PUB on this machine, then run:  bash $REPO/infra/runpod/start_gpu_node.sh keyonly"
    return 1
  fi
  local key
  key="$(awk 'NF>=2 {print $1" "$2; exit}' "$PUB")"
  [ -n "$key" ] || { echo "app node key: $PUB is not a public key"; return 1; }
  mkdir -p /root/.ssh && chmod 700 /root/.ssh && touch /root/.ssh/authorized_keys && chmod 600 /root/.ssh/authorized_keys
  if grep -qF "$key" /root/.ssh/authorized_keys; then
    echo "app node key: already installed"
  else
    printf 'restrict,port-forwarding,permitopen="127.0.0.1:%s",permitopen="127.0.0.1:%s" %s cdi-app-node\n' "${CDI_MLSERVE_PORT:-8077}" "${CDI_VLLM_PORT:-8078}" "$key" >> /root/.ssh/authorized_keys
    echo "app node key: installed (it can only forward ${CDI_MLSERVE_PORT:-8077} and ${CDI_VLLM_PORT:-8078} on this machine)"
  fi
}

if [ "$ACTION" = "keyonly" ]; then
  install_app_node_key
  exit $?
fi

echo "== 1. code from git =="
if [ -d "$REPO/.git" ]; then
  git -C "$REPO" fetch -q origin && git -C "$REPO" reset -q --hard origin/main
else
  git clone -q "$URL" "$REPO"
fi
git -C "$REPO" log --oneline -1
cd "$REPO"

echo "== 2. settings (.env) =="
cp .env.runpod .env
[ -f "$WS/gpu_node.env" ] && { set -a; . "$WS/gpu_node.env"; set +a; echo "overrides from $WS/gpu_node.env"; }
set -a; . ./.env; set +a
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -1 || echo 'none found')"
echo "model: ${CDI_VLM_MODEL_ID:-Qwen/Qwen2.5-VL-7B-Instruct}  quant: ${CDI_VLLM_QUANT:-bf16}  GPU memory share: ${CDI_VLLM_GPU_UTIL:-0.80}  sequences: ${CDI_VLLM_MAX_SEQS:-16}"

echo "== 3. python environment (base dependencies only: the model runs in vLLM's own environment) =="
[ -d .venv ] || python3 -m venv --system-site-packages .venv
. .venv/bin/activate
python -c "import cdi_adapter.mlserve.app" 2>/dev/null || pip install -q -e .
python -c "import cdi_adapter.mlserve.app" || { echo "the gateway code does not import: stop"; exit 1; }

echo "== 4. the app node's tunnel key =="
install_app_node_key || true

echo "== 5. model download (once per volume) =="
python - <<'PY'
import os, sys
try:
    from huggingface_hub import snapshot_download
except ImportError:
    sys.exit(3)
repo = os.environ.get("CDI_VLM_MODEL_ID", "Qwen/Qwen2.5-VL-7B-Instruct")
p = snapshot_download(repo, allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "*.jinja"])
print("ready", repo, p, flush=True)
PY
if [ $? -eq 3 ]; then
  pip install -q huggingface_hub hf_transfer
  python - <<'PY'
import os
from huggingface_hub import snapshot_download
repo = os.environ.get("CDI_VLM_MODEL_ID", "Qwen/Qwen2.5-VL-7B-Instruct")
print("ready", repo, snapshot_download(repo, allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "*.jinja"]), flush=True)
PY
fi

echo "== 6. vLLM  (:${CDI_VLLM_PORT:-8078}) =="
VLP="${CDI_VLLM_PORT:-8078}"
if curl -sf "http://127.0.0.1:${VLP}/v1/models" >/dev/null 2>&1; then
  echo "already up"
else
  bash "$REPO/infra/runpod/start_vllm.sh" || { echo "vLLM did not come up: tail $WS/logs/vllm.log"; tail -n 30 "$WS/logs/vllm.log"; exit 1; }
fi

echo "== 7. model gateway  (:${CDI_MLSERVE_PORT:-8077}) =="
MLP="${CDI_MLSERVE_PORT:-8077}"
if curl -s "http://127.0.0.1:${MLP}/healthz" | grep -q '"configured_backend":"vllm"'; then
  echo "already up"
else
  pkill -f cdi_adapter.mlserve 2>/dev/null || true; sleep 1
  CDI_MLSERVE_BACKEND=vllm setsid nohup python -m cdi_adapter.mlserve > "$WS/logs/mlserve.log" 2>&1 < /dev/null &
  echo "mlserve pid $!"
  for _ in $(seq 1 45); do
    curl -s "http://127.0.0.1:${MLP}/healthz" | grep -q configured_backend && break
    sleep 2
  done
fi
curl -s "http://127.0.0.1:${MLP}/healthz"; echo

echo "== 8. summary =="
IP="$(penv RUNPOD_PUBLIC_IP)"; SSHP="$(penv RUNPOD_TCP_PORT_22)"
echo "this GPU node:  public ip ${IP:-(see the pod's Connect menu)}   ssh port ${SSHP:-(see the pod's Connect menu, 'Direct TCP')}"
if [ -r /etc/ssh/ssh_host_ed25519_key.pub ]; then
  echo "ssh host key (the app node should show this when it first connects): $(ssh-keygen -lf /etc/ssh/ssh_host_ed25519_key.pub 2>/dev/null)"
fi
echo "on the app node:  bash /workspace/cdi/infra/runpod/model_tunnel.sh start <ip> <ssh port>"
echo "GPU NODE READY"
