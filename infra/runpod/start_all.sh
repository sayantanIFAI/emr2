#!/usr/bin/env bash
# THE single entrypoint. Run this once per pod boot:
#     bash /workspace/cdi/infra/runpod/start_all.sh
#
# Everything needed lives on /workspace (repo, venv, model cache, SeaweedFS object
# store, Postgres dump). This script re-creates only the ephemeral parts:
#   - apt packages (postgres/redis/cron)        [bootstrap_pod.sh]
#   - the Postgres cluster on the overlay,      [bootstrap_pod.sh]
#     restored from /workspace/backup/cdi.dump
#   - the model gateway (Qwen2.5-VL) process
#   - the CPU OCR host (RapidOCR, recognition v2)
#   - the web app process
#   - background agents: file listener + recovery. The FHIR builder and dispatch agents start ONLY
#     when the owner sets CDI_START_FHIR=1 / CDI_START_DISPATCH=1 (they are off by default).
set -uo pipefail
REPO=/workspace/cdi
cd "$REPO"
mkdir -p /workspace/logs

if ! mountpoint -q /workspace 2>/dev/null; then
  echo "######################################################################"
  echo "# WARNING: /workspace is this pod's own container disk, NOT a volume."
  echo "# A stop / reset / delete of the pod DELETES the models, the database,"
  echo "# the stored prescriptions and the passwords. Attach a RunPod volume"
  echo "# at /workspace, and until then copy the backup pack off the pod:"
  echo "#   bash /workspace/cdi/infra/runpod/prepare_stop.sh"
  echo "######################################################################"
fi

echo "########## 1. infra (bootstrap) ##########"
bash "$REPO/infra/runpod/bootstrap_pod.sh" || echo "(bootstrap returned non-zero; continuing)"

. .venv/bin/activate
set -a; . "$REPO/.env"; set +a
export HF_HOME=/workspace/hf-cache
bash "$REPO/infra/runpod/ensure_indian_codes.sh" || true       # the national lab / drug code lists the gate reads (not in git: see that script)
MLP="${CDI_MLSERVE_PORT:-8077}"
WBP="${CDI_WEBAPP_PORT:-8080}"

if [ "${CDI_MLSERVE_BACKEND:-hf}" = "vllm" ]; then
  echo "########## 1b. vLLM engine  (:${CDI_VLLM_PORT:-8078}) ##########"
  VLP="${CDI_VLLM_PORT:-8078}"
  if curl -sf "http://127.0.0.1:${VLP}/v1/models" >/dev/null 2>&1; then
    echo "already up"
  else
    bash "$REPO/infra/runpod/start_vllm.sh" || true
  fi
  if ! curl -sf "http://127.0.0.1:${VLP}/v1/models" >/dev/null 2>&1; then
    # never leave the pod without a reader: the slower transformers server, one request at a time
    echo "!! vLLM did not come up - using the slower transformers server instead (see /workspace/logs/vllm.log)"
    export CDI_MLSERVE_BACKEND=hf CDI_QWEN_LINE_CONCURRENCY=1 CDI_EXTRACT_CONCURRENCY=1
  fi
fi

echo "########## 2. model gateway  (:$MLP) ##########"
if curl -s "http://127.0.0.1:${MLP}/healthz" | grep -q configured_backend; then
  echo "already up"
else
  pkill -f cdi_adapter.mlserve 2>/dev/null || true; sleep 1
  setsid nohup python -m cdi_adapter.mlserve > /workspace/logs/mlserve.log 2>&1 < /dev/null &
  echo "mlserve pid $!"
  for _ in $(seq 1 45); do
    curl -s "http://127.0.0.1:${MLP}/healthz" | grep -q configured_backend && break
    sleep 2
  done
fi
curl -s "http://127.0.0.1:${MLP}/healthz"; echo

echo "########## 2b. CPU OCR host  (:${CDI_OCRHOST_PORT:-8079}) ##########"
bash "$REPO/infra/runpod/start_ocrhost.sh" || echo "(ocrhost failed - handwriting lines will be single-engine -> review)"

echo "########## 3. web app  (:$WBP) ##########"
# on these pods the only RunPod-edge-routed HTTP port is 8888 (Jupyter's) — take it
pkill -f jupyter 2>/dev/null || true
pkill -f cdi_adapter.webapp 2>/dev/null || true; sleep 3
setsid nohup python -m cdi_adapter.webapp > /workspace/logs/webapp.log 2>&1 < /dev/null &
echo "webapp pid $!"
for _ in $(seq 1 30); do
  curl -sf "http://127.0.0.1:${WBP}/healthz" >/dev/null && break
  sleep 1
done
curl -s "http://127.0.0.1:${WBP}/healthz"; echo

echo "########## 4. agents ##########"
# FHIR is built only when the owner says so: stop a leftover builder and do not start a new one
pkill -f "cdi_adapter.agents.fhir_builder" 2>/dev/null || true
if [ "${CDI_START_FHIR:-0}" = "1" ]; then
  setsid nohup python -m cdi_adapter.agents.fhir_builder > /workspace/logs/fhir_builder.log 2>&1 < /dev/null &
  echo "fhir_builder pid $!"
else
  echo "fhir_builder: not started (CDI_START_FHIR is not 1)"
fi
for mod in listener.recovery; do
  pkill -f "cdi_adapter.$mod" 2>/dev/null || true
  setsid nohup python -m "cdi_adapter.$mod" > "/workspace/logs/${mod##*.}.log" 2>&1 < /dev/null &
  echo "$mod pid $!"
done
# the listener runs only when asked (it moves files in the configured drive)
if [ "${CDI_START_LISTENER:-0}" = "1" ]; then
  pkill -f "cdi_adapter.listener.service" 2>/dev/null || true
  setsid nohup python -m cdi_adapter.listener.service > /workspace/logs/listener.log 2>&1 < /dev/null &
  echo "listener pid $!  (connector ${CDI_LISTENER_CONNECTOR:-local})"
fi
# dispatch (downstream screens) is off unless the owner asks for it
pkill -f "cdi_adapter.agents.dispatch" 2>/dev/null || true
if [ "${CDI_START_DISPATCH:-0}" = "1" ]; then
  setsid nohup python -m cdi_adapter.agents.dispatch run > /workspace/logs/dispatch.log 2>&1 < /dev/null &
  echo "dispatch pid $!"
else
  echo "dispatch: not started (CDI_START_DISPATCH is not 1)"
fi

POD=$(tr '\0' '\n' < /proc/1/environ | sed -n 's/^RUNPOD_POD_ID=//p')
echo
echo "======================================================================"
echo " LIVE URL:  https://${POD}-${WBP}.proxy.runpod.net"
echo "======================================================================"
