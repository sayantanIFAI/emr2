#!/usr/bin/env bash
# Update a RUNNING pod to the latest origin/main without touching the model server (vLLM keeps its GPU memory):
# pull -> migrate the database -> restart the web app, the OCR host and the retry agent -> wait until healthy.
# Settings in /workspace/cdi/.env are kept; pass KEY=VALUE pairs to change some:
#     bash infra/runpod/update_pod.sh CDI_RAPIDOCR_USE_CUDA=false
set -uo pipefail
REPO=/workspace/cdi
cd "$REPO"
git fetch -q origin && git reset -q --hard origin/main
git log --oneline -1
for kv in "$@"; do
  k="${kv%%=*}"
  if grep -q "^${k}=" .env; then sed -i "s|^${k}=.*|${kv}|" .env; else printf '%s\n' "$kv" >> .env; fi
  echo "set ${k}"
done
. .venv/bin/activate
set -a; . ./.env; set +a
export HF_HOME=/workspace/hf-cache
alembic upgrade head 2>&1 | tail -2
WBP="${CDI_WEBAPP_PORT:-8888}"
pkill -f cdi_adapter.webapp 2>/dev/null || true
pkill -f cdi_adapter.listener.recovery 2>/dev/null || true
bash "$REPO/infra/runpod/start_ocrhost.sh" || echo "(ocrhost did not answer; see /workspace/logs/ocrhost.log)"
sleep 3
setsid nohup python -m cdi_adapter.webapp > /workspace/logs/webapp.log 2>&1 < /dev/null &
setsid nohup python -m cdi_adapter.listener.recovery > /workspace/logs/recovery.log 2>&1 < /dev/null &
for _ in $(seq 1 60); do
  curl -sf "http://127.0.0.1:${WBP}/healthz" >/dev/null && break
  sleep 1
done
curl -s "http://127.0.0.1:${WBP}/healthz"; echo
tail -3 /workspace/logs/webapp.log | cut -c1-200
