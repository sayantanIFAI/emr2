#!/usr/bin/env bash
# Fresh pod -> running admin upload service, from git. Safe to re-run.
#
#   curl -fsSL https://raw.githubusercontent.com/sayantanIFAI/emr2/main/infra/runpod/deploy_fresh_pod.sh | bash
#   (or: bash /workspace/cdi/infra/runpod/deploy_fresh_pod.sh)
#
# What it does: clone/update the repo -> .env from .env.runpod + a generated admin password (kept in
# /workspace/secrets, never in git) -> pre-download the models -> start_all.sh (Postgres, Redis, object
# store, model gateway Qwen2.5-VL-7B, OCR host RapidOCR, web app) -> smoke test.
# The FHIR builder, dispatch and the review screens stay OFF (see .env.runpod).
set -uo pipefail
WS=/workspace
REPO=$WS/cdi
URL="${CDI_REPO_URL:-https://github.com/sayantanIFAI/emr2.git}"
mkdir -p "$WS/logs" "$WS/secrets" "$WS/hf-cache"
chmod 700 "$WS/secrets"
export DEBIAN_FRONTEND=noninteractive HF_HOME="$WS/hf-cache" PIP_BREAK_SYSTEM_PACKAGES=1 PIP_ROOT_USER_ACTION=ignore

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
PWF="$WS/secrets/admin_password"
if [ ! -s "$PWF" ]; then
  python3 -c 'import secrets;print(secrets.token_urlsafe(18))' > "$PWF"
  chmod 600 "$PWF"
fi
printf '\nCDI_ADMIN_PASSWORD=%s\n' "$(cat "$PWF")" >> .env
# no published default credentials in a running pod (the app refuses to start with them)
secret() { f="$WS/secrets/$1"; [ -s "$f" ] || { python3 -c 'import secrets;print(secrets.token_urlsafe(24))' > "$f"; chmod 600 "$f"; }; cat "$f"; }
S3K="app$(secret s3_access_key | tr -dc 'A-Za-z0-9' | head -c 16)"
S3S="$(secret s3_secret_key)"
DBP="$(secret db_password | tr -dc 'A-Za-z0-9' | head -c 28)"
printf 'CDI_S3_ACCESS_KEY=%s\nCDI_S3_SECRET_KEY=%s\nCDI_DB_PASSWORD=%s\n' "$S3K" "$S3S" "$DBP" >> .env
printf 'CDI_DATABASE_URL=postgresql+psycopg://cdi:%s@127.0.0.1:5432/cdi\n' "$DBP" >> .env
chmod 600 .env
echo "admin user: admin   password file: $PWF"

echo "== 3. models (downloaded once, pinned by the hub revision at download time) =="
pip install -q huggingface_hub hf_transfer 2>&1 | tail -1
cat > "$WS/prefetch_models.py" <<'PY'
import os
from huggingface_hub import snapshot_download
for repo, extra in (("Qwen/Qwen2.5-VL-7B-Instruct", {"allow_patterns": ["*.json", "*.safetensors", "*.txt", "*.model", "*.jinja"]}),):
    p = snapshot_download(repo, **extra)
    print("ready", repo, p, flush=True)
print("PREFETCH_DONE", flush=True)
PY
setsid nohup python3 "$WS/prefetch_models.py" > "$WS/logs/prefetch.log" 2>&1 < /dev/null &
echo "prefetch pid $!  (log $WS/logs/prefetch.log)"

echo "== 4. start everything =="
setsid nohup bash "$REPO/infra/runpod/start_all.sh" > "$WS/logs/start_all.log" 2>&1 < /dev/null &
echo "start_all pid $!  (log $WS/logs/start_all.log)"
echo "poll:  tail -f $WS/logs/start_all.log $WS/logs/prefetch.log"
