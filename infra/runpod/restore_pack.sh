#!/usr/bin/env bash
# Put a backup pack (backup_pack.sh) back onto a pod whose /workspace was lost, and start everything.
#
#   scp -P <port> cdi-state.tar.gz root@<ip>:/workspace/offpod/      # from your computer
#   bash <(curl -fsSL https://raw.githubusercontent.com/sayantanIFAI/emr2/main/infra/runpod/restore_pack.sh) /workspace/offpod/cdi-state.tar.gz
#
# Order matters: the pack is unpacked FIRST (the passwords are kept, and the database dump is there when
# the fresh Postgres is built), then the normal deploy clones the code, downloads the models and starts.
set -euo pipefail
PACK="${1:?usage: restore_pack.sh /path/to/cdi-state.tar.gz}"
WS="${WS:-/workspace}"
[ -s "$PACK" ] || { echo "no such pack: $PACK" >&2; exit 1; }
if [ -s "$PACK.sha256" ]; then
  ( cd "$(dirname "$PACK")" && sha256sum -c "$(basename "$PACK").sha256" ) || { echo "the pack is damaged" >&2; exit 1; }
fi
echo "== stopping anything running (files must not be live while they are put back)"
for pat in '[c]di_adapter.webapp' '[c]di_adapter.mlserve' '[c]di_adapter.ocrhost' '[c]di_adapter.listener' '[w]orkspace/bin/weed'; do
  pkill -f "$pat" 2>/dev/null || true
done
sleep 3
echo "== unpacking $PACK into $WS"
mkdir -p "$WS"
tar -xzf "$PACK" -C "$WS"
chmod 700 "$WS/secrets" 2>/dev/null || true
echo "== deploying from git (clones the code, downloads the models, starts everything)"
if [ -d "$WS/cdi/.git" ]; then
  git -C "$WS/cdi" fetch -q origin && git -C "$WS/cdi" reset -q --hard origin/main
  exec bash "$WS/cdi/infra/runpod/deploy_fresh_pod.sh"
fi
curl -fsSL https://raw.githubusercontent.com/sayantanIFAI/emr2/main/infra/runpod/deploy_fresh_pod.sh | bash
