#!/usr/bin/env bash
# Run on YOUR machine: put the data that is not in git onto the APP NODE (the CPU pod).
#
#   bash infra/runpod/push_to_app_node.sh <app-ip> <app-ssh-port> [--with-pages] [--dry-run]
#
# Needs the staging folder from  preflight_laptop.sh stage  (default ~/pod_stage, override with POD_STAGE=...).
# Sends: the national lists and the medicine-name file to /workspace/data, and the C-DAC zips to /workspace/data-src (so the pod can rebuild the lists).
# Prescription pages are patients' data: they are sent ONLY with --with-pages (to /workspace/loadtest/images).
# SSH rules for these pods: the direct address and -p <port>, scp with -P, never ssh.runpod.io, never -t.
set -euo pipefail
IP="${1:?usage: push_to_app_node.sh <app-ip> <app-ssh-port> [--with-pages] [--dry-run]}"
PORT="${2:?usage: push_to_app_node.sh <app-ip> <app-ssh-port> [--with-pages] [--dry-run]}"
shift 2
PAGES=0; DRY=0
for a in "$@"; do
  case "$a" in
    --with-pages) PAGES=1 ;;
    --dry-run) DRY=1 ;;
    *) echo "unknown option $a"; exit 2 ;;
  esac
done
KEY="${SSH_KEY:-$HOME/.ssh/id_ed25519}"
STAGE="${POD_STAGE:-$HOME/pod_stage}"
ZIPS="${C_DAC_DIR:-$HOME/Downloads}"
SSHO=(-p "$PORT" -i "$KEY" -o StrictHostKeyChecking=accept-new)
SCPO=(-P "$PORT" -i "$KEY" -o StrictHostKeyChecking=accept-new)
run() { if [ "$DRY" = 1 ]; then printf 'DRY RUN: %s\n' "$*"; else "$@"; fi; }

for f in codes/clci_labs.json codes/cdci_drugs.json codes/medicine_names.txt; do
  [ -s "$STAGE/$f" ] || { echo "missing $STAGE/$f: run  bash infra/runpod/preflight_laptop.sh stage"; exit 1; }
done
run ssh "${SSHO[@]}" "root@$IP" 'mkdir -p /workspace/data /workspace/data-src /workspace/loadtest/images'
run scp "${SCPO[@]}" "$STAGE/codes/clci_labs.json" "$STAGE/codes/cdci_drugs.json" "$STAGE/codes/medicine_names.txt" "root@$IP:/workspace/data/"
LABS="$(ls "$ZIPS"/common-lab-codes-for-india-*.zip 2>/dev/null | tail -1 || true)"
DRUGS="$(ls "$ZIPS"/CommonDrugCodesForIndia_FlatFilePackage*.zip 2>/dev/null | tail -1 || true)"
[ -n "$LABS" ] && [ -n "$DRUGS" ] && run scp "${SCPO[@]}" "$LABS" "$DRUGS" "root@$IP:/workspace/data-src/"
run ssh "${SSHO[@]}" "root@$IP" 'chmod 600 /workspace/data/*.json; ls -la /workspace/data'
if [ "$PAGES" = 1 ]; then
  n="$(ls "$STAGE/pages" 2>/dev/null | grep -c -i -E '\.(jpe?g|png|pdf|tiff?)$' || true)"
  [ "$n" -gt 0 ] || { echo "no pages in $STAGE/pages"; exit 1; }
  echo "sending $n page(s) (patients' data) to /workspace/loadtest/images"
  run scp "${SCPO[@]}" "$STAGE"/pages/* "root@$IP:/workspace/loadtest/images/"
fi
echo "done. Load the lists into the running app:  ssh -p $PORT -i $KEY root@$IP \"bash /workspace/cdi/infra/runpod/update_pod.sh\""
