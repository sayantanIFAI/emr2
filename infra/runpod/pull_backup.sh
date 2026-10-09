#!/usr/bin/env bash
# Run on YOUR computer: makes a fresh backup pack on the pod and copies it here, keeping the last 5.
#
#   bash infra/runpod/pull_backup.sh <ip> <ssh-port> [ssh-key] [local-dir]
#
# The pack holds what cannot be downloaded again: the generated passwords, the database dump, the stored prescriptions and their results.
# The code (git) and the models are not in it. Restore on a new pod with infra/runpod/restore_pack.sh.
# SSH rules for these pods: the direct port, never -t, scp with -P.
set -euo pipefail
IP="${1:?pod ip}"; PORT="${2:?ssh port}"; KEY="${3:-$HOME/.ssh/id_ed25519}"; DIR="${4:-$HOME/pod-backups}"
mkdir -p "$DIR"
SSH=(ssh -p "$PORT" -i "$KEY" -o StrictHostKeyChecking=no "root@$IP")
"${SSH[@]}" 'bash /workspace/cdi/infra/runpod/backup_pack.sh /workspace/offpod/cdi-state.tar.gz' | tail -1
STAMP="$(date +%Y%m%d-%H%M%S)"
scp -q -P "$PORT" -i "$KEY" -o StrictHostKeyChecking=no "root@$IP:/workspace/offpod/cdi-state.tar.gz" "$DIR/cdi-state-$STAMP.tar.gz"
scp -q -P "$PORT" -i "$KEY" -o StrictHostKeyChecking=no "root@$IP:/workspace/offpod/cdi-state.tar.gz.sha256" "$DIR/cdi-state-$STAMP.tar.gz.sha256"
( cd "$DIR" && ls -1t cdi-state-*.tar.gz | tail -n +6 | while read -r f; do rm -f "$f" "$f.sha256"; done )
echo "saved: $DIR/cdi-state-$STAMP.tar.gz ($(du -h "$DIR/cdi-state-$STAMP.tar.gz" | cut -f1))"
