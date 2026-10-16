#!/usr/bin/env bash
# APP NODE: keep an encrypted SSH tunnel to the GPU node's model gateway (8077) and vLLM (8078).
# The app calls them as http://127.0.0.1:8077 and http://127.0.0.1:8078, exactly as when the model is on the same machine.
#
#   bash infra/runpod/model_tunnel.sh keygen                        once: makes the tunnel's own key, prints its PUBLIC key (give it to the GPU node)
#   bash infra/runpod/model_tunnel.sh start <gpu-ip> <gpu-ssh-port>  starts the tunnel; it reconnects by itself within 5 s if it drops
#   bash infra/runpod/model_tunnel.sh status                        is it up, does the model answer, how long is one round trip
#   bash infra/runpod/model_tunnel.sh stop
#
# The key is only good for these two forwards (the GPU node installs it restricted: start_gpu_node.sh). SSH rules for these pods: the direct
# address and -p <port>, never ssh.runpod.io, never -t. The tunnel's own log: /workspace/logs/model_tunnel.log
set -uo pipefail
WS="${CDI_POD_WS:-/workspace}"
SEC="$WS/secrets"
KEY="$SEC/gpu_node_key"
KNOWN="$SEC/gpu_node_known_hosts"
TARGET="$SEC/gpu_node_target"
PIDF="$SEC/gpu_node_tunnel.pid"
LOG="$WS/logs/model_tunnel.log"
MLP="${CDI_MLSERVE_PORT:-8077}"
VLP="${CDI_VLLM_PORT:-8078}"

alive() { [ -s "$PIDF" ] && kill -0 "$(cat "$PIDF")" 2>/dev/null; }

case "${1:-}" in
  keygen)
    mkdir -p "$SEC" "$WS/logs" && chmod 700 "$SEC"
    [ -s "$KEY" ] || ssh-keygen -q -t ed25519 -N "" -C "cdi-app-node" -f "$KEY"
    chmod 600 "$KEY"
    echo "PUBLIC key of the tunnel. On the GPU node, save it as /workspace/secrets/app_node_key.pub and run: bash /workspace/cdi/infra/runpod/start_gpu_node.sh keyonly"
    cat "$KEY.pub"
    ;;
  start)
    IP="${2:?usage: model_tunnel.sh start <gpu-ip> <gpu-ssh-port>}"
    PORT="${3:?usage: model_tunnel.sh start <gpu-ip> <gpu-ssh-port>}"
    [ -s "$KEY" ] || { echo "no tunnel key yet: run  bash $0 keygen  first"; exit 1; }
    "$0" stop >/dev/null 2>&1
    mkdir -p "$WS/logs"
    echo "$IP $PORT" > "$TARGET"
    setsid nohup bash -c "while true; do
      ssh -N -L 127.0.0.1:$MLP:127.0.0.1:$MLP -L 127.0.0.1:$VLP:127.0.0.1:$VLP -p $PORT -i $KEY \
        -o IdentitiesOnly=yes -o BatchMode=yes -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile=$KNOWN \
        -o ServerAliveInterval=15 -o ServerAliveCountMax=3 -o ExitOnForwardFailure=yes root@$IP
      echo \"\$(date +%FT%T) the tunnel ended; trying again in 5 s\"
      sleep 5
    done" >> "$LOG" 2>&1 < /dev/null &
    echo $! > "$PIDF"
    echo "tunnel started (pid $(cat "$PIDF")) to $IP:$PORT; waiting for the model to answer ..."
    for _ in $(seq 1 30); do
      curl -sf -m 3 "http://127.0.0.1:${VLP}/v1/models" >/dev/null 2>&1 && { echo "vLLM answers through the tunnel"; break; }
      sleep 1
    done
    "$0" status
    ;;
  stop)
    if [ -s "$PIDF" ]; then
      p="$(cat "$PIDF")"
      kill -- "-$p" 2>/dev/null || kill "$p" 2>/dev/null      # the loop and its ssh are one process group: stopped by id, never pkill -f
      rm -f "$PIDF"
      echo "tunnel stopped"
    else
      echo "no tunnel is recorded as running"
    fi
    ;;
  status)
    if alive; then echo "tunnel: running (pid $(cat "$PIDF"), to $(cat "$TARGET" 2>/dev/null))"; else echo "tunnel: NOT running"; fi
    echo "gateway :${MLP}  -> $(curl -s -m 4 "http://127.0.0.1:${MLP}/healthz" | head -c 150 || true)"
    echo "vLLM    :${VLP}  -> $(curl -s -m 4 "http://127.0.0.1:${VLP}/v1/models" | head -c 90 || true)"
    if curl -sf -m 4 -o /dev/null "http://127.0.0.1:${MLP}/healthz" 2>/dev/null; then
      t=0; n=0
      for _ in 1 2 3 4 5; do
        s="$(curl -s -m 4 -o /dev/null -w '%{time_total}' "http://127.0.0.1:${MLP}/healthz" 2>/dev/null || echo 0)"
        t="$(awk -v a="$t" -v b="$s" 'BEGIN{print a+b}')"; n=$((n+1))
      done
      awk -v t="$t" -v n="$n" 'BEGIN{ printf "one round trip to the GPU node (gateway health check), average of %d: %.0f ms\n", n, t/n*1000 }'
    else
      echo "round trip: not measured (the gateway does not answer through the tunnel)"
    fi
    tail -n 3 "$LOG" 2>/dev/null | sed 's/^/log: /'
    ;;
  *)
    echo "usage: model_tunnel.sh keygen | start <gpu-ip> <gpu-ssh-port> | status | stop"; exit 2 ;;
esac
