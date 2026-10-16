#!/usr/bin/env bash
# Record what the pod is doing while a load test runs: GPU (use, memory, power), CPU, memory, the busiest processes and vLLM's own queue.
#
#   bash /workspace/cdi/infra/runpod/load_monitor.sh start run1      # before the load test
#   bash /workspace/cdi/infra/runpod/load_monitor.sh stop  run1      # after it; prints a short summary
#
# Files go to /workspace/loadtest/<run>/ : gpu.csv (every second), cpu_vmstat.txt (every second), mem.txt, top.txt and vllm_metrics.txt (every 5 s).
# The background loops are stopped by PROCESS ID from the pids file: never pkill -f over ssh (it kills your own ssh shell too).
set -uo pipefail
ACT="${1:?usage: load_monitor.sh start|stop <run-name>}"
NAME="${2:?usage: load_monitor.sh start|stop <run-name>}"
DIR="/workspace/loadtest/$NAME"
PIDS="$DIR/pids"
VLLM_PORT="${CDI_VLLM_PORT:-8078}"

case "$ACT" in
  start)
    mkdir -p "$DIR"
    : > "$PIDS"
    if command -v nvidia-smi >/dev/null 2>&1; then
      nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,temperature.gpu,clocks.sm \
        --format=csv -l 1 > "$DIR/gpu.csv" 2>&1 &
      echo $! >> "$PIDS"
      nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv > "$DIR/gpu_info.csv" 2>&1
    else
      echo "warning: nvidia-smi not found: no GPU samples" >&2
    fi
    command -v vmstat >/dev/null 2>&1 && { vmstat -t 1 > "$DIR/cpu_vmstat.txt" 2>&1 & echo $! >> "$PIDS"; }
    # the CPU this CONTAINER really used (vmstat shows the whole host, e.g. 128 cores, while the pod is limited to a share of them)
    if [ -r /sys/fs/cgroup/cpu.stat ]; then
      echo "$(date +%s.%N) $(awk '/^usage_usec/ {print $2}' /sys/fs/cgroup/cpu.stat)" > "$DIR/cgroup_cpu_start.txt"
      cat /sys/fs/cgroup/cpu.max > "$DIR/cgroup_cpu_max.txt" 2>/dev/null
    fi
    ( while true; do date +%FT%T; free -m | sed -n 1,2p; sleep 5; done ) > "$DIR/mem.txt" 2>&1 &
    echo $! >> "$PIDS"
    ( while true; do echo "# $(date +%FT%T)"; ps -eo pcpu,pmem,rss,comm --sort=-pcpu | head -9; sleep 5; done ) > "$DIR/top.txt" 2>&1 &
    echo $! >> "$PIDS"
    ( while true; do
        echo "# $(date +%FT%T)"
        curl -s -m 3 "http://127.0.0.1:${VLLM_PORT}/metrics" | grep -E '^vllm:(num_requests_running|num_requests_waiting|gpu_cache_usage_perc|kv_cache_usage_perc|generation_tokens_total|prompt_tokens_total|request_success_total)'
        sleep 5
      done ) > "$DIR/vllm_metrics.txt" 2>&1 &
    echo $! >> "$PIDS"
    echo "recording to $DIR  ($(wc -l < "$PIDS") background loops; stop with:  bash $0 stop $NAME)"
    ;;
  stop)
    [ -f "$PIDS" ] || { echo "no pids file in $DIR: nothing to stop"; exit 1; }
    while read -r p; do [ -n "$p" ] && kill "$p" 2>/dev/null; done < "$PIDS"
    sleep 1
    rm -f "$PIDS"
    echo "stopped. Summary of $DIR:"
    if [ -s "$DIR/gpu.csv" ]; then
      awk -F', ' 'NR>1 { gsub(/ %| MiB| W/,""); u+=$2; n++; if($2>um)um=$2; if($4>mm)mm=$4; if($6>pm)pm=$6 }
                  END { if(n) printf "GPU: %d samples, average use %.0f %%, peak use %d %%, peak memory %d MiB, peak power %.0f W\n", n, u/n, um, mm, pm }' "$DIR/gpu.csv"
    fi
    if [ -s "$DIR/cpu_vmstat.txt" ]; then
      awk 'NF>=17 && $1 ~ /^[0-9]+$/ { idle=$15; n++; busy+=100-idle; if(100-idle>pk)pk=100-idle }
           END { if(n) printf "CPU: %d samples, average busy %.0f %%, peak busy %d %% (of ALL cores the pod can see)\n", n, busy/n, pk }' "$DIR/cpu_vmstat.txt"
    fi
    if [ -s "$DIR/cgroup_cpu_start.txt" ] && [ -r /sys/fs/cgroup/cpu.stat ]; then
      read -r t0 u0 < "$DIR/cgroup_cpu_start.txt"
      t1="$(date +%s.%N)"; u1="$(awk '/^usage_usec/ {print $2}' /sys/fs/cgroup/cpu.stat)"
      awk -v t0="$t0" -v t1="$t1" -v u0="$u0" -v u1="$u1" -v mx="$(cut -d' ' -f1 "$DIR/cgroup_cpu_max.txt" 2>/dev/null) $(cut -d' ' -f2 "$DIR/cgroup_cpu_max.txt" 2>/dev/null)" \
        'BEGIN { s=t1-t0; if(s>0) printf "CPU used by this container: %.1f cores on average over %.0f s (limit: %s)\n", (u1-u0)/1e6/s, s, mx }'
    fi
    grep -E '^Mem:' "$DIR/mem.txt" 2>/dev/null | awk '{ if($3>m)m=$3 } END { if(m) printf "Memory: peak used %d MiB\n", m }'
    grep -E '^vllm:num_requests_running' "$DIR/vllm_metrics.txt" 2>/dev/null | awk '{ if($NF>m)m=$NF } END { if(m!="") printf "vLLM: most requests running at once %d\n", m }'
    grep -E '^vllm:num_requests_waiting' "$DIR/vllm_metrics.txt" 2>/dev/null | awk '{ if($NF>m)m=$NF } END { if(m!="") printf "vLLM: most requests waiting %d\n", m }'
    ;;
  *)
    echo "usage: load_monitor.sh start|stop <run-name>"; exit 2 ;;
esac
