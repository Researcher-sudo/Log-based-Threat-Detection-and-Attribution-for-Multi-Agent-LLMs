#!/usr/bin/env bash
set -euo pipefail
aid="$1"; aname="$2"; model="$3"; out="$4"

mkdir -p "$out"
printf "agent_id=%s\nagent_name=%s\nmodel=%s\n" "$aid" "$aname" "$model" > "$out/env.txt"

# start light collectors (best-effort)
( vmstat 1 > "$out/vmstat.log" 2>&1 ) &
VMPID=$!
if command -v nvidia-smi >/dev/null 2>&1; then
  ( nvidia-smi --query-gpu=timestamp,index,utilization.gpu,utilization.memory,memory.used,memory.total \
      --format=csv -l 1 > "$out/gpu.csv" 2>&1 ) &
  GPUPID=$!
else
  GPUPID=""
fi
if command -v tcpdump >/dev/null 2>&1; then
  # tcpdump may need caps; ignore if it fails
  ( tcpdump -i any -s 262144 -w "$out/traffic.pcap" ) >/dev/null 2>&1 &
  TCPPID=$! || TCPPID=""
else
  TCPPID=""
fi

# run the python worker under strace (per-agent files via -ff)
# (if strace missing / permission denied, still run worker)
if command -v strace >/dev/null 2>&1; then
  strace -ff -ttT -o "$out/strace" \
    python3 tools/agent_worker.py --model "$model" --role "$aname" --log "$out/app.log"
else
  python3 tools/agent_worker.py --model "$model" --role "$aname" --log "$out/app.log"
fi

# on exit, stop collectors; redact app.log into app.filtered.log
[ -n "${VMPID:-}" ]  && kill "$VMPID" || true
[ -n "${GPUPID:-}" ] && kill "$GPUPID" || true
[ -n "${TCPPID:-}" ] && kill "$TCPPID" || true

# trivial redaction (strip control chars)
tr -d '\000-\010\013\014\016-\037' < "$out/app.log" > "$out/app.filtered.log" || true
