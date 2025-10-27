# #!/usr/bin/env bash
# set -euo pipefail


# PYTHON=$(command -v python3 || true)
# if [ -z "$PYTHON" ]; then
#   PYTHON=$(command -v python || true)
# fi
# if [ -z "$PYTHON" ]; then
#   echo "[!] No python or python3 found in PATH"
#   exit 1
# fi


# # ---- inputs (edit these if paths differ) --------------------------------------
# BENIGN="full_pack/benign_preview.jsonl"
# ATTACKS="full_pack/compromised_preview.jsonl"
# MODEL="llama2-uncensored:georgesung/llama2_7b_chat_uncensored"

# # Optional pacing between prompts (seconds) – leave unset to disable
# SLEEP_BENIGN="${SLEEP_BENIGN:-}"
# SLEEP_ATTACKS="${SLEEP_ATTACKS:-}"

# # ---- tools (optional collectors) ---------------------------------------------
# MPROF=$(command -v mprof || true)     # memory_profiler (mprof) is optional
# TSHARK=$(command -v tshark || true)   # tshark is optional
# NVSMI=$(command -v nvidia-smi || true)

# run_one() {
#   local label="$1"        # "benign" or "attacks"
#   local jsonl="$2"        # path to JSONL (benign or attacks)
#   local extra_sleep="$3"  # e.g. "--sleep 0.05" or ""

#   local TS; TS=$(date +%Y%m%d-%H%M%S)
#   local LOG_DIR="$PWD/run_${label}_${TS}"
#   mkdir -p "$LOG_DIR"

#   local APP_LOG="$LOG_DIR/app.log"
#   local AGENT_JSON="$LOG_DIR/agent.jsonl"
#   local PCAP="$LOG_DIR/traffic.pcap"
#   local VMSTAT="$LOG_DIR/vmstat.log"
#   local GPUCSV="$LOG_DIR/gpu.csv"
#   local PSTATS="$LOG_DIR/profile.pstats"
#   local MEMPLOT="$LOG_DIR/mprof_$(date +%s).dat"
#   local STRACE_BASE="$LOG_DIR/strace"

#   # make per-message logger write here
#   export AGENT_JSON_LOG="$AGENT_JSON"

#   # snapshot the environment (best-effort)
#   env > "$LOG_DIR/env.txt"
#   conda list > "$LOG_DIR/conda_pkgs.txt" 2>/dev/null || true
#   git rev-parse HEAD              > "$LOG_DIR/git_sha.txt"   2>/dev/null || true
#   git diff --patch --stat         > "$LOG_DIR/git_diff.patch" 2>/dev/null || true

#   # ---- collectors -------------------------------------------------------------
#   # vmstat
#   vmstat 1 > "$VMSTAT" & VMSTAT_PID=$!

#   # GPU (if available)
#   if [ -n "$NVSMI" ]; then
#     echo "timestamp,util.gpu %,mem.used MB,temp C,power W" > "$GPUCSV"
#     (
#       while true; do
#         nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used,temperature.gpu,power.draw \
#                    --format=csv,noheader >> "$GPUCSV"
#         sleep 1
#       done
#     ) & GPU_PID=$!
#   else
#     GPU_PID=
#   fi

#   # Full packet capture (needs sudo)
#   if sudo -n true 2>/dev/null; then
#     # sudo tcpdump -i any -w "$PCAP" not port 22 & TCPDUMP_PID=$!
#     sudo tcpdump -U -i any -w "$PCAP" not port 22 & TCPDUMP_PID=$!
#   else
#     echo "[!] sudo not available non-interactively; skipping tcpdump for $label"
#     TCPDUMP_PID=
#   fi

#   # ---- run the program with cProfile + strace (+ mprof if present) -----------
#   echo "[*] launching generate_agent_logs.py for $label …  (logs → $LOG_DIR)"
#   # Build the left side of the pipeline (no pipes until tee):
#   # If mprof exists, prefix with `mprof run -o …`
#   if [ -n "$MPROF" ]; then
#     $MPROF run -o "$MEMPLOT" \
#       strace -qq -tt -ff -o "$STRACE_BASE" \
#       "$PYTHON" -m cProfile -o "$PSTATS" \
#       generate_agent_logs.py \
#       --model "$MODEL" \
#       --outdir "$LOG_DIR" \
#       $extra_sleep \
#       $( [ "$label" = "benign" ] && echo --benign "$jsonl" ) \
#       $( [ "$label" = "attacks" ] && echo --attacks "$jsonl" ) \
#       |& tee "$APP_LOG"
#     status=${PIPESTATUS[0]}   # exit of the left side (python / mprof wrapper)
#   else
#     strace -qq -tt -ff -o "$STRACE_BASE" \
#       "$PYTHON" -m cProfile -o "$PSTATS" \
#       generate_agent_logs.py \
#       --model "$MODEL" \
#       --outdir "$LOG_DIR" \
#       $extra_sleep \
#       $( [ "$label" = "benign" ] && echo --benign "$jsonl" ) \
#       $( [ "$label" = "attacks" ] && echo --attacks "$jsonl" ) \
#       |& tee "$APP_LOG"
#     status=${PIPESTATUS[0]}
#   fi

#   # ---- quick post-processing (if tshark present) ------------------------------
#   if [ -n "$TCPDUMP_PID" ] && [ -n "$TSHARK" ]; then
#     $TSHARK -r "$PCAP" -Y http \
#       -T fields -e frame.time -e ip.dst -e http.request.full_uri \
#       > "$LOG_DIR/http.tsv" || true
#     $TSHARK -r "$PCAP" -Y "dns.flags.response==0" \
#       -T fields -e frame.time -e dns.qry.name \
#       > "$LOG_DIR/dns.tsv" || true
#   fi

#   # ---- cleanup collectors -----------------------------------------------------
#   [ -n "${VMSTAT_PID:-}" ] && kill $VMSTAT_PID 2>/dev/null || true
#   [ -n "${GPU_PID:-}"    ] && kill $GPU_PID    2>/dev/null || true
#   # [ -n "${TCPDUMP_PID:-}" ] && sudo kill $TCPDUMP_PID 2>/dev/null || true
#   [ -n "${TCPDUMP_PID:-}" ] && sudo kill -2 "$TCPDUMP_PID" 2>/dev/null || true


#   echo "[*] done ($label, exit $status) – artefacts in $LOG_DIR"
#   return $status
# }

# # Run benign to its own folder
# run_one "benign"  "$BENIGN"  "${SLEEP_BENIGN:+--sleep $SLEEP_BENIGN}"   # I have changed it temporary to only run attacks now

# # Run attacks to its own folder
# # run_one "attacks" "$ATTACKS" "${SLEEP_ATTACKS:+--sleep $SLEEP_ATTACKS}"



# #!/usr/bin/env bash
# set -euo pipefail

# SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# TOOLS_DIR="$SCRIPT_DIR/tools"

# # --- Python detection ---
# PYTHON=$(command -v python3 || true)
# if [ -z "$PYTHON" ]; then
#   PYTHON=$(command -v python || true)
# fi
# if [ -z "$PYTHON" ]; then
#   echo "[!] No python3 or python found in PATH"; exit 1
# fi

# # --- Inputs ---
# BENIGN="full_pack/benign_preview.jsonl"
# ATTACKS="full_pack/compromised_preview.jsonl"
# MODEL="llama2-uncensored"
# # MODEL="llama2-uncensored:georgesung/llama2_7b_chat_uncensored"

# SLEEP_BENIGN="${SLEEP_BENIGN:-}"
# SLEEP_ATTACKS="${SLEEP_ATTACKS:-}"

# # --- Optional tools ---
# MPROF=$(command -v mprof || true)
# STRACE=$(command -v strace || true)
# TCPDUMP=$(command -v tcpdump || true)
# TSHARK=$(command -v tshark || true)
# NVSMI=$(command -v nvidia-smi || true)

# # --- Sessionization env for content-blind fingerprinting (CFP) ---
# export LOG_HMAC_KEY="${LOG_HMAC_KEY:-$(openssl rand -hex 32 2>/dev/null || echo deadbeef)}"
# export LOG_SALT="${LOG_SALT:-$(hostname || echo host)}"
# export CFP_BUCKET_SECS="${CFP_BUCKET_SECS:-300}"

# run_one() {
#   local label="$1" jsonl="$2" extra_sleep="$3"
#   local TS; TS=$(date +%Y%m%d-%H%M%S)
#   local LOG_DIR="$PWD/run_${label}_${TS}"
#   mkdir -p "$LOG_DIR"

#   local APP_LOG="$LOG_DIR/app.log"
#   local AGENT_JSON="$LOG_DIR/agent.jsonl"
#   local PCAP="$LOG_DIR/traffic.pcap"
#   local VMSTAT="$LOG_DIR/vmstat.log"
#   local GPUCSV="$LOG_DIR/gpu.csv"
#   local PSTATS="$LOG_DIR/profile.pstats"
#   local STRACE_BASE="$LOG_DIR/strace"

#   export AGENT_JSON_LOG="$AGENT_JSON"

#   env > "$LOG_DIR/env.txt" || true

#   # vmstat (best effort)
#   vmstat -t 1 > "$VMSTAT" & VMSTAT_PID=$! || VMSTAT_PID=

#   # GPU sampler (optional)
#   if [ -n "$NVSMI" ]; then
#     echo "timestamp,util.gpu %,mem.used MB,temp C,power W" > "$GPUCSV"
#     ( while true; do
#         nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used,temperature.gpu,power.draw \
#                    --format=csv,noheader >> "$GPUCSV"; sleep 1; done ) \
#       & GPU_PID=$!
#   else
#     GPU_PID=
#   fi

#   # Packet capture only if passwordless sudo is available
#   if [ -n "$TCPDUMP" ] && sudo -n true 2>/dev/null; then
#     sudo tcpdump -U -i any -w "$PCAP" not port 22 & TCPDUMP_PID=$!
#   else
#     echo "[i] Skipping tcpdump (no passwordless sudo)"; TCPDUMP_PID=
#   fi

#   echo "[*] launching generate_agent_logs.py for $label … (logs → $LOG_DIR)"

#   # Build the command: optional mprof/strace + cProfile
#   CMD=( "$PYTHON" -m cProfile -o "$PSTATS" generate_agent_logs.py
#         --model "$MODEL" --outdir "$LOG_DIR" $extra_sleep )
#   if [ "$label" = "benign" ];  then CMD+=( --benign "$jsonl" ); fi
#   if [ "$label" = "attacks" ]; then CMD+=( --attacks "$jsonl" ); fi

#   if [ -n "${STRACE:-}" ]; then
#     CMD=( "$STRACE" -qq -tt -ff -o "$STRACE_BASE" "${CMD[@]}" )
#   fi
#   if [ -n "$MPROF" ]; then
#     CMD=( "$MPROF" run "${CMD[@]}" )
#   fi

#   # Run and tee logs
#   set +e
#   "${CMD[@]}" |& tee "$APP_LOG"
#   status=${PIPESTATUS[0]}
#   set -e

#   # Post-process pcaps only if they exist
#   # --- per-agent post-processing (build bundles) ---
#   if [ -d "$LOG_DIR" ]; then
#     # If the app wrote to a separate runs/… dir, pull useful files into $LOG_DIR
#     if [ -f "$APP_LOG" ]; then
#       RUN_ART_DIR=$(grep -oE 'Writing run artifacts to: .*$' "$APP_LOG" | tail -n1 | sed 's/.*: //')
#       if [ -n "$RUN_ART_DIR" ] && [ -d "$RUN_ART_DIR" ]; then
#         for f in results.json traffic.pcap http.tsv dns.tsv; do
#           [ -e "$RUN_ART_DIR/$f" ] && ln -sf "$RUN_ART_DIR/$f" "$LOG_DIR/$f"
#         done
#       fi
#       # after the program finishes and app.log exists:
#       echo "[*] post: starting per-agent slicing …"
#       python3 tools/agent_jsonl_stats.py "$LOG_DIR" || true
#       python3 tools/pcap_per_agent.py    "$LOG_DIR" "$LOG_DIR/traffic.pcap" || true
#       python3 tools/slice_metric_per_agent.py "$LOG_DIR" || true
#       echo "[*] post: done"


#       # Always build per-agent bundles from app.log
#       "$PYTHON" tools/agent_jsonl_stats.py "$LOG_DIR" || true
#     fi

#     # Optional: summarize strace (unchanged)
#     if compgen -G "$STRACE_BASE.*" >/dev/null; then
#       [ -f tools/strace_stats.py ] && "$PYTHON" tools/strace_stats.py "$STRACE_BASE" "$LOG_DIR/strace_stats.tsv" || true
#     fi

#     # Optional: summarize traffic (unchanged)
#     if [ -f "$PCAP" ] && [ -n "$TSHARK" ] && [ -f tools/pcap_stats.py ]; then
#       "$PYTHON" tools/pcap_stats.py "$PCAP" "$LOG_DIR/traffic.tsv" || true
#     fi
#   fi



  
#   # --- per-agent post-processing (build bundles) ---
#   echo "[*] post: starting per-agent slicing …"
#   if [ -d "$LOG_DIR" ]; then
#     # Try to pull artifacts if the app wrote to ./runs/… (optional)
#     if [ -f "$APP_LOG" ]; then
#       # Robust scrape of the last "Writing run artifacts to:" line
#       RUN_ART_DIR=$(awk '/Writing run artifacts to:/ {last=$0} END{print last}' "$APP_LOG" | sed 's/.*to:[[:space:]]*//')
#       if [ -n "$RUN_ART_DIR" ] && [ -d "$RUN_ART_DIR" ]; then
#         echo "[*] post: linking run artifacts from $RUN_ART_DIR → $LOG_DIR"
#         for f in results.json traffic.pcap http.tsv dns.tsv; do
#           [ -e "$RUN_ART_DIR/$f" ] && ln -sf "$RUN_ART_DIR/$f" "$LOG_DIR/$f"
#         done
#       else
#         echo "[i] post: no external runs/ dir detected in app.log (ok)"
#       fi
#     fi

#     # Always slice app.log → per_agent/…
#     if [ -f "$APP_LOG" ]; then
#       echo "[*] post: slicing $APP_LOG"
#       set +e
#       "$PYTHON" tools/agent_jsonl_stats.py "$LOG_DIR"
#       slicer_rc=$?
#       set -e
#       if [ $slicer_rc -ne 0 ]; then
#         echo "[!] post: slicer failed with $slicer_rc — see $LOG_DIR for clues"
#       else
#         echo "[+] post: per-agent bundles → $LOG_DIR/per_agent"
#         ls -1 "$LOG_DIR/per_agent" 2>/dev/null || true
#       fi
#     else
#       echo "[!] post: no $APP_LOG found; cannot slice"
#     fi

#     # Optional summaries (unchanged)
#     if compgen -G "$STRACE_BASE.*" >/dev/null; then
#       [ -f tools/strace_stats.py ] && "$PYTHON" tools/strace_stats.py "$STRACE_BASE" "$LOG_DIR/strace_stats.tsv" || true
#     fi
#     if [ -f "$PCAP" ] && [ -n "$TSHARK" ] && [ -f tools/pcap_stats.py ]; then
#       "$PYTHON" tools/pcap_stats.py "$PCAP" "$LOG_DIR/traffic.tsv" || true
#     fi
#   fi
#   echo "[*] post: done"

#   [ -n "${VMSTAT_PID:-}" ] && kill "$VMSTAT_PID" 2>/dev/null || true
#   [ -n "${GPU_PID:-}"    ] && kill "$GPU_PID"    2>/dev/null || true

#   echo "[*] done ($label, exit $status) – artefacts in $LOG_DIR"
#   return $status
  
# }

# # Run benign; uncomment attacks if/when you want them too
# run_one "benign"  "$BENIGN"  "${SLEEP_BENIGN:+--sleep $SLEEP_BENIGN}"
# # run_one "attacks" "$ATTACKS" "${SLEEP_ATTACKS:+--sleep $SLEEP_ATTACKS}"


#!/usr/bin/env bash
set -euo pipefail

# --- Python detection ---
PYTHON=$(command -v python3 || true)
if [ -z "$PYTHON" ]; then PYTHON=$(command -v python || true); fi
if [ -z "$PYTHON" ]; then echo "[!] No python or python3 found in PATH"; exit 1; fi

# --- Inputs ---
BENIGN="full_pack/benign_remaining.jsonl"
ATTACKS="full_pack/compromised_preview.jsonl"
MODEL="llama2-uncensored"    # or: llama2-uncensored:georgesung/llama2_7b_chat_uncensored
SLEEP_BENIGN="${SLEEP_BENIGN:-}"
SLEEP_ATTACKS="${SLEEP_ATTACKS:-}"

# --- Optional tools ---
MPROF=$(command -v mprof || true)
STRACE=$(command -v strace || true)
TCPDUMP=$(command -v tcpdump || true)
TSHARK=$(command -v tshark || true)
NVSMI=$(command -v nvidia-smi || true)

# allow ENV overrides
BENIGN="${BENIGN:-full_pack/benign_remaining.jsonl}"
ATTACKS="${ATTACKS:-full_pack/compromised_preview.jsonl}"

# quiet HF warnings
export TRANSFORMERS_VERBOSITY=${TRANSFORMERS_VERBOSITY:-error}

run_one() {
  local label="$1" jsonl="$2" extra_sleep="$3"

  local TS; TS=$(date +%Y%m%d-%H%M%S)
  local LOG_DIR="$PWD/run_${label}_${TS}"
  mkdir -p "$LOG_DIR"

  local APP_LOG="$LOG_DIR/app.log"
  local AGENT_JSON="$LOG_DIR/agent.jsonl"
  local PCAP="$LOG_DIR/traffic.pcap"
  local VMSTAT="$LOG_DIR/vmstat.log"
  local GPUCSV="$LOG_DIR/gpu.csv"
  local PSTATS="$LOG_DIR/profile.pstats"
  local STRACE_BASE="$LOG_DIR/strace"
  local MEMPLOT="$LOG_DIR/mprof_$(date +%s).dat"

  # make per-message logger write here
  export AGENT_JSON_LOG="$AGENT_JSON"

  # snapshot env (best effort)
  env > "$LOG_DIR/env.txt" || true

  # collectors
  vmstat -t 1 > "$VMSTAT" & VMSTAT_PID=$! || VMSTAT_PID=
  if [ -n "$NVSMI" ]; then
    echo "timestamp,util.gpu %,mem.used MB,temp C,power W" > "$GPUCSV"
    ( while true; do
        nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used,temperature.gpu,power.draw \
                   --format=csv,noheader >> "$GPUCSV"; sleep 1; done ) & GPU_PID=$!
  else
    GPU_PID=
  fi

  if [ -n "$TCPDUMP" ]; then
    echo "[*] tcpdump: requesting sudo (for capture only)…"
    if sudo -v; then
      sudo tcpdump -U -i any -w "$PCAP" not port 22 & TCPDUMP_PID=$!
    else
      echo "[i] Skipping tcpdump (sudo auth failed)"; TCPDUMP_PID=
    fi
  else
    TCPDUMP_PID=
  fi

  echo "[*] launching generate_agent_logs.py for $label … (logs → $LOG_DIR)"

  # build command (cProfile; optional strace/mprof)
  CMD=( "$PYTHON" -m cProfile -o "$PSTATS" generate_agent_logs.py
        --model "$MODEL" --outdir "$LOG_DIR" $extra_sleep )
  if [ "$label" = "benign" ];  then CMD+=( --benign "$jsonl" ); fi
  if [ "$label" = "attacks" ]; then CMD+=( --attacks "$jsonl" ); fi

  if [ -n "$STRACE" ]; then
    CMD=( "$STRACE" -qq -tt -ff -o "$STRACE_BASE" "${CMD[@]}" )
  fi
  if [ -n "$MPROF" ]; then
    CMD=( "$MPROF" run -o "$MEMPLOT" "${CMD[@]}" )
  fi

  # Run (CRUCIAL: force core.py to write inside $LOG_DIR)
  export TRACEPARENT="${TRACEPARENT:-00-$(head -c16 /dev/urandom | xxd -p)-$(head -c8 /dev/urandom | xxd -p)-01}"
  export TRACESTATE="${TRACESTATE:-vendor=demo}"
  if command -v sha256sum >/dev/null 2>&1; then
    CFP_DEFAULT="$(printf '%s' "${LOG_HMAC_KEY:-dead}:${LOG_SALT:-host}:$(( $(date +%s) / ${CFP_BUCKET_SECS:-300} ))" \
                  | sha256sum | awk '{print $1}' | cut -c1-16)"
  else
    CFP_DEFAULT="$(printf '%s' "${LOG_HMAC_KEY:-dead}:${LOG_SALT:-host}:$(( $(date +%s) / ${CFP_BUCKET_SECS:-300} ))" \
                  | shasum -a 256 | awk '{print $1}' | cut -c1-16)"
  fi
  export CFP="${CFP:-$CFP_DEFAULT}"


  # ---- RUN_LOG_DIR injection (also fixes #2 below) ----
  set +e
  RUN_LOG_DIR="$LOG_DIR" "${CMD[@]}" |& tee "$APP_LOG"
  status=${PIPESTATUS[0]}
  set -e

  # quick packet summaries
  if [ -n "$TCPDUMP_PID" ] && [ -n "$TSHARK" ]; then
    $TSHARK -r "$PCAP" -Y http \
      -T fields -e frame.time -e ip.dst -e http.request.full_uri > "$LOG_DIR/http.tsv" || true
    $TSHARK -r "$PCAP" -Y "dns.flags.response==0" \
      -T fields -e frame.time -e dns.qry.name > "$LOG_DIR/dns.tsv" || true
  fi

  # post-processing
  if [ -f "$APP_LOG" ]; then
    # If your app still printed an external runs/… path, link the goodies
    RUN_ART_DIR=$(awk '/Writing run artifacts to:/ {last=$0} END{print last}' "$APP_LOG" | sed 's/.*to:[[:space:]]*//')
    if [ -n "$RUN_ART_DIR" ] && [ -d "$RUN_ART_DIR" ] && [ "$RUN_ART_DIR" != "$LOG_DIR" ]; then
      echo "[*] post: linking run artifacts from $RUN_ART_DIR → $LOG_DIR"
      for f in results.json traffic.pcap http.tsv dns.tsv; do
        [ -e "$RUN_ART_DIR/$f" ] && ln -sf "$RUN_ART_DIR/$f" "$LOG_DIR/$f"
      done
    fi

    "$PYTHON" tools/agent_jsonl_stats.py "$LOG_DIR" || true
    [ -f "$PCAP" ] && "$PYTHON" tools/pcap_per_agent.py "$LOG_DIR" "$PCAP" || true
    [ -f "$PCAP" ] && "$PYTHON" tools/slice_metric_per_agent.py "$LOG_DIR" || true
  fi

  if compgen -G "$STRACE_BASE.*" >/dev/null; then
    [ -f tools/strace_stats.py ] && "$PYTHON" tools/strace_stats.py "$STRACE_BASE" "$LOG_DIR/strace_stats.tsv" || true
  fi
  if [ -f "$PCAP" ] && [ -n "$TSHARK" ] && [ -f tools/pcap_stats.py ]; then
    "$PYTHON" tools/pcap_stats.py "$PCAP" "$LOG_DIR/traffic.tsv" || true
  fi

  # cleanup
  [ -n "${VMSTAT_PID:-}" ] && kill "$VMSTAT_PID" 2>/dev/null || true
  [ -n "${GPU_PID:-}"    ] && kill "$GPU_PID"    2>/dev/null || true
  [ -n "${TCPDUMP_PID:-}" ] && sudo kill -2 "$TCPDUMP_PID" 2>/dev/null || true

  echo "[*] done ($label, exit $status) – artefacts in $LOG_DIR"
  return $status
}

# Run benign; uncomment attacks if you want those too
# run_one "benign"  "$BENIGN"  "${SLEEP_BENIGN:+--sleep $SLEEP_BENIGN}"
run_one "attacks" "$ATTACKS" "${SLEEP_ATTACKS:+--sleep $SLEEP_ATTACKS}"
