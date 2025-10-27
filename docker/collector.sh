#!/usr/bin/env bash
set -euo pipefail


MODEL="${MODEL:-llama2-uncensored:georgesung/llama2_7b_chat_uncensored}"
LOG_DIR="${LOG_DIR:-/logs}"
PROFILE="${PROFILE:-1}"

# sensible defaults if not provided by .env.benign
BENIGN="${BENIGN:-/app/preview_pack/benign_preview.jsonl}"
ATTACKS="${ATTACKS:-/app/preview_pack/compromised_preview.jsonl}"
SLEEP="${SLEEP:-0}"

if [[ "$PROFILE" == "1" ]]; then
  CMD=( python3 -u -m cProfile -o "$LOG_DIR/profile.pstats"
        /app/generate_agent_logs.py
        --model "$MODEL" --benign "$BENIGN" --attacks "$ATTACKS" --outdir "$LOG_DIR" --sleep "$SLEEP" )
else
  CMD=( python3 -u /app/generate_agent_logs.py
        --model "$MODEL" --benign "$BENIGN" --attacks "$ATTACKS" --outdir "$LOG_DIR" --sleep "$SLEEP" )
fi

echo "[collector] exec:" "${CMD[@]}"
exec "${CMD[@]}"
