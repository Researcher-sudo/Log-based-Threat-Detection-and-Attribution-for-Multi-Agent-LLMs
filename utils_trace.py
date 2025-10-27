#!/usr/bin/env python3
import os, re, time, hmac, hashlib, secrets, json
from datetime import datetime, timezone
from pathlib import Path

# --- CFP/sessionization config from env (already exported by your run script) ---
CFP_BUCKET_SECS = int(os.getenv("CFP_BUCKET_SECS", "300"))
_raw_key = os.getenv("LOG_HMAC_KEY", "")
LOG_HMAC_KEY = (bytes.fromhex(_raw_key) if re.fullmatch(r"[0-9a-f]{64}", _raw_key or "", re.I)
                else (_raw_key or "default")).encode("utf-8")
LOG_SALT = os.getenv("LOG_SALT", "host").encode("utf-8")

def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")

def mint_trace_ids():
    # 128-bit trace + 64-bit span, W3C v00
    trace_id = secrets.token_hex(16)   # 32 hex
    span_id  = secrets.token_hex(8)    # 16 hex
    traceparent = f"00-{trace_id}-{span_id}-01"
    return trace_id, traceparent, ""

def _sha256hex(s: str) -> str:
    return hashlib.sha256((s or "").encode("utf-8")).hexdigest()

def cfp_hash(api_key:str="", route:str="", org_id:str="") -> str:
    now = int(time.time())
    bucket = now - (now % CFP_BUCKET_SECS)
    msg = f"{bucket}|{_sha256hex(api_key)}|{_sha256hex(route)}|{_sha256hex(org_id)}".encode("utf-8")
    return hmac.new(LOG_HMAC_KEY, LOG_SALT + msg, hashlib.sha256).hexdigest()[:16]

def append_jsonl(path: Path, row: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush(); os.fsync(f.fileno())

def log_edge_message(outdir: Path,
                     round_i: int, turn_i: int,
                     src: dict, dst: dict,
                     message: str,
                     model, tokenizer,
                     decode_args: dict,
                     prompt_ids, completion_ids,
                     elapsed_ms: float,
                     headers: dict, trace: dict):
    # Torch is optional (cpu fallback ok)
    try:
        import torch
        cuda_ok = torch.cuda.is_available()
        dev_name = torch.cuda.get_device_name(0) if cuda_ok else "cpu"
        vram_gb = int(torch.cuda.get_device_properties(0).total_memory / 1024**3) if cuda_ok else 0
        device = "cuda:0" if cuda_ok else "cpu"
    except Exception:
        device, dev_name, vram_gb = "cpu", "cpu", 0

    model_id = getattr(getattr(model, "config", None), "_name_or_path", type(model).__name__)
    try:
        p_count = int(prompt_ids.numel())  # torch tensor
    except Exception:
        try:
            p_count = int(prompt_ids.shape[-1])
        except Exception:
            p_count = len(prompt_ids)

    try:
        c_count = int(completion_ids.numel())
    except Exception:
        try:
            c_count = int(completion_ids.shape[-1])
        except Exception:
            c_count = len(completion_ids)

    meta = {
        "model_id": model_id,
        "device": device,
        "device_name": dev_name,
        "vram_gb": vram_gb,
        "decode": decode_args,
        "timing_ms": int(elapsed_ms),
        "tokens": {"prompt": p_count, "completion": c_count},
        "pid": os.getpid(),
        "headers": headers,
        "trace": trace,
    }
    row = {
        "type": "edge_message",
        "ts": utc_now(),
        "round": int(round_i),
        "turn": int(turn_i),
        "src": src,
        "dst": dst,
        "message": message,
        "direction": "sent",
        "meta": meta,
    }

    outdir = Path(outdir)
    append_jsonl(outdir / "interactions.jsonl", row)
    # per-sender file
    append_jsonl(outdir / f"agent_{src['idx']:02d}_{src['role']}.jsonl", row)
