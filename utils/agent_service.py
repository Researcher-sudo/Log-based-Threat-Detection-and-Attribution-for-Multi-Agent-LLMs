# #V1: This version was correct and was working
# import argparse, json, os, socket, sys, time, threading, traceback
# from pathlib import Path

# ROLE = os.environ.get("ROLE_NAME","agent")
# LOG_PATH = os.environ.get("AGENT_JSON_LOG","")

# _lock = threading.Lock()
# _t0 = time.time()

# def log_emit(kind, **kv):
#     """Minimal JSONL logger (per-agent)."""
#     if not LOG_PATH:
#         return
#     kv.update(kind=kind,
#               ts=time.time(),
#               t_rel_ms=int((time.time()-_t0)*1000),
#               role=ROLE,
#               pid=os.getpid())
#     line = json.dumps(kv, ensure_ascii=False)
#     with _lock, open(LOG_PATH, "a", encoding="utf-8") as fp:
#         fp.write(line + "\n")

# _PIPE = None
# def get_pipe():
#     """Lazy-load the HF pipeline on first request."""
#     global _PIPE
#     if _PIPE is not None:
#         return _PIPE
#     from transformers import pipeline
#     model_id = os.environ.get("MODEL_ID", "georgesung/llama2_7b_chat_uncensored")
#     log_emit("model_load_begin", model_id=model_id)
#     _PIPE = pipeline(
#         "text-generation",
#         model=model_id,
#         device_map="auto",
#         torch_dtype="auto",
#         model_kwargs={"attn_implementation": "sdpa"},
#         max_new_tokens=128,
#         temperature=0.7,
#     )
#     log_emit("model_load_done", model_id=model_id)
#     return _PIPE

# def handle(line:str)->dict:
#     try:
#         req = json.loads(line)
#     except Exception as e:
#         return {"ok": False, "error": f"bad_json:{e}"}

#     # health check
#     if req.get("ping"):
#         return {"ok": True, "pong": True}

#     text = (req.get("text") or "").strip()
#     if not text:
#         return {"ok": False, "error": "empty_text"}

#     log_emit("agent_msg", direction="in", text=text)
#     t0 = time.time()
#     try:
#         out = get_pipe()(text, do_sample=False)[0]["generated_text"]
#         dt = int((time.time()-t0)*1000)
#         log_emit("agent_msg", direction="out", text=out, ms=dt)
#         return {"ok": True, "text": out, "elapsed_ms": dt}
#     except Exception as e:
#         tb = traceback.format_exc(limit=3)
#         log_emit("run_error", err=str(e), trace=tb)
#         return {"ok": False, "error": str(e)}

# def serve(sock_path:str):
#     p = Path(sock_path)
#     p.parent.mkdir(parents=True, exist_ok=True)
#     if p.exists():
#         p.unlink()
#     s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
#     s.bind(sock_path)
#     os.chmod(sock_path, 0o777)  # easy host access
#     s.listen(8)
#     print(f"[agent_service] role={ROLE} listening on {sock_path}", flush=True)
#     log_emit("server_start", sock=sock_path)
#     try:
#         while True:
#             conn,_ = s.accept()
#             f = conn.makefile("rwb")
#             try:
#                 line = f.readline().decode().strip()
#                 if not line:
#                     f.write(b'{"ok":false,"error":"empty_line"}\n'); f.flush()
#                 else:
#                     resp = handle(line)
#                     f.write((json.dumps(resp, ensure_ascii=False)+"\n").encode()); f.flush()
#             except Exception as e:
#                 tb = traceback.format_exc(limit=2)
#                 f.write((json.dumps({"ok":False,"error":str(e),"trace":tb})+"\n").encode()); f.flush()
#             finally:
#                 conn.close()
#     finally:
#         s.close()
#         if p.exists():
#             p.unlink()

# def main():
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--role", required=True)
#     ap.add_argument("--sock", required=True)
#     args = ap.parse_args()
#     global ROLE; ROLE = args.role
#     if LOG_PATH:
#         print(f"[agent_service] logging to {LOG_PATH}", flush=True)
#     serve(args.sock)

# if __name__ == "__main__":
#     main()


# #V3:
# # utils/agent_service.py
# import argparse, json, os, socket, sys, time, threading, traceback, hmac, hashlib, base64, pathlib

# # optional psutil (skip if not installed)
# try:
#     import psutil  # type: ignore
# except Exception:
#     psutil = None

# ROLE         = os.environ.get("ROLE_NAME", "agent")
# LOG_PATH     = os.environ.get("AGENT_JSON_LOG", "")
# CONTAINER_ID = os.getenv("HOSTNAME", "")
# MODEL_ID     = os.environ.get("MODEL_ID", "georgesung/llama2_7b_chat_uncensored")

# # CFP inputs (PII-safe)
# HMAC_KEY = os.getenv("CFP_HMAC_KEY", "").encode() if os.getenv("CFP_HMAC_KEY") else b""
# ORG_ID   = os.getenv("ORG_ID", "")
# API_KEY  = os.getenv("API_KEY", "")
# # if only *_HASH are set, use them directly
# ORG_ID_HASH = os.getenv("ORG_ID_HASH", "")
# API_KEY_HASH = os.getenv("API_KEY_HASH", "")

# def _hash(s: str) -> str:
#     return hashlib.sha256(s.encode()).hexdigest()

# def _time_bucket(seconds=300) -> int:
#     return int(time.time() // seconds)

# def compute_cfp() -> str:
#     """Content-blind fingerprint: HMAC(timebucket|API|ROUTE|ORG)."""
#     if not HMAC_KEY:
#         return ""
#     tb = str(_time_bucket())
#     akh = _hash(API_KEY) if API_KEY else (API_KEY_HASH or "")
#     orh = _hash(ORG_ID)  if ORG_ID  else (ORG_ID_HASH  or "")
#     rth = _hash(os.environ.get("ROLE_NAME","route"))  # stable per role
#     msg = "|".join([tb, akh, rth, orh]).encode()
#     mac = hmac.new(HMAC_KEY, msg, hashlib.sha256).digest()
#     return base64.b64encode(mac).decode()

# # ── metrics helpers ──────────────────────────────────────────────────────────────
# def _rss_mb():
#     try:
#         with open("/proc/self/status","r",encoding="utf-8",errors="ignore") as f:
#             for line in f:
#                 if line.startswith("VmRSS:"):
#                     kb = int(line.split()[1])
#                     return round(kb/1024, 3)
#     except Exception:
#         return None

# def _net_bytes():
#     total_rx = 0
#     total_tx = 0
#     base = pathlib.Path("/sys/class/net")
#     try:
#         for iface in base.iterdir():
#             try:
#                 with open(iface/"statistics/rx_bytes") as rxf: total_rx += int(rxf.read().strip())
#                 with open(iface/"statistics/tx_bytes") as txf: total_tx += int(txf.read().strip())
#             except Exception:
#                 continue
#     except Exception:
#         pass
#     return total_rx, total_tx

# # ── tiny logger (JSONL) with per-thread correlation context ─────────────────────
# _lock = threading.Lock()
# _t0 = time.time()
# _thread_ctx = threading.local()

# def _set_ctx(ctx: dict): setattr(_thread_ctx, "ctx", ctx)
# def _get_ctx() -> dict:   return getattr(_thread_ctx, "ctx", {})

# def log_emit(kind: str, **kv):
#     if not LOG_PATH:
#         return
#     base = dict(
#         kind=kind, ts=time.time(), t_rel_ms=int((time.time()-_t0)*1000),
#         role=ROLE, pid=os.getpid(), container_id=CONTAINER_ID, model_id=MODEL_ID
#     )
#     base.update(_get_ctx())
#     base.update(kv)
#     line = json.dumps(base, ensure_ascii=False)
#     with _lock, open(LOG_PATH, "a", encoding="utf-8") as fp:
#         fp.write(line + "\n")

# # ── lazy HF pipeline ────────────────────────────────────────────────────────────
# _PIPE = None
# def get_pipe():
#     global _PIPE
#     if _PIPE is not None:
#         return _PIPE
#     from transformers import pipeline  # lazy import keeps startup fast
#     log_emit("model_load_begin", model_id=MODEL_ID)
#     _PIPE = pipeline(
#         "text-generation",
#         model=MODEL_ID,
#         device_map="auto",
#         torch_dtype="auto",
#         model_kwargs={"attn_implementation": "sdpa"},
#         max_new_tokens=128,
#         temperature=0.7,
#         return_full_text=True,
#     )
#     log_emit("model_load_done", model_id=MODEL_ID)
#     return _PIPE

# # ── core request ────────────────────────────────────────────────────────────────
# def handle_request(req: dict) -> dict:
#     # accept reset/trace fields; we don't keep chat state, so reset is a no-op
#     text = (req.get("text") or "").strip()
#     if req.get("ping"):
#         return {"ok": True, "pong": True}
#     if not text:
#         return {"ok": False, "error": "empty_text"}

#     # correlation (best effort)
#     trace_id       = (req.get("trace_id") or "")[:32]
#     pipeline_id    = (req.get("pipeline_id") or "")[:32]
#     span_id        = (req.get("span_id") or "")[:16]
#     parent_span_id = (req.get("parent_span_id") or "")[:16]
#     seq            = int(req.get("seq", 0))
#     traceparent    = f"00-{trace_id or '0'*32}-{span_id or '0'*16}-01"
#     cfp            = compute_cfp()
#     _set_ctx(dict(
#         trace_id=trace_id, pipeline_id=pipeline_id, span_id=span_id,
#         parent_span_id=parent_span_id, seq=seq, traceparent=traceparent, cfp=cfp
#     ))

#     # metrics (before)
#     rss0 = _rss_mb()
#     rx0, tx0 = _net_bytes()
#     p = psutil.Process() if psutil else None
#     if p: _ = p.cpu_percent(None)  # prime

#     log_emit("agent_msg", direction="in", text=text)
#     t0 = time.time()
#     try:
#         out = get_pipe()(text, do_sample=False)[0]["generated_text"]
#         dt_ms = int((time.time()-t0)*1000)
#         log_emit("agent_msg", direction="out", text=out, ms=dt_ms)
#         ok = True; err = None
#     except Exception as e:
#         out = ""; ok = False; err = str(e)
#         dt_ms = int((time.time()-t0)*1000)
#         log_emit("run_error", err=err, ms=dt_ms)

#     # metrics (after)
#     rss1 = _rss_mb()
#     rx1, tx1 = _net_bytes()
#     cpu_pct = (p.cpu_percent(None) if p else None)
#     log_emit("sys_metrics",
#              rss_mb_before=rss0, rss_mb_after=rss1,
#              net_rx_bytes_delta=(rx1-rx0) if (rx1 and rx0) else None,
#              net_tx_bytes_delta=(tx1-tx0) if (tx1 and tx0) else None,
#              cpu_pct_process=cpu_pct)

#     if ok:
#         return {"ok": True, "text": out, "elapsed_ms": dt_ms}
#     else:
#         return {"ok": False, "error": err, "elapsed_ms": dt_ms}

# # ── server ─────────────────────────────────────────────────────────────────────
# def serve(sock_path: str):
#     p = pathlib.Path(sock_path)
#     p.parent.mkdir(parents=True, exist_ok=True)
#     if p.exists():
#         p.unlink()
#     s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
#     s.bind(sock_path)
#     os.chmod(sock_path, 0o777)
#     s.listen(16)
#     print(f"[agent_service] role={ROLE} listening on {sock_path}", flush=True)
#     log_emit("server_start", sock=str(p))

#     try:
#         while True:
#             conn, _ = s.accept()
#             f = conn.makefile("rwb")
#             try:
#                 line = f.readline().decode("utf-8","ignore").strip()
#                 if not line:
#                     f.write(b'{"ok":false,"error":"empty_line"}\n'); f.flush()
#                     continue
#                 try:
#                     req = json.loads(line)
#                 except Exception as e:
#                     log_emit("bad_json", error=str(e), raw=line[:512])
#                     f.write((json.dumps({"ok":False,"error":f"bad_json:{e}"})+"\n").encode()); f.flush()
#                     continue
#                 resp = handle_request(req)
#                 f.write((json.dumps(resp, ensure_ascii=False) + "\n").encode()); f.flush()
#             except Exception as e:
#                 tb = traceback.format_exc(limit=2)
#                 log_emit("server_error", error=str(e), trace=tb)
#                 try:
#                     f.write((json.dumps({"ok":False,"error":str(e),"trace":tb})+"\n").encode()); f.flush()
#                 except Exception:
#                     pass
#             finally:
#                 try: conn.close()
#                 except Exception: pass
#     finally:
#         s.close()
#         try: p.unlink()
#         except Exception: pass

# def main():
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--role", required=True)
#     ap.add_argument("--sock", required=True)
#     args = ap.parse_args()
#     global ROLE; ROLE = args.role
#     if LOG_PATH:
#         print(f"[agent_service] logging to {LOG_PATH}", flush=True)
#     serve(args.sock)

# if __name__ == "__main__":
#     main()


#V4:
# utils/agent_service.py  (V3 → V4: uncapped gen, no prompt echo, richer metrics)
import argparse, json, os, socket, sys, time, threading, traceback, hmac, hashlib, base64, pathlib

# optional psutil (skip if not installed)
try:
    import psutil  # type: ignore
except Exception:
    psutil = None

ROLE         = os.environ.get("ROLE_NAME", "agent")
LOG_PATH     = os.environ.get("AGENT_JSON_LOG", "")
CONTAINER_ID = os.getenv("HOSTNAME", "")
MODEL_ID     = os.environ.get("MODEL_ID", "georgesung/llama2_7b_chat_uncensored")

# ---- Optional generation knobs (ALL are opt-in; unset => not passed) ------------------
# If you leave these unset, we do NOT pass caps to the pipeline call.
# NOTE: 🤖 Transformers has its own defaults (often ~20 tokens) if nothing is set.
#       If you truly want "no practical cap", set GEN_MAX_NEW_TOKENS to something big,
#       or rely on provider-side EOS/stop sequences.
GEN_MAX_NEW_TOKENS = os.getenv("GEN_MAX_NEW_TOKENS")  # int if set
GEN_MAX_TIME_S     = os.getenv("GEN_MAX_TIME_S")      # float if set
GEN_DO_SAMPLE      = os.getenv("GEN_DO_SAMPLE")       # "1"/"true" to enable
GEN_TEMPERATURE    = os.getenv("GEN_TEMPERATURE")     # float if set

def _as_bool(x: str) -> bool:
    return str(x).strip().lower() in ("1","true","t","yes","y")

# ---- CFP inputs (PII-safe) ------------------------------------------------------------
HMAC_KEY = os.getenv("CFP_HMAC_KEY", "").encode() if os.getenv("CFP_HMAC_KEY") else b""
ORG_ID   = os.getenv("ORG_ID", "")
API_KEY  = os.getenv("API_KEY", "")
# if only *_HASH are set, use them directly
ORG_ID_HASH  = os.getenv("ORG_ID_HASH", "")
API_KEY_HASH = os.getenv("API_KEY_HASH", "")

def _hash(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()

def _time_bucket(seconds=300) -> int:
    return int(time.time() // seconds)

def compute_cfp() -> str:
    """Content-blind fingerprint: HMAC(timebucket|API|ROUTE|ORG)."""
    if not HMAC_KEY:
        return ""
    tb = str(_time_bucket())
    akh = _hash(API_KEY) if API_KEY else (API_KEY_HASH or "")
    orh = _hash(ORG_ID)  if ORG_ID  else (ORG_ID_HASH  or "")
    rth = _hash(os.environ.get("ROLE_NAME","route"))  # stable per role
    msg = "|".join([tb, akh, rth, orh]).encode()
    mac = hmac.new(HMAC_KEY, msg, hashlib.sha256).digest()
    return base64.b64encode(mac).decode()

# ── metrics helpers ────────────────────────────────────────────────────────────────────
def _rss_mb():
    try:
        with open("/proc/self/status","r",encoding="utf-8",errors="ignore") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    kb = int(line.split()[1])
                    return round(kb/1024, 3)
    except Exception:
        return None

def _net_bytes():
    total_rx = 0
    total_tx = 0
    base = pathlib.Path("/sys/class/net")
    try:
        for iface in base.iterdir():
            try:
                with open(iface/"statistics/rx_bytes") as rxf: total_rx += int(rxf.read().strip())
                with open(iface/"statistics/tx_bytes") as txf: total_tx += int(txf.read().strip())
            except Exception:
                continue
    except Exception:
        pass
    return total_rx, total_tx

# ── tiny logger (JSONL) with per-thread correlation context ────────────────────────────
_lock = threading.Lock()
_t0 = time.time()
_thread_ctx = threading.local()

def _set_ctx(ctx: dict): setattr(_thread_ctx, "ctx", ctx)
def _get_ctx() -> dict:   return getattr(_thread_ctx, "ctx", {})

def log_emit(kind: str, **kv):
    if not LOG_PATH:
        return
    base = dict(
        kind=kind, ts=time.time(), t_rel_ms=int((time.time()-_t0)*1000),
        role=ROLE, pid=os.getpid(), container_id=CONTAINER_ID, model_id=MODEL_ID
    )
    base.update(_get_ctx())
    base.update(kv)
    line = json.dumps(base, ensure_ascii=False)
    with _lock, open(LOG_PATH, "a", encoding="utf-8") as fp:
        fp.write(line + "\n")

# ── lazy HF pipeline ───────────────────────────────────────────────────────────────────
_PIPE = None
def get_pipe():
    """
    IMPORTANT:
      - We do NOT set max_new_tokens here.
      - We do NOT set return_full_text here.
      Those are passed at *call time* so we can keep the default uncapped behavior
      (unless you opt-in via env).
    """
    global _PIPE
    if _PIPE is not None:
        return _PIPE
    from transformers import pipeline  # lazy import keeps startup fast
    log_emit("model_load_begin", model_id=MODEL_ID)
    _PIPE = pipeline(
        "text-generation",
        model=MODEL_ID,
        device_map="auto",
        torch_dtype="auto",
        model_kwargs={"attn_implementation": "sdpa"},
    )
    log_emit("model_load_done", model_id=MODEL_ID)
    return _PIPE

# ── core request ───────────────────────────────────────────────────────────────────────
def handle_request(req: dict) -> dict:
    # accept reset/trace fields; we don't keep chat state, so reset is a no-op
    text = (req.get("text") or "").strip()
    if req.get("ping"):
        return {"ok": True, "pong": True}
    if not text:
        return {"ok": False, "error": "empty_text"}

    # correlation (best effort)
    trace_id       = (req.get("trace_id") or "")[:32]
    pipeline_id    = (req.get("pipeline_id") or "")[:32]
    span_id        = (req.get("span_id") or "")[:16]
    parent_span_id = (req.get("parent_span_id") or "")[:16]
    seq            = int(req.get("seq", 0))
    traceparent    = req.get("traceparent") or f"00-{trace_id or '0'*32}-{span_id or '0'*16}-01"
    tracestate     = req.get("tracestate") or ""
    cfp            = compute_cfp()
    _set_ctx(dict(
        trace_id=trace_id, pipeline_id=pipeline_id, span_id=span_id,
        parent_span_id=parent_span_id, seq=seq, traceparent=traceparent,
        tracestate=tracestate, cfp=cfp
    ))

    # metrics (before)
    rss0 = _rss_mb()
    rx0, tx0 = _net_bytes()
    p = psutil.Process() if psutil else None
    if p: _ = p.cpu_percent(None)  # prime

    log_emit("agent_msg", direction="in", text=text)

    # ---- build generation kwargs (opt-in caps only) -----------------------------------
    gen_kwargs = {
        "return_full_text": False,  # don't echo prompt
        "do_sample": _as_bool(GEN_DO_SAMPLE) if GEN_DO_SAMPLE is not None else False,
    }
    if GEN_TEMPERATURE is not None and _as_bool(GEN_DO_SAMPLE or "0"):
        try:
            gen_kwargs["temperature"] = float(GEN_TEMPERATURE)
        except Exception:
            pass
    if GEN_MAX_NEW_TOKENS is not None:
        try:
            gen_kwargs["max_new_tokens"] = int(GEN_MAX_NEW_TOKENS)
        except Exception:
            pass
    if GEN_MAX_TIME_S is not None:
        try:
            gen_kwargs["max_time"] = float(GEN_MAX_TIME_S)
        except Exception:
            pass

    # ---- run generation ---------------------------------------------------------------
    t0 = time.time()
    ok, out_text, err = True, "", None
    try:
        pipe = get_pipe()
        out = pipe(text, **gen_kwargs)
        out_text = (out[0].get("generated_text") or "").strip()
        dt_ms = int((time.time()-t0)*1000)
        log_emit("agent_msg", direction="out", text=out_text, ms=dt_ms)
    except Exception as e:
        ok = False
        err = str(e)
        dt_ms = int((time.time()-t0)*1000)
        log_emit("run_error", err=err, ms=dt_ms)

    # metrics (after)
    rss1 = _rss_mb()
    rx1, tx1 = _net_bytes()
    cpu_pct = (p.cpu_percent(None) if p else None)
    log_emit("sys_metrics",
             rss_mb_before=rss0, rss_mb_after=rss1,
             net_rx_bytes_delta=(rx1-rx0) if (rx1 is not None and rx0 is not None) else None,
             net_tx_bytes_delta=(tx1-tx0) if (tx1 is not None and tx0 is not None) else None,
             cpu_pct_process=cpu_pct)

    # token stats (best-effort)
    prompt_tokens = output_tokens = total_tokens = None
    try:
        tok = getattr(get_pipe(), "tokenizer", None)
        if tok is not None:
            prompt_tokens = len(tok.encode(text, add_special_tokens=False))
            output_tokens = len(tok.encode(out_text, add_special_tokens=False))
            total_tokens  = (prompt_tokens or 0) + (output_tokens or 0)
    except Exception:
        pass

    # response payload
    if ok:
        resp = {
            "ok": True,
            "text": out_text,
            "elapsed_ms": dt_ms,
            "stats": {
                "prompt_tokens": prompt_tokens,
                "output_tokens": output_tokens,
                "total_tokens": total_tokens,
            },
            "trace_id": trace_id,
            "pipeline_id": pipeline_id,
            "span_id": span_id,
            "parent_span_id": parent_span_id,
            "seq": seq,
            "traceparent": traceparent,
            "tracestate": tracestate,
            "cfp": cfp,
        }
    else:
        resp = {
            "ok": False,
            "error": err,
            "elapsed_ms": dt_ms,
            "trace_id": trace_id,
            "pipeline_id": pipeline_id,
            "span_id": span_id,
            "parent_span_id": parent_span_id,
            "seq": seq,
            "traceparent": traceparent,
            "tracestate": tracestate,
            "cfp": cfp,
        }
    return resp

# ── server ─────────────────────────────────────────────────────────────────────────────
def serve(sock_path: str):
    p = pathlib.Path(sock_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.exists():
        p.unlink()
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.bind(sock_path)
    os.chmod(sock_path, 0o777)
    s.listen(16)
    print(f"[agent_service] role={ROLE} listening on {sock_path}", flush=True)
    log_emit("server_start", sock=str(p))

    try:
        while True:
            conn, _ = s.accept()
            f = conn.makefile("rwb")
            try:
                line = f.readline().decode("utf-8","ignore").strip()
                if not line:
                    f.write(b'{"ok":false,"error":"empty_line"}\n'); f.flush()
                    continue
                try:
                    req = json.loads(line)
                except Exception as e:
                    log_emit("bad_json", error=str(e), raw=line[:512])
                    f.write((json.dumps({"ok":False,"error":f"bad_json:{e}"})+"\n").encode()); f.flush()
                    continue
                resp = handle_request(req)
                f.write((json.dumps(resp, ensure_ascii=False) + "\n").encode()); f.flush()
            except Exception as e:
                tb = traceback.format_exc(limit=2)
                log_emit("server_error", error=str(e), trace=tb)
                try:
                    f.write((json.dumps({"ok":False,"error":str(e),"trace":tb})+"\n").encode()); f.flush()
                except Exception:
                    pass
            finally:
                try: conn.close()
                except Exception: pass
    finally:
        s.close()
        try: p.unlink()
        except Exception: pass

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--role", required=True)
    ap.add_argument("--sock", required=True)
    args = ap.parse_args()
    global ROLE; ROLE = args.role
    if LOG_PATH:
        print(f"[agent_service] logging to {LOG_PATH}", flush=True)
    serve(args.sock)

if __name__ == "__main__":
    main()

# #VCPU + GPU:
# # utils/agent_service.py  (V4 – GPU/CPU by env, safe I/O, long outputs)
# import argparse, json, os, socket, sys, time, threading, traceback, hmac, hashlib, base64, pathlib

# try:
#     import psutil  # optional
# except Exception:
#     psutil = None

# ROLE         = os.environ.get("ROLE_NAME", "agent")
# LOG_PATH     = os.environ.get("AGENT_JSON_LOG", "")
# CONTAINER_ID = os.getenv("HOSTNAME", "")
# MODEL_ID     = os.environ.get("MODEL_ID", "georgesung/llama2_7b_chat_uncensored")

# # === Fingerprint (optional) ============================================================
# HMAC_KEY    = os.getenv("CFP_HMAC_KEY", "").encode() if os.getenv("CFP_HMAC_KEY") else b""
# ORG_ID      = os.getenv("ORG_ID", "")
# API_KEY     = os.getenv("API_KEY", "")
# ORG_ID_HASH = os.getenv("ORG_ID_HASH", "")
# API_KEY_HASH= os.getenv("API_KEY_HASH", "")

# def _hash(s: str) -> str: return hashlib.sha256(s.encode()).hexdigest()
# def _time_bucket(seconds=300) -> int: return int(time.time() // seconds)
# def compute_cfp() -> str:
#     if not HMAC_KEY: return ""
#     tb  = str(_time_bucket())
#     akh = _hash(API_KEY) if API_KEY else (API_KEY_HASH or "")
#     orh = _hash(ORG_ID)  if ORG_ID  else (ORG_ID_HASH  or "")
#     rth = _hash(os.environ.get("ROLE_NAME","route"))
#     mac = hmac.new(HMAC_KEY, f"{tb}|{akh}|{rth}|{orh}".encode(), hashlib.sha256).digest()
#     return base64.b64encode(mac).decode()

# # === Metrics/helpers ==================================================================
# _lock = threading.Lock()
# _t0 = time.time()
# _thread_ctx = threading.local()
# def _set_ctx(ctx: dict): setattr(_thread_ctx, "ctx", ctx)
# def _get_ctx() -> dict:   return getattr(_thread_ctx, "ctx", {})

# def log_emit(kind: str, **kv):
#     if not LOG_PATH: return
#     base = dict(kind=kind, ts=time.time(), t_rel_ms=int((time.time()-_t0)*1000),
#                 role=ROLE, pid=os.getpid(), container_id=CONTAINER_ID, model_id=MODEL_ID)
#     base.update(_get_ctx()); base.update(kv)
#     with _lock, open(LOG_PATH, "a", encoding="utf-8") as fp:
#         fp.write(json.dumps(base, ensure_ascii=False) + "\n")

# def _rss_mb():
#     try:
#         with open("/proc/self/status","r",encoding="utf-8",errors="ignore") as f:
#             for line in f:
#                 if line.startswith("VmRSS:"):
#                     return round(int(line.split()[1])/1024, 3)
#     except Exception: pass

# def _net_bytes():
#     total_rx = total_tx = 0
#     try:
#         for iface in pathlib.Path("/sys/class/net").iterdir():
#             try:
#                 total_rx += int((iface/"statistics/rx_bytes").read_text().strip())
#                 total_tx += int((iface/"statistics/tx_bytes").read_text().strip())
#             except Exception: continue
#     except Exception: pass
#     return total_rx, total_tx

# # === Lazy HF pipeline =================================================================
# _PIPE = None
# def get_pipe():
#     """
#     GPU roles: QUANT_MODE=fp16 DEVICE_MAP=auto TORCH_DTYPE=float16
#     CPU roles: DEVICE_MAP=cpu (or FORCE_CPU=1)
#     """
#     global _PIPE
#     if _PIPE is not None: return _PIPE

#     from transformers import pipeline
#     import torch

#     quant_mode  = os.getenv("QUANT_MODE", "").lower()  # "fp16" (we don't do 4bit here)
#     force_cpu   = os.getenv("FORCE_CPU", "0") == "1"
#     device_map  = os.getenv("DEVICE_MAP", ("cpu" if force_cpu else "auto"))
#     torch_dtype = os.getenv("TORCH_DTYPE", "auto").lower()

#     if torch_dtype == "float16": dtype = torch.float16
#     elif torch_dtype == "bfloat16": dtype = torch.bfloat16
#     else: dtype = "auto"

#     gen_kwargs = dict(
#         # no hard token cap; use a generous default to avoid truncation
#         max_new_tokens=int(os.getenv("MAX_NEW_TOKENS", "2048")),
#         temperature=float(os.getenv("TEMPERATURE", "0.0")),
#         top_p=float(os.getenv("TOP_P", "1.0")),
#         do_sample=bool(int(os.getenv("DO_SAMPLE", "0"))),
#         return_full_text=True,
#     )

#     log_emit("model_load_begin", model_id=MODEL_ID,
#              device_map=device_map, dtype=str(dtype), gen=gen_kwargs)

#     _PIPE = pipeline(
#         "text-generation",
#         model=MODEL_ID,
#         device_map=device_map,       # "auto" → GPU; "cpu" → CPU
#         torch_dtype=dtype,           # float16 on GPU roles
#         model_kwargs={"attn_implementation": "sdpa", "low_cpu_mem_usage": True},
#     )

#     log_emit("model_load_done", model_id=MODEL_ID)
#     return _PIPE

# # === Core request =====================================================================
# def handle_request(req: dict) -> dict:
#     text = (req.get("text") or "").strip()
#     if req.get("ping"): return {"ok": True, "pong": True}
#     if not text:        return {"ok": False, "error": "empty_text"}

#     trace_id       = (req.get("trace_id") or "")[:32]
#     pipeline_id    = (req.get("pipeline_id") or "")[:32]
#     span_id        = (req.get("span_id") or "")[:16]
#     parent_span_id = (req.get("parent_span_id") or "")[:16]
#     seq            = int(req.get("seq", 0))
#     traceparent    = f"00-{trace_id or '0'*32}-{span_id or '0'*16}-01"
#     cfp            = compute_cfp()
#     _set_ctx(dict(trace_id=trace_id, pipeline_id=pipeline_id, span_id=span_id,
#                   parent_span_id=parent_span_id, seq=seq, traceparent=traceparent, cfp=cfp))

#     rss0 = _rss_mb(); rx0, tx0 = _net_bytes()
#     p = psutil.Process() if psutil else None
#     if p: _ = p.cpu_percent(None)

#     log_emit("agent_msg", direction="in", text=text)
#     t0 = time.time()
#     try:
#         pipe = get_pipe()
#         # use generation params from env (see get_pipe); we call with none here to avoid caps
#         out = pipe(text)[0]["generated_text"]
#         dt_ms = int((time.time()-t0)*1000)
#         log_emit("agent_msg", direction="out", text=out, ms=dt_ms)
#         ok, err = True, None
#     except Exception as e:
#         out, ok, err = "", False, f"{type(e).__name__}: {e}"
#         dt_ms = int((time.time()-t0)*1000)
#         log_emit("run_error", err=err, ms=dt_ms)

#     rss1 = _rss_mb(); rx1, tx1 = _net_bytes()
#     cpu_pct = (p.cpu_percent(None) if p else None)
#     log_emit("sys_metrics",
#              rss_mb_before=rss0, rss_mb_after=rss1,
#              net_rx_bytes_delta=(rx1-rx0) if (rx1 and rx0) else None,
#              net_tx_bytes_delta=(tx1-tx0) if (tx1 and tx0) else None,
#              cpu_pct_process=cpu_pct)

#     return ({"ok": True, "text": out, "elapsed_ms": dt_ms}
#             if ok else {"ok": False, "error": err, "elapsed_ms": dt_ms})

# # === Robust AF_UNIX server ============================================================
# def _safe_send(f, payload: dict):
#     try:
#         f.write((json.dumps(payload, ensure_ascii=False) + "\n").encode())
#         f.flush()
#     except BrokenPipeError:
#         # client already gone; never crash the server
#         pass

# def serve(sock_path: str):
#     p = pathlib.Path(sock_path)
#     p.parent.mkdir(parents=True, exist_ok=True)
#     if p.exists(): p.unlink()
#     s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
#     s.bind(sock_path)
#     os.chmod(sock_path, 0o777)
#     s.listen(16)
#     print(f"[agent_service] logging to {LOG_PATH}" if LOG_PATH else "[agent_service] no file logging", flush=True)
#     print(f"[agent_service] role={ROLE} listening on {sock_path}", flush=True)
#     log_emit("server_start", sock=str(p))

#     try:
#         while True:
#             conn, _ = s.accept()
#             f = conn.makefile("rwb")
#             try:
#                 line_bytes = f.readline()
#                 if not line_bytes:
#                     # don't try to reply—client closed immediately
#                     continue
#                 line = line_bytes.decode("utf-8","ignore").strip()
#                 try:
#                     req = json.loads(line)
#                 except Exception as e:
#                     log_emit("bad_json", error=str(e), raw=line[:512])
#                     _safe_send(f, {"ok": False, "error": f"bad_json:{e}"})
#                     continue
#                 _safe_send(f, handle_request(req))
#             except Exception as e:
#                 tb = traceback.format_exc(limit=2)
#                 log_emit("server_error", error=str(e), trace=tb)
#                 _safe_send(f, {"ok": False, "error": str(e), "trace": tb})
#             finally:
#                 try: conn.close()
#                 except Exception: pass
#     finally:
#         try: s.close()
#         except Exception: pass
#         try: p.unlink()
#         except Exception: pass

# def main():
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--role", required=True)
#     ap.add_argument("--sock", required=True)
#     args = ap.parse_args()
#     global ROLE; ROLE = args.role
#     serve(args.sock)

# if __name__ == "__main__":
#     main()
