# import argparse, json, os, socket, time, threading, traceback, hmac, hashlib, base64
# from pathlib import Path

# # ---- identity / environment ------------------------------------------------------------
# ROLE         = os.environ.get("ROLE_NAME", "agent")
# ROUTE        = os.environ.get("ROLE_NAME", "agent")          # used in CFP
# LOG_PATH     = os.environ.get("AGENT_JSON_LOG", "")
# CONTAINER_ID = os.getenv("HOSTNAME", "")

# # CFP inputs (PII-safe, hashed/salted)
# HMAC_KEY = os.getenv("CFP_HMAC_KEY", "dev-key-change-me").encode()
# ORG_ID   = os.getenv("ORG_ID", "")
# API_KEY  = os.getenv("API_KEY", "")

# # ---- CFP + trace helpers ---------------------------------------------------------------
# def _hash(s: str) -> str:
#     return hashlib.sha256(s.encode()).hexdigest()

# def _time_bucket(seconds=300) -> int:
#     return int(time.time() // seconds)

# def compute_cfp() -> str:
#     tb  = str(_time_bucket())
#     akh = _hash(API_KEY) if API_KEY else ""
#     rth = _hash(ROUTE)
#     orh = _hash(ORG_ID) if ORG_ID else ""
#     msg = "|".join([tb, akh, rth, orh]).encode()
#     mac = hmac.new(HMAC_KEY, msg, hashlib.sha256).digest()
#     return base64.b64encode(mac).decode()

# def make_traceparent(trace_id_hex: str, span_id_hex: str) -> str:
#     # W3C: 00-<32 hex trace>-<16 hex span>-01
#     t = (trace_id_hex or "").lower()
#     s = (span_id_hex or "").lower()
#     if len(t) != 32 or len(s) != 16:
#         t = "0"*32
#         s = "0"*16
#     return f"00-{t}-{s}-01"

# # ---- logging ---------------------------------------------------------------------------
# _lock = threading.Lock()
# _t0   = time.time()

# def log_line(obj: dict):
#     if not LOG_PATH:
#         return
#     obj.setdefault("ts", time.time())
#     obj.setdefault("t_rel_ms", int((time.time()-_t0)*1000))
#     obj.setdefault("role", ROLE)
#     obj.setdefault("proc_pid", os.getpid())
#     obj.setdefault("container_id", CONTAINER_ID)
#     line = json.dumps(obj, ensure_ascii=False)
#     with _lock:
#         with open(LOG_PATH, "a", encoding="utf-8") as fp:
#             fp.write(line + "\n")

# def log(kind: str, corr: dict | None = None, **kv):
#     base = {"kind": kind}
#     if corr:
#         base.update(corr)
#     base.update(kv)
#     log_line(base)

# # ---- model pipeline (lazy) -------------------------------------------------------------
# _PIPE = None
# def get_pipe():
#     global _PIPE
#     if _PIPE is not None:
#         return _PIPE
#     from transformers import pipeline
#     model_id = os.environ.get("MODEL_ID", "georgesung/llama2_7b_chat_uncensored")
#     # correlation-free here; log as server meta
#     log("model_load_begin", None, model_id=model_id)
#     _PIPE = pipeline(
#         "text-generation",
#         model=model_id,
#         device_map="auto",
#         torch_dtype="auto",
#         model_kwargs={"attn_implementation": "sdpa"},
#         max_new_tokens=128,
#         temperature=0.7,
#     )
#     log("model_load_done", None, model_id=model_id)
#     return _PIPE

# # ---- request handler -------------------------------------------------------------------
# def handle(line: str) -> tuple[dict, dict]:
#     """Return (response_json, correlation_dict_for_logging)"""
#     try:
#         req = json.loads(line)
#     except Exception as e:
#         return {"ok": False, "error": f"bad_json:{e}"}, {}

#     # correlation context from client
#     trace_id       = req.get("trace_id") or ""
#     pipeline_id    = req.get("pipeline_id") or ""
#     span_id        = req.get("span_id") or ""
#     parent_span_id = req.get("parent_span_id") or ""
#     seq            = req.get("seq", 0)

#     corr = dict(
#         trace_id=trace_id,
#         pipeline_id=pipeline_id,
#         span_id=span_id,
#         parent_span_id=parent_span_id,
#         seq=seq,
#         cfp=compute_cfp(),
#         traceparent=make_traceparent(trace_id, span_id)
#     )

#     # fast health path (no model, works even when GPU is busy)
#     if req.get("ping"):
#         log("agent_msg", corr, direction="in", text="[ping]")
#         resp = {"ok": True, "pong": True}
#         log("agent_msg", corr, direction="out", text=json.dumps(resp, ensure_ascii=False))
#         return resp, corr

#     text = (req.get("text") or "").strip()
#     if not text:
#         return {"ok": False, "error": "empty_text"}, corr

#     # normal generation path
#     log("agent_msg", corr, direction="in", text=text)
#     t0 = time.time()
#     try:
#         out = get_pipe()(text, do_sample=False)[0]["generated_text"]
#         dt = int((time.time()-t0)*1000)
#         log("agent_msg", corr, direction="out", text=out, ms=dt)
#         return {"ok": True, "text": out, "elapsed_ms": dt}, corr
#     except Exception as e:
#         tb = traceback.format_exc(limit=3)
#         log("error", corr, error=str(e), trace=tb)
#         return {"ok": False, "error": str(e)}, corr

# # ---- server loop -----------------------------------------------------------------------
# def serve(sock_path: str):
#     p = Path(sock_path)
#     p.parent.mkdir(parents=True, exist_ok=True)
#     if p.exists():
#         p.unlink()
#     s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
#     s.bind(sock_path)
#     os.chmod(sock_path, 0o777)
#     s.listen(8)
#     print(f"[agent_service] logging to {LOG_PATH}", flush=True)
#     print(f"[agent_service] role={ROLE} listening on {sock_path}", flush=True)
#     log("server_start", None, sock=sock_path)
#     try:
#         while True:
#             conn, _ = s.accept()
#             f = conn.makefile("rwb")
#             try:
#                 line = f.readline().decode(errors="ignore").strip()
#                 if not line:
#                     try:
#                         f.write(b'{"ok":false,"error":"empty_line"}\n'); f.flush()
#                     except BrokenPipeError:
#                         pass
#                     continue
#                 resp, _corr = handle(line)
#                 try:
#                     f.write((json.dumps(resp, ensure_ascii=False)+"\n").encode()); f.flush()
#                 except BrokenPipeError:
#                     pass
#             except Exception as e:
#                 tb = traceback.format_exc(limit=2)
#                 try:
#                     f.write((json.dumps({"ok":False,"error":str(e),"trace":tb})+"\n").encode()); f.flush()
#                 except BrokenPipeError:
#                     pass
#             finally:
#                 try:
#                     conn.close()
#                 except Exception:
#                     pass
#     finally:
#         try:
#             s.close()
#         finally:
#             if p.exists():
#                 p.unlink()

# def main():
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--role", required=True)
#     ap.add_argument("--sock", required=True)
#     args = ap.parse_args()
#     global ROLE; ROLE = args.role
#     serve(args.sock)

# if __name__ == "__main__":
#     main()



#v2:
import argparse, json, os, socket, time, threading, traceback, hmac, hashlib, base64
from pathlib import Path

ROLE         = os.environ.get("ROLE_NAME", "agent")
ROUTE        = os.environ.get("ROLE_NAME", "agent")
LOG_PATH     = os.environ.get("AGENT_JSON_LOG", "")
CONTAINER_ID = os.getenv("HOSTNAME", "")

HMAC_KEY = os.getenv("CFP_HMAC_KEY", "dev-key-change-me").encode()
ORG_ID   = os.getenv("ORG_ID", "")
API_KEY  = os.getenv("API_KEY", "")

# ---------- helpers ----------
def _hash(s):
    return hashlib.sha256(s.encode()).hexdigest()

def _time_bucket(seconds=300):
    return int(time.time() // seconds)

def compute_cfp():
    tb  = str(_time_bucket())
    akh = _hash(API_KEY) if API_KEY else ""
    rth = _hash(ROUTE)
    orh = _hash(ORG_ID) if ORG_ID else ""
    msg = "|".join([tb, akh, rth, orh]).encode()
    mac = hmac.new(HMAC_KEY, msg, hashlib.sha256).digest()
    return base64.b64encode(mac).decode()

def make_traceparent(trace_id_hex, span_id_hex):
    t = (trace_id_hex or "").lower()
    s = (span_id_hex or "").lower()
    if len(t) != 32 or len(s) != 16:
        t = "0"*32; s = "0"*16
    return f"00-{t}-{s}-01"

_lock = threading.Lock()
_t0   = time.time()

def _log_line(obj):
    if not LOG_PATH: return
    obj.setdefault("ts", time.time())
    obj.setdefault("t_rel_ms", int((time.time()-_t0)*1000))
    obj.setdefault("role", ROLE)
    obj.setdefault("proc_pid", os.getpid())
    obj.setdefault("container_id", CONTAINER_ID)
    line = json.dumps(obj, ensure_ascii=False)
    with _lock:
        with open(LOG_PATH, "a", encoding="utf-8") as fp:
            fp.write(line + "\n")

def _log(kind, corr=None, **kv):
    base = {"kind": kind}
    if corr: base.update(corr)
    base.update(kv)
    _log_line(base)

# ---------- OPTIONAL lazy model ----------
_PIPE = None
def get_pipe():
    global _PIPE
    if _PIPE is not None:
        return _PIPE
    from transformers import pipeline
    model_id = os.environ.get("MODEL_ID", "georgesung/llama2_7b_chat_uncensored")
    _log("model_load_begin", None, model_id=model_id)
    _PIPE = pipeline(
        "text-generation",
        model=model_id,
        device_map="auto",
        torch_dtype="auto",
        model_kwargs={"attn_implementation": "sdpa"},
        max_new_tokens=128,
        temperature=0.7,
    )
    _log("model_load_done", None, model_id=model_id)
    return _PIPE

# ---------- request handler ----------
def handle(line):
    # returns (resp_dict, corr_dict)
    try:
        req = json.loads(line)
    except Exception as e:
        return {"ok": False, "error": f"bad_json:{e}"}, {}

    trace_id       = req.get("trace_id","")
    pipeline_id    = req.get("pipeline_id","")
    span_id        = req.get("span_id","")
    parent_span_id = req.get("parent_span_id","")
    seq            = req.get("seq", 0)

    corr = dict(
        trace_id=trace_id,
        pipeline_id=pipeline_id,
        span_id=span_id,
        parent_span_id=parent_span_id,
        seq=seq,
        cfp=compute_cfp(),
        traceparent=make_traceparent(trace_id, span_id),
    )

    # zero-GPU health path
    if req.get("ping"):
        _log("agent_msg", corr, direction="in", text="[ping]")
        resp = {"ok": True, "pong": True}
        _log("agent_msg", corr, direction="out", text=json.dumps(resp, ensure_ascii=False))
        return resp, corr

    text = (req.get("text") or "").strip()
    if not text:
        return {"ok": False, "error": "empty_text"}, corr

    _log("agent_msg", corr, direction="in", text=text)
    t0 = time.time()
    try:
        out = get_pipe()(text, do_sample=False)[0]["generated_text"]
        dt  = int((time.time()-t0)*1000)
        _log("agent_msg", corr, direction="out", text=out, ms=dt)
        return {"ok": True, "text": out, "elapsed_ms": dt}, corr
    except Exception as e:
        tb = traceback.format_exc(limit=3)
        _log("error", corr, error=str(e), trace=tb)
        return {"ok": False, "error": str(e)}, corr

# ---------- server loop ----------
def serve(sock_path):
    p = Path(sock_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.exists(): p.unlink()
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.bind(sock_path)
    os.chmod(sock_path, 0o777)
    s.listen(8)
    print(f"[agent_service] logging to {LOG_PATH}", flush=True)
    print(f"[agent_service] role={ROLE} listening on {sock_path}", flush=True)
    _log("server_start", None, sock=sock_path)
    try:
        while True:
            conn, _ = s.accept()
            f = conn.makefile("rwb")
            try:
                line = f.readline().decode(errors="ignore")
                if not line:
                    try:
                        f.write(b'{"ok":false,"error":"empty_line"}\n'); f.flush()
                    except BrokenPipeError:
                        pass
                    continue
                line = line.strip()
                resp, _corr = handle(line)
                try:
                    f.write((json.dumps(resp, ensure_ascii=False)+"\n").encode()); f.flush()
                except BrokenPipeError:
                    pass
            except Exception as e:
                tb = traceback.format_exc(limit=2)
                try:
                    f.write((json.dumps({"ok":False,"error":str(e),"trace":tb})+"\n").encode()); f.flush()
                except BrokenPipeError:
                    pass
            finally:
                try: conn.close()
                except Exception: pass
    finally:
        try: s.close()
        finally:
            if p.exists(): p.unlink()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--role", required=True)
    ap.add_argument("--sock", required=True)
    args = ap.parse_args()
    global ROLE; ROLE = args.role
    serve(args.sock)

if __name__ == "__main__":
    main()
