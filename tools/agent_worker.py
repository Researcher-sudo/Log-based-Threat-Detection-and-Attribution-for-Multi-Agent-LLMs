# #!/usr/bin/env python3
# import os, sys, json, time, uuid, argparse, re
# from datetime import datetime, timezone

# # Optional: tag outbound HTTP if your stack uses requests
# def monkeypatch_requests(extra_headers: dict):
#     try:
#         import requests
#     except Exception:
#         return
#     old = requests.sessions.Session.request
#     def patched(self, method, url, **kw):
#         hdrs = kw.get("headers") or {}
#         hdrs = {**extra_headers, **hdrs}
#         kw["headers"] = hdrs
#         return old(self, method, url, **kw)
#     requests.sessions.Session.request = patched

# def now_iso():
#     return datetime.now(timezone.utc).isoformat()

# def load_model(model_id):
#     # >>> Replace with *your* model init <<<
#     # Example: HF text-generation pipeline
#     from transformers import pipeline
#     return pipeline("text-generation", model=model_id, device_map="auto")

# def generate_text(pipeline_obj, prompt: str, max_new_tokens=256):
#     t0 = time.time()
#     out = pipeline_obj(prompt, max_new_tokens=max_new_tokens, do_sample=False)
#     text = out[0]["generated_text"]
#     ms = int((time.time()-t0)*1000)
#     toks = None
#     return text, toks, ms

# def main():
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--agent-id", required=True)
#     ap.add_argument("--agent-name", required=True)
#     ap.add_argument("--model", required=True)
#     ap.add_argument("--outdir", required=True)
#     ap.add_argument("--pidfile", required=True)
#     ap.add_argument("--headers", default="")  # JSON dict or empty
#     args = ap.parse_args()

#     os.makedirs(args.outdir, exist_ok=True)
#     ap.add_argument("--headers", default="", help="extra headers as JSON")
#     args = ap.parse_args()

#     # Write our PID so the spawner can attach pidstat/NVML
#     with open(args.pidfile, "w") as f: f.write(str(os.getpid()))

#     # Per-agent HTTP headers
#     alarm_trace = f"agent_{args.agent-id}_{uuid.uuid4().hex}"
#     traceparent = f"00-{uuid.uuid4().hex}-{uuid.uuid4().hex[:16]}-01"
#     extra_headers = {
#         "X-Alarm-Trace": alarm_trace,
#         "traceparent": traceparent,
#         "tracestate": ""
#     }
#     if args.headers:
#         try: extra_headers.update(json.loads(args.headers))
#         except Exception: pass
#     monkeypatch_requests(extra_headers)

#     # Load model once
#     pipe = load_model(args.model)

#     # Simple JSONL protocol on stdio
#     for line in sys.stdin:
#         line = line.strip()
#         if not line: 
#             continue
#         try:
#             req = json.loads(line)
#             rid = req.get("id") or uuid.uuid4().hex
#             prompt = req.get("prompt") or ""
#             text, toks, ms = generate_text(pipe, prompt)
#             resp = {"id": rid, "text": text, "tokens": toks, "ms": ms,
#                     "ts": now_iso(), "agent_id": int(args.agent_id),
#                     "headers": extra_headers}
#         except Exception as e:
#             resp = {"id": req.get("id","?"), "error": str(e), "ts": now_iso()}
#         sys.stdout.write(json.dumps(resp, ensure_ascii=False) + "\n")
#         sys.stdout.flush()

# if __name__ == "__main__":
#     main()



# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-

# import os, sys, json, time, uuid, argparse, re, math, hmac, hashlib
# from datetime import datetime, timezone

# # -------------------------
# # Utilities
# # -------------------------

# def monkeypatch_requests(extra_headers: dict):
#     """Optionally tag outbound HTTP if your stack uses requests."""
#     try:
#         import requests  # type: ignore
#     except Exception:
#         return
#     old = requests.sessions.Session.request
#     def patched(self, method, url, **kw):
#         hdrs = kw.get("headers") or {}
#         hdrs = {**extra_headers, **hdrs}
#         kw["headers"] = hdrs
#         return old(self, method, url, **kw)
#     requests.sessions.Session.request = patched

# def now_iso():
#     return datetime.now(timezone.utc).isoformat()

# def load_model(model_id: str):
#     """Replace with your model init if needed; default: HF text-generation pipeline."""
#     from transformers import pipeline  # type: ignore
#     return pipeline("text-generation", model=model_id, device_map="auto")

# def generate_text(pipeline_obj, prompt: str, max_new_tokens=256, do_sample=False, temperature=None):
#     """
#     Calls the pipeline with safe kwargs.
#     We only include `temperature` when sampling to avoid HF warnings.
#     Returns: text, timing_ms, prompt_tokens, completion_tokens
#     """
#     t0 = time.time()
#     gen = {"max_new_tokens": int(max_new_tokens), "do_sample": bool(do_sample)}
#     if do_sample and temperature is not None:
#         gen["temperature"] = float(temperature)

#     out = pipeline_obj(prompt, **gen)
#     # HF pipelines usually return a list of dicts with "generated_text"
#     text = out[0].get("generated_text", out[0].get("summary_text", ""))

#     ms = int((time.time() - t0) * 1000)

#     in_tok = out_tok = None
#     try:
#         tok = getattr(pipeline_obj, "tokenizer", None)
#         if tok:
#             in_tok = len(tok.encode(prompt))
#             total_tok = len(tok.encode(text))
#             out_tok = max(0, total_tok - in_tok)
#     except Exception:
#         pass

#     return text, ms, in_tok, out_tok

# def main():
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--agent-id", required=True)
#     ap.add_argument("--agent-name", required=True)
#     ap.add_argument("--model", required=True)
#     ap.add_argument("--outdir", required=True)
#     ap.add_argument("--pidfile", required=True)
#     ap.add_argument("--headers", default="", help="extra headers as JSON")
#     args = ap.parse_args()

#     os.makedirs(args.outdir, exist_ok=True)

#     # Write our PID so the spawner can attach strace/NVML/etc.
#     with open(args.pidfile, "w") as f:
#         f.write(str(os.getpid()))

#     # Per-agent trace headers
#     alarm_trace = f"agent_{args.agent_id}_{uuid.uuid4().hex}"
#     traceparent = f"00-{uuid.uuid4().hex}-{uuid.uuid4().hex[:16]}-01"
#     extra_headers = {
#         "X-Alarm-Trace": alarm_trace,
#         "traceparent": traceparent,
#         "tracestate": ""
#     }
#     if args.headers:
#         try:
#             extra_headers.update(json.loads(args.headers))
#         except Exception:
#             pass
#     monkeypatch_requests(extra_headers)

#     # Model (once)
#     pipe = load_model(args.model)

#     # Device info (best-effort)
#     try:
#         import torch  # type: ignore
#         if torch.cuda.is_available():
#             device = f"cuda:{torch.cuda.current_device()}"
#             device_name = torch.cuda.get_device_name(0)
#             vram_gb = torch.cuda.get_device_properties(0).total_memory // (1024**3)
#         else:
#             device = "cpu"
#             device_name = "cpu"
#             vram_gb = 0
#     except Exception:
#         device = "cpu"
#         device_name = "cpu"
#         vram_gb = 0

#     # Content-blind fingerprint (CFP): no text, only hashed identifiers
#     def cfp(secret, api="", route="", org="", bucket_s=300):
#         tb = int(math.floor(time.time() / bucket_s))
#         msg = f"{tb}|{api}|{route}|{org}"
#         return hmac.new(secret.encode(), msg.encode(), hashlib.sha256).hexdigest()[:32]

#     CFP_SECRET  = os.getenv("CFP_SECRET", "CFP_DEV_SECRET")
#     API_KEY_HASH = os.getenv("API_KEY_HASH", "")
#     ROUTE_HASH   = os.getenv("ROUTE_HASH", "")
#     ORG_ID_HASH  = os.getenv("ORG_ID_HASH", "")

#     # Simple JSONL over stdio
#     for line in sys.stdin:
#         line = line.strip()
#         if not line:
#             continue
#         try:
#             req = json.loads(line)
#             rid = req.get("id") or uuid.uuid4().hex
#             prompt = req.get("prompt") or ""

#             max_new_tokens = int(req.get("max_new_tokens", 256))
#             do_sample = bool(req.get("do_sample", False))
#             temperature = req.get("temperature", None)

#             text, ms, in_tok, out_tok = generate_text(
#                 pipe,
#                 prompt,
#                 max_new_tokens=max_new_tokens,
#                 do_sample=do_sample,
#                 temperature=temperature
#             )

#             resp = {
#                 "id": rid,
#                 "text": text,
#                 "ts": now_iso(),
#                 "agent_id": int(args.agent_id),
#                 "meta": {
#                     "model_id": args.model,
#                     "device": device,
#                     "device_name": device_name,
#                     "vram_gb": int(vram_gb),
#                     "decode": {
#                         "max_new_tokens": max_new_tokens,
#                         "do_sample": do_sample,
#                         "temperature": (float(temperature) if (temperature is not None) else None)
#                     },
#                     "timing_ms": ms,
#                     "tokens": {"prompt": in_tok, "completion": out_tok},
#                     "pid": os.getpid(),
#                     "headers": extra_headers,
#                     "trace": {
#                         "trace_id": alarm_trace,
#                         "traceparent": traceparent,
#                         "tracestate": "",
#                         "cfp": cfp(CFP_SECRET, API_KEY_HASH, ROUTE_HASH, ORG_ID_HASH)
#                     }
#                 }
#             }
#         except Exception as e:
#             # Keep the request id if possible
#             try:
#                 rid = json.loads(line).get("id", "?")
#             except Exception:
#                 rid = "?"
#             resp = {"id": rid, "error": str(e), "ts": now_iso()}

#         sys.stdout.write(json.dumps(resp, ensure_ascii=False) + "\n")
#         sys.stdout.flush()

# if __name__ == "__main__":
#     main()
# #!/usr/bin/env python3
# import sys, json, argparse, os
# from utils import Agent  # uses your existing Agent (HF/GPT/Gemini supported)

# def main():
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--agent-id", required=True, type=int)
#     ap.add_argument("--agent-name", required=True)
#     ap.add_argument("--model", required=True)
#     ap.add_argument("--outdir", required=True)
#     ap.add_argument("--pidfile", required=True)
#     a = ap.parse_args()

#     # write PID so strace/tcpdump/etc can attach
#     os.makedirs(a.outdir, exist_ok=True)
#     with open(a.pidfile, "w") as f:
#         f.write(str(os.getpid()))
#         f.flush()

#     # We lazily create the Agent after receiving "init" (so we get instruction)
#     agent = None

#     def write(obj: dict):
#         sys.stdout.write(json.dumps(obj, ensure_ascii=False) + "\n")
#         sys.stdout.flush()

#     for line in sys.stdin:
#         line = line.strip()
#         if not line:
#             continue
#         try:
#             req = json.loads(line)
#         except Exception:
#             # Non-JSON noise; ignore (stdout must remain JSON lines ideally)
#             continue

#         cmd = req.get("cmd")
#         rid = req.get("req_id")

#         if cmd == "init":
#             instruction = req.get("instruction", f"You are {a.agent_name}.")
#             try:
#                 agent = Agent(instruction=instruction, role=a.agent_name, model_info=a.model)
#                 # seed chat with instruction to match your current agents
#                 agent.chat(instruction)
#                 write({"ok": True, "req_id": rid, "status": "initialized"})
#             except Exception as e:
#                 write({"ok": False, "req_id": rid, "error": f"init_failed: {e}"})

#         elif cmd == "chat":
#             if agent is None:
#                 write({"ok": False, "req_id": rid, "error": "not_initialized"})
#                 continue

#             msg = req.get("message", "")
#             chat_mode = bool(req.get("chat_mode", True))
#             gen = req.get("gen", {}) or {}
#             # Pass through common generation kwargs your Agent.chat supports
#             try:
#                 text = agent.chat(
#                     msg,
#                     chat_mode=chat_mode,
#                     **{k: v for k, v in gen.items()
#                        if k in ("max_new_tokens", "do_sample", "temperature", "top_p")}
#                 )
#                 write({"ok": True, "req_id": rid, "text": text, "usage": {}})
#             except Exception as e:
#                 write({"ok": False, "req_id": rid, "error": f"chat_failed: {e}"})
#         else:
#             write({"ok": False, "req_id": rid, "error": f"unknown_cmd: {cmd}"})

# if __name__ == "__main__":
#     main()
# #!/usr/bin/env python3
# import os, sys, json, time, uuid, argparse, re
# from datetime import datetime, timezone

# # Optional: tag outbound HTTP if your stack uses requests
# def monkeypatch_requests(extra_headers: dict):
#     try:
#         import requests
#     except Exception:
#         return
#     old = requests.sessions.Session.request
#     def patched(self, method, url, **kw):
#         hdrs = kw.get("headers") or {}
#         hdrs = {**extra_headers, **hdrs}
#         kw["headers"] = hdrs
#         return old(self, method, url, **kw)
#     requests.sessions.Session.request = patched

# def now_iso():
#     return datetime.now(timezone.utc).isoformat()

# def load_model(model_id):
#     # >>> Replace with *your* model init <<<
#     # Example: HF text-generation pipeline
#     from transformers import pipeline
#     return pipeline("text-generation", model=model_id, device_map="auto")

# def generate_text(pipeline_obj, prompt: str, max_new_tokens=256):
#     t0 = time.time()
#     out = pipeline_obj(prompt, max_new_tokens=max_new_tokens, do_sample=False)
#     text = out[0]["generated_text"]
#     ms = int((time.time()-t0)*1000)
#     toks = None
#     return text, toks, ms

# def main():
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--agent-id", required=True)
#     ap.add_argument("--agent-name", required=True)
#     ap.add_argument("--model", required=True)
#     ap.add_argument("--outdir", required=True)
#     ap.add_argument("--pidfile", required=True)
#     ap.add_argument("--headers", default="")  # JSON dict or empty
#     args = ap.parse_args()

#     os.makedirs(args.outdir, exist_ok=True)
#     ap.add_argument("--headers", default="", help="extra headers as JSON")
#     args = ap.parse_args()

#     # Write our PID so the spawner can attach pidstat/NVML
#     with open(args.pidfile, "w") as f: f.write(str(os.getpid()))

#     # Per-agent HTTP headers
#     alarm_trace = f"agent_{args.agent-id}_{uuid.uuid4().hex}"
#     traceparent = f"00-{uuid.uuid4().hex}-{uuid.uuid4().hex[:16]}-01"
#     extra_headers = {
#         "X-Alarm-Trace": alarm_trace,
#         "traceparent": traceparent,
#         "tracestate": ""
#     }
#     if args.headers:
#         try: extra_headers.update(json.loads(args.headers))
#         except Exception: pass
#     monkeypatch_requests(extra_headers)

#     # Load model once
#     pipe = load_model(args.model)

#     # Simple JSONL protocol on stdio
#     for line in sys.stdin:
#         line = line.strip()
#         if not line: 
#             continue
#         try:
#             req = json.loads(line)
#             rid = req.get("id") or uuid.uuid4().hex
#             prompt = req.get("prompt") or ""
#             text, toks, ms = generate_text(pipe, prompt)
#             resp = {"id": rid, "text": text, "tokens": toks, "ms": ms,
#                     "ts": now_iso(), "agent_id": int(args.agent_id),
#                     "headers": extra_headers}
#         except Exception as e:
#             resp = {"id": req.get("id","?"), "error": str(e), "ts": now_iso()}
#         sys.stdout.write(json.dumps(resp, ensure_ascii=False) + "\n")
#         sys.stdout.flush()

# if __name__ == "__main__":
#     main()



#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, sys, json, time, uuid, argparse, re, math, hmac, hashlib
from datetime import datetime, timezone

# -------------------------
# Utilities
# -------------------------

def monkeypatch_requests(extra_headers: dict):
    """Optionally tag outbound HTTP if your stack uses requests."""
    try:
        import requests  # type: ignore
    except Exception:
        return
    old = requests.sessions.Session.request
    def patched(self, method, url, **kw):
        hdrs = kw.get("headers") or {}
        hdrs = {**extra_headers, **hdrs}
        kw["headers"] = hdrs
        return old(self, method, url, **kw)
    requests.sessions.Session.request = patched

def now_iso():
    return datetime.now(timezone.utc).isoformat()

def load_model(model_id: str):
    """Replace with your model init if needed; default: HF text-generation pipeline."""
    from transformers import pipeline  # type: ignore
    return pipeline("text-generation", model=model_id, device_map="auto")

def generate_text(pipeline_obj, prompt: str, max_new_tokens=256, do_sample=False, temperature=None):
    """
    Calls the pipeline with safe kwargs.
    We only include `temperature` when sampling to avoid HF warnings.
    Returns: text, timing_ms, prompt_tokens, completion_tokens
    """
    t0 = time.time()
    gen = {"max_new_tokens": int(max_new_tokens), "do_sample": bool(do_sample)}
    if do_sample and temperature is not None:
        gen["temperature"] = float(temperature)

    out = pipeline_obj(prompt, **gen)
    # HF pipelines usually return a list of dicts with "generated_text"
    text = out[0].get("generated_text", out[0].get("summary_text", ""))

    ms = int((time.time() - t0) * 1000)

    in_tok = out_tok = None
    try:
        tok = getattr(pipeline_obj, "tokenizer", None)
        if tok:
            in_tok = len(tok.encode(prompt))
            total_tok = len(tok.encode(text))
            out_tok = max(0, total_tok - in_tok)
    except Exception:
        pass

    return text, ms, in_tok, out_tok

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent-id", required=True)
    ap.add_argument("--agent-name", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--pidfile", required=True)
    ap.add_argument("--headers", default="", help="extra headers as JSON")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # Write our PID so the spawner can attach strace/NVML/etc.
    with open(args.pidfile, "w") as f:
        f.write(str(os.getpid()))

    # Per-agent trace headers
    alarm_trace = f"agent_{args.agent_id}_{uuid.uuid4().hex}"
    traceparent = f"00-{uuid.uuid4().hex}-{uuid.uuid4().hex[:16]}-01"
    extra_headers = {
        "X-Alarm-Trace": alarm_trace,
        "traceparent": traceparent,
        "tracestate": ""
    }
    if args.headers:
        try:
            extra_headers.update(json.loads(args.headers))
        except Exception:
            pass
    monkeypatch_requests(extra_headers)

    # Model (once)
    pipe = load_model(args.model)

    # Device info (best-effort)
    try:
        import torch  # type: ignore
        if torch.cuda.is_available():
            device = f"cuda:{torch.cuda.current_device()}"
            device_name = torch.cuda.get_device_name(0)
            vram_gb = torch.cuda.get_device_properties(0).total_memory // (1024**3)
        else:
            device = "cpu"
            device_name = "cpu"
            vram_gb = 0
    except Exception:
        device = "cpu"
        device_name = "cpu"
        vram_gb = 0

    # Content-blind fingerprint (CFP): no text, only hashed identifiers
    def cfp(secret, api="", route="", org="", bucket_s=300):
        tb = int(math.floor(time.time() / bucket_s))
        msg = f"{tb}|{api}|{route}|{org}"
        return hmac.new(secret.encode(), msg.encode(), hashlib.sha256).hexdigest()[:32]

    CFP_SECRET  = os.getenv("CFP_SECRET", "CFP_DEV_SECRET")
    API_KEY_HASH = os.getenv("API_KEY_HASH", "")
    ROUTE_HASH   = os.getenv("ROUTE_HASH", "")
    ORG_ID_HASH  = os.getenv("ORG_ID_HASH", "")

    # Simple JSONL over stdio
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            rid = req.get("id") or uuid.uuid4().hex
            prompt = req.get("prompt") or ""

            max_new_tokens = int(req.get("max_new_tokens", 256))
            do_sample = bool(req.get("do_sample", False))
            temperature = req.get("temperature", None)

            text, ms, in_tok, out_tok = generate_text(
                pipe,
                prompt,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature
            )

            resp = {
                "id": rid,
                "text": text,
                "ts": now_iso(),
                "agent_id": int(args.agent_id),
                "meta": {
                    "model_id": args.model,
                    "device": device,
                    "device_name": device_name,
                    "vram_gb": int(vram_gb),
                    "decode": {
                        "max_new_tokens": max_new_tokens,
                        "do_sample": do_sample,
                        "temperature": (float(temperature) if (temperature is not None) else None)
                    },
                    "timing_ms": ms,
                    "tokens": {"prompt": in_tok, "completion": out_tok},
                    "pid": os.getpid(),
                    "headers": extra_headers,
                    "trace": {
                        "trace_id": alarm_trace,
                        "traceparent": traceparent,
                        "tracestate": "",
                        "cfp": cfp(CFP_SECRET, API_KEY_HASH, ROUTE_HASH, ORG_ID_HASH)
                    }
                }
            }
        except Exception as e:
            # Keep the request id if possible
            try:
                rid = json.loads(line).get("id", "?")
            except Exception:
                rid = "?"
            resp = {"id": rid, "error": str(e), "ts": now_iso()}

        sys.stdout.write(json.dumps(resp, ensure_ascii=False) + "\n")
        sys.stdout.flush()

if __name__ == "__main__":
    main()
