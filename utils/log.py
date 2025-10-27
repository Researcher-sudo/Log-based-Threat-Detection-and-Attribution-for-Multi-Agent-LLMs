# import json, os, time, socket, threading, traceback
# import json, time, os, sys

# # where raw JSONL lines will go
# LOG_PATH = os.getenv("AGENT_JSON_LOG") or "agent.jsonl"

# _lock = threading.Lock()           # so concurrent threads don’t mix lines
# _start   = time.time()

# def emit(kind: str, **kv) -> None:
#     """
#     Write one JSON‑object line.
#     Every record gets:
#        • absolute ts          (seconds since epoch)
#        • t_rel_ms             (milliseconds since program start)
#        • host, pid
#        • kind                 (event type)
#     """
#     kv.update(
#         kind = kind,
#         ts   = time.time(),
#         t_rel_ms = int((time.time() - _start) * 1000),
#         host = socket.gethostname(),
#         pid  = os.getpid(),
#     )
#     line = json.dumps(kv, ensure_ascii=False)
#     with _lock, open(LOG_PATH, "a", encoding="utf‑8") as fp:
#         fp.write(line + "\n")

# # ---------------------------------------------------------------------------
# # 2)  Convenience wrapper for agent traffic
# # ---------------------------------------------------------------------------
# def agent_msg(*, role: str, direction: str, text: str,
#               n_prompt_toks: int | None = None,
#               n_comp_toks:   int | None = None):
#     """
#     Log any inbound/outbound message exchanged by an Agent.

#     Args
#     ----
#     role            e.g. "User", "Assistant", "Cardiologist‑Agent"
#     direction       "in"  (user → agent)   or
#                     "out" (agent → user)
#     text            raw string content
#     n_prompt_toks   token count of the *prompt* portion (if you have it)
#     n_comp_toks     token count of the completion / reply (if you have it)
#     """
#     emit("agent_msg",
#          role=role,
#          dir =direction,
#          text=text.strip(),
#          n_prompt_toks=n_prompt_toks,
#          n_comp_toks  =n_comp_toks)

# # ---------------------------------------------------------------------------
# # 3)  Decorator to auto‑log uncaught exceptions (unchanged)
# # ---------------------------------------------------------------------------
# def catch_exceptions(fn):
#     def _wrap(*a, **kw):
#         try:
#             return fn(*a, **kw)
#         except Exception as e:
#             emit("py_error",
#                  fn=f"{fn.__module__}.{fn.__name__}",
#                  err=str(e),
#                  stack=traceback.format_exc(limit=4))
#             raise
#     return _wrap

# utils/log.py
import json, os, time, socket, threading, traceback
import os

LOG_PATH = os.getenv("AGENT_JSON_LOG") or "agent.jsonl"
_lock = threading.Lock()
_start = time.time()

def emit(kind: str, __path: str = None, **kv) -> None:
    kv.update(
        kind=kind,
        ts=time.time(),
        t_rel_ms=int((time.time() - _start) * 1000),
        host=socket.gethostname(),
        pid=os.getpid(),
    )
    path = __path or LOG_PATH
    line = json.dumps(kv, ensure_ascii=False)
    with _lock, open(path, "a", encoding="utf-8") as fp:
        fp.write(line + "\n")

def agent_msg(*, role: str, direction: str, text: str,
              n_prompt_toks: int = None, n_comp_toks: int = None,
              __path: str = None):
    emit("agent_msg",
         __path=__path,
         role=role,
         direction=direction,   # ← was dir=direction
         text=text.strip(),
         n_prompt_toks=n_prompt_toks,
         n_comp_toks=n_comp_toks)

def catch_exceptions(fn):
    def _wrap(*a, **kw):
        try:
            return fn(*a, **kw)
        except Exception as e:
            emit("py_error", err=str(e), stack=traceback.format_exc(limit=4))
            raise
    return _wrap
