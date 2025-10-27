# utils.py
import os, json, pathlib, threading, datetime
import os

_BASE = os.getenv("AGENT_JSON_LOG", "agent.jsonl")
_lock = threading.Lock()

def _resolve(path_override: str | None) -> pathlib.Path:
    p = pathlib.Path(path_override or _BASE)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p

def emit(kind: str, __path: str | None = None, **fields):
    row = {"ts": datetime.datetime.utcnow().isoformat() + "Z", "kind": kind}
    row.update(fields)
    with _lock:
        with _resolve(__path).open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

def agent_msg(*, role: str, direction: str, text: str,
              n_prompt_toks=None, n_comp_toks=None, __path: str | None = None):
    emit("agent_msg", __path=__path, role=role, direction=direction, text=text,
         n_prompt_toks=n_prompt_toks, n_comp_toks=n_comp_toks)
