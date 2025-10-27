# # # # #!/usr/bin/env python3
# # # # import os, json, time, argparse, uuid
# # # # from pathlib import Path

# # # # # set log path AFTER import by assigning to log.LOG_PATH
# # # # # so we can switch files in the same process.
# # # # def set_log_path(log, path: str):
# # # #     Path(path).parent.mkdir(parents=True, exist_ok=True)
# # # #     log.LOG_PATH = path

# # # # def iter_jsonl(path):
# # # #     with open(path, "r", encoding="utf-8") as f:
# # # #         for ln in f:
# # # #             ln = ln.strip()
# # # #             if ln:
# # # #                 yield json.loads(ln)

# # # # def run_file(jsonl_path: str, *, model: str, role: str, log_path: str, sleep_s: float = 0.0):
# # # #     # late import so we can set log.LOG_PATH before use
# # # #     from utils import log
# # # #     from utils import Agent

# # # #     set_log_path(log, log_path)
# # # #     n = 0
# # # #     for row in iter_jsonl(jsonl_path):
# # # #         sample_id   = row.get("id")                # e.g., medqa_attack_cand_...
# # # #         label       = row.get("label")             # "benign" or "attack"
# # # #         tail_family = row.get("tail_family")       # None for benign
# # # #         prompt      = row["full_prompt"]

# # # #         # Mark the start of a sample in the log
# # # #         run_id = str(uuid.uuid4())
# # # #         log.emit("run_start",
# # # #                  run_id=run_id,
# # # #                  sample_id=sample_id,
# # # #                  label=label,
# # # #                  tail_family=tail_family)

# # # #         # Fresh Agent per prompt so no conversation carryover
# # # #         agent = Agent(instruction=os.environ.get("AGENT_INSTRUCTION", "You are a helpful agent."), role=role, model_info=model)
# # # #         _ = agent.chat(prompt)

# # # #         # Mark the end of a sample in the log
# # # #         log.emit("run_end",
# # # #                  run_id=run_id,
# # # #                  sample_id=sample_id)

# # # #         n += 1
# # # #         if sleep_s:
# # # #             time.sleep(sleep_s)
# # # #     return n

# # # # def main():
# # # #     ap = argparse.ArgumentParser()
# # # #     ap.add_argument("--model", default="llama2-uncensored",
# # # #                     help="Model name passed to Agent(instruction=os.environ.get("AGENT_INSTRUCTION"), model_info=...).")
# # # #     ap.add_argument("--role", default="target",
# # # #                     help="Role string to tag in logs (e.g., 'target').")
# # # #     ap.add_argument("--benign", type=str, default=None,
# # # #                     help="Path to benign JSONL (e.g., full_pack/benign_preview.jsonl).")
# # # #     ap.add_argument("--attacks", type=str, default=None,
# # # #                     help="Path to compromised JSONL (e.g., full_pack/compromised_preview.jsonl).")
# # # #     ap.add_argument("--outdir", type=str, default="logs",
# # # #                     help="Directory for log files.")
# # # #     ap.add_argument("--sleep", type=float, default=0.0,
# # # #                     help="Optional sleep(seconds) between prompts.")
# # # #     args = ap.parse_args()

# # # #     outdir = Path(args.outdir)
# # # #     outdir.mkdir(parents=True, exist_ok=True)

# # # #     total = 0
# # # #     if args.benign:
# # # #         n = run_file(
# # # #             args.benign,
# # # #             model=args.model,
# # # #             role=args.role,
# # # #             log_path=str(outdir / "benign_agent.jsonl"),
# # # #             sleep_s=args.sleep,
# # # #         )
# # # #         print(f"[✓] Logged {n} benign prompts → {outdir/'benign_agent.jsonl'}")
# # # #         total += n

# # # #     if args.attacks:
# # # #         n = run_file(
# # # #             args.attacks,
# # # #             model=args.model,
# # # #             role=args.role,
# # # #             log_path=str(outdir / "compromised_agent.jsonl"),
# # # #             sleep_s=args.sleep,
# # # #         )
# # # #         print(f"[✓] Logged {n} compromised prompts → {outdir/'compromised_agent.jsonl'}")
# # # #         total += n

# # # #     if total == 0:
# # # #         print("No input files provided. Use --benign and/or --attacks.")

# # # # if __name__ == "__main__":
# # # #     main()
# # # #!/usr/bin/env python3
# # # """
# # # generate_agent_logs.py
# # # Run a set of MCQ prompts (benign + compromised) through utils.Agent
# # # and emit detailed per-message logs via utils/log.py.

# # # It expects rows shaped like your preview/full pack:
# # # {
# # #   "id": "...",
# # #   "label": "benign" | "attack",
# # #   "tail_family": "json_plain" | ... | null,
# # #   "full_prompt": "MCQ ... (A) ... (B) ... --- <tail-if-any>",
# # #   ...
# # # }
# # # """

# # # import argparse, json, os, sys, time
# # # import os
# # # from pathlib import Path
# # # from utils import Agent, log
# # # log = log  # back-compat alias

# # # # Your utilities (Agent + JSONL logger)
# # # from utils import Agent, log  # type: ignore
# # # log = log  # back-compat alias

# # # def now_ms() -> int:
# # #     return int(time.time() * 1000)

# # # def iter_jsonl(path: Path):
# # #     with path.open("r", encoding="utf-8") as f:
# # #         for line in f:
# # #             line = line.strip()
# # #             if not line:
# # #                 continue
# # #             try:
# # #                 yield json.loads(line)
# # #             except Exception as e:
# # #                 log.emit("jsonl_parse_error", file=str(path), err=str(e), raw=line[:500])

# # # def run_set(items_path: Path, *, model: str, out_path: Path,
# # #             label_hint: str | None, sleep_s: float = 0.0):
# # #     n = 0
# # #     kept = 0
# # #     with out_path.open("w", encoding="utf-8") as fout:
# # #         for row in iter_jsonl(items_path):
# # #             n += 1
# # #             sid   = row.get("id", f"sample_{n:06d}")
# # #             label = row.get("label", label_hint or "unknown")
# # #             fam   = row.get("tail_family")
# # #             text  = row.get("full_prompt") or row.get("question_text")

# # #             if not text:
# # #                 log.emit("skip_missing_text", id=sid, label=label)
# # #                 continue

# # #             # Fresh Agent for isolation (HF model is cached internally)
# # #             agent = Agent(instruction=os.environ.get("AGENT_INSTRUCTION", "You are a helpful agent."), role="intruder", model_info=model)

# # #             # Mark start
# # #             log.emit("run_start", id=sid, label=label, tail_family=fam)

# # #             err = None
# # #             rsp = None
# # #             t0 = now_ms()
# # #             try:
# # #                 rsp = agent.chat(text)
# # #             except Exception as e:
# # #                 err = str(e)
# # #                 log.emit("run_error", id=sid, label=label, tail_family=fam, err=err)
# # #             t1 = now_ms()

# # #             # Mark end (include short preview for quick grepping)
# # #             prev = (rsp or "")[:300]
# # #             log.emit("run_end", id=sid, label=label, tail_family=fam, ms=t1 - t0, preview=prev, error=err)

# # #             # Also write a compact result line file for convenience
# # #             rec = {
# # #                 "id": sid,
# # #                 "label": label,
# # #                 "tail_family": fam,
# # #                 "elapsed_ms": t1 - t0,
# # #                 "response_preview": prev,
# # #                 "error": err,
# # #             }
# # #             fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
# # #             kept += 1
# # #             if sleep_s > 0:
# # #                 time.sleep(sleep_s)        # ← delay between prompts            

# # #     print(f"[+] Ran {kept}/{n} from {items_path} → {out_path}")

# # # def main():
# # #     ap = argparse.ArgumentParser()
# # #     ap.add_argument("--model", required=True,
# # #                     help='e.g. "llama2-uncensored:georgesung/llama2_7b_chat_uncensored"')
# # #     ap.add_argument("--benign",   help="Path to benign JSONL")
# # #     ap.add_argument("--attacks",  help="Path to compromised/attack JSONL")
# # #     ap.add_argument("--outdir",   default="./run_out", help="Where to write compact result JSONLs")
# # #     ap.add_argument("--sleep", type=float, default=0.0,
# # #                 help="Optional delay (seconds) between prompts")
# # #     args = ap.parse_args()

# # #     outdir = Path(args.outdir)
# # #     outdir.mkdir(parents=True, exist_ok=True)

# # #     # Make sure log path is set (utils/log.py uses AGENT_JSON_LOG)
# # #     log_path = os.getenv("AGENT_JSON_LOG") or str(outdir / "agent.jsonl")
# # #     os.environ["AGENT_JSON_LOG"] = log_path
# # #     log.LOG_PATH = log_path
# # #     print(f"[*] agent log → {log_path}")

# # #     if args.benign:
# # #         run_set(Path(args.benign), model=args.model,
# # #                 out_path=outdir / "benign_agent_runs.jsonl", label_hint="benign", sleep_s=args.sleep)
# # #     if args.attacks:
# # #         run_set(Path(args.attacks), model=args.model,
# # #                 out_path=outdir / "compromised_agent_runs.jsonl", label_hint="attack",sleep_s=args.sleep)

# # # if __name__ == "__main__":
# # #     main()

# # #!/usr/bin/env python3
# # import argparse, json, os
# # from pathlib import Path
# # from datetime import datetime, timezone

# # from utils import process_intermediate_query, log
# # from redteam import compromised # is your llama2-uncensored id


# # def append_jsonl(path: Path, row: dict):
# #     path.parent.mkdir(parents=True, exist_ok=True)
# #     with open(path, "a", encoding="utf-8") as f:
# #         f.write(json.dumps(row, ensure_ascii=False) + "\n")
# #         f.flush()
# #         os.fsync(f.fileno())

# # def iter_jsonl(path: Path):
# #     with path.open("r", encoding="utf-8") as f:
# #         for ln in f:
# #             if ln.strip():
# #                 yield json.loads(ln)

# # # def run_one(question: str):
# # #     # drive the *intermediate* MDT chain (multiple agents → multiple logs)
# # #     args = type("A", (), {"dataset":"medqa","difficulty":"intermediate"})()
# # #     out  = process_intermediate_query(question=question, examplers=[], model_id=model, args=args)
# # #     txt  = json.dumps(out, ensure_ascii=False)
# # #     return txt, compromised(txt)

# # def run_one(question: str, model: str):
# #     args = type("A", (), {"dataset":"medqa","difficulty":"intermediate"})()
# #     out  = process_intermediate_query(
# #         question=question,
# #         examplers=[],          # you can pipe a few MedQA train few-shots later
# #         model_id=model,        # <-- now defined
# #         args=args
# #     )
# #     txt = json.dumps(out, ensure_ascii=False)
# #     return txt, compromised(txt)

# # def main():
# #     ap = argparse.ArgumentParser()
# #     ap.add_argument("--model", required=True)
# #     ap.add_argument("--outdir", required=True)
# #     ap.add_argument("--benign")
# #     ap.add_argument("--attacks")
# #     ap.add_argument("--sleep", type=float, default=0.0)
# #     a = ap.parse_args()

# #     outdir = Path(args.outdir)
# #     jsonl_path = outdir / "agent.jsonl"
# #     print(f"[*] agent log → {jsonl_path}")


# #     # wire utils/log.py to the destination file (per-agent files split by role by your Agent)
# #     agent_json = os.getenv("AGENT_JSON_LOG") or str(outdir / "agent.jsonl")
# #     os.environ["AGENT_JSON_LOG"] = agent_json
# #     log.LOG_PATH = agent_json
# #     print(f"[*] agent log → {agent_json}")

# #     record = {
# #     "label": current_label,           # e.g. "benign"
# #     "question": question_text,
# #     "response": model_output,         # whatever you already have
# #     "compromised": False,             # or from data
# #     "ts": time.strftime('%Y-%m-%dT%H:%M:%S%z')
# # }
# # append_jsonl(jsonl_path, record)

# #     rows = []
# #     def push(q, lbl):
# #         txt, comp = run_one(q, a.model)
# #         rows.append({
# #             "label": lbl,
# #             "question": q,
# #             "response": txt[:2000],
# #             "compromised": comp,
# #             "ts": datetime.now(timezone.utc).isoformat()
# #         })

# #     if a.benign:
# #         for r in iter_jsonl(Path(a.benign)):
# #             push(r.get("full_prompt") or r.get("question_text"), "benign")
# #     if a.attacks:
# #         for r in iter_jsonl(Path(a.attacks)):
# #             push(r.get("full_prompt") or r.get("question_text"), "attack")

# #     (outdir/"results.json").write_text(json.dumps(rows, indent=2), "utf-8")
# #     print(f"[+] wrote {len(rows)} rows")

# # if __name__ == "__main__":
# #     main()


# # #!/usr/bin/env python3
# # import argparse, json, os, time
# # from pathlib import Path
# # from datetime import datetime, timezone
# # import re, json
# # from utils import process_intermediate_query, log
# # from redteam import compromised as is_compromised  # bool(text) or function returning bool
# # from telemetry import mint_trace_ctx, compute_cfp, system_trace_token, otel_headers
# # from log_writer import JSONLLogger
# # import time, re

# # def append_jsonl(path: Path, row: dict):
# #     path.parent.mkdir(parents=True, exist_ok=True)
# #     with open(path, "a", encoding="utf-8") as f:
# #         f.write(json.dumps(row, ensure_ascii=False) + "\n")
# #         f.flush()
# #         os.fsync(f.fileno())

# # def iter_jsonl(path: Path):
# #     with path.open("r", encoding="utf-8") as f:
# #         for ln in f:
# #             ln = ln.strip()
# #             if ln:
# #                 yield json.loads(ln)

# # def pick_question(row: dict) -> str:
# #     # be liberal about field names across packs
# #     return (row.get("full_prompt")
# #             or row.get("question")
# #             or row.get("question_text")
# #             or row.get("prompt")
# #             or "").strip()

# # def run_one(question: str, model: str):
# #     # drive the *intermediate* MDT chain
# #     args = type("A", (), {"dataset": "medqa", "difficulty": "intermediate"})()
# #     out  = process_intermediate_query(
# #         question=question,
# #         examplers=[],           # add few-shots if you want
# #         model_id=model,
# #         args=args
# #     )
# #     txt = json.dumps(out, ensure_ascii=False)
# #     return txt, is_compromised(txt)

# # def main():
# #     ap = argparse.ArgumentParser()
# #     ap.add_argument("--model",   required=True)
# #     ap.add_argument("--outdir",  required=True)
# #     ap.add_argument("--benign")
# #     ap.add_argument("--attacks")
# #     ap.add_argument("--sleep", type=float, default=0.0)
# #     a = ap.parse_args()

# #     outdir = Path(a.outdir)
# #     outdir.mkdir(parents=True, exist_ok=True)

# #     jsonl_path = outdir / "agent.jsonl"
# #     print(f"[*] agent log → {jsonl_path}")

# #     logger = JSONLLogger(run_dir=outdir)
# #     run_t0 = time.time()

# #     # wire per-agent logs (if your Agent writes to LOG_PATH)
# #     agent_json = os.getenv("AGENT_JSON_LOG") or str(jsonl_path)
# #     os.environ["AGENT_JSON_LOG"] = agent_json
# #     try:
# #         log.LOG_PATH = agent_json
# #     except Exception:
# #         pass


# #     rows = []
# #     def push(q, lbl):
# #         if not q:
# #             return
# #         rec = {"label": lbl, "question": q, "final_choice": None,
# #             "response": "", "compromised": False,
# #             "ts": datetime.now(timezone.utc).isoformat()}
# #         try:
# #             txt, comp = run_one(q, a.model)
# #             raw_txt = txt if isinstance(txt, str) else json.dumps(txt, ensure_ascii=False)

# #             letter = None
# #             first_block = raw_txt
# #             try:
# #                 obj  = json.loads(raw_txt)
# #                 maj  = obj.get("majority", {})
# #                 first_block = (maj.get("0.0") or maj.get("0.7") or "") or raw_txt
# #                 m = re.search(r'^[ \t]*([ABCD])\b', first_block, re.M)
# #                 letter = m.group(1) if m else None
# #             except Exception:
# #                 pass

# #             rec["final_choice"] = letter
# #             rec["response"]     = first_block[:2000]
# #             rec["compromised"]  = comp
# #         except Exception as e:
# #             rec["response"] = f"[error] {e}"
# #         append_jsonl(jsonl_path, rec)
# #         rows.append(rec)
# #         if a.sleep > 0:
# #             time.sleep(a.sleep)



# #     try:
# #         if a.benign:
# #             for r in iter_jsonl(Path(a.benign)):
# #                 q = pick_question(r)
# #                 if q:
# #                     push(q, "benign")
# #         if a.attacks:
# #             for r in iter_jsonl(Path(a.attacks)):
# #                 q = pick_question(r)
# #                 if q:
# #                     push(q, "attack")
# #     finally:
# #         (outdir / "results.json").write_text(
# #             json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
# #         )
# #         print(f"[+] wrote {len(rows)} rows")

# # if __name__ == "__main__":
# #     main()



# #!/usr/bin/env python3
# from __future__ import annotations
# import argparse, json, os, time, re
# from pathlib import Path
# from datetime import datetime, timezone

# from utils import process_intermediate_query, log
# from redteam import compromised as is_compromised  # bool(text)
# from telemetry import mint_trace_ctx, compute_cfp, system_trace_token, otel_headers
# from log_writer import JSONLLogger

# # ---------- helpers ----------
# def append_jsonl(path: Path, row: dict):
#     path.parent.mkdir(parents=True, exist_ok=True)
#     with open(path, "a", encoding="utf-8") as f:
#         f.write(json.dumps(row, ensure_ascii=False) + "\n")
#         f.flush()
#         os.fsync(f.fileno())

# def iter_jsonl(path: Path):
#     with path.open("r", encoding="utf-8") as f:
#         for ln in f:
#             ln = ln.strip()
#             if ln:
#                 yield json.loads(ln)

# def pick_question(row: dict) -> str:
#     # be liberal about field names across packs
#     return (row.get("full_prompt")
#             or row.get("question")
#             or row.get("question_text")
#             or row.get("prompt")
#             or "").strip()

# def _has_ctrl_chars(s: str) -> bool:
#     # quick scan for invisibles / control chars
#     return bool(re.search(r'[\u200b-\u200f\u202a-\u202e\p{Cc}]', s))

# # single place to run the MDT chain
# def run_one(question: str, model: str):
#     args = type("A", (), {"dataset": "medqa", "difficulty": "intermediate"})()
#     out = process_intermediate_query(
#         question=question,
#         examplers=[],           # add few-shots if you want
#         model_id=model,
#         args=args
#     )
#     txt = json.dumps(out, ensure_ascii=False)
#     return txt, is_compromised(txt)

# # ---------- main ----------
# def main():
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--model",   required=True)
#     ap.add_argument("--outdir",  required=True)
#     ap.add_argument("--benign")
#     ap.add_argument("--attacks")
#     ap.add_argument("--sleep", type=float, default=0.0)
#     a = ap.parse_args()

#     outdir = Path(a.outdir); outdir.mkdir(parents=True, exist_ok=True)

#     # message log file used by agents/tools (do NOT mix “row” records into this)
#     agent_jsonl_path = outdir / "agent.jsonl"
#     print(f"[*] agent message log → {agent_jsonl_path}")

#     # simple per-prompt “row” log (your summary rows go here)
#     rows_jsonl_path = outdir / "rows.jsonl"

#     # allow inside-agent logging to point at agent.jsonl
#     os.environ["AGENT_JSON_LOG"] = str(agent_jsonl_path)
#     try:
#         log.LOG_PATH = str(agent_jsonl_path)  # if utils.log uses this attr
#     except Exception:
#         pass

#     # orchestrator logger (also writes merged agent.jsonl)
#     logger = JSONLLogger(run_dir=outdir)

#     rows = []

#     def push(q_text: str, lbl: str):
#         if not q_text:
#             return

#         # ---- Sessionization shim per request ----
#         api_key = os.getenv("OPENAI_API_KEY", "")
#         route   = os.getenv("ROUTE_HINT", "/chat/completions")
#         org_id  = os.getenv("ORG_ID", "")
#         cfp = compute_cfp(api_key=api_key, route=route, org_id=org_id)
#         trace_ctx = mint_trace_ctx(sampled=True)
#         # expose for any tools/plugins that pick from env
#         os.environ["TRACE_ID"]      = trace_ctx.trace_id
#         os.environ["TRACEPARENT"]   = trace_ctx.traceparent
#         os.environ["TRACESTATE"]    = trace_ctx.tracestate or ""
#         os.environ["CFP"]           = cfp
#         os.environ["SYSTEM_TRACE_TOKEN"] = system_trace_token(trace_ctx.trace_id, cfp)
#         os.environ["OTEL_HTTP_HEADERS_JSON"] = json.dumps(otel_headers(trace_ctx))

#         # ---- top-level “orchestrator” logging (agent_id "0") ----
#         t_in = time.time()
#         logger.write("0", {
#             "role": "user",
#             "label": lbl,
#             "out": False,
#             "content_len": len(q_text),
#             "has_backticks": ("```" in q_text),
#             "json_brace_count": q_text.count("{") + q_text.count("}"),
#             "has_unicode_ctrl": _has_ctrl_chars(q_text),
#             "trace_id": trace_ctx.trace_id,
#             "traceparent": trace_ctx.traceparent,
#             "tracestate": trace_ctx.tracestate,
#             "cfp": cfp,
#         })

#         # ---- run the chain ----
#         rec = {
#             "label": lbl,
#             "question": q_text,
#             "final_choice": None,
#             "response": "",
#             "compromised": False,
#             "ts": datetime.now(timezone.utc).isoformat()
#         }

#         try:
#             txt, comp = run_one(q_text, a.model)
#             raw_txt = txt if isinstance(txt, str) else json.dumps(txt, ensure_ascii=False)

#             # best-effort “first block” + ABCD extraction for your rows
#             first_block = raw_txt
#             letter = None
#             try:
#                 obj  = json.loads(raw_txt)
#                 maj  = obj.get("majority", {})
#                 first_block = (maj.get("0.0") or maj.get("0.7") or "") or raw_txt
#                 m = re.search(r'^[ \t]*([ABCD])\b', first_block, re.M)
#                 letter = m.group(1) if m else None
#             except Exception:
#                 pass

#             # orchestrator assistant message
#             dt_ms = (time.time() - t_in) * 1000.0
#             logger.write("0", {
#                 "role": "assistant",
#                 "label": lbl,
#                 "out": True,
#                 "content_len": len(first_block),
#                 "has_backticks": ("```" in first_block),
#                 "json_brace_count": first_block.count("{") + first_block.count("}"),
#                 "has_unicode_ctrl": _has_ctrl_chars(first_block),
#                 "latency_ms": round(dt_ms, 1),
#                 "trace_id": trace_ctx.trace_id,
#                 "traceparent": trace_ctx.traceparent,
#                 "tracestate": trace_ctx.tracestate,
#                 "cfp": cfp,
#             })

#             rec["final_choice"] = letter
#             rec["response"]     = first_block[:2000]
#             rec["compromised"]  = comp

#         except Exception as e:
#             rec["response"] = f"[error] {e}"

#         # write summary row (separate file!)
#         append_jsonl(rows_jsonl_path, rec)
#         rows.append(rec)

#         if a.sleep > 0:
#             time.sleep(a.sleep)

#     # ---- drive benign/attack sets ----
#     try:
#         if a.benign:
#             for r in iter_jsonl(Path(a.benign)):
#                 q = pick_question(r)
#                 if q:
#                     push(q, "benign")

#         if a.attacks:
#             for r in iter_jsonl(Path(a.attacks)):
#                 q = pick_question(r)
#                 if q:
#                     push(q, "attack")

#     finally:
#         (outdir / "results.json").write_text(
#             json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
#         )
#         print(f"[+] wrote {len(rows)} rows (rows.jsonl + results.json)")
#         print("[i] per-agent message logs continue to append to agent.jsonl "
#               "and agents/<id>/agent.jsonl via your internal agent logger.")

# if __name__ == "__main__":
#     main()


# #!/usr/bin/env python3
# import argparse
# import json
# import os
# import time
# from pathlib import Path
# from datetime import datetime, timezone
# import re
# import hmac, hashlib

# # Your existing utilities
# from utils import process_intermediate_query, log
# from redteam import compromised as is_compromised  # bool(text) or function returning bool
# import argparse, json, os, time, subprocess, threading, queue, uuid, atexit, re, hmac, hashlib
# from pathlib import Path
# from datetime import datetime, timezone
# # Thin sessionization shim
# # (these helpers are expected in your repo per earlier discussion; we guard them just in case)
# try:
#     from telemetry import mint_trace_ctx, compute_cfp, system_trace_token, otel_headers
# except Exception:  # fallback no-ops if telemetry.py is missing
#     def mint_trace_ctx(**kw):
#         return {"trace_id": os.urandom(16).hex(), "traceparent": "", "tracestate": "", "cfp": ""}
#     def compute_cfp(*a, **kw): return ""
#     def system_trace_token(trace_id): return f"<trace:{trace_id}>"
#     def otel_headers(ctx): return {"X-Alarm-Trace": ctx.get("trace_id", "")}




# # ---------------- NEW: tiny worker pool ----------------
# class AgentWorker:
#     def __init__(self, agent_id:int, agent_name:str, model:str, base_out:Path):
#         self.agent_id = agent_id
#         self.agent_name = agent_name
#         safe = re.sub(r'\W+', '_', agent_name.lower()).strip('_') or f'agent_{agent_id}'
#         self.dir = base_out / "per_agent" / f"agent_{agent_id}.{safe}"
#         self.dir.mkdir(parents=True, exist_ok=True)
#         # Launch the per-agent wrapper (starts strace, pidstat, NVML sampler, cProfile)

#         env = os.environ.copy()
#         run_tag = env.get("RUN_ID") or datetime.now().strftime("%Y%m%d%H%M%S")
#         env["X_ALARM_TRACE"] = f"{run_tag}-agent{agent_id}"
#         env["X_ALARM_AGENT"] = str(agent_id)  # optional extra header/env
#         self.proc = subprocess.Popen(
#             ["bash", "tools/agent_spawn.sh", str(agent_id), agent_name, model, str(self.dir)],
#             stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1, env=env
#         )
#         self._q = queue.Queue()
#         self._reader = threading.Thread(target=self._read_loop, daemon=True); self._reader.start()

#     def _read_loop(self):
#         for line in self.proc.stdout:
#             try: self._q.put(json.loads(line))
#             except Exception: pass

#     def ask(self, prompt:str, rid:str=None, **extra) -> dict:
#         rid = rid or uuid.uuid4().hex
#         payload = {"id": rid, "prompt": prompt, **extra}
#         self.proc.stdin.write(json.dumps(payload) + "\n"); self.proc.stdin.flush()
#         while True:
#             resp = self._q.get()
#             if resp.get("id") == rid:
#                 return resp

#     def close(self):
#         try: self.proc.stdin.close()
#         except Exception: pass
#         try: self.proc.terminate()
#         except Exception: pass

# class WorkerPool:
#     def __init__(self, model:str, outdir:Path):
#         self.model = model
#         self.outdir = outdir
#         self.workers = {}

#     def start(self, agents:list[dict]):
#         # agents is like [{"id":1,"name":"Internist"}, ...]
#         for a in agents:
#             aid = int(a["id"]); nm = a.get("name", f"agent_{aid}")
#             self.workers[aid] = AgentWorker(aid, nm, self.model, self.outdir)
#         atexit.register(self.close)

#     def llm(self, agent_id:int, prompt:str) -> str:
#         resp = self.workers[int(agent_id)].ask(prompt, rid=f"{agent_id}-{uuid.uuid4().hex}")
#         if "error" in resp:
#             raise RuntimeError(resp["error"])
#         return resp.get("text", "")

#     def close(self):
#         for w in self.workers.values():
#             try: w.close()
#             except Exception: pass
# # ---------------- END NEW ----------------

# # ------------------------- helper IO -------------------------
# def append_jsonl(path: Path, row: dict):
#     path.parent.mkdir(parents=True, exist_ok=True)
#     with open(path, "a", encoding="utf-8") as f:
#         f.write(json.dumps(row, ensure_ascii=False) + "\n")
#         f.flush()
#         os.fsync(f.fileno())

# def iter_jsonl(path: Path):
#     with path.open("r", encoding="utf-8") as f:
#         for ln in f:
#             ln = ln.strip()
#             if ln:
#                 yield json.loads(ln)

# def pick_question(row: dict) -> str:
#     # be liberal about field names across packs
#     return (row.get("full_prompt")
#             or row.get("question")
#             or row.get("question_text")
#             or row.get("prompt")
#             or "").strip()

# # ------------------------- model driver -------------------------
# # def run_one(question: str, model: str):
# #     """Drive the *intermediate* MDT chain, return raw JSON string and compromise flag."""
# #     args = type("A", (), {"dataset": "medqa", "difficulty": "intermediate"})()
# #     out = process_intermediate_query(
# #         question=question,
# #         examplers=[],           # add few-shots if you want
# #         model_id=model,
# #         args=args
# #     )
# #     txt = json.dumps(out, ensure_ascii=False)
# #     return txt, is_compromised(txt)

# def run_one(question: str, model: str, *, llm=None, on_agent_roster=None):
#     """Drive the MDT chain, but allow per-agent LLM + roster callback."""
#     args = type("A", (), {"dataset": "medqa", "difficulty": "intermediate"})()
#     # Try new signature first (with hooks), fall back to old if not supported
#     try:
#         out = process_intermediate_query(
#             question=question,
#             examplers=[],
#             model_id=model,
#             args=args,
#             llm=llm,
#             on_agent_roster=on_agent_roster,
#         )
#     except TypeError:
#         # Your utils not patched yet -> old behavior
#         out = process_intermediate_query(
#             question=question,
#             examplers=[],
#             model_id=model,
#             args=args,
#         )
#     txt = json.dumps(out, ensure_ascii=False)
#     return txt, is_compromised(txt)

# # ------------------------- main -------------------------
# def main():
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--model", required=True)
#     ap.add_argument("--outdir", required=True)
#     ap.add_argument("--benign")
#     ap.add_argument("--attacks")
#     ap.add_argument("--sleep", type=float, default=0.0)
#     a = ap.parse_args()

#     hmac_key = os.getenv("LOG_HMAC_KEY", "")
#     salt     = os.getenv("LOG_SALT", "")
#     bucket_s = int(os.getenv("CFP_BUCKET_SECS", "300") or "300")

#     outdir = Path(a.outdir); outdir.mkdir(parents=True, exist_ok=True)
#     jsonl_path = outdir / "agent.jsonl"
#     print(f"[*] agent log → {jsonl_path}")

#     # wire your logger env (unchanged)
#     agent_json = os.getenv("AGENT_JSON_LOG") or str(jsonl_path)
#     os.environ["AGENT_JSON_LOG"] = agent_json
#     try: log.LOG_PATH = agent_json
#     except Exception: pass

#     # ---- NEW: prepare worker pool; start after we learn the roster
#     pool = WorkerPool(a.model, outdir)
#     def _on_roster(agents):  # called once from utils after roster is formed
#         """
#         agents should be a list of dicts like [{"id":1,"name":"Internist"}, ...].
#         We normalize a few common shapes and start one worker per agent.
#         """
#         print(f"[workers] on_agent_roster called with: {agents}")
#         norm = []
#         if isinstance(agents, dict):
#             # e.g. {"agents":[...]} or {"roster":[...]}
#             candidates = agents.get("agents") or agents.get("roster") or []
#         else:
#             candidates = agents or []

#         if candidates and isinstance(candidates[0], dict):
#             for i, a in enumerate(candidates, 1):
#                 aid = int(a.get("id", i))
#                 nm  = a.get("name") or a.get("role") or a.get("title") or f"agent_{aid}"
#                 norm.append({"id": aid, "name": nm})
#         else:
#             # fallback: just enumerate
#             for i, nm in enumerate(candidates, 1):
#                 norm.append({"id": i, "name": str(nm)})

#         if not norm:
#             # ultra-conservative fallback so we still see something
#             norm = [{"id": 1, "name": "agent_1"}]

#         pool.start(norm)
#         for a in norm:
#             print(f"[workers] spawned target for Agent {a['id']} ({a['name']})")

#     def _llm(agent_id:int, prompt:str) -> str:
#         return pool.llm(agent_id, prompt)

#     rows = []
#     # ap = argparse.ArgumentParser()
#     # ap.add_argument("--model",   required=True)
#     # ap.add_argument("--outdir",  required=True)
#     # ap.add_argument("--benign")
#     # ap.add_argument("--attacks")
#     # ap.add_argument("--sleep", type=float, default=0.0)
#     # a = ap.parse_args()

#     # outdir = Path(a.outdir)
#     # outdir.mkdir(parents=True, exist_ok=True)

#     # jsonl_path = outdir / "agent.jsonl"
#     # print(f"[*] agent log → {jsonl_path}")

#     # # if your per-agent logger reads this env, wire it up
#     # agent_json = os.getenv("AGENT_JSON_LOG") or str(jsonl_path)
#     # os.environ["AGENT_JSON_LOG"] = agent_json
#     # try:
#     #     log.LOG_PATH = agent_json
#     # except Exception:
#     #     pass

#     # # sessionization shim inputs (safe defaults are exported by the bash launcher)
#     # hmac_key = os.getenv("LOG_HMAC_KEY", "")
#     # salt     = os.getenv("LOG_SALT", "")
#     # bucket_s = int(os.getenv("CFP_BUCKET_SECS", "300") or "300")

#     # rows = []

#     def make_trace(route: str = "mdagents.generate", org_id_hash: str = "") -> dict:
#         try:
#             ctx = mint_trace_ctx(route=route, org_id_hash=org_id_hash)
#             if not ctx.get("cfp"):
#                 # try library impl
#                 ctx["cfp"] = compute_cfp(
#                     key=hmac_key,
#                     time_bucket=int(time.time() // bucket_s),
#                     api_key_hash="",
#                     route_hash=re.sub(r"[^a-z0-9_.-]+", "_", route),
#                     org_id_hash=org_id_hash or "",
#                     salt=salt,
#                 ) or ""

#             if not ctx.get("cfp"):
#                 # local HMAC fallback (content-blind fingerprint)
#                 bucket_start = int(time.time() // bucket_s) * bucket_s
#                 msg = f"{bucket_start}|{'':s}|{re.sub(r'[^a-z0-9_.-]+','_',route)}|{org_id_hash or ''}|{salt}".encode()
#                 key = (hmac_key or "deadbeef").encode()
#                 ctx["cfp"] = hmac.new(key, msg, hashlib.sha256).hexdigest()
            
#             # if not ctx.get("cfp"):
#             #     ctx["cfp"] = compute_cfp(
#             #         key=hmac_key,
#             #         time_bucket=int(time.time() // bucket_s),
#             #         api_key_hash="",
#             #         route_hash=re.sub(r"[^a-z0-9_.-]+", "_", route),
#             #         org_id_hash=org_id_hash or "",
#             #         salt=salt,
#             #     )
#             ctx["headers"] = otel_headers(ctx)
#             ctx["token"] = system_trace_token(ctx["trace_id"])
#             return ctx
#         except Exception:
#             # ultra-conservative fallback
#             trace_id = os.urandom(16).hex()
#             return {"trace_id": trace_id, "traceparent": "", "tracestate": "",
#                     "cfp": "", "headers": {"X-Alarm-Trace": trace_id},
#                     "token": f"<trace:{trace_id}>"}

#     def extract_majority_letter(raw_txt: str):
#         letter = None
#         first_block = raw_txt
#         try:
#             obj  = json.loads(raw_txt)
#             maj  = obj.get("majority", {})
#             first_block = (maj.get("0.0") or maj.get("0.7") or "") or raw_txt
#             m = re.search(r'^[ \t]*([ABCD])\b', first_block, re.M)
#             letter = m.group(1) if m else None
#         except Exception:
#             pass
#         return letter, first_block

#     def push(q, lbl):
#         if not q: return
#         tctx = make_trace()
#         q_with_token = f"{tctx['token']} {q}".strip()

#         # Export trace headers so tools/retrievers can pick them up implicitly
#         os.environ["X_ALARM_TRACE"] = tctx.get("trace_id", "") or ""
#         if tctx.get("traceparent"): os.environ["TRACEPARENT"] = tctx["traceparent"]
#         if tctx.get("tracestate"):  os.environ["TRACESTATE"]  = tctx["tracestate"]
#         for hk, hv in (tctx.get("headers") or {}).items():
#             if isinstance(hk, str) and isinstance(hv, str):
#                 os.environ[hk] = hv


#         rec = {"label": lbl, "question": q, "final_choice": None,
#                "response": "", "compromised": False,
#                "ts": datetime.now(timezone.utc).isoformat(),
#                "trace_id": tctx.get("trace_id"),
#                "traceparent": tctx.get("traceparent"),
#                "tracestate": tctx.get("tracestate"),
#                "cfp": tctx.get("cfp"),
#                "x_alarm_trace": (tctx.get("headers") or {}).get("X-Alarm-Trace", tctx.get("trace_id"))}
        
#         # print("[workers] DI hooks ACTIVE: llm + on_agent_roster")
#         # txt, comp = run_one(q_with_token, a.model, llm=_llm, on_agent_roster=_on_roster)


#         try:
#             # >>> HERE: pass hooks so each agent routes to its worker PID <<<
#             txt, comp = run_one(q_with_token, a.model, llm=_llm, on_agent_roster=_on_roster)
#             letter, block = extract_majority_letter(txt)
#             rec["final_choice"] = letter
#             rec["response"]     = block[:2000]
#             rec["compromised"]  = comp
#         except Exception as e:
#             rec["response"] = f"[error] {e}"

#         append_jsonl(jsonl_path, rec); rows.append(rec)
#         if a.sleep > 0: time.sleep(a.sleep)

#     try:
#         if a.benign:
#             for r in iter_jsonl(Path(a.benign)):
#                 q = pick_question(r)
#                 if q: push(q, "benign")
#         if a.attacks:
#             for r in iter_jsonl(Path(a.attacks)):
#                 q = pick_question(r)
#                 if q: push(q, "attack")
#     finally:
#         (outdir / "results.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
#         try: pool.close()
#         except Exception: pass
#         print(f"[+] wrote {len(rows)} rows")

# if __name__ == "__main__":
#     main()


# #         rec = {
# #             "label": lbl,
# #             "question": q,
# #             "final_choice": None,
# #             "response": "",
# #             "compromised": False,
# #             "ts": datetime.now(timezone.utc).isoformat(),
# #             # sessionization fields
# #             "trace_id": tctx.get("trace_id"),
# #             "traceparent": tctx.get("traceparent"),
# #             "tracestate": tctx.get("tracestate"),
# #             "cfp": tctx.get("cfp"),
# #             "x_alarm_trace": (tctx.get("headers") or {}).get("X-Alarm-Trace", tctx.get("trace_id")),
# #         }


# #         # light-weight “request” event for per-agent loggers (if used by your Agents)
# #         try:
# #             log.write_event(
# #                 agent="orchestrator",
# #                 event="request",
# #                 meta={"trace_id": rec["trace_id"], "cfp": rec["cfp"], "headers": tctx.get("headers", {})},
# #                 text=q_with_token,
# #             )
# #         except Exception:
# #             pass

# #         try:
# #             txt, comp = run_one(q_with_token, a.model)
# #             letter, block = extract_majority_letter(txt)
# #             rec["final_choice"] = letter
# #             rec["response"]     = block[:2000]
# #             rec["compromised"]  = comp
# #         except Exception as e:
# #             rec["response"] = f"[error] {e}"

# #         append_jsonl(jsonl_path, rec)
# #         rows.append(rec)
# #         if a.sleep > 0:
# #             time.sleep(a.sleep)

# #     try:
# #         if a.benign:
# #             for r in iter_jsonl(Path(a.benign)):
# #                 q = pick_question(r)
# #                 if q:
# #                     push(q, "benign")
# #         if a.attacks:
# #             for r in iter_jsonl(Path(a.attacks)):
# #                 q = pick_question(r)
# #                 if q:
# #                     push(q, "attack")
# #     finally:
# #         (outdir / "results.json").write_text(
# #             json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
# #         )
# #         print(f"[+] wrote {len(rows)} rows")

# # if __name__ == "__main__":
# #     main()
# # # # #!/usr/bin/env python3
# # # # import os, json, time, argparse, uuid
# # # # from pathlib import Path

# # # # # set log path AFTER import by assigning to log.LOG_PATH
# # # # # so we can switch files in the same process.
# # # # def set_log_path(log, path: str):
# # # #     Path(path).parent.mkdir(parents=True, exist_ok=True)
# # # #     log.LOG_PATH = path

# # # # def iter_jsonl(path):
# # # #     with open(path, "r", encoding="utf-8") as f:
# # # #         for ln in f:
# # # #             ln = ln.strip()
# # # #             if ln:
# # # #                 yield json.loads(ln)

# # # # def run_file(jsonl_path: str, *, model: str, role: str, log_path: str, sleep_s: float = 0.0):
# # # #     # late import so we can set log.LOG_PATH before use
# # # #     from utils import log
# # # #     from utils import Agent

# # # #     set_log_path(log, log_path)
# # # #     n = 0
# # # #     for row in iter_jsonl(jsonl_path):
# # # #         sample_id   = row.get("id")                # e.g., medqa_attack_cand_...
# # # #         label       = row.get("label")             # "benign" or "attack"
# # # #         tail_family = row.get("tail_family")       # None for benign
# # # #         prompt      = row["full_prompt"]

# # # #         # Mark the start of a sample in the log
# # # #         run_id = str(uuid.uuid4())
# # # #         log.emit("run_start",
# # # #                  run_id=run_id,
# # # #                  sample_id=sample_id,
# # # #                  label=label,
# # # #                  tail_family=tail_family)

# # # #         # Fresh Agent per prompt so no conversation carryover
# # # #         agent = Agent(instruction=os.environ.get("AGENT_INSTRUCTION", "You are a helpful agent."), role=role, model_info=model)
# # # #         _ = agent.chat(prompt)

# # # #         # Mark the end of a sample in the log
# # # #         log.emit("run_end",
# # # #                  run_id=run_id,
# # # #                  sample_id=sample_id)

# # # #         n += 1
# # # #         if sleep_s:
# # # #             time.sleep(sleep_s)
# # # #     return n

# # # # def main():
# # # #     ap = argparse.ArgumentParser()
# # # #     ap.add_argument("--model", default="llama2-uncensored",
# # # #                     help="Model name passed to Agent(instruction=os.environ.get("AGENT_INSTRUCTION"), model_info=...).")
# # # #     ap.add_argument("--role", default="target",
# # # #                     help="Role string to tag in logs (e.g., 'target').")
# # # #     ap.add_argument("--benign", type=str, default=None,
# # # #                     help="Path to benign JSONL (e.g., full_pack/benign_preview.jsonl).")
# # # #     ap.add_argument("--attacks", type=str, default=None,
# # # #                     help="Path to compromised JSONL (e.g., full_pack/compromised_preview.jsonl).")
# # # #     ap.add_argument("--outdir", type=str, default="logs",
# # # #                     help="Directory for log files.")
# # # #     ap.add_argument("--sleep", type=float, default=0.0,
# # # #                     help="Optional sleep(seconds) between prompts.")
# # # #     args = ap.parse_args()

# # # #     outdir = Path(args.outdir)
# # # #     outdir.mkdir(parents=True, exist_ok=True)

# # # #     total = 0
# # # #     if args.benign:
# # # #         n = run_file(
# # # #             args.benign,
# # # #             model=args.model,
# # # #             role=args.role,
# # # #             log_path=str(outdir / "benign_agent.jsonl"),
# # # #             sleep_s=args.sleep,
# # # #         )
# # # #         print(f"[✓] Logged {n} benign prompts → {outdir/'benign_agent.jsonl'}")
# # # #         total += n

# # # #     if args.attacks:
# # # #         n = run_file(
# # # #             args.attacks,
# # # #             model=args.model,
# # # #             role=args.role,
# # # #             log_path=str(outdir / "compromised_agent.jsonl"),
# # # #             sleep_s=args.sleep,
# # # #         )
# # # #         print(f"[✓] Logged {n} compromised prompts → {outdir/'compromised_agent.jsonl'}")
# # # #         total += n

# # # #     if total == 0:
# # # #         print("No input files provided. Use --benign and/or --attacks.")

# # # # if __name__ == "__main__":
# # # #     main()
# # # #!/usr/bin/env python3
# # # """
# # # generate_agent_logs.py
# # # Run a set of MCQ prompts (benign + compromised) through utils.Agent
# # # and emit detailed per-message logs via utils/log.py.

# # # It expects rows shaped like your preview/full pack:
# # # {
# # #   "id": "...",
# # #   "label": "benign" | "attack",
# # #   "tail_family": "json_plain" | ... | null,
# # #   "full_prompt": "MCQ ... (A) ... (B) ... --- <tail-if-any>",
# # #   ...
# # # }
# # # """

# # # import argparse, json, os, sys, time
# # # import os
# # # from pathlib import Path
# # # from utils import Agent, log
# # # log = log  # back-compat alias

# # # # Your utilities (Agent + JSONL logger)
# # # from utils import Agent, log  # type: ignore
# # # log = log  # back-compat alias

# # # def now_ms() -> int:
# # #     return int(time.time() * 1000)

# # # def iter_jsonl(path: Path):
# # #     with path.open("r", encoding="utf-8") as f:
# # #         for line in f:
# # #             line = line.strip()
# # #             if not line:
# # #                 continue
# # #             try:
# # #                 yield json.loads(line)
# # #             except Exception as e:
# # #                 log.emit("jsonl_parse_error", file=str(path), err=str(e), raw=line[:500])

# # # def run_set(items_path: Path, *, model: str, out_path: Path,
# # #             label_hint: str | None, sleep_s: float = 0.0):
# # #     n = 0
# # #     kept = 0
# # #     with out_path.open("w", encoding="utf-8") as fout:
# # #         for row in iter_jsonl(items_path):
# # #             n += 1
# # #             sid   = row.get("id", f"sample_{n:06d}")
# # #             label = row.get("label", label_hint or "unknown")
# # #             fam   = row.get("tail_family")
# # #             text  = row.get("full_prompt") or row.get("question_text")

# # #             if not text:
# # #                 log.emit("skip_missing_text", id=sid, label=label)
# # #                 continue

# # #             # Fresh Agent for isolation (HF model is cached internally)
# # #             agent = Agent(instruction=os.environ.get("AGENT_INSTRUCTION", "You are a helpful agent."), role="intruder", model_info=model)

# # #             # Mark start
# # #             log.emit("run_start", id=sid, label=label, tail_family=fam)

# # #             err = None
# # #             rsp = None
# # #             t0 = now_ms()
# # #             try:
# # #                 rsp = agent.chat(text)
# # #             except Exception as e:
# # #                 err = str(e)
# # #                 log.emit("run_error", id=sid, label=label, tail_family=fam, err=err)
# # #             t1 = now_ms()

# # #             # Mark end (include short preview for quick grepping)
# # #             prev = (rsp or "")[:300]
# # #             log.emit("run_end", id=sid, label=label, tail_family=fam, ms=t1 - t0, preview=prev, error=err)

# # #             # Also write a compact result line file for convenience
# # #             rec = {
# # #                 "id": sid,
# # #                 "label": label,
# # #                 "tail_family": fam,
# # #                 "elapsed_ms": t1 - t0,
# # #                 "response_preview": prev,
# # #                 "error": err,
# # #             }
# # #             fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
# # #             kept += 1
# # #             if sleep_s > 0:
# # #                 time.sleep(sleep_s)        # ← delay between prompts            

# # #     print(f"[+] Ran {kept}/{n} from {items_path} → {out_path}")

# # # def main():
# # #     ap = argparse.ArgumentParser()
# # #     ap.add_argument("--model", required=True,
# # #                     help='e.g. "llama2-uncensored:georgesung/llama2_7b_chat_uncensored"')
# # #     ap.add_argument("--benign",   help="Path to benign JSONL")
# # #     ap.add_argument("--attacks",  help="Path to compromised/attack JSONL")
# # #     ap.add_argument("--outdir",   default="./run_out", help="Where to write compact result JSONLs")
# # #     ap.add_argument("--sleep", type=float, default=0.0,
# # #                 help="Optional delay (seconds) between prompts")
# # #     args = ap.parse_args()

# # #     outdir = Path(args.outdir)
# # #     outdir.mkdir(parents=True, exist_ok=True)

# # #     # Make sure log path is set (utils/log.py uses AGENT_JSON_LOG)
# # #     log_path = os.getenv("AGENT_JSON_LOG") or str(outdir / "agent.jsonl")
# # #     os.environ["AGENT_JSON_LOG"] = log_path
# # #     log.LOG_PATH = log_path
# # #     print(f"[*] agent log → {log_path}")

# # #     if args.benign:
# # #         run_set(Path(args.benign), model=args.model,
# # #                 out_path=outdir / "benign_agent_runs.jsonl", label_hint="benign", sleep_s=args.sleep)
# # #     if args.attacks:
# # #         run_set(Path(args.attacks), model=args.model,
# # #                 out_path=outdir / "compromised_agent_runs.jsonl", label_hint="attack",sleep_s=args.sleep)

# # # if __name__ == "__main__":
# # #     main()

# # #!/usr/bin/env python3
# # import argparse, json, os
# # from pathlib import Path
# # from datetime import datetime, timezone

# # from utils import process_intermediate_query, log
# # from redteam import compromised # is your llama2-uncensored id


# # def append_jsonl(path: Path, row: dict):
# #     path.parent.mkdir(parents=True, exist_ok=True)
# #     with open(path, "a", encoding="utf-8") as f:
# #         f.write(json.dumps(row, ensure_ascii=False) + "\n")
# #         f.flush()
# #         os.fsync(f.fileno())

# # def iter_jsonl(path: Path):
# #     with path.open("r", encoding="utf-8") as f:
# #         for ln in f:
# #             if ln.strip():
# #                 yield json.loads(ln)

# # # def run_one(question: str):
# # #     # drive the *intermediate* MDT chain (multiple agents → multiple logs)
# # #     args = type("A", (), {"dataset":"medqa","difficulty":"intermediate"})()
# # #     out  = process_intermediate_query(question=question, examplers=[], model_id=model, args=args)
# # #     txt  = json.dumps(out, ensure_ascii=False)
# # #     return txt, compromised(txt)

# # def run_one(question: str, model: str):
# #     args = type("A", (), {"dataset":"medqa","difficulty":"intermediate"})()
# #     out  = process_intermediate_query(
# #         question=question,
# #         examplers=[],          # you can pipe a few MedQA train few-shots later
# #         model_id=model,        # <-- now defined
# #         args=args
# #     )
# #     txt = json.dumps(out, ensure_ascii=False)
# #     return txt, compromised(txt)

# # def main():
# #     ap = argparse.ArgumentParser()
# #     ap.add_argument("--model", required=True)
# #     ap.add_argument("--outdir", required=True)
# #     ap.add_argument("--benign")
# #     ap.add_argument("--attacks")
# #     ap.add_argument("--sleep", type=float, default=0.0)
# #     a = ap.parse_args()

# #     outdir = Path(args.outdir)
# #     jsonl_path = outdir / "agent.jsonl"
# #     print(f"[*] agent log → {jsonl_path}")


# #     # wire utils/log.py to the destination file (per-agent files split by role by your Agent)
# #     agent_json = os.getenv("AGENT_JSON_LOG") or str(outdir / "agent.jsonl")
# #     os.environ["AGENT_JSON_LOG"] = agent_json
# #     log.LOG_PATH = agent_json
# #     print(f"[*] agent log → {agent_json}")

# #     record = {
# #     "label": current_label,           # e.g. "benign"
# #     "question": question_text,
# #     "response": model_output,         # whatever you already have
# #     "compromised": False,             # or from data
# #     "ts": time.strftime('%Y-%m-%dT%H:%M:%S%z')
# # }
# # append_jsonl(jsonl_path, record)

# #     rows = []
# #     def push(q, lbl):
# #         txt, comp = run_one(q, a.model)
# #         rows.append({
# #             "label": lbl,
# #             "question": q,
# #             "response": txt[:2000],
# #             "compromised": comp,
# #             "ts": datetime.now(timezone.utc).isoformat()
# #         })

# #     if a.benign:
# #         for r in iter_jsonl(Path(a.benign)):
# #             push(r.get("full_prompt") or r.get("question_text"), "benign")
# #     if a.attacks:
# #         for r in iter_jsonl(Path(a.attacks)):
# #             push(r.get("full_prompt") or r.get("question_text"), "attack")

# #     (outdir/"results.json").write_text(json.dumps(rows, indent=2), "utf-8")
# #     print(f"[+] wrote {len(rows)} rows")

# # if __name__ == "__main__":
# #     main()


# # #!/usr/bin/env python3
# # import argparse, json, os, time
# # from pathlib import Path
# # from datetime import datetime, timezone
# # import re, json
# # from utils import process_intermediate_query, log
# # from redteam import compromised as is_compromised  # bool(text) or function returning bool
# # from telemetry import mint_trace_ctx, compute_cfp, system_trace_token, otel_headers
# # from log_writer import JSONLLogger
# # import time, re

# # def append_jsonl(path: Path, row: dict):
# #     path.parent.mkdir(parents=True, exist_ok=True)
# #     with open(path, "a", encoding="utf-8") as f:
# #         f.write(json.dumps(row, ensure_ascii=False) + "\n")
# #         f.flush()
# #         os.fsync(f.fileno())

# # def iter_jsonl(path: Path):
# #     with path.open("r", encoding="utf-8") as f:
# #         for ln in f:
# #             ln = ln.strip()
# #             if ln:
# #                 yield json.loads(ln)

# # def pick_question(row: dict) -> str:
# #     # be liberal about field names across packs
# #     return (row.get("full_prompt")
# #             or row.get("question")
# #             or row.get("question_text")
# #             or row.get("prompt")
# #             or "").strip()

# # def run_one(question: str, model: str):
# #     # drive the *intermediate* MDT chain
# #     args = type("A", (), {"dataset": "medqa", "difficulty": "intermediate"})()
# #     out  = process_intermediate_query(
# #         question=question,
# #         examplers=[],           # add few-shots if you want
# #         model_id=model,
# #         args=args
# #     )
# #     txt = json.dumps(out, ensure_ascii=False)
# #     return txt, is_compromised(txt)

# # def main():
# #     ap = argparse.ArgumentParser()
# #     ap.add_argument("--model",   required=True)
# #     ap.add_argument("--outdir",  required=True)
# #     ap.add_argument("--benign")
# #     ap.add_argument("--attacks")
# #     ap.add_argument("--sleep", type=float, default=0.0)
# #     a = ap.parse_args()

# #     outdir = Path(a.outdir)
# #     outdir.mkdir(parents=True, exist_ok=True)

# #     jsonl_path = outdir / "agent.jsonl"
# #     print(f"[*] agent log → {jsonl_path}")

# #     logger = JSONLLogger(run_dir=outdir)
# #     run_t0 = time.time()

# #     # wire per-agent logs (if your Agent writes to LOG_PATH)
# #     agent_json = os.getenv("AGENT_JSON_LOG") or str(jsonl_path)
# #     os.environ["AGENT_JSON_LOG"] = agent_json
# #     try:
# #         log.LOG_PATH = agent_json
# #     except Exception:
# #         pass


# #     rows = []
# #     def push(q, lbl):
# #         if not q:
# #             return
# #         rec = {"label": lbl, "question": q, "final_choice": None,
# #             "response": "", "compromised": False,
# #             "ts": datetime.now(timezone.utc).isoformat()}
# #         try:
# #             txt, comp = run_one(q, a.model)
# #             raw_txt = txt if isinstance(txt, str) else json.dumps(txt, ensure_ascii=False)

# #             letter = None
# #             first_block = raw_txt
# #             try:
# #                 obj  = json.loads(raw_txt)
# #                 maj  = obj.get("majority", {})
# #                 first_block = (maj.get("0.0") or maj.get("0.7") or "") or raw_txt
# #                 m = re.search(r'^[ \t]*([ABCD])\b', first_block, re.M)
# #                 letter = m.group(1) if m else None
# #             except Exception:
# #                 pass

# #             rec["final_choice"] = letter
# #             rec["response"]     = first_block[:2000]
# #             rec["compromised"]  = comp
# #         except Exception as e:
# #             rec["response"] = f"[error] {e}"
# #         append_jsonl(jsonl_path, rec)
# #         rows.append(rec)
# #         if a.sleep > 0:
# #             time.sleep(a.sleep)



# #     try:
# #         if a.benign:
# #             for r in iter_jsonl(Path(a.benign)):
# #                 q = pick_question(r)
# #                 if q:
# #                     push(q, "benign")
# #         if a.attacks:
# #             for r in iter_jsonl(Path(a.attacks)):
# #                 q = pick_question(r)
# #                 if q:
# #                     push(q, "attack")
# #     finally:
# #         (outdir / "results.json").write_text(
# #             json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
# #         )
# #         print(f"[+] wrote {len(rows)} rows")

# # if __name__ == "__main__":
# #     main()



# #!/usr/bin/env python3
# from __future__ import annotations
# import argparse, json, os, time, re
# from pathlib import Path
# from datetime import datetime, timezone

# from utils import process_intermediate_query, log
# from redteam import compromised as is_compromised  # bool(text)
# from telemetry import mint_trace_ctx, compute_cfp, system_trace_token, otel_headers
# from log_writer import JSONLLogger

# # ---------- helpers ----------
# def append_jsonl(path: Path, row: dict):
#     path.parent.mkdir(parents=True, exist_ok=True)
#     with open(path, "a", encoding="utf-8") as f:
#         f.write(json.dumps(row, ensure_ascii=False) + "\n")
#         f.flush()
#         os.fsync(f.fileno())

# def iter_jsonl(path: Path):
#     with path.open("r", encoding="utf-8") as f:
#         for ln in f:
#             ln = ln.strip()
#             if ln:
#                 yield json.loads(ln)

# def pick_question(row: dict) -> str:
#     # be liberal about field names across packs
#     return (row.get("full_prompt")
#             or row.get("question")
#             or row.get("question_text")
#             or row.get("prompt")
#             or "").strip()

# def _has_ctrl_chars(s: str) -> bool:
#     # quick scan for invisibles / control chars
#     return bool(re.search(r'[\u200b-\u200f\u202a-\u202e\p{Cc}]', s))

# # single place to run the MDT chain
# def run_one(question: str, model: str):
#     args = type("A", (), {"dataset": "medqa", "difficulty": "intermediate"})()
#     out = process_intermediate_query(
#         question=question,
#         examplers=[],           # add few-shots if you want
#         model_id=model,
#         args=args
#     )
#     txt = json.dumps(out, ensure_ascii=False)
#     return txt, is_compromised(txt)

# # ---------- main ----------
# def main():
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--model",   required=True)
#     ap.add_argument("--outdir",  required=True)
#     ap.add_argument("--benign")
#     ap.add_argument("--attacks")
#     ap.add_argument("--sleep", type=float, default=0.0)
#     a = ap.parse_args()

#     outdir = Path(a.outdir); outdir.mkdir(parents=True, exist_ok=True)

#     # message log file used by agents/tools (do NOT mix “row” records into this)
#     agent_jsonl_path = outdir / "agent.jsonl"
#     print(f"[*] agent message log → {agent_jsonl_path}")

#     # simple per-prompt “row” log (your summary rows go here)
#     rows_jsonl_path = outdir / "rows.jsonl"

#     # allow inside-agent logging to point at agent.jsonl
#     os.environ["AGENT_JSON_LOG"] = str(agent_jsonl_path)
#     try:
#         log.LOG_PATH = str(agent_jsonl_path)  # if utils.log uses this attr
#     except Exception:
#         pass

#     # orchestrator logger (also writes merged agent.jsonl)
#     logger = JSONLLogger(run_dir=outdir)

#     rows = []

#     def push(q_text: str, lbl: str):
#         if not q_text:
#             return

#         # ---- Sessionization shim per request ----
#         api_key = os.getenv("OPENAI_API_KEY", "")
#         route   = os.getenv("ROUTE_HINT", "/chat/completions")
#         org_id  = os.getenv("ORG_ID", "")
#         cfp = compute_cfp(api_key=api_key, route=route, org_id=org_id)
#         trace_ctx = mint_trace_ctx(sampled=True)
#         # expose for any tools/plugins that pick from env
#         os.environ["TRACE_ID"]      = trace_ctx.trace_id
#         os.environ["TRACEPARENT"]   = trace_ctx.traceparent
#         os.environ["TRACESTATE"]    = trace_ctx.tracestate or ""
#         os.environ["CFP"]           = cfp
#         os.environ["SYSTEM_TRACE_TOKEN"] = system_trace_token(trace_ctx.trace_id, cfp)
#         os.environ["OTEL_HTTP_HEADERS_JSON"] = json.dumps(otel_headers(trace_ctx))

#         # ---- top-level “orchestrator” logging (agent_id "0") ----
#         t_in = time.time()
#         logger.write("0", {
#             "role": "user",
#             "label": lbl,
#             "out": False,
#             "content_len": len(q_text),
#             "has_backticks": ("```" in q_text),
#             "json_brace_count": q_text.count("{") + q_text.count("}"),
#             "has_unicode_ctrl": _has_ctrl_chars(q_text),
#             "trace_id": trace_ctx.trace_id,
#             "traceparent": trace_ctx.traceparent,
#             "tracestate": trace_ctx.tracestate,
#             "cfp": cfp,
#         })

#         # ---- run the chain ----
#         rec = {
#             "label": lbl,
#             "question": q_text,
#             "final_choice": None,
#             "response": "",
#             "compromised": False,
#             "ts": datetime.now(timezone.utc).isoformat()
#         }

#         try:
#             txt, comp = run_one(q_text, a.model)
#             raw_txt = txt if isinstance(txt, str) else json.dumps(txt, ensure_ascii=False)

#             # best-effort “first block” + ABCD extraction for your rows
#             first_block = raw_txt
#             letter = None
#             try:
#                 obj  = json.loads(raw_txt)
#                 maj  = obj.get("majority", {})
#                 first_block = (maj.get("0.0") or maj.get("0.7") or "") or raw_txt
#                 m = re.search(r'^[ \t]*([ABCD])\b', first_block, re.M)
#                 letter = m.group(1) if m else None
#             except Exception:
#                 pass

#             # orchestrator assistant message
#             dt_ms = (time.time() - t_in) * 1000.0
#             logger.write("0", {
#                 "role": "assistant",
#                 "label": lbl,
#                 "out": True,
#                 "content_len": len(first_block),
#                 "has_backticks": ("```" in first_block),
#                 "json_brace_count": first_block.count("{") + first_block.count("}"),
#                 "has_unicode_ctrl": _has_ctrl_chars(first_block),
#                 "latency_ms": round(dt_ms, 1),
#                 "trace_id": trace_ctx.trace_id,
#                 "traceparent": trace_ctx.traceparent,
#                 "tracestate": trace_ctx.tracestate,
#                 "cfp": cfp,
#             })

#             rec["final_choice"] = letter
#             rec["response"]     = first_block[:2000]
#             rec["compromised"]  = comp

#         except Exception as e:
#             rec["response"] = f"[error] {e}"

#         # write summary row (separate file!)
#         append_jsonl(rows_jsonl_path, rec)
#         rows.append(rec)

#         if a.sleep > 0:
#             time.sleep(a.sleep)

#     # ---- drive benign/attack sets ----
#     try:
#         if a.benign:
#             for r in iter_jsonl(Path(a.benign)):
#                 q = pick_question(r)
#                 if q:
#                     push(q, "benign")

#         if a.attacks:
#             for r in iter_jsonl(Path(a.attacks)):
#                 q = pick_question(r)
#                 if q:
#                     push(q, "attack")

#     finally:
#         (outdir / "results.json").write_text(
#             json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
#         )
#         print(f"[+] wrote {len(rows)} rows (rows.jsonl + results.json)")
#         print("[i] per-agent message logs continue to append to agent.jsonl "
#               "and agents/<id>/agent.jsonl via your internal agent logger.")

# if __name__ == "__main__":
#     main()

# # # # #!/usr/bin/env python3
# # # # import os, json, time, argparse, uuid
# # # # from pathlib import Path

# # # # # set log path AFTER import by assigning to log.LOG_PATH
# # # # # so we can switch files in the same process.
# # # # def set_log_path(log, path: str):
# # # #     Path(path).parent.mkdir(parents=True, exist_ok=True)
# # # #     log.LOG_PATH = path

# # # # def iter_jsonl(path):
# # # #     with open(path, "r", encoding="utf-8") as f:
# # # #         for ln in f:
# # # #             ln = ln.strip()
# # # #             if ln:
# # # #                 yield json.loads(ln)

# # # # def run_file(jsonl_path: str, *, model: str, role: str, log_path: str, sleep_s: float = 0.0):
# # # #     # late import so we can set log.LOG_PATH before use
# # # #     from utils import log
# # # #     from utils import Agent

# # # #     set_log_path(log, log_path)
# # # #     n = 0
# # # #     for row in iter_jsonl(jsonl_path):
# # # #         sample_id   = row.get("id")                # e.g., medqa_attack_cand_...
# # # #         label       = row.get("label")             # "benign" or "attack"
# # # #         tail_family = row.get("tail_family")       # None for benign
# # # #         prompt      = row["full_prompt"]

# # # #         # Mark the start of a sample in the log
# # # #         run_id = str(uuid.uuid4())
# # # #         log.emit("run_start",
# # # #                  run_id=run_id,
# # # #                  sample_id=sample_id,
# # # #                  label=label,
# # # #                  tail_family=tail_family)

# # # #         # Fresh Agent per prompt so no conversation carryover
# # # #         agent = Agent(instruction=os.environ.get("AGENT_INSTRUCTION", "You are a helpful agent."), role=role, model_info=model)
# # # #         _ = agent.chat(prompt)

# # # #         # Mark the end of a sample in the log
# # # #         log.emit("run_end",
# # # #                  run_id=run_id,
# # # #                  sample_id=sample_id)

# # # #         n += 1
# # # #         if sleep_s:
# # # #             time.sleep(sleep_s)
# # # #     return n

# # # # def main():
# # # #     ap = argparse.ArgumentParser()
# # # #     ap.add_argument("--model", default="llama2-uncensored",
# # # #                     help="Model name passed to Agent(instruction=os.environ.get("AGENT_INSTRUCTION"), model_info=...).")
# # # #     ap.add_argument("--role", default="target",
# # # #                     help="Role string to tag in logs (e.g., 'target').")
# # # #     ap.add_argument("--benign", type=str, default=None,
# # # #                     help="Path to benign JSONL (e.g., full_pack/benign_preview.jsonl).")
# # # #     ap.add_argument("--attacks", type=str, default=None,
# # # #                     help="Path to compromised JSONL (e.g., full_pack/compromised_preview.jsonl).")
# # # #     ap.add_argument("--outdir", type=str, default="logs",
# # # #                     help="Directory for log files.")
# # # #     ap.add_argument("--sleep", type=float, default=0.0,
# # # #                     help="Optional sleep(seconds) between prompts.")
# # # #     args = ap.parse_args()

# # # #     outdir = Path(args.outdir)
# # # #     outdir.mkdir(parents=True, exist_ok=True)

# # # #     total = 0
# # # #     if args.benign:
# # # #         n = run_file(
# # # #             args.benign,
# # # #             model=args.model,
# # # #             role=args.role,
# # # #             log_path=str(outdir / "benign_agent.jsonl"),
# # # #             sleep_s=args.sleep,
# # # #         )
# # # #         print(f"[✓] Logged {n} benign prompts → {outdir/'benign_agent.jsonl'}")
# # # #         total += n

# # # #     if args.attacks:
# # # #         n = run_file(
# # # #             args.attacks,
# # # #             model=args.model,
# # # #             role=args.role,
# # # #             log_path=str(outdir / "compromised_agent.jsonl"),
# # # #             sleep_s=args.sleep,
# # # #         )
# # # #         print(f"[✓] Logged {n} compromised prompts → {outdir/'compromised_agent.jsonl'}")
# # # #         total += n

# # # #     if total == 0:
# # # #         print("No input files provided. Use --benign and/or --attacks.")

# # # # if __name__ == "__main__":
# # # #     main()
# # # #!/usr/bin/env python3
# # # """
# # # generate_agent_logs.py
# # # Run a set of MCQ prompts (benign + compromised) through utils.Agent
# # # and emit detailed per-message logs via utils/log.py.

# # # It expects rows shaped like your preview/full pack:
# # # {
# # #   "id": "...",
# # #   "label": "benign" | "attack",
# # #   "tail_family": "json_plain" | ... | null,
# # #   "full_prompt": "MCQ ... (A) ... (B) ... --- <tail-if-any>",
# # #   ...
# # # }
# # # """

# # # import argparse, json, os, sys, time
# # # import os
# # # from pathlib import Path
# # # from utils import Agent, log
# # # log = log  # back-compat alias

# # # # Your utilities (Agent + JSONL logger)
# # # from utils import Agent, log  # type: ignore
# # # log = log  # back-compat alias

# # # def now_ms() -> int:
# # #     return int(time.time() * 1000)

# # # def iter_jsonl(path: Path):
# # #     with path.open("r", encoding="utf-8") as f:
# # #         for line in f:
# # #             line = line.strip()
# # #             if not line:
# # #                 continue
# # #             try:
# # #                 yield json.loads(line)
# # #             except Exception as e:
# # #                 log.emit("jsonl_parse_error", file=str(path), err=str(e), raw=line[:500])

# # # def run_set(items_path: Path, *, model: str, out_path: Path,
# # #             label_hint: str | None, sleep_s: float = 0.0):
# # #     n = 0
# # #     kept = 0
# # #     with out_path.open("w", encoding="utf-8") as fout:
# # #         for row in iter_jsonl(items_path):
# # #             n += 1
# # #             sid   = row.get("id", f"sample_{n:06d}")
# # #             label = row.get("label", label_hint or "unknown")
# # #             fam   = row.get("tail_family")
# # #             text  = row.get("full_prompt") or row.get("question_text")

# # #             if not text:
# # #                 log.emit("skip_missing_text", id=sid, label=label)
# # #                 continue

# # #             # Fresh Agent for isolation (HF model is cached internally)
# # #             agent = Agent(instruction=os.environ.get("AGENT_INSTRUCTION", "You are a helpful agent."), role="intruder", model_info=model)

# # #             # Mark start
# # #             log.emit("run_start", id=sid, label=label, tail_family=fam)

# # #             err = None
# # #             rsp = None
# # #             t0 = now_ms()
# # #             try:
# # #                 rsp = agent.chat(text)
# # #             except Exception as e:
# # #                 err = str(e)
# # #                 log.emit("run_error", id=sid, label=label, tail_family=fam, err=err)
# # #             t1 = now_ms()

# # #             # Mark end (include short preview for quick grepping)
# # #             prev = (rsp or "")[:300]
# # #             log.emit("run_end", id=sid, label=label, tail_family=fam, ms=t1 - t0, preview=prev, error=err)

# # #             # Also write a compact result line file for convenience
# # #             rec = {
# # #                 "id": sid,
# # #                 "label": label,
# # #                 "tail_family": fam,
# # #                 "elapsed_ms": t1 - t0,
# # #                 "response_preview": prev,
# # #                 "error": err,
# # #             }
# # #             fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
# # #             kept += 1
# # #             if sleep_s > 0:
# # #                 time.sleep(sleep_s)        # ← delay between prompts            

# # #     print(f"[+] Ran {kept}/{n} from {items_path} → {out_path}")

# # # def main():
# # #     ap = argparse.ArgumentParser()
# # #     ap.add_argument("--model", required=True,
# # #                     help='e.g. "llama2-uncensored:georgesung/llama2_7b_chat_uncensored"')
# # #     ap.add_argument("--benign",   help="Path to benign JSONL")
# # #     ap.add_argument("--attacks",  help="Path to compromised/attack JSONL")
# # #     ap.add_argument("--outdir",   default="./run_out", help="Where to write compact result JSONLs")
# # #     ap.add_argument("--sleep", type=float, default=0.0,
# # #                 help="Optional delay (seconds) between prompts")
# # #     args = ap.parse_args()

# # #     outdir = Path(args.outdir)
# # #     outdir.mkdir(parents=True, exist_ok=True)

# # #     # Make sure log path is set (utils/log.py uses AGENT_JSON_LOG)
# # #     log_path = os.getenv("AGENT_JSON_LOG") or str(outdir / "agent.jsonl")
# # #     os.environ["AGENT_JSON_LOG"] = log_path
# # #     log.LOG_PATH = log_path
# # #     print(f"[*] agent log → {log_path}")

# # #     if args.benign:
# # #         run_set(Path(args.benign), model=args.model,
# # #                 out_path=outdir / "benign_agent_runs.jsonl", label_hint="benign", sleep_s=args.sleep)
# # #     if args.attacks:
# # #         run_set(Path(args.attacks), model=args.model,
# # #                 out_path=outdir / "compromised_agent_runs.jsonl", label_hint="attack",sleep_s=args.sleep)

# # # if __name__ == "__main__":
# # #     main()

# # #!/usr/bin/env python3
# # import argparse, json, os
# # from pathlib import Path
# # from datetime import datetime, timezone

# # from utils import process_intermediate_query, log
# # from redteam import compromised # is your llama2-uncensored id


# # def append_jsonl(path: Path, row: dict):
# #     path.parent.mkdir(parents=True, exist_ok=True)
# #     with open(path, "a", encoding="utf-8") as f:
# #         f.write(json.dumps(row, ensure_ascii=False) + "\n")
# #         f.flush()
# #         os.fsync(f.fileno())

# # def iter_jsonl(path: Path):
# #     with path.open("r", encoding="utf-8") as f:
# #         for ln in f:
# #             if ln.strip():
# #                 yield json.loads(ln)

# # # def run_one(question: str):
# # #     # drive the *intermediate* MDT chain (multiple agents → multiple logs)
# # #     args = type("A", (), {"dataset":"medqa","difficulty":"intermediate"})()
# # #     out  = process_intermediate_query(question=question, examplers=[], model_id=model, args=args)
# # #     txt  = json.dumps(out, ensure_ascii=False)
# # #     return txt, compromised(txt)

# # def run_one(question: str, model: str):
# #     args = type("A", (), {"dataset":"medqa","difficulty":"intermediate"})()
# #     out  = process_intermediate_query(
# #         question=question,
# #         examplers=[],          # you can pipe a few MedQA train few-shots later
# #         model_id=model,        # <-- now defined
# #         args=args
# #     )
# #     txt = json.dumps(out, ensure_ascii=False)
# #     return txt, compromised(txt)

# # def main():
# #     ap = argparse.ArgumentParser()
# #     ap.add_argument("--model", required=True)
# #     ap.add_argument("--outdir", required=True)
# #     ap.add_argument("--benign")
# #     ap.add_argument("--attacks")
# #     ap.add_argument("--sleep", type=float, default=0.0)
# #     a = ap.parse_args()

# #     outdir = Path(args.outdir)
# #     jsonl_path = outdir / "agent.jsonl"
# #     print(f"[*] agent log → {jsonl_path}")


# #     # wire utils/log.py to the destination file (per-agent files split by role by your Agent)
# #     agent_json = os.getenv("AGENT_JSON_LOG") or str(outdir / "agent.jsonl")
# #     os.environ["AGENT_JSON_LOG"] = agent_json
# #     log.LOG_PATH = agent_json
# #     print(f"[*] agent log → {agent_json}")

# #     record = {
# #     "label": current_label,           # e.g. "benign"
# #     "question": question_text,
# #     "response": model_output,         # whatever you already have
# #     "compromised": False,             # or from data
# #     "ts": time.strftime('%Y-%m-%dT%H:%M:%S%z')
# # }
# # append_jsonl(jsonl_path, record)

# #     rows = []
# #     def push(q, lbl):
# #         txt, comp = run_one(q, a.model)
# #         rows.append({
# #             "label": lbl,
# #             "question": q,
# #             "response": txt[:2000],
# #             "compromised": comp,
# #             "ts": datetime.now(timezone.utc).isoformat()
# #         })

# #     if a.benign:
# #         for r in iter_jsonl(Path(a.benign)):
# #             push(r.get("full_prompt") or r.get("question_text"), "benign")
# #     if a.attacks:
# #         for r in iter_jsonl(Path(a.attacks)):
# #             push(r.get("full_prompt") or r.get("question_text"), "attack")

# #     (outdir/"results.json").write_text(json.dumps(rows, indent=2), "utf-8")
# #     print(f"[+] wrote {len(rows)} rows")

# # if __name__ == "__main__":
# #     main()


# # #!/usr/bin/env python3
# # import argparse, json, os, time
# # from pathlib import Path
# # from datetime import datetime, timezone
# # import re, json
# # from utils import process_intermediate_query, log
# # from redteam import compromised as is_compromised  # bool(text) or function returning bool
# # from telemetry import mint_trace_ctx, compute_cfp, system_trace_token, otel_headers
# # from log_writer import JSONLLogger
# # import time, re

# # def append_jsonl(path: Path, row: dict):
# #     path.parent.mkdir(parents=True, exist_ok=True)
# #     with open(path, "a", encoding="utf-8") as f:
# #         f.write(json.dumps(row, ensure_ascii=False) + "\n")
# #         f.flush()
# #         os.fsync(f.fileno())

# # def iter_jsonl(path: Path):
# #     with path.open("r", encoding="utf-8") as f:
# #         for ln in f:
# #             ln = ln.strip()
# #             if ln:
# #                 yield json.loads(ln)

# # def pick_question(row: dict) -> str:
# #     # be liberal about field names across packs
# #     return (row.get("full_prompt")
# #             or row.get("question")
# #             or row.get("question_text")
# #             or row.get("prompt")
# #             or "").strip()

# # def run_one(question: str, model: str):
# #     # drive the *intermediate* MDT chain
# #     args = type("A", (), {"dataset": "medqa", "difficulty": "intermediate"})()
# #     out  = process_intermediate_query(
# #         question=question,
# #         examplers=[],           # add few-shots if you want
# #         model_id=model,
# #         args=args
# #     )
# #     txt = json.dumps(out, ensure_ascii=False)
# #     return txt, is_compromised(txt)

# # def main():
# #     ap = argparse.ArgumentParser()
# #     ap.add_argument("--model",   required=True)
# #     ap.add_argument("--outdir",  required=True)
# #     ap.add_argument("--benign")
# #     ap.add_argument("--attacks")
# #     ap.add_argument("--sleep", type=float, default=0.0)
# #     a = ap.parse_args()

# #     outdir = Path(a.outdir)
# #     outdir.mkdir(parents=True, exist_ok=True)

# #     jsonl_path = outdir / "agent.jsonl"
# #     print(f"[*] agent log → {jsonl_path}")

# #     logger = JSONLLogger(run_dir=outdir)
# #     run_t0 = time.time()

# #     # wire per-agent logs (if your Agent writes to LOG_PATH)
# #     agent_json = os.getenv("AGENT_JSON_LOG") or str(jsonl_path)
# #     os.environ["AGENT_JSON_LOG"] = agent_json
# #     try:
# #         log.LOG_PATH = agent_json
# #     except Exception:
# #         pass


# #     rows = []
# #     def push(q, lbl):
# #         if not q:
# #             return
# #         rec = {"label": lbl, "question": q, "final_choice": None,
# #             "response": "", "compromised": False,
# #             "ts": datetime.now(timezone.utc).isoformat()}
# #         try:
# #             txt, comp = run_one(q, a.model)
# #             raw_txt = txt if isinstance(txt, str) else json.dumps(txt, ensure_ascii=False)

# #             letter = None
# #             first_block = raw_txt
# #             try:
# #                 obj  = json.loads(raw_txt)
# #                 maj  = obj.get("majority", {})
# #                 first_block = (maj.get("0.0") or maj.get("0.7") or "") or raw_txt
# #                 m = re.search(r'^[ \t]*([ABCD])\b', first_block, re.M)
# #                 letter = m.group(1) if m else None
# #             except Exception:
# #                 pass

# #             rec["final_choice"] = letter
# #             rec["response"]     = first_block[:2000]
# #             rec["compromised"]  = comp
# #         except Exception as e:
# #             rec["response"] = f"[error] {e}"
# #         append_jsonl(jsonl_path, rec)
# #         rows.append(rec)
# #         if a.sleep > 0:
# #             time.sleep(a.sleep)



# #     try:
# #         if a.benign:
# #             for r in iter_jsonl(Path(a.benign)):
# #                 q = pick_question(r)
# #                 if q:
# #                     push(q, "benign")
# #         if a.attacks:
# #             for r in iter_jsonl(Path(a.attacks)):
# #                 q = pick_question(r)
# #                 if q:
# #                     push(q, "attack")
# #     finally:
# #         (outdir / "results.json").write_text(
# #             json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
# #         )
# #         print(f"[+] wrote {len(rows)} rows")

# # if __name__ == "__main__":
# #     main()



# #!/usr/bin/env python3
# from __future__ import annotations
# import argparse, json, os, time, re
# from pathlib import Path
# from datetime import datetime, timezone

# from utils import process_intermediate_query, log
# from redteam import compromised as is_compromised  # bool(text)
# from telemetry import mint_trace_ctx, compute_cfp, system_trace_token, otel_headers
# from log_writer import JSONLLogger

# # ---------- helpers ----------
# def append_jsonl(path: Path, row: dict):
#     path.parent.mkdir(parents=True, exist_ok=True)
#     with open(path, "a", encoding="utf-8") as f:
#         f.write(json.dumps(row, ensure_ascii=False) + "\n")
#         f.flush()
#         os.fsync(f.fileno())

# def iter_jsonl(path: Path):
#     with path.open("r", encoding="utf-8") as f:
#         for ln in f:
#             ln = ln.strip()
#             if ln:
#                 yield json.loads(ln)

# def pick_question(row: dict) -> str:
#     # be liberal about field names across packs
#     return (row.get("full_prompt")
#             or row.get("question")
#             or row.get("question_text")
#             or row.get("prompt")
#             or "").strip()

# def _has_ctrl_chars(s: str) -> bool:
#     # quick scan for invisibles / control chars
#     return bool(re.search(r'[\u200b-\u200f\u202a-\u202e\p{Cc}]', s))

# # single place to run the MDT chain
# def run_one(question: str, model: str):
#     args = type("A", (), {"dataset": "medqa", "difficulty": "intermediate"})()
#     out = process_intermediate_query(
#         question=question,
#         examplers=[],           # add few-shots if you want
#         model_id=model,
#         args=args
#     )
#     txt = json.dumps(out, ensure_ascii=False)
#     return txt, is_compromised(txt)

# # ---------- main ----------
# def main():
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--model",   required=True)
#     ap.add_argument("--outdir",  required=True)
#     ap.add_argument("--benign")
#     ap.add_argument("--attacks")
#     ap.add_argument("--sleep", type=float, default=0.0)
#     a = ap.parse_args()

#     outdir = Path(a.outdir); outdir.mkdir(parents=True, exist_ok=True)

#     # message log file used by agents/tools (do NOT mix “row” records into this)
#     agent_jsonl_path = outdir / "agent.jsonl"
#     print(f"[*] agent message log → {agent_jsonl_path}")

#     # simple per-prompt “row” log (your summary rows go here)
#     rows_jsonl_path = outdir / "rows.jsonl"

#     # allow inside-agent logging to point at agent.jsonl
#     os.environ["AGENT_JSON_LOG"] = str(agent_jsonl_path)
#     try:
#         log.LOG_PATH = str(agent_jsonl_path)  # if utils.log uses this attr
#     except Exception:
#         pass

#     # orchestrator logger (also writes merged agent.jsonl)
#     logger = JSONLLogger(run_dir=outdir)

#     rows = []

#     def push(q_text: str, lbl: str):
#         if not q_text:
#             return

#         # ---- Sessionization shim per request ----
#         api_key = os.getenv("OPENAI_API_KEY", "")
#         route   = os.getenv("ROUTE_HINT", "/chat/completions")
#         org_id  = os.getenv("ORG_ID", "")
#         cfp = compute_cfp(api_key=api_key, route=route, org_id=org_id)
#         trace_ctx = mint_trace_ctx(sampled=True)
#         # expose for any tools/plugins that pick from env
#         os.environ["TRACE_ID"]      = trace_ctx.trace_id
#         os.environ["TRACEPARENT"]   = trace_ctx.traceparent
#         os.environ["TRACESTATE"]    = trace_ctx.tracestate or ""
#         os.environ["CFP"]           = cfp
#         os.environ["SYSTEM_TRACE_TOKEN"] = system_trace_token(trace_ctx.trace_id, cfp)
#         os.environ["OTEL_HTTP_HEADERS_JSON"] = json.dumps(otel_headers(trace_ctx))

#         # ---- top-level “orchestrator” logging (agent_id "0") ----
#         t_in = time.time()
#         logger.write("0", {
#             "role": "user",
#             "label": lbl,
#             "out": False,
#             "content_len": len(q_text),
#             "has_backticks": ("```" in q_text),
#             "json_brace_count": q_text.count("{") + q_text.count("}"),
#             "has_unicode_ctrl": _has_ctrl_chars(q_text),
#             "trace_id": trace_ctx.trace_id,
#             "traceparent": trace_ctx.traceparent,
#             "tracestate": trace_ctx.tracestate,
#             "cfp": cfp,
#         })

#         # ---- run the chain ----
#         rec = {
#             "label": lbl,
#             "question": q_text,
#             "final_choice": None,
#             "response": "",
#             "compromised": False,
#             "ts": datetime.now(timezone.utc).isoformat()
#         }

#         try:
#             txt, comp = run_one(q_text, a.model)
#             raw_txt = txt if isinstance(txt, str) else json.dumps(txt, ensure_ascii=False)

#             # best-effort “first block” + ABCD extraction for your rows
#             first_block = raw_txt
#             letter = None
#             try:
#                 obj  = json.loads(raw_txt)
#                 maj  = obj.get("majority", {})
#                 first_block = (maj.get("0.0") or maj.get("0.7") or "") or raw_txt
#                 m = re.search(r'^[ \t]*([ABCD])\b', first_block, re.M)
#                 letter = m.group(1) if m else None
#             except Exception:
#                 pass

#             # orchestrator assistant message
#             dt_ms = (time.time() - t_in) * 1000.0
#             logger.write("0", {
#                 "role": "assistant",
#                 "label": lbl,
#                 "out": True,
#                 "content_len": len(first_block),
#                 "has_backticks": ("```" in first_block),
#                 "json_brace_count": first_block.count("{") + first_block.count("}"),
#                 "has_unicode_ctrl": _has_ctrl_chars(first_block),
#                 "latency_ms": round(dt_ms, 1),
#                 "trace_id": trace_ctx.trace_id,
#                 "traceparent": trace_ctx.traceparent,
#                 "tracestate": trace_ctx.tracestate,
#                 "cfp": cfp,
#             })

#             rec["final_choice"] = letter
#             rec["response"]     = first_block[:2000]
#             rec["compromised"]  = comp

#         except Exception as e:
#             rec["response"] = f"[error] {e}"

#         # write summary row (separate file!)
#         append_jsonl(rows_jsonl_path, rec)
#         rows.append(rec)

#         if a.sleep > 0:
#             time.sleep(a.sleep)

#     # ---- drive benign/attack sets ----
#     try:
#         if a.benign:
#             for r in iter_jsonl(Path(a.benign)):
#                 q = pick_question(r)
#                 if q:
#                     push(q, "benign")

#         if a.attacks:
#             for r in iter_jsonl(Path(a.attacks)):
#                 q = pick_question(r)
#                 if q:
#                     push(q, "attack")

#     finally:
#         (outdir / "results.json").write_text(
#             json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
#         )
#         print(f"[+] wrote {len(rows)} rows (rows.jsonl + results.json)")
#         print("[i] per-agent message logs continue to append to agent.jsonl "
#               "and agents/<id>/agent.jsonl via your internal agent logger.")

# if __name__ == "__main__":
#     main()


#!/usr/bin/env python3
import argparse
import json
import os
import time
from pathlib import Path
from datetime import datetime, timezone
import re
import hmac, hashlib

# Your existing utilities
from utils import process_intermediate_query, log
from redteam import compromised as is_compromised  # bool(text) or function returning bool

# Thin sessionization shim
# (these helpers are expected in your repo per earlier discussion; we guard them just in case)
try:
    from telemetry import mint_trace_ctx, compute_cfp, system_trace_token, otel_headers
except Exception:  # fallback no-ops if telemetry.py is missing
    def mint_trace_ctx(**kw):
        return {"trace_id": os.urandom(16).hex(), "traceparent": "", "tracestate": "", "cfp": ""}
    def compute_cfp(*a, **kw): return ""
    def system_trace_token(trace_id): return f"<trace:{trace_id}>"
    def otel_headers(ctx): return {"X-Alarm-Trace": ctx.get("trace_id", "")}

# ------------------------- helper IO -------------------------
def append_jsonl(path: Path, row: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())

def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if ln:
                yield json.loads(ln)

def pick_question(row: dict) -> str:
    # be liberal about field names across packs
    return (row.get("full_prompt")
            or row.get("question")
            or row.get("question_text")
            or row.get("prompt")
            or "").strip()

# ------------------------- model driver -------------------------
def run_one(question: str, model: str):
    """Drive the *intermediate* MDT chain, return raw JSON string and compromise flag."""
    args = type("A", (), {"dataset": "medqa", "difficulty": "intermediate"})()
    out = process_intermediate_query(
        question=question,
        examplers=[],           # add few-shots if you want
        model_id=model,
        args=args
    )
    txt = json.dumps(out, ensure_ascii=False)
    return txt, is_compromised(txt)

# ------------------------- main -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model",   required=True)
    ap.add_argument("--outdir",  required=True)
    ap.add_argument("--benign")
    ap.add_argument("--attacks")
    ap.add_argument("--sleep", type=float, default=0.0)
    a = ap.parse_args()

    outdir = Path(a.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    jsonl_path = outdir / "agent.jsonl"
    print(f"[*] agent log → {jsonl_path}")

    # if your per-agent logger reads this env, wire it up
    agent_json = os.getenv("AGENT_JSON_LOG") or str(jsonl_path)
    os.environ["AGENT_JSON_LOG"] = agent_json
    try:
        log.LOG_PATH = agent_json
    except Exception:
        pass

    # sessionization shim inputs (safe defaults are exported by the bash launcher)
    hmac_key = os.getenv("LOG_HMAC_KEY", "")
    salt     = os.getenv("LOG_SALT", "")
    bucket_s = int(os.getenv("CFP_BUCKET_SECS", "300") or "300")

    rows = []

    def make_trace(route: str = "mdagents.generate", org_id_hash: str = "") -> dict:
        try:
            ctx = mint_trace_ctx(route=route, org_id_hash=org_id_hash)
            if not ctx.get("cfp"):
                # try library impl
                ctx["cfp"] = compute_cfp(
                    key=hmac_key,
                    time_bucket=int(time.time() // bucket_s),
                    api_key_hash="",
                    route_hash=re.sub(r"[^a-z0-9_.-]+", "_", route),
                    org_id_hash=org_id_hash or "",
                    salt=salt,
                ) or ""

            if not ctx.get("cfp"):
                # local HMAC fallback (content-blind fingerprint)
                bucket_start = int(time.time() // bucket_s) * bucket_s
                msg = f"{bucket_start}|{'':s}|{re.sub(r'[^a-z0-9_.-]+','_',route)}|{org_id_hash or ''}|{salt}".encode()
                key = (hmac_key or "deadbeef").encode()
                ctx["cfp"] = hmac.new(key, msg, hashlib.sha256).hexdigest()
            
            # if not ctx.get("cfp"):
            #     ctx["cfp"] = compute_cfp(
            #         key=hmac_key,
            #         time_bucket=int(time.time() // bucket_s),
            #         api_key_hash="",
            #         route_hash=re.sub(r"[^a-z0-9_.-]+", "_", route),
            #         org_id_hash=org_id_hash or "",
            #         salt=salt,
            #     )
            ctx["headers"] = otel_headers(ctx)
            ctx["token"] = system_trace_token(ctx["trace_id"])
            return ctx
        except Exception:
            # ultra-conservative fallback
            trace_id = os.urandom(16).hex()
            return {"trace_id": trace_id, "traceparent": "", "tracestate": "",
                    "cfp": "", "headers": {"X-Alarm-Trace": trace_id},
                    "token": f"<trace:{trace_id}>"}

    def extract_majority_letter(raw_txt: str):
        letter = None
        first_block = raw_txt
        try:
            obj  = json.loads(raw_txt)
            maj  = obj.get("majority", {})
            first_block = (maj.get("0.0") or maj.get("0.7") or "") or raw_txt
            m = re.search(r'^[ \t]*([ABCD])\b', first_block, re.M)
            letter = m.group(1) if m else None
        except Exception:
            pass
        return letter, first_block

    def push(q, lbl):
        if not q:
            return
        tctx = make_trace()
        q_with_token = f"{tctx['token']} {q}".strip()

        # Export trace headers so tools/retrievers can pick them up implicitly
        os.environ["X_ALARM_TRACE"] = tctx.get("trace_id", "") or ""
        if tctx.get("traceparent"): os.environ["TRACEPARENT"] = tctx["traceparent"]
        if tctx.get("tracestate"):  os.environ["TRACESTATE"]  = tctx["tracestate"]
        for hk, hv in (tctx.get("headers") or {}).items():
            if isinstance(hk, str) and isinstance(hv, str):
                os.environ[hk] = hv



        rec = {
            "label": lbl,
            "question": q,
            "final_choice": None,
            "response": "",
            "compromised": False,
            "ts": datetime.now(timezone.utc).isoformat(),
            # sessionization fields
            "trace_id": tctx.get("trace_id"),
            "traceparent": tctx.get("traceparent"),
            "tracestate": tctx.get("tracestate"),
            "cfp": tctx.get("cfp"),
            "x_alarm_trace": (tctx.get("headers") or {}).get("X-Alarm-Trace", tctx.get("trace_id")),
        }


        # light-weight “request” event for per-agent loggers (if used by your Agents)
        try:
            log.write_event(
                agent="orchestrator",
                event="request",
                meta={"trace_id": rec["trace_id"], "cfp": rec["cfp"], "headers": tctx.get("headers", {})},
                text=q_with_token,
            )
        except Exception:
            pass

        try:
            txt, comp = run_one(q_with_token, a.model)
            letter, block = extract_majority_letter(txt)
            rec["final_choice"] = letter
            rec["response"]     = block[:2000]
            rec["compromised"]  = comp
        except Exception as e:
            rec["response"] = f"[error] {e}"

        append_jsonl(jsonl_path, rec)
        rows.append(rec)
        if a.sleep > 0:
            time.sleep(a.sleep)

    try:
        if a.benign:
            for r in iter_jsonl(Path(a.benign)):
                q = pick_question(r)
                if q:
                    push(q, "benign")
        if a.attacks:
            for r in iter_jsonl(Path(a.attacks)):
                q = pick_question(r)
                if q:
                    push(q, "attack")
    finally:
        (outdir / "results.json").write_text(
            json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"[+] wrote {len(rows)} rows")

if __name__ == "__main__":
    main()

