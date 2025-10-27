#!/usr/bin/env python3
import re, os, sys, json, hmac, hashlib, time, shutil
from pathlib import Path
from datetime import datetime, timezone

AGENT_DEF_RE = re.compile(r'^Agent\s+(\d+)\s+\((?:[^)]*?\s)?(?:\d+\.\s*)?([^)]+?)\):\s*(.*)$')
INTERACTION_HDR_RE = re.compile(r'^\s*Agent\s+(\d+)\s+\(([^)]*)\)\s*->\s*Agent\s+(\d+)\s+\(([^)]*)\)\s*:\s*$')
ROUND_RE = re.compile(r'^==\s*Round\s+(\d+)\s*==')
TURN_RE  = re.compile(r'^\|_Turn\s+(\d+)')
NOISE_PREFIXES = ("RAW participate", "RAW chosen_expert", "[INFO]", "[LOG]", "[USER]",
                  "The following generation flags", "Loading checkpoint shards", "Device set to use")

def slugify(s: str) -> str:
    s = re.sub(r'[\W_]+', '_', s.strip().lower())
    return re.sub(r'_+', '_', s).strip('_') or "agent"

def now_iso(): return datetime.now(timezone.utc).isoformat()

def mint_trace_id(): return os.urandom(16).hex()     # 128-bit hex
def mint_span_id():  return os.urandom(8).hex()      # 64-bit hex
def build_traceparent(trace_id: str): return f"00-{trace_id}-{mint_span_id()}-01"

def compute_cfp(now_ts: int, route_hash="gen_agent_logs", api_key_hash="na", org_id_hash="na"):
    key = (os.environ.get("LOG_HMAC_KEY") or "deadbeef").encode()
    salt = os.environ.get("LOG_SALT") or "host"
    bucket_secs = int(os.environ.get("CFP_BUCKET_SECS") or 300)
    bucket = (now_ts // bucket_secs) * bucket_secs
    msg = f"{bucket}|{api_key_hash}|{route_hash}|{org_id_hash}|{salt}".encode()
    return hmac.new(key, msg, hashlib.sha256).hexdigest()

def link_or_copy(src: Path, dst: Path):
    if not src.exists(): return
    try:
        if dst.exists() or dst.is_symlink(): dst.unlink()
    except: pass
    try:
        os.symlink(src, dst)
    except Exception:
        if src.is_file(): shutil.copy2(src, dst)

def parse_app_log(app_log: Path):
    agents, events = {}, []
    round_no, turn_no = None, None
    lines = app_log.read_text(encoding="utf-8", errors="ignore").splitlines()

    # agent roster
    for ln in lines:
        m = AGENT_DEF_RE.match(ln.strip())
        if m:
            aid, name, tagline = m.groups()
            agents[int(aid)] = {"name": name.strip(), "tagline": tagline.strip()}

    # fallback roster to avoid empty
    if not agents:
        for i in range(1, 10): agents[i] = {"name": f"agent_{i}", "tagline": ""}

    # messages
    i, seen = 0, set()
    while i < len(lines):
        s = lines[i].strip()
        mr = ROUND_RE.match(s)
        if mr: round_no = int(mr.group(1)); i += 1; continue
        mt = TURN_RE.match(s)
        if mt: turn_no  = int(mt.group(1)); i += 1; continue
        if any(s.startswith(p) for p in NOISE_PREFIXES): i += 1; continue

        mh = INTERACTION_HDR_RE.match(s)
        if mh:
            frm_id, to_id = int(mh.group(1)), int(mh.group(3))
            content_lines, j = [], i + 1
            while j < len(lines):
                nxts = lines[j].strip()
                if (INTERACTION_HDR_RE.match(nxts) or AGENT_DEF_RE.match(nxts) or
                    ROUND_RE.match(nxts) or TURN_RE.match(nxts) or
                    any(nxts.startswith(p) for p in NOISE_PREFIXES)):
                    break
                if nxts: content_lines.append(lines[j])
                j += 1
            content = "\n".join(content_lines).strip()
            key = (frm_id, to_id, content[:200])
            if content and key not in seen:
                seen.add(key)
                events.append({
                    "round": round_no, "turn":  turn_no,
                    "from":  {"id": frm_id, "name": agents.get(frm_id, {}).get("name", f"agent_{frm_id}")},
                    "to":    {"id": to_id,  "name": agents.get(to_id,  {}).get("name",  f"agent_{to_id}")},
                    "text":  content,
                    "header": lines[i]
                })
            i = j; continue
        i += 1
    return agents, events

def write_per_agent_bundles(run_dir: Path, agents: dict, events: list):
    per_root = run_dir / "per_agent"
    per_root.mkdir(parents=True, exist_ok=True)

    # run-level metadata from results.json if present
    results_path = run_dir / "results.json"
    results = []
    if results_path.exists():
        try: results = json.loads(results_path.read_text(encoding="utf-8"))
        except Exception: results = []
    meta = results[0] if results else {}

    # sessionization shim (shared across the run)
    trace_id    = meta.get("trace_id")    or mint_trace_id()
    traceparent = meta.get("traceparent") or build_traceparent(trace_id)
    tracestate  = meta.get("tracestate")  or ""
    cfp         = meta.get("cfp")         or compute_cfp(int(time.time()))

    # write shared context
    (per_root / "trace_ctx.json").write_text(
        json.dumps({"trace_id": trace_id, "traceparent": traceparent, "tracestate": tracestate, "cfp": cfp},
                   ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    # prepare per-agent handles + filtered app.log accumulators
    agent_files, filtered_lines = {}, {}
    try:
        for aid, meta_a in agents.items():
            slug = slugify(meta_a.get("name", f"agent_{aid}"))
            agent_dir = per_root / f"agent_{aid}.{slug}"
            agent_dir.mkdir(parents=True, exist_ok=True)

            # per-agent agent.jsonl
            agent_jsonl = agent_dir / "agent.jsonl"
            agent_files[aid] = agent_jsonl.open("w", encoding="utf-8")

            # start filtered log capture
            filtered_lines[aid] = []

            # convenience symlinks/copies
            for fname in ["results.json","app.log","vmstat.log","gpu.csv","profile.pstats",
                          "env.txt","traffic.pcap","http.tsv","dns.tsv"]:
                src = run_dir / fname
                dst = agent_dir / fname
                link_or_copy(src, dst)
            # all strace.* files
            for s in run_dir.glob("strace.*"):
                link_or_copy(s, agent_dir / s.name)

        # write per-message rows (+ filtered logs)
        for ev in events:
            base_row = {
                "ts": now_iso(),
                "round": ev.get("round"),
                "turn":  ev.get("turn"),
                "from_id": ev["from"]["id"],
                "from_name": ev["from"]["name"],
                "to_id": ev["to"]["id"],
                "to_name": ev["to"]["name"],
                "text": ev["text"],
                # run-level metadata expected by you
                "label": meta.get("label"),
                "question": meta.get("question"),
                "final_choice": meta.get("final_choice"),
                "response": meta.get("response"),
                "compromised": meta.get("compromised"),
                # sessionization shim
                "trace_id": trace_id,
                "traceparent": traceparent,
                "tracestate": tracestate,
                "cfp": cfp,
                "x_alarm_trace": trace_id,
            }
            aid = ev["from"]["id"]
            f = agent_files.get(aid)
            if f:
                f.write(json.dumps(base_row, ensure_ascii=False) + "\n")
                # also keep the original header + content for filtered log
                filtered_lines[aid].append(ev["header"])
                if ev["text"]:
                    filtered_lines[aid].extend(ev["text"].splitlines())

        # write filtered app.log for each agent
        for aid, lines in filtered_lines.items():
            meta_a = agents.get(aid, {})
            slug = slugify(meta_a.get("name", f"agent_{aid}"))
            agent_dir = per_root / f"agent_{aid}.{slug}"
            with (agent_dir / "app.filtered.log").open("w", encoding="utf-8") as lf:
                lf.write("\n".join(lines) + "\n")

        # write quick summaries + matrix
        sent = {aid:0 for aid in agents}; recv = {aid:0 for aid in agents}
        matrix = {}
        for ev in events:
            fid, tid = ev["from"]["id"], ev["to"]["id"]
            sent[fid] = sent.get(fid,0)+1; recv[tid] = recv.get(tid,0)+1
            matrix[(fid,tid)] = matrix.get((fid,tid),0)+1

        with (per_root / "agent_summaries.tsv").open("w", encoding="utf-8") as sf:
            sf.write("agent_id\tagent_name\tsent\treceived\n")
            for aid, meta_a in agents.items():
                sf.write(f"{aid}\t{meta_a.get('name','')}\t{sent.get(aid,0)}\t{recv.get(aid,0)}\n")

        with (run_dir / "interactions_matrix.tsv").open("w", encoding="utf-8") as mf:
            mf.write("from_id\tto_id\tcount\n")
            for (fid,tid), c in sorted(matrix.items()):
                mf.write(f"{fid}\t{tid}\t{c}\n")

        # tiny readme per agent
        for aid, meta_a in agents.items():
            slug = slugify(meta_a.get("name", f"agent_{aid}"))
            agent_dir = per_root / f"agent_{aid}.{slug}"
            (agent_dir / "README.txt").write_text(
                "This folder contains per-agent views.\n"
                "- agent.jsonl: per-message events for this agent with trace ctx and CFP.\n"
                "- app.filtered.log: only this agent’s chat lines.\n"
                "- other files: symlinks/copies for convenience.\n",
                encoding="utf-8"
            )
    finally:
        for f in agent_files.values():
            try: f.close()
            except: pass

def main():
    if len(sys.argv) != 2:
        print("Usage: tools/agent_jsonl_stats.py <RUN_DIR>", file=sys.stderr); sys.exit(2)
    run_dir = Path(sys.argv[1]).expanduser().resolve()
    app_log = run_dir / "app.log"
    if not app_log.exists():
        print(f"[!] app.log not found in {run_dir}", file=sys.stderr); sys.exit(1)
    agents, events = parse_app_log(app_log)
    write_per_agent_bundles(run_dir, agents, events)
    print(f"[+] per-agent bundles -> {run_dir/'per_agent'}")
    print(f"[i] agents: {len(agents)}   events: {len(events)}")

if __name__ == "__main__":
    main()
