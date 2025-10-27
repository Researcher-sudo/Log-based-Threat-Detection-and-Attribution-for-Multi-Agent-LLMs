#!/usr/bin/env python3
import sys, json, subprocess
from pathlib import Path

def main():
    if len(sys.argv) != 3:
        print("Usage: tools/pcap_per_agent.py <RUN_DIR> <PCAP>", file=sys.stderr); sys.exit(2)
    run_dir = Path(sys.argv[1]).resolve()
    pcap = Path(sys.argv[2]).resolve()
    per_agent = run_dir / "per_agent"
    if not per_agent.exists() or not pcap.exists():
        sys.exit(0)

    for adir in sorted(per_agent.glob("agent_*.*")):
        # Try to collect all trace ids seen in this agent’s JSONL (x_alarm_trace if present)
        ids = set()
        aj = adir / "agent.jsonl"
        if not aj.exists():
            continue
        for ln in aj.read_text(encoding="utf-8", errors="ignore").splitlines():
            if not ln.strip():
                continue
            try:
                row = json.loads(ln)
            except Exception:
                continue
            tid = (row.get("x_alarm_trace") or row.get("trace_id") or "").strip()
            if tid:
                ids.add(tid)

        if not ids:
            continue

        # Build a Tshark display filter that matches any of the headers
        # We check both X-Alarm-Trace and traceparent
        parts = []
        for tid in ids:
            parts.append(f'http.request.line matches "X-Alarm-Trace: {tid}"')
            parts.append(f'http.request.line matches "traceparent: .*{tid}.*"')
        dfilter = "http && (" + " or ".join(parts) + ")"

        out_tsv = adir / "net.tsv"
        cmd = [
            "tshark", "-r", str(pcap),
            "-Y", dfilter,
            "-T", "fields",
            "-e", "frame.time",
            "-e", "ip.src",
            "-e", "ip.dst",
            "-e", "http.request.method",
            "-e", "http.host",
            "-e", "http.request.uri",
        ]
        try:
            res = subprocess.run(cmd, capture_output=True, text=True, check=False)
            if res.stdout.strip():
                out_tsv.write_text(res.stdout, encoding="utf-8")
        except FileNotFoundError:
            pass

    print("[+] wrote per-agent net.tsv where headers matched")

if __name__ == "__main__":
    main()
