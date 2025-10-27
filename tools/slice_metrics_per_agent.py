#!/usr/bin/env python3
import sys, json, re
from pathlib import Path
from datetime import datetime, timezone, timedelta

def parse_vmstat_line(s):
    # vmstat -t puts date+time at end; keep flexible
    # Example tail: "... 2025-09-10 16:05:22"
    m = re.search(r'(\d{4}-\d{2}-\d{2})\s+(\d{2}:\d{2}:\d{2})\s*$', s)
    if not m:
        return None, s
    ts = datetime.fromisoformat(f"{m.group(1)} {m.group(2)}")
    return ts, s

def parse_nvsmi_ts(s):
    # nvidia-smi --query-gpu=timestamp,... gives e.g. '2025/09/10 16:05:49'
    for fmt in ("%Y/%m/%d %H:%M:%S", "%Y-%m-%d %H:%M:%S", "%m/%d/%Y %H:%M:%S"):
        try:
            return datetime.strptime(s.strip(), fmt)
        except Exception:
            pass
    return None

def load_agent_windows(per_agent_dir, half_window_sec=5):
    """Return dict[agent_dir] -> list[(start,end)] from per_agent/*/agent.jsonl"""
    windows = {}
    for adir in sorted(per_agent_dir.glob("agent_*.*")):
        aj = adir / "agent.jsonl"
        spans = []
        if aj.exists():
            for ln in aj.read_text(encoding="utf-8", errors="ignore").splitlines():
                if not ln.strip():
                    continue
                try:
                    row = json.loads(ln)
                except Exception:
                    continue
                ts = row.get("ts")
                if not ts:
                    continue
                try:
                    t = datetime.fromisoformat(ts.replace("Z","+00:00")).astimezone(timezone.utc)
                except Exception:
                    continue
                half = timedelta(seconds=half_window_sec)
                spans.append((t-half, t+half))
        if spans:
            windows[adir] = spans
    return windows

def slice_vmstat(run_dir: Path, windows: dict):
    vm = run_dir / "vmstat.log"
    if not vm.exists():
        return
    lines = vm.read_text(encoding="utf-8", errors="ignore").splitlines()
    stamped = []
    for ln in lines:
        ts, full = parse_vmstat_line(ln)
        if ts:
            stamped.append((ts.astimezone(timezone.utc), full))
    for adir, spans in windows.items():
        out = adir / "vmstat.slice.tsv"
        with out.open("w", encoding="utf-8") as f:
            f.write("timestamp\tvmstat_row\n")
            for ts, row in stamped:
                if any(a<=ts<=b for (a,b) in spans):
                    f.write(f"{ts.isoformat()}\t{row}\n")

def slice_gpu(run_dir: Path, windows: dict):
    gpu = run_dir / "gpu.csv"
    if not gpu.exists():
        return
    rows = []
    hdr = None
    for i, ln in enumerate(gpu.read_text(encoding="utf-8", errors="ignore").splitlines()):
        if i == 0:
            hdr = ln
            continue
        parts = [p.strip() for p in ln.split(",")]
        if not parts:
            continue
        ts = parse_nvsmi_ts(parts[0])
        if not ts:
            continue
        rows.append((ts.replace(tzinfo=None), ln))  # treat as naive local
    for adir, spans in windows.items():
        out = adir / "gpu.slice.csv"
        with out.open("w", encoding="utf-8") as f:
            if hdr:
                f.write(hdr + "\n")
            for ts, line in rows:
                if any(a.replace(tzinfo=None)<=ts<=b.replace(tzinfo=None) for (a,b) in spans):
                    f.write(line + "\n")

def main():
    if len(sys.argv) != 2:
        print("Usage: tools/slice_metrics_per_agent.py <RUN_DIR>", file=sys.stderr)
        sys.exit(2)
    run_dir = Path(sys.argv[1]).resolve()
    per_agent = run_dir / "per_agent"
    if not per_agent.exists():
        print(f"[!] {per_agent} not found (run agent_jsonl_stats.py first)", file=sys.stderr)
        sys.exit(1)
    windows = load_agent_windows(per_agent, half_window_sec=5)
    slice_vmstat(run_dir, windows)
    slice_gpu(run_dir, windows)
    print("[+] wrote per-agent vmstat.slice.tsv and gpu.slice.csv where data matched windows")

if __name__ == "__main__":
    main()
