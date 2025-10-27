#!/usr/bin/env python3
import sys, glob, re, collections
if len(sys.argv) < 2:
    print("usage: strace_stats.py STRACE_BASE [OUT_TSV]", file=sys.stderr); sys.exit(2)
base = sys.argv[1]
out = sys.argv[2] if len(sys.argv) > 2 else base + "_stats.tsv"
counts = collections.Counter()
for path in glob.glob(base + ".*"):
    try:
        with open(path, "r", errors="ignore") as f:
            for ln in f:
                m = re.match(r".*?\s+([a-z_][a-z0-9_]*)\(", ln)
                if m: counts[m.group(1)] += 1
    except Exception:
        pass
with open(out, "w") as w:
    w.write("syscall\tcount\n")
    for k, v in counts.most_common():
        w.write(f"{k}\t{v}\n")
print(f"[strace] wrote {out}")
