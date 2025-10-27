#!/usr/bin/env python3
import json, csv, re, sys
from pathlib import Path
from collections import defaultdict, Counter
from datetime import datetime, timezone

# Optional plotting
try:
    import networkx as nx
    HAVE_NX = True
except Exception:
    HAVE_NX = False

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAVE_MPL = True
except Exception:
    HAVE_MPL = False


# ---------- time helpers ----------
def parse_iso(ts: str):
    # "2025-09-12T14:08:28.899403"
    try:
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except Exception:
        return None

def parse_nvsmits(ts: str):
    """
    nvidia-smi "timestamp" formats often look like:
      2025/09/12 14:08:29.123
      2025/09/12 14:08:29
    """
    for fmt in ("%Y/%m/%d %H:%M:%S.%f", "%Y/%m/%d %H:%M:%S"):
        try:
            return datetime.strptime(ts, fmt).replace(tzinfo=timezone.utc)
        except Exception:
            pass
    return None


# ---------- GPU sampler load ----------
def load_gpu_csv(run_dir: Path):
    f = run_dir / "gpu.csv"
    rows = []
    if not f.exists():
        return rows
    with f.open(newline="", encoding="utf-8") as fh:
        rdr = csv.reader(fh)
        header = next(rdr, None)
        # Expect: timestamp,util.gpu %,mem.used MB,temp C,power W
        # Be defensive on positions
        for row in rdr:
            if not row:
                continue
            ts = parse_nvsmits(row[0].strip())
            if not ts:
                continue
            try:
                util = float(row[1].split()[0])
            except Exception:
                util = None
            try:
                mem = float(row[2].split()[0])
            except Exception:
                mem = None
            try:
                power = float(row[4].split()[0])
            except Exception:
                power = None
            rows.append((ts, util, mem, power))
    return rows


def avg(vals):
    vals = [v for v in vals if v is not None]
    return sum(vals)/len(vals) if vals else None


# ---------- core extraction ----------
def find_q_dirs(run_dir: Path):
    return sorted([p for p in run_dir.iterdir() if p.is_dir() and p.name.startswith("q_")])

def guess_agents_from_files(qdir: Path):
    """
    Return {idx: role} based on agent_XX_*.jsonl files.
    """
    mapping = {}
    for f in qdir.glob("agent_*.jsonl"):
        m = re.match(r"agent_(\d+)_(.+)\.jsonl$", f.name)
        if m:
            idx = int(m.group(1))
            role = m.group(2).replace("_", " ")
            mapping[idx] = role
    return mapping

def load_edges(qdir: Path):
    """
    interactions.jsonl lines of form:
      {"type":"edge_message", "ts": "...", "round":..., "turn":..., 
       "src":{"idx":i,"role":"..."}, "dst":{"idx":j,"role":"..."},
       "message":"...", "meta":{...}}
    """
    edges = []
    f = qdir / "interactions.jsonl"
    if not f.exists():
        return edges
    with f.open(encoding="utf-8") as fh:
        for line in fh:
            try:
                j = json.loads(line)
            except Exception:
                continue
            if j.get("type") != "edge_message":
                continue
            ts = parse_iso(j.get("ts",""))
            src = j.get("src", {})
            dst = j.get("dst", {})
            meta = j.get("meta", {}) or {}
            msg = (j.get("message") or "").strip()
            edges.append({
                "ts": ts,
                "round": j.get("round"),
                "turn":  j.get("turn"),
                "src_idx": int(src.get("idx", 0) or 0),
                "src_role": src.get("role",""),
                "dst_idx": int(dst.get("idx", 0) or 0),
                "dst_role": dst.get("role",""),
                "msg": msg,
                "meta": meta
            })
    return edges


def build_features_for_q(run_dir: Path, qdir: Path):
    gpu_rows = load_gpu_csv(run_dir)   # [(ts, util, mem, power), ...]
    edges = load_edges(qdir)
    if not edges:
        print(f"[i] {qdir.name}: no edges found; skipping")
        return

    # Map agents
    idx2role = guess_agents_from_files(qdir)
    # fallback from edges if needed
    for e in edges:
        if e["src_idx"] and e["src_idx"] not in idx2role and e["src_role"]:
            idx2role[e["src_idx"]] = e["src_role"]
        if e["dst_idx"] and e["dst_idx"] not in idx2role and e["dst_role"]:
            idx2role[e["dst_idx"]] = e["dst_role"]

    # Aggregators
    node_feats = defaultdict(lambda: {
        "agent_idx": None, "role": "", "model_id": None, "device": None,
        "device_name": None, "vram_gb": None,
        "msgs_sent": 0, "msgs_recv": 0,
        "chars_sent": 0, "chars_recv": 0,
        "prompt_tokens_sum": 0, "completion_tokens_sum": 0,
        "peers_out": Counter(), "peers_in": Counter(),
        "first_ts": None, "last_ts": None
    })
    edge_feats = defaultdict(lambda: {
        "count": 0, "chars_sum": 0,
        "prompt_tokens_sum": 0, "completion_tokens_sum": 0,
        "first_ts": None, "last_ts": None
    })

    # Fill from edges
    for e in edges:
        si, di = e["src_idx"], e["dst_idx"]
        msg = e["msg"]
        meta = e["meta"] or {}
        tokens = (meta.get("tokens") or {})
        p_tok = int(tokens.get("prompt") or 0)
        c_tok = int(tokens.get("completion") or 0)

        # nodes
        n_src = node_feats[si]; n_dst = node_feats[di]
        n_src["agent_idx"] = si; n_src["role"] = idx2role.get(si, n_src["role"])
        n_dst["agent_idx"] = di; n_dst["role"] = idx2role.get(di, n_dst["role"])

        # capture last-seen model/device info from meta
        for n in (n_src, n_dst):
            if meta.get("model_id"):    n["model_id"] = meta.get("model_id")
            if meta.get("device"):      n["device"] = meta.get("device")
            if meta.get("device_name"): n["device_name"] = meta.get("device_name")
            if "vram_gb" in meta:       n["vram_gb"] = meta.get("vram_gb")

        n_src["msgs_sent"] += 1
        n_src["chars_sent"] += len(msg)
        n_src["prompt_tokens_sum"] += p_tok
        n_src["completion_tokens_sum"] += c_tok
        n_src["peers_out"][di] += 1

        n_dst["msgs_recv"] += 1
        n_dst["chars_recv"] += len(msg)
        n_dst["peers_in"][si] += 1

        for n in (n_src, n_dst):
            t = e["ts"]
            if t:
                if n["first_ts"] is None or t < n["first_ts"]:
                    n["first_ts"] = t
                if n["last_ts"] is None or t > n["last_ts"]:
                    n["last_ts"] = t

        # edges
        key = (si, di)
        ef = edge_feats[key]
        ef["count"] += 1
        ef["chars_sum"] += len(msg)
        ef["prompt_tokens_sum"] += p_tok
        ef["completion_tokens_sum"] += c_tok
        t = e["ts"]
        if t:
            if ef["first_ts"] is None or t < ef["first_ts"]:
                ef["first_ts"] = t
            if ef["last_ts"] is None or t > ef["last_ts"]:
                ef["last_ts"] = t

    # Attach GPU aggregates over each agent’s active window
    if gpu_rows:
        for idx, nf in node_feats.items():
            ft, lt = nf["first_ts"], nf["last_ts"]
            if ft and lt:
                util, mem, power = [], [], []
                for (ts, u, m, p) in gpu_rows:
                    if ts and ft <= ts <= lt:
                        util.append(u); mem.append(m); power.append(p)
                nf["gpu_util_avg"]  = avg(util)
                nf["gpu_mem_avgMB"] = avg(mem)
                nf["gpu_power_avgW"]= avg(power)
            else:
                nf["gpu_util_avg"] = nf["gpu_mem_avgMB"] = nf["gpu_power_avgW"] = None

    # Write per-question CSVs
    nodes_csv = qdir / "features_nodes.csv"
    edges_csv = qdir / "features_edges.csv"

    node_cols = [
        "agent_idx","role","model_id","device","device_name","vram_gb",
        "msgs_sent","msgs_recv","chars_sent","chars_recv",
        "prompt_tokens_sum","completion_tokens_sum",
        "peers_out_distinct","peers_in_distinct",
        "first_ts","last_ts","window_sec",
        "gpu_util_avg","gpu_mem_avgMB","gpu_power_avgW"
    ]
    with nodes_csv.open("w", newline="", encoding="utf-8") as fh:
        wr = csv.writer(fh)
        wr.writerow(node_cols)
        for idx in sorted(node_feats):
            nf = node_feats[idx]
            ft = nf["first_ts"]; lt = nf["last_ts"]
            window_sec = (lt - ft).total_seconds() if (ft and lt) else None
            wr.writerow([
                nf["agent_idx"], nf["role"], nf["model_id"], nf["device"], nf["device_name"], nf["vram_gb"],
                nf["msgs_sent"], nf["msgs_recv"], nf["chars_sent"], nf["chars_recv"],
                nf["prompt_tokens_sum"], nf["completion_tokens_sum"],
                len(nf["peers_out"]), len(nf["peers_in"]),
                ft.isoformat() if ft else "", lt.isoformat() if lt else "", window_sec,
                nf.get("gpu_util_avg"), nf.get("gpu_mem_avgMB"), nf.get("gpu_power_avgW"),
            ])

    edge_cols = [
        "src_idx","src_role","dst_idx","dst_role",
        "count","chars_mean","prompt_tokens_sum","completion_tokens_sum",
        "first_ts","last_ts","span_sec"
    ]
    with edges_csv.open("w", newline="", encoding="utf-8") as fh:
        wr = csv.writer(fh)
        wr.writerow(edge_cols)
        for (si, di), ef in sorted(edge_feats.items()):
            ft, lt = ef["first_ts"], ef["last_ts"]
            mean_chars = (ef["chars_sum"]/ef["count"]) if ef["count"] else 0
            wr.writerow([
                si, idx2role.get(si, ""), di, idx2role.get(di, ""),
                ef["count"], round(mean_chars,2), ef["prompt_tokens_sum"], ef["completion_tokens_sum"],
                ft.isoformat() if ft else "", lt.isoformat() if lt else "",
                (lt - ft).total_seconds() if (ft and lt) else ""
            ])

    # Optional: write graph files
    if HAVE_NX:
        G = nx.DiGraph()
        # nodes
        for idx in node_feats:
            nf = node_feats[idx]
            G.add_node(idx, **{
                "role": nf["role"],
                "msgs": nf["msgs_sent"] + nf["msgs_recv"],
                "model": nf["model_id"] or "",
                "gpu_util_avg": nf.get("gpu_util_avg", None),
            })
        # edges
        for (si, di), ef in edge_feats.items():
            G.add_edge(si, di, weight=ef["count"])
        try:
            nx.write_gexf(G, str(qdir / "graph.gexf"))
        except Exception as e:
            print(f"[!] could not write GEXF: {e}")
        # PNG preview
        if HAVE_MPL:
            try:
                pos = nx.spring_layout(G, seed=42)
                sizes = [300 + 40*G.nodes[n]["msgs"] for n in G.nodes]
                widths = [1 + 0.6*G[u][v]["weight"] for u,v in G.edges]
                nx.draw_networkx_nodes(G, pos, node_size=sizes)
                nx.draw_networkx_edges(G, pos, width=widths, arrows=True, arrowsize=12)
                nx.draw_networkx_labels(G, pos, {n: f"{n}: {G.nodes[n]['role']}" for n in G.nodes}, font_size=8)
                plt.axis("off")
                plt.tight_layout()
                plt.savefig(qdir / "graph.png", dpi=160)
                plt.close()
            except Exception as e:
                print(f"[!] could not render PNG: {e}")

    print(f"[+] {qdir.name}: wrote {nodes_csv.name}, {edges_csv.name}, graph.gexf" + (", graph.png" if HAVE_NX and HAVE_MPL else ""))


def main():
    if len(sys.argv) != 2:
        print("Usage: python3 tools/extract_features_graph.py <RUN_DIR>")
        sys.exit(2)

    run_dir = Path(sys.argv[1]).resolve()
    if not run_dir.exists():
        print(f"[!] no such dir: {run_dir}")
        sys.exit(1)

    qdirs = find_q_dirs(run_dir)
    if not qdirs:
        print(f"[i] no q_* subfolders in {run_dir}; are you pointing at the run_benign_* folder?")
        sys.exit(0)

    for q in qdirs:
        build_features_for_q(run_dir, q)

if __name__ == "__main__":
    main()
