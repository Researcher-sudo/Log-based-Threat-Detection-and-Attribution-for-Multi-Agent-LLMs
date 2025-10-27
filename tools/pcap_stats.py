# tools/pcap_stats.py
from __future__ import annotations
import sys, subprocess, json, statistics as stats
from datetime import datetime

def tshark_lines(pcap: str, fields: list[str], display: str|None=None, extra: list[str]|None=None):
    cmd = ["tshark","-r",pcap,"-T","fields"]
    for f in fields: cmd += ["-e", f]
    if display: cmd += ["-Y", display]
    if extra: cmd += extra
    out = subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL)
    for line in out.splitlines():
        yield [col for col in line.split("\t")]

def parse_float(s, default=0.0):
    try: return float(s)
    except: return default

def main(pcap: str, out_tsv: str):
    # basic pkt lens + times
    rows = list(tshark_lines(pcap, ["frame.time_epoch","frame.len"]))
    if not rows:
        open(out_tsv,"w").write("")
        return
    times = [parse_float(t) for t,_ in rows]
    lens  = [parse_float(l) for _,l in rows]
    start, end = min(times), max(times)
    dur = max(1e-6, end - start)

    # TCP/UDP counts
    n_tcp = len(list(tshark_lines(pcap, ["frame.len"], display="tcp")))
    n_udp = len(list(tshark_lines(pcap, ["frame.len"], display="udp")))

    # distinct src/dst IPs
    n_src = len(set(x[0] for x in tshark_lines(pcap, ["ip.src"]) if x and x[0]))
    n_dst = len(set(x[0] for x in tshark_lines(pcap, ["ip.dst"]) if x and x[0]))

    # unique 5-tuple flows (src,dst,sport,dport,proto)
    flows = set()
    for s,d,sp,dp,pr in tshark_lines(pcap, ["ip.src","ip.dst","tcp.srcport","tcp.dstport","ip.proto"], display="ip"):
        flows.add((s,d,sp,dp,pr))
    for s,d,sp,dp,pr in tshark_lines(pcap, ["ip.src","ip.dst","udp.srcport","udp.dstport","ip.proto"], display="udp"):
        flows.add((s,d,sp,dp,pr))

    # percentiles
    p95 = stats.quantiles(lens, n=100)[94] if len(lens) >= 100 else sorted(lens)[int(0.95*len(lens))-1]

    out = {
        "pcap_pkts": len(lens),
        "pcap_bytes": int(sum(lens)),
        "pcap_pkts_per_sec": round(len(lens)/dur, 3),
        "pcap_bytes_per_sec": round(sum(lens)/dur, 1),
        "pcap_avg_len": round(sum(lens)/len(lens), 1),
        "pcap_p95_len": int(p95),
        "pcap_pkts_tcp": n_tcp,
        "pcap_pkts_udp": n_udp,
        "pcap_n_flows_5tuple": len(flows),
        "pcap_n_src_ips": n_src,
        "pcap_n_dst_ips": n_dst,
    }
    with open(out_tsv,"w") as fp:
        fp.write("feature\tvalue\n")
        for k,v in out.items():
            fp.write(f"{k}\t{v}\n")

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
