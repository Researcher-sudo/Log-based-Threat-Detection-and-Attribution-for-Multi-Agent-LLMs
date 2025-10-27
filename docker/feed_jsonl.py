# import os, sys, socket, json, time, argparse
# p=argparse.ArgumentParser()
# p.add_argument("--sock", required=True)
# p.add_argument("--jsonl", required=True)
# p.add_argument("--wait", type=float, default=120)
# p.add_argument("--delay", type=float, default=0.25)
# a=p.parse_args()

# deadline=time.time()+a.wait
# while time.time()<deadline:
#     if os.path.exists(a.sock): break
#     time.sleep(0.5)
# else:
#     print("[coordinator] socket not found:", a.sock, file=sys.stderr); sys.exit(1)

# print(f"[coordinator] feeding {a.jsonl} -> {a.sock}")
# with open(a.jsonl, "r", encoding="utf-8", errors="ignore") as f:
#     for ln in f:
#         ln=ln.strip()
#         if not ln: continue
#         try:
#             obj=json.loads(ln)
#             text=obj.get("text") or obj.get("prompt") or ln
#         except Exception:
#             text=ln
#         s=socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
#         try:
#             s.connect(a.sock); s.sendall(text.encode("utf-8")+b"\n")
#         finally:
#             s.close()
#         time.sleep(a.delay)
# print("[coordinator] done")

#!/usr/bin/env python3
import argparse, json, os, socket, time, sys
import os

def wait_for_socket(path: str, wait_secs: int) -> bool:
    for _ in range(wait_secs * 10):
        if os.path.exists(path):
            try:
                s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                s.settimeout(0.2)
                s.connect(path)
                s.close()
                return True
            except Exception:
                pass
        time.sleep(0.1)
    return os.path.exists(path)

def iter_jsonl(jsonl_path):
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            try:
                obj = json.loads(ln)
                text = obj.get("text") or obj.get("prompt") or ln
            except Exception:
                text = ln
            yield text

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sock", required=True, help="UNIX socket path, e.g. /runs/agents/recruiter.sock")
    ap.add_argument("--jsonl", required=True, help="JSONL file with prompts")
    ap.add_argument("--wait", type=int, default=60, help="seconds to wait for socket")
    # backward-compat alias; stores into the same 'wait'
    ap.add_argument("--wait-secs", type=int, dest="wait", help=argparse.SUPPRESS)
    args = ap.parse_args()

    if not wait_for_socket(args.sock, args.wait):
        print(f"[feed_jsonl] socket not found: {args.sock}", file=sys.stderr)
        sys.exit(1)

    total = 0
    for text in iter_jsonl(args.jsonl):
        total += 1
        try:
            s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            s.connect(args.sock)
            s.sendall((text + "\n").encode("utf-8"))
            s.close()
        except Exception as e:
            print(f"[feed_jsonl] send error on line {total}: {e}", file=sys.stderr)
            time.sleep(0.25)
            continue
        time.sleep(0.25)
    print(f"[feed_jsonl] sent {total} lines from {args.jsonl} -> {args.sock}")

if __name__ == "__main__":
    main()
