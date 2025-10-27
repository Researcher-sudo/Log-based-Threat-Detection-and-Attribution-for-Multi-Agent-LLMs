# log_writer.py
from __future__ import annotations
import json, os, time, pathlib
from typing import Any, Dict

class JSONLLogger:
    def __init__(self, run_dir: str):
        self.run_dir = pathlib.Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.merged = (self.run_dir / "agent.jsonl").open("a", buffering=1)

    def _agent_fp(self, agent_id: str):
        d = self.run_dir / "agents" / agent_id
        d.mkdir(parents=True, exist_ok=True)
        return (d / "agent.jsonl").open("a", buffering=1)

    def write(self, agent_id: str, rec: Dict[str, Any]):
        rec["ts"] = time.time()
        rec["agent_id"] = agent_id
        line = json.dumps(rec, ensure_ascii=False)
        # per-agent
        with self._agent_fp(str(agent_id)) as fp:
            fp.write(line + "\n")
        # merged
        self.merged.write(line + "\n")
