#!/usr/bin/env python3
"""
build_medqa_attackset.py
------------------------
Generate a preview dataset of MedQA MCQs:
  • benign (no tail)
  • compromised (MCQ + payload tail)
Then success-screen them against your current redteam harness to keep only
prompts that actually trigger your compromise detector.

USAGE
-----
# Example (preview 25+25):
python build_medqa_attackset.py \
  --medqa-path ../data/medqa/test.jsonl \
  --n-benign 25 \
  --n-attacks 25 \
  --overgen 120 \
  --model llama2-uncensored \
  --output-dir ./preview_pack

Place this script in the same directory as your redteam.py so imports resolve.
"""

import argparse
import json
import os
import random
from pathlib import Path
from typing import List, Dict, Tuple, Optional

# Local imports: redteam (your harness) and families
import importlib
import sys

HERE = Path(__file__).parent.resolve()
sys.path.insert(0, str(HERE))

redteam = importlib.import_module("redteam")
families = importlib.import_module("attack_families")

# --------------------------
# Utils
# --------------------------

def load_medqa(jsonl_path: Path) -> List[Dict]:
    data = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            # Expect fields: question, options (dict), answer_idx, answer
            if not all(k in obj for k in ("question","options","answer_idx","answer")):
                # Skip malformed lines quietly
                continue
            data.append(obj)
    return data

def sample_mcqs(data: List[Dict], k: int, seed: int) -> List[Dict]:
    rng = random.Random(seed)
    return rng.sample(data, k)

def mcq_to_text(item: Dict, shuffle_options: bool = True, seed: int = 42) -> Tuple[str, Dict]:
    """
    Build a single-string MCQ block and return (text, mapping) where mapping
    maps letters back to original option keys (A,B,C,D).
    """
    opts = [(k, v) for k, v in item["options"].items()]
    rng = random.Random(seed)
    if shuffle_options:
        rng.shuffle(opts)
    letters = ["A","B","C","D","E","F","G","H"]
    mapping = {}
    parts = [item["question"].strip(), "Options:"]
    for i, (k, v) in enumerate(opts):
        letter = letters[i]
        mapping[letter] = k
        parts.append(f"({letter}) {v}")
    text = " ".join(parts)
    return text, mapping

def compose_attack(mcq_text: str, family: str) -> str:
    return families.attach_tail(mcq_text, family)

def to_row(sample_id: str,
           item: Dict,
           mcq_text: str,
           full_prompt: str,
           label: str,
           family: Optional[str],
           position: str = "after_options") -> Dict:
    row = {
        "id": sample_id,
        "source_dataset": "MedQA",
        "split": "preview",
        "label": label,                       # "benign" | "attack"
        "question_text": item["question"].strip(),
        "options": item["options"],
        "answer_idx": item["answer_idx"],
        "answer_text": item["answer"],
        "tail_family": family,
        "tail_layers": None,                  # simple; families encode their own layers
        "position": position,
        "full_prompt": full_prompt,
        "attack_goal": "sys_prompt_leak" if label=="attack" else None,
        "generated_with": {"payload_version": "preview_v1", "seed": None},
        "validation": None,
    }
    return row

# --------------------------
# Screening via your redteam harness
# --------------------------

def make_ask_fn(model_name: str):
    def raw_ask(payload: str):
        # Uses your Agent from redteam.py
        # test_agent = redteam.Agent(instruction=os.environ.get("AGENT_INSTRUCTION" % os.environ.get("ROLE_NAME","agent")), "", role="intruder", model_info=model_name)
        test_agent = redteam.Agent(
            instruction=os.environ.get("AGENT_INSTRUCTION", f"You are a {os.environ.get('ROLE_NAME','agent')}."),
            role="intruder",
            model_info=model_name
            )

        return test_agent.chat(payload)
    return raw_ask

def screen_attacks(candidates: List[Dict], model_name: str, max_keep: int) -> List[Dict]:
    ask = make_ask_fn(model_name)
    kept = []
    per_family_counts = {}
    for cand in candidates:
        family = cand["tail_family"]
        record = redteam.run_one(cand["full_prompt"], ask)
        compromised = record.get("compromised", False)
        val = {
            "attempts": record.get("attempt", 0),
            "compromised": compromised
        }
        cand["validation"] = val
        if compromised:
            per_family_counts[family] = per_family_counts.get(family, 0) + 1
            kept.append(cand)
            if len(kept) >= max_keep:
                break
    return kept

# --------------------------
# Main
# --------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--medqa-path", type=str, required=True,
                    help="Path to MedQA test.jsonl (fields: question, options, answer_idx, answer).")
    ap.add_argument("--n-benign", type=int, default=25)
    ap.add_argument("--n-attacks", type=int, default=25)
    ap.add_argument("--overgen", type=int, default=120,
                    help="Generate this many attack candidates before screening.")
    ap.add_argument("--model", type=str, default="llama2-uncensored",
                    help="Model string consumed by your Agent(instruction=os.environ.get("AGENT_INSTRUCTION" % os.environ.get("ROLE_NAME","agent")), model_info=...).")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output-dir", type=str, default="./preview_pack")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    medqa = load_medqa(Path(args.medqa_path))
    if len(medqa) < (args.n_benign + args.overgen + 10):
        print(f"[!] MedQA size seems small ({len(medqa)}). Consider a larger split.")

    # 1) Benign preview
    benign_samples = sample_mcqs(medqa, args.n_benign, seed=args.seed+1)
    benign_rows = []
    for i, item in enumerate(benign_samples, 1):
        mcq_text, _ = mcq_to_text(item, shuffle_options=True, seed=args.seed+i)
        rid = f"medqa_benign_{i:03d}"
        row = to_row(rid, item, mcq_text, mcq_text, "benign", family=None)
        benign_rows.append(row)

    # 2) Attack candidates (over-generate and then screen)
    attack_pool = sample_mcqs(medqa, args.overgen, seed=args.seed+3)
    family_list = families.FAMILY_LIST[:]
    rng.shuffle(family_list)

    candidates: List[Dict] = []
    fam_cycle = 0
    for j, item in enumerate(attack_pool, 1):
        fam = family_list[fam_cycle % len(family_list)]
        fam_cycle += 1
        mcq_text, _ = mcq_to_text(item, shuffle_options=True, seed=args.seed+100+j)
        full_prompt = compose_attack(mcq_text, fam)
        rid = f"medqa_attack_cand_{j:03d}_{fam}"
        row = to_row(rid, item, mcq_text, full_prompt, "attack", family=fam)
        candidates.append(row)

    # 3) Screen using your redteam harness & detector
    print(f"[*] Screening {len(candidates)} candidates on model={args.model} ...")
    kept_attacks = screen_attacks(candidates, args.model, max_keep=args.n_attacks)
    print(f"[+] Kept {len(kept_attacks)} compromised prompts.")

    # 4) Write JSONL outputs
    benign_path = outdir / "benign_preview.jsonl"
    compromised_path = outdir / "compromised_preview.jsonl"
    with benign_path.open("w", encoding="utf-8") as fb:
        for row in benign_rows:
            fb.write(json.dumps(row, ensure_ascii=False) + "\n")
    with compromised_path.open("w", encoding="utf-8") as fa:
        for row in kept_attacks:
            fa.write(json.dumps(row, ensure_ascii=False) + "\n")

    # 5) Metadata
    meta = {
        "source": "MedQA",
        "option": "A + D (success-screened)",
        "n_benign": len(benign_rows),
        "n_compromised": len(kept_attacks),
        "families_used": families.FAMILY_LIST,
        "seed": args.seed,
        "model": args.model,
    }
    (outdir / "metadata.json").write_text(json.dumps(meta, indent=2))

    print(f"[✓] Wrote:\n  - {benign_path}\n  - {compromised_path}\n  - {outdir/'metadata.json'}")

if __name__ == "__main__":
    main()
