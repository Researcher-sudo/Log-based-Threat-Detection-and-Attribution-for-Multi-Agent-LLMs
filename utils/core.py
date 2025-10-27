import os
import json
import random
import re
from tqdm import tqdm
from prettytable import PrettyTable 
from termcolor import cprint
from pptree import Node, print_tree
import google.generativeai as genai
from openai import OpenAI
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from transformers import BitsAndBytesConfig
import torch, os
import re     
from pathlib import Path
from datetime import datetime
# _REPO = "georgesung/llama2_7b_chat_uncensored"
# _TOKENIZER = None   # shared singleton
# _MODEL     = None
# _PIPE_CACHE: dict[tuple[int,float], pipeline] = {}
import re
import tempfile
import traceback



def _trim_to_three_lines(txt: str) -> str:
    lines = [l.strip() for l in (txt or "").splitlines() if l.strip() and not l.startswith("[")]
    pick = lambda tag: next((l for l in lines if l.lower().startswith(tag)), "")
    out = [pick("hypothesis:"), pick("question:"), pick("next step:")]
    tags = ["Hypothesis:", "Question:", "Next step:"]
    out = [o if o else t + " (missing)" for o, t in zip(out, tags)]
    return "\n".join(out)

class RunLogger:
    """JSONL logger that records all edges and per-agent logs."""
    def __init__(self, run_dir: Path, question: str, roles: list[str]):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.all_path = self.run_dir / "interactions.jsonl"
        self.roles = roles
        # write a header
        self._append(self.all_path, {
            "type": "run_start",
            "ts": datetime.utcnow().isoformat(),
            "question": question,
            "roles": [{"idx": i+1, "role": r} for i, r in enumerate(roles)]
        })

    def _agent_path(self, idx: int) -> Path:
        safe_role = re.sub(r'[^a-zA-Z0-9_-]+', '_', self.roles[idx-1])[:50]
        return self.run_dir / f"agent_{idx:02d}_{safe_role}.jsonl"

    def _append(self, path: Path, obj: dict):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")


    def log_edge(self, round_no:int, turn_no:int, src:int, dst:int, message:str, meta:dict|None=None):
        rec = {
            "type": "edge_message",
            "ts": datetime.utcnow().isoformat(),
            "round": round_no,
            "turn": turn_no,
            "src": {"idx": src, "role": self.roles[src-1]},
            "dst": {"idx": dst, "role": self.roles[dst-1]},
            "message": message
        }
        if meta: rec["meta"] = meta
        self._append(self.all_path, rec)
        self._append(self._agent_path(src), {**rec, "direction": "sent"})
        self._append(self._agent_path(dst), {**rec, "direction": "received"})

    @staticmethod
    def build_edge_meta(pipe, prompt: str, reply: str, decode_args: dict) -> dict:
        try:
            import torch
            cuda = torch.cuda.is_available()
            device = "cuda:0" if cuda else "cpu"
            name = torch.cuda.get_device_name(0) if cuda else "cpu"
            vram = int(torch.cuda.get_device_properties(0).total_memory/1024**3) if cuda else 0
        except Exception:
            device, name, vram = "cpu", "cpu", 0

        model_id = None
        if pipe is not None:
            model = getattr(pipe, "model", None)
            model_id = getattr(getattr(model, "config", None), "_name_or_path", type(model).__name__)

        tok = getattr(pipe, "tokenizer", None) if pipe is not None else None
        p_ids = tok.encode(prompt) if tok else []
        c_ids = tok.encode(reply)  if tok else []

        headers = {
            "X-Alarm-Trace": os.environ.get("X_ALARM_TRACE",""),
            "traceparent":   os.environ.get("TRACEPARENT",""),
            "tracestate":    os.environ.get("TRACESTATE",""),
        }
        trace = {
            "trace_id":    headers["X-Alarm-Trace"],
            "traceparent": headers["traceparent"],
            "tracestate":  headers["tracestate"],
            "cfp":         os.environ.get("CFP",""),
        }
        return {
            "model_id": model_id or "unknown",
            "device": device, "device_name": name, "vram_gb": vram,
            "decode": decode_args,
            "timing_ms": float(decode_args.get("timing_ms", 0.0)),
            "tokens": {"prompt": len(p_ids), "completion": len(c_ids)},
            "pid": os.getpid(),
            "headers": headers, "trace": trace,
        }

    def log_status(self, round_no: int, turn_no: int, agent_idx: int, status: str, reason: str = ""):
        rec = {
            "type": "agent_status",
            "ts": datetime.utcnow().isoformat(),
            "round": round_no,
            "turn": turn_no,
            "agent": {"idx": agent_idx, "role": self.roles[agent_idx-1]},
            "status": status,  # e.g., "pass", "no_targets", "parse_failed"
            "reason": reason
        }
        self._append(self.all_path, rec)
        self._append(self._agent_path(agent_idx), rec)

def wants_to_talk(text: str) -> bool:
    first = (text or "").splitlines()[0].strip().lower()
    # accept "yes", "yes,", "yes." "y", "yeah" etc. (first token starts with y/yes)
    return bool(re.match(r'^(y|yes)\b', first))



def parse_targets(s: str, n_agents: int) -> list[int]:
    # Extract all integers in range [1..n_agents]
    nums = [int(m.group()) for m in re.finditer(r"\d+", s or "")]
    return [x for x in nums if 1 <= x <= n_agents]


def _is_writable_dir(p: Path) -> bool:
    try:
        p.mkdir(parents=True, exist_ok=True)
        test = p / ".write_test"
        with test.open("w") as f:
            f.write("ok")
        test.unlink(missing_ok=True)
        return True
    except Exception:
        return False

def choose_run_dir() -> Path:
    """
    Single source of truth for where RunLogger writes.
    If RUN_LOG_DIR is set (the launcher injects it), write *there*.
    Otherwise, fall back to ./runs/<ts> under the current working dir.
    """
    env_path = os.environ.get("RUN_LOG_DIR")
    if env_path:
        p = Path(env_path).expanduser().resolve()
        p.mkdir(parents=True, exist_ok=True)
        return p  # ← IMPORTANT: no extra timestamp folder

    ts = datetime.now().strftime('%Y%m%d-%H%M%S')
    p = Path.cwd() / "runs" / ts
    p.mkdir(parents=True, exist_ok=True)
    return p

def choose_run_dir_for_question(question: str) -> Path:
    base = choose_run_dir()
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "_", (question or "").strip().lower())[:60] or "q"
    from hashlib import sha1
    qhash = sha1((question or "").encode("utf-8")).hexdigest()[:8]
    ts = datetime.now().strftime('%H%M%S')
    p = base / f"q_{qhash}_{slug}_{ts}"
    p.mkdir(parents=True, exist_ok=True)
    return p


_REPO = "georgesung/llama2_7b_chat_uncensored"
_TOKENIZER = None
_MODEL = None
_PIPE_CACHE: dict[tuple[int,float], pipeline] = {}

def _tail(tokens: list[int], max_len: int = 2048) -> list[int]:
    return tokens[-max_len:]


# def get_llama_pipe(*, max_new_tokens: int = 128, temperature: float = 0.7):
from transformers import BitsAndBytesConfig  # make sure this import exists

def get_llama_pipe(*, max_new_tokens: int = int(os.getenv("MAX_NEW_TOKENS", "128")),
                   temperature: float = float(os.getenv("TEMP", "0.7"))):
    global _TOKENIZER, _MODEL, _PIPE_CACHE

    key = (max_new_tokens, round(temperature, 4))
    if key in _PIPE_CACHE:
        return _PIPE_CACHE[key]

    if _TOKENIZER is None:
        _TOKENIZER = AutoTokenizer.from_pretrained(_REPO, use_fast=True, legacy=False)
    if _MODEL is None:
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16
        )
        _MODEL = AutoModelForCausalLM.from_pretrained(
            _REPO,
            quantization_config=bnb,
            device_map="auto",
            low_cpu_mem_usage=True,
        )

    _PIPE_CACHE[key] = pipeline(
        "text-generation",
        model=_MODEL,
        tokenizer=_TOKENIZER,
        max_new_tokens=max_new_tokens,
        repetition_penalty=1.1,
        return_full_text=True,
    )
    return _PIPE_CACHE[key]   # ← this was missing



# utils/core.py

def _smart_head_tail_trim(pipe, prompt: str, max_new_tokens: int) -> str:
    tok = pipe.tokenizer
    ids = tok(prompt, return_tensors="pt").input_ids[0]

    cfg = getattr(getattr(pipe, "model", None), "config", None)
    ctx_limit = int(getattr(cfg, "max_position_embeddings", 2048))
    budget = max(64, ctx_limit - int(max_new_tokens) - 32)

    if ids.size(0) <= budget:
        return prompt

    head_keep = min(max(256, ctx_limit // 5), budget // 2)
    tail_keep = budget - head_keep
    kept = torch.cat([ids[:head_keep], ids[-tail_keep:]], dim=0)
    return tok.decode(kept, skip_special_tokens=True)


# utils/core.py
def llama_generate(pipe, prompt, *, max_new_tokens=128, do_sample=False, temperature=0.7, top_p=0.95, **kw):
    prompt = _smart_head_tail_trim(pipe, prompt, max_new_tokens=max_new_tokens)
    gen = dict(max_new_tokens=int(max_new_tokens), do_sample=bool(do_sample))
    if do_sample:
        gen.update(temperature=float(temperature), top_p=float(top_p))
    eos_id = getattr(pipe.tokenizer, "eos_token_id", None)
    if eos_id is not None:
        gen["eos_token_id"] = eos_id
    out = pipe(prompt, **gen, **kw)
    return (out[0]["generated_text"] if isinstance(out, list) else out["generated_text"])

# def get_llama_pipe(max_new_tokens: int = 512, *, temperature: float = 0.7):
#     """Return a *shared* HF pipeline keyed by (max_new_tokens, temperature).

#     • Keeps GPU memory down – the model + tokenizer load only once.
#     • `temperature` is part of the cache‑key so you can safely request both
#       greedy (≈0) and sampling (0.7, 1.0 …) pipelines in the same process.
#     • Sets `return_full_text=True` so we can always split the assistant slice
#       out of the prompt without truncation issues.
#     """
#     global _TOKENIZER, _MODEL, _PIPE_CACHE

#     key = (max_new_tokens, round(temperature, 4))
#     if key in _PIPE_CACHE:
#         return _PIPE_CACHE[key]

#     if _TOKENIZER is None:
#         _TOKENIZER = AutoTokenizer.from_pretrained(_REPO, use_fast=True)
#     if _MODEL is None:
#         _MODEL = AutoModelForCausalLM.from_pretrained(
#             _REPO,
#             torch_dtype=torch.float16,
#             device_map="auto",
#             low_cpu_mem_usage=True,
#         )

#     _PIPE_CACHE[key] = pipeline(
#         "text-generation",
#         model=_MODEL,
#         tokenizer=_TOKENIZER,
#         max_new_tokens=max_new_tokens,
#         temperature=max(1e-5, temperature),  # HF requires >0 if sampling
#         repetition_penalty=1.1,
#         return_full_text=True,
#     )
#     return _PIPE_CACHE[key]


class Agent:
    def __init__(self, instruction, role, examplers=None, model_info='gpt-4o-mini', img_path=None):
        self.instruction = instruction
        self.role = role
        self.model_info = model_info
        self.img_path = img_path

        if self.model_info == 'gemini-pro':
            self.model = genai.GenerativeModel('gemini-pro')
            self._chat = self.model.start_chat(history=[])
        elif self.model_info in ['gpt-3.5', 'gpt-4', 'gpt-4o', 'gpt-4o-mini']:
            self.client = OpenAI(api_key=os.environ['openai_api_key'])
            self.messages = [
                {"role": "system", "content": instruction},
            ]
            if examplers is not None:
                for exampler in examplers:
                    self.messages.append({"role": "user", "content": exampler['question']})
                    self.messages.append({"role": "assistant", "content": exampler['answer'] + "\n\n" + exampler['reason']})
        elif self.model_info.startswith("llama2-uncensored"):
            # use the single shared llama-2 pipeline instead of reloading
            self._pipe     = get_llama_pipe()
            self.tokenizer = self._pipe.tokenizer
            self.history   = [f"[SYSTEM] {instruction.strip()}"]

    def chat(self, message, img_path=None, chat_mode=True):
        if self.model_info == 'gemini-pro':
            for _ in range(10):
                try:
                    response = self._chat.send_message(message, stream=True)
                    responses = ""
                    for chunk in response:
                        responses += chunk.text + "\n"
                    return responses
                except:
                    continue
            return "Error: Failed to get response from Gemini."

        elif self.model_info.startswith("llama2-uncensored"):
            # # build chat-style prompt
            # prompt = "\n".join(
            #     self.history + [f"[USER] {message.strip()}", "[ASSISTANT]"]
            # )
             # ephemeral mode: ignore history, be deterministic & short
            if chat_mode is False:
                base = f"[SYSTEM] {self.instruction.strip()}\n[USER] {message.strip()}\n[ASSISTANT]"
                # generated = llama_generate(
                #     self._pipe, base,
                #     max_new_tokens=16,   # tiny; we only need "yes"/"no" or a few digits
                #     temperature=0.0,     # greedy for compliance
                #     do_sample=False
                # )
                generated = llama_generate(self._pipe, base, max_new_tokens=16, do_sample=False)
                reply = generated.rsplit("[ASSISTANT]", 1)[-1].strip()
                del generated
                torch.cuda.empty_cache()
                return reply

            # normal chat: use running history
            prompt = "\n".join(self.history + [f"[USER] {message.strip()}", "[ASSISTANT]"])

            # ---------- optional debug aid ----------
            #   export DEBUG_PROMPT=1 before running to see
            #   the *exact* context fed into the model.
            if os.getenv("DEBUG_PROMPT"):
                print("\n══ PROMPT (tail) ═════════════════════════════════════")
                print(prompt[-2000:])      # last 2 kB usually enough
                print("══ END PROMPT ═══════════════════════════════════════\n")
            # ----------------------------------------

            # grab a *sampling* pipeline (≈temperature 0.7 is fine here)
            # pipe = get_llama_pipe(max_new_tokens=128, temperature=0.7)

            # # generate
            # generated = pipe(
            #     prompt,
            #     do_sample=True,
            #     eos_token_id=self.tokenizer.eos_token_id,
            # )[0]["generated_text"]

            pipe = self._pipe            # reuse the one built in __init__
            _MAX_NEW = int(os.getenv("MAX_NEW_TOKENS", "64"))
            _TEMP    = float(os.getenv("TEMP", "0.7"))
            generated = llama_generate(self._pipe, prompt,
                                    max_new_tokens=_MAX_NEW, temperature=_TEMP, do_sample=True)

        #     generated = llama_generate(
        #     pipe,
        #     prompt,
        #     max_new_tokens=128,
        #     temperature=0.7,
        # )
            # generated = pipe(
            #     prompt,
            #     do_sample=True,
            #     eos_token_id=pipe.tokenizer.eos_token_id,
            # )[0]["generated_text"]

            # assistant_reply = generated.split("[ASSISTANT]", 1)[-1].strip()
            assistant_reply = generated.rsplit("[ASSISTANT]", 1)[-1].strip()

            # ─── free the bulky string & caches ──────────────────────
            del generated
            torch.cuda.empty_cache()
            # --------------------------------------------------------
            self.history.extend(
                [f"[USER] {message.strip()}", f"[ASSISTANT] {assistant_reply}"]
            )
            return assistant_reply    
        
        # elif self.model_info.startswith("llama2-uncensored"):
        #     prompt = "\n".join(self.history + [f"[USER] {message.strip()}", "[ASSISTANT]"])
        #     raw = self._pipe(prompt, do_sample=True, eos_token_id=self.tokenizer.eos_token_id)[0]["generated_text"]
        #     assistant_reply = raw.split("[ASSISTANT]")[-1].strip()
        #     self.history.extend([f"[USER] {message.strip()}", f"[ASSISTANT] {assistant_reply}"])
        #     return assistant_reply
        
        elif ("/" in self.model_info) or any(x in self.model_info.lower() for x in
                                     ("llama","mistral","mixtral","qwen","phi","vicuna","falcon","zephyr")):
            # Hugging Face model id → monkeypatch handles chat(); keep a history buffer.
            self.history = [f"[SYSTEM] {self.instruction.strip()}"]


        elif self.model_info in ['gpt-3.5', 'gpt-4', 'gpt-4o', 'gpt-4o-mini']:
            self.messages.append({"role": "user", "content": message})
            
            if self.model_info == 'gpt-3.5':
                model_name = "gpt-3.5-turbo"
            else:
                model_name = "gpt-4o-mini"

            response = self.client.chat.completions.create(
                model=model_name,
                messages=self.messages
            )

            self.messages.append({"role": "assistant", "content": response.choices[0].message.content})
            return response.choices[0].message.content

    def temp_responses(self, message, img_path=None):
        """
        Return {0.0: greedy_text, 0.7: sampled_text} without touching chat history.
        Keeps the prompt short (system + one user turn) to avoid context bleed.
        """
        if self.model_info in ['gpt-3.5', 'gpt-4', 'gpt-4o', 'gpt-4o-mini']:
            self.messages.append({"role": "user", "content": message})
            responses = {}
            for temperature in [0.0, 0.7]:
                model_info = 'gpt-3.5-turbo' if self.model_info == 'gpt-3.5' else 'gpt-4o-mini'
                resp = self.client.chat.completions.create(
                    model=model_info,
                    messages=self.messages,
                    temperature=temperature,
                )
                responses[temperature] = resp.choices[0].message.content
            return responses

        elif self.model_info.startswith("llama2-uncensored"):
            # Build a short, ephemeral prompt (don’t use self.history)
            base = f"[SYSTEM] {self.instruction.strip()}\n[USER] {message.strip()}\n[ASSISTANT]"
            out = {}

            _MAX_NEW = int(os.getenv("MAX_NEW_TOKENS", "128"))
            # Greedy (temperature ignored when do_sample=False)
            greedy = llama_generate(self._pipe, base, max_new_tokens=_MAX_NEW,
                                    temperature=0.0, do_sample=False)
            out[0.0] = greedy.rsplit("[ASSISTANT]", 1)[-1].strip()

            # Sampled
            samp_T = 0.7
            sample = llama_generate(self._pipe, base, max_new_tokens=_MAX_NEW,
                                    temperature=samp_T, do_sample=True)
            out[samp_T] = sample.rsplit("[ASSISTANT]", 1)[-1].strip()

            # free memory
            del greedy, sample
            torch.cuda.empty_cache()
            return out

        # HF monkeypatched backend (other model IDs)
        return {0.0: self.chat(message, chat_mode=False)}


        # elif self.model_info.startswith("llama2-uncensored"):
        #     out = {}
        #     base_prompt = "\n".join(self.history + [f"[USER] {message}", "[ASSISTANT]"])

        #     #   a) greedy pass (do_sample=False) --------------------------------
        #     pipe = get_llama_pipe(max_new_tokens=128, temperature=0.7)  # temp ignored when do_sample=False
        #     _MAX_NEW = int(os.getenv("MAX_NEW_TOKENS", "128"))
        #     greedy = llama_generate(self._pipe, base_prompt,
        #                 max_new_tokens=_MAX_NEW, do_sample=False, temperature=0.0)

        #     # greedy = llama_generate(
        #     #     pipe, base_prompt,
        #     #     max_new_tokens=128, temperature=0.0,  # do_sample ignored
        #     # )
        #     out[0.0] = greedy.split("[ASSISTANT]")[-1].strip()
        #     #new
        #     del greedy
        #     torch.cuda.empty_cache()

        #     #   b) sampling pass (e.g. T=0.7) -----------------------------------
        #     samp_T = 0.7
        #     # pipe_samp = get_llama_pipe(max_new_tokens=128, temperature=samp_T)
        #     pipe_samp = self._pipe
        #     # sample = pipe_samp(base_prompt, do_sample=True, eos_token_id=self.tokenizer.eos_token_id)[0]["generated_text"]
        #     sample = llama_generate(pipe_samp, base_prompt,
        #                 max_new_tokens=_MAX_NEW, temperature=samp_T)

        # #     sample = llama_generate(
        # #     pipe_samp, base_prompt,
        # #     max_new_tokens=128, temperature=samp_T,
        # # )
        #     out[samp_T] = sample.split("[ASSISTANT]")[-1].strip()
        #     #new
        #     del sample
        #     torch.cuda.empty_cache()

        #     return out


model_id = "georgesung/llama2_7b_chat_uncensored"

class Group:
    def __init__(self, goal, members, question, examplers=None):
        self.goal = goal
        self.members = []
        for member_info in members:
            _agent = Agent('You are a {} who {}.'.format(member_info['role'], member_info['expertise_description'].lower()), role=member_info['role'], model_info=model_id)
            _agent.chat('You are a {} who {}.'.format(member_info['role'], member_info['expertise_description'].lower()))
            self.members.append(_agent)
        self.question = question
        self.examplers = examplers

    def interact(self, comm_type, message=None, img_path=None):
        if comm_type == 'internal':
            lead_member = None
            assist_members = []
            for member in self.members:
                member_role = member.role

                if 'lead' in member_role.lower():
                    lead_member = member
                else:
                    assist_members.append(member)

            if lead_member is None:
                lead_member = assist_members[0]
            
            delivery_prompt = f'''You are the lead of the medical group which aims to {self.goal}. You have the following assistant clinicians who work for you:'''
            for a_mem in assist_members:
                delivery_prompt += "\n{}".format(a_mem.role)
            
            delivery_prompt += "\n\nNow, given the medical query, provide a short answer to what kind investigations are needed from each assistant clinicians.\nQuestion: {}".format(self.question)
            try:
                delivery = lead_member.chat(delivery_prompt)
            except:
                delivery = assist_members[0].chat(delivery_prompt)

            investigations = []
            for a_mem in assist_members:
                investigation = a_mem.chat("You are in a medical group where the goal is to {}. Your group lead is asking for the following investigations:\n{}\n\nPlease remind your expertise and return your investigation summary that contains the core information.".format(self.goal, delivery))
                investigations.append([a_mem.role, investigation])
            
            gathered_investigation = ""
            for investigation in investigations:
                gathered_investigation += "[{}]\n{}\n".format(investigation[0], investigation[1])

            if self.examplers is not None:
                investigation_prompt = f"""The gathered investigation from your asssitant clinicians is as follows:\n{gathered_investigation}.\n\nNow, after reviewing the following example cases, return your answer to the medical query among the option provided:\n\n{self.examplers}\nQuestion: {self.question}"""
            else:
                investigation_prompt = f"""The gathered investigation from your asssitant clinicians is as follows:\n{gathered_investigation}.\n\nNow, return your answer to the medical query among the option provided.\n\nQuestion: {self.question}"""

            response = lead_member.chat(investigation_prompt)

            return response

        elif comm_type == 'external':
            return

# def parse_hierarchy(info, emojis):
#     moderator = Node('moderator (\U0001F468\u200D\u2696\uFE0F)')
#     agents = [moderator]

#     count = 0
#     for expert, hierarchy in info:
#         try:
#             expert = expert.split('-')[0].split('.')[1].strip()
#         except:
#             expert = expert.split('-')[0].strip()
        
#         if hierarchy is None:
#             hierarchy = 'Independent'
        
#         if 'independent' not in hierarchy.lower():
#             # parent = hierarchy.split(">")[0].strip()
#             # child = hierarchy.split(">")[1].strip()
#             # changes..............................
#             parent, sep, rest = hierarchy.partition('>')
#             if not sep:
#                 # no “>” found—either skip or treat as all-independent
#                 print(f"⚠️  malformed hierarchy, no ‘>’: {hierarchy!r}")
#                 continue

#             parent = parent.strip()
#             child  = rest.strip()
#             #........................................
#             for agent in agents:
#                 if agent.name.split("(")[0].strip().lower() == parent.strip().lower():
#                     child_agent = Node("{} ({})".format(child, emojis[count]), agent)
#                     agents.append(child_agent)

#         else:
#             agent = Node("{} ({})".format(expert, emojis[count]), moderator)
#             agents.append(agent)

#         count += 1

#     return agents

def parse_hierarchy(info, emojis):
    """
    Build pptree Node hierarchy from [(expert_line, hierarchy), …].

    • Works even if there are more experts than emojis.
    • Treats malformed “A > B > C” defensively.
    """
    moderator = Node('moderator (\U0001F468\u200D\u2696\uFE0F)')
    agents    = [moderator]

    for idx, (expert_line, hierarchy) in enumerate(info):
        emoji = emojis[idx % len(emojis)]              # <-- never out of range

        # plain title before first “-”
        expert_title = expert_line.split('-', 1)[0].split(':', 1)[-1].strip()

        # independent if no hierarchy or keyword present
        if not hierarchy or 'independent' in hierarchy.lower():
            agents.append(Node(f"{expert_title} ({emoji})", moderator))
            continue

        # hierarchy like “Parent > Child”
        parent, _, child = (s.strip() for s in hierarchy.partition('>'))
        if not child:                                   # malformed ⇒ treat as independent
            agents.append(Node(f"{expert_title} ({emoji})", moderator))
            continue

        # find parent we have already created (fallback: moderator)
        parent_node = next((n for n in agents
                            if n.name.split('(')[0].strip().lower() == parent.lower()),
                           moderator)
        agents.append(Node(f"{child} ({emoji})", parent_node))

    return agents


def parse_group_info(group_info):
    lines = group_info.split('\n')
    
    parsed_info = {
        'group_goal': '',
        'members': []
    }

    parsed_info['group_goal'] = "".join(lines[0].split('-')[1:])
    
    for line in lines[1:]:
        if line.startswith('Member'):
            member_info = line.split(':')
            member_role_description = member_info[1].split('-')
            
            member_role = member_role_description[0].strip()
            member_expertise = member_role_description[1].strip() if len(member_role_description) > 1 else ''
            
            parsed_info['members'].append({
                'role': member_role,
                'expertise_description': member_expertise
            })
    
    return parsed_info

def setup_model(model_name):
    if 'gemini' in model_name:
        genai.configure(api_key=os.environ['genai_api_key'])
        return genai, None
    elif 'gpt' in model_name:
        client = OpenAI(api_key=os.environ['openai_api_key'])
        return None, client
    else:
        raise ValueError(f"Unsupported model: {model_name}")

def load_data(dataset):
    test_qa = []
    examplers = []

    test_path = f'../data/{dataset}/test.jsonl'
    with open(test_path, 'r') as file:
        for line in file:
            test_qa.append(json.loads(line))

    train_path = f'../data/{dataset}/train.jsonl'
    with open(train_path, 'r') as file:
        for line in file:
            examplers.append(json.loads(line))

    return test_qa, examplers

def create_question(sample, dataset):
    if dataset == 'medqa':
        question = sample['question'] + " Options: "
        options = []
        for k, v in sample['options'].items():
            options.append("({}) {}".format(k, v))
        random.shuffle(options)
        question += " ".join(options)
        return question, None
    return sample['question'], None

def determine_difficulty(question, difficulty):
    if difficulty != 'adaptive':
        return difficulty
    
    difficulty_prompt = f"""Now, given the medical query as below, you need to decide the difficulty/complexity of it:\n{question}.\n\nPlease indicate the difficulty/complexity of the medical query among below options:\n1) basic: a single medical agent can output an answer.\n2) intermediate: number of medical experts with different expertise should dicuss and make final decision.\n3) advanced: multiple teams of clinicians from different departments need to collaborate with each other to make final decision."""
    
    medical_agent = Agent(instruction='You are a medical expert who conducts initial assessment and your job is to decide the difficulty/complexity of the medical query.', role='medical expert', model_info=model_id)
    medical_agent.chat('You are a medical expert who conducts initial assessment and your job is to decide the difficulty/complexity of the medical query.')
    response = medical_agent.chat(difficulty_prompt)

    if 'basic' in response.lower() or '1)' in response.lower():
        return 'basic'
    elif 'intermediate' in response.lower() or '2)' in response.lower():
        return 'intermediate'
    elif 'advanced' in response.lower() or '3)' in response.lower():
        return 'advanced'

def process_basic_query(question, examplers, model, args):
    medical_agent = Agent(instruction='You are a helpful medical agent.', role='medical expert', model_info=model_id)
    new_examplers = []
    if args.dataset == 'medqa':
        random.shuffle(examplers)
        for ie, exampler in enumerate(examplers[:5]):
            tmp_exampler = {}
            exampler_question = exampler['question']
            choices = [f"({k}) {v}" for k, v in exampler['options'].items()]
            random.shuffle(choices)
            exampler_question += " " + ' '.join(choices)
            exampler_answer = f"Answer: ({exampler['answer_idx']}) {exampler['answer']}\n\n"
            exampler_reason = medical_agent.chat(f"You are a helpful medical agent. Below is an example of medical knowledge question and answer. After reviewing the below medical question and answering, can you provide 1-2 sentences of reason that support the answer as you didn't know the answer ahead?\n\nQuestion: {exampler_question}\n\nAnswer: {exampler_answer}")

            tmp_exampler['question'] = exampler_question
            tmp_exampler['reason'] = exampler_reason
            tmp_exampler['answer'] = exampler_answer
            new_examplers.append(tmp_exampler)
    
    single_agent = Agent(instruction='You are a helpful assistant that answers multiple choice questions about medical knowledge.', role='medical expert', examplers=new_examplers, model_info=model_id)
    single_agent.chat('You are a helpful assistant that answers multiple choice questions about medical knowledge.')
    final_decision = single_agent.temp_responses(f'''The following are multiple choice questions (with answers) about medical knowledge. Let's think step by step.\n\n**Question:** {question}\nAnswer: ''', img_path=None)
    
    return final_decision

# from . import log  # add this near top of core.py (with your other imports)
try:
    from . import log  # if part of a package
except Exception:
    log = None  # running as a script; logging module unavailable is fine
import time, psutil, pathlib

_RX = pathlib.Path("/sys/class/net/eth0/statistics/rx_bytes")
_TX = pathlib.Path("/sys/class/net/eth0/statistics/tx_bytes")
def _read_bytes(p: pathlib.Path) -> int:
    try: return int(p.read_text().strip())
    except Exception: return -1

def process_intermediate_query(question, examplers, model_id, args):
    cprint("[INFO] Step 1. Expert Recruitment", 'yellow', attrs=['blink'])
    recruit_prompt = (
        "You are an experienced medical expert who recruits a group of experts "
        "with diverse identity and ask them to discuss and solve the given medical query."
    )

    tmp_agent = Agent(instruction=recruit_prompt, role='recruiter', model_info=model_id)
    tmp_agent.chat(recruit_prompt)

    num_agents = 5
    recruited = tmp_agent.chat(
        f"Question: {question}\n"
        f"You can recruit {num_agents} experts in different medical expertise. Considering the medical question and the options for the answer, "
        f"what kind of experts will you recruit to better make an accurate answer?\n"
        f"Also, you need to specify the communication structure between experts "
        f"(e.g., Pulmonologist == Neonatologist == Medical Geneticist == Pediatrician > Cardiologist), "
        f"or indicate if they are independent.\n\n"
        f"For example, if you want to recruit five experts, your answer can be like:\n"
        f"1. Pediatrician - Specializes in ... - Hierarchy: Independent\n"
        f"2. Cardiologist - Focuses on ...   - Hierarchy: Pediatrician > Cardiologist\n"
        f"3. Pulmonologist - ...             - Hierarchy: Independent\n"
        f"4. Neonatologist - ...             - Hierarchy: Independent\n"
        f"5. Medical Geneticist - ...        - Hierarchy: Independent\n\n"
        f"Please answer in the above format, and do not include your reason."
    )

    # --- robust recruiter-line parser
    def _parse_agent_lines(raw: str):
        out = []
        for line in (raw or "").splitlines():
            line = line.strip()
            if not line: 
                continue
            head, *tail = line.split(" - Hierarchy:", 1)
            hier = tail[0].strip() if tail else None
            out.append((head, hier))
        return out

    # keep only “N. … - Hierarchy: …” lines so we don’t accidentally parse junk
    clean_lines = []
    for ln in (recruited or "").splitlines():
        ln = ln.strip()
        if re.match(r"^\d+\.\s+[^-]+ - .*Hierarchy:", ln, re.I):
            clean_lines.append(ln)

    agents_data = _parse_agent_lines("\n".join(clean_lines))

    # --- fallback if parsing failed
    # core.py  (inside process_intermediate_query, before using agents_data)
    def _fallback_specialists(question: str):
        q = (question or "").lower()
        if any(k in q for k in [
            "newborn","neonate","cyanot","egg-shaped","transposition","septostomy","ductus"
        ]):
            return [
                ("1. Neonatologist - Ventilation & PGE1 in duct-dependent lesions.", "Independent"),
                ("2. Pediatric Cardiologist - CHD dx; TGA vs others; echo planning.", "Independent"),
                ("3. Cardiothoracic Surgeon - Arterial switch; shunts; timing.", "Independent"),
                ("4. Pediatric Cardiac Anesthesiologist - Peri-op stabilization.", "Independent"),
                ("5. Pediatric Intensivist - ICU support; gases; lines; inotropes.", "Independent"),
            ]
        # generic fallback
        return [
            ("1. Internist - General differential & stabilization.", "Independent"),
            ("2. Cardiologist - Cardiac causes & tests.", "Independent"),
            ("3. Radiologist - Imaging strategy.", "Independent"),
            ("4. Pharmacologist - Drug/infusion plan.", "Independent"),
            ("5. Surgeon - Operative considerations.", "Independent"),
        ]

    # use it here:
    if not agents_data:
        agents_data = _fallback_specialists(question)

    # if not agents_data:
    #     agents_data = [
    #         ("1. Medical Ethicist - Disclosure, duty to inform, autonomy.", "Independent"),
    #         ("2. Orthopedic Surgeon - Operative standards & documentation.", "Independent"),
    #         ("3. Patient Safety Officer - Disclosure & RCA process.", "Independent"),
    #         ("4. Risk Manager - Incident reporting & policy.", "Independent"),
    #         ("5. Hospital Legal Counsel - Regulatory/mandated disclosure.", "Independent"),
    #     ]

    agent_emoji = ['\U0001F468\u200D\u2695\uFE0F', '\U0001F468\U0001F3FB\u200D\u2695\uFE0F', '\U0001F469\U0001F3FC\u200D\u2695\uFE0F', '\U0001F469\U0001F3FB\u200D\u2695\uFE0F', '\U0001f9d1\u200D\u2695\uFE0F', '\U0001f9d1\U0001f3ff\u200D\u2695\uFE0F', '\U0001f468\U0001f3ff\u200D\u2695\uFE0F', '\U0001f468\U0001f3fd\u200D\u2695\uFE0F', '\U0001f9d1\U0001f3fd\u200D\u2695\uFE0F', '\U0001F468\U0001F3FD\u200D\u2695\uFE0F']
    random.shuffle(agent_emoji)
    def emo(i: int) -> str:
        return agent_emoji[i % len(agent_emoji)]

    hierarchy_agents = parse_hierarchy(agents_data, agent_emoji)

    agent_list = ""
    for i, agent in enumerate(agents_data, 1):
        agent_role = agent[0].split('-', 1)[0].split('.', 1)[-1].strip().lower()
        parts = agent[0].split('-', 1)
        description = parts[1].strip().lower() if len(parts) > 1 else agent_role
        agent_list += f"Agent {i}: {agent_role} - {description}\n"

    agent_dict = {}
    medical_agents = []
    for agent in agents_data:
        text = agent[0]
        agent_role = text.split('-', 1)[0].split('.', 1)[-1].strip().lower()
        parts = text.split('-', 1)
        description = parts[1].strip().lower() if len(parts) > 1 else agent_role

        inst_prompt = f"You are a {agent_role} who {description}. Your job is to collaborate with other medical experts in a team."
        _agent = Agent(instruction=inst_prompt, role=agent_role, model_info=model_id)
        _agent.chat(inst_prompt)
        agent_dict[agent_role] = _agent
        medical_agents.append(_agent)

    for idx, (agent_line, _) in enumerate(agents_data):
        emoji = agent_emoji[idx % len(agent_emoji)]
        parts = agent_line.split('-', 1)
        role = parts[0].strip()
        desc = parts[1].strip() if len(parts) > 1 else ""
        if desc:
            print(f"Agent {idx+1} ({emo(idx)} {role}): {desc}")
        else:
            print(f"Agent {idx+1} ({emo(idx)} {role})")

    fewshot_examplers = ""
    medical_agent = Agent(instruction='You are a helpful medical agent.', role='medical expert', model_info=model_id)
    if getattr(args, "dataset", "") == 'medqa':
        random.shuffle(examplers or [])
        for ie, ex in enumerate((examplers or [])[:5]):
            ex_q = f"[Example {ie+1}]\n" + ex['question']
            opts = [f"({k}) {v}" for k, v in ex['options'].items()]
            random.shuffle(opts)
            ex_q += " " + " ".join(opts)
            ex_ans = f"Answer: ({ex['answer_idx']}) {ex['answer']}"
            ex_reason = tmp_agent.chat(
                "Below is an example of medical knowledge question and answer. "
                "After reviewing them, provide 1–2 sentence reason that supports the answer as if you didn’t know it ahead.\n\n"
                f"Question: {ex_q}\n\nAnswer: {ex_ans}"
            )
            ex_q += f"\n{ex_ans}\n{ex_reason}\n\n"
            fewshot_examplers += ex_q
    roles_list = [a.role for a in medical_agents]

    try:
        run_dir = choose_run_dir_for_question(question)
        run_logger = RunLogger(run_dir, question, roles_list)
    except Exception as e:
        print(f"[WARN] RunLogger init failed at {str(run_dir) if 'run_dir' in locals() else '<unset>'}: {e}")
        print(traceback.format_exc())
        # last-ditch temp fallback
        run_dir = Path(tempfile.mkdtemp(prefix="mdagents_run_"))
        run_logger = RunLogger(run_dir, question, roles_list)

    print(f"[LOG] Writing run artifacts to: {run_dir}")


    print()
    cprint("[INFO] Step 2. Collaborative Decision Making", 'yellow', attrs=['blink'])
    cprint("[INFO] Step 2.1. Hierarchy Selection", 'yellow', attrs=['blink'])
    print_tree(hierarchy_agents[0], horizontal=False)
    print()

    # num_rounds, num_turns = 5, 5
    num_rounds = int(os.getenv("MDT_ROUNDS", "2"))
    num_turns  = int(os.getenv("MDT_TURNS",  "2"))

    num_agents = len(medical_agents)
    interaction_log = {
        f'Round {r}': {
            f'Turn {t}': {f'Agent {i}': {f'Agent {j}': None for j in range(1, num_agents+1)}
                          for i in range(1, num_agents+1)}
            for t in range(1, num_turns+1)
        } for r in range(1, num_rounds+1)
    }

    cprint("[INFO] Step 2.2. Participatory Debate", 'yellow', attrs=['blink'])
    round_opinions = {n: {} for n in range(1, num_rounds+1)}
    round_answers  = {n: None for n in range(1, num_rounds+1)}

    initial_report = ""
    for k, v in agent_dict.items():
        opinion = v.chat(
            f"Given the examplers, please return your answer to the medical query among the option provided.\n\n"
            f"{fewshot_examplers}\n\nQuestion: {question}\n\nYour answer should be like:\n\nAnswer: "
        )
        initial_report += f"({k.lower()}): {opinion}\n"
        round_opinions[1][k.lower()] = opinion

    final_answer = None
    for n in range(1, num_rounds+1):
        already = set() 
        print(f"== Round {n} ==")
        round_name = f"Round {n}"
        agent_rs = Agent(
            instruction="You are a medical assistant who excels at summarizing and synthesizing based on multiple experts from various domain experts.",
            role="medical assistant", model_info=model_id
        )
        agent_rs.chat(agent_rs.instruction)

        assessment = "".join(f"({k.lower()}): {v}\n" for k, v in round_opinions[n].items())
        report = agent_rs.chat(
            "Here are some reports from different medical domain experts.\n\n"
            f"{assessment}\n\n"
            "You need to complete the following steps\n"
            "1. Take careful and comprehensive consideration of the following reports.\n"
            "2. Extract key knowledge from the following reports.\n"
            "3. Derive the comprehensive and summarized analysis based on the knowledge\n"
            "4. Your ultimate goal is to derive a refined and synthesized report based on the following reports.\n\n"
            "You should output in exactly the same format as: Key Knowledge:; Total Analysis:"
        )

        num_yes = 0
        for turn_num in range(num_turns):
            turn_name = f"Turn {turn_num + 1}"
            print(f"|_{turn_name}")

            for idx, v in enumerate(medical_agents):
                all_comments = "".join(
                    f"{_k} -> Agent {idx+1}: {_v[f'Agent {idx+1}']}\n"
                    for _k, _v in interaction_log[round_name][turn_name].items()
                )

                # Keep the context short; use the summarized report if available
                context = assessment if n == 1 else all_comments
                context = (context or "")
                if len(context) > 600:
                    context = context[:600] + " …"
                # 1) ask: talk or not (strict YES/NO on first line)
                participate = v.chat(
                    "FIRST LINE ONLY: reply YES or NO.\nDo you want to talk to any expert?\n",
                    chat_mode=False,  # greedy, short
                )
                print(f"RAW participate[{idx+1}]: {repr(participate)}")
                want = wants_to_talk(participate)
                force = os.getenv("MDT_FORCE_TALK", "0") == "1"
                if not (want or force):
                    print(f" Agent {idx+1} ({emo(idx)} {v.role}): 🤐")
                    run_logger.log_status(n, turn_num + 1, idx + 1, "pass", reason="declined")
                    continue


                # 2) choose targets (digits only)
                choose = v.chat(
                    f"FIRST LINE ONLY: list expert numbers (1..{num_agents}) separated by commas. "
                    "Return 0 for none. Examples: 1  or  1,3  or  0",
                    chat_mode=False,  # greedy, short
                )
                print(f"RAW chosen_expert[{idx+1}]: {repr(choose)}")

                targets = [t for t in parse_targets(choose, num_agents) if t != (idx+1)]

                # dedupe + cap fan-out
                seen, deduped = set(), []
                for t in targets:
                    if t not in seen:
                        deduped.append(t); seen.add(t)
                targets = deduped[:2]

                if not targets:
                    print(f" Agent {idx+1} ({emo(idx)} {v.role}): 🤐")
                    run_logger.log_status(n, turn_num + 1, idx + 1, "pass", reason="no_targets")
                    continue

                for ce in targets:
                    key = (idx+1, ce, n, turn_num+1)
                    if key in already:
                        continue
                    already.add(key)
                    msg_prompt = (
                        f"You are {v.role}. The team is answering this MCQ:\n{question}\n\n"
                        f"Write ONLY THREE LINES to Agent {ce} ({medical_agents[ce-1].role}).\n"
                        "Line 1: Hypothesis: <one clause>\n"
                        "Line 2: Question: <one direct question for them>\n"
                        "Line 3: Next step: <one concrete action>\n"
                        "Output EXACTLY those three lines. No quotes, no extra lines, nothing else."
                    )
                    msg = v.chat(msg_prompt, chat_mode=False)  # greedy, short
                    msg = _trim_to_three_lines(msg)

                    if any("(missing)" in line.lower() for line in msg.splitlines()):
                        # one strict retry
                        msg_retry = v.chat(
                            msg_prompt + "\nRespond again, exactly 3 lines starting with "
                                        "'Hypothesis:', 'Question:', 'Next step:'.",
                            chat_mode=False
                        )
                        msg2 = _trim_to_three_lines(msg_retry)
                        if not any("(missing)" in line.lower() for line in msg2.splitlines()):
                            msg = msg2


                    # NEW: ask the recipient to reply in 3 short lines
                    # NEW: ask the recipient to reply in 3 short lines
                    # reply = medical_agents[ce-1].chat(
                    #     (
                    #         f"You are Agent {ce} ({medical_agents[ce-1].role}). "
                    #         f"You just received this message from Agent {idx+1} ({medical_agents[idx].role}):\n"
                    #         f"{msg}\n\n"
                    #         "Respond to the AGENT ONLY (not the user).\n"
                    #         "Do NOT apologize. Do NOT say the system is unavailable.\n"
                    #         "Reply in EXACTLY THREE LINES with this format:\n"
                    #         "Answer: <one short sentence>\n"
                    #         "Rationale: <very short reason>\n"
                    #         "Action: <concrete next step>\n"
                    #         "No extra text."
                    #     ),
                    #     # IMPORTANT: use your chat wrapper if it builds role/system templates correctly.
                    #     # If your .chat defaults to greedy, that’s fine, but we’ll keep it short:
                    #     chat_mode=True,           # <-- this usually restores proper chat templating
                    #     max_new_tokens=60,
                    #     do_sample=False           # deterministic, matches the 3-line post-trim you do
                    # )


                    # # keep only the first 3 non-empty lines
                    # r_lines = [l.strip() for l in reply.splitlines() if l.strip()]
                    # reply3 = "\n".join(r_lines[:3]) if r_lines else "Answer: (none)\nRationale: (none)\nAction: (none)"

                    # decode_args = {"max_new_tokens": 60, "do_sample": False, "temperature": None}
                    # pipe_for_meta = getattr(medical_agents[ce-1], "_pipe", None)  # may be None for HF path; ok
                    # meta = RunLogger.build_edge_meta(pipe_for_meta, msg, reply3, decode_args)


                    # # log reverse edge (dst -> src)
                    # interaction_log[round_name][turn_name][f'Agent {ce}'][f'Agent {idx+1}'] = reply3
                    # run_logger.log_edge(n, turn_num + 1, ce, idx + 1, reply3, meta=meta)
                    # TIME the recipient's generation
                    t0 = time.perf_counter_ns()
                    reply = medical_agents[ce-1].chat(
                        (
                            f"You are Agent {ce} ({medical_agents[ce-1].role}). "
                            f"You just received this message from Agent {idx+1} ({medical_agents[idx].role}):\n"
                            f"{msg}\n\n"
                            "Respond to the AGENT ONLY (not the user).\n"
                            "Do NOT apologize. Do NOT say the system is unavailable.\n"
                            "Reply in EXACTLY THREE LINES with this format:\n"
                            "Answer: <one short sentence>\n"
                            "Rationale: <very short reason>\n"
                            "Action: <concrete next step>\n"
                            "No extra text."
                        ),
                        chat_mode=True,
                        do_sample=False,
                        max_new_tokens=60
                    )
                    t1 = time.perf_counter_ns()
                    elapsed_ms = round((t1 - t0) / 1e6, 3)  # monotonic, high-res

                    # trim to 3 lines
                    r_lines = [l.strip() for l in reply.splitlines() if l.strip()]
                    reply3 = "\n".join(r_lines[:3]) if r_lines else "Answer: (none)\nRationale: (none)\nAction: (none)"

                    # include timing in decode_args so the logger picks it up
                    decode_args = {"max_new_tokens": 60, "do_sample": False, "temperature": None, "timing_ms": elapsed_ms}

                    pipe_for_meta = getattr(medical_agents[ce-1], "_pipe", None)
                    meta = RunLogger.build_edge_meta(pipe_for_meta, msg, reply3, decode_args)

                    # record & log
                    interaction_log[round_name][turn_name][f'Agent {ce}'][f'Agent {idx+1}'] = reply3
                    run_logger.log_edge(n, turn_num + 1, ce, idx + 1, reply3, meta=meta)

                    # reply = medical_agents[ce-1].chat(
                    #     (
                    #         f"You are Agent {ce} ({medical_agents[ce-1].role}). "
                    #         f"You just received this message from Agent {idx+1} ({medical_agents[idx].role}):\n"
                    #         f"{msg}\n\n"
                    #         "Respond to the AGENT ONLY (not the user).\n"
                    #         "Do NOT apologize. Do NOT say the system is unavailable.\n"
                    #         "Reply in EXACTLY THREE LINES with this format:\n"
                    #         "Answer: <one short sentence>\n"
                    #         "Rationale: <very short reason>\n"
                    #         "Action: <concrete next step>\n"
                    #         "No extra text."
                    #     ),
                    #     chat_mode=True,      # ensure chat template/roles are applied
                    #     do_sample=False,     # deterministic & short
                    #     max_new_tokens=60
                    # )

                    # r_lines = [l.strip() for l in reply.splitlines() if l.strip()]
                    # reply3 = "\n".join(r_lines[:3]) if r_lines else \
                    #         "Answer: (none)\nRationale: (none)\nAction: (none)"

                    # # log reverse edge (dst -> src)
                    # interaction_log[round_name][turn_name][f'Agent {ce}'][f'Agent {idx+1}'] = reply3
                    # run_logger.log_edge(n, turn_num + 1, ce, idx + 1, reply3)

                    # print(f" Agent {ce} ({emo(ce-1)} {medical_agents[ce-1].role}) -> "
                    #     f"Agent {idx+1} ({emo(idx)} {medical_agents[idx].role}) :\n{reply3}")
                    



                    # print(f" Agent {ce} ({emo(ce-1)} {medical_agents[ce-1].role}) -> Agent {idx+1} ({emo(idx)} {medical_agents[idx].role}) :\n{reply3}")

                    # num_yes += 1

                # else:
                #     print(f" Agent {idx+1} ({emo(idx)} {v.role}): \U0001f910")
                    # No targets case
                    # run_logger.log_status(n, turn_num + 1, idx + 1, "pass", reason="no_targets_or_declined")


                # participate = v.chat(
                #     "Given the opinions from other medical experts in your team, please indicate whether you want to talk to any expert (yes/no)\n\n"
                #     "Opinions:\n" + (assessment if n == 1 else all_comments)
                # )

                # if 'yes' in (participate or "").lower().strip():
                #     chosen_expert = v.chat(
                #         f"Enter the number of the expert you want to talk to:\n{agent_list}\n"
                #         f"For example, if you want to talk with Agent 1. Pediatrician, return just 1. "
                #         f"If you want to talk with more than one expert, please return 1,2 and don't return the reasons."
                #     )
                #     chosen_experts = [int(ce) for ce in str(chosen_expert).replace('.', ',').split(',') if ce.strip().isdigit()]
                #     for ce in chosen_experts:
                #         specific_question = v.chat(
                #             f"Please remind your medical expertise and then leave your opinion to an expert you chose (Agent {ce}. {medical_agents[ce-1].role}). "
                #             f"You should deliver your opinion once you are confident enough and in a way to convince other expert with a short reason."
                #         )
                #         print(f" Agent {idx+1} ({emo(idx)} {medical_agents[idx].role}) -> Agent {ce} ({emo(ce-1)} {medical_agents[ce-1].role}) : {specific_question}")
                #         interaction_log[round_name][turn_name][f'Agent {idx+1}'][f'Agent {ce}'] = specific_question
                #     num_yes += 1
                # else:
                #     print(f" Agent {idx+1} ({emo(idx)} {v.role}): \U0001f910")

            if num_yes == 0:
                break
        if num_yes == 0:
            break

        tmp_final_answer = {}
        for i, agent in enumerate(medical_agents):
            response = agent.chat(
                "Now that you've interacted with other medical experts, remind your expertise and the comments from other experts and make your final answer to the given question:\n"
                f"{question}\nAnswer: "
            )
            tmp_final_answer[agent.role] = response

        round_answers[round_name] = tmp_final_answer
        final_answer = tmp_final_answer

    print('\nInteraction Log')
    myTable = PrettyTable([''] + [f"Agent {i+1} ({emo(i)})" for i in range(len(medical_agents))])
    for i in range(1, len(medical_agents)+1):
        row = [f"Agent {i} ({emo(i-1)})"]
        for j in range(1, len(medical_agents)+1):
            if i == j:
                row.append('')
            else:
                i2j = any(interaction_log[f'Round {k}'][f'Turn {l}'][f'Agent {i}'][f'Agent {j}'] is not None
                          for k in range(1, len(interaction_log)+1)
                          for l in range(1, len(interaction_log['Round 1'])+1))
                j2i = any(interaction_log[f'Round {k}'][f'Turn {l}'][f'Agent {j}'][f'Agent {i}'] is not None
                          for k in range(1, len(interaction_log)+1)
                          for l in range(1, len(interaction_log['Round 1'])+1))
                if not i2j and not j2i:
                    row.append(' ')
                elif i2j and not j2i:
                    row.append(f'\u270B ({i}->{j})')
                elif j2i and not i2j:
                    row.append(f'\u270B ({i}<-{j})')
                elif i2j and j2i:
                    row.append(f'\u270B ({i}<->{j})')
        myTable.add_row(row)
        if i != len(medical_agents):
            myTable.add_row(['' for _ in range(len(medical_agents)+1)])
    print(myTable)
    print("\n--- Transcript (trimmed) ---")
    for r in range(1, num_rounds+1):
        rn = f"Round {r}"
        for t in range(1, num_turns+1):
            tn = f"Turn {t}"
            for i in range(1, num_agents+1):
                for j in range(1, num_agents+1):
                    if i == j: 
                        continue
                    m = interaction_log[rn][tn][f'Agent {i}'][f'Agent {j}']
                    if m:
                        head = m.replace("\n", " | ")[:180]
                        print(f"[R{r} T{t}] Agent {i}({roles_list[i-1]}) → Agent {j}({roles_list[j-1]}): {head}")


    cprint("\n[INFO] Step 3. Final Decision", 'yellow', attrs=['blink'])
    moderator = Agent(
    "You are a final medical decision maker who outputs only the letter choice and one short reason.",
    "Moderator", model_info=model_id
    )
    # moderator.chat('You are a final medical decision maker who reviews all opinions from different medical experts and make final decision.')
    # _decision = moderator.temp_responses(
    #     "Given each agent's final answer, please review each agent's opinion and make the final answer to the question by taking majority vote. "
    #     "Your answer should be like below format:\nAnswer: C) 2th pharyngeal arch\n"
    #     f"{final_answer}\n\nQuestion: {question}"
    # )
    # final_decision = {'majority': _decision}
    prompt = (
    "Review the experts' final answers and choose the single best option.\n"
    "FIRST LINE: only the letter A/B/C/D.\n"
    "SECOND LINE: one short reason.\n\n"
    f"Experts:\n{final_answer}\n\nQuestion:\n{question}\nChoice:"
    )
    _decision = moderator.temp_responses(prompt)
    final_decision = {"majority": _decision}
    print()
    return final_decision

# def process_intermediate_query(question, examplers, model_id, args):
#     cprint("[INFO] Step 1. Expert Recruitment", 'yellow', attrs=['blink'])
#     recruit_prompt = f"""You are an experienced medical expert who recruits a group of experts with diverse identity and ask them to discuss and solve the given medical query."""
    
#     tmp_agent = Agent(instruction=recruit_prompt, role='recruiter', model_info=model_id)
#     tmp_agent.chat(recruit_prompt)
    
#     num_agents = 5  # You can adjust this number as needed
#     recruited = tmp_agent.chat(f"Question: {question}\nYou can recruit {num_agents} experts in different medical expertise. Considering the medical question and the options for the answer, what kind of experts will you recruit to better make an accurate answer?\nAlso, you need to specify the communication structure between experts (e.g., Pulmonologist == Neonatologist == Medical Geneticist == Pediatrician > Cardiologist), or indicate if they are independent.\n\nFor example, if you want to recruit five experts, you answer can be like:\n1. Pediatrician - Specializes in the medical care of infants, children, and adolescents. - Hierarchy: Independent\n2. Cardiologist - Focuses on the diagnosis and treatment of heart and blood vessel-related conditions. - Hierarchy: Pediatrician > Cardiologist\n3. Pulmonologist - Specializes in the diagnosis and treatment of respiratory system disorders. - Hierarchy: Independent\n4. Neonatologist - Focuses on the care of newborn infants, especially those who are born prematurely or have medical issues at birth. - Hierarchy: Independent\n5. Medical Geneticist - Specializes in the study of genes and heredity. - Hierarchy: Independent\n\nPlease answer in above format, and do not include your reason.")

#     # agents_info = [agent_info.split(" - Hierarchy: ") for agent_info in recruited.split('\n') if agent_info]
#     # agents_data = [(info[0], info[1]) if len(info) > 1 else (info[0], None) for info in agents_info]
# # ## INTERMEDIATE RECRUITER FALLBACK
# if not agents_data:
#     agents_data = [
#         ("1. Medical Ethicist - Disclosure, duty to inform, autonomy.", "Independent"),
#         ("2. Orthopedic Surgeon - Operative standards & documentation.", "Independent"),
#         ("3. Patient Safety Officer - Disclosure & RCA process.", "Independent"),
#         ("4. Risk Manager - Incident reporting & policy.", "Independent"),
#         ("5. Hospital Legal Counsel - Regulatory/mandated disclosure.", "Independent"),
#     ]

#     # --- NEW: robust recruiter-line parser ---------------------------------
#     def _parse_agent_lines(raw: str):
#         """Return [(line_before_hierarchy, hierarchy_or_None), …] without IndexError."""
#         out = []
#         for line in raw.splitlines():
#             line = line.strip()
#             if not line:
#                 continue                      # skip blank lines
#             head, *tail = line.split(" - Hierarchy:", 1)
#             hier = tail[0].strip() if tail else None   # None when no ‘ - Hierarchy: ’
#             out.append((head, hier))
#         return out
#     # -----------------------------------------------------------------------

#     # agents_data = _parse_agent_lines(recruited)
#     clean_lines = []
#     for ln in recruited.splitlines():
#         ln = ln.strip()
#         # keep only lines that *start* with a number + dot + some text + " - Hierarchy:"
#         if re.match(r"^\d+\.\s+[^-]+ - .*Hierarchy:", ln, re.I):
#             clean_lines.append(ln)
#     agents_data = _parse_agent_lines("\n".join(clean_lines))


#     agent_emoji = ['\U0001F468\u200D\u2695\uFE0F', '\U0001F468\U0001F3FB\u200D\u2695\uFE0F', '\U0001F469\U0001F3FC\u200D\u2695\uFE0F', '\U0001F469\U0001F3FB\u200D\u2695\uFE0F', '\U0001f9d1\u200D\u2695\uFE0F', '\U0001f9d1\U0001f3ff\u200D\u2695\uFE0F', '\U0001f468\U0001f3ff\u200D\u2695\uFE0F', '\U0001f468\U0001f3fd\u200D\u2695\uFE0F', '\U0001f9d1\U0001f3fd\u200D\u2695\uFE0F', '\U0001F468\U0001F3FD\u200D\u2695\uFE0F']
#     random.shuffle(agent_emoji)
#     def emo(i: int) -> str:
#         """Modulo-index into agent_emoji so we never run out."""
#         return agent_emoji[i % len(agent_emoji)]

#     hierarchy_agents = parse_hierarchy(agents_data, agent_emoji)

#     agent_list = ""
#     for i, agent in enumerate(agents_data, 1):
#         # role: everything after the leading number / dot
#         agent_role = agent[0].split('-', 1)[0].split('.', 1)[-1].strip().lower()

#         # description: text after the first dash, if any; otherwise same as role
#         parts = agent[0].split('-', 1)          # at most 2 pieces
#         description = parts[1].strip().lower() if len(parts) > 1 else agent_role
#         agent_list += f"Agent {i}: {agent_role} - {description}\n"
#     # for i, agent in enumerate(agents_data):   
#     #     agent_role = agent[0].split('-')[0].split('.')[1].strip().lower()
#     #     description = agent[0].split('-')[1].strip().lower()
#     #     agent_list += f"Agent {i+1}: {agent_role} - {description}\n"

#     agent_dict = {}
#     medical_agents = []
#     for agent in agents_data:
#         text = agent[0]

#         agent_role = text.split('-', 1)[0].split('.', 1)[-1].strip().lower()
#         parts = text.split('-', 1)
#         description = parts[1].strip().lower() if len(parts) > 1 else agent_role

#         # try:
#         #     agent_role = agent[0].split('-')[0].split('.')[1].strip().lower()
#         #     description = agent[0].split('-')[1].strip().lower()
#         # except:
#         #     continue
        
#         inst_prompt = f"""You are a {agent_role} who {description}. Your job is to collaborate with other medical experts in a team."""
#         _agent = Agent(instruction=inst_prompt, role=agent_role, model_info=model_id)
        
#         _agent.chat(inst_prompt)
#         agent_dict[agent_role] = _agent
#         medical_agents.append(_agent)

#     for idx, (agent_line, _) in enumerate(agents_data):
#         emoji = agent_emoji[idx % len(agent_emoji)]
#         parts = agent_line.split('-', 1)          # at most 2 pieces
#         role = parts[0].strip()                   # always exists
#         desc = parts[1].strip() if len(parts) > 1 else ""   # may be absent

#         if desc:
#             print(f"Agent {idx+1} ({emo(idx)} {role}): {desc}")
#         else:
#             print(f"Agent {idx+1} ({emo(idx)} {role})")

#     fewshot_examplers = ""
#     medical_agent = Agent(instruction='You are a helpful medical agent.', role='medical expert', model_info=model_id)
#     if args.dataset == 'medqa':
#         random.shuffle(examplers)
#         for ie, exampler in enumerate(examplers[:5]):
#             exampler_question = f"[Example {ie+1}]\n" + exampler['question']
#             options = [f"({k}) {v}" for k, v in exampler['options'].items()]
#             random.shuffle(options)
#             exampler_question += " " + " ".join(options)
#             exampler_answer = f"Answer: ({exampler['answer_idx']}) {exampler['answer']}"
#             exampler_reason = tmp_agent.chat(f"Below is an example of medical knowledge question and answer. After reviewing the below medical question and answering, can you provide 1-2 sentences of reason that support the answer as you didn't know the answer ahead?\n\nQuestion: {exampler_question}\n\nAnswer: {exampler_answer}")
            
#             exampler_question += f"\n{exampler_answer}\n{exampler_reason}\n\n"
#             fewshot_examplers += exampler_question

#     print()
#     cprint("[INFO] Step 2. Collaborative Decision Making", 'yellow', attrs=['blink'])
#     cprint("[INFO] Step 2.1. Hierarchy Selection", 'yellow', attrs=['blink'])
#     print_tree(hierarchy_agents[0], horizontal=False)
#     print()

#     num_rounds = 5
#     num_turns = 5
#     num_agents = len(medical_agents)

#     interaction_log = {f'Round {round_num}': {f'Turn {turn_num}': {f'Agent {source_agent_num}': {f'Agent {target_agent_num}': None for target_agent_num in range(1, num_agents + 1)} for source_agent_num in range(1, num_agents + 1)} for turn_num in range(1, num_turns + 1)} for round_num in range(1, num_rounds + 1)}

#     cprint("[INFO] Step 2.2. Participatory Debate", 'yellow', attrs=['blink'])

#     round_opinions = {n: {} for n in range(1, num_rounds+1)}
#     round_answers = {n: None for n in range(1, num_rounds+1)}
#     initial_report = ""
#     for k, v in agent_dict.items():
#         opinion = v.chat(f'''Given the examplers, please return your answer to the medical query among the option provided.\n\n{fewshot_examplers}\n\nQuestion: {question}\n\nYour answer should be like below format.\n\nAnswer: ''', img_path=None)
#         initial_report += f"({k.lower()}): {opinion}\n"
#         round_opinions[1][k.lower()] = opinion

#     final_answer = None
#     for n in range(1, num_rounds+1):
#         print(f"== Round {n} ==")
#         round_name = f"Round {n}"
#         agent_rs = Agent(instruction="You are a medical assistant who excels at summarizing and synthesizing based on multiple experts from various domain experts.", role="medical assistant", model_info=model_id)
#         agent_rs.chat("You are a medical assistant who excels at summarizing and synthesizing based on multiple experts from various domain experts.")
        
#         assessment = "".join(f"({k.lower()}): {v}\n" for k, v in round_opinions[n].items())

#         report = agent_rs.chat(f'''Here are some reports from different medical domain experts.\n\n{assessment}\n\nYou need to complete the following steps\n1. Take careful and comprehensive consideration of the following reports.\n2. Extract key knowledge from the following reports.\n3. Derive the comprehensive and summarized analysis based on the knowledge\n4. Your ultimate goal is to derive a refined and synthesized report based on the following reports.\n\nYou should output in exactly the same format as: Key Knowledge:; Total Analysis:''')
        
#         for turn_num in range(num_turns):
#             turn_name = f"Turn {turn_num + 1}"
#             print(f"|_{turn_name}")

#             num_yes = 0
#             for idx, v in enumerate(medical_agents):
#                 all_comments = "".join(f"{_k} -> Agent {idx+1}: {_v[f'Agent {idx+1}']}\n" for _k, _v in interaction_log[round_name][turn_name].items())
                
#                 participate = v.chat("Given the opinions from other medical experts in your team, please indicate whether you want to talk to any expert (yes/no)\n\nOpinions:\n{}".format(assessment if n == 1 else all_comments))
                
#                 if 'yes' in participate.lower().strip():                
#                     chosen_expert = v.chat(f"Enter the number of the expert you want to talk to:\n{agent_list}\nFor example, if you want to talk with Agent 1. Pediatrician, return just 1. If you want to talk with more than one expert, please return 1,2 and don't return the reasons.")
                    
#                     chosen_experts = [int(ce) for ce in chosen_expert.replace('.', ',').split(',') if ce.strip().isdigit()]

#                     for ce in chosen_experts:
#                         specific_question = v.chat(f"Please remind your medical expertise and then leave your opinion to an expert you chose (Agent {ce}. {medical_agents[ce-1].role}). You should deliver your opinion once you are confident enough and in a way to convince other expert with a short reason.")
                        
#                         print(f" Agent {idx+1} ({emo(idx)} {medical_agents[idx].role}) -> Agent {ce} ({emo(ce-1)} {medical_agents[ce-1].role}) : {specific_question}")
#                         interaction_log[round_name][turn_name][f'Agent {idx+1}'][f'Agent {ce}'] = specific_question
                
#                     num_yes += 1
#                 else:
#                     print(f" Agent {idx+1} ({emo(idx)} {v.role}): \U0001f910")

#             if num_yes == 0:
#                 break
        
#         if num_yes == 0:
#             break

#         tmp_final_answer = {}
#         for i, agent in enumerate(medical_agents):
#             response = agent.chat(f"Now that you've interacted with other medical experts, remind your expertise and the comments from other experts and make your final answer to the given question:\n{question}\nAnswer: ")
#             tmp_final_answer[agent.role] = response

#         round_answers[round_name] = tmp_final_answer
#         final_answer = tmp_final_answer

#     print('\nInteraction Log')        
#     myTable = PrettyTable([''] + [f"Agent {i+1} ({emo(i)})" for i in range(len(medical_agents))])

#     for i in range(1, len(medical_agents)+1):
#         row = [f"Agent {i} ({emo(i-1)})"]
#         for j in range(1, len(medical_agents)+1):
#             if i == j:
#                 row.append('')
#             else:
#                 i2j = any(interaction_log[f'Round {k}'][f'Turn {l}'][f'Agent {i}'][f'Agent {j}'] is not None
#                           for k in range(1, len(interaction_log)+1)
#                           for l in range(1, len(interaction_log['Round 1'])+1))
#                 j2i = any(interaction_log[f'Round {k}'][f'Turn {l}'][f'Agent {j}'][f'Agent {i}'] is not None
#                           for k in range(1, len(interaction_log)+1)
#                           for l in range(1, len(interaction_log['Round 1'])+1))
                
#                 if not i2j and not j2i:
#                     row.append(' ')
#                 elif i2j and not j2i:
#                     row.append(f'\u270B ({i}->{j})')
#                 elif j2i and not i2j:
#                     row.append(f'\u270B ({i}<-{j})')
#                 elif i2j and j2i:
#                     row.append(f'\u270B ({i}<->{j})')

#         myTable.add_row(row)
#         if i != len(medical_agents):
#             myTable.add_row(['' for _ in range(len(medical_agents)+1)])
    
#     print(myTable)

#     cprint("\n[INFO] Step 3. Final Decision", 'yellow', attrs=['blink'])
    
#     moderator = Agent("You are a final medical decision maker who reviews all opinions from different medical experts and make final decision.", "Moderator", model_info=model_id)
#     moderator.chat('You are a final medical decision maker who reviews all opinions from different medical experts and make final decision.')
    
#     _decision = moderator.temp_responses(f"Given each agent's final answer, please review each agent's opinion and make the final answer to the question by taking majority vote. Your answer should be like below format:\nAnswer: C) 2th pharyngeal arch\n{final_answer}\n\nQuestion: {question}", img_path=None)
#     final_decision = {'majority': _decision}

#     #    print(f"{'\U0001F468\u200D\u2696\uFE0F'} moderator's final decision (by majority vote):", _decision)
#     print()

#     return final_decision

def process_advanced_query(question, model, args):
    print("[STEP 1] Recruitment")
    group_instances = []

    recruit_prompt = f"""You are an experienced medical expert. Given the complex medical query, you need to organize Multidisciplinary Teams (MDTs) and the members in MDT to make accurate and robust answer."""

    tmp_agent = Agent(instruction=recruit_prompt, role='recruiter', model_info=model_id)
    tmp_agent.chat(recruit_prompt)

    num_teams = 3  # You can adjust this number as needed
    num_agents = 3  # You can adjust this number as needed

    recruited = tmp_agent.chat(f"Question: {question}\n\nYou should organize {num_teams} MDTs with different specialties or purposes and each MDT should have {num_agents} clinicians. Considering the medical question and the options, please return your recruitment plan to better make an accurate answer.\n\nFor example, the following can an example answer:\nGroup 1 - Initial Assessment Team (IAT)\nMember 1: Otolaryngologist (ENT Surgeon) (Lead) - Specializes in ear, nose, and throat surgery, including thyroidectomy. This member leads the group due to their critical role in the surgical intervention and managing any surgical complications, such as nerve damage.\nMember 2: General Surgeon - Provides additional surgical expertise and supports in the overall management of thyroid surgery complications.\nMember 3: Anesthesiologist - Focuses on perioperative care, pain management, and assessing any complications from anesthesia that may impact voice and airway function.\n\nGroup 2 - Diagnostic Evidence Team (DET)\nMember 1: Endocrinologist (Lead) - Oversees the long-term management of Graves' disease, including hormonal therapy and monitoring for any related complications post-surgery.\nMember 2: Speech-Language Pathologist - Specializes in voice and swallowing disorders, providing rehabilitation services to improve the patient's speech and voice quality following nerve damage.\nMember 3: Neurologist - Assesses and advises on nerve damage and potential recovery strategies, contributing neurological expertise to the patient's care.\n\nGroup 3 - Patient History Team (PHT)\nMember 1: Psychiatrist or Psychologist (Lead) - Addresses any psychological impacts of the chronic disease and its treatments, including issues related to voice changes, self-esteem, and coping strategies.\nMember 2: Physical Therapist - Offers exercises and strategies to maintain physical health and potentially support vocal function recovery indirectly through overall well-being.\nMember 3: Vocational Therapist - Assists the patient in adapting to changes in voice, especially if their profession relies heavily on vocal communication, helping them find strategies to maintain their occupational roles.\n\nGroup 4 - Final Review and Decision Team (FRDT)\nMember 1: Senior Consultant from each specialty (Lead) - Provides overarching expertise and guidance in decision\nMember 2: Clinical Decision Specialist - Coordinates the different recommendations from the various teams and formulates a comprehensive treatment plan.\nMember 3: Advanced Diagnostic Support - Utilizes advanced diagnostic tools and techniques to confirm the exact extent and cause of nerve damage, aiding in the final decision.\n\nAbove is just an example, thus, you should organize your own unique MDTs but you should include Initial Assessment Team (IAT) and Final Review and Decision Team (FRDT) in your recruitment plan. When you return your answer, please strictly refer to the above format.")

    groups = [group.strip() for group in recruited.split("Group") if group.strip()]
    group_strings = ["Group " + group for group in groups]
    
    for i1, gs in enumerate(group_strings):
        res_gs = parse_group_info(gs)
        print(f"Group {i1+1} - {res_gs['group_goal']}")
        for i2, member in enumerate(res_gs['members']):
            print(f" Member {i2+1} ({member['role']}): {member['expertise_description']}")
        print()

        group_instance = Group(res_gs['group_goal'], res_gs['members'], question)
        group_instances.append(group_instance)

    # STEP 2. initial assessment from each group
    # STEP 2.1. IAP Process
    initial_assessments = []
    for group_instance in group_instances:
        if 'initial' in group_instance.goal.lower() or 'iap' in group_instance.goal.lower():
            init_assessment = group_instance.interact(comm_type='internal')
            initial_assessments.append([group_instance.goal, init_assessment])

    initial_assessment_report = ""
    for idx, init_assess in enumerate(initial_assessments):
        initial_assessment_report += f"Group {idx+1} - {init_assess[0]}\n{init_assess[1]}\n\n"

    # STEP 2.2. other MDTs Process
    assessments = []
    for group_instance in group_instances:
        if 'initial' not in group_instance.goal.lower() and 'iap' not in group_instance.goal.lower():
            assessment = group_instance.interact(comm_type='internal')
            assessments.append([group_instance.goal, assessment])
    
    assessment_report = ""
    for idx, assess in enumerate(assessments):
        assessment_report += f"Group {idx+1} - {assess[0]}\n{assess[1]}\n\n"
    
    # STEP 2.3. FRDT Process
    final_decisions = []
    for group_instance in group_instances:
        if 'review' in group_instance.goal.lower() or 'decision' in group_instance.goal.lower() or 'frdt' in group_instance.goal.lower():
            decision = group_instance.interact(comm_type='internal')
            final_decisions.append([group_instance.goal, decision])
    
    compiled_report = ""
    for idx, decision in enumerate(final_decisions):
        compiled_report += f"Group {idx+1} - {decision[0]}\n{decision[1]}\n\n"

    # STEP 3. Final Decision
    decision_prompt = f"""You are an experienced medical expert. Now, given the investigations from multidisciplinary teams (MDT), please review them very carefully and return your final decision to the medical query."""
    tmp_agent = Agent(instruction=decision_prompt, role='decision maker', model_info=model_id)
    tmp_agent.chat(decision_prompt)

    final_decision = tmp_agent.temp_responses(f"""Investigation:\n{initial_assessment_report}\n\nQuestion: {question}""", img_path=None)

    return final_decision
# === HF (Transformers) backend monkeypatch (auto) ============================
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch
    _HF_CACHE = {}

    # def _is_hf_id(model_info: str) -> bool:
    #     mi = (model_info or "").lower()
    #     return ("/" in mi) or any(k in mi for k in ("llama","mistral","mixtral","qwen","phi","vicuna","falcon","zephyr"))

    def _is_hf_id(model_info: str) -> bool:
        mi = (model_info or "").lower()
        if mi.startswith("llama2-uncensored"):
            return False
        return "/" in mi



    def _hf_ensure(model_id: str):
        if model_id not in _HF_CACHE:
            tok = AutoTokenizer.from_pretrained(model_id, use_fast=True, legacy=False)
            try:
                model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")
            except Exception:
                model = AutoModelForCausalLM.from_pretrained(model_id)
            _HF_CACHE[model_id] = (tok, model)
        return _HF_CACHE[model_id]

    def _llama_template(sys_msg: str, user_msg: str) -> str:
        sys_msg  = (sys_msg or "").strip()
        user_msg = (user_msg or "").strip()
        return f"<<SYS>>\n{sys_msg}\n<</SYS>>\n\n[INST] {user_msg} [/INST]" if sys_msg else f"[INST] {user_msg} [/INST]"

    # Monkeypatch Agent.chat
    _orig_chat = Agent.chat
    def _chat_hf(self, message, img_path=None, chat_mode=True, **gen_kw):
        """
        HF path: accept extra generation kwargs (max_new_tokens, do_sample, temperature, top_p, etc.)
        and pass them to model.generate() appropriately.
        """
        if _is_hf_id(getattr(self, "model_info", "")):
            tok, model = _hf_ensure(self.model_info)

            # build a simple chat template for LLaMA-style models
            sys_msg  = getattr(self, "instruction", "") or ""
            user_msg = message or ""
            prompt = _llama_template(sys_msg, user_msg)

            inputs = tok(prompt, return_tensors="pt")
            device = next(model.parameters()).device
            inputs = {k: v.to(device) for k,v in inputs.items()}

            # default generation args
            do_sample      = bool(gen_kw.pop("do_sample", False))
            max_new_tokens = int(gen_kw.pop("max_new_tokens", 128))
            temperature    = float(gen_kw.pop("temperature", 0.7))
            top_p          = float(gen_kw.pop("top_p", 0.95))

            # keep greedy for short “probe” turns
            if chat_mode is False:
                do_sample = False
                max_new_tokens = min(max_new_tokens, 16)

            # only pass temp/top_p when sampling (prevents “ignored flags” warnings)
            gen_args = dict(
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                pad_token_id=tok.eos_token_id,
            )
            if do_sample:
                gen_args.update(temperature=temperature, top_p=top_p)

            # allow any *other* recognized kwargs to flow through
            # (e.g., repetition_penalty, eos_token_id, etc.)
            gen_args.update(gen_kw)

            with torch.no_grad():
                out = model.generate(**inputs, **gen_args)

            text = tok.decode(out[0], skip_special_tokens=True)
            if "[/INST]" in text:
                text = text.split("[/INST]", 1)[-1].strip()

            # For NON-chatty probes (chat_mode=False), we only need a short first line
            return text.splitlines()[0].strip() if chat_mode is False else text

        # non-HF ids fall back to original behavior
        return _orig_chat(self, message, img_path, chat_mode)

    Agent.chat = _chat_hf

    # Monkeypatch Agent.temp_responses
    _orig_temp = Agent.temp_responses
    def _temp_hf(self, message, img_path=None, **kw):
        if _is_hf_id(getattr(self, "model_info", "")):
            tok, model = _hf_ensure(self.model_info)
            prompt = _llama_template(getattr(self, "instruction",""), message or "")
            inputs = tok(prompt, return_tensors="pt")
            device = next(model.parameters()).device
            inputs = {k: v.to(device) for k,v in inputs.items()}

            with torch.no_grad():
                out = model.generate(
                    **inputs,
                    max_new_tokens=int(kw.get("max_new_tokens", 128)),
                    do_sample=False,  # greedy for consistency
                    pad_token_id=tok.eos_token_id
                )
            text = tok.decode(out[0], skip_special_tokens=True)
            if "[/INST]" in text:
                text = text.split("[/INST]", 1)[-1].strip()
            return {0.0: text}
        return _orig_temp(self, message, img_path)

    Agent.temp_responses = _temp_hf
except Exception as _e:
    # Keep the OpenAI/Gemini path working even if HF isn't available
    pass
# ============================================================================

