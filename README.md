This repository contains the code and notebooks used to (1) generate compromised prompts, (2) execute both benign and compromised workloads through a multi-agent LLM stack, (3) collect system/network/application logs for each run, (4) build per-prompt graphs from those logs, and (5) perform threat detection and attribution.

The pipeline below is exactly what was used to produce the reported results in the accompanying manuscript.

---

## 1. Generate the Compromised Dataset

We first construct an "attack-set" version of the QA data.

```bash
python build_medqa_attackset.py
```

This script programmatically takes benign medical-style questions and produces paired malicious / policy-violating variants.
Output: a dataset containing both **normal (benign)** and **compromised (adversarial)** prompts.

You should now have two sets:

* Normal dataset
* Compromised dataset

These sets are later executed against the multi-agent system.

---

## 2. Execute the Workloads Under Instrumentation

We execute both datasets (normal + compromised) through the multi-agent LLM framework (MDAgents) under logging.

```bash
bash run_with_logging.sh
```

`run_with_logging.sh` acts as a wrapper/orchestrator. It:

* Runs the normal dataset through MDAgents.
* Runs the compromised dataset through MDAgents.
* Collects runtime telemetry for each prompt/response pair.

Captured logs include (per prompt):

* Application-level interaction traces between agents.
* System-level resource/activity traces.
* Network-level traces.

This produces a structured log bundle for every prompt, for both benign and compromised cases.

---

## 3. Build Per-Prompt Graphs From Logs

We convert raw logs into a directed, attributed multi-agent interaction graph on a per-prompt basis.

Open and run:

* `Graph Creation.ipynb`

This notebook:

1. Parses the collected logs for each prompt.
2. Aligns application / system / network views.
3. Builds a per-prompt graph where:

   * Nodes = agents involved in handling the prompt.
   * Edges = directed communications / interactions.
   * Node and edge attributes = behavioral, resource, and communication statistics aggregated over the prompt’s lifetime.

Output: serialized graph objects plus feature matrices suitable for downstream classification.

---

## 4. Threat Detection and Attribution

Open and run:

* `Threat Detection and Attribution.ipynb`

This notebook performs two tasks:

### 4.1 Threat Detection

It trains / evaluates a graph-based classifier that labels each per-prompt graph as **benign** or **compromised** using only runtime metadata (no prompt text, no model output text).

It also supports early-detection experiments, where only a prefix of each trace is used.

### 4.2 Threat Attribution

It attributes the model’s decision back to specific elements of the graph:

* Which agents (nodes) and which communications (edges) contributed most to the “compromised” classification.
* Which runtime features (e.g., abnormal CPU usage patterns, unusual message fan-out, atypical communication rates) were decisive.

This produces both the binary compromise decision and an interpretable explanation.

---

## Environment / Requirements

The core environment for data generation and agent execution uses the following Python packages:

```text
transformers==4.43.3
accelerate==0.33.0
bitsandbytes==0.43.3
sentencepiece>=0.1.99
protobuf<5
uvicorn>=0.30.0
fastapi>=0.111.0
requests>=2.32.3
```

Notes:

* `bitsandbytes` is used for efficient inference / quantization.
* `uvicorn` + `fastapi` are included for serving / orchestration components.
* If you rely on GPU inference, make sure your CUDA/toolkit setup matches the versions required by `bitsandbytes`.

Any additional local utilities (e.g., dataset loaders, graph feature extraction helpers, etc.) should be installed or placed in-path before running the notebooks.
---

## Minimal Reproduction Flow (Summary)

1. **Build adversarial dataset**

   ```bash
   python build_medqa_attackset.py
   ```

2. **Run agent framework with logging on both benign + compromised sets**

   ```bash
   bash run_with_logging.sh
   ```

3. **Construct per-prompt graphs from the collected logs**

   * Run `Graph Creation.ipynb`

4. **Detect and attribute compromise**

   * Run `Threat Detection and Attribution.ipynb`

At the end of this flow you have:

* Labeled graphs for each prompt (benign vs. compromised)
* A trained detector that flags compromised activity from runtime behavior alone
* An attribution report that identifies which agents / links in the graph are responsible for the alert

---

## Repository Layout (reference)

* `build_medqa_attackset.py`
  Generates paired benign / compromised prompts.

* `run_with_logging.sh`
  Executes workloads in the multi-agent environment and captures logs.

* `Graph Creation.ipynb`
  Parses logs and builds per-prompt directed graphs with node/edge features.

* `Threat Detection and Attribution.ipynb`
  Classifies each graph (benign / compromised) and produces local explanations (which agent/edge caused the alert).

---

## Anonymity / Contact

This artifact is provided for review and reproducibility of the submitted manuscript.
For questions, please contact the authors through the submission system.
