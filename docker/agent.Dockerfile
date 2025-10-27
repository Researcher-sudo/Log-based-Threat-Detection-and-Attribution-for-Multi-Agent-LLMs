# # FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

# # RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
# #     python3 python3-pip git curl \
# #     tcpdump tshark strace procps sysstat \
# #     && apt-get clean && rm -rf /var/lib/apt/lists/*

# # # Optional: pin your Python deps here or use requirements.txt from the repo
# # RUN pip3 install --no-cache-dir \
# #     memory_profiler termcolor flask uvicorn \
# #     "transformers>=4.40" "accelerate>=0.30" "bitsandbytes>=0.43" "torch>=2.2" "openai==1.14.2" "tqdm==4.66.1" "requests==2.31.0" "prettytable==3.9.0" "termcolor==2.4.0" "pptree==3.1" "climage==0.2.0"

# # WORKDIR /app
# # docker/agent.Dockerfile
# # docker/agent.Dockerfile
# FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

# ARG DEBIAN_FRONTEND=noninteractive

# # Base OS deps (vmstat is in procps). tcpdump/tshark are for your collector.
# RUN apt-get update && apt-get install -y --no-install-recommends \
#     python3 python3-pip python3-venv git curl \
#     tcpdump tshark strace procps iproute2 ca-certificates \
#  && rm -rf /var/lib/apt/lists/*

# # Make "python" available (some scripts call `python`, others `python3`)
# RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 1

# WORKDIR /app
# # Bring the full repo (agents import utils/* etc.)
# COPY . /app

# # Python deps for agents
# # If you have a requirements.txt for agents, install it first (best practice).
# RUN if [ -f requirements.txt ]; then pip3 install --no-cache-dir -r requirements.txt; fi && \
#     pip3 install --no-cache-dir \
#       memory_profiler termcolor flask uvicorn pptree
# # NOTE: If you truly need Torch/CUDA/bitsandbytes, install the exact CUDA 12.1 wheels.
# # Example:
# #   pip3 install --no-cache-dir --index-url https://download.pytorch.org/whl/cu121 \
# #       torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2
# #   pip3 install --no-cache-dir bitsandbytes==0.43.1 transformers==4.41.0 accelerate==0.30.1
# # Only do this if your agents actually use them.

# # Collector script (does profiling, tcpdump, vmstat, writes /logs/*)
# COPY docker/collector.sh /usr/local/bin/collector.sh
# RUN sed -i 's/\r$//' /usr/local/bin/collector.sh && chmod +x /usr/local/bin/collector.sh

# ENV PYTHONUNBUFFERED=1 \
#     PIP_NO_CACHE_DIR=1

# # IMPORTANT:
# # We don't set CMD here. The collector expects AGENT_CMD from the environment
# # (provided by docker-compose per-service). It injects `-m cProfile` correctly.
# ENTRYPOINT ["/usr/local/bin/collector.sh"]


# docker/agent.Dockerfile (add/replace the relevant lines)
FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
      bash python3 python3-pip python3-venv git curl \
      tcpdump tshark strace procps iproute2 ca-certificates \
 && apt-get clean && rm -rf /var/lib/apt/lists/*

# docker/agent.Dockerfile  (only the pip line shown)
# docker/agent.Dockerfile  (snippet)
# Python deps for agents
RUN pip3 install --no-cache-dir sentencepiece tiktoken \
    memory_profiler termcolor flask uvicorn \
    "transformers>=4.40" "accelerate>=0.30" "bitsandbytes>=0.43" "torch>=2.2" \
    "openai==1.1.1" \
    pptree prettytable google-generativeai




WORKDIR /app
COPY . /app

COPY docker/collector.sh /usr/local/bin/collector.sh
RUN sed -i 's/\r$//' /usr/local/bin/collector.sh \
 && awk 'NR==1{sub(/^\xef\xbb\xbf/,"")}1' /usr/local/bin/collector.sh > /tmp/c.sh \
 && mv /tmp/c.sh /usr/local/bin/collector.sh \
 && chmod +x /usr/local/bin/collector.sh

ENV PYTHONUNBUFFERED=1

# run under bash explicitly
ENTRYPOINT ["/usr/bin/env","bash","/usr/local/bin/collector.sh"]

# --- PyTorch w/ CUDA 12.1 wheels ---
RUN pip3 install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cu121 \
    torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1
