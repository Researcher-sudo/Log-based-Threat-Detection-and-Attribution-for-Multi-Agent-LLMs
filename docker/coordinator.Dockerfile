# FROM python:3.10-slim

# ARG DEBIAN_FRONTEND=noninteractive
# RUN apt-get update && apt-get install -y --no-install-recommends \
#       tcpdump tshark strace procps iproute2 ca-certificates \
#     && rm -rf /var/lib/apt/lists/*

# WORKDIR /app
# COPY . /app
# RUN pip install --no-cache-dir -r requirements.txt || true \
#  && pip install --no-cache-dir memory_profiler termcolor

# COPY docker/collector.sh /usr/local/bin/collector.sh
# RUN sed -i 's/\r$//' /usr/local/bin/collector.sh && chmod +x /usr/local/bin/collector.sh

# ENV PYTHONUNBUFFERED=1
# ENTRYPOINT ["/usr/local/bin/collector.sh"]


# docker/coordinator.Dockerfile (add/replace the relevant lines)
FROM python:3.10-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
      bash tcpdump tshark strace procps iproute2 ca-certificates \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . /app

RUN pip install --no-cache-dir -r requirements.txt || true \
 && pip install --no-cache-dir memory_profiler termcolor

# put collector in place, strip CRLF and UTF-8 BOM, make executable
COPY docker/collector.sh /usr/local/bin/collector.sh
RUN sed -i 's/\r$//' /usr/local/bin/collector.sh \
 && awk 'NR==1{sub(/^\xef\xbb\xbf/,"")}1' /usr/local/bin/collector.sh > /tmp/c.sh \
 && mv /tmp/c.sh /usr/local/bin/collector.sh \
 && chmod +x /usr/local/bin/collector.sh

ENV PYTHONUNBUFFERED=1

# call it via bash explicitly to bypass any shebang weirdness
ENTRYPOINT ["/usr/bin/env","bash","/usr/local/bin/collector.sh"]
