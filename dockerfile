# Base — CUDA runtime for GPU support on RunPod and similar platforms
FROM nvidia/cuda:12.2.2-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-venv python3-dev python3-pip python-is-python3 \
    build-essential git wget curl unzip ca-certificates nano less tmux ffmpeg \
    libsndfile1 \
 && rm -rf /var/lib/apt/lists/* \
 && mkdir -p /data

# Clone the trainer repo so users can `git pull` to get updates
# without rebuilding the Docker image.
RUN git clone https://github.com/BigPappy098/microWakeWord-Trainer-Nvidia-Docker.git /root/mww-scripts \
 && chmod -R a+x /root/mww-scripts/cli \
 && chmod +x /root/mww-scripts/train_wake_word \
              /root/mww-scripts/setup \
              /root/mww-scripts/entrypoint.sh \
              /root/mww-scripts/github_push.sh \
 && ln -sf /root/mww-scripts/.bashrc /root/.bashrc

WORKDIR /root/mww-scripts

# No args = interactive bash shell; "train <wake_word>" = full pipeline
ENTRYPOINT ["/root/mww-scripts/entrypoint.sh"]
CMD ["/bin/bash", "-l"]
