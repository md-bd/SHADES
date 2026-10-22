FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install Python 3.8 and required Linux libraries.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        python3.8 \
        python3.8-dev \
        python3-pip \
        ca-certificates \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Make Python 3.8 available through python and python3.
RUN ln -sf /usr/bin/python3.8 /usr/local/bin/python \
    && ln -sf /usr/bin/python3.8 /usr/local/bin/python3

# Install packaging tools that still support Python 3.8.
RUN python -m pip install --upgrade \
        "pip<25.1" \
        "setuptools<76" \
        wheel

# Install CUDA-enabled PyTorch and Torchvision.
RUN python -m pip install \
        torch==2.2.1 \
        torchvision==0.17.1 \
        --index-url https://download.pytorch.org/whl/cu118

# Copy the requirements file separately for Docker layer caching.
COPY requirements.txt ./requirements.txt

# Exclude torch and torchvision because their CUDA-enabled versions
# were installed explicitly above.
RUN grep -vE '^(torch|torchvision)([=<>!~]|$)' requirements.txt \
        > requirements-docker.txt \
    && python -m pip install -r requirements-docker.txt

# Copy project source code and configuration files.
COPY src/ ./src/
COPY configs/ ./configs/  
# TODO config should be mounted

# Download the pretrained weights, verify their SHA-256 checksum,
# save them under models/, and install them in the PyTorch cache.
RUN python src/setup_weights.py

# Create directories intended for runtime bind mounts.
RUN mkdir -p /app/data /app/results

# Allows commands such as:
# docker run IMAGE src/training_script.py --config configs/config.yaml
ENTRYPOINT ["python"]
CMD ["--help"]
