FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

RUN apt-get update && apt-get install -y \
    git python3 python3-pip libopus-dev \
 && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade pip

# Install Torch CUDA wheels (adjust if your CUDA differs)
RUN pip install --no-cache-dir \
    torch==2.2.1 \
    torchaudio==2.2.1 \
    --index-url https://download.pytorch.org/whl/cu121

# Install our runtime deps (client libs included; harmless on server)
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Install NVIDIA PersonaPlex Moshi from source (official pattern)
RUN git clone --depth 1 https://github.com/NVIDIA/personaplex.git /opt/personaplex
RUN pip install --no-cache-dir /opt/personaplex/moshi

# Copy project
COPY . /app

EXPOSE 8000

CMD ["python3", "main.py"]
