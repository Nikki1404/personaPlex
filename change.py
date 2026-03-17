FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ARG USE_PROXY=false

ENV http_proxy=${USE_PROXY:+http://163.116.128.80:8080}
ENV https_proxy=${USE_PROXY:+http://163.116.128.80:8080}
ENV PYTHONUNBUFFERED=1

ENV HF_HOME=/srv/model_cache
ENV TRANSFORMERS_CACHE=/srv/model_cache
ENV TORCH_HOME=/srv/model_cache

WORKDIR /srv

# Install Python + system deps
COPY requirements.txt .

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3-pip \
    python3-dev \
    ffmpeg \
    git \
 && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip setuptools wheel \
 && pip install --no-cache-dir -r requirements.txt

# Download and prepare models
RUN python3 - <<EOF
import nemo.collections.asr as nemo_asr
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

print("Downloading Nemotron from HuggingFace")

model = nemo_asr.models.ASRModel.from_pretrained(
    "nvidia/nemotron-speech-streaming-en-0.6b"
)
model.save_to("/srv/nemotron-speech-streaming-en-0.6b.nemo")
print("Nemotron saved as .nemo")
print("Warm loading Nemotron (CPU)")

model = nemo_asr.models.ASRModel.restore_from(
    restore_path="/srv/nemotron-speech-streaming-en-0.6b.nemo",
    map_location="cpu"
)
print("Nemotron warm load complete")
print("Downloading Whisper Turbo")
AutoModelForSpeechSeq2Seq.from_pretrained(
    "openai/whisper-large-v3-turbo"
)
AutoProcessor.from_pretrained(
    "openai/whisper-large-v3-turbo"
)
print("Whisper downloaded and cached")

EOF

COPY app ./app
COPY app/google_credentials.json google_credentials.json

ENV GOOGLE_APPLICATION_CREDENTIALS=/srv/google_credentials.json
ENV GOOGLE_RECOGNIZER=projects/eci-ugi-digital-ccaipoc/locations/us-central1/recognizers/google-stt-default
ENV GOOGLE_REGION=us-central1
ENV GOOGLE_LANGUAGE=en-US
ENV GOOGLE_MODEL=latest_short
ENV GOOGLE_INTERIM=true
ENV GOOGLE_EXPLICIT_DECODING=true

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8002"]
