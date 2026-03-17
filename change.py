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


when runnig docker run getting this 
(base) root@EC03-E01-AICOE1:/home/CORP/re_nikitav# docker logs 81521e3e9435

==========
== CUDA ==
==========

CUDA Version 12.4.1

Container image Copyright (c) 2016-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

This container image and its contents are governed by the NVIDIA Deep Learning Container License.
By pulling and using the container, you accept the terms and conditions of this license:
https://developer.nvidia.com/ngc/nvidia-deep-learning-container-license

A copy of this license is made available in this container at /NGC-DL-CONTAINER-LICENSE for your convenience.

/usr/local/lib/python3.10/dist-packages/google/api_core/_python_version_support.py:275: FutureWarning: You are using a Python version (3.10.12) which Google will stop supporting in new releases of google.api_core once it reaches its end of life (2026-10-04). Please upgrade to the latest Python version, or at least Python 3.11, to continue receiving updates for google.api_core past that date.
  warnings.warn(message, FutureWarning)
/usr/local/lib/python3.10/dist-packages/google/api_core/_python_version_support.py:275: FutureWarning: You are using a Python version (3.10.12) which Google will stop supporting in new releases of google.cloud.speech_v2 once it reaches its end of life (2026-10-04). Please upgrade to the latest Python version, or at least Python 3.11, to continue receiving updates for google.cloud.speech_v2 past that date.
  warnings.warn(message, FutureWarning)
DEBUG: Startup cfg.model_name='nemotron-speech-streaming-en-0.6b.nemo' cfg.asr_backend='nemotron'
INFO:     Started server process [1]
INFO:     Waiting for application startup.
2026-03-17 09:11:20,152 | INFO | asr_server | Preloading ASR engines...
2026-03-17 09:11:20,366 | INFO | httpx | HTTP Request: HEAD https://huggingface.co/openai/whisper-large-v3-turbo/resolve/main/processor_config.json "HTTP/1.1 404 Not Found"
2026-03-17 09:11:20,410 | INFO | httpx | HTTP Request: HEAD https://huggingface.co/openai/whisper-large-v3-turbo/resolve/main/preprocessor_config.json "HTTP/1.1 307 Temporary Redirect"
2026-03-17 09:11:20,424 | INFO | httpx | HTTP Request: HEAD https://huggingface.co/api/resolve-cache/models/openai/whisper-large-v3-turbo/41f01f3fe87f28c78e2fbf8b568835947dd65ed9/preprocessor_config.json "HTTP/1.1 200 OK"
2026-03-17 09:11:20,454 | INFO | httpx | HTTP Request: HEAD https://huggingface.co/openai/whisper-large-v3-turbo/resolve/main/processor_config.json "HTTP/1.1 404 Not Found"
2026-03-17 09:11:20,487 | INFO | httpx | HTTP Request: HEAD https://huggingface.co/openai/whisper-large-v3-turbo/resolve/main/preprocessor_config.json "HTTP/1.1 307 Temporary Redirect"
2026-03-17 09:11:20,501 | INFO | httpx | HTTP Request: HEAD https://huggingface.co/api/resolve-cache/models/openai/whisper-large-v3-turbo/41f01f3fe87f28c78e2fbf8b568835947dd65ed9/preprocessor_config.json "HTTP/1.1 200 OK"
2026-03-17 09:11:20,536 | INFO | httpx | HTTP Request: GET https://huggingface.co/api/models/openai/whisper-large-v3-turbo/tree/main/additional_chat_templates?recursive=false&expand=false "HTTP/1.1 404 Not Found"
2026-03-17 09:11:20,571 | INFO | httpx | HTTP Request: HEAD https://huggingface.co/openai/whisper-large-v3-turbo/resolve/main/processor_config.json "HTTP/1.1 404 Not Found"
2026-03-17 09:11:20,571 | WARNING | huggingface_hub.utils._http | Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
2026-03-17 09:11:20,608 | INFO | httpx | HTTP Request: HEAD https://huggingface.co/openai/whisper-large-v3-turbo/resolve/main/chat_template.json "HTTP/1.1 404 Not Found"
2026-03-17 09:11:20,640 | INFO | httpx | HTTP Request: HEAD https://huggingface.co/openai/whisper-large-v3-turbo/resolve/main/chat_template.jinja "HTTP/1.1 404 Not Found"
2026-03-17 09:11:20,673 | INFO | httpx | HTTP Request: HEAD https://huggingface.co/openai/whisper-large-v3-turbo/resolve/main/audio_tokenizer_config.json "HTTP/1.1 404 Not Found"
2026-03-17 09:11:20,750 | INFO | httpx | HTTP Request: HEAD https://huggingface.co/openai/whisper-large-v3-turbo/resolve/main/processor_config.json "HTTP/1.1 404 Not Found"
2026-03-17 09:11:20,778 | INFO | httpx | HTTP Request: HEAD https://huggingface.co/openai/whisper-large-v3-turbo/resolve/main/preprocessor_config.json "HTTP/1.1 307 Temporary Redirect"
2026-03-17 09:11:20,793 | INFO | httpx | HTTP Request: HEAD https://huggingface.co/api/resolve-cache/models/openai/whisper-large-v3-turbo/41f01f3fe87f28c78e2fbf8b568835947dd65ed9/preprocessor_config.json "HTTP/1.1 200 OK"
2026-03-17 09:11:20,834 | INFO | httpx | HTTP Request: HEAD https://huggingface.co/openai/whisper-large-v3-turbo/resolve/main/processor_config.json "HTTP/1.1 404 Not Found"
2026-03-17 09:11:20,863 | INFO | httpx | HTTP Request: HEAD https://huggingface.co/openai/whisper-large-v3-turbo/resolve/main/preprocessor_config.json "HTTP/1.1 307 Temporary Redirect"
2026-03-17 09:11:20,877 | INFO | httpx | HTTP Request: HEAD https://huggingface.co/api/resolve-cache/models/openai/whisper-large-v3-turbo/41f01f3fe87f28c78e2fbf8b568835947dd65ed9/preprocessor_config.json "HTTP/1.1 200 OK"
2026-03-17 09:11:20,908 | INFO | httpx | HTTP Request: HEAD https://huggingface.co/openai/whisper-large-v3-turbo/resolve/main/config.json "HTTP/1.1 307 Temporary Redirect"
2026-03-17 09:11:20,921 | INFO | httpx | HTTP Request: HEAD https://huggingface.co/api/resolve-cache/models/openai/whisper-large-v3-turbo/41f01f3fe87f28c78e2fbf8b568835947dd65ed9/config.json "HTTP/1.1 200 OK"
2026-03-17 09:11:20,952 | INFO | httpx | HTTP Request: HEAD https://huggingface.co/openai/whisper-large-v3-turbo/resolve/main/tokenizer_config.json "HTTP/1.1 307 Temporary Redirect"
2026-03-17 09:11:20,967 | INFO | httpx | HTTP Request: HEAD https://huggingface.co/api/resolve-cache/models/openai/whisper-large-v3-turbo/41f01f3fe87f28c78e2fbf8b568835947dd65ed9/tokenizer_config.json "HTTP/1.1 200 OK"
2026-03-17 09:11:21,004 | INFO | httpx | HTTP Request: GET https://huggingface.co/api/models/openai/whisper-large-v3-turbo/tree/main/additional_chat_templates?recursive=false&expand=false "HTTP/1.1 404 Not Found"
2026-03-17 09:11:21,038 | INFO | httpx | HTTP Request: GET https://huggingface.co/api/models/openai/whisper-large-v3-turbo/tree/main?recursive=true&expand=false "HTTP/1.1 200 OK"
2026-03-17 09:11:21,553 | INFO | httpx | HTTP Request: HEAD https://huggingface.co/openai/whisper-large-v3-turbo/resolve/main/config.json "HTTP/1.1 307 Temporary Redirect"
2026-03-17 09:11:21,568 | INFO | httpx | HTTP Request: HEAD https://huggingface.co/api/resolve-cache/models/openai/whisper-large-v3-turbo/41f01f3fe87f28c78e2fbf8b568835947dd65ed9/config.json "HTTP/1.1 200 OK"
`torch_dtype` is deprecated! Use `dtype` instead!
Loading weights: 100%|██████████| 587/587 [00:00<00:00, 3312.03it/s]
2026-03-17 09:11:21,976 | INFO | httpx | HTTP Request: HEAD https://huggingface.co/openai/whisper-large-v3-turbo/resolve/main/generation_config.json "HTTP/1.1 307 Temporary Redirect"
2026-03-17 09:11:21,992 | INFO | httpx | HTTP Request: HEAD https://huggingface.co/api/resolve-cache/models/openai/whisper-large-v3-turbo/41f01f3fe87f28c78e2fbf8b568835947dd65ed9/generation_config.json "HTTP/1.1 200 OK"
Using custom `forced_decoder_ids` from the (generation) config. This is deprecated in favor of the `task` and `language` flags/config options.
The attention mask is not set and cannot be inferred from input because pad token is same as eos token. As a consequence, you may observe unexpected behavior. Please pass your input's `attention_mask` to obtain reliable results.
Both `max_new_tokens` (=16) and `max_length`(=448) seem to have been set. `max_new_tokens` will take precedence. Please refer to the documentation for more information. (https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)
A custom logits processor of type <class 'transformers.generation.logits_process.SuppressTokensLogitsProcessor'> has been passed to `.generate()`, but it was also created in `.generate()`, given its parameterization. The custom <class 'transformers.generation.logits_process.SuppressTokensLogitsProcessor'> will take precedence. Please check the docstring of <class 'transformers.generation.logits_process.SuppressTokensLogitsProcessor'> to see related `.generate()` flags.
A custom logits processor of type <class 'transformers.generation.logits_process.SuppressTokensAtBeginLogitsProcessor'> has been passed to `.generate()`, but it was also created in `.generate()`, given its parameterization. The custom <class 'transformers.generation.logits_process.SuppressTokensAtBeginLogitsProcessor'> will take precedence. Please check the docstring of <class 'transformers.generation.logits_process.SuppressTokensAtBeginLogitsProcessor'> to see related `.generate()` flags.
2026-03-17 09:11:23,243 | INFO | asr_server | Preloaded whisper in 3.09s
[NeMo I 2026-03-17 09:11:32 mixins:176] Tokenizer SentencePieceTokenizer initialized with 1024 tokens
[NeMo W 2026-03-17 09:11:32 modelPT:176] If you intend to do training or fine-tuning, please call the ModelPT.setup_training_data() method and provide a valid configuration file to setup the train data loader.

