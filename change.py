#Dockerfile-
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ARG USE_PROXY=false

ENV http_proxy=${USE_PROXY:+http://163.116.128.80:8080}
ENV https_proxy=${USE_PROXY:+http://163.116.128.80:8080}
ENV PYTHONUNBUFFERED=1

# cache dirs (cleaner)
ENV HF_HOME=/srv/hf_cache
ENV TRANSFORMERS_CACHE=/srv/hf_cache
ENV TORCH_HOME=/srv/torch_cache

WORKDIR /srv

COPY requirements.txt .

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3-pip \
    python3-dev \
    ffmpeg \
 && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip setuptools wheel \
 && pip install --no-cache-dir -r requirements.txt

# Pre-download models
RUN python3 - <<EOF
import nemo.collections.asr as nemo_asr
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

print("Downloading Nemotron...")
nemo_asr.models.ASRModel.from_pretrained(
    "nvidia/nemotron-speech-streaming-en-0.6b"
)

print("Downloading Whisper...")
AutoModelForSpeechSeq2Seq.from_pretrained(
    "openai/whisper-large-v3-turbo"
)

AutoProcessor.from_pretrained(
    "openai/whisper-large-v3-turbo"
)

print("Model download complete.")
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

#main.py-
import json
import logging
import sys
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware

from app.config import load_config, Config, MODEL_MAP
from app.factory import build_engine
from app.streaming_session import StreamingSession
from app.asr_engines.base import ASREngine


cfg = load_config()

logging.basicConfig(
    level=cfg.log_level,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

log = logging.getLogger("asr_server")

app = FastAPI()

ENGINE_CACHE: dict[str, ASREngine] = {}


async def preload_engines():

    backends = ["whisper", "nemotron", "google"]

    log.info("Preloading ASR engines...")

    for backend in backends:

        try:

            model_name = MODEL_MAP[backend]

            tmp_cfg = Config()
            object.__setattr__(tmp_cfg, "asr_backend", backend)
            object.__setattr__(tmp_cfg, "model_name", model_name)
            object.__setattr__(tmp_cfg, "device", cfg.device)
            object.__setattr__(tmp_cfg, "sample_rate", cfg.sample_rate)
            object.__setattr__(tmp_cfg, "context_right", cfg.context_right)

            engine = build_engine(tmp_cfg)

            load_sec = engine.load()

            ENGINE_CACHE[backend] = engine

            log.info(f"Preloaded {backend} in {load_sec:.2f}s")

        except Exception as e:

            log.error(f"Failed to preload {backend}: {e}")

    log.info("All engines preloaded.")


@app.on_event("startup")
async def startup_event():

    await preload_engines()


def get_engine(backend: str) -> ASREngine:

    if backend not in ENGINE_CACHE:
        raise ValueError(f"Engine '{backend}' not preloaded")

    return ENGINE_CACHE[backend]


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.websocket("/asr/realtime-custom-vad")
async def ws_asr(ws: WebSocket):

    log.info(f"WebSocket connection request received from {ws.client}")

    await ws.accept()

    init = json.loads(await ws.receive_text())

    backend = init.get("backend")

    if backend not in ("nemotron", "whisper", "google"):

        log.warning("Invalid backend requested")

        await ws.close(code=4000)
        return

    engine = get_engine(backend)

    log.info(f"WebSocket connected backend={backend} client={ws.client}")

    session = StreamingSession(engine, cfg)

    try:

        while True:

            msg = await ws.receive()

            if msg["type"] == "websocket.disconnect":
                break

            data = msg.get("bytes")

            if data is None:
                continue

            events = session.process_chunk(data)

            for ev_type, text, ttfb in events:

                await ws.send_text(json.dumps({
                    "type": ev_type,
                    "text": text,
                    "t_start": ttfb
                }))

    finally:

        log.info(f"WebSocket disconnected client={ws.client}")

        await ws.close()


#streaming_seesion.py-
import time
from app.vad import AdaptiveEnergyVAD


class StreamingSession:

    def __init__(self, engine, cfg):

        self.engine = engine
        self.cfg = cfg

        self.vad = AdaptiveEnergyVAD(
            cfg.sample_rate,
            cfg.vad_frame_ms,
            cfg.vad_start_margin,
            cfg.vad_min_noise_rms,
            cfg.pre_speech_ms,
        )

        self.session = engine.new_session(max_buffer_ms=cfg.max_utt_ms)

        self.frame_bytes = int(cfg.sample_rate * cfg.vad_frame_ms / 1000) * 2
        self.raw_buf = bytearray()

        self.utt_started = False
        self.utt_audio_ms = 0
        self.t_utt_start = None
        self.t_first_partial = None
        self.silence_ms = 0


    def process_chunk(self, pcm):

        events = []

        self.raw_buf.extend(pcm)

        while len(self.raw_buf) >= self.frame_bytes:

            frame = bytes(self.raw_buf[:self.frame_bytes])
            del self.raw_buf[:self.frame_bytes]

            is_speech, pre = self.vad.push_frame(frame)

            self.silence_ms = 0 if is_speech else self.silence_ms + self.cfg.vad_frame_ms

            if pre and not self.utt_started:

                self.utt_started = True
                self.utt_audio_ms = 0
                self.t_utt_start = time.time()
                self.t_first_partial = None

                self.session.accept_pcm16(pre)

            if not self.utt_started:
                continue

            self.session.accept_pcm16(frame)
            self.utt_audio_ms += self.cfg.vad_frame_ms

            if self.engine.caps.partials:

                text = self.session.step_if_ready()

                if text:

                    if self.t_first_partial is None:
                        self.t_first_partial = time.time()

                    ttfb_ms = int((self.t_first_partial - self.t_utt_start) * 1000)

                    events.append(("partial", text, ttfb_ms))

            if (
                not is_speech
                and self.utt_audio_ms >= self.cfg.min_utt_ms
                and self.silence_ms >= self.cfg.end_silence_ms
            ):

                final = self.session.finalize(self.cfg.post_speech_pad_ms)

                if final:

                    ttfb_ms = (
                        int((self.t_first_partial - self.t_utt_start) * 1000)
                        if self.t_first_partial
                        else None
                    )

                    events.append(("final", final, ttfb_ms))

                self.reset()

        return events


    def reset(self):

        self.vad.reset()
        self.utt_started = False
        self.utt_audio_ms = 0
        self.silence_ms = 0

when using this dockerfile 
getting this while docker run 
(base) root@EC03-E01-AICOE1:/home/CORP/re_nikitav/asr-realtime-custom-vad-updated# docker logs 7dc78473419d

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
2026-03-16 17:31:47,665 | INFO | asr_server | Preloading ASR engines...
2026-03-16 17:31:47,879 | INFO | httpx | HTTP Request: HEAD https://huggingface.co/openai/whisper-large-v3-turbo/resolve/main/processor_config.json "HTTP/1.1 404 Not Found"
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
2026-03-16 17:31:47,879 | WARNING | huggingface_hub.utils._http | Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
2026-03-16 17:31:47,926 | INFO | httpx | HTTP Request: HEAD https://huggingface.co/openai/whisper-large-v3-turbo/resolve/main/preprocessor_config.json "HTTP/1.1 307 Temporary Redirect"
2026-03-16 17:31:47,941 | INFO | httpx | HTTP Request: HEAD https://huggingface.co/api/resolve-cache/models/openai/whisper-large-v3-turbo/41f01f3fe87f28c78e2fbf8b568835947dd65ed9/preprocessor_config.json "HTTP/1.1 200 OK"
2026-03-16 17:31:47,970 | INFO | httpx | HTTP Request: HEAD https://huggingface.co/openai/whisper-large-v3-turbo/resolve/main/processor_config.json "HTTP/1.1 404 Not Found"
2026-03-16 17:31:47,997 | INFO | httpx | HTTP Request: HEAD https://huggingface.co/openai/whisper-large-v3-turbo/resolve/main/preprocessor_config.json "HTTP/1.1 307 Temporary Redirect"
2026-03-16 17:31:48,012 | INFO | httpx | HTTP Request: HEAD https://huggingface.co/api/resolve-cache/models/openai/whisper-large-v3-turbo/41f01f3fe87f28c78e2fbf8b568835947dd65ed9/preprocessor_config.json "HTTP/1.1 200 OK"
2026-03-16 17:31:48,048 | INFO | httpx | HTTP Request: GET https://huggingface.co/api/models/openai/whisper-large-v3-turbo/tree/main/additional_chat_templates?recursive=false&expand=false "HTTP/1.1 404 Not Found"
2026-03-16 17:31:48,086 | INFO | httpx | HTTP Request: HEAD https://huggingface.co/openai/whisper-large-v3-turbo/resolve/main/processor_config.json "HTTP/1.1 404 Not Found"
2026-03-16 17:31:48,115 | INFO | httpx | HTTP Request: HEAD https://huggingface.co/openai/whisper-large-v3-turbo/resolve/main/chat_template.json "HTTP/1.1 404 Not Found"
2026-03-16 17:31:48,156 | INFO | httpx | HTTP Request: HEAD https://huggingface.co/openai/whisper-large-v3-turbo/resolve/main/chat_template.jinja "HTTP/1.1 404 Not Found"
2026-03-16 17:31:48,188 | INFO | httpx | HTTP Request: HEAD https://huggingface.co/openai/whisper-large-v3-turbo/resolve/main/audio_tokenizer_config.json "HTTP/1.1 404 Not Found"
2026-03-16 17:31:48,221 | INFO | httpx | HTTP Request: HEAD https://huggingface.co/openai/whisper-large-v3-turbo/resolve/main/processor_config.json "HTTP/1.1 404 Not Found"
2026-03-16 17:31:48,253 | INFO | httpx | HTTP Request: HEAD https://huggingface.co/openai/whisper-large-v3-turbo/resolve/main/preprocessor_config.json "HTTP/1.1 307 Temporary Redirect"
2026-03-16 17:31:48,266 | INFO | httpx | HTTP Request: HEAD https://huggingface.co/api/resolve-cache/models/openai/whisper-large-v3-turbo/41f01f3fe87f28c78e2fbf8b568835947dd65ed9/preprocessor_config.json "HTTP/1.1 200 OK"
2026-03-16 17:31:48,310 | INFO | httpx | HTTP Request: HEAD https://huggingface.co/openai/whisper-large-v3-turbo/resolve/main/processor_config.json "HTTP/1.1 404 Not Found"
2026-03-16 17:31:48,337 | INFO | httpx | HTTP Request: HEAD https://huggingface.co/openai/whisper-large-v3-turbo/resolve/main/preprocessor_config.json "HTTP/1.1 307 Temporary Redirect"
2026-03-16 17:31:48,350 | INFO | httpx | HTTP Request: HEAD https://huggingface.co/api/resolve-cache/models/openai/whisper-large-v3-turbo/41f01f3fe87f28c78e2fbf8b568835947dd65ed9/preprocessor_config.json "HTTP/1.1 200 OK"
2026-03-16 17:31:48,395 | INFO | httpx | HTTP Request: HEAD https://huggingface.co/openai/whisper-large-v3-turbo/resolve/main/config.json "HTTP/1.1 307 Temporary Redirect"
2026-03-16 17:31:48,408 | INFO | httpx | HTTP Request: HEAD https://huggingface.co/api/resolve-cache/models/openai/whisper-large-v3-turbo/41f01f3fe87f28c78e2fbf8b568835947dd65ed9/config.json "HTTP/1.1 200 OK"
2026-03-16 17:31:48,446 | INFO | httpx | HTTP Request: HEAD https://huggingface.co/openai/whisper-large-v3-turbo/resolve/main/tokenizer_config.json "HTTP/1.1 307 Temporary Redirect"
2026-03-16 17:31:48,459 | INFO | httpx | HTTP Request: HEAD https://huggingface.co/api/resolve-cache/models/openai/whisper-large-v3-turbo/41f01f3fe87f28c78e2fbf8b568835947dd65ed9/tokenizer_config.json "HTTP/1.1 200 OK"
2026-03-16 17:31:48,509 | INFO | httpx | HTTP Request: GET https://huggingface.co/api/models/openai/whisper-large-v3-turbo/tree/main/additional_chat_templates?recursive=false&expand=false "HTTP/1.1 404 Not Found"
2026-03-16 17:31:48,541 | INFO | httpx | HTTP Request: GET https://huggingface.co/api/models/openai/whisper-large-v3-turbo/tree/main?recursive=true&expand=false "HTTP/1.1 200 OK"
2026-03-16 17:31:49,084 | INFO | httpx | HTTP Request: HEAD https://huggingface.co/openai/whisper-large-v3-turbo/resolve/main/config.json "HTTP/1.1 307 Temporary Redirect"
2026-03-16 17:31:49,097 | INFO | httpx | HTTP Request: HEAD https://huggingface.co/api/resolve-cache/models/openai/whisper-large-v3-turbo/41f01f3fe87f28c78e2fbf8b568835947dd65ed9/config.json "HTTP/1.1 200 OK"
`torch_dtype` is deprecated! Use `dtype` instead!
Loading weights: 100%|██████████| 587/587 [00:00<00:00, 1119.29it/s]
2026-03-16 17:31:49,859 | INFO | httpx | HTTP Request: HEAD https://huggingface.co/openai/whisper-large-v3-turbo/resolve/main/generation_config.json "HTTP/1.1 307 Temporary Redirect"
2026-03-16 17:31:49,873 | INFO | httpx | HTTP Request: HEAD https://huggingface.co/api/resolve-cache/models/openai/whisper-large-v3-turbo/41f01f3fe87f28c78e2fbf8b568835947dd65ed9/generation_config.json "HTTP/1.1 200 OK"
Using custom `forced_decoder_ids` from the (generation) config. This is deprecated in favor of the `task` and `language` flags/config options.
The attention mask is not set and cannot be inferred from input because pad token is same as eos token. As a consequence, you may observe unexpected behavior. Please pass your input's `attention_mask` to obtain reliable results.
Both `max_new_tokens` (=16) and `max_length`(=448) seem to have been set. `max_new_tokens` will take precedence. Please refer to the documentation for more information. (https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)
A custom logits processor of type <class 'transformers.generation.logits_process.SuppressTokensLogitsProcessor'> has been passed to `.generate()`, but it was also created in `.generate()`, given its parameterization. The custom <class 'transformers.generation.logits_process.SuppressTokensLogitsProcessor'> will take precedence. Please check the docstring of <class 'transformers.generation.logits_process.SuppressTokensLogitsProcessor'> to see related `.generate()` flags.
A custom logits processor of type <class 'transformers.generation.logits_process.SuppressTokensAtBeginLogitsProcessor'> has been passed to `.generate()`, but it was also created in `.generate()`, given its parameterization. The custom <class 'transformers.generation.logits_process.SuppressTokensAtBeginLogitsProcessor'> will take precedence. Please check the docstring of <class 'transformers.generation.logits_process.SuppressTokensAtBeginLogitsProcessor'> to see related `.generate()` flags.
2026-03-16 17:32:01,604 | INFO | asr_server | Preloaded whisper in 13.94s
[NeMo W 2026-03-16 17:32:03 nemo_logging:361] /usr/local/lib/python3.10/dist-packages/pydub/utils.py:170: RuntimeWarning: Couldn't find ffmpeg or avconv - defaulting to ffmpeg, but may not work
      warn("Couldn't find ffmpeg or avconv - defaulting to ffmpeg, but may not work", RuntimeWarning)

2026-03-16 17:32:04,042 | ERROR | asr_server | Failed to preload nemotron: Can't find /srv/nemotron-speech-streaming-en-0.6b.nemo
2026-03-16 17:32:04,099 | INFO | asr_server | Preloaded google in 0.06s
2026-03-16 17:32:04,099 | INFO | asr_server | All engines preloaded.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8002 (Press CTRL+C to quit)


