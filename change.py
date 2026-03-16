#Dockerfile-
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ARG USE_PROXY=false

ENV http_proxy=${USE_PROXY:+http://163.116.128.80:8080}
ENV https_proxy=${USE_PROXY:+http://163.116.128.80:8080}
ENV PYTHONUNBUFFERED=1

WORKDIR /srv
COPY requirements.txt .
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3-pip \
    python3-dev \
 && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip setuptools wheel \
 && pip install --no-cache-dir -r requirements.txt

# Download + preload models
RUN python3 - <<EOF
import nemo.collections.asr as nemo_asr
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

print("Downloading Nemotron model...")
nemo_asr.models.ASRModel.from_pretrained(
    "nvidia/nemotron-speech-streaming-en-0.6b"
)

print("Downloading Whisper model...")
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
