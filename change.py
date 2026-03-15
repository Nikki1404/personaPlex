app/config.py-
from dataclasses import dataclass, replace
import os

@dataclass(frozen=True)
class BackendParams:
    end_silence_ms: int
    short_pause_flush_ms: int
    min_utt_ms: int
    finalize_pad_ms: int

@dataclass(frozen=True)
class Config:
    asr_backend: str = os.getenv("ASR_BACKEND", "nemotron")
    model_name: str = os.getenv("MODEL_NAME", "")
    device: str = os.getenv("DEVICE", "cuda")

    sample_rate: int = int(os.getenv("SAMPLE_RATE", "16000"))
    context_right: int = int(os.getenv("CONTEXT_RIGHT", "1"))

    vad_frame_ms: int = int(os.getenv("VAD_FRAME_MS", "20"))
    vad_start_margin: float = float(os.getenv("VAD_START_MARGIN", "2.5"))
    vad_min_noise_rms: float = float(os.getenv("VAD_MIN_NOISE_RMS", "0.003"))
    pre_speech_ms: int = int(os.getenv("PRE_SPEECH_MS", "300"))

    min_utt_ms: int = 250
    end_silence_ms: int = 900
    max_utt_ms: int = int(os.getenv("MAX_UTT_MS", "30000"))
    post_speech_pad_ms: int = int(os.getenv("FINALIZE_PAD_MS", "400"))

    google_recognizer: str = os.getenv(
        "GOOGLE_RECOGNIZER",
        "projects/eci-ugi-digital-ccaipoc/locations/us-central1/recognizers/google-stt-default"
    )

    google_region: str = os.getenv("GOOGLE_REGION", "us-central1")
    google_language: str = os.getenv("GOOGLE_LANGUAGE", "en-US")
    google_model: str = os.getenv("GOOGLE_MODEL", "latest_short")

    google_interim: bool = os.getenv("GOOGLE_INTERIM", "true").lower() == "true"
    google_explicit_decoding: bool = os.getenv(
        "GOOGLE_EXPLICIT_DECODING", "true"
    ).lower() == "true"

    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    backend_params: BackendParams | None = None

MODEL_MAP = {
    "whisper": "openai/whisper-large-v3-turbo",
    "nemotron": "nemotron-speech-streaming-en-0.6b.nemo",
    "google": "google-stt-v2-streaming",
}

def load_config() -> Config:
    cfg = Config()

    if not cfg.model_name:
        cfg = replace(cfg, model_name=MODEL_MAP["nemotron"])
    backend_params = BackendParams(
        end_silence_ms=int(os.getenv("NEMO_END_SILENCE_MS", "900")),
        short_pause_flush_ms=0,
        min_utt_ms=int(os.getenv("NEMO_MIN_UTT_MS", "250")),
        finalize_pad_ms=cfg.post_speech_pad_ms,
    )
    if cfg.asr_backend == "whisper":
        backend_params = BackendParams(
            end_silence_ms=int(os.getenv("WHISPER_END_SILENCE_MS", "900")),
            short_pause_flush_ms=int(os.getenv("WHISPER_SHORT_PAUSE_FLUSH_MS", "350")),
            min_utt_ms=int(os.getenv("WHISPER_MIN_UTT_MS", "250")),
            finalize_pad_ms=cfg.post_speech_pad_ms,
        )

    if cfg.asr_backend == "google":
        backend_params = BackendParams(
            end_silence_ms=int(os.getenv("GOOGLE_END_SILENCE_MS", "700")),
            short_pause_flush_ms=0,
            min_utt_ms=int(os.getenv("GOOGLE_MIN_UTT_MS", "200")),
            finalize_pad_ms=int(os.getenv("GOOGLE_FINALIZE_PAD_MS", str(cfg.post_speech_pad_ms))),
        )

        cfg = replace(cfg, model_name=MODEL_MAP["google"])

    cfg = replace(cfg, backend_params=backend_params)
    cfg = replace(
        cfg,
        end_silence_ms=backend_params.end_silence_ms,
        min_utt_ms=backend_params.min_utt_ms,
        post_speech_pad_ms=backend_params.finalize_pad_ms,
    )

    print(
        f"DEBUG: Startup cfg.model_name='{cfg.model_name}' "
        f"cfg.asr_backend='{cfg.asr_backend}'"
    )

    return cfg


app/endpointing.py-
class Endpointing:
    def __init__(self, end_silence_ms: int, min_utt_ms: int, max_utt_ms: int):
        self.end_silence_ms = end_silence_ms
        self.min_utt_ms = min_utt_ms
        self.max_utt_ms = max_utt_ms
        self.reset()

    def reset(self):
        self.silence_ms = 0

    def update(self, is_speech: bool, frame_ms: int, utt_audio_ms: int) -> bool:
        if is_speech:
            self.silence_ms = 0
        else:
            self.silence_ms += frame_ms

        if utt_audio_ms >= self.max_utt_ms:
            return True

        if utt_audio_ms >= self.min_utt_ms and self.silence_ms >= self.end_silence_ms:
            return True

        return False

app/factory.py-
from app.config import Config
from app.asr_engines.nemotron_asr import NemotronStreamingASR
from app.asr_engines.whisper_asr import WhisperTurboASR
from app.asr_engines.google_streaming_asr import GoogleStreamingASR


def build_engine(cfg: Config):

    if cfg.asr_backend == "nemotron":
        return NemotronStreamingASR(
            model_name=cfg.model_name,
            device=cfg.device,
            sample_rate=cfg.sample_rate,
            context_right=cfg.context_right,
        )

    if cfg.asr_backend == "whisper":
        return WhisperTurboASR(
            model_name=cfg.model_name,
            device=cfg.device,
            sample_rate=cfg.sample_rate,
        )
    if cfg.asr_backend == "google":
        # Google engine reads everything from environment variables
        return GoogleStreamingASR(
            sample_rate=cfg.sample_rate,
        )

    raise ValueError(f"Unsupported ASR_BACKEND={cfg.asr_backend}")

app/main.py-
#app/main.py-
import asyncio
import json
import time
import logging
import os
import numpy as np
import resampy

from fastapi import FastAPI, WebSocket

from app.config import load_config, Config, MODEL_MAP
from app.vad import AdaptiveEnergyVAD
from app.factory import build_engine
from app.asr_engines.base import ASREngine

cfg = load_config()
logging.basicConfig(level=cfg.log_level, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger("asr_server")

app = FastAPI()
ENGINE_CACHE: dict[str, ASREngine] = {}


async def preload_engines():
    backends = ["whisper", "nemotron", "google"]
    log.info("Preloading ASR engines...")

    for backend in backends:
        try:
            model_name = MODEL_MAP[backend]
            print(f"Loading {backend} ({model_name})...")

            tmp_cfg = Config()
            object.__setattr__(tmp_cfg, 'asr_backend', backend)
            object.__setattr__(tmp_cfg, 'model_name', model_name)
            object.__setattr__(tmp_cfg, 'device', cfg.device)
            object.__setattr__(tmp_cfg, 'sample_rate', cfg.sample_rate)
            object.__setattr__(tmp_cfg, 'context_right', cfg.context_right)

            engine = build_engine(tmp_cfg)
            load_sec = engine.load()

            log.info(f"Preloaded {backend} in {load_sec:.2f}s")
            ENGINE_CACHE[backend] = engine

        except Exception as e:
            log.error(f"Failed to preload {backend}: {e}")
            continue

    log.info("All engines preloaded.")


@app.on_event("startup")
async def startup_event():
    await preload_engines()


def get_engine(backend: str) -> ASREngine:
    if backend not in ENGINE_CACHE:
        raise ValueError(f"Engine '{backend}' not preloaded. Available: {list(ENGINE_CACHE.keys())}")
    log.info(f"Using cached {backend} engine")
    return ENGINE_CACHE[backend]


from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.websocket("/asr/realtime-custom-vad")
async def ws_asr(ws: WebSocket):
    print(f'{time.time()} connection request received', flush=True)
    print(ws.headers, flush=True)
    await ws.accept()

    init = await ws.receive_text()
    init_obj = json.loads(init)

    backend = init_obj.get("backend")
    if backend not in ("nemotron", "whisper", "google"):
        await ws.close(code=4000)
        return

    client_sample_rate = init_obj.get("sample_rate", cfg.sample_rate)

    def upsample_if_needed(pcm: bytes) -> bytes:
        if not pcm or client_sample_rate == cfg.sample_rate:
            return pcm
        x = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
        y = resampy.resample(x, client_sample_rate, cfg.sample_rate)
        y = np.clip(y, -1.0, 1.0)
        return (y * 32767.0).astype(np.int16).tobytes()

    engine = get_engine(backend)
    log.info(f"WS connected ({backend}) {ws.client}")

    vad = AdaptiveEnergyVAD(
        cfg.sample_rate,
        cfg.vad_frame_ms,
        cfg.vad_start_margin,
        cfg.vad_min_noise_rms,
        cfg.pre_speech_ms,
    )

    session = engine.new_session(max_buffer_ms=cfg.max_utt_ms)

    frame_bytes = int(cfg.sample_rate * cfg.vad_frame_ms / 1000) * 2
    raw_buf = bytearray()

    utt_started = False
    utt_audio_ms = 0
    t_utt_start = None
    t_first_partial = None
    silence_ms = 0

    try:
        while True:
            msg = await ws.receive()
            if msg["type"] == "websocket.disconnect":
                break

            data = msg.get("bytes")
            if data is None:
                continue

            data = upsample_if_needed(data)

            # EOS from client
            if data == b"":
                if utt_started:
                    final = session.finalize(cfg.post_speech_pad_ms)
                    await _emit_final(ws, final, t_utt_start, t_first_partial)
                break

            raw_buf.extend(data)

            while len(raw_buf) >= frame_bytes:
                frame = bytes(raw_buf[:frame_bytes])
                del raw_buf[:frame_bytes]

                is_speech, pre = vad.push_frame(frame)
                silence_ms = 0 if is_speech else silence_ms + cfg.vad_frame_ms

                if pre and not utt_started:
                    utt_started = True
                    utt_audio_ms = 0
                    t_utt_start = time.time()
                    t_first_partial = None
                    silence_ms = 0
                    session.accept_pcm16(pre)

                if not utt_started:
                    continue

                session.accept_pcm16(frame)
                utt_audio_ms += cfg.vad_frame_ms

                # PARTIALS
                if engine.caps.partials:
                    text = session.step_if_ready()
                    if text:
                        if t_first_partial is None:
                            t_first_partial = time.time()

                        ttfb_ms = int((t_first_partial - t_utt_start) * 1000)

                        await ws.send_text(json.dumps({
                            "type": "partial",
                            "text": text,
                            "t_start": ttfb_ms
                        }))

                # ENDPOINT
                if (
                    not is_speech
                    and utt_audio_ms >= cfg.min_utt_ms
                    and silence_ms >= cfg.end_silence_ms
                ):
                    final = session.finalize(cfg.post_speech_pad_ms)
                    await _emit_final(ws, final, t_utt_start, t_first_partial)

                    vad.reset()
                    utt_started = False
                    utt_audio_ms = 0
                    silence_ms = 0

    finally:
        await ws.close()
        log.info("WS disconnected")


async def _emit_final(ws, final_text, t_start, t_first_partial):
    if not final_text:
        return

    ttfb_ms = (
        int((t_first_partial - t_start) * 1000)
        if (t_first_partial and t_start)
        else None
    )

    await ws.send_text(json.dumps({
        "type": "final",
        "text": final_text,
        "t_start": ttfb_ms
    }))


app/vad.py-
from collections import deque
import numpy as np

class AdaptiveEnergyVAD:
    """
    Adaptive noise-floor VAD.
    Speech if RMS > noise_rms * start_margin.
    Keeps a pre-speech ring buffer so we don't cut initial phonemes.
    """
    def __init__(self, sample_rate: int, frame_ms: int, start_margin: float, min_noise_rms: float, pre_speech_ms: int):
        self.sr = sample_rate
        self.frame_ms = frame_ms
        self.start_margin = start_margin
        self.min_noise_rms = min_noise_rms

        self.frame_samples = int(self.sr * self.frame_ms / 1000)
        self.frame_bytes = self.frame_samples * 2

        self.pre_frames = max(1, int(pre_speech_ms / frame_ms))
        self.ring = deque(maxlen=self.pre_frames)

        self.in_speech = False
        self.noise_rms = min_noise_rms

    def reset(self):
        self.ring.clear()
        self.in_speech = False
        self.noise_rms = self.min_noise_rms

    def _rms(self, pcm16: bytes) -> float:
        x = np.frombuffer(pcm16, dtype=np.int16).astype(np.float32) / 32768.0
        return float(np.sqrt(np.mean(x * x) + 1e-12))

    def push_frame(self, frame_pcm16: bytes):
        e = self._rms(frame_pcm16)

        # Update noise floor when not in speech (EMA)
        if not self.in_speech:
            alpha = 0.95
            self.noise_rms = max(self.min_noise_rms, alpha * self.noise_rms + (1 - alpha) * e)

        th = max(self.min_noise_rms, self.noise_rms) * self.start_margin
        is_speech = e >= th

        self.ring.append(frame_pcm16)

        pre_roll = None
        if (not self.in_speech) and is_speech:
            self.in_speech = True
            pre_roll = b"".join(self.ring)

        if self.in_speech and (not is_speech):
            # endpointing handles finalization
            pass

        return is_speech, pre_roll

#Dockerfile-
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ARG USE_PROXY=false

ENV http_proxy=${USE_PROXY:+http://163.116.128.80:8080}
ENV https_proxy=${USE_PROXY:+http://163.116.128.80:8080}

WORKDIR /srv

COPY requirements.txt .

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3-pip \
    python3-dev

RUN pip install --upgrade pip setuptools wheel \
 && pip install --no-cache-dir -r requirements.txt

COPY download_model/nemotron-speech-streaming/nemotron-speech-streaming-en-0.6b.nemo .
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


i was told to do this 
- code simplification (you can create a new branch from the main branch):
			- remove endpointing.py
			- main.py
				- move heavy lifting code from "vad = AdaptiveEnergyVAD(" to another file
				- ideally main.py should only have fastapi related things
			- add time in logs and flush log.info as soon as it's printed.  for this I think setting "ENV PYTHONUNBUFFERED=1" should do.
				- also we need to print logs for when a websocket connection was made
			- download_model directly in docker image and copy to the docker image.

there is asr_engines folder but don't need to touch that 
