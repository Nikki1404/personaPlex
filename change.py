FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ARG USE_PROXY=false

ENV http_proxy=${USE_PROXY:+http://163.116.128.80:8080}
ENV https_proxy=${USE_PROXY:+http://163.116.128.80:8080}
ENV PYTHONUNBUFFERED=1

# Optional cache paths (cleaner containers)
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

# Download ASR models during build
RUN python3 - <<EOF
import nemo.collections.asr as nemo_asr
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

print("===== Downloading Nemotron from HuggingFace =====")

model = nemo_asr.models.ASRModel.from_pretrained(
    "nvidia/nemotron-speech-streaming-en-0.6b"
)

model.save_to("/srv/nemotron-speech-streaming-en-0.6b.nemo")

print("Nemotron saved at /srv/nemotron-speech-streaming-en-0.6b.nemo")


print("===== Downloading Whisper Turbo =====")

AutoModelForSpeechSeq2Seq.from_pretrained(
    "openai/whisper-large-v3-turbo"
)

AutoProcessor.from_pretrained(
    "openai/whisper-large-v3-turbo"
)

print("Whisper downloaded and cached.")

EOF

# Copy application
COPY app ./app
COPY app/google_credentials.json google_credentials.json

# Google STT configuration
ENV GOOGLE_APPLICATION_CREDENTIALS=/srv/google_credentials.json
ENV GOOGLE_RECOGNIZER=projects/eci-ugi-digital-ccaipoc/locations/us-central1/recognizers/google-stt-default
ENV GOOGLE_REGION=us-central1
ENV GOOGLE_LANGUAGE=en-US
ENV GOOGLE_MODEL=latest_short
ENV GOOGLE_INTERIM=true
ENV GOOGLE_EXPLICIT_DECODING=true

# Run server
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


# Download ASR models during build
RUN python3 - <<EOF
import nemo.collections.asr as nemo_asr
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

print("===== Downloading Nemotron from HuggingFace =====")

model = nemo_asr.models.ASRModel.from_pretrained(
    "nvidia/nemotron-speech-streaming-en-0.6b"
)

model.save_to("/srv/nemotron-speech-streaming-en-0.6b.nemo")

print("Nemotron saved at /srv/nemotron-speech-streaming-en-0.6b.nemo")


print("===== Downloading Whisper Turbo =====")

AutoModelForSpeechSeq2Seq.from_pretrained(
    "openai/whisper-large-v3-turbo"
)

AutoProcessor.from_pretrained(
    "openai/whisper-large-v3-turbo"
)

print("Whisper downloaded and cached.")

EOF

COPY app ./app
COPY app/google_credentials.json google_credentials.json

# Google STT configuration
ENV GOOGLE_APPLICATION_CREDENTIALS=/srv/google_credentials.json
ENV GOOGLE_RECOGNIZER=projects/eci-ugi-digital-ccaipoc/locations/us-central1/recognizers/google-stt-default
ENV GOOGLE_REGION=us-central1
ENV GOOGLE_LANGUAGE=en-US
ENV GOOGLE_MODEL=latest_short
ENV GOOGLE_INTERIM=true
ENV GOOGLE_EXPLICIT_DECODING=true

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8002"]


import time
from dataclasses import dataclass
from typing import Optional, Any, Tuple

import numpy as np
import torch
from omegaconf import OmegaConf

# 🔧 ADDED: engine interface + capability metadata
from app.asr_engines.base import ASREngine, EngineCaps


def safe_text(h: Any) -> str:
    """
    Robust extraction for NeMo outputs (Hypothesis / str / list / None).
    Never throws, always returns a string.
    """
    if h is None:
        return ""
    if isinstance(h, str):
        return h
    # sometimes texts is like [Hypothesis(...)] or nested
    if isinstance(h, (list, tuple)) and len(h) > 0:
        return safe_text(h[0])
    if hasattr(h, "text"):
        try:
            return h.text or ""
        except Exception:
            return ""
    try:
        return str(h)
    except Exception:
        return ""


@dataclass
class StreamTimings:
    preproc_sec: float = 0.0
    infer_sec: float = 0.0
    flush_sec: float = 0.0


class NemotronStreamingASR(ASREngine):
    """
    True streaming ASR using NeMo's conformer_stream_step().
    Avoids calling model.transcribe() repeatedly (which is slow + noisy).
    """

    # 🔧 ADDED: engine capability declaration (used by server & metrics)
    caps = EngineCaps(
        streaming=True,
        partials=True,
        ttft_meaningful=True,
    )

    def __init__(self, model_name: str, device: str, sample_rate: int, context_right: int):
        self.model_name = model_name
        self.device = device
        self.sr = sample_rate
        self.context_right = context_right

        self.model = None

        # streaming params (set at load)
        self.shift_frames: int = 0
        self.pre_cache_frames: int = 0
        self.hop_samples: int = 0
        self.drop_extra: int = 0

        # helpful derived values
        self._frame_stride_sec: float = 0.01  # default; overwritten at load

    @property
    def chunk_samples(self) -> int:
        """
        Approx audio samples consumed per streaming step.
        shift_frames * hop_samples is a good approximation.
        """
        if self.shift_frames <= 0 or self.hop_samples <= 0:
            return int(0.08 * self.sr)  # safe fallback 80ms
        return int(self.shift_frames * self.hop_samples)

    def _to_device(self, x: torch.Tensor) -> torch.Tensor:
        if self.device == "cuda":
            return x.cuda(non_blocking=True)
        return x.cpu()

    def _move_cache_to_device(self, cache: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]):
        c0, c1, c2 = cache
        return (self._to_device(c0), self._to_device(c1), self._to_device(c2))

    def load(self):
        import nemo.collections.asr as nemo_asr

        t0 = time.time()

        # HF model name or local .nemo
        if self.model_name.endswith(".nemo"):
            self.model = nemo_asr.models.ASRModel.restore_from(self.model_name, map_location="cpu")
        else:
            self.model = nemo_asr.models.ASRModel.from_pretrained(self.model_name, map_location="cpu")

        if self.device == "cuda":
            self.model = self.model.cuda()
        else:
            self.model = self.model.cpu()

        # streaming attention context
        # keep your logic intact
        try:
            self.model.encoder.set_default_att_context_size([70, int(self.context_right)])
        except Exception:
            # some NeMo variants accept tuple
            self.model.encoder.set_default_att_context_size((70, int(self.context_right)))

        # IMPORTANT: disable cuda-graph decoder to avoid "CUDA failure 35" / fallback spam
        self.model.change_decoding_strategy(
            decoding_cfg=OmegaConf.create({
                "strategy": "greedy",
                "greedy": {
                    "max_symbols": 10,
                    "loop_labels": False,
                    "use_cuda_graph_decoder": False,
                }
            })
        )

        self.model.eval()
        try:
            self.model.preprocessor.featurizer.dither = 0.0
        except Exception:
            pass

        # streaming cfg
        scfg = self.model.encoder.streaming_cfg
        self.shift_frames = scfg.shift_size[1] if isinstance(scfg.shift_size, (list, tuple)) else scfg.shift_size
        pre_cache = scfg.pre_encode_cache_size
        self.pre_cache_frames = pre_cache[1] if isinstance(pre_cache, (list, tuple)) else pre_cache
        self.drop_extra = int(getattr(scfg, "drop_extra_pre_encoded", 0))

        # hop size in samples (audio->feature stride)
        self._frame_stride_sec = float(self.model.cfg.preprocessor.get("window_stride", 0.01))
        self.hop_samples = int(self._frame_stride_sec * self.sr)

        # warmup streaming kernels (FIXED)
        self._warmup()

        return time.time() - t0

    @torch.inference_mode()
    def _warmup(self):
        """
        ✅ FIX: Warmup using the real StreamingSession path.
        Older code sometimes called stream_transcribe() with wrong args.
        """
        try:
            sess = self.new_session(max_buffer_ms=3000)
            silence = np.zeros(int(self.sr * 1.0), dtype=np.float32)
            # Feed as pcm16-like float32 (convert back to pcm16 bytes)
            pcm16 = (np.clip(silence, -1.0, 1.0) * 32767).astype(np.int16).tobytes()
            sess.accept_pcm16(pcm16)
            _ = sess.finalize(pad_ms=400)
        except Exception:
            # Warmup should never crash startup
            pass

    def new_session(self, max_buffer_ms: int):
        return StreamingSession(self, max_buffer_ms=max_buffer_ms)

    @torch.inference_mode()
    def stream_transcribe(
        self,
        audio_f32: np.ndarray,
        cache: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        prev_hyp: Any,
        prev_pred_out: Any,
        emitted_frames: int,
        force_flush: bool = False,
    ):
        """
        Process enough frames for ONE streaming step (shift_frames),
        or flush if force_flush=True.
        """
        assert self.model is not None
        timings = StreamTimings()

        # preprocess all current buffer
        t0 = time.perf_counter()
        audio_tensor = torch.from_numpy(audio_f32).unsqueeze(0)
        audio_tensor = self._to_device(audio_tensor)
        audio_len = torch.tensor([len(audio_f32)], device=audio_tensor.device)
        mel, mel_len = self.model.preprocessor(input_signal=audio_tensor, length=audio_len)
        timings.preproc_sec += (time.perf_counter() - t0)

        # available frames excluding edge
        available = int(mel.shape[-1]) - 1
        if available <= 0:
            return None, cache, prev_hyp, prev_pred_out, emitted_frames, timings

        need_frames = self.shift_frames
        enough = (available - emitted_frames) >= need_frames

        if not enough and not force_flush:
            return None, cache, prev_hyp, prev_pred_out, emitted_frames, timings

        # build chunk mel
        if emitted_frames == 0:
            chunk_start = 0
            chunk_end = min(self.shift_frames, available)
            drop_extra = 0
        else:
            chunk_start = max(0, emitted_frames - self.pre_cache_frames)
            chunk_end = min(emitted_frames + self.shift_frames, available)
            drop_extra = self.drop_extra

        chunk_mel = mel[:, :, chunk_start:chunk_end]
        chunk_len = torch.tensor([chunk_mel.shape[-1]], device=chunk_mel.device)

        # ensure cache on correct device
        cache = self._move_cache_to_device(cache)

        # infer step
        t1 = time.perf_counter()
        (prev_pred_out, texts, cache0, cache1, cache2, prev_hyp) = self.model.conformer_stream_step(
            processed_signal=chunk_mel,
            processed_signal_length=chunk_len,
            cache_last_channel=cache[0],
            cache_last_time=cache[1],
            cache_last_channel_len=cache[2],
            keep_all_outputs=False,
            previous_hypotheses=prev_hyp,
            previous_pred_out=prev_pred_out,
            drop_extra_pre_encoded=drop_extra,
            return_transcription=True,
        )
        timings.infer_sec += (time.perf_counter() - t1)

        new_cache = (cache0, cache1, cache2)

        if emitted_frames < available:
            emitted_frames = min(emitted_frames + self.shift_frames, available)

        text = ""
        if texts is not None:
            text = safe_text(texts).strip()

        return text, new_cache, prev_hyp, prev_pred_out, emitted_frames, timings


class StreamingSession:
    """
    Holds per-websocket streaming state.
    Maintains ring buffer so preprocessing cost doesn't grow unbounded.
    """

    def __init__(self, engine: NemotronStreamingASR, max_buffer_ms: int):
        self.engine = engine
        self.max_buffer_samples = int(engine.sr * (max_buffer_ms / 1000.0))

        # audio buffer float32
        self.audio = np.array([], dtype=np.float32)

        # encoder cache
        self.cache = None
        self.prev_hyp = None
        self.prev_pred = None
        self.emitted_frames = 0

        # text tracking
        self.current_text = ""
        self.last_final_text = ""

        # timing accumulators per utterance
        self.utt_preproc = 0.0
        self.utt_infer = 0.0
        self.utt_flush = 0.0
        self.chunks = 0

        # track trimming to prevent emitted_frames desync
        self._trimmed_since_last_step = False

        self.reset_stream_state()

    def reset_stream_state(self):
        # initial cache state
        cache = self.engine.model.encoder.get_initial_cache_state(batch_size=1)
        self.cache = self.engine._move_cache_to_device((cache[0], cache[1], cache[2]))

        self.prev_hyp = None
        self.prev_pred = None
        self.emitted_frames = 0

        self.current_text = ""
        self.audio = np.array([], dtype=np.float32)

        self.utt_preproc = 0.0
        self.utt_infer = 0.0
        self.utt_flush = 0.0
        self.chunks = 0

        self._trimmed_since_last_step = False

    def accept_pcm16(self, pcm16: bytes):
        x = np.frombuffer(pcm16, dtype=np.int16).astype(np.float32) / 32768.0
        self.audio = np.concatenate([self.audio, x])

        # bound audio buffer
        if len(self.audio) > self.max_buffer_samples:
            self.audio = self.audio[-self.max_buffer_samples:]
            # ✅ IMPORTANT: trimming breaks alignment with emitted_frames -> protect accuracy
            self._trimmed_since_last_step = True

    def backlog_ms(self) -> int:
        return int(1000 * (len(self.audio) / self.engine.sr))

    def step_if_ready(self) -> Optional[str]:
        # ✅ If buffer trimmed, safest is to reset stream cache alignment (prevents missing words)
        if self._trimmed_since_last_step and self.emitted_frames > 0:
            # keep already-produced text, but reset streaming state to realign
            self.cache = self.engine.model.encoder.get_initial_cache_state(batch_size=1)
            self.cache = self.engine._move_cache_to_device((self.cache[0], self.cache[1], self.cache[2]))
            self.prev_hyp = None
            self.prev_pred = None
            self.emitted_frames = 0
            self._trimmed_since_last_step = False

        text, self.cache, self.prev_hyp, self.prev_pred, self.emitted_frames, t = self.engine.stream_transcribe(
            audio_f32=self.audio,
            cache=self.cache,
            prev_hyp=self.prev_hyp,
            prev_pred_out=self.prev_pred,
            emitted_frames=self.emitted_frames,
            force_flush=False,
        )

        self.utt_preproc += t.preproc_sec
        self.utt_infer += t.infer_sec

        if text is None or text == "" or text == self.current_text:
            return None

        self.current_text = text
        self.chunks += 1
        return text

    def finalize(self, pad_ms: int) -> str:
        # pad zeros to flush last words
        pad = np.zeros(int(self.engine.sr * (pad_ms / 1000.0)), dtype=np.float32)
        self.audio = np.concatenate([self.audio, pad])

        t0 = time.perf_counter()
        text, self.cache, self.prev_hyp, self.prev_pred, self.emitted_frames, t = self.engine.stream_transcribe(
            audio_f32=self.audio,
            cache=self.cache,
            prev_hyp=self.prev_hyp,
            prev_pred_out=self.prev_pred,
            emitted_frames=self.emitted_frames,
            force_flush=True,
        )
        self.utt_preproc += t.preproc_sec
        self.utt_infer += t.infer_sec
        self.utt_flush += (time.perf_counter() - t0)

        if text:
            self.current_text = text.strip()

        final = self.current_text.strip()

        # keep last_final_text accumulation
        self.last_final_text = (self.last_final_text + " " + final).strip() if final else self.last_final_text

        # hard reset state (fresh caches)
        self.reset_stream_state()
        return final
