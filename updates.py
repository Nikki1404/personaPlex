this is my code for asr-realtime which has nemotron, whisper, google which I am working on simplifying the code to productionize it . 
here is my complete project -
  
#app/asr_engines/base.py-
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Protocol


@dataclass(frozen=True)
class EngineCaps:
    """
    Declares behavioral capabilities of an ASR engine.
    """
    streaming: bool          # true streaming model with incremental state
    partials: bool           # supports partial outputs during speech
    ttft_meaningful: bool    # whether TTFT is meaningful (streaming only)


class ASRSession(Protocol):
    """
    Session interface the server expects.
    Session may optionally expose timing fields:
      - utt_preproc, utt_infer, utt_flush, chunks
    """
    def accept_pcm16(self, pcm16: bytes) -> None: ...
    def step_if_ready(self) -> Optional[str]: ...
    def finalize(self, pad_ms: int) -> str: ...


class ASREngine(ABC):
    """
    Engine interface.
    """
    caps: EngineCaps

    @abstractmethod
    def load(self) -> float:
        ...

    @abstractmethod
    def new_session(self, max_buffer_ms: int) -> ASRSession:
        ...

#app/asr_engines/google_streaming_asr.py-
import os
import time
import queue
import threading
from typing import Optional

from google.api_core.client_options import ClientOptions
from google.cloud import speech_v2
from google.cloud.speech_v2.types import cloud_speech

from app.asr_engines.base import ASREngine, EngineCaps


class GoogleStreamingASR(ASREngine):
    """
    Google Cloud Speech-to-Text v2 streaming engine.

    True streaming:
    - partials supported
    - final supported
    - TTFT meaningful (network dependent)
    """

    caps = EngineCaps(
        streaming=True,
        partials=True,
        ttft_meaningful=True,
    )

    def __init__(self, sample_rate: int):
        self.sr = sample_rate
        self.client: Optional[speech_v2.SpeechClient] = None

        self.region = os.getenv("GOOGLE_REGION", "us-central1")
        self.recognizer = os.getenv("GOOGLE_RECOGNIZER", "").strip()
        self.language_code = os.getenv("GOOGLE_LANG", "en-US")
        self.model = os.getenv("GOOGLE_MODEL", "latest_short")

    @property
    def model_name(self) -> str:
        return f"google:{self.model}"

    def load(self) -> float:
        t0 = time.time()

        if not self.recognizer:
            raise ValueError("GOOGLE_RECOGNIZER env variable not set")

        endpoint = f"{self.region}-speech.googleapis.com"
        self.client = speech_v2.SpeechClient(
            client_options=ClientOptions(api_endpoint=endpoint)
        )

        return time.time() - t0

    def new_session(self, max_buffer_ms: int):
        return GoogleStreamingSession(self)


# =============================================================


class GoogleStreamingSession:
    """
    Per-websocket session.

    IMPORTANT:
    Your server calls step_if_ready() every VAD frame.
    So we must:
      - return partial ONLY when it changes (dedupe)
      - reset internal accumulators after finalize() (like Nemotron/Whisper)
    """

    def __init__(self, engine: GoogleStreamingASR):
        self.engine = engine
        if self.engine.client is None:
            raise RuntimeError("Google engine not loaded")

        # metrics parity with Nemotron
        self.utt_preproc = 0.0
        self.utt_infer = 0.0
        self.utt_flush = 0.0
        self.chunks = 0

        # state init
        self._reset_utterance_state()

    # -------------------------
    # session lifecycle
    # -------------------------

    def _reset_utterance_state(self):
        # audio queue + thread state
        self._audio_q: "queue.Queue[Optional[bytes]]" = queue.Queue()
        self._closed = False

        # transcript state
        self._latest_partial: str = ""
        self._last_partial_sent: str = ""   # DEDUPE KEY
        self._final_parts: list[str] = []

        # chunk counter should reset per utterance (matches Nemotron behavior)
        self.chunks = 0

        # restart worker thread
        self._thread = threading.Thread(target=self._run_stream, daemon=True)
        self._thread.start()

    # -------------------------
    # ASRSession interface
    # -------------------------

    def accept_pcm16(self, pcm16: bytes):
        if self._closed:
            return
        if not pcm16:
            return
        self._audio_q.put(pcm16)
        self.chunks += 1

    def step_if_ready(self) -> Optional[str]:
        """
        Return a partial ONLY when it's NEW (prevents infinite repeat prints).
        """
        if not self._latest_partial:
            return None

        if self._latest_partial == self._last_partial_sent:
            return None

        self._last_partial_sent = self._latest_partial
        return self._latest_partial

    def finalize(self, pad_ms: int) -> str:
        """
        Close current Google stream, wait briefly for final results, return final text,
        then RESET for next utterance (critical fix).
        """
        # Optional pad to help last phonemes (similar to your other engines)
        if pad_ms and pad_ms > 0:
            pad_samples = int(self.engine.sr * (pad_ms / 1000.0))
            pad_bytes = b"\x00\x00" * pad_samples
            try:
                self._audio_q.put(pad_bytes)
            except Exception:
                pass

        if not self._closed:
            self._closed = True
            self._audio_q.put(None)

        t0 = time.perf_counter()
        self._thread.join(timeout=2.5)
        self.utt_flush += (time.perf_counter() - t0)

        final_text = " ".join(self._final_parts).strip()

        # ✅ CRITICAL: reset for next utterance so we don't repeat the same final forever
        self._reset_utterance_state()

        return final_text

    # -------------------------
    # internal: google request/response
    # -------------------------

    def _request_generator(self):
        """
        Uses ExplicitDecodingConfig for raw PCM16 streaming.
        """

        recognition_config = cloud_speech.RecognitionConfig(
            explicit_decoding_config=cloud_speech.ExplicitDecodingConfig(
                encoding=cloud_speech.ExplicitDecodingConfig.AudioEncoding.LINEAR16,
                sample_rate_hertz=self.engine.sr,
                audio_channel_count=1,
            ),
            language_codes=[self.engine.language_code],
            model=self.engine.model,
        )

        streaming_config = cloud_speech.StreamingRecognitionConfig(
            config=recognition_config,
        )

        # First request must contain config
        yield cloud_speech.StreamingRecognizeRequest(
            recognizer=self.engine.recognizer,
            streaming_config=streaming_config,
        )

        while True:
            chunk = self._audio_q.get()
            if chunk is None:
                break

            yield cloud_speech.StreamingRecognizeRequest(audio=chunk)

    def _run_stream(self):
        try:
            t_infer0 = time.perf_counter()

            responses = self.engine.client.streaming_recognize(
                requests=self._request_generator()
            )

            for response in responses:
                self.utt_infer += (time.perf_counter() - t_infer0)
                t_infer0 = time.perf_counter()

                for result in response.results:
                    if not result.alternatives:
                        continue

                    transcript = (result.alternatives[0].transcript or "").strip()
                    if not transcript:
                        continue

                    if result.is_final:
                        self._final_parts.append(transcript)
                        # once final arrives, clear partial so dedupe resets naturally
                        self._latest_partial = ""
                        self._last_partial_sent = ""
                    else:
                        self._latest_partial = transcript

        except Exception as e:
            print("Google streaming error:", e)
            self._latest_partial = ""
            self._last_partial_sent = ""
            return 

#app/asr_engines/nemotron_asr.py-
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
#app/asr_engines/whisper_asr.py-
import time
from typing import Optional
import unicodedata
import re

import numpy as np
import torch
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq

from app.asr_engines.base import ASREngine, EngineCaps


_DEVANAGARI_RE = re.compile(r"[\u0900-\u097F]")  # Hindi script range
_ALLOWED_ASCII_RE = re.compile(r"^[\x00-\x7F]+$")  # basic ASCII


def _ascii_fold(s: str) -> str:
    """
    Convert accented Latin chars to closest ASCII (Halló -> Hallo).
    Keeps ASCII punctuation/numbers.
    """
    s = unicodedata.normalize("NFKD", s)
    return s.encode("ascii", "ignore").decode("ascii", "ignore")


def _english_only_filter(text: str) -> str:
    """
    Enforce 'flush only English' in a practical way:
    - If Devanagari appears -> drop completely.
    - Otherwise fold accents to ASCII.
    - If still contains non-ASCII -> drop.
    """
    if not text:
        return ""

    # Hard block Hindi/Devanagari output
    if _DEVANAGARI_RE.search(text):
        return ""

    folded = _ascii_fold(text).strip()

    # If anything non-ascii remains after folding -> drop
    if folded and not _ALLOWED_ASCII_RE.match(folded):
        return ""

    return folded


class WhisperTurboASR(ASREngine):
    """
    Chunked (non-streaming) ASR using Whisper Turbo.

    - English prompt enforced (no language auto-detection prompt)
    - Post-filter drops Hindi/Devanagari and non-ASCII outputs
    - Final transcription only
    """

    caps = EngineCaps(
        streaming=False,
        partials=False,
        ttft_meaningful=False,
    )

    def __init__(self, model_name: str, device: str, sample_rate: int):
        self.model_name = model_name
        self.device = device
        self.sr = sample_rate

        self.model = None
        self.processor = None
        self.forced_decoder_ids = None

    def load(self) -> float:
        t0 = time.time()

        self.processor = AutoProcessor.from_pretrained(self.model_name)
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            low_cpu_mem_usage=True,
        )

        if self.device == "cuda":
            self.model = self.model.cuda()
        else:
            self.model = self.model.cpu()

        self.model.eval()

        # Force decoder prompt to English transcription
        self.forced_decoder_ids = self.processor.get_decoder_prompt_ids(
            language="en",
            task="transcribe",
        )

        self._warmup()
        return time.time() - t0

    @torch.inference_mode()
    def _warmup(self):
        try:
            silence = np.zeros(int(self.sr * 1.0), dtype=np.float32)
            inputs = self.processor(
                silence,
                sampling_rate=self.sr,
                return_tensors="pt",
            )
            inputs = {k: v.to(device=self.model.device, dtype=self.model.dtype) for k, v in inputs.items()}
            _ = self.model.generate(
                **inputs,
                forced_decoder_ids=self.forced_decoder_ids,
                max_new_tokens=16,
            )
        except Exception:
            pass

    def new_session(self, max_buffer_ms: int):
        return WhisperSession(self, max_buffer_ms=max_buffer_ms)


class WhisperSession:
    def __init__(self, engine: WhisperTurboASR, max_buffer_ms: int):
        self.engine = engine
        self.max_buffer_samples = int(engine.sr * (max_buffer_ms / 1000.0))
        self.audio = np.array([], dtype=np.float32)

        self.utt_preproc = 0.0
        self.utt_infer = 0.0
        self.utt_flush = 0.0
        self.chunks = 0

    def accept_pcm16(self, pcm16: bytes):
        x = np.frombuffer(pcm16, dtype=np.int16).astype(np.float32) / 32768.0
        self.audio = np.concatenate([self.audio, x])
        if len(self.audio) > self.max_buffer_samples:
            self.audio = self.audio[-self.max_buffer_samples:]

    def step_if_ready(self) -> Optional[str]:
        return None

    @torch.inference_mode()
    def finalize(self, pad_ms: int) -> str:
        if len(self.audio) == 0:
            return ""

        pad = np.zeros(int(self.engine.sr * (pad_ms / 1000.0)), dtype=np.float32)
        audio = np.concatenate([self.audio, pad])

        t0 = time.perf_counter()
        inputs = self.engine.processor(
            audio,
            sampling_rate=self.engine.sr,
            return_tensors="pt",
        )
        self.utt_preproc += (time.perf_counter() - t0)

        inputs = {k: v.to(device=self.engine.model.device, dtype=self.engine.model.dtype) for k, v in inputs.items()}

        t1 = time.perf_counter()
        generated_ids = self.engine.model.generate(
            **inputs,
            max_new_tokens=444,
            forced_decoder_ids=self.engine.forced_decoder_ids,
        )
        self.utt_infer += (time.perf_counter() - t1)

        self.chunks += 1

        text = self.engine.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
        )[0].strip()

        # 🔥 English-only output gate (drop Hindi/non-ASCII)
        text = _english_only_filter(text)

        self.audio = np.array([], dtype=np.float32)
        return text

#app/config.py-
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

#app/endpointing.py-
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

#app/factory.py-
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

app/gpu_monitor.py-
import time
import threading
import logging
from app.metrics import GPU_UTIL, GPU_MEM_USED_MB, GPU_MEM_TOTAL_MB

log = logging.getLogger("asr_server")

def start_gpu_monitor(enable: bool, gpu_index: int):
    if not enable:
        return
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(int(gpu_index))

        def loop():
            while True:
                try:
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
                    mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    GPU_UTIL.set(util)
                    GPU_MEM_USED_MB.set(mem.used / (1024 * 1024))
                    GPU_MEM_TOTAL_MB.set(mem.total / (1024 * 1024))
                except Exception:
                    pass
                time.sleep(2)

        t = threading.Thread(target=loop, daemon=True)
        t.start()
        log.info("GPU monitor started")
    except Exception as e:
        log.warning(f"GPU monitor disabled: {e}")

#app/main.py-
import asyncio
import json
import time
import logging
import os
import numpy as np
import resampy

from fastapi import FastAPI, WebSocket
from fastapi.responses import Response

from app.config import load_config, Config, MODEL_MAP
from app.metrics import *
from app.vad import AdaptiveEnergyVAD
from app.factory import build_engine
from app.asr_engines.base import ASREngine

cfg = load_config()
logging.basicConfig(level=cfg.log_level, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger("asr_server")

app = FastAPI()
ENGINE_CACHE: dict[str, ASREngine] = {}

#  PRELOAD BOTH MODELS AT STARTUP (takes ~30-60s once)
async def preload_engines():
    """Preload both Whisper + Nemotron models into cache"""
    backends = ["whisper", "nemotron", "google"]
    
    log.info(" Preloading ASR engines (this happens once at startup)...")
    for backend in backends:
        try:
            model_name = MODEL_MAP[backend]
            print(f"   Loading {backend} ({model_name})...")
            
            tmp_cfg = Config()
            object.__setattr__(tmp_cfg, 'asr_backend', backend)
            object.__setattr__(tmp_cfg, 'model_name', model_name)
            object.__setattr__(tmp_cfg, 'device', cfg.device)
            object.__setattr__(tmp_cfg, 'sample_rate', cfg.sample_rate)
            object.__setattr__(tmp_cfg, 'context_right', cfg.context_right)
             
            os.environ["https_proxy"] = "http://163.116.128.80:8080"
            os.environ["http_proxy"] = "http://163.116.128.80:8080"
            
            engine = build_engine(tmp_cfg)
            load_sec = engine.load()
            
            os.environ.pop("https_proxy", None)
            os.environ.pop("http_proxy", None)
            
            log.info(f" Preloaded {backend} ({model_name}) in {load_sec:.2f}s")
            ENGINE_CACHE[backend] = engine
            
        except Exception as e:
            log.error(f" Failed to preload {backend}: {e}")
            continue
    
    log.info(" All engines preloaded! Client requests will be INSTANT.")

#  STARTUP EVENT - Preload happens automatically
@app.on_event("startup")
async def startup_event():
    await preload_engines()

def get_engine(backend: str) -> ASREngine:
    """Instant lookup from preloaded cache"""
    if backend not in ENGINE_CACHE:
        raise ValueError(f"Engine '{backend}' not preloaded. Available: {list(ENGINE_CACHE.keys())}")
    
    log.info(f" Using cached {backend} engine (0ms latency!)")
    return ENGINE_CACHE[backend]


@app.websocket("/asr/realtime-custom-vad")
async def ws_asr(ws: WebSocket):
    await ws.accept()

    #  FIRST MESSAGE MUST BE CONFIG
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

    labels = (backend, engine.model_name)

    active_streams = ACTIVE_STREAMS.labels(*labels)
    partials_total = PARTIALS_TOTAL.labels(*labels)
    finals_total = FINALS_TOTAL.labels(*labels)
    utterances_total = UTTERANCES_TOTAL.labels(*labels)

    ttft_wall = TTFT_WALL.labels(*labels)
    ttf_wall = TTF_WALL.labels(*labels)

    infer_sec = INFER_SEC.labels(*labels)
    preproc_sec = PREPROC_SEC.labels(*labels)
    flush_sec = FLUSH_SEC.labels(*labels)

    audio_sec_hist = AUDIO_SEC.labels(*labels)
    rtf_hist = RTF.labels(*labels)
    backlog_ms_gauge = BACKLOG_MS.labels(*labels)

    active_streams.inc()
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

            if data == b"":
                if utt_started:
                    final = session.finalize(cfg.post_speech_pad_ms)
                    await _emit_final(
                        ws,
                        session,
                        final,
                        utt_audio_ms,
                        t_utt_start,
                        t_first_partial,
                        "eos",
                        utterances_total,
                        finals_total,
                        ttf_wall,
                        audio_sec_hist,
                        rtf_hist,
                        engine,
                    )
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

                if engine.caps.partials:
                    text = session.step_if_ready()
                    if text:
                        partials_total.inc()
                        if t_first_partial is None:
                            t_first_partial = time.time()
                            if engine.caps.ttft_meaningful:
                                ttft_wall.observe(t_first_partial - t_utt_start)

                        log.info(f"CLIENT: {ws.client}, TEXT: {text}, START_TIME : {int(t_utt_start * 1000)}")
                        await ws.send_text(json.dumps({
                            "type": "partial", 
                            "text": text,
                            "t_start": int(t_utt_start * 1000)
                        }))

                if (
                    not is_speech
                    and utt_audio_ms >= cfg.min_utt_ms
                    and silence_ms >= cfg.end_silence_ms
                ):
                    final = session.finalize(cfg.post_speech_pad_ms)
                    await _emit_final(
                        ws,
                        session,
                        final,
                        utt_audio_ms,
                        t_utt_start,
                        t_first_partial,
                        "silence",
                        utterances_total,
                        finals_total,
                        ttf_wall,
                        audio_sec_hist,
                        rtf_hist,
                        engine,
                    )
                    vad.reset()
                    utt_started = False
                    utt_audio_ms = 0
                    silence_ms = 0

    finally:
        active_streams.dec()
        await ws.close()
        log.info("WS disconnected")


async def _emit_final(
    ws,
    session,
    final_text,
    audio_ms,
    t_start,
    t_first_partial,
    reason,
    utterances_total,
    finals_total,
    ttf_wall,
    audio_sec_hist,
    rtf_hist,
    engine,
):
    if not final_text:
        return

    utterances_total.inc()
    finals_total.inc()

    audio_sec = audio_ms / 1000.0
    ttf = time.time() - t_start

    ttf_wall.observe(ttf)
    audio_sec_hist.observe(audio_sec)

    compute_sec = session.utt_preproc + session.utt_infer + session.utt_flush
    if audio_sec > 0:
        rtf_hist.observe(compute_sec / audio_sec)

    log.info(f"CLIENT: {ws.client}, TEXT: {final_text}")

    await ws.send_text(json.dumps({
        "type": "final",
        "text": final_text,
        "reason": reason,
        "t_start": int(t_start * 1000),
        "audio_ms": audio_ms,
        "ttf_ms": int(ttf * 1000),
        "ttft_ms": (
            int((t_first_partial - t_start) * 1000)
            if t_first_partial and engine.caps.ttft_meaningful
            else None
        ),
        "chunks": session.chunks,
        "model_preproc_ms": int(session.utt_preproc * 1000),
        "model_infer_ms": int(session.utt_infer * 1000),
        "model_flush_ms": int(session.utt_flush * 1000),
        "rtf": (compute_sec / audio_sec) if audio_sec > 0 else None,
    }))

app/metrics.py-
from prometheus_client import Counter, Histogram, Gauge

LABELS = ["backend", "model"]

ACTIVE_STREAMS = Gauge("asr_active_streams", "Active websocket streams", LABELS)

PARTIALS_TOTAL = Counter("asr_partials_total", "Partial messages sent", LABELS)
FINALS_TOTAL = Counter("asr_finals_total", "Final messages sent", LABELS)
UTTERANCES_TOTAL = Counter("asr_utterances_total", "Utterances finalized", LABELS)

# NOTE: TTFT only recorded when engine.caps.ttft_meaningful == True (Nemotron)
TTFT_WALL = Histogram("asr_ttft_wall_sec", "Wall TTFT seconds (streaming only)", LABELS)
TTF_WALL  = Histogram("asr_ttf_wall_sec", "Wall TTF seconds", LABELS)

INFER_SEC = Histogram("asr_infer_sec", "Model inference seconds", LABELS)
PREPROC_SEC = Histogram("asr_preproc_sec", "Model preproc seconds", LABELS)
FLUSH_SEC = Histogram("asr_flush_sec", "Finalize/flush wall seconds", LABELS)

AUDIO_SEC = Histogram("asr_audio_sec", "Audio seconds per utterance", LABELS)
RTF = Histogram("asr_rtf", "Real-time factor (infer/audio)", LABELS)

BACKLOG_MS = Gauge("asr_backlog_ms", "Buffered audio backlog (ms)", LABELS)


GPU_UTIL = Gauge("asr_gpu_util", "GPU utilization percent")
GPU_MEM_USED_MB = Gauge("asr_gpu_mem_used_mb", "GPU memory used MB")
GPU_MEM_TOTAL_MB = Gauge("asr_gpu_mem_total_mb", "GPU memory total MB")

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
FROM python:3.11-slim

ARG USE_PROXY=false
ARG HTTP_PROXY
ARG HTTPS_PROXY

ENV http_proxy=""
ENV https_proxy=""

RUN if [ "$USE_PROXY" = "true" ]; then \
        echo " Enabling proxy"; \
        export http_proxy=${HTTP_PROXY}; \
        export https_proxy=${HTTPS_PROXY}; \
    else \
        echo " Proxy disabled"; \
    fi

WORKDIR /srv

COPY download_model/nemotron-speech-streaming/nemotron-speech-streaming-en-0.6b.nemo nemotron-speech-streaming-en-0.6b.nemo

COPY requirements.txt requirements.txt

RUN pip install --upgrade pip setuptools wheel
RUN python3 -m pip install --no-cache-dir -r requirements.txt

COPY app app

COPY app/google_credentials.json google_credentials.json


ENV GOOGLE_APPLICATION_CREDENTIALS=/srv/google_credentials.json
ENV GOOGLE_RECOGNIZER=projects/eci-ugi-digital-ccaipoc/locations/us-central1/recognizers/google-stt-default
ENV GOOGLE_REGION=us-central1
ENV GOOGLE_LANGUAGE=en-US
ENV GOOGLE_MODEL=latest_short
ENV GOOGLE_INTERIM=true
ENV GOOGLE_EXPLICIT_DECODING=true
 
ENV http_proxy=""
ENV https_proxy=""

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8002"]

#requirements.txt-
fastapi
google-cloud-speech>=2.25.0
google-auth>=2.28.0
google-api-core>=2.19.0
omegaconf
resampy
transformers
uvicorn[standard]
nemo_toolkit[asr]
prometheus_client


please remove gpu_monitor.py
and please remove app/metrics.py and prometheus_client from requirements.txt. 
We just need the following values in return: type, text, and t_start
