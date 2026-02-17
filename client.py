import asyncio
import logging
import os
import queue
from urllib.parse import quote_plus

import aiohttp
import numpy as np
import sounddevice as sd
from scipy.signal import resample_poly
import sphn

from config import settings

logger = logging.getLogger("client")


def setup_logging():
    logging.basicConfig(
        level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
    )


def energy_vad(int16_audio: np.ndarray, threshold: float) -> bool:
    # VAD before resampling (cheap)
    # Mean absolute amplitude
    return float(np.mean(np.abs(int16_audio))) > threshold


def to_float_mono(int16_audio: np.ndarray) -> np.ndarray:
    # sphn expects float PCM in [-1..1] typically; their examples pass raw PCM arrays.
    # We'll normalize to float32 in [-1, 1].
    return (int16_audio.astype(np.float32) / 32768.0)


def resample_to_24k(float_audio: np.ndarray, in_sr: int, out_sr: int) -> np.ndarray:
    if in_sr == out_sr:
        return float_audio
    # resample_poly is stable and fast:
    # out = in * up/down
    # Choose integer ratio by gcd
    from math import gcd
    g = gcd(in_sr, out_sr)
    up = out_sr // g
    down = in_sr // g
    return resample_poly(float_audio, up, down).astype(np.float32)


async def recv_loop(ws: aiohttp.ClientWebSocketResponse, opus_decoder: sphn.OpusStreamReader, audio_q: queue.Queue):
    async for msg in ws:
        if msg.type == aiohttp.WSMsgType.BINARY:
            data = msg.data
            if not data:
                continue
            kind = data[0:1]
            payload = data[1:]

            if kind == b"\x00":
                logger.info("Received handshake from server.")
            elif kind == b"\x01":
                # audio opus
                opus_decoder.append_bytes(payload)
            elif kind == b"\x02":
                # text token stream (optional)
                try:
                    txt = payload.decode("utf-8", errors="ignore")
                    logger.debug(f"TEXT: {txt}")
                except Exception:
                    pass
        elif msg.type in (aiohttp.WSMsgType.CLOSE, aiohttp.WSMsgType.CLOSED):
            logger.info("Server closed connection.")
            break
        elif msg.type == aiohttp.WSMsgType.ERROR:
            logger.error("WebSocket error.")
            break

        # Drain decoded PCM into queue
        while True:
            pcm = opus_decoder.read_pcm()
            if pcm is None or pcm.shape[-1] == 0:
                break
            audio_q.put(pcm.astype(np.float32))


def audio_play_callback(audio_q: queue.Queue):
    def cb(outdata, frames, time, status):
        if not audio_q.empty():
            chunk = audio_q.get()
            # Ensure correct shape
            if chunk.ndim == 1:
                chunk = chunk.reshape(-1, 1)
            out = chunk[:frames]
            if out.shape[0] < frames:
                pad = np.zeros((frames - out.shape[0], 1), dtype=np.float32)
                out = np.vstack([out, pad])
            outdata[:] = out
        else:
            outdata.fill(0)
    return cb


async def send_loop(ws: aiohttp.ClientWebSocketResponse, opus_encoder: sphn.OpusStreamWriter, pcm_push_q: asyncio.Queue):
    while True:
        pcm = await pcm_push_q.get()
        if pcm is None:
            return
        opus_encoder.append_pcm(pcm)
        await asyncio.sleep(0)  # yield
        data = opus_encoder.read_bytes()
        if data:
            await ws.send_bytes(b"\x01" + data)


def mic_callback_factory(pcm_push_q: asyncio.Queue, in_sr: int):
    def cb(indata, frames, time, status):
        # indata: float32/int16 depending on dtype; we use int16 for VAD
        audio_i16 = indata[:, 0].copy()

        # VAD BEFORE resampling (requirement)
        if not energy_vad(audio_i16, settings.VAD_ENERGY_THRESHOLD):
            return

        if in_sr != settings.MODEL_SR:
            logger.info(f"Input {in_sr} Hz detected → resampling to {settings.MODEL_SR} Hz")
        # Convert and resample to 24k float
        audio_f = to_float_mono(audio_i16)
        audio_24k = resample_to_24k(audio_f, in_sr, settings.MODEL_SR)

        # Push to async queue
        try:
            pcm_push_q.put_nowait(audio_24k)
        except asyncio.QueueFull:
            # Drop if overloaded; better than blocking realtime audio callback
            pass

    return cb


async def main():
    setup_logging()

    # Client config (set via env or defaults)
    host = os.getenv("S2S_HOST", "localhost")
    port = int(os.getenv("S2S_PORT", str(settings.PORT)))
    in_sr = int(os.getenv("INPUT_SR", "8000"))  # test telephony by default

    # Frame size: Moshi uses 24k and a 12.5Hz frame rate => ~80ms frames.
    # We'll record in ~80ms chunks at input SR.
    blocksize = max(160, int(in_sr * 0.08))

    # Build text prompt from file
    prompt_text = open(settings.PROMPT_FILE, "r", encoding="utf-8").read().strip()

    # WS URL (moshi.server uses /api/chat) :contentReference[oaicite:8]{index=8}
    url = (
        f"ws://{host}:{port}/api/chat"
        f"?text_prompt={quote_plus(prompt_text)}"
        f"&voice_prompt={quote_plus(settings.VOICE_PROMPT)}"
    )

    logger.info(f"Connecting to {url}")
    logger.info(f"Mic input SR: {in_sr} Hz | Model SR: {settings.MODEL_SR} Hz | blocksize: {blocksize}")

    audio_q: queue.Queue = queue.Queue()
    pcm_push_q: asyncio.Queue = asyncio.Queue(maxsize=8)

    # Opus encoder/decoder must use MODEL_SR (server expects MODEL_SR) :contentReference[oaicite:9]{index=9}
    opus_encoder = sphn.OpusStreamWriter(settings.MODEL_SR)
    opus_decoder = sphn.OpusStreamReader(settings.MODEL_SR)

    async with aiohttp.ClientSession() as session:
        async with session.ws_connect(url) as ws:
            # Wait for handshake byte 0x00 :contentReference[oaicite:10]{index=10}
            logger.info("Waiting for server handshake...")
            while True:
                msg = await ws.receive()
                if msg.type == aiohttp.WSMsgType.BINARY and msg.data[:1] == b"\x00":
                    logger.info("Handshake OK. Starting realtime streaming.")
                    break
                if msg.type in (aiohttp.WSMsgType.CLOSE, aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                    raise RuntimeError("Handshake failed / connection closed.")

            # Start audio output
            out_stream = sd.OutputStream(
                samplerate=settings.MODEL_SR,
                channels=1,
                dtype="float32",
                blocksize=int(settings.MODEL_SR * 0.08),
                callback=audio_play_callback(audio_q),
            )

            # Start mic input (int16 for VAD)
            in_stream = sd.InputStream(
                samplerate=in_sr,
                channels=1,
                dtype="int16",
                blocksize=blocksize,
                callback=mic_callback_factory(pcm_push_q, in_sr),
            )

            with in_stream, out_stream:
                await asyncio.gather(
                    recv_loop(ws, opus_decoder, audio_q),
                    send_loop(ws, opus_encoder, pcm_push_q),
                )


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
