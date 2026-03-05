import asyncio
import websockets
import json
import pyaudio
import numpy as np
import sys

print('''After you are done with dev work please test the following:
1. test for backend = "nemotron"
2. test for backend = "google"
3. test for backend = "whisper"
4. test for the combination: backend = "nemotron" and TARGET_SR = 16000
5. test for the combination: backend = "nemotron" and TARGET_SR = 8000
''')

# CONFIG
WEBSOCKET_ADDRESS = "ws://127.0.0.1:8002/asr/realtime-custom-vad"

TARGET_SR = 16000
CHANNELS = 1

CHUNK_MS = 80
CHUNK_FRAMES = int(TARGET_SR * CHUNK_MS / 1000)
SLEEP_SEC = CHUNK_MS / 1000.0

WHISPER_FLUSH_INTERVAL_SEC = 0.35
WHISPER_FLUSH_SILENCE_MS = 80

websocket = None
stream = None
is_recording = False


# RECEIVE LOOP
async def receive_data():
    try:
        async for msg in websocket:
            if isinstance(msg, str):
                obj = json.loads(msg)
                typ = obj.get("type")

                if typ == "partial":
                    txt = obj.get("text", "")
                    print(f"\r[PARTIAL] {txt[:120]}", end="", flush=True)

                elif typ == "final":
                    print(f"\n[FINAL] {obj.get('text')}")

                else:
                    print("[SERVER EVENT]", obj)

    except websockets.exceptions.ConnectionClosed:
        print("\nWebSocket closed")


# CONNECT
async def connect_websocket():
    global websocket
    websocket = await websockets.connect(
        WEBSOCKET_ADDRESS,
        max_size=None,
    )
    print(f"Connected to {WEBSOCKET_ADDRESS}")


# SEND BACKEND CONFIG
async def send_audio_config(backend: str):
    audio_config = {
        "backend": backend,
        "sample_rate": TARGET_SR
    }

    await websocket.send(json.dumps(audio_config))
    print(f"Backend config sent: {backend}")


# MIC START
async def start_recording():
    global stream, is_recording

    p = pyaudio.PyAudio()
    stream = p.open(
        format=pyaudio.paInt16,
        channels=CHANNELS,
        rate=TARGET_SR,
        input=True,
        frames_per_buffer=CHUNK_FRAMES,
    )

    is_recording = True
    print("Recording started (Ctrl+C to stop)")


# MIC STOP + EOS
async def stop_recording():
    global stream, is_recording
    is_recording = False

    try:
        await websocket.send(b"\x00\x00" * int(TARGET_SR * 0.6))
        await asyncio.sleep(0.5)
        await websocket.send(b"")
    except Exception:
        pass

    if stream:
        stream.stop_stream()
        stream.close()

    print("Recording stopped")


# MAIN LOOP
async def main():
    backend = "nemotron"
    if len(sys.argv) > 1:
        backend = sys.argv[1]

    if backend not in ("nemotron", "whisper", "google"):
        print("Usage: python client.py [nemotron|whisper|google]")
        return

    await connect_websocket()
    await send_audio_config(backend)
    await start_recording()

    recv_task = asyncio.create_task(receive_data())

    last_flush_time = asyncio.get_event_loop().time()

    try:
        while True:
            data = stream.read(CHUNK_FRAMES, exception_on_overflow=False)
            pcm = np.frombuffer(data, dtype=np.int16)

            await websocket.send(pcm.tobytes())

            if backend == "whisper":
                now = asyncio.get_event_loop().time()
                if now - last_flush_time >= WHISPER_FLUSH_INTERVAL_SEC:
                    silence_frames = int(
                        TARGET_SR * (WHISPER_FLUSH_SILENCE_MS / 1000.0)
                    )
                    silence = b"\x00\x00" * silence_frames
                    await websocket.send(silence)
                    last_flush_time = now

            await asyncio.sleep(SLEEP_SEC)

    except KeyboardInterrupt:
        print("\nKeyboard interrupt")

    finally:
        await stop_recording()
        recv_task.cancel()
        try:
            await websocket.close()
        except Exception:
            pass


if __name__ == "__main__":
    asyncio.run(main())
