(venv) PS C:\Users\re_nikitav\Desktop\bu-digital-cx-speech-asr-realtime-custom-vad\scrippython .\ws_client.py
After you are done with dev work please test the following:
1. test for backend = "nemotron"
2. test for backend = "google"
3. test for backend = "whisper"
4. test for the combination: backend = "nemotron" and TARGET_SR = 16000
5. test for the combination: backend = "nemotron" and TARGET_SR = 8000

---------------------
STARTING TESTING
---------------------


Traceback (most recent call last):
  File "C:\Users\re_nikitav\Desktop\bu-digital-cx-speech-asr-realtime-custom-vad\scripts\venv\Lib\site-packages\websockets\http11.py", line 244, in parse
    status_line = yield from parse_line(read_line)
                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\re_nikitav\Desktop\bu-digital-cx-speech-asr-realtime-custom-vad\scripts\venv\Lib\site-packages\websockets\http11.py", line 320, in parse_line
    line = yield from read_line(MAX_LINE_LENGTH)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\re_nikitav\Desktop\bu-digital-cx-speech-asr-realtime-custom-vad\scripts\venv\Lib\site-packages\websockets\streams.py", line 46, in read_line
    raise EOFError(f"stream ends after {p} bytes, before end of line")
EOFError: stream ends after 0 bytes, before end of line

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "C:\Users\re_nikitav\Desktop\bu-digital-cx-speech-asr-realtime-custom-vad\scripts\venv\Lib\site-packages\websockets\client.py", line 303, in parse
    response = yield from Response.parse(
               ^^^^^^^^^^^^^^^^^^^^^^^^^^
    ...<3 lines>...
    )
    ^
  File "C:\Users\re_nikitav\Desktop\bu-digital-cx-speech-asr-realtime-custom-vad\scripts\venv\Lib\site-packages\websockets\http11.py", line 246, in parse
    raise EOFError("connection closed while reading HTTP status line") from exc
EOFError: connection closed while reading HTTP status line

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "C:\Users\re_nikitav\Desktop\bu-digital-cx-speech-asr-realtime-custom-vad\scripts\ws_client.py", line 204, in <module>
    asyncio.run(main())
    ~~~~~~~~~~~^^^^^^^^
  File "C:\Program Files\Python313\Lib\asyncio\runners.py", line 195, in run
    return runner.run(main)
           ~~~~~~~~~~^^^^^^
  File "C:\Program Files\Python313\Lib\asyncio\runners.py", line 118, in run
    return self._loop.run_until_complete(task)
           ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^
  File "C:\Program Files\Python313\Lib\asyncio\base_events.py", line 725, in run_until_complete
    return future.result()
           ~~~~~~~~~~~~~^^
  File "C:\Users\re_nikitav\Desktop\bu-digital-cx-speech-asr-realtime-custom-vad\scripts\ws_client.py", line 151, in main
    await connect_websocket()
  File "C:\Users\re_nikitav\Desktop\bu-digital-cx-speech-asr-realtime-custom-vad\scripts\ws_client.py", line 77, in connect_websocket
    websocket = await websockets.connect(
                ^^^^^^^^^^^^^^^^^^^^^^^^^
    ...<2 lines>...
    )
    ^
  File "C:\Users\re_nikitav\Desktop\bu-digital-cx-speech-asr-realtime-custom-vad\scripts\venv\Lib\site-packages\websockets\asyncio\client.py", line 546, in __await_impl__
    await self.connection.handshake(
    ...<2 lines>...
    )
  File "C:\Users\re_nikitav\Desktop\bu-digital-cx-speech-asr-realtime-custom-vad\scripts\venv\Lib\site-packages\websockets\asyncio\client.py", line 115, in handshake
    raise self.protocol.handshake_exc
websockets.exceptions.InvalidMessage: did not receive a valid HTTP response


why gettingthis for this ws_client.py-
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

---------------------
STARTING TESTING
---------------------

''')

# CONFIG
WEBSOCKET_ADDRESS = "ws://127.0.0.1:8002/asr/realtime-custom-vad"
#WEBSOCKET_ADDRESS = "wss://cx-asr.exlservice.com/asr/realtime-custom-vad"

TARGET_SR = 16000
CHANNELS = 1

CHUNK_MS = 80
CHUNK_FRAMES = int(TARGET_SR * CHUNK_MS / 1000)
SLEEP_SEC = CHUNK_MS / 1000.0

# Whisper flush tuning
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
                    t_start = obj.get("t_start")

                    print(
                        f"\r[PARTIAL] {txt[:120]} (t_start={t_start} ms)",
                        end="",
                        flush=True,
                    )

                elif typ == "final":
                    txt = obj.get("text", "")
                    t_start = obj.get("t_start")

                    print(f"\n[FINAL] {txt}")
                    print(f"[SERVER] t_start={t_start} ms")

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
        "sample_rate": TARGET_SR,
    }

    await websocket.send(json.dumps(audio_config))
    print(f"Sent backend config: {backend}")


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
        # trailing silence
        await websocket.send(b"\x00\x00" * int(TARGET_SR * 0.6))
        await asyncio.sleep(0.5)

        # explicit EOS
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

            # Whisper flush logic
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


even after doing this 
PS C:\Users\re_nikitav>  ssh -L 8002:localhost:8002 re_nikitav@10.90.126.61
re_nikitav@10.90.126.61's password:
Welcome to Ubuntu 22.04.5 LTS (GNU/Linux 6.8.0-1047-aws x86_64)

 * Documentation:  https://help.ubuntu.com
 * Management:     https://landscape.canonical.com
 * Support:        https://ubuntu.com/pro

 System information as of Mon Mar 16 06:18:27 UTC 2026

 * Ubuntu Pro delivers the most comprehensive open source security and
   compliance features.

   https://ubuntu.com/aws/pro

Expanded Security Maintenance for Applications is not enabled.

140 updates can be applied immediately.
2 of these updates are standard security updates.
To see these additional updates run: apt list --upgradable

47 additional security updates can be applied with ESM Apps.
Learn more about enabling ESM Apps service at https://ubuntu.com/esm

New release '24.04.4 LTS' available.
Run 'do-release-upgrade' to upgrade to it.


Last login: Mon Mar 16 05:27:19 2026 from 10.54.67.114
re_nikitav@EC03-E01-AICOE1:~$ channel 3: open failed: connect failed: Connection refused
channel 3: open failed: connect failed: Connection refused
channel 3: open failed: connect failed: Connection refused
getting this 

in server logs there is nothing 
      language_model: null
      softmax_temperature: 1.0
      preserve_alignments: false
      ngram_lm_model: null
      ngram_lm_alpha: 0.0
      hat_subtract_ilm: false
      hat_ilm_weight: 0.0
    temperature: 1.0
    durations: []
    big_blank_durations: []

2026-03-16 06:08:29,655 - INFO - Preloaded nemotron in 13.05s
2026-03-16 06:08:29,763 - INFO - Preloaded google in 0.11s
2026-03-16 06:08:29,763 - INFO - All engines preloaded.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8002 (Press CTRL+C to quit)
(base) root@EC03-E01-AICOE1:/home/CORP/re_nikitav/asr
