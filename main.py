import os
import sys
import logging
from pathlib import Path

from config import settings

logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
)
logger = logging.getLogger("main")


def _load_prompt_text() -> str:
    p = Path(settings.PROMPT_FILE)
    if not p.exists():
        raise RuntimeError(f"Prompt file not found: {p}")
    return p.read_text(encoding="utf-8").strip()


def main():
    if not settings.HF_TOKEN:
        raise RuntimeError("HF_TOKEN env var must be set (do not hardcode).")

    # moshi uses HF_TOKEN from env
    os.environ["HF_TOKEN"] = settings.HF_TOKEN

    # Validate prompt file exists (used by client as query param)
    prompt_preview = _load_prompt_text()
    logger.info("Prompt file loaded OK.")
    logger.info(f"PersonaPlex native sample rate: {settings.MODEL_SR} Hz")

    # Start moshi.server (official)
    # WS endpoint will be: ws://HOST:PORT/api/chat  (from server.py) :contentReference[oaicite:6]{index=6}
    argv = [
        "moshi.server",
        "--host", settings.HOST,
        "--port", str(settings.PORT),
        "--hf-repo", settings.HF_REPO,
        "--device", settings.DEVICE,
        "--static", settings.STATIC,
    ]
    if settings.CPU_OFFLOAD:
        argv.append("--cpu-offload")

    logger.info(f"Starting moshi.server: {' '.join(argv)}")

    # Run moshi.server as module main
    sys.argv = ["python", "-m"] + argv
    from moshi.server import main as moshi_main  # noqa: E402
    moshi_main()


if __name__ == "__main__":
    main()
