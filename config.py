import os
import torch


class Settings:
    # HuggingFace auth (DO NOT hardcode)
    HF_TOKEN = os.getenv("HF_TOKEN", "")
    HF_REPO = os.getenv("HF_REPO", "nvidia/personaplex-7b-v1")

    # Server
    HOST = os.getenv("HOST", "0.0.0.0")
    PORT = int(os.getenv("PORT", "8000"))
    STATIC = os.getenv("STATIC", "none")  # "none" disables UI/static

    # Device
    DEVICE = os.getenv("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
    CPU_OFFLOAD = os.getenv("CPU_OFFLOAD", "0") == "1"

    # Persona control
    PROMPT_FILE = os.getenv("PROMPT_FILE", "prompts/dob_agent.txt")
    VOICE_PROMPT = os.getenv("VOICE_PROMPT", "NATM1.pt")  # must exist in voices dir

    # Audio
    MODEL_SR = int(os.getenv("MODEL_SR", "24000"))  # PersonaPlex native SR
    VAD_ENERGY_THRESHOLD = float(os.getenv("VAD_ENERGY_THRESHOLD", "450.0"))

    # Logging
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")


settings = Settings()
