# app/config.py
import os
from pydantic_settings import BaseSettings  # Changed from pydantic import BaseSettings
from functools import lru_cache

class Settings(BaseSettings):
    eden_ai_api_key: str = os.getenv("EDEN_AI_API_KEY", "")
    speech_to_text_url: str = "https://api.edenai.run/v2/audio/speech_to_text_async/"
    speech_to_text_result_url: str = "https://api.edenai.run/v2/audio/speech_to_text_async/{public_id}"
    default_language: str = "en"
    default_providers: list = ["google"]

    class Config:
        env_file = ".env"

@lru_cache()
def get_settings():
    return Settings()