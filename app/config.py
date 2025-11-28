# app/config.py

from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env")

    # Routing thresholds
    REGEX_CONFIDENCE: float = 0.95
    ML_CONFIDENCE: float = 0.70

    # Paths
    LR_MODEL_PATH: str = "models/lr_model.pkl"
    LABEL_ENCODER_PATH: str = "models/label_encoder.pkl"

    # LLM
    GEMINI_API_KEY: str = ""
    LLM_MODEL_NAME: str = "models/gemini-2.5-flash"

settings = Settings()
