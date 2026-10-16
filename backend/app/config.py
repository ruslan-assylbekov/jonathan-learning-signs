from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = "postgresql+asyncpg://sign:sign@localhost:5432/sign"
    redis_url: str = "redis://localhost:6379/0"
    jwt_secret: str = "change-me"
    jwt_ttl_min: int = 60
    model_path: str = "ml/artifacts/model-v1.joblib"
    confidence_threshold: float = 0.7
    hold_frames: int = 10
    max_fps_per_conn: int = 30
    cors_origins: str = "http://localhost:5173"
    log_level: str = "INFO"


settings = Settings()
