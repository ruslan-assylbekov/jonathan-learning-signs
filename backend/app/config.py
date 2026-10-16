from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# The repository root, from this file rather than the working directory: the
# backend is normally started from backend/, where a bare ".env" would look for
# backend/.env and silently fall back to every default. Both locations are
# accepted, root first.
_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=(_ROOT / ".env", ".env"), extra="ignore")

    database_url: str = "postgresql+asyncpg://sign:sign@localhost:5432/sign"
    redis_url: str = "redis://localhost:6379/0"
    jwt_secret: str = "change-me"
    jwt_ttl_min: int = 60
    model_path: str = "ml/artifacts/model-v2.joblib"
    confidence_threshold: float = 0.7
    hold_frames: int = 10
    max_fps_per_conn: int = 30
    # Local development only: accepts any non-empty token on /ws/practice so
    # the camera loop can be exercised before auth exists (M3). Never deploy
    # with this on; with it off the socket is closed instead of open.
    dev_auth: bool = False
    cors_origins: str = "http://localhost:5173"
    log_level: str = "INFO"

    @property
    def model_file(self) -> Path:
        """``model_path`` resolved against the repository root.

        Its presence is what decides whether the app runs on a real model or on
        the stub, so it must not depend on the working directory.
        """
        path = Path(self.model_path)
        return path if path.is_absolute() else _ROOT / path


settings = Settings()
