import os
import tempfile
from pathlib import Path

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict, YamlConfigSettingsSource
from pydantic_settings.sources import PydanticBaseSettingsSource


ROOT_DIR = Path(__file__).parent.parent.parent.parent


def _get_yaml_path() -> str:
    # Support CONFIG_FILE env var for testing
    if config_file := os.environ.get("CONFIG_FILE"):
        return config_file

    config_path = Path("/etc/bioscopeai-classifier-service-config.yaml")
    if not config_path.exists():
        config_path = ROOT_DIR / "docs/bioscopeai-classifier-service-config.yaml"
    return str(config_path)


class AppSettings(BaseSettings):
    DEBUG: bool = False
    LOG_LEVEL: str = "info"
    LOG_FILE_LEVEL: str = "debug"
    LOG_FILE_PATH: str = "classifier-service.log"
    PROJECT_NAME: str = "BioScopeAI Classifier Service"
    PROJECT_VERSION: str = "0.0.1"


class SentrySettings(BaseSettings):
    SENTRY_DSN: SecretStr | None = None


class AuthSettings(BaseSettings):
    ACCESS_TOKEN_TTL_MINUTES: int = 15 * 15  # 15 minutes
    REFRESH_TOKEN_TTL_MINUTES: int = 60 * 24 * 7  # 7 days
    PUBLIC_KEY: str
    PRIVATE_KEY: SecretStr


class ServiceAuthSettings(BaseSettings):
    SERVICE_TOKEN: SecretStr
    CORE_API_BASE_URL: str = "http://bioscopeai-core:8000"


class ImageSettings(BaseSettings):
    UPLOAD_DIR: str
    ALLOWED_MIME: set[str]
    ALLOWED_EXT: set[str] = {".jpg", ".jpeg", ".png"}
    MAX_FILE_SIZE: int = 10 * 1024 * 1024


class KafkaSettings(BaseSettings):
    BOOTSTRAP_SERVERS: str

    # Topics and consumer group for classification jobs
    CLASSIFICATION_JOBS_TOPIC: str = "classification-job"
    CLASSIFICATION_RESULTS_TOPIC: str = "classification-result"
    CLASSIFICATION_CONSUMER_GROUP: str = "classification-result-group"

    # ---- SSL ---- #
    SSL_ENABLED: bool = False
    SSL_CAFILE: str | None = None
    SSL_CERTFILE: str | None = None
    SSL_KEYFILE: str | None = None
    # ---- SASL ---- #
    SASL_ENABLED: bool = False
    SASL_USERNAME: str | None = None
    SASL_PASSWORD: SecretStr | None = None
    SASL_MECHANISM: str = "SCRAM-SHA-512"


class MLModelSettings(BaseSettings):
    # Local model path
    MODEL_PATH: str = ""

    # HuggingFace Hub settings
    USE_HF_HUB: bool = False
    HF_REPO_ID: str = ""
    HF_TOKEN: SecretStr | None = None
    HF_CACHE_DIR: str = str(Path(tempfile.gettempdir()) / "hf_cache")

    # Model verification
    VERIFY_CHECKSUM: bool = True

    # Model configurration
    IMG_SIZE: tuple[int, int] = (300, 300)
    CLASS_NAMES: list[str] = [
        "bone_cells_group",
        "bone_cells_individual",
        "other",
        "rbc_group",
        "rbc_individual",
        "vascular_fragments",
    ]


class Settings(BaseSettings):
    app: AppSettings
    sentry: SentrySettings
    auth: AuthSettings
    service_auth: ServiceAuthSettings
    image: ImageSettings
    kafka: KafkaSettings
    ml_model: MLModelSettings

    model_config = SettingsConfigDict(
        yaml_file=_get_yaml_path(),
        yaml_file_encoding="utf-8",
        case_sensitive=True,
        extra="allow",
    )

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        return (
            init_settings,
            env_settings,
            YamlConfigSettingsSource(settings_cls),
            dotenv_settings,
            file_secret_settings,
        )


settings = Settings()
