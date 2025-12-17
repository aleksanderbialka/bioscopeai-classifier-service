import hashlib
import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from huggingface_hub import hf_hub_download
from keras.models import load_model
from loguru import logger

from bioscopeai_classifier_service.config import settings


@lru_cache(maxsize=1)
def load_metadata() -> dict[str, Any]:
    """Load model metadata from HuggingFace or local path."""
    logger.info("Loading model metadata...")

    if settings.ml_model.USE_HF_HUB:
        metadata_path = hf_hub_download(
            repo_id=settings.ml_model.HF_REPO_ID,
            filename="metadata.json",
            cache_dir=settings.ml_model.HF_CACHE_DIR,
            token=settings.ml_model.HF_TOKEN.get_secret_value()
            if settings.ml_model.HF_TOKEN
            else None,
        )
        logger.info(f"Metadata downloaded from HuggingFace: {metadata_path}")
    else:
        metadata_path = Path(settings.ml_model.MODEL_PATH).parent / "metadata.json"
        if not metadata_path.exists():
            logger.warning(f"Metadata not found at: {metadata_path}")
            return {}
        logger.info(f"Using local metadata: {metadata_path}")

    with Path(metadata_path).open(encoding="utf-8") as f:
        metadata = json.load(f)

    logger.info(
        f"Model: {metadata.get('model_name')} v{metadata.get('model_version')} "
        f"({metadata.get('architecture')})"
    )
    return metadata  # type: ignore[no-any-return]


def verify_checksum(file_path: Path, expected_checksum: str) -> bool:
    """Verify file integrity using SHA256 checksum."""
    logger.info("Verifying model checksum...")

    sha256_hash = hashlib.sha256()
    with Path(file_path).open("rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)

    actual_checksum = sha256_hash.hexdigest()

    if actual_checksum != expected_checksum:
        logger.error(
            f"Checksum mismatch! Expected: {expected_checksum}, Got: {actual_checksum}"
        )
        return False

    logger.info("Checksum verification passed ✓")
    return True


def download_checksum() -> str | None:
    """Download checksum file from HuggingFace."""
    if not settings.ml_model.USE_HF_HUB:
        checksum_path = Path(settings.ml_model.MODEL_PATH).parent / "checksum.sha256"
        if checksum_path.exists():
            content = checksum_path.read_text(encoding="utf-8").strip()
            return content.split()[0]
        return None

    try:
        checksum_path = hf_hub_download(
            repo_id=settings.ml_model.HF_REPO_ID,
            filename="checksum.sha256",
            cache_dir=settings.ml_model.HF_CACHE_DIR,
            token=settings.ml_model.HF_TOKEN.get_secret_value()
            if settings.ml_model.HF_TOKEN
            else None,
        )
        content = Path(checksum_path).read_text(encoding="utf-8").strip()
        return content.split()[0]
    except Exception as e:  # noqa: BLE001
        logger.warning(f"Could not download checksum: {e}")
        return None


@lru_cache(maxsize=1)
def load_classifier_model():  # type: ignore[no-untyped-def]
    """Load and cache the classification model from HuggingFace Hub or local path."""
    logger.info("Loading classification model...")

    metadata = load_metadata()

    if settings.ml_model.USE_HF_HUB:
        model_filename = metadata.get("model_file", "bioscopeai_classifier_model.keras")
        logger.info(
            f"Downloading model from HuggingFace Hub: {settings.ml_model.HF_REPO_ID}"
        )

        model_path = Path(
            hf_hub_download(
                repo_id=settings.ml_model.HF_REPO_ID,
                filename=model_filename,
                cache_dir=settings.ml_model.HF_CACHE_DIR,
                token=settings.ml_model.HF_TOKEN.get_secret_value()
                if settings.ml_model.HF_TOKEN
                else None,
            )
        )
        logger.info(f"Model downloaded to: {model_path}")
    else:
        model_path = Path(settings.ml_model.MODEL_PATH)
        if not model_path.exists():
            error_msg = f"Model file not found at: {model_path}"
            raise FileNotFoundError(error_msg)
        logger.info(f"Using local model: {model_path}")

    if settings.ml_model.VERIFY_CHECKSUM:
        expected_checksum = download_checksum()
        if expected_checksum:
            if not verify_checksum(model_path, expected_checksum):
                msg = "Model checksum verification failed!"
                raise ValueError(msg)
        else:
            logger.warning("Checksum not available, skipping verification")

    logger.info("Loading model into memory...")
    model = load_model(str(model_path))
    logger.info("Model loaded successfully ✓")

    return model
