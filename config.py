import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv


BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")


def _env_truthy(name: str) -> bool | None:
    v = os.getenv(name)
    if v is None:
        return None
    return v.strip().lower() in ("1", "true", "yes")


def _openai_extraction_from_env() -> bool:
    """OPENAI_EXTRACTION prime ; sinon OPENAI_OVERRIDE_BL_FIELDS (legacy) ; défaut True."""
    ex = _env_truthy("OPENAI_EXTRACTION")
    if ex is not None:
        return ex
    leg = _env_truthy("OPENAI_OVERRIDE_BL_FIELDS")
    if leg is not None:
        return leg
    return True


@dataclass(frozen=True)
class Settings:
    input_folder: Path = Path(os.getenv("INPUT_FOLDER", BASE_DIR / "input_folder"))
    processed_folder: Path = Path(os.getenv("PROCESSED_FOLDER", BASE_DIR / "processed"))
    failed_folder: Path = Path(os.getenv("FAILED_FOLDER", BASE_DIR / "failed"))
    logs_folder: Path = Path(os.getenv("LOGS_FOLDER", BASE_DIR / "logs"))

    log_file: Path = Path(os.getenv("LOG_FILE", BASE_DIR / "logs/pipeline.log"))
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    log_max_bytes: int = int(os.getenv("LOG_MAX_BYTES", 10 * 1024 * 1024))
    log_backup_count: int = int(os.getenv("LOG_BACKUP_COUNT", 10))

    queue_max_size: int = int(os.getenv("QUEUE_MAX_SIZE", 5000))
    worker_multiplier: int = int(os.getenv("WORKER_MULTIPLIER", 2))
    worker_count_override: int = int(os.getenv("WORKER_COUNT_OVERRIDE", 0))
    retry_limit: int = int(os.getenv("RETRY_LIMIT", 2))

    batch_size: int = int(os.getenv("DB_BATCH_SIZE", 50))
    batch_flush_seconds: int = int(os.getenv("DB_BATCH_FLUSH_SECONDS", 2))

    mysql_host: str = os.getenv("MYSQL_HOST", "127.0.0.1")
    mysql_port: int = int(os.getenv("MYSQL_PORT", 3306))
    mysql_user: str = os.getenv("MYSQL_USER", "root")
    mysql_password: str = os.getenv("MYSQL_PASSWORD", "")
    mysql_database: str = os.getenv("MYSQL_DATABASE", "bl_pipeline")
    mysql_pool_name: str = os.getenv("MYSQL_POOL_NAME", "bl_pool")
    # Only used if DatabaseClient(use_pool=True). Writer defaults to a single connection instead.
    mysql_pool_size: int = int(os.getenv("MYSQL_POOL_SIZE", 2))
    # Cap total pipeline DB connections: workers (each 1) + writer (1) + ~1 reserve. 0 = unlimited.
    mysql_connection_budget: int = int(os.getenv("MYSQL_CONNECTION_BUDGET", 0))

    supported_extensions: tuple[str, ...] = (".pdf", ".jpg", ".jpeg", ".png")
    min_pdf_text_chars: int = int(os.getenv("MIN_PDF_TEXT_CHARS", 100))

    stats_interval_seconds: int = int(os.getenv("STATS_INTERVAL_SECONDS", 60))

    monitor_enabled: bool = os.getenv("MONITOR_ENABLED", "true").lower() in (
        "1",
        "true",
        "yes",
    )
    monitor_host: str = os.getenv("MONITOR_HOST", "127.0.0.1")
    monitor_port: int = int(os.getenv("MONITOR_PORT", 8765))
    monitor_broadcast_seconds: float = float(os.getenv("MONITOR_BROADCAST_SECONDS", "1"))

    openai_api_key: str = os.getenv("OPENAI_API_KEY", "").strip()
    openai_extraction_enabled: bool = field(default_factory=_openai_extraction_from_env)
    # false = jamais de regex pour les champs B/L (IA ou champs vides). true = repli regex si pas de clé / IA off.
    openai_regex_fallback: bool = os.getenv("OPENAI_REGEX_FALLBACK", "false").lower() in (
        "1",
        "true",
        "yes",
    )
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    openai_max_input_chars: int = int(os.getenv("OPENAI_MAX_INPUT_CHARS", "24000"))
    # Vision: render PDF / image pages and send with OCR text (better layout than text-only).
    openai_use_vision: bool = os.getenv("OPENAI_USE_VISION", "true").lower() in (
        "1",
        "true",
        "yes",
    )
    openai_vision_max_pages: int = int(os.getenv("OPENAI_VISION_MAX_PAGES", "3"))
    openai_vision_max_side: int = int(os.getenv("OPENAI_VISION_MAX_SIDE", "1800"))
    openai_vision_dpi: int = int(os.getenv("OPENAI_VISION_DPI", "160"))
    openai_vision_jpeg_quality: int = int(os.getenv("OPENAI_VISION_JPEG_QUALITY", "82"))
    openai_vision_extensions: tuple[str, ...] = (
        ".pdf",
        ".png",
        ".jpg",
        ".jpeg",
        ".webp",
    )


settings = Settings()


def ensure_directories() -> None:
    for path in (
        settings.input_folder,
        settings.processed_folder,
        settings.failed_folder,
        settings.logs_folder,
    ):
        path.mkdir(parents=True, exist_ok=True)
