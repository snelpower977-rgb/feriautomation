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


def _gemini_extraction_from_env() -> bool:
    """GEMINI_EXTRACTION prime ; sinon GEMINI_OVERRIDE_BL_FIELDS (legacy) ; défaut False."""
    ex = _env_truthy("GEMINI_EXTRACTION")
    if ex is not None:
        return ex
    leg = _env_truthy("GEMINI_OVERRIDE_BL_FIELDS")
    if leg is not None:
        return leg
    return False


def _deepseek_extraction_from_env() -> bool:
    """DEEPSEEK_EXTRACTION prime ; défaut False."""
    ex = _env_truthy("DEEPSEEK_EXTRACTION")
    if ex is not None:
        return ex
    return False


def _ocr_psm_modes_from_env() -> tuple[int, ...]:
    """Modes Page Segmentation Tesseract (0–13), CSV dans OCR_TESSERACT_PSM."""
    raw = (os.getenv("OCR_TESSERACT_PSM") or "6,4,11,3").replace(" ", "")
    out: list[int] = []
    for part in raw.split(","):
        if not part:
            continue
        try:
            n = int(part)
            if 0 <= n <= 13:
                out.append(n)
        except ValueError:
            continue
    return tuple(out) if out else (6, 4, 11, 3)


def _default_ocr_psm_modes() -> tuple[int, ...]:
    return _ocr_psm_modes_from_env()


@dataclass(frozen=True)
class Settings:
    input_folder: Path = Path(os.getenv("INPUT_FOLDER", BASE_DIR / "input_folder"))
    processed_folder: Path = Path(os.getenv("PROCESSED_FOLDER", BASE_DIR / "processed"))
    failed_folder: Path = Path(os.getenv("FAILED_FOLDER", BASE_DIR / "failed"))
    skipped_folder: Path = Path(os.getenv("SKIPPED_FOLDER", BASE_DIR / "skipped"))
    logs_folder: Path = Path(os.getenv("LOGS_FOLDER", BASE_DIR / "logs"))

    log_file: Path = Path(os.getenv("LOG_FILE", BASE_DIR / "logs/pipeline.log"))
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    log_max_bytes: int = int(os.getenv("LOG_MAX_BYTES", 10 * 1024 * 1024))
    log_backup_count: int = int(os.getenv("LOG_BACKUP_COUNT", 10))

    queue_max_size: int = int(os.getenv("QUEUE_MAX_SIZE", 5000))
    worker_multiplier: int = int(os.getenv("WORKER_MULTIPLIER", 2))
    worker_count_override: int = int(os.getenv("WORKER_COUNT_OVERRIDE", 0))
    retry_limit: int = int(os.getenv("RETRY_LIMIT", 2))
    # Extra retries for transient AI/API errors (timeouts, 5xx, throttling).
    ai_extra_retries: int = int(os.getenv("AI_EXTRA_RETRIES", 3))
    ai_retry_backoff_max_seconds: int = int(os.getenv("AI_RETRY_BACKOFF_MAX_SECONDS", 30))
    dedup_enabled: bool = os.getenv("DEDUP_ENABLED", "true").lower() in (
        "1",
        "true",
        "yes",
    )
    # Keep DB clean: store only successfully processed records.
    store_processed_only: bool = os.getenv("STORE_PROCESSED_ONLY", "true").lower() in (
        "1",
        "true",
        "yes",
    )
    # Cap concurrent AI-heavy workers to avoid API overload on large batches.
    ai_max_concurrent_workers: int = int(os.getenv("AI_MAX_CONCURRENT_WORKERS", 4))
    # Minimum number of non-empty structured fields required to accept extraction as processed.
    min_structured_fields: int = int(os.getenv("MIN_STRUCTURED_FIELDS", 2))
    # Mode OCR+regex: souvent moins de champs remplissables qu’avec l’IA (1 suffit si BL/booking présents).
    min_structured_fields_ocr: int = int(os.getenv("MIN_STRUCTURED_FIELDS_OCR", 1))
    # File should remain unchanged for these checks before processing starts.
    file_stability_checks: int = int(os.getenv("FILE_STABILITY_CHECKS", 4))
    file_stability_delay_seconds: float = float(os.getenv("FILE_STABILITY_DELAY_SECONDS", "0.8"))
    # Enforce per-file processing duration (scaled by file size).
    min_processing_seconds: int = int(os.getenv("MIN_PROCESSING_SECONDS", "20"))
    max_processing_seconds: int = int(os.getenv("MAX_PROCESSING_SECONDS", "45"))
    processing_scale_max_mb: int = int(os.getenv("PROCESSING_SCALE_MAX_MB", "20"))

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
    ocr_only_mode: bool = os.getenv("OCR_ONLY_MODE", "false").lower() in (
        "1",
        "true",
        "yes",
    )
    # OCR (pdf2image + Tesseract) — utilisé pour PDF scannés et images.
    ocr_pdf_dpi: int = int(os.getenv("OCR_PDF_DPI", "300"))
    # 0 = toutes les pages (attention aux gros PDF).
    ocr_max_pdf_pages: int = int(os.getenv("OCR_MAX_PDF_PAGES", "0"))
    ocr_tesseract_oem: int = int(os.getenv("OCR_TESSERACT_OEM", "3"))
    # ex. eng, fra, eng+fra (nécessite les paquets langue installés pour tesseract)
    ocr_tesseract_lang: str = os.getenv("OCR_TESSERACT_LANG", "eng").strip() or "eng"
    ocr_tesseract_psm_modes: tuple[int, ...] = field(default_factory=_default_ocr_psm_modes)
    ocr_max_tesseract_calls_per_page: int = int(os.getenv("OCR_MAX_TESSERACT_CALLS_PER_PAGE", "12"))
    ocr_min_side_upscale: int = int(os.getenv("OCR_MIN_SIDE_UPSCALE", "1400"))
    ocr_upscale_target_min_side: int = int(os.getenv("OCR_UPSCALE_TARGET_MIN_SIDE", "1800"))
    ocr_upscale_max_side: int = int(os.getenv("OCR_UPSCALE_MAX_SIDE", "4000"))

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

    gemini_api_key: str = os.getenv("GEMINI_API_KEY", "").strip()
    gemini_extraction_enabled: bool = field(default_factory=_gemini_extraction_from_env)
    # false = jamais de regex pour les champs B/L (IA ou champs vides). true = repli regex si pas de clé / IA off.
    gemini_regex_fallback: bool = os.getenv("GEMINI_REGEX_FALLBACK", "false").lower() in (
        "1",
        "true",
        "yes",
    )
    gemini_model: str = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    gemini_max_input_chars: int = int(os.getenv("GEMINI_MAX_INPUT_CHARS", "24000"))
    gemini_timeout_seconds: int = int(os.getenv("GEMINI_TIMEOUT_SECONDS", "180"))
    gemini_use_vision: bool = os.getenv("GEMINI_USE_VISION", "true").lower() in (
        "1",
        "true",
        "yes",
    )
    gemini_vision_max_pages: int = int(os.getenv("GEMINI_VISION_MAX_PAGES", "3"))
    gemini_vision_max_side: int = int(os.getenv("GEMINI_VISION_MAX_SIDE", "1800"))
    gemini_vision_dpi: int = int(os.getenv("GEMINI_VISION_DPI", "160"))
    gemini_vision_jpeg_quality: int = int(os.getenv("GEMINI_VISION_JPEG_QUALITY", "82"))
    gemini_vision_extensions: tuple[str, ...] = (
        ".pdf",
        ".png",
        ".jpg",
        ".jpeg",
        ".webp",
    )

    deepseek_api_key: str = os.getenv("DEEPSEEK_API_KEY", "").strip()
    deepseek_extraction_enabled: bool = field(default_factory=_deepseek_extraction_from_env)
    # false = jamais de regex pour les champs B/L (IA ou champs vides). true = repli regex si pas de clé / IA off.
    deepseek_regex_fallback: bool = os.getenv("DEEPSEEK_REGEX_FALLBACK", "false").lower() in (
        "1",
        "true",
        "yes",
    )
    deepseek_model: str = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
    deepseek_max_input_chars: int = int(os.getenv("DEEPSEEK_MAX_INPUT_CHARS", "24000"))
    deepseek_timeout_seconds: int = int(os.getenv("DEEPSEEK_TIMEOUT_SECONDS", "180"))


settings = Settings()


def ensure_directories() -> None:
    for path in (
        settings.input_folder,
        settings.processed_folder,
        settings.failed_folder,
        settings.skipped_folder,
        settings.logs_folder,
    ):
        path.mkdir(parents=True, exist_ok=True)
