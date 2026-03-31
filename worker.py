from __future__ import annotations

import queue
import time
import uuid
from pathlib import Path

from config import settings
from extractor import empty_structured_fields, extract_structured_fields, extract_text
from database import DatabaseClient
from utils.file_utils import compute_file_hash, move_with_unique_name, wait_for_file_stability
from utils.logging_utils import configure_logging
from utils.pipeline_events import push_pipeline_event

_LARGE_FILE_BYTES = 4 * 1024 * 1024
_LARGE_FILE_EXTRA_RETRIES = 2


def _target_processing_seconds(file_size_bytes: int) -> float:
    min_s = max(1, settings.min_processing_seconds)
    max_s = max(min_s, settings.max_processing_seconds)
    max_mb = max(1, settings.processing_scale_max_mb)
    size_mb = max(0.0, file_size_bytes / (1024 * 1024))
    ratio = min(1.0, size_mb / float(max_mb))
    return min_s + (max_s - min_s) * ratio


def _is_transient_ai_error(message: str) -> bool:
    m = (message or "").lower()
    markers = (
        "timeout",
        "timed out",
        "temporarily",
        "rate limit",
        "too many requests",
        "service unavailable",
        "connection reset",
        "connection aborted",
        "bad gateway",
        "gateway timeout",
        " 502",
        " 503",
        " 504",
    )
    return any(k in m for k in markers)


def _is_low_quality_error(message: str) -> bool:
    m = (message or "").lower()
    return (
        "low-quality" in m
        or "neither bl_number nor booking_number" in m
        or "extraction low-quality" in m
    )


def _has_minimum_bl_identity(record: dict) -> bool:
    """A processed BL must expose at least BL number or booking number."""
    bl = str(record.get("bl_number") or "").strip()
    booking = str(record.get("booking_number") or "").strip()
    return bool(bl or booking)


def _non_empty_structured_count(record: dict) -> int:
    keys = (
        "bl_number",
        "booking_number",
        "vessel",
        "port_loading",
        "port_discharge",
        "weight",
        "shipper",
        "consignee",
    )
    return sum(1 for k in keys if str(record.get(k) or "").strip())


def _effective_min_structured_fields() -> int:
    if settings.ocr_only_mode:
        return max(1, settings.min_structured_fields_ocr)
    return max(1, settings.min_structured_fields)


def _is_structured_extraction_acceptable(record: dict) -> bool:
    if not _has_minimum_bl_identity(record):
        return False
    need = _effective_min_structured_fields()
    return _non_empty_structured_count(record) >= need


def _build_record(
    file_path: Path,
    file_hash: str,
    raw_text: str,
    status: str,
    *,
    job_id: str = "",
) -> dict:
    if status == "processed" and raw_text:
        parsed = extract_structured_fields(raw_text, file_path=file_path)
    else:
        parsed = empty_structured_fields()
    return {
        "id": str(uuid.uuid4()),
        "job_id": job_id,
        "source_path": str(file_path),
        "file_name": file_path.name,
        "file_hash": file_hash,
        "bl_number": parsed.get("bl_number"),
        "booking_number": parsed.get("booking_number"),
        "vessel": parsed.get("vessel"),
        "port_loading": parsed.get("port_loading"),
        "port_discharge": parsed.get("port_discharge"),
        "weight": parsed.get("weight"),
        "shipper": parsed.get("shipper"),
        "consignee": parsed.get("consignee"),
        "raw_text": raw_text,
        "status": status,
    }


def run_worker(worker_id, ingest_queue, result_queue, stop_event, seen_cache, stats) -> None:
    logger = configure_logging()
    db_client = DatabaseClient(use_pool=False)
    logger.info("Worker-%s started", worker_id)
    while not stop_event.is_set():
        try:
            job = ingest_queue.get(timeout=1)
        except queue.Empty:
            continue

        if job is None:
            break

        with stats["lock"]:
            stats["ingest_pending"].value = max(0, stats["ingest_pending"].value - 1)

        file_path = Path(job["path"])
        job_id = str(job.get("job_id") or "")
        retries = int(job.get("retries", 0))
        start = time.perf_counter()
        file_size = file_path.stat().st_size if file_path.exists() else 0
        max_retries = settings.retry_limit + (
            _LARGE_FILE_EXTRA_RETRIES if file_size >= _LARGE_FILE_BYTES else 0
        )

        push_pipeline_event(
            stats,
            kind="processing",
            file=file_path.name,
            job_id=job_id,
            worker_id=worker_id,
        )

        try:
            if not file_path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")

            wait_for_file_stability(
                file_path,
                checks=max(2, settings.file_stability_checks),
                delay=max(0.2, settings.file_stability_delay_seconds),
            )
            file_hash = compute_file_hash(file_path)
            job["file_hash"] = file_hash
            if settings.dedup_enabled and db_client.file_hash_exists(file_hash):
                logger.info("Worker-%s skipped duplicate %s", worker_id, file_path.name)
                push_pipeline_event(
                    stats,
                    kind="skipped_duplicate",
                    file=file_path.name,
                    job_id=job_id,
                    worker_id=worker_id,
                )
                move_with_unique_name(file_path, settings.skipped_folder)
                with stats["lock"]:
                    stats["skipped"].value += 1
                continue

            raw_text = extract_text(file_path)
            if not raw_text:
                raise ValueError("No extractable text from file")

            record = _build_record(
                file_path, file_hash, raw_text, "processed", job_id=job_id
            )
            if not _is_structured_extraction_acceptable(record):
                present = _non_empty_structured_count(record)
                engine = "OCR" if settings.ocr_only_mode else "AI"
                need = _effective_min_structured_fields()
                raise ValueError(
                    f"{engine} extraction low-quality: only {present} structured fields (< {need})"
                )
            # Keep runtime proportional to document volume to let AI fully settle.
            elapsed_before_queue = time.perf_counter() - start
            target_seconds = _target_processing_seconds(file_size)
            if elapsed_before_queue < target_seconds:
                time.sleep(target_seconds - elapsed_before_queue)

            result_queue.put(record, timeout=2)
            with stats["lock"]:
                stats["result_pending"].value += 1

            elapsed = time.perf_counter() - start
            push_pipeline_event(
                stats,
                kind="done",
                file=file_path.name,
                job_id=job_id,
                worker_id=worker_id,
                seconds=round(elapsed, 2),
            )
            logger.info("Worker-%s processed %s in %.2fs", worker_id, file_path.name, elapsed)
        except FileNotFoundError as exc:
            processed_twin = settings.processed_folder / file_path.name
            if processed_twin.exists():
                push_pipeline_event(
                    stats,
                    kind="already_done",
                    file=file_path.name,
                    job_id=job_id,
                    worker_id=worker_id,
                )
                logger.info(
                    "Worker-%s detected already finalized file %s in processed folder; skip retry",
                    worker_id,
                    file_path.name,
                )
                continue

            # Some files appear a little later (copy/write not fully committed yet).
            # Retry missing files before finally marking them as skipped.
            if retries < max_retries:
                job["retries"] = retries + 1
                push_pipeline_event(
                    stats,
                    kind="retry",
                    file=file_path.name,
                    job_id=job_id,
                    worker_id=worker_id,
                    attempt=retries + 1,
                    error="file not found yet",
                )
                ingest_queue.put(job)
                with stats["lock"]:
                    stats["ingest_pending"].value += 1
                backoff = 2 ** retries
                logger.warning(
                    "Worker-%s retrying missing file %s (attempt %s/%s)",
                    worker_id,
                    file_path.name,
                    retries + 1,
                    max_retries,
                )
                time.sleep(backoff)
                continue

            push_pipeline_event(
                stats,
                kind="missing_after_retries",
                file=file_path.name,
                job_id=job_id,
                worker_id=worker_id,
            )
            logger.warning(
                "Worker-%s skipped missing file %s after %s retries: %s",
                worker_id,
                file_path,
                retries,
                exc,
            )
            continue
        except Exception as exc:  # pylint: disable=broad-except
            msg = str(exc)
            no_retry = "AI provider misconfigured" in msg or "API key not valid" in msg
            transient_ai = _is_transient_ai_error(msg)
            low_quality = _is_low_quality_error(msg)
            effective_max_retries = max_retries + (
                max(0, settings.ai_extra_retries) if (transient_ai or low_quality) else 0
            )
            if retries < effective_max_retries and not no_retry:
                job["retries"] = retries + 1
                push_pipeline_event(
                    stats,
                    kind="retry",
                    file=file_path.name,
                    job_id=job_id,
                    worker_id=worker_id,
                    attempt=retries + 1,
                    error=str(exc)[:180],
                )
                ingest_queue.put(job)
                with stats["lock"]:
                    stats["ingest_pending"].value += 1
                backoff = min(2 ** retries, max(1, settings.ai_retry_backoff_max_seconds))
                logger.warning(
                    "Worker-%s retrying %s (attempt %s/%s) after error: %s",
                    worker_id,
                    file_path.name,
                    retries + 1,
                    effective_max_retries,
                    exc,
                )
                time.sleep(backoff)
                continue

            try:
                if file_path.exists():
                    move_with_unique_name(file_path, settings.failed_folder)
                push_pipeline_event(
                    stats,
                    kind="failed_no_db",
                    file=file_path.name,
                    job_id=job_id,
                    worker_id=worker_id,
                    error=str(exc)[:220],
                )
                if not settings.store_processed_only:
                    file_hash = str(job.get("file_hash") or "")
                    if not file_hash:
                        file_hash = compute_file_hash(file_path) if file_path.exists() else "missing"
                    failed_record = _build_record(
                        file_path, file_hash, str(exc), "failed", job_id=job_id
                    )
                    result_queue.put(failed_record, timeout=2)
                    with stats["lock"]:
                        stats["result_pending"].value += 1
            except Exception as inner_exc:  # pylint: disable=broad-except
                logger.exception("Worker-%s failed while handling error: %s", worker_id, inner_exc)
            finally:
                with stats["lock"]:
                    stats["failed"].value += 1
                logger.exception("Worker-%s failed processing %s: %s", worker_id, file_path, exc)
        finally:
            # Keep seen marker while file still exists in input folder to avoid
            # duplicate enqueues of the same path during processing/retries.
            # Once moved out (processed/failed), release marker so a future file
            # with the same name can be re-ingested.
            if not file_path.exists():
                seen_cache.pop(str(file_path.resolve()), None)

    logger.info("Worker-%s stopped", worker_id)
