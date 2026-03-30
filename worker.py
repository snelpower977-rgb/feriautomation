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


def _build_record(file_path: Path, file_hash: str, raw_text: str, status: str) -> dict:
    if status == "processed" and raw_text:
        parsed = extract_structured_fields(raw_text, file_path=file_path)
    else:
        parsed = empty_structured_fields()
    return {
        "id": str(uuid.uuid4()),
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
        retries = int(job.get("retries", 0))
        start = time.perf_counter()

        try:
            if not file_path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")

            wait_for_file_stability(file_path)
            file_hash = compute_file_hash(file_path)
            if db_client.file_hash_exists(file_hash):
                logger.info("Worker-%s skipped duplicate %s", worker_id, file_path.name)
                move_with_unique_name(file_path, settings.processed_folder)
                with stats["lock"]:
                    stats["skipped"].value += 1
                continue

            raw_text = extract_text(file_path)
            if not raw_text:
                raise ValueError("No extractable text from file")

            record = _build_record(file_path, file_hash, raw_text, "processed")
            result_queue.put(record, timeout=2)
            with stats["lock"]:
                stats["result_pending"].value += 1
            move_with_unique_name(file_path, settings.processed_folder)

            elapsed = time.perf_counter() - start
            with stats["lock"]:
                stats["processed"].value += 1
            logger.info("Worker-%s processed %s in %.2fs", worker_id, file_path.name, elapsed)
        except Exception as exc:  # pylint: disable=broad-except
            if retries < settings.retry_limit:
                job["retries"] = retries + 1
                ingest_queue.put(job)
                with stats["lock"]:
                    stats["ingest_pending"].value += 1
                backoff = 2 ** retries
                logger.warning(
                    "Worker-%s retrying %s (attempt %s/%s) after error: %s",
                    worker_id,
                    file_path.name,
                    retries + 1,
                    settings.retry_limit,
                    exc,
                )
                time.sleep(backoff)
                continue

            try:
                file_hash = compute_file_hash(file_path) if file_path.exists() else "missing"
                failed_record = _build_record(file_path, file_hash, str(exc), "failed")
                result_queue.put(failed_record, timeout=2)
                with stats["lock"]:
                    stats["result_pending"].value += 1
                if file_path.exists():
                    move_with_unique_name(file_path, settings.failed_folder)
            except Exception as inner_exc:  # pylint: disable=broad-except
                logger.exception("Worker-%s failed while handling error: %s", worker_id, inner_exc)
            finally:
                with stats["lock"]:
                    stats["failed"].value += 1
                logger.exception("Worker-%s failed processing %s: %s", worker_id, file_path, exc)
        finally:
            seen_cache.pop(str(file_path.resolve()), None)

    logger.info("Worker-%s stopped", worker_id)
