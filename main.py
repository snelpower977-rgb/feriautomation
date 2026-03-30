from __future__ import annotations

import argparse
import multiprocessing as mp
import queue
import signal
import time

from config import ensure_directories, settings
from database import DatabaseClient
from utils.logging_utils import configure_logging
from watcher import run_watcher
from worker import run_worker


def _db_writer_loop(result_queue, stop_event) -> None:
    logger = configure_logging()
    # One writer process → one persistent connection (no pool) to save server slots.
    db_client = DatabaseClient(use_pool=False)
    buffer: list[dict] = []
    last_flush = time.time()
    logger.info("DB writer started")

    while not stop_event.is_set() or not result_queue.empty():
        try:
            item = result_queue.get(timeout=1)
            if item is None:
                break
            buffer.append(item)
        except queue.Empty:
            pass

        should_flush = (
            len(buffer) >= settings.batch_size
            or (buffer and time.time() - last_flush >= settings.batch_flush_seconds)
        )
        if should_flush:
            inserted = db_client.insert_batch(buffer)
            logger.info("DB writer inserted batch of %s", inserted)
            buffer.clear()
            last_flush = time.time()

    if buffer:
        inserted = db_client.insert_batch(buffer)
        logger.info("DB writer inserted final batch of %s", inserted)
    logger.info("DB writer stopped")


def _stats_loop(stop_event, stats) -> None:
    logger = configure_logging()
    while not stop_event.is_set():
        stop_event.wait(settings.stats_interval_seconds)
        if stop_event.is_set():
            break
        with stats["lock"]:
            processed = stats["processed"].value
            failed = stats["failed"].value
            skipped = stats["skipped"].value
        per_minute = processed / max(1, settings.stats_interval_seconds / 60)
        logger.info(
            "Stats | processed=%s failed=%s skipped=%s approx_rate=%.2f files/min",
            processed,
            failed,
            skipped,
            per_minute,
        )


def _worker_count() -> int:
    if settings.worker_count_override > 0:
        count = settings.worker_count_override
    else:
        count = mp.cpu_count() * settings.worker_multiplier
    if settings.mysql_connection_budget > 0:
        # Budget covers: N workers + 1 db-writer + 1 margin (reconnects / admin).
        max_workers = max(1, settings.mysql_connection_budget - 2)
        count = min(count, max_workers)
    return max(1, count)


def _show_stats() -> None:
    logger = configure_logging()
    count = 0
    db_client = DatabaseClient(use_pool=False)
    try:
        count = db_client.count_recent_processed(minutes=1)
    finally:
        db_client.close()
    logger.info("Processed in last 1 minute: %s", count)


def run_pipeline() -> None:
    ensure_directories()
    logger = configure_logging()

    db_client = DatabaseClient(use_pool=False)
    db_client.ensure_schema()
    db_client.close()

    manager = mp.Manager()
    ingest_queue = mp.Queue(maxsize=settings.queue_max_size)
    result_queue = mp.Queue(maxsize=settings.queue_max_size)
    stop_event = mp.Event()
    seen_cache = manager.dict()

    stats = {
        "processed": manager.Value("i", 0),
        "failed": manager.Value("i", 0),
        "skipped": manager.Value("i", 0),
        "lock": manager.Lock(),
    }

    def handle_shutdown(signum, _frame):
        logger.warning("Received signal %s, shutting down gracefully", signum)
        stop_event.set()

    signal.signal(signal.SIGINT, handle_shutdown)
    signal.signal(signal.SIGTERM, handle_shutdown)

    watcher_process = mp.Process(target=run_watcher, args=(ingest_queue, seen_cache, stop_event), name="watcher")
    db_writer_process = mp.Process(
        target=_db_writer_loop,
        args=(result_queue, stop_event),
        name="db-writer",
    )
    stats_process = mp.Process(
        target=_stats_loop,
        args=(stop_event, stats),
        name="stats",
    )

    worker_count = _worker_count()
    if settings.mysql_connection_budget > 0:
        logger.info(
            "MySQL connection budget=%s → using up to %s workers (+ 1 db-writer)",
            settings.mysql_connection_budget,
            worker_count,
        )
    workers = [
        mp.Process(
            target=run_worker,
            args=(idx + 1, ingest_queue, result_queue, stop_event, seen_cache, stats),
            name=f"worker-{idx + 1}",
        )
        for idx in range(worker_count)
    ]

    watcher_process.start()
    db_writer_process.start()
    stats_process.start()
    for process in workers:
        process.start()
    logger.info("Pipeline started with %s workers", worker_count)

    try:
        while not stop_event.is_set():
            time.sleep(1)
    finally:
        stop_event.set()
        for _ in workers:
            ingest_queue.put(None)
        result_queue.put(None)

        watcher_process.join(timeout=10)
        for process in workers:
            process.join(timeout=10)
        db_writer_process.join(timeout=10)
        stats_process.join(timeout=10)
        logger.info("Pipeline shutdown complete")


def main() -> None:
    parser = argparse.ArgumentParser(description="Bill of Lading pipeline")
    parser.add_argument(
        "command",
        nargs="?",
        default="run",
        choices=["run", "stats"],
        help="run pipeline or show stats",
    )
    args = parser.parse_args()

    if args.command == "stats":
        _show_stats()
        return

    run_pipeline()


if __name__ == "__main__":
    main()
