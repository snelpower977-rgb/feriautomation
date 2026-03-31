from __future__ import annotations

import queue
import uuid
from pathlib import Path

from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer

from config import settings
from utils.logging_utils import configure_logging
from utils.pipeline_events import push_pipeline_event


class NewFileHandler(FileSystemEventHandler):
    def __init__(self, ingest_queue, seen_cache, stats):
        super().__init__()
        self.ingest_queue = ingest_queue
        self.seen_cache = seen_cache
        self.stats = stats

    def _enqueue(self, file_path: Path) -> None:
        # Ignore transient or out-of-scope paths (common with move/rename events).
        if not file_path.exists():
            return
        try:
            file_path.resolve().relative_to(settings.input_folder.resolve())
        except Exception:
            return
        if file_path.suffix.lower() not in settings.supported_extensions:
            return
        key = str(file_path.resolve())
        if key in self.seen_cache:
            return
        self.seen_cache[key] = True
        job = {"path": key, "retries": 0, "job_id": str(uuid.uuid4())}
        try:
            self.ingest_queue.put(job, timeout=2)
            with self.stats["lock"]:
                self.stats["ingest_pending"].value += 1
            push_pipeline_event(
                self.stats,
                kind="queued",
                file=file_path.name,
                job_id=job["job_id"],
            )
        except queue.Full:
            # Remove from cache so it can be retried by rescans.
            self.seen_cache.pop(key, None)

    def on_created(self, event: FileSystemEvent) -> None:
        if event.is_directory:
            return
        self._enqueue(Path(event.src_path))

    def on_moved(self, event: FileSystemEvent) -> None:
        if event.is_directory:
            return
        self._enqueue(Path(event.dest_path))


def seed_existing_files(ingest_queue, seen_cache, stats) -> None:
    for path in settings.input_folder.iterdir():
        if (
            path.exists()
            and path.is_file()
            and path.suffix.lower() in settings.supported_extensions
        ):
            key = str(path.resolve())
            if key not in seen_cache:
                seen_cache[key] = True
                job = {"path": key, "retries": 0, "job_id": str(uuid.uuid4())}
                ingest_queue.put(job)
                with stats["lock"]:
                    stats["ingest_pending"].value += 1
                push_pipeline_event(
                    stats,
                    kind="queued",
                    file=path.name,
                    job_id=job["job_id"],
                )


def run_watcher(ingest_queue, seen_cache, stop_event, stats) -> None:
    logger = configure_logging()
    event_handler = NewFileHandler(ingest_queue, seen_cache, stats)
    observer = Observer()
    observer.schedule(event_handler, str(settings.input_folder), recursive=False)
    seed_existing_files(ingest_queue, seen_cache, stats)
    observer.start()
    logger.info("Watcher started on %s", settings.input_folder)

    try:
        while not stop_event.is_set():
            stop_event.wait(1)
    finally:
        observer.stop()
        observer.join(timeout=5)
        logger.info("Watcher stopped")
