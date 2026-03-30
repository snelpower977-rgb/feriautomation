import hashlib
import os
import shutil
import time
from pathlib import Path


def compute_file_hash(file_path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with file_path.open("rb") as source:
        while True:
            chunk = source.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def wait_for_file_stability(file_path: Path, checks: int = 3, delay: float = 0.5) -> None:
    last_size = -1
    stable_count = 0
    while stable_count < checks:
        size = os.path.getsize(file_path)
        if size == last_size:
            stable_count += 1
        else:
            stable_count = 0
            last_size = size
        time.sleep(delay)


def move_with_unique_name(source: Path, destination_folder: Path) -> Path:
    destination_folder.mkdir(parents=True, exist_ok=True)
    target = destination_folder / source.name
    if not target.exists():
        shutil.move(str(source), str(target))
        return target

    stem = source.stem
    suffix = source.suffix
    counter = 1
    while True:
        candidate = destination_folder / f"{stem}_{counter}{suffix}"
        if not candidate.exists():
            shutil.move(str(source), str(candidate))
            return candidate
        counter += 1
