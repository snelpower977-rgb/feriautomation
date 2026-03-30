# B/L Ingestion Pipeline

Production-style Python pipeline that watches `input_folder/`, extracts structured data from Bill of Lading documents, and stores it in MySQL.

## Features

- Auto-detect new files with `watchdog` (`.pdf`, `.jpg`, `.jpeg`, `.png`)
- Multiprocessing workers (`CPU cores * 2` by default)
- Smart extraction path:
  - `pdfplumber` first for text PDFs
  - OCR fallback via `pytesseract` for scanned PDFs/images
- Retry support (2 attempts by default)
- Batch database writes for throughput
- Deduplication via SHA-256 hash + DB unique index
- File routing:
  - success -> `processed/`
  - failures -> `failed/`
- Rotating logs + periodic throughput stats
- Graceful shutdown on `SIGINT` / `SIGTERM`

## Project Layout

```
project/
├── main.py
├── watcher.py
├── worker.py
├── extractor.py
├── database.py
├── config.py
├── utils/
├── logs/
├── input_folder/
├── processed/
└── failed/
```

## Setup

1. Install system packages:
   - Tesseract OCR must be installed and available on PATH.
   - Poppler must be installed for `pdf2image` PDF rasterization.
2. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Copy environment file:
   ```bash
   cp .env.example .env
   ```
4. Edit MySQL credentials in `.env`.
5. Ensure target DB exists (e.g. `bl_pipeline`).

## Run

```bash
python main.py run
```

Drop files into `input_folder/`.

Optional one-shot stats (last minute):

```bash
python main.py stats
```
