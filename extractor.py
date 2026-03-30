import re
from pathlib import Path

import pdfplumber
import pytesseract
from pdf2image import convert_from_path
from PIL import Image

from config import settings


FIELD_PATTERNS = {
    "bl_number": [
        r"(?:B\/?L(?:\s*No\.?| Number)?)[\s:]*([A-Z0-9\-\/]{5,})",
        r"(?:Bill\s+of\s+Lading(?:\s*No\.?| Number)?)[\s:]*([A-Z0-9\-\/]{5,})",
    ],
    "booking_number": [
        r"(?:Booking(?:\s*No\.?| Number)?)[\s:]*([A-Z0-9\-\/]{5,})",
        r"(?:Bk(?:\s*No\.?| Number)?)[\s:]*([A-Z0-9\-\/]{5,})",
    ],
    "vessel": [
        r"(?:Vessel(?:\s*Name)?)[\s:]*([A-Z0-9 \-\/]{3,})",
    ],
    "port_loading": [
        r"(?:Port\s+of\s+Loading|POL)[\s:]*([A-Z][A-Z \-\/]{2,})",
    ],
    "port_discharge": [
        r"(?:Port\s+of\s+Discharge|POD)[\s:]*([A-Z][A-Z \-\/]{2,})",
    ],
    "weight": [
        r"(?:Gross\s+Weight|Weight|WGT)[\s:]*([0-9][0-9,\. ]{0,20}(?:KG|KGS|LBS|MT|TONS?)?)",
    ],
}


def _normalize_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _extract_text_from_pdf(pdf_path: Path) -> str:
    pages = []
    with pdfplumber.open(str(pdf_path)) as pdf:
        for page in pdf.pages:
            pages.append(page.extract_text() or "")
    text = "\n".join(pages).strip()
    if len(text) >= settings.min_pdf_text_chars:
        return _normalize_text(text)
    return ""


def _ocr_image(image: Image.Image) -> str:
    return pytesseract.image_to_string(image)


def _extract_text_from_scanned_pdf(pdf_path: Path) -> str:
    images = convert_from_path(str(pdf_path), dpi=250)
    text_parts = [_ocr_image(image) for image in images]
    return _normalize_text("\n".join(text_parts))


def _extract_text_from_image(image_path: Path) -> str:
    with Image.open(image_path) as image:
        text = _ocr_image(image)
    return _normalize_text(text)


def extract_text(file_path: Path) -> str:
    suffix = file_path.suffix.lower()
    if suffix == ".pdf":
        text = _extract_text_from_pdf(file_path)
        if text:
            return text
        return _extract_text_from_scanned_pdf(file_path)
    return _extract_text_from_image(file_path)


def _extract_with_patterns(text: str, patterns: list[str]) -> str | None:
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE | re.MULTILINE)
        if match:
            return match.group(1).strip(" :;-")
    return None


def parse_bl_fields(raw_text: str) -> dict[str, str | None]:
    compact_text = _normalize_text(raw_text)
    output: dict[str, str | None] = {}
    for field, patterns in FIELD_PATTERNS.items():
        output[field] = _extract_with_patterns(compact_text, patterns)

    # Fallback strategy: derive numbers if labels were missed.
    if not output["bl_number"]:
        fallback = re.search(r"\b([A-Z]{2,5}[0-9]{6,12})\b", compact_text)
        if fallback:
            output["bl_number"] = fallback.group(1)

    if not output["booking_number"]:
        fallback = re.search(r"\b(BK|BOOK|BKG)[A-Z0-9\-]{4,}\b", compact_text, flags=re.IGNORECASE)
        if fallback:
            output["booking_number"] = fallback.group(0)

    return output
