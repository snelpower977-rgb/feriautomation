import re
import unicodedata
from pathlib import Path

import pdfplumber
import pytesseract
from pdf2image import convert_from_path
from PIL import Image, ImageFilter, ImageOps, ImageStat

from config import settings


FIELD_PATTERNS = {
    "bl_number": [
        r"(?:B\/?L(?:\s*No\.?| Number)?)[\s:]*([A-Z0-9\-\/]{5,})",
        r"(?:Bill\s+of\s+Lading(?:\s*No\.?| Number)?)[\s:]*([A-Z0-9\-\/]{5,})",
        r"(?:Connaissement|SWB)(?:\s*N[°os]\.?)?[\s:]*([A-Z0-9\-\/]{5,})",
        r"(?:BL|B\.L\.?)\s*(?:N[°o]|#|No\.?)\s*[:\s]*([A-Z0-9\-\/]{5,})",
    ],
    "booking_number": [
        r"(?:Booking(?:\s*No\.?| Number)?)[\s:]*([A-Z0-9\-\/]{5,})",
        r"(?:Bk(?:\s*No\.?| Number)?)[\s:]*([A-Z0-9\-\/]{5,})",
        r"(?:R[ée]servation|Rés\.?)(?:\s*N[°os]\.?)?[\s:]*([A-Z0-9\-\/]{5,})",
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
    text = unicodedata.normalize("NFC", text)
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


_OCR_HINTS = (
    "b/l",
    "bill of lading",
    "booking",
    "consignee",
    "shipper",
    "vessel",
    "gross weight",
    "port of loading",
    "port of discharge",
    "pol",
    "pod",
    "connaissement",
)


def _ocr_quality_score(raw: str) -> float:
    """Préfère un texte lisible (mots-clés B/L) plutôt que du bruit long."""
    s = (raw or "").strip()
    if not s:
        return 0.0
    n = len(s)
    letters = sum(1 for c in s if c.isalpha())
    digits = sum(1 for c in s if c.isdigit())
    alnum_ratio = (letters + digits) / max(n, 1)
    wordish = len(re.findall(r"[A-Za-zÀ-ÿ]{3,}", s))
    low = s.lower()
    hint = sum(3.0 for k in _OCR_HINTS if k in low)
    # Pénalise les lignes presque uniquement non-alphanum (scanner bruité).
    noise_penalty = max(0.0, 0.45 - alnum_ratio) * 120.0
    return n * 0.08 + wordish * 2.5 + hint + alnum_ratio * 40.0 - noise_penalty


def _tesseract(img: Image.Image, psm: int) -> str:
    oem = max(0, min(3, settings.ocr_tesseract_oem))
    cfg = f"--oem {oem} --psm {psm}"
    return pytesseract.image_to_string(
        img, config=cfg, lang=settings.ocr_tesseract_lang
    )


def _maybe_invert_if_dark(gray: Image.Image) -> Image.Image:
    """Fonds sombres / capture écran : texte clair sur noir."""
    mean = ImageStat.Stat(gray).mean[0]
    if mean < 118:
        return ImageOps.invert(gray)
    return gray


def _resize_min_side(img: Image.Image, target_min: int, max_side: int) -> Image.Image:
    w, h = img.size
    m = min(w, h)
    if m >= target_min:
        return img
    scale = target_min / max(m, 1)
    nw, nh = int(w * scale), int(h * scale)
    if max(nw, nh) > max_side:
        r = max_side / max(nw, nh)
        nw, nh = int(nw * r), int(nh * r)
    if nw < 1 or nh < 1:
        return img
    return img.resize((nw, nh), Image.Resampling.LANCZOS)


def _binary_threshold(gray: Image.Image, cutoff: int = 2, thresh: int = 155) -> Image.Image:
    g = ImageOps.autocontrast(gray, cutoff=cutoff)
    return g.point(lambda p: 255 if p > thresh else 0)


def _preprocess_variants(rgb_or_gray: Image.Image) -> list[Image.Image]:
    """Génère plusieurs versions pour Tesseract (contraste, netteté, binarisation)."""
    base = rgb_or_gray.convert("L")
    base = _maybe_invert_if_dark(base)

    out: list[Image.Image] = []
    # 1) Nettoyage doux
    v_soft = ImageOps.autocontrast(base)
    v_soft = v_soft.filter(ImageFilter.MedianFilter(size=3))
    out.append(v_soft)

    # 2) Renforcement contours
    sharp = ImageOps.autocontrast(base).filter(
        ImageFilter.UnsharpMask(radius=1.2, percent=130, threshold=2)
    )
    out.append(sharp)

    # 3) Binarisation (seuils voisins pour scans dégradés)
    out.append(_binary_threshold(base, 2, 150))
    out.append(_binary_threshold(base, 2, 168))

    # 5) Agrandissement cible si image petite (meilleur taux sur petites captures)
    if min(base.size) < settings.ocr_min_side_upscale:
        up = _resize_min_side(
            v_soft,
            settings.ocr_upscale_target_min_side,
            settings.ocr_upscale_max_side,
        )
        if up is not v_soft:
            out.append(up)

    return out[:6]


def _ocr_image(image: Image.Image) -> str:
    variants = _preprocess_variants(image)
    psms = list(settings.ocr_tesseract_psm_modes) or [6]
    cap = max(4, settings.ocr_max_tesseract_calls_per_page)
    best_text = ""
    best_score = -1.0
    calls = 0

    for v in variants:
        for psm in psms:
            if calls >= cap:
                break
            try:
                raw = _tesseract(v, psm)
            except Exception:
                raw = ""
            calls += 1
            sc = _ocr_quality_score(raw)
            if sc > best_score:
                best_score = sc
                best_text = raw
        if calls >= cap:
            break

    return best_text or ""


def _extract_text_from_scanned_pdf(pdf_path: Path) -> str:
    dpi = max(72, settings.ocr_pdf_dpi)
    max_p = settings.ocr_max_pdf_pages
    if max_p > 0:
        images = convert_from_path(
            str(pdf_path), dpi=dpi, first_page=1, last_page=max_p
        )
    else:
        images = convert_from_path(str(pdf_path), dpi=dpi)
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


def parse_bl_fields(
    raw_text: str,
    *,
    loose_regex_fallback: bool = True,
) -> dict[str, str | None]:
    compact_text = _normalize_text(raw_text)
    output: dict[str, str | None] = {}
    for field, patterns in FIELD_PATTERNS.items():
        output[field] = _extract_with_patterns(compact_text, patterns)

    # Repli agressif = beaucoup de faux positifs (ex. autre ref. transport / segment alphabétique proche).
    # Désactivé quand OpenAI est utilisé: voir extract_structured_fields.
    if loose_regex_fallback:
        if not output["bl_number"]:
            fallback = re.search(r"\b([A-Z]{2,5}[0-9]{6,12})\b", compact_text)
            if fallback:
                output["bl_number"] = fallback.group(1)

        if not output["booking_number"]:
            fallback = re.search(
                r"\b(BK|BOOK|BKG)[A-Z0-9\-]{4,}\b", compact_text, flags=re.IGNORECASE
            )
            if fallback:
                output["booking_number"] = fallback.group(0)

    output["shipper"] = None
    output["consignee"] = None
    return output


def empty_structured_fields() -> dict[str, str | None]:
    """Champs structurés tous absents (pipeline IA exclusive sans repli regex)."""
    return {
        "bl_number": None,
        "booking_number": None,
        "vessel": None,
        "port_loading": None,
        "port_discharge": None,
        "weight": None,
        "shipper": None,
        "consignee": None,
    }


def extract_structured_fields(
    raw_text: str, *, file_path: Path | None = None
) -> dict[str, str | None]:
    """IA DeepSeek/Gemini/OpenAI puis regex optionnel selon configuration."""
    if settings.ocr_only_mode:
        return parse_bl_fields(raw_text, loose_regex_fallback=True)

    deepseek_key_ok = bool((settings.deepseek_api_key or "").strip())
    if settings.deepseek_extraction_enabled and deepseek_key_ok:
        from deepseek_extract import extract_bl_with_deepseek

        ai = extract_bl_with_deepseek(file_path, raw_text)
        return _fields_from_ai_only(ai)

    gem_key_ok = bool((settings.gemini_api_key or "").strip())
    if settings.gemini_extraction_enabled and gem_key_ok:
        from gemini_extract import extract_bl_with_gemini

        ai = extract_bl_with_gemini(file_path, raw_text)
        if ai.get("_error") == "API_KEY_INVALID":
            raise ValueError("AI provider misconfigured: Gemini API key invalid")
        return _fields_from_ai_only(ai)

    key_ok = bool((settings.openai_api_key or "").strip())
    if settings.openai_extraction_enabled and key_ok:
        from openai_extract import extract_bl_with_openai

        ai = extract_bl_with_openai(file_path, raw_text)
        return _fields_from_ai_only(ai)

    if settings.gemini_extraction_enabled and settings.gemini_regex_fallback:
        return parse_bl_fields(raw_text, loose_regex_fallback=True)
    if settings.openai_extraction_enabled and settings.openai_regex_fallback:
        return parse_bl_fields(raw_text, loose_regex_fallback=True)
    if settings.deepseek_extraction_enabled and settings.deepseek_regex_fallback:
        return parse_bl_fields(raw_text, loose_regex_fallback=True)
    return empty_structured_fields()


def _fields_from_ai_only(ai: dict) -> dict[str, str | None]:
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
    out: dict[str, str | None] = {}
    for k in keys:
        v = ai.get(k)
        if v is None:
            out[k] = None
            continue
        s = str(v).strip()
        out[k] = s if s else None
    return out
