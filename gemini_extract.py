from __future__ import annotations

import base64
import io
import json
import logging
import re
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

from PIL import Image

from config import settings

log = logging.getLogger("bl_pipeline")

_JSON_SANITIZE = re.compile(r"^[\s`]*json\s*", re.IGNORECASE)
_AI_KEYS = (
    "bl_number",
    "booking_number",
    "port_loading",
    "port_discharge",
    "weight",
    "shipper",
    "consignee",
    "vessel",
)


def extract_bl_with_gemini(file_path: Path | None, raw_text: str) -> dict[str, Any]:
    key = (settings.gemini_api_key or "").strip()
    if not key:
        return {}
    file_size = file_path.stat().st_size if file_path and file_path.exists() else 0
    ocr_budget = max(4000, settings.gemini_max_input_chars)
    if file_size >= 4 * 1024 * 1024:
        # Large files benefit from less noisy OCR context when vision is provided.
        ocr_budget = min(ocr_budget, 12000)
    sample = _sample_text_for_ai(raw_text, ocr_budget)
    instructions = _user_instructions_only()

    # Prefer vision first.
    if (
        file_path
        and settings.gemini_use_vision
        and file_path.exists()
        and file_path.suffix.lower() in settings.gemini_vision_extensions
    ):
        try:
            image_parts = _document_image_parts_for_gemini(file_path)
            if image_parts:
                prompt_parts: list[Any] = [
                    _system_prompt(),
                    _user_intro_vision(len(image_parts)),
                    instructions,
                ]
                prompt_parts.extend(image_parts)
                prompt_parts.append("Texte OCR (secours, souvent imparfait):\n---\n" + sample + "\n---")
                result = _complete_and_parse(prompt_parts, key=key)
                if result and _has_minimum_identity(result):
                    return result
                if result:
                    repaired = _repair_identity_pass(
                        key=key,
                        image_parts=image_parts,
                        sample_text=sample,
                        previous=result,
                    )
                    if repaired:
                        return repaired
                log.warning("Gemini vision response empty; retrying text-only")
        except Exception as exc:  # pylint: disable=broad-except
            log.warning("Gemini vision path failed (%s); using text-only", exc)

    result = _complete_and_parse([_system_prompt(), _user_block_text(sample)], key=key)
    if result and _has_minimum_identity(result):
        return result
    if result:
        repaired = _repair_identity_pass(
            key=key,
            image_parts=[],
            sample_text=sample,
            previous=result,
        )
        if repaired:
            return repaired
    return result


def _system_prompt() -> str:
    return (
        "Tu es un expert en connaissements maritimes (Bill of Lading / SWB). "
        "Si des images de pages sont fournies, tu dois t’en servir en priorité pour lire les champs "
        "(mise en page, encadrés Shipper/Consignee, lignes B/L et Booking). "
        "Le texte OCR peut être faux ou dans le mauvais ordre : ne le suis que pour départager si une zone image est illisible. "
        "Réponds UNIQUEMENT avec un objet JSON valide, sans markdown. null si absent. "
        "Clés obligatoires: bl_number, booking_number, port_loading, port_discharge, weight, shipper, consignee, vessel."
    )


def _user_intro_vision(n_pages: int) -> str:
    return (
        f"Ci-dessous : {n_pages} image(s) de page(s), puis le texte OCR. "
        "Lis en priorité les images pour le connaissement officiel (souvent 'B/L No'), les ports, les poids et les blocs Shipper/Consignee."
    )


def _user_instructions_only() -> str:
    return (
        "Schéma JSON exact: {\n"
        '  "bl_number": string|null,\n'
        '  "booking_number": string|null,\n'
        '  "port_loading": string|null,\n'
        '  "port_discharge": string|null,\n'
        '  "weight": string|null,\n'
        '  "shipper": string|null,\n'
        '  "consignee": string|null,\n'
        '  "vessel": string|null\n'
        "}\n\n"
        "Règles bl_number: numéro du connaissement/SWB principal (ex. COKA04793). "
        "Jamais un booking séparé, jamais un code transport (ex. MABUS...) sauf s'il est clairement étiqueté B/L.\n"
        "booking_number: réservation / Booking / S.O. (ex. 545192).\n"
        "port_loading / port_discharge: POL et POD. Si Place of Delivery diffère, ajoute à la fin de port_discharge.\n"
        "weight: inclure net et brut si présents.\n"
        "shipper / consignee: texte des blocs.\n"
        "vessel: navire / voyage.\n"
    )


def _user_block_text(sample: str) -> str:
    return _user_instructions_only() + ("\nTexte OCR (secours):\n---\n" + sample + "\n---")


def _complete_and_parse(parts: list[Any], *, key: str) -> dict[str, Any]:
    try:
        payload = {
            "contents": [{"role": "user", "parts": _to_api_parts(parts)}],
            "generationConfig": {
                "temperature": 0.05,
                "maxOutputTokens": 1600,
                "responseMimeType": "application/json",
            },
        }
        url = (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            + urllib.parse.quote(settings.gemini_model, safe="")
            + ":generateContent?key="
            + urllib.parse.quote(key, safe="")
        )
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=max(30, settings.gemini_timeout_seconds)) as resp:  # nosec B310
            body = resp.read().decode("utf-8", errors="replace")
        data = json.loads(body)
        raw = _extract_text_from_gemini_response(data).strip()
        raw = _JSON_SANITIZE.sub("", raw).strip()
        parsed = _parse_json_lenient(raw)
        if not isinstance(parsed, dict):
            return {}
        return {k: _clean_val(parsed.get(k)) for k in _AI_KEYS}
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        msg = body[:500]
        if "API_KEY_INVALID" in body or "API key not valid" in body:
            return {"_error": "API_KEY_INVALID"}
        log.warning("Gemini extraction failed HTTP %s: %s", exc.code, msg)
        return {}
    except Exception as exc:  # pylint: disable=broad-except
        log.warning("Gemini extraction failed: %s", exc)
        return {}


def _repair_identity_pass(
    *,
    key: str,
    image_parts: list[dict[str, Any]],
    sample_text: str,
    previous: dict[str, Any],
) -> dict[str, Any]:
    """Second focused pass when main extraction missed BL/Booking identity."""
    prev_json = json.dumps({k: previous.get(k) for k in _AI_KEYS}, ensure_ascii=False)
    repair_prompt = (
        "Relecture ciblée: ton extraction précédente manque l'identité du document.\n"
        "Trouve en priorité bl_number et booking_number. Si introuvable, laisse null.\n"
        "Ne renvoie QUE du JSON valide avec les mêmes clés.\n"
        "Extraction précédente:\n"
        + prev_json
        + "\n\nTexte OCR (secours):\n---\n"
        + sample_text
        + "\n---"
    )
    parts: list[Any] = [_system_prompt(), repair_prompt]
    parts.extend(image_parts)
    out = _complete_and_parse(parts, key=key)
    if not out:
        return {}
    if _has_minimum_identity(out):
        return out
    return {}


def _parse_json_lenient(raw: str) -> dict[str, Any]:
    """Parse model JSON with small repair for occasional malformed text."""
    try:
        return json.loads(raw)
    except Exception:
        pass

    # Strip markdown fences / surrounding text and keep first balanced JSON object.
    core = _extract_first_balanced_object(raw)

    # Escape literal newlines/tabs inside quoted strings.
    repaired_chars: list[str] = []
    in_string = False
    escaped = False
    for ch in core:
        if in_string:
            if escaped:
                repaired_chars.append(ch)
                escaped = False
                continue
            if ch == "\\":
                repaired_chars.append(ch)
                escaped = True
                continue
            if ch == '"':
                repaired_chars.append(ch)
                in_string = False
                continue
            if ch == "\n":
                repaired_chars.append("\\n")
                continue
            if ch == "\r":
                repaired_chars.append("\\r")
                continue
            if ch == "\t":
                repaired_chars.append("\\t")
                continue
            repaired_chars.append(ch)
            continue

        repaired_chars.append(ch)
        if ch == '"':
            in_string = True

    repaired = "".join(repaired_chars)
    return json.loads(repaired)


def _extract_first_balanced_object(text: str) -> str:
    start = text.find("{")
    if start < 0:
        return text
    depth = 0
    in_string = False
    escaped = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escaped:
                escaped = False
                continue
            if ch == "\\":
                escaped = True
                continue
            if ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return text[start:]


def _to_api_parts(parts: list[Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for p in parts:
        if isinstance(p, str):
            out.append({"text": p})
            continue
        if isinstance(p, dict) and "mime_type" in p and "data" in p:
            out.append({"inline_data": {"mime_type": p["mime_type"], "data": p["data"]}})
    return out


def _extract_text_from_gemini_response(data: dict[str, Any]) -> str:
    cands = data.get("candidates") or []
    if not cands:
        return ""
    content = (cands[0] or {}).get("content") or {}
    parts = content.get("parts") or []
    texts = [str(p.get("text") or "") for p in parts if isinstance(p, dict)]
    return "\n".join([t for t in texts if t]).strip()


def _document_image_parts_for_gemini(path: Path) -> list[dict[str, Any]]:
    suf = path.suffix.lower()
    max_side = max(800, settings.gemini_vision_max_side)
    max_pages = max(1, settings.gemini_vision_max_pages)
    images: list[Image.Image] = []

    if suf == ".pdf":
        from pdf2image import convert_from_path

        images = convert_from_path(
            str(path),
            dpi=settings.gemini_vision_dpi,
            first_page=1,
            last_page=max_pages,
        )
    elif suf in (".png", ".jpg", ".jpeg", ".webp"):
        with Image.open(path) as im:
            images = [im.convert("RGB")]
    else:
        return []

    parts: list[dict[str, Any]] = []
    for img in images:
        b = _image_to_jpeg_bytes(img, max_side=max_side)
        parts.append({"mime_type": "image/jpeg", "data": base64.b64encode(b).decode("ascii")})
    return parts


def _image_to_jpeg_bytes(img: Image.Image, max_side: int = 1600) -> bytes:
    rgb = img.convert("RGB")
    w, h = rgb.size
    if max(w, h) > max_side:
        ratio = max_side / max(w, h)
        rgb = rgb.resize((int(w * ratio), int(h * ratio)), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    rgb.save(
        buf,
        format="JPEG",
        quality=settings.gemini_vision_jpeg_quality,
        optimize=True,
    )
    return buf.getvalue()


def _sample_text_for_ai(raw_text: str, budget: int) -> str:
    if len(raw_text) <= budget:
        return raw_text
    sep = "\n\n[ ... tronque ... ]\n\n"
    reserve = len(sep) * 2
    usable = max(1000, budget - reserve)
    head = usable * 45 // 100
    tail = usable * 30 // 100
    mid = usable - head - tail
    mid_start = max(0, len(raw_text) // 2 - mid // 2)
    return raw_text[:head] + sep + raw_text[mid_start : mid_start + mid] + sep + raw_text[-tail:]


def _clean_val(v: Any) -> str | None:
    if v is None:
        return None
    s = str(v).strip()
    return s if s else None


def _has_minimum_identity(data: dict[str, Any]) -> bool:
    return bool(str(data.get("bl_number") or "").strip() or str(data.get("booking_number") or "").strip())
