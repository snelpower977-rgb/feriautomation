from __future__ import annotations

import base64
import io
import json
import logging
import re
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


def extract_bl_with_openai(file_path: Path | None, raw_text: str) -> dict[str, Any]:
    """Extrait les champs B/L via OpenAI. Si ``OPENAI_USE_VISION`` et un fichier image/PDF sont fournis, envoie les pages en image + texte OCR (priorité visuelle)."""
    key = (settings.openai_api_key or "").strip()
    if not key:
        return {}

    try:
        from openai import OpenAI
    except ImportError:
        log.warning("openai package not installed; skip AI extraction")
        return {}

    client = OpenAI(api_key=key, timeout=180.0)
    sample = _sample_text_for_openai(raw_text, max(4000, settings.openai_max_input_chars))
    system = _system_prompt()
    user_text_only = _user_block_text(sample)
    user_instructions = _user_instructions_only()

    if (
        file_path
        and settings.openai_use_vision
        and file_path.exists()
        and file_path.suffix.lower() in settings.openai_vision_extensions
    ):
        try:
            data_urls = _document_images_as_data_urls(file_path)
            if data_urls:
                content: list[dict[str, Any]] = [
                    {"type": "text", "text": _user_intro_vision(len(data_urls))},
                    {"type": "text", "text": user_instructions},
                ]
                for i, url in enumerate(data_urls):
                    content.append(
                        {"type": "text", "text": f"--- Page {i + 1} (image) ---"}
                    )
                    content.append({"type": "image_url", "image_url": {"url": url}})
                content.append(
                    {
                        "type": "text",
                        "text": "Texte OCR (secours, souvent imparfait):\n---\n"
                        + sample
                        + "\n---",
                    }
                )
                result = _complete_and_parse(client, system, content)
                if result:
                    return result
                log.warning("Vision response empty; retrying text-only")
        except Exception as exc:  # pylint: disable=broad-except
            log.warning("Vision path failed (%s); using text-only", exc)

    return _complete_and_parse(client, system, user_text_only)


def _system_prompt() -> str:
    return (
        "Tu es un expert en connaissements maritimes (Bill of Lading / SWB). "
        "Si des images de pages sont fournies, tu dois t’en servir en priorité pour lire les champs "
        "(mise en page, encadrés Shipper/Consignee, lignes B/L et Booking). "
        "Le texte OCR ci-dessous peut être faux ou dans le mauvais ordre : ne le suis que pour départager si une zone de l’image est illisible. "
        "Réponds UNIQUEMENT avec un objet JSON valide, sans markdown. null si absent. "
        "Clés obligatoires dans le JSON: bl_number, booking_number, port_loading, port_discharge, weight, shipper, consignee, vessel."
    )


def _user_intro_vision(n_pages: int) -> str:
    return (
        f"Ci-dessous : {n_pages} image(s) de page(s), puis le texte OCR. "
        "Lis **en priorité les images** pour le connaissement officiel (souvent « B/L No »), les ports, les poids et les blocs Shipper / Consignee.\n"
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
        "Règles bl_number: numéro du **connaissement / SWB principal** (ex. COKA04793). "
        "Jamais un booking séparé, jamais un code routier/prestataire (ex. MABUS…) sauf s’il est clairement étiqueté B/L.\n"
        "booking_number: réservation / Booking / S.O. (ex. 545192).\n"
        "port_loading / port_discharge: POL et POD. Si « Place of Delivery » diffère, ajoute à la fin de port_discharge "
        'ex. "Izmir | Lieu de livraison: Izmir".\n'
        "weight: inclure net et brut si présents (ex. « Net: 15,000 KGS ; Gross: 15,750 KGS »).\n"
        "shipper / consignee: texte des blocs (plusieurs lignes possibles dans la chaîne JSON, \\n pour retours ligne).\n"
        "vessel: navire / voyage.\n"
    )


def _user_block_text(sample: str) -> str:
    return _user_instructions_only() + (
        "\nTexte OCR (secours, souvent imparfait):\n---\n" + sample + "\n---"
    )


def _complete_and_parse(
    client: Any,
    system: str,
    user_content: str | list[dict[str, Any]],
) -> dict[str, Any]:
    try:
        resp = client.chat.completions.create(
            model=settings.openai_model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user_content},
            ],
            temperature=0.05,
            max_tokens=1600,
            response_format={"type": "json_object"},
        )
        raw = (resp.choices[0].message.content or "").strip()
        raw = _JSON_SANITIZE.sub("", raw).strip()
        data = json.loads(raw)
        if not isinstance(data, dict):
            return {}
        return {k: _clean_val(data.get(k)) for k in _AI_KEYS}
    except Exception as exc:  # pylint: disable=broad-except
        log.warning("OpenAI extraction failed: %s", exc)
        return {}


def _document_images_as_data_urls(path: Path) -> list[str]:
    suf = path.suffix.lower()
    max_side = max(800, settings.openai_vision_max_side)
    max_pages = max(1, settings.openai_vision_max_pages)
    images: list[Image.Image] = []

    if suf == ".pdf":
        from pdf2image import convert_from_path

        images = convert_from_path(
            str(path),
            dpi=settings.openai_vision_dpi,
            first_page=1,
            last_page=max_pages,
        )
    elif suf in (".png", ".jpg", ".jpeg", ".webp"):
        with Image.open(path) as im:
            # webp / rgba
            images = [im.convert("RGB")]
    else:
        return []

    urls = []
    for img in images:
        urls.append(_image_to_jpeg_data_url(img, max_side=max_side))
    return urls


def _image_to_jpeg_data_url(img: Image.Image, max_side: int = 1600) -> str:
    rgb = img.convert("RGB")
    w, h = rgb.size
    if max(w, h) > max_side:
        r = max_side / max(w, h)
        rgb = rgb.resize((int(w * r), int(h * r)), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    rgb.save(buf, format="JPEG", quality=settings.openai_vision_jpeg_quality, optimize=True)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


def _sample_text_for_openai(raw_stext: str, budget: int) -> str:
    if len(raw_stext) <= budget:
        return raw_stext
    sep = "\n\n[ … tronqué … ]\n\n"
    reserve = len(sep) * 2
    usable = max(1000, budget - reserve)
    head = usable * 45 // 100
    tail = usable * 30 // 100
    mid = usable - head - tail
    mid_start = max(0, len(raw_stext) // 2 - mid // 2)
    return (
        raw_stext[:head]
        + sep
        + raw_stext[mid_start : mid_start + mid]
        + sep
        + raw_stext[-tail:]
    )


def _clean_val(v: Any) -> str | None:
    if v is None:
        return None
    s = str(v).strip()
    return s if s else None
