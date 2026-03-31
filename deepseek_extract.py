from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

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


def extract_bl_with_deepseek(_file_path: Path | None, raw_text: str) -> dict[str, Any]:
    key = (settings.deepseek_api_key or "").strip()
    if not key:
        return {}
    try:
        from openai import OpenAI
    except ImportError:
        log.warning("openai package not installed; skip DeepSeek extraction")
        return {}

    client = OpenAI(
        api_key=key,
        base_url="https://api.deepseek.com",
        timeout=float(max(30, settings.deepseek_timeout_seconds)),
    )
    prompt = _user_block_text(
        _sample_text_for_ai(raw_text, max(4000, settings.deepseek_max_input_chars))
    )
    try:
        resp = client.chat.completions.create(
            model=settings.deepseek_model,
            messages=[
                {"role": "system", "content": _system_prompt()},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
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
        log.warning("DeepSeek extraction failed: %s", exc)
        return {}


def _system_prompt() -> str:
    return (
        "Tu es un expert en connaissements maritimes (Bill of Lading / SWB). "
        "Retourne UNIQUEMENT un objet JSON valide, sans markdown. "
        "Si absent, utilise null. "
        "Clés obligatoires: bl_number, booking_number, port_loading, port_discharge, "
        "weight, shipper, consignee, vessel."
    )


def _user_block_text(sample: str) -> str:
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
        "Règles: bl_number = numéro B/L ou SWB principal; booking_number = numéro de réservation; "
        "ports = POL/POD; shipper/consignee = blocs texte; vessel = navire/voyage.\n\n"
        "Texte OCR:\n---\n"
        + sample
        + "\n---"
    )


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
    return (
        raw_text[:head]
        + sep
        + raw_text[mid_start : mid_start + mid]
        + sep
        + raw_text[-tail:]
    )


def _clean_val(v: Any) -> str | None:
    if v is None:
        return None
    s = str(v).strip()
    return s if s else None
