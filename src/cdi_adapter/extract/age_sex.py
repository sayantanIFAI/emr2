"""The age and sex written together beside the name ("74/F", "30 Y/M", "73 yrs | Female"), read from their own small crop.

MEASURED on a real page: the whole-page answer gave age 74 and sex M for a paper that says "74/F" (the F read as M, or taken from the title), and
a line reader wrote "741f." / "7416." for the same token. The token is cut out of the name's row, enlarged to three sizes and read as {age, sex};
a value is used only when at least TWO of the three readings agree (a lone reading decides nothing). It replaces the whole-page answer only when
it is clear; a conflict is recorded in ``payload['_age_sex']``."""
from __future__ import annotations

import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from ..logging import get_logger
from . import resolve_llm

log = get_logger(__name__)

SCHEMA: dict[str, Any] = {"type": "object", "properties": {"age": {"type": ["string", "null"]}, "sex": {"type": ["string", "null"]}}, "required": ["age", "sex"]}
PROMPT = ("This is the age and sex a doctor wrote on a prescription beside the patient's name, for example 74/F, 30 Y/M or 8 yrs F. Write the age "
          "exactly as written (the number, with its unit if there is one) and the sex as M or F. If the age or the sex cannot be read, answer null for "
          "it. Never guess from the name or the title. Answer ONLY as JSON: {\"age\": \"...\", \"sex\": \"...\"}")
SCALES = (1.0, 2.0, 3.0)
# a short piece that starts with the age: "741f.", "74|F", "73yrs", "30 Y", "8 yrs"
_TOKEN = re.compile(r"^\W{0,2}\d{1,3}\s*[A-Za-z0-9|/\\.\-]{0,6}\W{0,2}$")
_UNITS = {"m": "months", "mo": "months", "month": "months", "months": "months", "d": "days", "day": "days", "days": "days",
          "w": "weeks", "wk": "weeks", "week": "weeks", "weeks": "weeks"}


def find_token(blocks: list[dict[str, Any]] | None, name: str | None) -> tuple[int, int, int, int] | None:
    """The box of the age / sex piece in the name's row (the piece that starts with a number), or None."""
    best = resolve_llm.best_name_block(blocks, name)
    if best is None:
        return None
    left, top, right, bottom = resolve_llm.name_row_box(best, blocks)
    lh = max(8, int(best["bbox"][3]) - int(best["bbox"][1]))
    for b in blocks or []:
        try:
            x0, y0, x1, y1 = (int(float(v)) for v in b["bbox"][:4])
        except (KeyError, TypeError, ValueError, IndexError):
            continue
        cy = (y0 + y1) / 2
        if abs(cy - (top + bottom) / 2) <= 1.2 * max(lh, bottom - top) and x0 >= left and _TOKEN.match(str(b.get("text") or "").strip()):
            return x0, y0, x1, y1
    return None


def crops_for(image: bytes, box: tuple[int, int, int, int]) -> list[bytes]:
    import cv2
    import numpy as np

    arr = cv2.imdecode(np.frombuffer(image, np.uint8), cv2.IMREAD_COLOR)
    if arr is None:
        return []
    h, w = arr.shape[:2]
    x0, y0, x1, y1 = box
    px, py = max(20, int(0.02 * w)), max(10, int(0.01 * h))
    crop = arr[max(0, y0 - py):min(h, y1 + py), max(0, x0 - px):min(w, x1 + px)]
    out = []
    for f in SCALES:
        c = crop if f == 1.0 else cv2.resize(crop, None, fx=f, fy=f, interpolation=cv2.INTER_CUBIC)
        ok, png = cv2.imencode(".png", c)
        if ok:
            out.append(png.tobytes())
    return out


def _age_key(text: Any) -> str | None:
    m = re.match(r"^\s*(\d{1,3})\s*([A-Za-z]+)?", str(text or ""))
    if not m:
        return None
    return f"{int(m.group(1))} " + _UNITS.get((m.group(2) or "").lower(), "yrs")


def vote(reads: list[dict[str, Any]]) -> tuple[str | None, str | None]:
    """``(age text, sex)`` each only when at least two readings agree."""
    ages = Counter(k for k in (_age_key(r.get("age")) for r in reads) if k)
    sexes = Counter(s for s in (str(r.get("sex") or "").strip()[:1].upper() for r in reads) if s in ("M", "F"))
    age = ages.most_common(1)[0][0] if ages and ages.most_common(1)[0][1] >= 2 else None
    sex = sexes.most_common(1)[0][0] if sexes and sexes.most_common(1)[0][1] >= 2 else None
    return age, sex


def apply(client: Any, image: bytes, blocks: list[dict[str, Any]], payload: dict[str, Any]) -> dict[str, Any]:
    """Set the patient's age / sex from the crop reading when it is clear. Returns what was found (also kept in ``payload['_age_sex']``)."""
    pat = payload.get("patient")
    if not isinstance(pat, dict):
        return {}
    box = find_token(blocks, pat.get("name"))
    if box is None:
        return {}
    crops = crops_for(image, box)
    if not crops:
        return {}

    def ask(png: bytes) -> dict[str, Any]:
        try:
            resp, _ = client.vlm_json_ex(png, PROMPT, SCHEMA, max_tokens=40, retries=1)
            return resp or {}
        except Exception as exc:  # noqa: BLE001 - an extra look must never cost the document
            log.warning("age_sex_read_failed", error=str(exc)[:160])
            return {}

    with ThreadPoolExecutor(max_workers=len(crops)) as pool:
        reads = list(pool.map(ask, crops))
    age, sex = vote(reads)
    found: dict[str, Any] = {"reads": [{"age": r.get("age"), "sex": r.get("sex")} for r in reads], "age": age, "sex": sex}
    if age:
        pat["age_text"] = age
    if sex:
        was = str(pat.get("sex") or "").strip()[:1].upper()
        if was and was != sex:
            found["sex_whole_page_said"] = was                  # the line says one thing and the whole-page answer another: the line wins, and it is recorded
        pat["sex"] = {"M": "Male", "F": "Female"}[sex]
    payload["_age_sex"] = found
    return found
