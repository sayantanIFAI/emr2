"""Which OCR blocks support a value the model wrote, found by the program instead of written by the model.

The model was asked to add an ``evidence`` list of block ids ("b12") to every value it wrote. Those lists are pure overhead in the answer: tokens
at about 70 per second on the pod. The compact answer (``extract_compact_answer``) leaves them out, and this module puts the same links back afterwards:
for each written text, the blocks whose words it shares (at least 60 % of the text's words, up to four blocks, best first). The ids keep the prompt's
numbering (``[bN]`` = position in ``blocks``), so provenance, bounding boxes and the confidence from the readers work as before."""
from __future__ import annotations

import re
from typing import Any

_WORD = re.compile(r"[a-z0-9]+")
_LISTS = ("investigations", "advice", "diagnoses", "medications", "investigation_preparation", "vitals")


def _tokens(text: str) -> set[str]:
    return {w for w in _WORD.findall(text.casefold()) if len(w) >= 2}


def blocks_for(text: str | None, blocks: list[dict[str, Any]], limit: int = 4, floor: float = 0.6) -> list[str]:
    """``["b3", "b7"]``: the blocks (by the prompt's numbering) that hold the words of ``text``."""
    want = _tokens(text or "")
    if not want:
        return []
    scored: list[tuple[float, int]] = []
    for i, b in enumerate(blocks, start=1):
        have = _tokens(str(b.get("text") or ""))
        if not have:
            continue
        shared = len(want & have)
        if shared == 0:
            continue
        cover = shared / len(want)                       # how much of the written text this block holds
        inside = shared / len(have)                      # how much of the block is the written text (a line split over blocks)
        if cover >= floor or (inside >= 0.8 and shared >= 2):
            scored.append((max(cover, inside * 0.9), i))
    scored.sort(key=lambda t: (-t[0], t[1]))
    return [f"b{i}" for _s, i in scored[:limit]]


def attach(payload: dict[str, Any], blocks: list[dict[str, Any]]) -> int:
    """Fill ``evidence`` where the answer has none. Returns how many values got links."""
    n = 0

    def put(obj: dict[str, Any], text: Any) -> None:
        nonlocal n
        if isinstance(obj, dict) and not obj.get("evidence") and isinstance(text, str) and text.strip():
            ev = blocks_for(text, blocks)
            if ev:
                obj["evidence"] = ev
                n += 1

    for key in ("patient", "prescriber"):
        sub = payload.get(key)
        if isinstance(sub, dict):
            put(sub, sub.get("name"))
            clinic = sub.get("clinic")
            if isinstance(clinic, dict):
                put(clinic, clinic.get("name"))
    for key in _LISTS:
        for item in payload.get(key) or []:
            if isinstance(item, dict):
                put(item, item.get("text") or item.get("drug_text") or item.get("name"))
    return n
