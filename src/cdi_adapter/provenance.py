"""Which models and components produced a result (OUT-S1, SW-S1 AC4, SW-S2 AC2).

Every value is read at the moment it is asked for, from the code that is actually installed and the
settings that are actually in force: nothing here is typed in by hand, so it cannot drift from what ran.
"""
from __future__ import annotations

import hashlib
from importlib import metadata
from typing import Any

from . import swap
from .config import settings


def _dist_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def prompt_version() -> str:
    """A fingerprint of the extraction prompt text (base prompt + per-document-type additions).
    It changes exactly when the words sent to the model change."""
    from .extract import prompt as P

    text = P._BASE + repr(sorted(P._EXTRA.items()))
    return "p-" + hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def engine_versions(served_vlm: str | None = None) -> dict[str, Any]:
    """The component and model identities in force. ``served_vlm`` is what the model gateway
    reported for the call (it can differ from the setting when the OOM fallback answered)."""
    from .compliance.models import pinned_revision

    return {
        "vlm_configured": settings.vlm_model_id,
        "vlm_served": served_vlm,
        "vlm_revision": pinned_revision(served_vlm or settings.vlm_model_id),
        "printed_ocr": f"rapidocr-onnxruntime {_dist_version('rapidocr-onnxruntime') or '?'}",
        "pdf_renderer": swap.current_version("pdf_renderer"),
        "image_library": f"Pillow {_dist_version('Pillow') or '?'}; opencv {_dist_version('opencv-python-headless') or '?'}",
        "qwen_line_mode": settings.qwen_line_mode,
        "recognition_v2": settings.recognition_v2,
    }
