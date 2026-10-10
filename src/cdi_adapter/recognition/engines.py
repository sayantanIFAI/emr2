"""Line recognizers. Each returns *transcription only* (text + confidence), never a code.

- ``QwenLineEngine`` contextual reader: Qwen2.5-VL via the model gateway, one crop at a time,
  prompted WITHOUT any other engine's answer (no anchoring - ARCHITECTURE §15.3).
"""
from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from ..config import settings
from ..logging import get_logger

log = get_logger(__name__)


@dataclass
class Reading:
    """One engine's reading of one crop. ``conf`` is None when the engine gives none -
    a VLM transcription has no calibrated confidence and we do not invent one."""

    engine: str
    engine_version: str
    text: str
    conf: float | None
    token_confidences: list[float] = field(default_factory=list)
    prompt_hash: str | None = None
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None


class EngineUnavailable(RuntimeError):
    pass


def is_fallback_model(model: str | None) -> bool:
    """True when ``model`` is the OOM fallback the gateway loaded instead of the primary
    (``vlm_fallback_model_id``). Anything it read is held for review (``gate_fallback_review``)."""
    return bool(model) and model != settings.vlm_model_id and model == settings.vlm_fallback_model_id


_RUN = re.compile(r"(.)\1{5,}")
_SPACED_Q = re.compile(r"(?:\s*\?){4,}")


def clean_line(txt: str | None) -> str:
    """What the model wrote for a crop, with the degenerate cases removed: nothing but punctuation ('?', '????...') is
    no reading, and a long run of one character or of spaced question marks is cut back. Letters and digits that were
    really read are never touched."""
    t = (txt or "").strip()
    if not re.search(r"[A-Za-z0-9]", t):
        return ""
    t = _SPACED_Q.sub(" ?", t)
    return _RUN.sub(lambda m: m.group(1) * 3, t).strip()


QWEN_LINE_PROMPT = (
    "Transcribe exactly the text written in this image crop. It is one line from a "
    "medical prescription and may be handwritten. Copy letters, numbers, units and "
    "dosing notation exactly as they appear. Do not correct spelling, expand "
    "abbreviations, or guess drug names. Write an unreadable character as ?. "
    "Output only the transcription on one line."
)


class QwenLineEngine:
    """Independent contextual reading of ONE crop through the model gateway.

    The prompt is a constant: it never carries another engine's candidate, so agreement
    with a second look at the same line is genuine independent evidence rather than anchoring."""

    name = "qwen2.5-vl"

    @property
    def version(self) -> str:
        """The configured model. A reading carries the model the gateway REPORTED for it: after
        an out-of-memory load that is the fallback, and recording the primary would be false."""
        return settings.vlm_model_id

    def recognize(self, crops_png: list[bytes]) -> list[Reading]:
        import hashlib

        from ..ml.client import get_client

        if settings.qwen_line_mode == "off":
            return [Reading(self.name, self.version, "", None, error="qwen line mode off")
                    for _ in crops_png]
        client = get_client()
        ph = hashlib.sha256(QWEN_LINE_PROMPT.encode()).hexdigest()[:16]

        def one(png: bytes) -> Reading:
            try:
                txt, served = client.vlm_generate_ex(png, QWEN_LINE_PROMPT,
                                                     max_tokens=settings.qwen_line_max_tokens)
                line = next((s.strip() for s in (txt or "").splitlines() if s.strip()), "")
                cleaned = clean_line(line)
                if line and not cleaned:
                    # the model returned only punctuation: not a reading. What it said stays on record in the error.
                    return Reading(self.name, served or self.version, "", None, prompt_hash=ph,
                                   error=f"no readable text (the model wrote: {line[:30]!r})")
                return Reading(self.name, served or self.version, cleaned, None, prompt_hash=ph)
            except Exception as exc:  # noqa: BLE001
                return Reading(self.name, self.version, "", None, prompt_hash=ph, error=str(exc)[:200])

        workers = max(1, int(settings.qwen_line_concurrency))
        if workers == 1 or len(crops_png) < 2:
            return [one(p) for p in crops_png]
        # a batching server (vLLM) reads many lines in the time of a few: send them together. The
        # answers come back in the order of the crops, whatever order they finish in.
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="qline") as ex:
            return list(ex.map(one, crops_png))
