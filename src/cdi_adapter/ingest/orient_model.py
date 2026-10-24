"""Which way up, when the ink shape and the printed text cannot say: ask the reader, never refuse the page.

A handwriting-only picture (a screenshot of a note, a page with no print) is turned sideways by the camera as often as any other, and the two cheaper
witnesses (``geometry.decide_sideways``: ink above / below the line middle; ``orient_ocr``: how well the PRINT reads at each turn) give no verdict on it.
The page used to be held for a retake. It is now read: the model is shown the page at each of the two candidate turns and asked whether the writing is upright.
Exactly one "upright" decides. Anything else (both, neither, the model unreachable) falls back to the clearer ink score, and to the first turn
when there is none; the page then goes on with a warning that the way up was chosen, not known. Nothing here reads content."""
from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from ..logging import get_logger

log = get_logger(__name__)

PROMPT = ("Look at this picture of a page with handwriting or print. Is the writing upright, so that it can be read normally: the lines run from left "
          'to right and the tops of the letters point up? Answer ONLY as JSON {"upright": true} or {"upright": false}.')
SCHEMA: dict[str, Any] = {"type": "object", "properties": {"upright": {"type": "boolean"}}, "required": ["upright"]}


def _png(arr: np.ndarray, long_side: int = 1000) -> bytes:
    h, w = arr.shape[:2]
    s = long_side / max(h, w)
    small = cv2.resize(arr, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else arr
    ok, enc = cv2.imencode(".png", small)
    if not ok:  # pragma: no cover
        raise RuntimeError("encode failed")
    return enc.tobytes()


def which_way(arr: np.ndarray, info: dict[str, Any] | None = None, candidates: tuple[int, ...] = (1, 3), client: Any = None) -> tuple[int, dict[str, Any]]:
    """``(k, how)``: the ``numpy.rot90`` turn (one of ``candidates``) to apply, and how it was chosen (``by`` = model | ink score | default)."""
    info = info or {}
    answers: dict[int, bool] = {}
    try:
        if client is None:
            from ..ml.client import get_client

            client = get_client()
        for k in candidates:
            resp, _ = client.vlm_json_ex(_png(np.ascontiguousarray(np.rot90(arr, k))), PROMPT, SCHEMA, max_tokens=12, retries=1)
            answers[k] = isinstance(resp, dict) and resp.get("upright") is True
    except Exception as exc:  # noqa: BLE001 - the page must still be read when the model cannot be asked
        log.warning("orient_model_failed", error=str(exc)[:160])
        answers = {}
    upright = [k for k in candidates if answers.get(k)]
    if len(upright) == 1:
        return upright[0], {"by": "model", "answers": {str(k): v for k, v in answers.items()}}
    s1, s3 = info.get("upright_score_k1"), info.get("upright_score_k3")
    if s1 is not None and s3 is not None and s1 != s3:
        return (1 if s1 > s3 else 3), {"by": "ink score", "answers": {str(k): v for k, v in answers.items()}, "upright_score_k1": s1, "upright_score_k3": s3}
    return candidates[0], {"by": "default", "answers": {str(k): v for k, v in answers.items()}}
