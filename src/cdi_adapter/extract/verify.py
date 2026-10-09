"""Is this test really written on the page? A yes / no question to the same model, with its own probability of "yes".

A candidate test (from the full-page answer, the text scan, a second look) is put to the page image as a CLOSED question: "is the laboratory test X
written here as a test the doctor ordered?". The answer is the model's token probability of yes against no, not a self-reported confidence. It only
RANKS and flags candidates; nothing is accepted or rejected by it alone (a person still checks every handwritten test). How well the probability
separates real tests from wrong ones is MEASURED by ``scripts/measure_verify.py`` on the labelled pages before any threshold is used."""
from __future__ import annotations

import base64
import math
from concurrent.futures import ThreadPoolExecutor

import httpx

from ..config import settings
from ..logging import get_logger

log = get_logger(__name__)

QUESTION = ("This is a handwritten medical prescription. Is the laboratory test \"{name}\" written on it as a test the doctor ordered "
            "(not a past result, not a medicine, not an imaging or ECG order)? Answer with one word: yes or no.")


def _p_yes(top_logprobs: list[dict]) -> float | None:
    yes = no = 0.0
    for t in top_logprobs or []:
        tok = str(t.get("token", "")).strip().casefold()
        p = math.exp(float(t.get("logprob", -99)))
        if tok in ("yes", "y"):
            yes += p
        elif tok in ("no", "n"):
            no += p
    return yes / (yes + no) if (yes + no) > 0 else None


def ask_one(image: bytes, name: str, client: httpx.Client | None = None) -> float | None:
    """The model's probability that ``name`` is written on ``image`` as an ordered test (None when the call failed)."""
    url = settings.vllm_url.rstrip("/") + "/chat/completions"
    body = {"model": settings.vllm_model or settings.vlm_model_id, "temperature": 0, "max_tokens": 1, "logprobs": True, "top_logprobs": 8,
            "messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(image).decode()}},
                {"type": "text", "text": QUESTION.format(name=name.replace('"', "'")[:80])}]}]}
    try:
        c = client or httpx.Client(timeout=settings.vllm_timeout_s, trust_env=False)
        r = c.post(url, json=body)
        r.raise_for_status()
        lp = r.json()["choices"][0]["logprobs"]["content"][0]["top_logprobs"]
        return _p_yes(lp)
    except Exception as exc:  # noqa: BLE001 - a missing score must never cost the page
        log.warning("verify_failed", name=name[:40], error=str(exc)[:120])
        return None


def verify_tests(images: list[bytes], names: list[str], workers: int = 4) -> dict[str, float]:
    """``{name: highest probability of "yes" over the given views of the page}``; a name with no answer is left out."""
    if not names or not images or settings.mlserve_backend != "vllm":
        return {}
    jobs = [(i, n) for n in names for i in images]
    with httpx.Client(timeout=settings.vllm_timeout_s, trust_env=False) as c, ThreadPoolExecutor(max_workers=workers) as ex:
        got = list(ex.map(lambda j: (j[1], ask_one(j[0], j[1], c)), jobs))
    out: dict[str, float] = {}
    for n, p in got:
        if p is not None:
            out[n] = max(out.get(n, 0.0), p)
    return out
