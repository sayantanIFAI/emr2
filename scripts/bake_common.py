"""Helpers shared by the Qwen benchmark scripts (scoring set-up, per-doctor vocabulary, leave-one-out, snapping to a vocabulary)."""
from __future__ import annotations

import json
import re
from pathlib import Path

from cdi_adapter.recognition.scoring import _norm, score, wilson_lower


def _alnum(s: str) -> str:
    return re.sub(r"[^0-9a-z]", "", (s or "").casefold())

def _rows(p: Path) -> list[dict]:
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]

def _group(line_id: str) -> str:
    """The doctor = the prescription photo the line came from (ids are <photo>_<n>)."""
    return line_id.rsplit("_", 1)[0]


def doctor_vocab(rows: list[dict]) -> dict[str, list[tuple[str, set[str]]]]:
    """Per doctor: the words in that doctor's OTHER confirmed lines. Used leave-one-out, so a line never sees its own truth."""
    out: dict[str, list[tuple[str, set[str]]]] = {}
    for r in rows:
        toks = {t for t in re.findall(r"[A-Za-z][A-Za-z0-9.\-]*", r["truth"]) if len(t) >= 2}
        out.setdefault(_group(r["id"]), []).append((r["id"], {t.casefold() for t in toks}))
    return out


def vocab_for(line_id: str, vocab: dict[str, list[tuple[str, set[str]]]]) -> list[str]:
    words: set[str] = set()
    for other, toks in vocab.get(_group(line_id), []):
        if other != line_id:
            words |= toks
    return sorted(words)

def snap_to_vocab(text: str, words: list[str]) -> str:
    """Deterministic memory: a word one or two letters away from exactly ONE of the doctor's confirmed words becomes that word."""
    from cdi_adapter.recognition.scoring import levenshtein

    def fix(tok: str) -> str:
        core = tok.casefold()
        if len(core) < 3 or core in words:
            return tok
        lim = 1 if len(core) < 6 else 2
        near = [w for w in words if abs(len(w) - len(core)) <= lim and levenshtein(core, w) <= lim]
        return near[0] if len(near) == 1 else tok
    return re.sub(r"[A-Za-z][A-Za-z0-9]*", lambda m: fix(m.group(0)), text)

def _setup(name: str, truths: list[str], preds: list[str]) -> dict:
    ok = sum(_norm(t) == _norm(p) for t, p in zip(truths, preds))
    n = len(truths)
    ok_alnum = sum(_alnum(t) == _alnum(p) for t, p in zip(truths, preds))
    return {"name": name, **score(truths, preds), "accepted": n, "coverage": 1.0, "exact": ok, "silent_errors": n - ok,
            "exact_ignoring_punctuation_and_spacing": round(ok_alnum / n, 4) if n else None,
            "silent_error_rate_ignoring_punctuation_and_spacing": round((n - ok_alnum) / n, 4) if n else None,
            "silent_error_rate": round((n - ok) / n, 4) if n else None,
            "precision_lower_bound": round(wilson_lower(ok, n), 4)}
