"""The Indian national code sets, as deterministic lookups.

* labs   Common Lab Codes for India (CLCI, a curated LOINC subset)      -> ``labs()``  /  ``lab_candidates(text)``
* drugs  Common Drug Codes for India (CDCI, SNOMED CT national extension) -> ``drugs()`` / ``drug_lookup(text)``

Built from the NRCeS / C-DAC packages by ``scripts/build_indian_codes.py`` into ``settings.indian_codes_dir``. The
packages are not in the repository (their licence limits redistribution to India), so without the directory every
lookup answers "unknown" and the other rules still apply.

A lookup NEVER guesses: it answers with what the list contains (``exact`` or within a misread letter, ``fuzzy``),
or nothing. When a name has several codes (urine creatinine, spot or 24 hour), all are returned and the caller
must treat the binding as a candidate, not a fact.
"""
from __future__ import annotations

import difflib
import json
import re
import unicodedata
import threading
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from ..config import settings

LOINC = "http://loinc.org"
CDCI = "urn:cdac:cdci"             # Common Drug Codes for India: SNOMED CT identifiers + the India extension

_lock = threading.Lock()
_cache: dict[str, object] = {}
_FORM_PREFIX = re.compile(r"^\s*(?:inj|in|tab|t|cap|c|syp|sy|syr|susp|oint|cream|gel|lotion|drops?|neb|inh|iv|im|sc|po|rx)\b[\s.:\-]*", re.I)
_GENERIC_WORDS = frozenset("test tests level levels total quantitative qualitative blood serum plasma urine".split())


def norm(s: str | None) -> str:
    # superscript / subscript digits and signs ("Ca²⁺", "D₃") are their plain forms: the model writes them that way
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", unicodedata.normalize("NFKC", s or "").casefold())).strip()


def _dir(path: str | None = None) -> Path | None:
    p = path if path is not None else settings.indian_codes_dir
    return Path(p) if p else None


# ------------------------------------------------------------------------------------------------- labs
@dataclass
class LabCode:
    name: str
    loinc: str
    lcn: str
    specimen: str | None
    aliases: list[str] = field(default_factory=list)


def labs(path: str | None = None) -> dict[str, list[LabCode]]:
    """normalised name or alias -> every CLCI test that carries it."""
    d = _dir(path)
    f = d / "clci_labs.json" if d else None
    if f is None or not f.is_file():
        return {}
    key = f"labs:{f}"
    with _lock:
        if key not in _cache:
            idx: dict[str, list[LabCode]] = defaultdict(list)
            for t in json.loads(f.read_text(encoding="utf-8"))["tests"]:
                lc = LabCode(t["name"], t["loinc"], t.get("lcn") or t.get("fsn") or t["name"], t.get("specimen"), t.get("aliases", []))
                for nm in {t["name"], *t.get("aliases", [])}:
                    k = norm(nm)
                    if k and lc not in idx[k]:
                        idx[k].append(lc)
            _cache[key] = dict(idx)
        return _cache[key]   # type: ignore[return-value]


def lab_candidates(text: str | None, path: str | None = None) -> list[LabCode]:
    """The CLCI tests this name is: exactly, or without the words every lab name carries. [] = not in the list."""
    idx = labs(path)
    n = norm(text)
    if not idx or not n:
        return []
    bare = " ".join(w for w in n.split() if w not in _GENERIC_WORDS) or n
    for k in (n, bare):
        if k in idx:
            return idx[k]
    return []


def suggest_labs(text: str | None, k: int = 3, floor: float = 0.6, path: str | None = None) -> list[str]:
    """Reference names a misread entry might be (for the model to choose from; never applied on its own)."""
    idx = labs(path)
    n = norm(text)
    if not idx or len(n) < 3:
        return []
    scored = []
    for name in idx:
        if len(name) < 3 or abs(len(name) - len(n)) > max(3, len(n) // 2):
            continue
        r = difflib.SequenceMatcher(None, n, name).ratio()
        if r >= floor:
            scored.append((r, name))
    scored.sort(reverse=True)
    return [idx[name][0].name if len(name.split()) > 1 or name == norm(idx[name][0].name) else name.upper() for _r, name in scored[:k]]


# ------------------------------------------------------------------------------------------------- drugs
@dataclass(frozen=True)
class DrugMatch:
    kind: str                 # "brand" | "generic" | "substance"
    matched: str              # the name in the list
    product_id: str | None
    generic_ids: tuple[str, ...]
    generic_names: tuple[str, ...]
    fuzzy: bool


class DrugIndex:
    def __init__(self, data: dict) -> None:
        self.brands: dict[str, list] = data["brands"]
        self.generic_stems: dict[str, str] = data["generic_stems"]
        self.substances: dict[str, str] = data["substances"]
        self.generic_names: dict[str, str] = data["generic_names"]
        self.first_brand: dict[str, str] = {}
        for k in self.brands:
            self.first_brand.setdefault(k.split(" ", 1)[0], k)
        self.first_generic: dict[str, tuple[str, str]] = {}
        for k, gid in self.generic_stems.items():
            self.first_generic.setdefault(k.split(" ", 1)[0], ("generic", k))
        for k in self.substances:
            self.first_generic.setdefault(k.split(" ", 1)[0], ("substance", k))
        self.buckets: dict[tuple[str, int], list[str]] = defaultdict(list)
        for w in set(self.first_brand) | set(self.first_generic):
            if len(w) >= 6:
                self.buckets[(w[0], len(w))].append(w)

    def _brand(self, key: str, fuzzy: bool) -> DrugMatch:
        recs = self.brands[key]
        gids = tuple(sorted({g for _p, _n, gs in recs for g in gs}))
        names = tuple(dict.fromkeys(self.generic_names[g] for g in gids if g in self.generic_names))
        return DrugMatch("brand", key, recs[0][0], gids, names, fuzzy)

    def _generic(self, kind: str, key: str, fuzzy: bool) -> DrugMatch:
        gid = self.generic_stems.get(key) if kind == "generic" else self.substances.get(key)
        return DrugMatch(kind, key, None, (gid,) if gid else (), (self.generic_names.get(gid or "", key),), fuzzy)

    def _word(self, w: str, fuzzy: bool) -> DrugMatch | None:
        if w in self.brands:
            return self._brand(w, fuzzy)
        if w in self.first_brand:
            return self._brand(self.first_brand[w], fuzzy)
        if w in self.generic_stems:
            return self._generic("generic", w, fuzzy)
        if w in self.substances:
            return self._generic("substance", w, fuzzy)
        if w in self.first_generic:
            kind, key = self.first_generic[w]
            return self._generic(kind, key, fuzzy)
        return None

    def lookup(self, text: str | None) -> DrugMatch | None:
        """The medicine named by the first word after the form prefix (a brand, a generic or a substance)."""
        t = _FORM_PREFIX.sub("", text or "", count=1)
        words = re.findall(r"[a-z0-9]+", t.casefold())
        if not words:
            return None
        for key in (" ".join(words[:3]), " ".join(words[:2])):                # a two or three word brand first
            if key in self.brands:
                return self._brand(key, False)
        w = words[0]
        if len(w) < 3:
            return None
        if len(w) == 3:                                                        # a short brand ("Pan 40"): exact brand only
            return self._brand(w, False) if w in self.brands else None
        hit = self._word(w, False)
        if hit or len(w) < 6:
            return hit
        sm = difflib.SequenceMatcher(None, "", w)                             # one misread letter or two
        best, score = None, 0.0
        for n in (len(w) - 1, len(w), len(w) + 1):
            for cand in self.buckets.get((w[0], n), ()):
                sm.set_seq1(cand)
                if sm.real_quick_ratio() < 0.86 or sm.quick_ratio() < 0.86:
                    continue
                r = sm.ratio()
                if r > score:
                    best, score = cand, r
        return self._word(best, True) if best and score >= 0.86 else None


def drugs(path: str | None = None) -> DrugIndex | None:
    d = _dir(path)
    f = d / "cdci_drugs.json" if d else None
    if f is None or not f.is_file():
        return None
    key = f"drugs:{f}"
    with _lock:
        if key not in _cache:
            _cache[key] = DrugIndex(json.loads(f.read_text(encoding="utf-8")))
        return _cache[key]   # type: ignore[return-value]


def drug_lookup(text: str | None, path: str | None = None) -> DrugMatch | None:
    idx = drugs(path)
    return idx.lookup(text) if idx else None


def clear_cache() -> None:
    with _lock:
        _cache.clear()
