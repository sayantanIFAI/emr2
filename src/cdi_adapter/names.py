"""Comparing person names as handwriting gives them: titles, spacing and small misreadings do not make a different person."""
from __future__ import annotations

import difflib
import re

_TITLES = re.compile(r"^(?:mr|mrs|ms|miss|master|baby|dr|smt|shri|sri|sh|late)\b\.?\s*", re.I)


def name_key(name: str | None) -> str:
    """A name for comparing: lower case, no title, letters and single spaces only."""
    n = _TITLES.sub("", (name or "").strip())
    return re.sub(r"\s+", " ", re.sub(r"[^a-z ]", "", n.casefold())).strip()


_NOT_NAME_WORDS = frozenset("yrs yr yrs. yr. years year yo y/o age aged sex gender male female m/f f/m yrsfemale yrsmale".split())
_GLUED_TITLE = re.compile(r"^(mr|mrs|ms|miss|smt|shri|sri|dr)\.(?=[A-Za-z])", re.I)
SEPARATORS = re.compile(r"[\s/,;|]+")             # "Gangopadhyay./72yrs/Female" is three words: the name, the age, the sex


def clean_name(name: str | None) -> str | None:
    """A patient's name is words only. Everything from the first number, or the first age or sex word ("74 yrs Female", "yrs Female", "M/F"),
    is the age and sex written beside the name, never part of it; stray digits inside a word are dropped; a title glued to the first word
    ("Mrs.Sumita") gets its space. ``None`` when no letters are left."""
    text = _GLUED_TITLE.sub(lambda m: m.group(0) + " ", str(name or "").strip())
    words: list[str] = []
    toks = SEPARATORS.split(text)
    for k, tok in enumerate(toks):
        bare = tok.strip(" ,;:|()[]{}\"'")
        low = bare.lower()
        if not bare:
            continue
        if bare[0].isdigit() or low in _NOT_NAME_WORDS or (low in ("m", "f") and k + 1 < len(toks) and toks[k + 1].lower() in ("f", "m")):
            break
        tok = re.sub(r"\d+", "", tok)
        if any(c.isalpha() for c in tok):
            words.append(tok)
    out = " ".join(words).strip(" ,;:|-")
    return out if sum(c.isalpha() for c in out) >= 2 else None


_ORG_WORDS = re.compile(r"\b(?:limited|ltd|pvt|private|hospital|hospitals|clinic|clinics|centre|center|health|healthcare|lifestyle|"
                        r"diagnostic|diagnostics|pharmacy|medical|laboratory|laboratories|nursing|polyclinic|institute|foundation|"
                        r"trust|corporation|enterprises|company)\b", re.I)


def org_like(name: str | None) -> bool:
    """True for a company / hospital name ("Sanjeevani Health and Lifestyle Private Limited"): never a patient's name."""
    return bool(name and _ORG_WORDS.search(name))


def similarity(a: str | None, b: str | None) -> float:
    ka, kb = name_key(a), name_key(b)
    if not ka or not kb:
        return 1.0 if ka == kb else 0.0
    return 1.0 if ka == kb else difflib.SequenceMatcher(None, ka, kb).ratio()


def alike(a: str | None, b: str | None, threshold: float) -> bool:
    return similarity(a, b) >= threshold


def prefer_complete(chosen: str | None, readings: list[str], threshold: float = 0.85) -> str | None:
    """A model often cuts a long name short ("Smita Gupta" for "Smita Gupta Gangopadhyay"). When another reading begins with the
    same words as ``chosen`` and goes on, that fuller reading is the better suggestion. Never shortens, never invents: it only
    picks among the readings made. Titles are ignored; the first of equally full readings wins."""
    if not chosen:
        return chosen
    base = name_key(chosen).split()
    best, best_n = chosen, len(base)
    for r in readings:
        toks = name_key(r).split()
        if len(toks) > best_n and len(toks) <= best_n + 2 and all(alike(a, b, threshold) for a, b in zip(base, toks[:len(base)])):
            best, best_n = r, len(toks)
    return best


def consensus(candidates: list[str], threshold: float = 0.85) -> tuple[str | None, int, int]:
    """``(the name most readings agree on, how many agree, how many readings)``. The shown name is the reading closest to the
    others in the winning group (its medoid); empty readings are ignored; ties go to the earlier reading."""
    cands = [c.strip() for c in candidates if isinstance(c, str) and name_key(c)]
    if not cands:
        return None, 0, 0
    groups: list[list[str]] = []
    for c in cands:
        for g in groups:
            if alike(g[0], c, threshold):
                g.append(c)
                break
        else:
            groups.append([c])
    best = max(groups, key=len)                                   # max keeps the earliest group on a tie
    medoid = max(best, key=lambda x: (sum(similarity(x, y) for y in best), -best.index(x)))
    return medoid, len(best), len(cands)


# Common surnames of the patients (Kolkata and India wide). Used ONLY to offer spellings the front desk can pick from when a surname
# is read badly ("San?ar", "Sanwar" for Sarkar); never to decide a name. A surname that is not here is simply not offered.
COMMON_SURNAMES = tuple("""Sarkar Sarker Sircar Sen Sengupta Sanyal Saha Shaw Sinha Singh Sharma Shah Sheikh Chowdhury Chaudhuri Choudhury
Chaudhary Chatterjee Chattopadhyay Banerjee Bandyopadhyay Mukherjee Mukhopadhyay Ganguly Gangopadhyay Ghosh Ghoshal Bose Basu Das Dasgupta
Dutta Datta Dey Roy Rai Majumder Majumdar Mondal Mandal Biswas Halder Haldar Paul Pal Pramanik Manna Naskar Mitra Mitter Guha Gupta Kundu
Bhattacharya Bhattacharjee Chakraborty Chakrabarti Chakravarty Dhar Laha Nandi Nag Karmakar Kar Kumar Kumari Yadav Verma Mishra Misra
Pandey Tiwari Agarwal Aggarwal Jain Patel Mehta Joshi Reddy Rao Nair Iyer Iyengar Pillai Khan Ali Ahmed Hussain Begum Khatun Bibi Devi
Prasad Thakur Santra Samanta Sahoo Panda Patra Behera Mahato Murmu Soren Tudu Roychowdhury Sankar Shankar Sanwar""".split())


def surname_options(readings: list[str], limit: int = 7) -> list[str]:
    """Spellings of the surname to choose between: the readings made (a reading with a ``?`` for a letter it could not read is not
    offered as it is), then the list names a ``?`` reading fits exactly (``San?ar`` -> Sankar, Sansar), then the list names closest
    to what was read. ``readings`` are surnames (the last word of each reading of the name)."""
    reads = [r for r in dict.fromkeys(x for x in readings if isinstance(x, str) and len(x) >= 3 and re.fullmatch(r"[A-Za-z?]+", x))]
    plain = [r for r in reads if "?" not in r][:3]
    fits: list[str] = []
    close: dict[str, float] = {}
    for r in reads:
        pat = re.compile(re.escape(r.casefold()).replace("\\?", "."))
        for s in COMMON_SURNAMES:
            if "?" in r and pat.fullmatch(s.casefold()):
                fits.append(s)
            ratio = difflib.SequenceMatcher(None, r.casefold().replace("?", ""), s.casefold()).ratio()
            if ratio >= 0.5:
                close[s] = max(close.get(s, 0.0), ratio)
    ranked = fits + [s for s, _ in sorted(close.items(), key=lambda kv: -kv[1])]
    have = {p.casefold() for p in plain}
    listed = [s for s in dict.fromkeys(ranked) if s.casefold() not in have][: max(0, limit - len(plain))]
    return plain + listed

