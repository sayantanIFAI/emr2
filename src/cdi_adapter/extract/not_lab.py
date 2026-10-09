"""Is this entry a LABORATORY test? The rules that keep imaging, ECG, physiotherapy, a clinic's printed list of services and the
words of a medicine line out of the tests, with the reason, so nothing is lost silently.

MEASURED on 17 real prescriptions (92 entries shown as tests): "Physio - UST (IFT)" produced "UST" -> AST (SGOT) and "RFT" -> creatinine,
"MRI Brain - N" and "EEG: N" (results) and "-MRI LS spine" were listed, a printed footer "CT Scan, X-Ray, OPG ..." produced "CT" (clotting
time) on a page that orders none, and "Zincovit" produced "Zinc". None is a laboratory test the doctor ordered.

Everything here only ever REMOVES an entry from the tests (the entry is kept in the result as ``rejected`` with its reason); it never
invents one. ``settings.lab_tests_only`` (default on) turns the imaging / ECG rule on: a site that wants ECG and imaging listed as
investigations switches it off."""
from __future__ import annotations

import re
from typing import Any

from .test_names import looks_like_medicine, split_tests

_IMAGING = re.compile(
    r"(?i)(?<![a-z0-9])(?:mri|x-?\s?rays?|cxr|usg|ultra\s?sound|ultrasonography|sonography|opg|dexa|bmd|mammo(?:gram|graphy)?|doppler|"
    r"angio(?:graphy|gram)?|echo(?:\s?cardio\w*)?|2d\s?echo|ncv|fundoscopy|endoscopy|colonoscopy|"
    r"ct\s*[-\s]?\s*(?:scan|abdomen|chest|brain|head|spine|kub|thorax|pns))(?![a-z0-9])")
_CARDIO_NEURO = re.compile(r"(?i)(?<![a-z0-9])(?:ecg|ekg|eeg|emg|tmt|holter|spirometry|pft)(?![a-z0-9])")
_PHYSIO = re.compile(r"(?i)(?<![a-z0-9])(?:physio(?:therapy)?|ust|ift|tens|kegel|exercises?)(?![a-z0-9])")
# a printed line that lists a clinic's services ("Pain & Laser Clinic ● Sugar Clinic ● Dental Clinic ● MRI ● CT Scan ● X-Ray")
_SERVICE_WORDS = re.compile(r"(?i)\b(?:clinic|centre|center|endoscopy|ultrasonography|mammography|physiotherapy|pharmacy|consultation|"
                            r"health check|dental|laser|eye|ent|echocardiography|pathology|microbiology|biopsy|doppler)\b")
_BULLETS = re.compile(r"[●•·▪■|]")
_MEDICINE_MARK = re.compile(r"(?i)\b(?:tabs?|tablets?|caps?|capsules?|syp|syr|inj|drops?|oint|cream|gel|susp)\b\.?|\d\s*(?:mg|mcg|ml|gm|iu)\b|\(\s*\d")


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", (s or "").casefold())


def entry_reason(text: str | None, *, lab_only: bool = True) -> str | None:
    """Why this entry is not a laboratory test, judged from its own words (``None`` = it may be one)."""
    t = text or ""
    if not t.strip():
        return None
    if _PHYSIO.search(t):
        return "physiotherapy, not a laboratory test"
    if lab_only and _IMAGING.search(t):
        return "imaging, not a laboratory test"
    if lab_only and _CARDIO_NEURO.search(t):
        return "ECG / EEG type test, not a laboratory test"
    return None


def _blocks_with(key: str, blocks: list[dict[str, Any]]) -> list[tuple[str, bool]]:
    """(text, whole_word) of every page line that holds the name; whole_word False = only as the start of a longer word ("zinc" in "Zincovit")."""
    k = re.escape(re.sub(r"\s+", " ", key.strip().casefold()))
    if len(_norm(key)) < 2:
        return []
    whole = re.compile(r"(?<![a-z0-9])" + k.replace(r"\ ", r"[\s.\-]*") + r"(?![a-z0-9])")
    prefix = re.compile(r"(?<![a-z0-9])" + k + r"[a-z]{2,}") if len(_norm(key)) >= 4 else None
    out: list[tuple[str, bool]] = []
    for b in blocks:
        t = str(b.get("text") or "")
        low = t.casefold()
        if whole.search(low):
            out.append((t, True))
        elif prefix is not None and prefix.search(low):
            out.append((t, False))
    return out


def line_reason(key: str, blocks: list[dict[str, Any]] | None, *, lab_only: bool = True) -> str | None:
    """Why a name came from a line that is not an order for a laboratory test: EVERY page line that holds it is an imaging / physio line, a
    printed list of the clinic's services, or a medicine line. A name that also stands in a line of its own is left alone."""
    if not blocks:
        return None
    hits = _blocks_with(key, blocks)
    if not hits:
        return None
    reasons: list[str] = []
    k = _norm(key)
    for text, whole in hits:
        # judge the name's OWN piece of the line: "Digital OPG, FBS, BJS CT" is imaging for OPG and a list of tests for FBS and CT
        own = [p for p in (split_tests(text) or [text]) if k in _norm(p)] or [text]
        r = entry_reason(" ".join(own), lab_only=lab_only)
        if r:
            reasons.append(f"written in a line that is {r.split(',')[0]}")
        elif len(_BULLETS.findall(text)) >= 2 or len(_SERVICE_WORDS.findall(text)) >= 3:
            reasons.append("part of the clinic's printed list of services")
        elif ((not whole and (looks_like_medicine(text) or _MEDICINE_MARK.search(text)))
              or (whole and re.search(r"(?i)\b(?:tabs?|caps?|inj|syp|syr)\b\.?\s*" + re.escape(key.strip()) + r"(?![a-z0-9])", text))):
            # the name is the start of a brand ("Zincovit") or directly follows a form word ("Tab Zinc"); a test written beside a medicine stays
            reasons.append("part of a medicine line")
        else:
            return None                        # at least one line is a plain test line: keep it
    return reasons[0] if reasons else None


def classify(names: list[str], blocks: list[dict[str, Any]] | None, *, lab_only: bool = True) -> dict[str, str]:
    """``{name: reason}`` for the names that are not laboratory tests the doctor ordered."""
    out: dict[str, str] = {}
    for n in names:
        r = entry_reason(n, lab_only=lab_only) or line_reason(n, blocks, lab_only=lab_only)
        if r:
            out[n] = r
    return out
