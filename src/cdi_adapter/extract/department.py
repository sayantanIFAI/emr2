"""Which tests does this kind of clinic order? The department is read from the printed header (the doctor's degree, the clinic's name) and is used
ONLY as a check on what was read, never to add, change or choose a test:

* a test that belongs to a department (OPG: dental; angiogram / CAG / CABG: cardiology; MRI brain: neurology and general medicine ...) written on a
  page of ANOTHER department is marked "unusual for this clinic: check it first" (it stays a candidate, a person decides);
* a test usual for the page's department says so in its reason;
* the common tests (CBC, FBS, PPBS, HbA1c, TSH, FT4, LFT, KFT, lipid profile, urine R/E ...) belong to every department and are never questioned.

No department found (no degree or clinic word in the header) = no note at all. Nothing here guesses a department for a page that does not show one."""
from __future__ import annotations

import re
from typing import Any

# department id -> words that show it in the header (degree, designation, clinic name)
_DEPARTMENTS: dict[str, re.Pattern[str]] = {
    "dental": re.compile(r"(?i)\b(?:bds|mds|dental|dentist|oral\s+(?:and\s+)?(?:dental|surgeon|surgery|medicine)|orthodont\w*|endodont\w*|periodont\w*|maxillofacial)\b"),
    "cardiology": re.compile(r"(?i)\b(?:cardiolog\w*|cardiac|cardiothoracic|interventional|heart)\b"),
    "neurology": re.compile(r"(?i)\b(?:neurolog\w*|neurosurg\w*|neuro\s*sciences?|neurophysician|neuro)\b"),
    "orthopaedics": re.compile(r"(?i)\b(?:ortho\w*|arthroscop\w*|arthroplast\w*|spine\s+surgeon)\b"),
    "ent": re.compile(r"(?i)\b(?:ent\s+clinic|otorhino\w*|otolaryng\w*|ms\s*\(\s*ent\s*\)|ent\s+specialist|ent\s+surgeon)\b"),
    "gynaecology": re.compile(r"(?i)\b(?:gynae\w*|gynec\w*|obstetric\w*|obg|obs\s*&\s*gyn\w*)\b"),
    "ophthalmology": re.compile(r"(?i)\b(?:ophthalm\w*|eye\s+(?:clinic|surgeon|specialist))\b"),
    "paediatrics": re.compile(r"(?i)\b(?:paediatric\w*|pediatric\w*|child\s+(?:health|specialist))\b"),
    "endocrinology": re.compile(r"(?i)\b(?:endocrin\w*|diabet\w*)\b"),
    "rheumatology": re.compile(r"(?i)\b(?:rheumatolog\w*)\b"),
    "general medicine": re.compile(r"(?i)\b(?:general\s+physician|physician|internal\s+medicine|md\s*\(\s*medicine\s*\)|family\s+(?:physician|medicine))\b"),
    "surgery": re.compile(r"(?i)\b(?:general\s+surgery|surgeon|laparoscopic)\b"),
    "veterinary": re.compile(r"(?i)\b(?:veterinar\w*|canine|feline)\b"),
}
_PRIORITY = ["dental", "veterinary", "ent", "cardiology", "neurology", "orthopaedics", "gynaecology", "ophthalmology", "paediatrics", "rheumatology",
             "endocrinology", "surgery", "general medicine"]

# a test that belongs to some departments only: (written name or standard name, departments where it is usual)
_SPECIFIC: list[tuple[re.Pattern[str], frozenset[str]]] = [
    (re.compile(r"(?i)\b(?:opg|orthopantomogram|iopa|bitewing|cbct)\b"), frozenset({"dental"})),
    (re.compile(r"(?i)\b(?:angiogra\w*|angiogram|cag|cabg|ptca|tmt|stress\s+test|holter|ck-?mb|troponin|bnp)\b"), frozenset({"cardiology", "general medicine", "surgery"})),
    (re.compile(r"(?i)\b(?:mri\s*(?:brain|head)|ct\s*(?:brain|head)|eeg|ncv|nerve\s+conduction)\b"), frozenset({"neurology", "general medicine", "paediatrics"})),
    (re.compile(r"(?i)\b(?:mri\s*(?:spine|knee|shoulder|ls)|dexa|bmd)\b"), frozenset({"orthopaedics", "rheumatology", "neurology", "general medicine", "surgery"})),
    (re.compile(r"(?i)\b(?:audiometry|pure\s+tone|tympanometry|x-?ray\s*(?:mastoid|pns|nose))\b"), frozenset({"ent"})),
    (re.compile(r"(?i)\b(?:pap\s*smear|usg\s*pelvi\w*|tvs|hcg|beta\s*hcg)\b"), frozenset({"gynaecology", "general medicine", "surgery"})),
    (re.compile(r"(?i)\b(?:oct|fundus|visual\s+field|fundoscopy)\b"), frozenset({"ophthalmology", "endocrinology", "general medicine"})),
]


def detect(header_texts: list[str] | None) -> str | None:
    """The department the header shows (first match in priority order), or None."""
    text = " ".join(t for t in (header_texts or []) if t)
    if not text.strip():
        return None
    for dept in _PRIORITY:
        if _DEPARTMENTS[dept].search(text):
            return dept
    return None


def header_texts(blocks: list[dict[str, Any]] | None, fraction: float = 0.35) -> list[str]:
    """The PRINTED text of the first page (the letterhead, wherever the clinic puts it: a dental pad prints the doctor's degree at the foot).
    Handwritten lines are never used: MEASURED on a real page, a garbled handwritten "DCH-" made a dental clinic a paediatric one."""
    out = []
    for b in blocks or []:
        rec = b.get("recognition")
        if isinstance(rec, dict) and rec.get("state") and rec.get("state") != "printed":
            continue
        out.append(str(b.get("text") or ""))
    return out


def note(name: str | None, department: str | None) -> tuple[str, str] | None:
    """``("usual" | "unusual", sentence)`` for a test that belongs to some departments only, else None (a common test, or no department)."""
    if not name or not department:
        return None
    for rx, allowed in _SPECIFIC:
        if rx.search(name):
            if department in allowed:
                return "usual", f"usual for a {department} clinic"
            return "unusual", f"not usual for a {department} clinic (it is ordered by {', '.join(sorted(allowed))}): check it first"
    return None
