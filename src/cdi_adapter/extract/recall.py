"""Never lose a test because the page was given the wrong type.

MEASURED on 17 real prescriptions: a dental prescription ("Adv Digital OPG, FBS, BJS CT") was typed ``operative_note``, a hormone clinic's
note was typed ``other`` (its page text holds "TSH"), and a pad with a printed review checklist and handwritten medicines was typed
``lab_report``. None of the prescription steps ran for them, so 7 ordered tests (FBS, BT, CT, TSH and the whole checklist) never reached a
person. A page whose own text holds a test the lab lists place, beside the marks of a prescription, is handled as a prescription.

This only changes which steps run. Every test found is still a candidate that a person checks."""
from __future__ import annotations

import re
from typing import Any

from . import not_lab, test_cluster
from .test_names import looks_like_medicine

RX_TYPES = frozenset({"prescription", "opd_note", "referral"})
_REROUTE_FROM = frozenset({"other", "operative_note", "lab_report"})
_RX_CUES = re.compile(r"(?i)\badv(?:ice|ised)?\b|\br[x\u00d7]\b|\u211e|r[eo]v[il]?ew\s*after|follow\s*up|\bc/o\b|\bcomplain|\binv(?:estigations?)?\b")
_OPERATIVE_WORDS = re.compile(r"(?i)\boperative?\s+note\b|\bprocedure\s+performed\b|\bname\s+of\s+(?:the\s+)?(?:operation|procedure)\b|"
                              r"\bsurgeon\s*[:=]|\banaesthesi|\banesthesi|\bestimated\s+blood\s+loss\b")
_RESULT_TABLE = re.compile(r"(?i)\breference\s+(?:range|interval)\b|\bbio\.?\s*ref|\bnormal\s+range\b|\bunits?\b\s+\bresult\b|\bspecimen\b|\bsample\s+collected\b")


def reroute(doc_type: str | None, blocks: list[dict[str, Any]] | None) -> tuple[str | None, str]:
    """``(new doc type, why)`` when this page should be handled as a prescription, else ``(None, "")``."""
    if doc_type in RX_TYPES or doc_type not in _REROUTE_FROM or not blocks:
        return None, ""
    texts = [str(b.get("text") or "") for b in blocks]
    page = " ".join(texts)
    if _OPERATIVE_WORDS.search(page):
        return None, ""                                    # it says it is an operative note
    found = [f for f in test_cluster.scan(blocks) if f.why in ("exact", "marker", "beside") and not not_lab.entry_reason(f.test)]    # laboratory tests only
    meds = sum(1 for t in texts if t.strip() and looks_like_medicine(t))
    cue = bool(_RX_CUES.search(page))
    if doc_type == "lab_report":
        # a real lab report is a table of results with reference ranges: never a medicine order. A pad with handwritten medicines and a
        # "review after" line is a prescription, whatever the printed form says.
        if meds >= 2 and cue and not _RESULT_TABLE.search(page) and found:
            return "prescription", f"typed lab_report, but {meds} medicine lines and a prescription cue are on the page ({len(found)} tests found in its text)"
        return None, ""
    if found and (cue or meds >= 1 or len(found) >= 2):
        return "prescription", f"typed {doc_type}, but its text holds {len(found)} test(s) the lab lists place ({found[0].test}) beside the marks of a prescription"
    return None, ""
