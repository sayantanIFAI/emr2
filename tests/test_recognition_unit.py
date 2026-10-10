"""Recognition v2 pure logic: grammar, engine disagreement, grounding, alias cascade,
plausibility, doctor matching, gate policy, evidence hierarchy, quality gate, regions."""
from __future__ import annotations

import io

import cv2
import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFont

from cdi_adapter.ocr.rapid import OcrLine
from cdi_adapter.recognition import grammar as G
from cdi_adapter.recognition import plausibility as P
from cdi_adapter.recognition import practitioner as DR
from cdi_adapter.recognition.alias import KB, candidates, mark_collisions, norm_alias
from cdi_adapter.recognition.disagreement import (AGREE, DISAGREE, NONE, SINGLE,
                                                  compare_engines, compare_texts,
                                                  strength_fits_product)
from cdi_adapter.recognition.engines import QWEN_LINE_PROMPT, Reading
from cdi_adapter.recognition.grounding import ground_value, numbers_supported
from cdi_adapter.recognition.hierarchy import assess_fact
from cdi_adapter.recognition.quality import assess
from cdi_adapter.recognition.regions import HANDWRITTEN, PRINTED, detect_regions
from cdi_adapter.validate.policy import rule_for, worst_state


# ---------------- grammar ----------------
def test_numeric_context_only_inside_numbers():
    assert G.normalize_numeric_context("Telma 4O") == "telma 40"
    assert G.normalize_numeric_context("SOS") == "sos"          # a word is never rewritten
    assert G.numbers_in("Telma 4O mg 1-0-1") == ["40", "1", "0", "1"]


@pytest.mark.parametrize("txt,ok", [("1-0-1", True), ("½-0-½", True), ("1-1-1-1", True),
                                    ("1-0", False), ("9-0-1", False), ("a-0-1", False)])
def test_dose_pattern(txt, ok):
    assert G.parse_dose_pattern(txt).valid is ok


def test_frequency_and_duration():
    assert G.parse_frequency("BD").normalized == 2
    assert G.parse_frequency("1-0-1").normalized == 2
    assert not G.parse_frequency("thrice-ish").valid
    assert G.parse_duration("x 5 days").normalized == 5
    assert G.parse_duration("x 2 wk").normalized == 14
    assert G.parse_duration("x 1/52").normalized == 7


def test_medication_grammar_flags_bad_unit():
    bad = G.check_medication_fields({"strength_num": 40, "strength_unit": "kg"})
    assert bad and bad[0][0] == "strength"


# ---------------- engine disagreement ----------------
def _r(engine, text, conf=0.9, error=None):
    return Reading(engine, "v", text, conf, error=error)


def test_engines_agree_prefers_literal_text():
    v = compare_engines([_r("reader-a", "Telma 40 1-0-1"), _r("qwen2.5-vl", "Telma 4O 1-0-1", None)])
    assert v.state == AGREE and v.display_text == "Telma 40 1-0-1"


def test_number_difference_is_material_even_when_text_is_close():
    agree, det = compare_texts("Telma 40", "Telma 20")
    assert not agree and not det["numbers_equal"]
    v = compare_engines([_r("reader-a", "Telma 40"), _r("qwen2.5-vl", "Telma 20", None)])
    assert v.state == DISAGREE and "⟂" in v.display_text


def test_single_and_no_reading():
    assert compare_engines([_r("reader-a", "Telma 40"), _r("qwen2.5-vl", "", None, "down")]).state == SINGLE
    assert compare_engines([_r("reader-a", "", None, "x"), _r("qwen2.5-vl", "", None, "y")]).state == NONE


def test_qwen_prompt_never_carries_other_readings():
    assert "{" not in QWEN_LINE_PROMPT and "reader-a" not in QWEN_LINE_PROMPT.lower()


def test_marketed_strength():
    assert strength_fits_product(40, {"strengths_available": [20, 40, 80]}) is True
    assert strength_fits_product(30, {"strengths_available": [20, 40, 80]}) is False
    assert strength_fits_product(30, {}) is None


# ---------------- grounding ----------------
def test_grounding_vetoes_unsupported_number():
    assert ground_value("Telma 40", ["Tab Telma 4O 1-0-1"]).grounded
    g = ground_value("Telma 20", ["Tab Telma 40 1-0-1"])
    assert not g.grounded and not g.numbers_ok


def test_grounding_uses_reread_only_when_needed():
    calls = []
    ground_value("Telma 40", ["Telma 40"], reread=lambda: calls.append(1) or [])
    assert calls == []
    g = ground_value("Sorbitrate 5", ["other line"], reread=lambda: ["Sorbitrate 5 mg"])
    assert g.grounded and g.method == "reread"


def test_numbers_multiset():
    assert numbers_supported("1-0-1", "1-0-1 x5d")
    assert not numbers_supported("1-1-1", "1-0-1")


# ---------------- alias cascade + plausibility ----------------
@pytest.fixture()
def kb():
    concepts = {
        "drug:telma-40": {"id": "drug:telma-40", "domain": "drug", "canonical_name": "Telma 40",
                          "attrs": {"indications": ["hypertension"], "strengths_available": [20, 40, 80]}},
        "drug:telma-20": {"id": "drug:telma-20", "domain": "drug", "canonical_name": "Telma 20",
                          "attrs": {"indications": ["hypertension"]}},
        "drug:sorbitrate-5": {"id": "drug:sorbitrate-5", "domain": "drug",
                              "canonical_name": "Sorbitrate 5", "attrs": {"indications": ["cardiac"]}},
        "drug:pregabalin-75": {"id": "drug:pregabalin-75", "domain": "drug",
                               "canonical_name": "Pregabalin 75",
                               "attrs": {"indications": ["neuropathic", "msk_spine"]}},
    }
    aliases = [{"concept_id": c, "alias": concepts[c]["canonical_name"],
                "alias_norm": norm_alias(concepts[c]["canonical_name"]), "alias_class": "A",
                "practitioner_id": None} for c in concepts]
    groups = {"msk_spine": ["back pain", "lumbar", "spondylosis"], "cardiac": ["chest pain", "angina"],
              "pain_fever": ["pain", "fever"]}
    return KB(concepts, aliases, groups)


def test_alias_levels_and_number_safety(kb):
    assert candidates(kb, "Telma 40", "drug")[0]["resolved_by_level"] == 1
    assert candidates(kb, "Tab Telma 40 mg", "drug")[0]["resolved_by_level"] == 2
    c = candidates(kb, "Telma4O", "drug")
    assert c[0]["concept_id"] == "drug:telma-40"
    # numbers are never fuzzed across strengths
    assert all(x["concept_id"] != "drug:telma-40" for x in candidates(kb, "Telma 30", "drug"))


def test_collision_marks_ties():
    cands = [{"concept_id": "a", "score": 0.9}, {"concept_id": "b", "score": 0.89}]
    assert mark_collisions(cands) and all(c.get("collision") for c in cands)


def test_spine_context_flags_cardiac_drug(kb):
    ctx = P.context_groups(["Low back pain", "Lumbar spondylosis"], kb.groups)
    assert "msk_spine" in ctx
    assert P.assess(["cardiac"], ctx).status == "mismatch"
    assert P.assess(["neuropathic", "msk_spine"], ctx).status == "fits"
    ranked = P.rerank([{"score": 0.95, "attrs": {"indications": ["cardiac"]}},
                       {"score": 0.90, "attrs": {"indications": ["msk_spine"]}}], ctx)
    assert ranked[0]["attrs"]["indications"] == ["msk_spine"] and len(ranked) == 2


# ---------------- doctor matching ----------------
MASTERS = [
    {"id": "1", "registration_number": "12345", "name_full": "Dr. A. Sen",
     "name_aliases": ["A Sen"], "specialty": ["General Medicine"], "department": "General Medicine",
     "designation": "Consultant Physician"},
    {"id": "2", "registration_number": "12346", "name_full": "Dr. Arindam Sen",
     "name_aliases": [], "specialty": ["Orthopaedics"], "department": "Orthopaedics",
     "designation": "Consultant Orthopaedic Surgeon"},
    {"id": "3", "registration_number": "22001", "name_full": "Dr. R. Iyer",
     "name_aliases": [], "specialty": ["Biochemistry"], "department": "Biochemistry",
     "designation": "Consultant Biochemist"},
]


def test_registration_number_links():
    m = DR.match(MASTERS, name="Dr. A. Sen", reg_no=None,
                 header_texts=["Dr. A. Sen (Reg. No. 12345)"])
    assert m.linked and m.practitioner_id == "1" and m.method == "registration_no"


def test_ambiguous_name_is_not_linked_but_shows_db_candidates():
    m = DR.match(MASTERS, name="Dr A Sen", reg_no=None, header_texts=[])
    assert not m.linked and m.db_name in ("Dr. A. Sen", "Dr. Arindam Sen")
    assert len(m.candidates) >= 2


def test_unknown_doctor():
    m = DR.match(MASTERS, name="Dr. Nobody Known", reg_no=None, header_texts=[])
    assert not m.linked and m.practitioner_id is None


# ---------------- policy ----------------
def test_structural_rules_cannot_be_loosened():
    assert rule_for("medication", "disagree").action == "review"
    assert rule_for("condition", "no_reading").action == "review"
    assert rule_for("medication", "single_engine").action == "review"
    assert rule_for("condition", "printed").action == "gate"
    assert worst_state(["printed", "agree", "disagree"]) == "disagree"
    assert worst_state([None]) == "legacy"


# ---------------- hierarchy ----------------
def _block(text, state, engines=None):
    return {"text": text, "recognition": {"state": state, "engines": engines or {}},
            "observation_ids": []}


def test_hierarchy_disagreement_and_grounding_veto():
    f = {"fact_type": "medication", "local_text": "Telma 20"}
    md = {"drug_text": "Telma", "strength_num": 20, "frequency_code": "1-0-1"}
    a = assess_fact(f, md, [_block("Telma 40 ⟂ Telma 20", "disagree",
                                   {"reader-a": "Telma 40 1-0-1", "qwen2.5-vl": "Telma 20 1-0-1"})], [])
    codes = {c for _s, c, _m in a.findings}
    assert a.evidence_state == "disagree" and "engine-disagreement" in codes
    b = assess_fact(f, md, [_block("Tab Telma 40 1-0-1", "agree",
                                   {"reader-a": "Tab Telma 40 1-0-1"})], [])
    assert "grounding-failed" in {c for _s, c, _m in b.findings}


def test_hierarchy_indication_mismatch_blocks_medication():
    f = {"fact_type": "medication", "local_text": "Sorbitrate 5"}
    md = {"drug_text": "Sorbitrate 5", "strength_num": 5, "frequency_code": "SOS"}
    cands = [{"id": "c1", "concept_id": "drug:sorbitrate-5", "normalized_text": "Sorbitrate 5",
              "is_selected": True, "collision": False, "resolved_by_level": 1, "score": 1.0,
              "alias_class": "A",
              "evidence": {"plausibility": {"status": "mismatch", "reason": "cardiac vs msk_spine"},
                           "attrs": {"strengths_available": [5, 10]}}}]
    a = assess_fact(f, md, [_block("Sorbitrate 5 SOS", "printed")], cands)
    assert ("blocker", "med-indication-mismatch") in {(s, c) for s, c, _m in a.findings}
    assert a.winning_candidate_ids == ["c1"]


# ---------------- L8 adjudication (advisory) ----------------
class _Adj:
    def __init__(self, answers):
        self.answers, self.prompts = list(answers), []

    def vlm_generate(self, image, prompt, *, max_tokens=8, json_schema=None):
        self.prompts.append(prompt)
        return self.answers.pop(0)


def test_adjudication_needs_both_orders_to_agree():
    from cdi_adapter.recognition.adjudicate import adjudicate

    texts = ["Pregabalin 75", "Pregabalin 150"]
    c = _Adj(["A", "B"])            # A/B order picks #0, B/A order picks #0 -> consistent
    a = adjudicate(b"png", texts, client=c)
    assert a.verdict == "prefers" and a.preferred_text == "Pregabalin 75"
    assert "A: Pregabalin 75" in c.prompts[0] and "A: Pregabalin 150" in c.prompts[1]
    assert adjudicate(b"png", texts, client=_Adj(["A", "A"])).verdict == "inconsistent"
    assert adjudicate(b"png", texts, client=_Adj(["NEITHER", "neither."])).verdict == "neither"
    assert adjudicate(b"png", texts, client=_Adj(["???", "B"])).verdict == "inconsistent"


def test_adjudication_failure_is_harmless():
    from cdi_adapter.recognition.adjudicate import adjudicate

    class Boom:
        def vlm_generate(self, *a, **k):
            raise RuntimeError("gateway down")

    a = adjudicate(b"png", ["x 1", "x 2"], client=Boom())
    assert a.verdict == "failed" and a.preferred_index is None


def test_adjudication_never_resolves_disagreement():
    f = {"fact_type": "medication", "local_text": "Pregabalin 75"}
    md = {"drug_text": "Pregabalin", "strength_num": 75, "frequency_code": "0-0-1"}
    blk = {"text": "Pregabalin 75 0-0-1 ⟂ Pregabalin 150 0-0-1", "observation_ids": [],
           "recognition": {"state": "disagree",
                           "engines": {"reader-a": "Pregabalin 75 0-0-1",
                                       "qwen2.5-vl": "Pregabalin 150 0-0-1"},
                           "adjudication": {"verdict": "prefers", "preferred_index": 0,
                                            "preferred_text": "Pregabalin 75 0-0-1"}}}
    a = assess_fact(f, md, [blk], [])
    msg = next(m for s, c, m in a.findings if c == "engine-disagreement")
    assert a.evidence_state == "disagree" and "advisory" in msg
    assert rule_for("medication", a.evidence_state).action == "review"


# ---------------- quality gate + regions ----------------
def _page(text_lines, blur=0):
    im = Image.new("RGB", (1400, 900), "white")
    d = ImageDraw.Draw(im)
    try:
        f = ImageFont.truetype("arial.ttf", 34)
    except OSError:
        f = ImageFont.load_default()
    for i, t in enumerate(text_lines):
        d.text((80, 80 + i * 90), t, fill="black", font=f)
    arr = cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)
    if blur:
        arr = cv2.GaussianBlur(arr, (blur, blur), 0)
    return arr


def test_quality_gate_passes_sharp_and_holds_blurred():
    assert assess(_page(["Rx Telma 40 1-0-1", "Pregabalin 75 0-0-1"])).passed
    bad = assess(_page(["Rx Telma 40 1-0-1"], blur=61))
    assert not bad.passed and any("blur" in r for r in bad.reasons)


def test_regions_printed_vs_unexplained_ink():
    img = _page(["Tab Telma 40 1-0-1"])
    # a scribble no OCR line explains -> handwriting engines
    cv2.polylines(img, [np.array([[100, 400], [160, 380], [220, 420], [300, 390], [380, 415]])],
                  False, (0, 0, 0), 4)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ocr = [OcrLine("Tab Telma 40 1-0-1", [80, 80, 460, 120], 0.97, [])]
    regs = detect_regions(gray, ocr)
    kinds = {r.kind for r in regs}
    assert PRINTED in kinds and HANDWRITTEN in kinds


def test_crop_png_roundtrip():
    from cdi_adapter.recognition.regions import crop_png

    png = crop_png(_page(["x"]), [10, 10, 110, 60], margin_frac=0.1, min_margin_px=2)
    assert Image.open(io.BytesIO(png)).size[0] >= 100
