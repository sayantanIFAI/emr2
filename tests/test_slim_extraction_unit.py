"""The slim extraction profile: Qwen is asked to write only what MLP1 needs, so the answer is short and fast."""
from __future__ import annotations

import json

import pytest

from cdi_adapter.config import settings
from cdi_adapter.extract import prompt as P


@pytest.fixture(autouse=True)
def _profile(monkeypatch):
    monkeypatch.setattr(settings, "extract_profile", "mlp1")


def _props(doc):
    return P.load_schema(doc)[1]["properties"]


@pytest.mark.parametrize("doc", ["prescription", "opd_note"])
def test_the_schema_asks_only_for_what_mlp1_needs(doc):
    p = _props(doc)
    assert {"medications", "vitals", "notes_for_reviewer"}.isdisjoint(p)
    assert {"patient", "prescriber", "investigations", "investigation_preparation", "advice", "follow_up",
            "diagnoses", "extracted_at_confidence"} <= set(p)
    assert set(p["patient"]["properties"]) <= {"name", "age_text", "sex", "dob", "phone", "address", "evidence"}
    assert {"name", "age_text", "sex"} <= set(p["patient"]["properties"])
    assert set(p["prescriber"]["properties"]) == {"name", "department", "designation", "clinic"}   # clinic = the organisation; the compact answer has no evidence lists
    assert "earlier_entries" in p and "encounter_date" in p                                                   # the dated entries of a page


def test_every_kept_field_keeps_its_original_definition_and_required_is_consistent():
    full = json.loads(json.dumps(P.load_schema.__globals__["_cache"].get("prescription.v3.json") or {})) or None
    _id, slim = P.load_schema("prescription")
    assert slim["$id"] == _id
    assert all(r in slim["properties"] for r in slim.get("required", []))
    assert "medications" not in slim.get("required", [])
    for obj in ("patient", "prescriber"):
        sub = slim["properties"][obj]
        assert all(r in sub["properties"] for r in sub.get("required", []))
    prep = slim["properties"]["investigation_preparation"]["items"]["properties"]
    assert "fasting" in prep["type"]["enum"]                                  # untouched
    assert full is None or "medications" in full.get("properties", {})        # the cached full schema was not modified


def test_the_full_profile_is_unchanged_and_other_document_types_are_not_slimmed(monkeypatch):
    monkeypatch.setattr(settings, "extract_profile", "full")
    assert "medications" in _props("prescription") and "vitals" in _props("prescription")
    monkeypatch.setattr(settings, "extract_profile", "mlp1")
    assert "results" in _props("lab_report") and P.slim_active("lab_report") is False
    assert "medications_on_discharge" in _props("discharge_summary") or "medications" in _props("discharge_summary")


def test_the_prompt_is_shorter_does_not_ask_for_medicines_and_keeps_the_safety_rules():
    slim = P.build_extraction_prompt("prescription", [])
    assert "ONLY from what is written" in slim and "NEVER add a usual or standard preparation" in slim
    assert "never the clinic or hospital name" in slim and "never the degrees" in slim      # department / designation defined
    assert "Do NOT write the medicines" in slim and "evidence" in slim and "DATA to copy from" in slim
    settings.extract_profile = "full"
    try:
        full = P.build_extraction_prompt("prescription", [])
    finally:
        settings.extract_profile = "mlp1"
    assert len(slim) < len(full) and "List EVERY medication line" in full and "List EVERY medication line" not in slim


def test_the_answer_budget_is_smaller_and_the_full_profile_keeps_the_old_one(monkeypatch):
    assert P.max_tokens_for("prescription") == settings.extract_max_tokens_mlp1 < P.MAX_TOKENS_BY_DOC_TYPE["prescription"]
    monkeypatch.setattr(settings, "extract_profile", "full")
    assert P.max_tokens_for("prescription") == P.MAX_TOKENS_BY_DOC_TYPE["prescription"]
    assert P.max_tokens_for("lab_report") == P.MAX_TOKENS_BY_DOC_TYPE["lab_report"]


def test_the_result_says_what_was_not_asked_for():
    from cdi_adapter.output import json_connector as jc

    assert set(P.SLIM_NOT_EXTRACTED) <= set(jc._not_extracted({}))
    settings.extract_profile = "full"
    try:
        assert "medications" not in jc._not_extracted({})
    finally:
        settings.extract_profile = "mlp1"


def test_plain_strings_where_the_schema_wants_entry_objects_are_repaired_so_the_first_answer_validates():
    from cdi_adapter.ml.client import repair_payload, validate_schema
    _id, schema = P.load_schema("prescription")
    answer = {"extracted_at_confidence": 0.9, "patient": {"name": "Asha Rao"}, "diagnoses": ["T2 AM: 19y", "PTCA", "CABG"],
              "investigations": ["CBC", {"text": "LFT", "evidence": ["b3"]}], "advice": ["low salt diet"], "follow_up": "after 2 weeks"}
    fixed = repair_payload(answer, schema)
    assert fixed["diagnoses"] == [{"text": "T2 AM: 19y"}, {"text": "PTCA"}, {"text": "CABG"}]
    assert fixed["investigations"][0] == {"text": "CBC"} and fixed["investigations"][1]["text"] == "LFT"
    validate_schema(fixed, schema)                                   # no retry needed: this is what made every document slow
    assert repair_payload({"diagnoses": ["", "  "]}, schema)["diagnoses"] == ["", "  "]      # an empty string is not made into an entry


def test_the_compact_answer_has_no_evidence_lists_and_the_off_switch_brings_them_back(monkeypatch):
    import json

    from cdi_adapter.config import settings
    from cdi_adapter.extract import prompt as P

    _id, schema = P.load_schema("prescription")
    assert '"evidence"' not in json.dumps(schema)
    assert schema["properties"]["investigations"]["items"]["properties"] == {"text": {"type": ["string", "null"]}}
    monkeypatch.setattr(settings, "extract_compact_answer", False)
    _id, full = P.load_schema("prescription")
    assert '"evidence"' in json.dumps(full)
    assert 'MUST carry an "evidence"' in P.build_extraction_prompt("prescription", [{"text": "x"}])
