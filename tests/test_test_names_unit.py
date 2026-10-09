"""A medicine must never be shown as a lab test; a real test must never be taken for a medicine."""
from __future__ import annotations

import pytest

from cdi_adapter.output.json_connector import _capture_flagged  # noqa: F401  (module imports cleanly)
from cdi_adapter.extract.test_names import UNCONFIRMED, is_grounded, is_known_test, is_test_list, looks_like_medicine, split_tests

MEDICINES = [
    "In Candilock (lots)", "In Candilock (10/5)", "In Candizem (lots) 1t 3x10d", "Inj Cardiloc (10/5)",
    "Tab. Montana fx 1 tab at bedtime", "Cap. Austflu (#5)", "Syp Cough Relief 5 ml TDS", "Tab Metformin 500 mg",
    "Inj DNS - 20 ml 0.d x 5d", "Tab Paracetamol 650 mg 1-0-1", "T. Zincovit x 5 days", "Cap Omez BD",
]
TESTS = [
    "CBC", "KFT", "LFT", "Blood: CBC, Urea, Creatinine, FBS, HBA1C, TSH, PT, Na/K+", "CXR-PA", "ECG", "Echo cardiology",
    "COVID-19 RT-PCR", "Blood for fever profile", "X-ray LS spine", "Urine R/E", "HbA1c (fasting)", "Serum creatinine",
    "Fasting blood sugar (fasting 8-10 hrs)", "USG abdomen", "Lipid profile", "2D echo", "T3 T4 TSH", "Vitamin B12",
]


@pytest.mark.parametrize("text", MEDICINES)
def test_a_medicine_is_recognised_as_one(text):
    assert looks_like_medicine(text), text


@pytest.mark.parametrize("text", TESTS)
def test_a_real_test_is_never_taken_for_a_medicine(text):
    assert not looks_like_medicine(text), text
    assert is_known_test(text), text


def test_an_unknown_name_without_medicine_marks_is_left_alone():
    assert not looks_like_medicine("IPOM. Ventral hernia.")             # not a known test, not a medicine: shown, flagged
    assert not looks_like_medicine("") and not looks_like_medicine(None)


def test_a_known_test_wins_over_a_medicine_mark():
    assert not looks_like_medicine("TSH (0.5 mg/dl ref)")               # a number and a unit, but the name is a test


# ---- the doctor's convention: a medicine has a dose / schedule / "x Nd"; a test is only names -----------------

@pytest.mark.parametrize("text", ["Tab Xyz x 10d", "Inj Abc (10/5) x 5 days", "1 tab x 5d", "Syp Foo 5 ml x 7 d", "Cap Bar 1-0-1",
                                  "Tab Xyz ×10d", "In Abc (lots) 1t 3x10d"])
def test_a_dose_schedule_or_x_n_d_duration_marks_a_medicine(text):
    assert looks_like_medicine(text), text


@pytest.mark.parametrize("text,expected", [
    ("CBC/KFT/LFT", True), ("FBS, HbA1c, TSH", True), ("ECG", True), ("Blood: CBC, Urea, Creatinine, FBS", True),
    ("Calcium + Vit D3", False), ("Iron tab", False), ("Tab Xyz 500 mg", False), ("Vitamin B12 injection", False),
    ("CBC x 10d", False), ("Zincovit", False), ("", False), (None, False)])
def test_a_list_of_only_test_abbreviations_is_a_test_list_and_nothing_else_is(text, expected):
    assert is_test_list(text) is expected


@pytest.mark.parametrize("text,expected", [
    ("CBC/KFT/LFT", ["CBC", "KFT", "LFT"]),
    ("Blood: CBC, Urea, Creatinine, FBS", ["CBC", "Urea", "Creatinine", "FBS"]),
    ("FBS, HbA1c and TSH", ["FBS", "HbA1c", "TSH"]),
    ("Urine R/E", ["Urine R/E"]), ("X-ray LS spine", ["X-ray LS spine"]), ("COVID-19 RT-PCR", ["COVID-19 RT-PCR"]),
    ("ECG", ["ECG"]), ("", [])])
def test_a_list_of_tests_is_split_into_one_entry_per_test_and_a_single_test_is_not_cut(text, expected):
    assert split_tests(text) == expected


class _Rec:
    """Stands in for the fact writer: records what would be stored."""

    sess = None

    def __init__(self):
        self.facts = []

    def add(self, *, fact_type, local_text, **kw):
        self.facts.append((fact_type, local_text))
        return len(self.facts)


def test_a_test_list_filed_as_a_medicine_is_stored_as_tests_and_a_real_medicine_is_not(monkeypatch):
    from cdi_adapter.extract import service

    stored = []
    monkeypatch.setattr(service.repo, "insert_medication_detail", lambda sess, fid, **kw: stored.append(kw["drug_text"]))
    c = _Rec()
    service._facts_prescription(c, {
        "medications": [{"drug_text": "CBC/KFT/LFT", "evidence": ["b3"]},                       # tests, wrongly in medications
                        {"drug_text": "Tab Xyz", "strength": "500 mg", "frequency_text": "1-0-1", "evidence": ["b4"]},
                        {"drug_text": "Calcium + Vit D3", "evidence": ["b5"]}],                  # a medicine with no dose
        "investigations": [{"text": "FBS, HbA1c", "evidence": ["b6"]}]})
    tests = [t for k, t in c.facts if k == "investigation_order"]
    assert tests == ["CBC", "KFT", "LFT", "FBS", "HbA1c"]
    assert any("Xyz" in m for m in stored) and any("Calcium" in m for m in stored) and len(stored) == 2


# ---- a name that is not a test is set aside, never listed as one --------------------------------------------

NOT_TESTS = ["Cardiology", "Intravenous fluid therapy", "Candilock (lot)", "Candilock", "IPOM. Ventral hernia.", "Reaplerology"]
REAL_TESTS = ["Dengue NS1", "Serum ferritin", "Vit D3", "Holter monitoring", "Pap smear", "2D echo", "Spirometry", "TFT", "ABG",
              "Urine culture", "Fasting lipid profile", "Mantoux test", "Doppler leg"]


@pytest.mark.parametrize("text", NOT_TESTS)
def test_a_name_with_no_test_in_it_is_not_known_as_a_test(text):
    assert not is_known_test(text), text


@pytest.mark.parametrize("text", REAL_TESTS)
def test_less_common_real_tests_are_still_recognised(text):
    assert is_known_test(text), text


def test_the_result_sets_unrecognised_entries_aside_and_keeps_real_tests(monkeypatch):
    from cdi_adapter.output import json_connector as jc
    from cdi_adapter.extract.test_names import UNRECOGNISED

    facts = [{"fact_type": "investigation_order", "id": str(i), "local_text": t, "status": "needs_check",
              "code": None, "confidence_overall": 0.5}
             for i, t in enumerate(["CBC", "Cardiology", "In Candilock (lots)", "Candilock (lot)"])]
    res = jc.build_result(jc.ResultInputs(document={"id": "d", "status": "validated", "original_filename": "x.jpg"},
                                          facts=facts, blocks=[], pages=[], payload={}))
    by = {t["as_written"]: t for t in res["lab_tests"]}
    assert by["CBC"]["status"] != "rejected" and not (by["CBC"].get("reason") or "").startswith(UNRECOGNISED)
    assert by["Cardiology"]["reason"].startswith(UNRECOGNISED)
    assert by["Candilock (lot)"]["reason"].startswith(UNRECOGNISED)
    assert by["In Candilock (lots)"]["status"] == "rejected"


def test_a_vitamin_with_a_dose_is_a_medicine_but_the_same_name_alone_is_a_test():
    assert looks_like_medicine("Vit D3 60000 IU")            # "iu" is not a test name: the dose decides
    assert looks_like_medicine("Calcium 500 mg x 30d")
    assert not looks_like_medicine("Vit D3") and is_known_test("Vit D3")


# ---- a test nothing on the page supports is not listed as one ------------------------------------------------

PAGE = "Dr A Sen  Patient Anil  Adv: CBC, FBS, HbA1c   ECG  Blood for fever profile  review after 2 weeks"


@pytest.mark.parametrize("test", ["CBC", "FBS", "HbA1c", "ECG", "Blood for fever profile", "Blood for fever profle",
                                  "K+", "CBC and FBS"])
def test_a_test_whose_words_are_on_the_page_is_grounded(test):
    assert is_grounded(test, PAGE), test


@pytest.mark.parametrize("test", ["Liver function tests", "Thyroid function tests", "Chest X-ray", "Holter monitoring",
                                  "Abdominal ultrasound", "Urinalysis", "Complete blood count"])
def test_a_plausible_test_nothing_on_the_page_supports_is_not_grounded(test):
    assert not is_grounded(test, PAGE), test


def test_the_result_does_not_list_a_test_the_page_does_not_support(monkeypatch):
    from cdi_adapter.output import json_connector as jc

    facts = [{"fact_type": "investigation_order", "id": str(i), "local_text": t, "status": "needs_check", "code": None,
              "confidence_overall": 0.5} for i, t in enumerate(["CBC", "Holter monitoring"])]
    blocks = [{"id": "1", "text": "Adv: CBC, FBS"}]
    res = jc.build_result(jc.ResultInputs(document={"id": "d", "status": "validated", "original_filename": "x.jpg"},
                                          facts=facts, blocks=blocks, pages=[], payload={}))
    by = {t["as_written"]: t for t in res["lab_tests"]}
    assert not (by["CBC"].get("reason") or "").startswith(UNCONFIRMED)
    assert by["Holter monitoring"]["reason"].startswith(UNCONFIRMED)


# ---- a list of tests written on one line is cut into its tests, never kept whole as well
@pytest.mark.parametrize("written,expected", [
    ("HbA1c/FBS/PPBS/S LIPASE TSH/FT4", ["HbA1c", "FBS", "PPBS", "S LIPASE", "TSH", "FT4"]),
    ("HbA1c\FBS\PPBS", ["HbA1c", "FBS", "PPBS"]),
    ("HbA1c|FBS|TSH", ["HbA1c", "FBS", "TSH"]),
    ("TSH + FT4 + FT3", ["TSH", "FT4", "FT3"]),
    ("CBC, LFT; KFT & TSH", ["CBC", "LFT", "KFT", "TSH"]),
    ("HbA1c - FBS - PPBS", ["HbA1c", "FBS", "PPBS"]),
    ("HbA1c-FBS-TSH", ["HbA1c", "FBS", "TSH"]),
    ("HbA1c. FBS. PPBS", ["HbA1c", "FBS", "PPBS"]),
    ("S LIPASE TSH", ["S LIPASE", "TSH"]),
    ("HbA1c FBS PPBS", ["HbA1c", "FBS", "PPBS"]),
])
def test_a_list_of_tests_on_one_line_is_cut_into_its_tests(written, expected):
    from cdi_adapter.extract.test_names import split_tests
    assert split_tests(written) == expected


@pytest.mark.parametrize("written", ["A/G ratio", "Urine R/E", "Urine R/E & C/S", "C/S", "CK-MB", "D-dimer", "Vitamin B-12", "S. Lipase",
                                     "X-ray LS spine", "Blood urea nitrogen", "Lipid profile", "Urine culture", "Serum creatinine"])
def test_a_name_that_is_one_test_is_never_cut(written):
    from cdi_adapter.extract.test_names import split_tests
    assert split_tests(written) == [written]


# ---- a test the doctor ticked is not rejected because the text reader ran the ticked words together
@pytest.mark.parametrize("test", ["CBC", "CRP", "LFT", "Creatinine", "KFT"])
def test_ticked_words_run_together_by_the_text_reader_still_support_the_test(test):
    from cdi_adapter.extract.test_names import is_grounded
    page = "Review after. 2 months CBCCRP vLFT ? Creatinine vKFT"                      # what the real page read as
    assert is_grounded(test, page)


def test_a_test_nothing_on_the_page_supports_is_still_not_grounded():
    from cdi_adapter.extract.test_names import is_grounded
    assert not is_grounded("HbA1c", "Review after. 2 months CBCCRP vLFT Creatinine")
    assert not is_grounded("TSH", "Tab Metformin 500 mg 1-0-1 after food")


# ---- any separator the doctor used is used to find ALL the tests: ticks, bullets, a stuck "v", words run together
@pytest.mark.parametrize("written,expected", [
    ("✓CBC ✓CRP ✓LFT ✓Creatinine", ["CBC", "CRP", "LFT", "Creatinine"]),
    ("✔ CBC ✔ CRP ✔ LFT", ["CBC", "CRP", "LFT"]),
    ("• CBC • CRP", ["CBC", "CRP"]),
    ("* HbA1c * FBS * PPBS", ["HbA1c", "FBS", "PPBS"]),
    ("CBC → CRP → LFT", ["CBC", "CRP", "LFT"]),
    ("vCBC vCRP vLFT vCreatinine", ["CBC", "CRP", "LFT", "Creatinine"]),
    ("CBCCRP vLFT", ["CBC", "CRP", "LFT"]),
    ("CBCCRP", ["CBC", "CRP"]),
    ("vLFT", ["LFT"]),
    ("CBC CRP LFT Creatinine", ["CBC", "CRP", "LFT", "Creatinine"]),
])
def test_every_test_in_a_list_is_found_whatever_separates_them(written, expected):
    from cdi_adapter.extract.test_names import split_tests
    assert split_tests(written) == expected


@pytest.mark.parametrize("written", ["VLDL", "Vitamin D", "Vitamin B12", "Serum creatinine", "Lipid profile", "Urine culture", "A/G ratio",
                                     "CK-MB", "Blood urea nitrogen", "X-ray LS spine", "vitamin"])
def test_a_name_that_is_one_test_is_still_never_cut_by_the_new_rules(written):
    from cdi_adapter.extract.test_names import split_tests
    assert split_tests(written) == [written]


# ---- the similarity score is shown, the gate result decides, and a score never rejects a test
def test_the_page_support_score_is_one_for_a_word_on_the_page_lower_for_run_together_and_low_for_nothing():
    from cdi_adapter.extract.test_names import page_support
    page = "Review after 2 months CBCCRP vLFT Creatinine"
    assert page_support("Creatinine", page) == 1.0
    assert page_support("CBC", page) == 0.9 and page_support("LFT", page) == 0.9              # inside a run-together word
    assert page_support("HbA1c", page) < 0.5 and page_support("HbA1c", "") is None and page_support("", page) is None


def test_a_test_the_lists_place_is_never_rejected_and_a_medicine_line_with_a_loose_test_word_still_is(monkeypatch):
    import test_json_connector_unit as T
    from cdi_adapter.extract import lab_resolve
    from cdi_adapter.output import json_connector as jc
    blocks = [{"text": "unrelated words only", "page_id": "p1"}]
    med = T._fact("investigation_order", "Tab Sompraz (40 mg) 1 tab OD AC x load", state="in_review", code_status="unmapped", conf=0.4)
    r = jc.build_result(T._inputs([med], payload=T.PAYLOAD, blocks=blocks))
    assert r["lab_tests"][0]["status"] == "rejected" and r["lab_tests"][0]["gate_recognised"] is False      # "load" (viral load) is not a lab-list hit
    placed = lab_resolve.Resolved("test", "2160-0", "bound", ("2160-0",), "Creatinine", "mapping", False)
    monkeypatch.setattr(lab_resolve, "resolve", lambda x: placed if "creat" in (x or "").lower() else None)
    fact = T._fact("investigation_order", "Tab Creat 500 mg 1 tab OD", state="in_review", code_status="unmapped", conf=0.4)    # looks like a medicine
    t = jc.build_result(T._inputs([fact], payload=T.PAYLOAD, blocks=blocks))["lab_tests"][0]
    assert t["gate_recognised"] is True and t["page_support"] is not None and t["page_support"] < 0.5
    # NEW RULE (owner): a test no line reader saw and no second look agrees with is not listed (MEASURED: the whole-page answer adds INR to "PT / APTT")
    from cdi_adapter.extract.test_names import SECOND_LOOK, UNCONFIRMED
    assert t["status"] == "rejected" and t["reason"].startswith(UNCONFIRMED)
    t2 = jc.build_result(T._inputs([fact], payload={**T.PAYLOAD, "_corroborated": ["Tab Creat 500 mg 1 tab OD"]}, blocks=blocks))["lab_tests"][0]
    assert t2["status"] != "rejected" and t2["reason"].startswith(SECOND_LOOK)       # a second look read it again in enough views: listed


def test_an_unseen_test_the_model_confirms_when_asked_directly_is_listed_for_a_check(monkeypatch):
    import test_json_connector_unit as T
    from cdi_adapter.extract import lab_resolve
    from cdi_adapter.extract.test_names import UNCONFIRMED
    from cdi_adapter.output import json_connector as jc

    placed = lab_resolve.Resolved("test", "2160-0", "bound", ("2160-0",), "Creatinine", "mapping", False)
    monkeypatch.setattr(lab_resolve, "resolve", lambda x: placed if "creat" in (x or "").lower() else None)
    fact = T._fact("investigation_order", "Creatinine", state="in_review", code_status="unmapped", conf=0.4)
    blocks = [{"text": "unrelated words only", "page_id": "p1"}]
    sure = jc.build_result(jc.ResultInputs(**{**T._inputs([fact], payload={**T.PAYLOAD, "_verify": {"Creatinine": 0.92}}, blocks=blocks).__dict__}))["lab_tests"][0]
    assert sure["status"] == "needs_check" and "asked directly" in sure["reason"]
    unsure = jc.build_result(jc.ResultInputs(**{**T._inputs([fact], payload={**T.PAYLOAD, "_verify": {"Creatinine": 0.2}}, blocks=blocks).__dict__}))["lab_tests"][0]
    assert unsure["status"] == "rejected" and unsure["reason"].startswith(UNCONFIRMED)
