"""Fixes from the 24 Oct 2026 parallel test of 13 real pages and the owner's reports (USG W/A, TORCH, ABO + Rh typing, TVS, Echo-2D, Karyotype).
Each case is a MEASURED failure on a real page; the expected values are what the page says."""
import pytest

from cdi_adapter import names as N
from cdi_adapter.extract import lab_resolve, not_lab, service, test_cluster as T
from cdi_adapter.extract.test_names import split_tests

PRINTED = {"state": "printed"}


def std(text):
    r = lab_resolve.resolve(text)
    return r.long_name if r else None


# ---------------------------------------------------------------- names the lists did not know
@pytest.mark.parametrize("written,standard", [
    ("USG W/A", "USG whole abdomen"), ("(USG W/A)", "USG whole abdomen"), ("usg wa", "USG whole abdomen"), ("USG W/Abd", "USG whole abdomen"),
    ("TVS", "USG TVS"), ("T.V.S.", "USG TVS"), ("USG TVS", "USG TVS"),
    ("Echo-2D", "2D Echo"), ("Echo 2D", "2D Echo"), ("2D Echo", "2D Echo"),
    ("TORCH", "TORCH panel"), ("Bl for TORCH", "TORCH panel"), ("TORCH profile", "TORCH panel"),
    ("ABO + Rh typing", "ABO and Rh blood group"), ("ABO/Rh", "ABO and Rh blood group"), ("blood grouping", "ABO and Rh blood group"),
    ("Karyotyping", "Karyotype"), ("Bl for Karyotyping (both parents)", "Karyotype"),
])
def test_names_the_owner_reported_are_known(written, standard):
    assert std(written) == standard


def test_abo_and_rh_is_one_test_not_rh_alone():
    assert split_tests("ABO + Rh typing") == ["ABO + Rh typing"]
    assert split_tests("ABO/Rh") == ["ABO + Rh"]
    assert split_tests("Bl for TORCH, HbA1c, CBC, ABO + Rh typing, TSH") == ["Bl for TORCH", "HbA1c", "CBC", "ABO + Rh typing", "TSH"]
    assert split_tests("CBC + CRP") == ["CBC", "CRP"]                      # a + between other tests still separates them


def test_echo_2d_is_no_longer_dropped_as_unplaced_imaging():
    assert not_lab.entry_reason("Echo-2D") is None


# ---------------------------------------------------------------- PT / APTT
@pytest.mark.parametrize("piece", ["PT APT", "PT/APT", "PT-APTT"])
def test_pt_aptt_pair_keeps_both(piece):
    name, note = T.repair_piece(piece, evidence=True)
    assert name == "PT, APTT" and "APTT" in note


def test_a_trailing_test_abbreviation_is_still_dropped():
    assert T.repair_piece("Lipid Profile Tr", evidence=True)[0] == "Lipid Profile"


def test_pt_aptt_in_a_real_entry():
    payload = {"investigations": ["CBC Blood Engn <h PP HbA1c HIV HCV", "LFT", "PT [APT", "ECG"]}
    T.repair_investigations(payload)
    assert "PT, APTT" in payload["investigations"]


# ---------------------------------------------------------------- not a test
def test_medicine_orders_filed_under_investigations_are_not_tests():
    for line in ("2 of-lm) - 0.6n 1/w b.d x 5d", "Sredox w -> once sth x10d", "Tab Zinc 20 mg 1-0-1 x 5d"):
        assert not_lab.dosing_reason(line), line
    for line in ("FBS 12 hrs fasting", "CBC, CRP x 2 weeks", "TSH once", "Hb-11.9", "HbA1c/FBS/PPBS/S LIPASE TSH/FT4"):
        assert not_lab.dosing_reason(line) is None, line


def test_a_footer_read_badly_is_still_a_list_of_services():
    blocks = [{"text": "CT Scan @ 4? Digital X-ray"}]
    got = not_lab.classify(["CT Scan", "X ray"], blocks)
    assert set(got) == {"CT Scan", "X ray"}
    assert not_lab.classify(["CT Scan"], [{"text": "Adv: CT Scan brain, X-ray chest"}]) == {}          # an order line is not a footer


def test_a_test_the_enlarged_reads_confirmed_is_not_thrown_out_for_a_result_elsewhere():
    blocks = [{"text": "S.Lipase-916"}, {"text": "HbA1c/FBS review"}]
    assert "result is already written" in not_lab.classify(["Lipase"], blocks)["Lipase"]
    assert not_lab.classify(["Lipase"], blocks, confirmed=["Lipase"]) == {}
    assert "result is already written" in not_lab.classify(["Hb"], [{"text": "Hb-11.9"}], confirmed=["Lipase"])["Hb"]      # others are still judged


# ---------------------------------------------------------------- names
@pytest.mark.parametrize("written,clean", [
    ("For Mr Oukar Chowdhury", "Mr Oukar Chowdhury"), ("Name: Sita Roy", "Sita Roy"), ("Patient Name Mr Amit Sen", "Mr Amit Sen"), ("For Sita", "Sita"),
    ("Forum Sen", "Forum Sen"), ("Nameeta Roy", "Nameeta Roy"), ("Patil Rao", "Patil Rao"),
])
def test_label_words_in_front_of_a_name_are_not_part_of_it(written, clean):
    assert N.clean_name(written) == clean


B66 = [{"text": "NameofPatient:Mr.ONKARCHOUDHURY", "recognition": PRINTED, "ocr_conf": 0.984}, {"text": "Gender:MAge:68(years)", "recognition": PRINTED, "ocr_conf": 0.958}]


def test_the_printed_name_field_is_found_with_its_exact_letters():
    assert N.printed_name_field(B66) == {"letters": "onkarchoudhury", "raw": "ONKARCHOUDHURY", "title": "Mr"}
    assert N.printed_name_field([{"text": "NameofDoctor:Dr.SHATABDI", "recognition": PRINTED, "ocr_conf": 0.99}]) is None
    assert N.printed_name_field([{"text": "NameofPatient:Mr.ONKAR", "recognition": {"state": "single_engine"}, "ocr_conf": 0.99}]) is None     # not print
    assert N.printed_name_field([{"text": "NameofPatient:Mr.ONKARCHOUDHURY", "recognition": PRINTED, "ocr_conf": 0.5}]) is None              # not sure


def test_the_printed_spelling_decides_and_the_reading_gives_the_word_breaks():
    p = {"patient": {"name": "Mr. ONKAR CHAUDHURY", "age_text": "68", "sex": "M"}}
    service._anchor_to_printed_banner(p, B66, ["Mr. ONKAR CHAUDHURY", "Onkar Chaudhary"])
    assert p["patient"]["name"] == "Mr ONKAR CHOUDHURY" and p["patient"]["sex"] == "M"          # a name field gives no age or sex


def test_a_reading_with_the_exact_printed_letters_is_preferred():
    p = {"patient": {"name": "Mr. ONKAR CHAUDHURY"}}
    service._anchor_to_printed_banner(p, B66, ["Mr. ONKAR CHAUDHURY", "ONKAR CHOUDHURY"])
    assert p["patient"]["name"] == "ONKAR CHOUDHURY"


def test_no_printed_name_no_change_and_nothing_invented():
    p = {"patient": {"name": "Sita Roy"}}
    service._anchor_to_printed_banner(p, [{"text": "Patient Name:", "recognition": PRINTED, "ocr_conf": 0.97}], ["Rita Roy"])
    assert p["patient"]["name"] == "Sita Roy"
    p = {"patient": {"name": None}}
    service._anchor_to_printed_banner(p, B66, ["Amit Kumar"])
    assert p["patient"]["name"] is None


# ---------------------------------------------------------------- imaging: modality + body part is ONE test
@pytest.mark.parametrize("written,standard", [
    ("X-ray of pelvis c S/C hip - AP", "X-ray pelvis c S/C hip - AP"), ("xray of pelvis with S/I hip AP", "X-ray pelvis with S/I hip AP"),
    ("X-ray of LS spine - AP / lat", "X-ray LS spine - AP / lat"), ("MRI of C spine", "MRI C spine"), ("MRI of LS spine", "MRI LS spine"),
    ("CT scan of brain", "CT scan brain"), ("X-ray chest PA", "X-ray chest PA"), ("X-ray RT mastoid lateral view", "X-ray RT mastoid lateral view"),
])
def test_an_imaging_order_with_its_body_part_is_one_known_test(written, standard):
    assert std(written) == standard
    assert not_lab.entry_reason(written) is None                         # no longer thrown out as imaging the lists do not place


def test_a_bare_modality_is_still_the_generic_test_and_nothing_is_guessed():
    assert std("MRI") == "MRI" and std("X-ray") == "X-ray"
    assert std("MRI of something") is None or std("MRI of something") == "MRI of something"      # no body part, no part invented


def test_the_views_written_after_an_imaging_order_stay_with_it():
    assert split_tests("xray ls spine ap and lat") == ["xray ls spine ap lat"]
    assert split_tests("X-ray of LS spine - AP / lat") == ["X-ray of LS spine - AP / lat"]
    assert split_tests("X-ray chest, CBC, LFT") == ["X-ray chest", "CBC", "LFT"]            # a list of tests is still a list
    assert split_tests("CBC / LFT") == ["CBC", "LFT"]


def test_a_bare_modality_beside_a_specific_order_of_the_same_kind_is_dropped():
    from cdi_adapter.extract import unplaced as U

    inv = [{"text": "MRI"}, {"text": "MRI of C spine"}, {"text": "X ray"}, {"text": "X-ray pelvis"}, {"text": "USG"}, {"text": "CBC"}]
    kept, dropped = U.drop_generic_modality(inv)
    assert [i["text"] for i in kept] == ["MRI of C spine", "X-ray pelvis", "USG", "CBC"] and dropped == ["MRI", "X ray"]
    kept, dropped = U.drop_generic_modality([{"text": "MRI"}, {"text": "X ray"}])
    assert len(kept) == 2 and dropped == []


def test_the_same_line_read_twice_is_listed_once():
    from cdi_adapter.extract import unplaced as U

    kept, dropped = U.drop_near_duplicates([{"text": "Blood for fever ?ipple"}, {"text": "fever profile"}, {"text": "CBC"}, {"text": "Dengue NS1"}, {"text": "TSH"}])
    assert dropped == ["Blood for fever ?ipple"] and [i["text"] for i in kept] == ["fever profile", "CBC", "Dengue NS1", "TSH"]


# ---------------------------------------------------------------- the department prior no longer turns an ADVICE line into a test
class _Client:
    """A model that always picks option 1."""

    def vlm_generate_ex(self, *_a, **_k):
        return "com(RL)", None

    def vlm_json_ex(self, *_a, **_k):
        return {"choice": 1}, None


def _png():
    import cv2
    import numpy as np

    ok, b = cv2.imencode(".png", np.full((120, 400, 3), 255, np.uint8))
    return b.tobytes()


def test_an_ent_page_complaint_among_the_advice_is_not_turned_into_laryngoscopy():
    from cdi_adapter.extract import unplaced as U

    payload = {"investigations": [], "advice": ["com[RL]"]}
    blocks = [{"text": "com[RL]", "bbox": [10, 10, 200, 60]}]
    U.reread(_Client(), _png(), blocks, payload, "ENT")
    assert not any(isinstance(i, dict) and i.get("source") == "choice_vote" for i in payload["investigations"])
    assert "Laryngoscopy" not in str(payload["investigations"])
