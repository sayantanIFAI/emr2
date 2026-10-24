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


# ---------------------------------------------------------------- a result line read with one unreadable letter
def test_a_result_line_with_one_unreadable_letter_still_makes_the_name_a_result():
    from cdi_adapter.extract import results_written as W

    blocks = [{"text": "? Cr?at 0.85", "bbox": [810, 670, 975, 701]}, {"text": "CBC CRP LFT", "bbox": [100, 100, 300, 130]}]
    assert "result is already written" in W.page_result_reason("S creat", blocks)               # p63 / p64: "Creat 0.86" among the findings is not an order
    assert W.page_result_reason("CBC", blocks) is None
    assert W.page_result_reason("S creat", blocks + [{"text": "Review: S creat", "bbox": [100, 400, 300, 430]}]) is None     # also ordered plainly: stays a test
    assert W.page_result_reason("Lipase", [{"text": "Hb 11.9"}]) is None


def test_the_box_reader_hint_list_is_the_measured_one():
    from cdi_adapter.extract import test_box

    assert "APTT" not in test_box.LIST and "Blood group" not in test_box.LIST         # extra names were echoed onto pages that do not have them


# ---------------------------------------------------------------- the 24 Oct page: "USA where aldoros", "TCOE", "LEFT", "xy chd Panies"
@pytest.mark.parametrize("written,standard", [
    ("USA where aldoros", "USG whole abdomen"), ("USA where al?doros", "USG whole abdomen"), ("USQ whole abdomen", "USG whole abdomen"),
    ("xy chd Panies", "X-ray chest PA"), ("xy chd Panies (visit)", "X-ray chest PA"),
    ("TC DC", "Total count and differential count"), ("TCDC", "Total count and differential count"), ("TC/DC", "Total count and differential count"),
    ("TLC / DLC", "Total count and differential count"),
])
def test_the_lines_of_the_page_that_missed_usg_and_tcdc(written, standard):
    assert std(written) == standard


@pytest.mark.parametrize("text", ["USA", "usa today", "Xmas chest", "US tour", "xyz"])
def test_a_stray_word_is_not_an_imaging_order(text):
    assert std(text) is None


def test_a_short_unknown_word_beside_tests_is_matched_to_the_one_abbreviation_it_can_only_be():
    assert T.near_miss("TCOE") == "TCDC" and T.near_miss("LEFT") == "LFT"
    for w in ("Case", "Tabs", "Rest", "LFT", "CBC", "Blood"):
        assert T.near_miss(w) is None, w


def test_the_advice_list_of_that_page_gives_every_test_it_holds():
    got = {name for name, _r, _p in T.list_entries("Blood for Hb, TCOE placed on Vaca each LEFT FBS")}
    assert "TCOE" in got and "FBS" in got
    assert [n for n, _r, _p in T.list_entries("USA where aldoros")] == ["USA where aldoros"]
    assert [n for n, _r, _p in T.list_entries("xy chd Panies (visit)")] == ["xy chd Panies"]


# ---------------------------------------------------------------- USG whole abdomen must never be missed, however it is written
USG_WA = ["USG whole abdomen", "USG Whole Abdomen", "USG W/A", "USG (W/A)", "(USG W/A)", "USG W/A.", "USG WA", "USG-WA", "USG - W/A", "U.S.G. whole abdomen", "U.S.G. W/A",
          "USG of whole abdomen", "USG for whole abdomen", "USG Whole Abd", "USG W/Abd", "USG whole", "USG whole abdoman", "USG whole abdomem", "USG (whole abdomen)",
          "Whole abdomen", "whole abdomen USG", "WA USG", "USS whole abdomen", "VSGT (whole Abdomen)", "VSG W/A", "USA whole abdomen", "USA where aldoros",
          "UGS whole abdomen", "USG wholeabdomen", "USGWA", "Ultrasound whole abdomen", "Sonography whole abdomen", "USG whole abdomen (PRE-OP)", "Ultra sound whole abdomen"]


@pytest.mark.parametrize("written", USG_WA)
def test_usg_whole_abdomen_is_extracted_however_it_is_written(written):
    pieces = split_tests(written)
    assert any(std(p) == "USG whole abdomen" and not_lab.entry_reason(p) is None for p in pieces), (written, pieces)


@pytest.mark.parametrize("written", ["USG abdomen", "USG lower abdomen", "USG KUB", "USG w/ abdomen", "USG pelvis", "USA", "USG neck", "wholesale", "X-ray whole spine"])
def test_other_ultrasounds_and_stray_words_are_not_turned_into_whole_abdomen(written):
    assert not any(std(p) == "USG whole abdomen" for p in split_tests(written))


def test_a_small_screenshot_is_no_longer_held_for_a_rescan():
    from cdi_adapter.config import settings
    from cdi_adapter.recognition import quality

    assert settings.quality_min_short_side_px <= 538                                    # the 538 px screenshot the owner uploaded
    import numpy as np

    rng = np.random.default_rng(1)
    img = np.full((700, 538, 3), 245, np.uint8)
    for y in range(40, 660, 36):
        img[y:y + 3, 40:500] = rng.integers(0, 90, (3, 460, 1))                          # ink-like lines
    assert quality.RESOLUTION_LOW not in quality.assess(img).reason_codes


def test_ugi_endoscopy_rut_and_usg_wa_lookalikes_are_placed():
    """Owner page 24 Oct 2026: 'UGI Endoscopy + RUT', 'USG W/A' (read 'USG Loin'), 'CBC, TSH, RBS' (read 'CBC, Bn, RBS'): none may be lost."""
    from cdi_adapter.extract import lab_mapping, test_cluster

    assert lab_mapping.lookup("UGI Endoscopy").canonical == "Upper GI endoscopy"
    assert lab_mapping.lookup("RUT").canonical == "Rapid urease test"
    for t in ("USG Loin", "USG LD/A", "usg ldia"):
        assert lab_mapping.lookup(t).canonical == "USG whole abdomen"
    for w in ("Bn", "BH", "bh"):
        assert test_cluster.repair_piece(w, True)[0] == "TSH"
    assert test_cluster.repair_piece("Bn", False) is None            # no other listed test beside it: left alone
    assert test_cluster.joined_letters("BUN") is None and test_cluster.joined_letters("Bp") is None
    p = {"investigations": [{"text": "UGI Endoscopy + RUT"}, {"text": "CBC, Bn, RBS"}]}
    test_cluster.repair_investigations(p)
    assert p["investigations"][1]["text"] == "CBC, TSH, RBS"
