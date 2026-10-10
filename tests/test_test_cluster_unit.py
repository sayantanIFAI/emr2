"""Tests are written together: the page's own text is searched by position, with no model (extract/test_cluster.py)."""
from __future__ import annotations

import pytest

from cdi_adapter.extract import test_cluster as T


def B(text, x0, y0, x1, y1):
    return {"text": text, "bbox": [x0, y0, x1, y1]}


# the lines and boxes of a real prescription (565 x 956 px), as the readers produced them
DEBABRATA = [B("MR. Debabrata San?ar(63/M) DATE 16/03/24", 3, 190, 546, 231), B("STOP SMOKING", 325, 312, 503, 340),
             B("Tabs. Glimepiride 10?/Generic", 132, 541, 483, 586), B("Tab. Aliv?ng(20) 1tros afm Dmndly Cap. Vit?ria", 149, 712, 506, 785),
             B("x cap. lid?niz Ds(bow)/DVRK.", 3, 751, 515, 956), B("fas, fas", 9, 771, 97, 804), B("100mg/5mL", 3, 799, 82, 819),
             B("vitam?? D", 3, 804, 112, 837), B("l?p", 281, 814, 332, 842), B("pn/kx2oh", 346, 849, 511, 875), B("1m??", 384, 876, 486, 928)]


def test_the_real_page_gives_the_fasting_sugar_slip_and_vitamin_d_beside_it():
    got = [(f.test, f.why) for f in T.scan(DEBABRATA)]
    assert ("FBS", "near") in got and ("Vitamin d", "beside") in got
    assert not any(t in ("Tabs. Glimepiride", "STOP SMOKING") for t, _ in got)          # medicines and advice are not tests


def test_vitamin_d_on_its_own_or_among_medicines_is_not_taken_but_beside_tests_it_is():
    assert T.scan([B("vitamin D", 3, 804, 112, 837)]) == []
    assert T.scan([B("Tab Metformin 500mg", 3, 100, 300, 130), B("vitamin D", 3, 135, 110, 165)]) == []
    assert [f.test for f in T.scan([B("CBC", 3, 100, 60, 130), B("vitamin D", 3, 135, 110, 165)])] == ["CBC", "vitamin D"]
    assert [f.test for f in T.scan([B("CBC", 3, 100, 60, 130), B("vitamin D", 300, 600, 410, 630)])] == ["CBC"]    # far away: not beside


def test_an_ambiguous_name_inside_a_sentence_is_never_a_list_entry():
    got = T.scan([B("CBC", 3, 100, 60, 130), B("calcium rich diet and exercise daily here", 3, 135, 400, 165)])
    assert [f.test for f in got] == ["CBC"]


@pytest.mark.parametrize("line", ["25(OH) vit D", "25 OH Vit D", "pn/kx2oh"])
def test_the_25_oh_mark_alone_is_the_vitamin_d_test(line):
    assert any(f.test.casefold().startswith("vit") for f in T.scan([B(line, 3, 100, 200, 130)]))


def test_strong_names_are_tests_wherever_they_are_and_a_list_is_taken_whole():
    got = [f.test for f in T.scan([B("CBC, LFT, RFT", 3, 100, 200, 130), B("FBS PPBS", 3, 135, 150, 165), B("25 OH Vit D", 3, 170, 200, 200)])]
    assert got == ["CBC", "LFT", "RFT", "FBS", "PPBS", "Vit D"]
    assert [f.test for f in T.scan([B("TSH", 400, 900, 450, 930)])] == ["TSH"]


@pytest.mark.parametrize("word,expected", [("fas", "FBS"), ("lfl", "LFT"), ("far", None), ("for", None), ("cbc", None), ("tsh", None), ("ab", None)])
def test_only_a_one_letter_handwriting_slip_of_one_abbreviation_counts(word, expected):
    assert T.near_miss(word) == expected


def test_a_slip_on_its_own_is_not_taken_but_two_in_a_list_or_one_beside_a_test_are():
    assert T.scan([B("fas", 3, 100, 60, 130)]) == []
    assert [f.test for f in T.scan([B("fas, fas", 3, 100, 100, 130)])] == ["FBS"]
    assert [f.test for f in T.scan([B("CBC", 3, 100, 60, 130), B("fas", 3, 135, 60, 165)])] == ["CBC", "FBS"]


def test_where_a_test_came_from_is_said_in_words():
    notes = {f.test: f.note for f in T.scan(DEBABRATA)}
    assert "one letter from FBS" in notes["FBS"] and "beside other tests" in notes["Vitamin d"]


def test_lines_without_boxes_are_grouped_in_reading_order():
    got = [f.test for f in T.scan([{"text": "CBC"}, {"text": "vitamin D"}])]
    assert got == ["CBC", "vitamin D"]


def test_pt_is_the_test_only_beside_tests_never_as_the_word_patient():
    assert T.scan([B("Currently moderate articolar activity as conveyed by pt henc", 196, 560, 672, 692)]) == []     # MEASURED: a real page
    assert [f.test for f in T.scan([B("pt", 3, 100, 40, 130)])] == []
    assert [f.test for f in T.scan([B("CBC", 3, 100, 60, 130), B("PT", 3, 135, 40, 165)])] == ["CBC", "PT"]


def test_a_word_that_can_be_read_two_ways_is_not_taken():
    assert T.near_miss("apt") is None and T.near_miss("fas") == "FBS"                # apt: AST by one letter or APTT by a doubled letter


def test_a_test_name_inside_a_long_line_of_other_words_is_weak_evidence_not_a_test():
    # MEASURED on a real page: the clinic's footer "?oscopy ?NT Endoscopy ? Ultras?nography ? Echocardiography ... ECG" gave ECG
    footer = B("Endoscopy Ultrasonography Echocardiography ECG Colonoscopy Treadmill Spirometry", 138, 1491, 1123, 1511)
    assert T.scan([footer]) == []
    assert [f.test for f in T.scan([footer, B("CBC, LFT", 140, 1520, 300, 1550)])] == ["Endoscopy", "Echocardiography", "ECG", "Colonoscopy", "Spirometry", "CBC", "LFT"]       # beside tests it counts (page order); the footer names are then rejected by extract/not_lab.py as "part of the clinic's printed list of services"
    assert [f.test for f in T.scan([B("CBC CRP LFT KFT TSH FT4", 3, 100, 400, 130)])] == ["CBC", "CRP", "LFT", "KFT", "TSH", "FT4"]   # a real list is taken


def test_a_line_the_reader_calls_printed_is_still_read_because_handwritten_tests_are_often_called_that():
    # MEASURED: "CBCCRP" and "VCRP" are handwritten and were labelled printed; skipping printed lines lost them on two real pages
    assert [f.test for f in T.scan([{"text": "CBCCRP", "bbox": [989, 917, 1065, 945], "recognition": {"state": "printed"}}])] == ["CBC", "CRP"]


def test_a_handwritten_line_with_one_garbled_word_keeps_its_tests():
    # MEASURED on a real page: "! CBC Blood Engn ? PP ?" is 6 words, 2 of them tests: taken (the printed footer is 1 of 8 or fewer)
    line = B("! CBC Blood Engn ? PP ?", 6, 846, 413, 999)
    assert [f.test for f in T.scan([line])] == ["CBC", "PP"]



# ---- the model's own entries, put right (MEASURED on a real prescription: "Chest ECO", "Nat & Kit Level Test")
@pytest.mark.parametrize("piece,expected", [
    ("Nat & Kit Level Test", "Na+ & K+"), ("Na+ & K+ Level Test", None), ("Na+ & K+", None), ("Nat q Kit", "Na+ & K+"), ("S. Na & K", None),      # this one the lists already place
    ("Chest ECO", "Chest ECG"), ("CBC", None), ("Chest pain", None), ("Level Test", None),
])
def test_a_misread_test_is_put_right_only_when_the_lists_then_place_it(piece, expected):
    got = T.repair_piece(piece, evidence=True)
    assert (got[0] if got else None) == expected


def test_a_slip_needs_other_listed_tests_beside_it_but_sodium_and_potassium_does_not():
    assert T.repair_piece("Chest ECO", evidence=False) is None
    assert T.repair_piece("Nat & Kit Level Test", evidence=False)[0] == "Na+ & K+"


def test_the_models_list_is_corrected_in_place_and_says_what_was_read():
    payload = {"investigations": [{"text": "Chest ECO", "evidence": []}, "CBC Test", {"text": "Nat & Kit Level Test"}, "Lipid Profile Test"]}
    notes = T.repair_investigations(payload)
    assert [x["text"] if isinstance(x, dict) else x for x in payload["investigations"]] == ["Chest ECG", "CBC", "Na+ & K+", "Lipid Profile"]
    assert "ECG" in notes["Chest ECG"] and "'Chest ECO'" in notes["Chest ECG"] and "sodium and potassium" in notes["Na+ & K+"]
    alone = {"investigations": ["Chest ECO"]}
    assert T.repair_investigations(alone) == {} and alone["investigations"] == ["Chest ECO"]          # no other test beside it: not taken


def test_chest_ecg_sodium_potassium_and_fever_profile_are_in_the_mapping_table():
    from cdi_adapter.extract import lab_mapping
    for name, std in (("Chest ECG", "ECG"), ("Na+ & K+", "Sodium and potassium (electrolytes)"), ("Blood for fever profile", "Fever profile"),
                      ("Serum electrolytes", "Sodium and potassium (electrolytes)")):
        assert lab_mapping.lookup(name).canonical == std


# ---- MEASURED on the real Dr. Mondal page: what each enlarged view answered (3 rounds), and where CBC / Na+ & K+ were lost
@pytest.mark.parametrize("answer,expected", [
    ("CBC w. NPT & K+ Level F", ["CBC", "Na+ & K+"]), ("CBC w. diff.", ["CBC"]), ("CBC + WBC", ["CBC"]),
    ("NAT & Kit Level Test", ["Na+ & K+"]), ("NAT + Q Kit Level Test", ["Na+ & K+"]), ("NAT-2 Kit Level Th", ["Na+ & K+"]),
    ("Chest ECG", ["Chest ECG"]), ("Lipid Profile", ["Lipid Profile"]), ("case m.", []), ("Na+ & K+ Level Test", ["Na+ & K+"]),
])
def test_the_tests_inside_a_garbled_model_answer_are_taken_word_by_word(answer, expected):
    assert T.answer_tests(answer) == expected


def test_the_models_nat_2_kit_entry_is_put_right():
    got = T.repair_piece("NAT-2 Kit Level Th", evidence=False)
    assert got and got[0] == "Na+ & K+" and "sodium and potassium" in got[1]


# ---- MEASURED once the national lab list was loaded on the pod: printed form labels and the doctor's "Tr" after a test
@pytest.mark.parametrize("line", ["Height:", "Weight: 77 kg", "Pulse:", "BP: 110/70 mmHg", "Blood", "Vital Signs", "Investigation:"])
def test_a_printed_form_label_is_never_a_lab_test_even_when_the_national_list_holds_the_word(line):
    assert T.scan([B(line, 3, 100, 160, 130)]) == []


@pytest.mark.parametrize("piece,expected", [
    ("lipid profile fr", "lipid profile"), ("Lipid Profile Tr", "Lipid Profile"), ("CBC Tr", "CBC"), ("Lipid Profile Test", "Lipid Profile"),
    ("NAT Kit Level th", "Na+ & K+"), ("Nat Kit", "Na+ & K+"),
])
def test_the_doctors_abbreviated_test_after_a_name_is_not_part_of_the_name(piece, expected):
    got = T.repair_piece(piece, evidence=False)
    assert got and got[0] == expected


def test_words_that_only_look_like_sodium_potassium_are_left_alone():
    for word in ("Napkin", "Nakshatra", "Nature kit", "Panel"):
        assert T.repair_piece(word, evidence=False) is None


# ---- MEASURED on a real page: the reader merged several lines into one; "Digital OPG, FBS, BT, CT" sat after "Adv"
def test_the_words_after_a_heading_that_orders_tests_are_a_declared_test_region():
    line = "4 Difficulty in ? 5 Gren or Strain Adv Digital OP? 2 FBS, BJS CT"
    got = [f.test for f in T.scan([B(line, 60, 580, 403, 758)])]
    assert "FBS" in got and "CT (clotting time or CT scan)" not in got and any(x.casefold().startswith("digital op") for x in got)
    assert "FBS" not in [f.test for f in T.scan([B(line.replace("Adv ", ""), 60, 580, 403, 758)])]       # without the heading it is a long mixed line


@pytest.mark.parametrize("line", ["Adv: FBS, BT, CT", "Inv - CBC, CRP", "Investigations: LFT", "Ix FBS"])
def test_each_heading_word_that_orders_tests_opens_a_test_region(line):
    assert T.scan([B("some words before this " + line, 3, 100, 600, 130)])


def test_opg_and_ct_are_in_the_lists_with_their_ambiguity_said():
    from cdi_adapter.extract import lab_mapping
    assert lab_mapping.lookup("Digital OPG").canonical == "OPG (orthopantomogram)"
    ct = lab_mapping.lookup("CT")
    assert ct.canonical.startswith("CT (clotting time or CT scan") and "ambiguous" in ct.note


# ---- a list with one test the lists place is a list of tests (MEASURED: the model wrote "Digital OPG, FBS, BJS CT" as the booking text)
def test_every_entry_of_a_list_that_holds_one_placed_test_is_a_test():
    got = T.list_entries("Digital OPG, FBS, BJS CT")
    assert [(n, placed) for n, _a, placed in got] == [("Digital OPG", True), ("FBS", True), ("BJS", False), ("CT", True)]       # as read; the lists name them
    assert T.list_entries("FBS, BP, OD, KT") == [("FBS", "FBS", True), ("KT", "KT", False)]            # BP and OD are labels, not entries
    assert T.list_entries("Adv: 1) Digital OPG 2) FBS, BT, CT")[1][0] == "FBS"


@pytest.mark.parametrize("text", ["Rest, drink water, avoid sugar", "Review after 2 weeks", "BJS, XYZ", "", None, "Take tablet after food"])
def test_a_list_with_no_placed_test_is_not_a_list_of_tests(text):
    assert T.list_entries(text) == []


def test_ordinary_words_beside_a_test_are_not_entries_only_capital_abbreviations_are():
    got = [n for n, _a, _p in T.list_entries("Adv: FBS, rest, Diet control, KT")]
    assert got == ["FBS", "KT"]


def test_a_composite_entry_of_words_already_listed_one_by_one_is_dropped():
    entries = T.list_entries("Digital OPG, FBS, BJS CT")
    inv = [{"text": "BJS CT"}, {"text": "FBS"}, {"text": "Digital OPG"}, "Chest pain", {"text": "BJS"}]
    got = [x["text"] if isinstance(x, dict) else x for x in T.drop_composites(inv, entries)]
    assert got == ["FBS", "Digital OPG", "Chest pain", "BJS"]
