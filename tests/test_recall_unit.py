from cdi_adapter.extract import recall


def B(*texts):
    return [{"text": t, "bbox": [0, 40 * i, 500, 40 * i + 30]} for i, t in enumerate(texts)]


def test_dental_page_typed_operative_note_becomes_prescription():
    page = B("Patient complains of mobility", "4 Difficulty in ? 5 Gren or Strain Adv Digital OPG, FBS, BCS CT", "Tab Taxim-O2 1 tab BDPC x 3 days")
    new, why = recall.reroute("operative_note", page)
    assert new == "prescription" and "FBS" in why or new == "prescription"


def test_other_with_a_test_and_medicines():
    new, _ = recall.reroute("other", B("TAB solpect (5mg) 1x1 - 30d", "TAB Mira (25) 1x1", "L", "TSH"))
    assert new == "prescription"


def test_a_real_operative_note_stays():
    assert recall.reroute("operative_note", B("Operative note", "Surgeon: Dr X", "Pre-operative diagnosis hernia", "Hb CBC done")) == (None, "")


def test_prescription_types_and_unknowns_are_untouched():
    assert recall.reroute("prescription", B("CBC")) == (None, "")
    assert recall.reroute("radiology_report", B("CBC", "Adv")) == (None, "")


def test_lab_report_pad_with_medicines_and_review_line():
    page = B("Review after 3 month", "FPG, CREATININE, LIPID PROFILE", "Free T4, TSH", "Tab Cetapin (0.3) 1 tab", "Tab Rivotril (0.5) 1 tab")
    new, _ = recall.reroute("lab_report", page)
    assert new == "prescription"


def test_real_lab_report_with_a_reference_range_stays():
    page = B("Reference range", "Hb 12.5 g/dL 12-16", "CBC", "Tab", "Review after")
    assert recall.reroute("lab_report", page) == (None, "")
