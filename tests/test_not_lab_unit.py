from cdi_adapter.extract import not_lab as N


def B(*texts):
    return [{"text": t} for t in texts]


def test_entry_reasons():
    assert N.entry_reason("MRI LS spine") == "imaging, not a laboratory test"
    assert N.entry_reason("Chest ECG").startswith("ECG")
    assert N.entry_reason("Physio - UST (IFT)").startswith("physiotherapy")
    assert N.entry_reason("CBC") is None and N.entry_reason("CT") is None and N.entry_reason("FBS") is None
    assert N.entry_reason("MRI LS spine", lab_only=False) is None          # a site that wants imaging listed switches the rule off


def test_line_context_physio_and_footer_and_medicine():
    page = B("- Physio - UST (RFT)", "Tab Zincovit 1 tab after lunch x 15 days",
             "Pain & Laser Clinic ● Sugar Clinic ● Dental Clinic ● MRI ● CT Scan ● X-Ray (Digital)")
    assert "physio" in N.line_reason("RFT", page)
    assert N.line_reason("UST", page)
    assert N.line_reason("Zinc", page) == "part of a medicine line"
    assert N.line_reason("CT", page) is not None                              # only the footer holds it


def test_mixed_line_keeps_the_tests_and_drops_only_imaging():
    page = B("4 Difficulty in ? 5 Gren or Strain Adv Digital OPG, FBS, BJS CT")
    got = N.classify(["Digital OPG", "FBS", "CT"], page)
    assert "Digital OPG" in got and "FBS" not in got and "CT" not in got


def test_a_name_with_its_own_line_is_kept():
    page = B("Tab Zincovit 1 tab", "Zinc level")
    assert N.line_reason("Zinc", page) is None
