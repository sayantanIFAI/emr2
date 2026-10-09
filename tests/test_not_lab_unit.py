from cdi_adapter.extract import not_lab as N


def B(*texts):
    return [{"text": t} for t in texts]


def test_entry_reasons():
    assert N.entry_reason("MRI LS spine").startswith("imaging")                  # the lists do not place it
    assert N.entry_reason("OPG") is None and N.entry_reason("Digital OPG") is None    # the mapping table places it: a test the gate knows
    assert N.entry_reason("ECG") is None and N.entry_reason("Chest ECG") is None
    assert N.entry_reason("Ph") and N.entry_reason("Ph.:") and N.entry_reason("Phone")    # a printed phone label (MEASURED: "pH of Blood" on a letterhead)
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


def test_a_test_written_on_the_same_line_as_a_medicine_is_kept():
    assert N.line_reason("TSHH", B("tshh  tab pantocid")) is None
    assert N.line_reason("CBC", B("Tab Pantocid 40 1 tab OD  CBC, LFT")) is None
    assert N.line_reason("Zinc", B("Tab Zinc 1 tab after lunch")) == "part of a medicine line"
