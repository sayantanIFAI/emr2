from cdi_adapter.extract import department as D


def test_department_from_the_header():
    assert D.detect(["Dr. Soumalya Das", "BDS(HON'S) (WBUHS)", "Consultant Oral and Dental Surgeon"]) == "dental"
    assert D.detect(["Prof. (Dr.) Aniruddha Majumder", "MBBS, MS (ENT)", "Speciality ENT CLINIC"]) == "ent"
    assert D.detect(["Dr. Debaditya Roy", "Consultant Rheumatologist", "MD (Medicine)"]) == "rheumatology"
    assert D.detect(["Dr. B. Mondal", "General Physician"]) == "general medicine"
    assert D.detect(["SONOSCAN", "Diagnostic"]) is None


def test_specific_tests_are_checked_against_the_department_common_ones_never():
    assert D.note("Digital OPG", "dental")[0] == "usual"
    assert D.note("OPG (orthopantomogram)", "cardiology")[0] == "unusual"
    assert D.note("Coronary angiogram", "dental")[0] == "unusual"
    assert D.note("MRI brain", "neurology")[0] == "usual" and D.note("MRI brain", "general medicine")[0] == "usual"
    for common in ("CBC", "FBS", "PPBS", "HbA1c", "TSH", "FT4", "LFT", "KFT", "Lipid profile"):
        assert D.note(common, "dental") is None and D.note(common, "cardiology") is None
    assert D.note("OPG", None) is None                          # no department found: no note


def test_header_texts_takes_the_top_of_the_page():
    blocks = [{"bbox": [0, 10, 100, 30], "text": "Dr X BDS"}, {"bbox": [0, 900, 100, 930], "text": "footer cardiology services"}]
    assert D.header_texts(blocks) == ["Dr X BDS"]


def test_header_texts_ignore_handwriting_and_use_the_whole_printed_page():
    blocks = [{"bbox": [0, 10, 9, 20], "text": "DCH-", "recognition": {"state": "single_engine"}},
              {"bbox": [0, 900, 9, 920], "text": "Dr Soumalya Das BDS(HON'S)", "recognition": {"state": "printed"}},
              {"bbox": [0, 950, 9, 970], "text": "DENTAL CENTRE", "recognition": {"state": "printed"}}]
    assert D.detect(D.header_texts(blocks)) == "dental"
