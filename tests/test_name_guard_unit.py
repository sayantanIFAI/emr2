"""A patient's name is never a line of the Rx, and a printed patient line ("SAYAN DAS (40 Y / MALE)") is the name line.

MEASURED on a real page: the name line was cut from the Rx handwriting (the printed patient line was skipped as 'printed'), six readings of
"Rx gargle warm water betadine ..." agreed with each other and became the shown name; the age came out "5 days" and the sex Female."""
from cdi_adapter import names
from cdi_adapter.extract import resolve_llm, service

PRINTED = {"state": "printed"}
BANNER = {"text": "SAYANDAS(40Y/MALE)", "bbox": [457, 134, 610, 150], "recognition": PRINTED}
DOCTOR = {"text": "OR.SUSHMITAPALSANTRA", "bbox": [3, 139, 219, 154], "recognition": PRINTED}
RX = {"text": "Rx Gargle warm water", "bbox": [180, 360, 400, 380], "recognition": {"state": "single_engine"}}
SENTENCE = "Rx gargle warm water betadine steam inhalation kaolin plus rest enough fluids OR"


def test_a_line_of_the_rx_is_not_a_name():
    assert names.sentence_like(SENTENCE)
    assert names.sentence_like("Rx Google warm water betadine")
    assert names.sentence_like("Tab Montena fx")
    assert names.sentence_like("one two three four five six")


def test_real_names_are_not_sentences():
    for n in ("Sayan Das", "Mrs. Sumita Gupta Gangopadhyay", "Shibaji Sen", "Mr. Onkar Kumar Chowdhury", "Sayantani Sarkar Ghosh", "SAYAN DAS"):
        assert not names.sentence_like(n), n


def test_the_signature_label_is_not_part_of_a_name():
    assert names.clean_name("Smita Gupta Consultant") == "Smita Gupta"
    assert names.clean_name("Smita Gupta Consultant Signature") == "Smita Gupta"


def test_the_printed_patient_line_is_found_with_its_age_and_sex():
    assert names.printed_banner([DOCTOR, BANNER, RX]) == {"letters": "sayandas", "raw": "SAYANDAS", "title": "", "age": "40", "sex": "Male"}
    assert names.printed_banner([{"text": "Mrs Sita Roy (45/F)", "recognition": PRINTED}]) == {"letters": "sitaroy", "raw": "SitaRoy", "title": "Mrs", "age": "45", "sex": "Female"}
    assert names.name_on_page([BANNER])                                            # "(40Y/MALE)" is an age and a sex


def test_a_handwritten_or_doctor_line_is_not_a_banner():
    assert names.printed_banner([DOCTOR, RX]) is None
    assert names.printed_banner([{"text": "SAYANDAS(40Y/MALE)", "recognition": {"state": "single_engine"}}]) is None       # handwriting is not exact
    assert names.printed_banner([{"text": "Dr Sushmita Pal 52Y/F", "recognition": PRINTED}]) is None


def test_the_name_line_crop_is_cut_from_the_printed_patient_line():
    assert resolve_llm.best_name_block([DOCTOR, BANNER, RX], "SAYAN DAS") is BANNER           # was skipped as 'printed': the crop fell on the Rx
    assert resolve_llm.best_name_block([DOCTOR, RX], "Sushmita Pal") is None                  # the doctor's printed line is still never the patient's


def test_the_printed_line_decides_over_a_wrong_reading():
    p = {"patient": {"name": "Rx Gargle", "age_text": "5 days", "sex": "Female"}}
    service._anchor_to_printed_banner(p, [BANNER], ["Rx Gargle", "SAYAN DAS", "Sayan Das"])
    assert p["patient"] == {"name": "SAYAN DAS", "age_text": "40 yrs", "sex": "Male"}
    assert "sayandas" in p["_name_note"]


def test_the_printed_line_never_makes_up_a_name():
    p = {"patient": {"name": None, "age_text": None, "sex": None}}
    service._anchor_to_printed_banner(p, [BANNER], [SENTENCE, "Amit Kumar"])               # no reading has the printed letters
    assert p["patient"]["name"] is None


def test_a_matching_name_is_kept_as_written():
    p = {"patient": {"name": "Sayan Das", "age_text": "40", "sex": "Male"}}
    service._anchor_to_printed_banner(p, [BANNER], ["Shayan Das"])
    assert p["patient"]["name"] == "Sayan Das" and "_name_note" not in p


def test_no_printed_line_changes_nothing():
    p = {"patient": {"name": "Sita Roy", "age_text": "45", "sex": "Female"}}
    service._anchor_to_printed_banner(p, [DOCTOR, RX], ["Rita Roy"])
    assert p["patient"] == {"name": "Sita Roy", "age_text": "45", "sex": "Female"}
