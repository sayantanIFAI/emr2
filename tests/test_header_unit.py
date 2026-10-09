"""The doctor's and the clinic's name from the printed header, when the model left them empty (extract/header.py)."""
from __future__ import annotations

import pytest

from cdi_adapter.extract import header as H


@pytest.mark.parametrize("line,expected", [
    ("Dr. Kumar Sourav", "Dr. Kumar Sourav"),                                               # MEASURED: real pages
    ("Dr.Kumar Sourav MBBSPATMDPATalMine ConsuPyin/Nerndogist KSHEALTHCARE /", "Dr. Kumar Sourav"),
    ("Dr.Debasis Giri", "Dr. Debasis Giri"),
    ("Prof.（Dr.)Aniruddha Majumder", "Prof. Dr. Aniruddha Majumder"),                   # a full-width bracket from the reader
    ("Dr. K. Sourav MD", "Dr. K. Sourav"), ("Dr. Giri Mob. 98302", "Dr. Giri"),
    ("Medicine OPD", None), ("Dr", None), ("Dr. MBBS MD", None), ("", None),
])
def test_the_doctors_name_is_the_name_words_after_dr_up_to_the_first_qualification(line, expected):
    assert H.doctor_name(line) == expected


@pytest.mark.parametrize("line,expected", [
    ("KSHEALTHCARE", "KS HEALTHCARE"), ("Core Clinic Mob.9830228483 ConsultantEndocrinolgist&Dlabetologist Reg.", "Core Clinic"),
    ("LIFE CENTRE POLYCLINIC", "LIFE CENTRE POLYCLINIC"),
    ("Sinus Vertigo Clinic|Ear Clinic|Allergy Clinic| Voice Clinic", None),                  # a list of clinics is not the organisation
    ("HORMONE SCIENCETO HEALTH", None), ("Contacts:+919836834614", None), ("Clinic", None),
    # MEASURED on a real page: a long garbled line holding the word is not the clinic's name
    ("Dr.Kumar Sourav MBBSPATMDPATalMine ConsuPyin/Nerndogist KSHEALTHCARE /", None),
    ("Sugar & Thyroid Clinic", "Sugar & Thyroid Clinic"),
])
def test_the_clinics_name_is_the_line_with_exactly_one_organisation_word(line, expected):
    assert H.clinic_name(line) == expected


def B(text, y0, y1):
    return {"text": text, "bbox": [4, y0, 700, y1]}


PAGE = [B("Dr. Kumar Sourav", 0, 60), B("KSHEALTHCARE", 70, 100), B("HORMONE SCIENCETO HEALTH", 110, 130), B("Tab. Glimepiride 10mg", 500, 560),
        B("Diabetic Clinic follow up here", 900, 960), B("x", 1000, 1100)]


def test_the_header_fills_the_doctor_and_the_clinic_when_the_model_left_them_empty():
    payload = {"patient": {"name": "x"}}
    assert H.fill(payload, PAGE) == ["doctor", "clinic"]
    assert payload["prescriber"]["name"] == "Dr. Kumar Sourav" and payload["prescriber"]["clinic"]["name"] == "KS HEALTHCARE"


def test_a_value_the_model_gave_is_never_replaced_and_only_the_top_of_the_page_is_read():
    payload = {"prescriber": {"name": "Dr. A Banerjee", "clinic": {"name": "City Hospital", "phone": "1"}}}
    assert H.fill(payload, PAGE) == [] and payload["prescriber"]["name"] == "Dr. A Banerjee"
    assert payload["prescriber"]["clinic"] == {"name": "City Hospital", "phone": "1"}
    low = {}
    assert H.fill(low, [B("Dr. Someone", 900, 960), B("pad", 0, 20), B("end", 1000, 1100)]) == []            # a "Dr" far down the page is not the letterhead
    assert H.fill({}, []) == [] and H.fill({"prescriber": "text"}, PAGE) == []


def test_only_the_empty_part_is_filled():
    payload = {"prescriber": {"name": "Dr. A Banerjee", "clinic": {"name": None, "address": "Patna"}}}
    assert H.fill(payload, PAGE) == ["clinic"]
    assert payload["prescriber"]["clinic"] == {"name": "KS HEALTHCARE", "address": "Patna"} and payload["prescriber"]["name"] == "Dr. A Banerjee"


def test_the_garbled_long_line_is_skipped_and_the_clean_printed_one_is_the_clinic():
    page = [B("Dr. Kumar Sourav", 0, 60), B("Dr.Kumar Sourav MBBSPATMDPATalMine ConsuPyin/Nerndogist KSHEALTHCARE /", 40, 330), B("KSHEALTHCARE", 70, 100),
            B("x", 1000, 1100)]
    payload = {}
    H.fill(payload, page)
    assert payload["prescriber"]["clinic"]["name"] == "KS HEALTHCARE" and payload["prescriber"]["name"] == "Dr. Kumar Sourav"



# ---- the patient printed in the header (Apollo Sugar Clinics): a name with "(40 Y / MALE)" beside it
def BX(text, y0, y1):
    return {"text": text, "bbox": [400, y0, 600, y1]}


APOLLO = [BX("Apollo Sugar Clinics", 10, 60), BX("SAYANDAS(40Y/MALE)", 120, 150), BX("OPDBNO:6065/15", 160, 185), BX("M:8697709557,", 190, 215),
          BX("OR.SUSHMITAPALSANTRA", 250, 280), BX("Mob: 7980988837", 300, 320), BX("x", 900, 1000)]          # MEASURED: the readers drop the spaces


def test_the_patient_is_the_name_before_the_age_and_sex_in_the_header_with_the_phone_beside_it():
    got = H.patient_in_header(APOLLO)
    assert got == {"name": "Sayandas", "age_text": "40 Y", "sex": "M", "phone": "8697709557"}      # not the doctor's 7980988837
    spaced = [BX("SAYAN DAS (40 Y / MALE)", 120, 150)] + APOLLO[2:]
    assert H.patient_in_header(spaced)["name"] == "Sayan Das"


@pytest.mark.parametrize("line", ["Dr. B. Mondal (50 Y / MALE)", "KS Healthcare Clinic (5 Y / M)", "Patient Name Mr. Shibaji Sen", "Age 40 Y Sex MALE", ""])
def test_a_doctor_an_organisation_or_a_line_without_age_and_sex_is_not_the_patient(line):
    assert H.patient_in_header([BX(line, 10, 40), BX("x", 900, 1000)]) is None


def test_the_header_patient_fills_only_what_is_empty_and_text_null_counts_as_empty():
    payload = {"patient": {"name": "null", "sex": "F", "phone": "None"}}
    assert H.fill_patient(payload, APOLLO) == ["name", "age_text"]
    assert payload["patient"] == {"name": "Sayandas", "sex": "F", "phone": "None", "age_text": "40 Y"}      # the sex the model gave stays; the phone is never taken from the page
    kept = {"patient": {"name": "Sayan Das", "age_text": "40", "sex": "M", "phone": "1"}}
    assert H.fill_patient(kept, APOLLO) == [] and kept["patient"]["name"] == "Sayan Das"
    assert H.fill_patient({"patient": "text"}, APOLLO) == [] and H.fill_patient({}, []) == []
