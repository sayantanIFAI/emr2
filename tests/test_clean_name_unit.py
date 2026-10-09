import pytest

from cdi_adapter.extract import resolve_llm
from cdi_adapter.names import clean_name


@pytest.mark.parametrize("raw,want", [
    ("Mrs.Sumita Gupta Gangopadhyay yrs Female", "Mrs. Sumita Gupta Gangopadhyay"),
    ("Mrs. Sumita Gupta Gangopadhyay 74 yrs Female", "Mrs. Sumita Gupta Gangopadhyay"),
    ("Sumita Gupta Gangopadhyay 74/F", "Sumita Gupta Gangopadhyay"),
    ("Sumita Gupta Gangopadhyay, Female", "Sumita Gupta Gangopadhyay"),
    ("Sumita Gupta Gangopadhyay M/F", "Sumita Gupta Gangopadhyay"),
    ("Sumi7ta Das", "Sumita Das"),
    ("A K Das", "A K Das"),
    ("Debabrata Sanwar", "Debabrata Sanwar"),
])
def test_a_name_is_words_only(raw, want):
    assert clean_name(raw) == want


def test_nothing_left_is_no_name():
    assert clean_name("74 yrs Female") is None
    assert clean_name("yrs Female") is None
    assert clean_name(None) is None


def test_a_transcription_stops_at_the_age_or_sex_words():
    assert resolve_llm.name_from_transcription("Mrs. Sumita Gupta Gangopadhyay yrs Female") == "Sumita Gupta Gangopadhyay"


def test_a_glued_age_and_sex_after_the_surname_are_cut_off():
    # MEASURED on a real page: the line reads "Mrs.Sumita Gupta Gangopadhyay./72yrs/Female"
    assert resolve_llm.name_from_transcription("Mrs.Sumita Gupta Gangopadhyay./72yrs/Female") == "Sumita Gupta Gangopadhyay"
    assert clean_name("Sumita Gupta Gangopadhyay Yrsfemale") == "Sumita Gupta Gangopadhyay"
    assert clean_name("Mrs.Sumita Gupta Gangopadhyay./72yrs/Female") == "Mrs. Sumita Gupta Gangopadhyay."
