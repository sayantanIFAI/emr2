from cdi_adapter.extract.fields import normalise_age


def test_age_with_a_slash_is_the_number_before_it():
    assert normalise_age("74/1") == ("74 yrs", None)
    assert normalise_age("73 yrs / Female") == ("73 yrs", "F")
    assert normalise_age("68 Y/M") == ("68 Y", "M")
    assert normalise_age("41|F") == ("41 yrs", "F")


def test_other_texts_are_left_alone():
    assert normalise_age("73 yrs") == ("73 yrs", None)
    assert normalise_age("6 months") == ("6 months", None)
    assert normalise_age(None) == (None, None)


def test_the_age_is_checked_against_a_number_with_its_unit_on_the_page():
    from cdi_adapter.extract.fields import _age_on_page
    assert _age_on_page("73 yrs", "Sumita Gupta | 73yrs | Female")
    assert _age_on_page("73 yrs", "Sumita Gupta Gangopadhyay 73 Yrs/ Female")
    assert not _age_on_page("73 yrs", "Plt 73 lakhs, ESR 12")
    assert not _age_on_page("74 yrs", "Sumita Gupta | 73yrs | Female")
