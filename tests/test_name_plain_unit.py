from cdi_adapter.extract import resolve_llm as R


def test_plain_transcription_to_a_name():
    f = R.name_from_transcription
    assert f("Mr. Sumita. gupta. Gangopadhyay. 7416.") == "Sumita gupta Gangopadhyay"          # MEASURED reading of a real name line
    assert f("Mx. Sumita Gupta. Gangopalhyay. 74/6.") == "Sumita Gupta Gangopalhyay"
    assert f("Mrs Smita Gupta 73 yrs") == "Smita Gupta"
    assert f("S. K. Roy") == "S. K. Roy"                                                     # initials keep their stop
    assert f("```json\nnull\n```") is None and f("") is None and f("74/F") is None
    assert f("Mr.") is None
    assert f("Onkar Chowdhury (63/M)") == "Onkar Chowdhury"
