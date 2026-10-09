import pytest

from cdi_adapter.extract import results_written as W

# every one of these was listed as a test to be done on a real page; each carries a value, so each is a result already done
RESULTS = ["Hb-11.9", "Hb-11.3", "Platelet-2.25 lac", "platelet-2·25lac", "ILC-6900", "TLC-6900", "Creat-0.35", "CRP-0.02 (<0.8)", "SGPT-28/22",
           "SHOT(SAPT-28/22)", "ISH-0.550", "FMS-103.6", "Fg-133", "TG-133", "LDL 94", "LDL94", "TSH 1.27", "HIV -ve", "HBsAg negative", "Urine R/E normal",
           "DXA L1-L4 -0.8 Femur -1.4"]


@pytest.mark.parametrize("text", RESULTS)
def test_a_value_next_to_a_test_makes_the_entry_a_result(text):
    assert W.is_result_entry(text), text
    assert W.clean_entry(text)[0] == ""


def test_tests_to_be_done_are_untouched():
    for text in ["CBC", "CBC, CRP, Creatinine, SGPT, SGOT", "HbA1c", "FT4", "T3, T4, TSH", "Vitamin B12", "25 OH Vitamin D", "FBS 12 hrs fasting",
                 "Review after 2 weeks", "HbA1c/FBS/PPBS/S LIPASE TSH/FT4", "Lipid Profile", "PT/APTT", "X-ray chest PA"]:
        assert W.clean_entry(text) == (text, []), text


def test_a_list_keeps_the_ordered_tests_and_drops_the_result():
    assert W.clean_entry("CBC, CRP, Hb-11.9")[0] == "CBC, CRP"
    assert W.clean_entry("Hb-11.9, platelet 2.2 lac, LFT")[0] == "LFT"
    assert W.clean_entry("CBC , SGPT-28/22 , SGOT")[0] == "CBC , SGOT" or W.clean_entry("CBC , SGPT-28/22 , SGOT")[0] == "CBC, SGOT"


def test_a_name_is_a_result_only_when_every_place_it_is_written_has_a_value():
    blocks = [{"text": "CRP-0.02 (<0.8)"}, {"text": "Hb-11.9"}, {"text": "Review after 3 months"}, {"text": "CBC CRP Creatinine SGPT SGOT"}]
    assert W.page_result_reason("Hb", blocks).startswith("a result is already written")
    assert W.page_result_reason("CRP", blocks) is None                 # also written plainly in the ordered list
    assert W.page_result_reason("CBC", blocks) is None
    assert W.page_result_reason("TSH", blocks) is None                 # not on the page text at all: left alone


def test_a_value_in_the_next_piece_of_the_same_row_counts():
    blocks = [{"text": "TSH"}, {"text": "1.27"}, {"text": "Adv"}, {"text": "LFT"}]
    assert W.page_result_reason("TSH", blocks) is not None
    assert W.page_result_reason("LFT", blocks) is None


def test_ordinary_advice_with_a_number_is_not_a_result():
    for text in ["Stop smoking - 2 weeks", "Diet - 1600 kcal per day", "Walk for 30 minutes", "Take rest - 3 days", "Brisk walk 30-45 min/day"]:
        assert W.clean_entry(text) == (text, []), text
