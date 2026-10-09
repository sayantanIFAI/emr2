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


def test_a_middle_dot_between_name_and_value_and_a_glued_ordered_list():
    assert W.is_result_entry("Hb·11.9") and W.is_result_entry("Creat·0.35")
    blocks = [{"text": "?CRP-0.02 (<0.8)"}, {"text": "4CBCCRP"}, {"text": "Creatinine ? SAPT/"}]
    assert W.page_result_reason("CRP", blocks) is None                 # written in the ordered list too, glued to CBC by the reader
    assert W.page_result_reason("Creatinine", blocks) is None


def test_one_result_written_for_a_group_of_names_joined_by_slashes():
    # MEASURED on a real page: "HBsAg/Anti-HCV" then, on the wrapped next piece, "HIV?? non reactive": all three are results, none a test to be done
    blocks = [{"text": "?HBsAg/Anti-HCV"}, {"text": "HIV?? non reactive."}, {"text": "CBC Creatinine ?APT/SGOT CRP"}]
    assert W.page_result_reason("HBsAg", blocks).startswith("a result is already written")
    assert W.page_result_reason("Anti-HCV", blocks).startswith("a result is already written")
    assert W.page_result_reason("CBC", blocks) is None
    assert W.page_result_reason("SGOT", blocks) is None


def test_a_group_result_in_the_same_line_and_a_list_without_a_result():
    same = [{"text": "HBsAg / Anti-HCV / HIV :- non reactive"}]
    assert W.page_result_reason("HBsAg", same) is not None and W.page_result_reason("HIV", same) is not None
    plain = [{"text": "HbA1c/FBS/PPBS/TSH"}, {"text": "Normal diet and walk"}]
    assert W.page_result_reason("FBS", plain) is None and W.page_result_reason("TSH", plain) is None
    assert W.page_result_reason("HbA1c", plain) is None


def test_the_piece_written_below_is_found_by_its_place_not_by_its_order():
    # MEASURED: the OCR order put "metatinine-0.?3" and "64PT-17" between "?HBsAg/Anti-HCV" and the wrapped "HIV?? non reactive."
    blocks = [{"text": "?HBsAg/Anti-HCV", "bbox": [294, 1099, 521, 1152]}, {"text": "metatinine-0.?3", "bbox": [75, 1114, 259, 1144]},
              {"text": "64PT-17", "bbox": [80, 1151, 180, 1186]}, {"text": "HIV?? non reactive.", "bbox": [341, 1154, 580, 1200]},
              {"text": "CBC Creatinine ?APT/SGOT CRP", "bbox": [660, 1178, 896, 1322]}]
    assert W.page_result_reason("HBsAg", blocks).startswith("a result is already written")
    assert W.page_result_reason("Anti-HCV", blocks).startswith("a result is already written")
    assert W.page_result_reason("CRP", blocks) is None


def test_anti_hiv_is_the_hiv_written_on_the_page():
    blocks = [{"text": "?HBsAg/Anti-HCV", "bbox": [294, 1099, 521, 1152]}, {"text": "HIV?? non reactive.", "bbox": [341, 1154, 580, 1200]}]
    assert W.page_result_reason("Anti-HIV", blocks).startswith("a result is already written")
    assert W.page_result_reason("HIV", blocks).startswith("a result is already written")
