from cdi_adapter.extract import resolve_llm as R


def B(*texts):
    return [{"text": t, "bbox": [0, 40 * i, 600, 40 * i + 30], "recognition": {"state": "single_engine"}} for i, t in enumerate(texts)]


def test_agree_when_text_and_answer_name_the_same_tests():
    assert R.texts_agree(["CBC", "LFT", "FBS"], B("Adv: CBC, LFT, FBS")) is True


def test_no_skip_when_they_differ_or_one_is_empty():
    assert R.texts_agree(["CBC", "LFT"], B("Adv: CBC, LFT, FBS")) is False        # the text holds one the answer lacks
    assert R.texts_agree(["CBC", "LFT", "FBS"], B("Adv: CBC, LFT")) is False      # the answer holds one the text lacks
    assert R.texts_agree([], B("Adv: CBC")) is False
    assert R.texts_agree(["CBC"], B("Tab Pantocid 40")) is False


def test_no_skip_when_an_order_line_could_not_be_read():
    page = B("Adv: CBC, LFT", "Rx")
    page.append({"text": "Ponl:@mt", "bbox": [0, 400, 600, 430], "recognition": {"state": "no_reading"}})
    assert R.texts_agree(["CBC", "LFT"], page) is False
