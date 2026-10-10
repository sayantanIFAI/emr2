"""Clean-ups of the investigations: an entry that repeats tests already listed, and an unreadable investigation chosen from a short list."""
from __future__ import annotations

import cv2
import numpy as np

from cdi_adapter.extract import unplaced as U


def test_an_entry_that_only_repeats_tests_listed_on_their_own_is_dropped():
    inv = [{"text": "Br Av TSH FT4"}, {"text": "TSH"}, {"text": "FT4"}, {"text": "CBC TSH"}, {"text": "Detox 12 14 mg x 2"}, {"text": "Adv: TSH, FT4"}]
    kept, dropped = U.drop_covered(inv)
    assert sorted(dropped) == ["Adv: TSH, FT4", "Br Av TSH FT4"]
    assert [i["text"] for i in kept] == ["TSH", "FT4", "CBC TSH", "Detox 12 14 mg x 2"]          # CBC is not listed on its own: that entry stays


def test_a_lone_entry_is_never_dropped():
    kept, dropped = U.drop_covered([{"text": "TSH FT4"}])
    assert dropped == [] and len(kept) == 1                                                   # nothing else lists them: this is the only place they are


def test_the_names_most_alike_the_readings_are_offered_and_an_unrelated_one_is_not():
    got = U.candidates(["Lang?erseid?", "Langueschy", "Langnesby", "Lanfugsenty"])
    assert got[0] == "Laryngoscopy" and "Endoscopy" not in got[:1]
    assert U.candidates(["Qzxv"]) == []


def test_only_an_unreadable_investigation_is_looked_at_again():
    assert U._qualifies("Lang?erseid?", {})
    assert not U._qualifies("Detox 12 14 mg x 2", {}) and not U._qualifies("Tab Lanex 5 mg", {}) and not U._qualifies("TSH", {}) and not U._qualifies("Adv", {})


PAGE = np.full((800, 1200, 3), 255, np.uint8)
ok, _png = cv2.imencode(".png", PAGE)
IMAGE = _png.tobytes()
BLOCKS = [{"text": "Lang?erseid?", "bbox": [526, 642, 887, 703]}, {"text": "Br ?v", "bbox": [577, 534, 671, 587]}]


class Chooser:
    """Reads the line badly and, when asked to choose, always picks the option named ``pick`` (or 'none of these')."""

    def __init__(self, pick):
        self.pick = pick

    def vlm_generate_ex(self, png, prompt, max_tokens=40):
        return "Langueschy", "m"

    def vlm_json_ex(self, png, prompt, schema, **k):
        import re
        opts = dict((name, int(n)) for n, name in re.findall(r"(\d)\) ([^.]*?)(?= \d\)|\.)", prompt))
        return {"choice": opts.get(self.pick, 0)}, None


def test_an_unreadable_investigation_is_taken_only_when_most_answers_pick_the_same_name():
    p = {"investigations": [{"text": "Lang?erseid?"}, {"text": "TSH"}]}
    got = U.reread(Chooser("Laryngoscopy"), IMAGE, BLOCKS, p)
    assert got == {"Lang?erseid?": "Laryngoscopy"}
    assert [i["text"] for i in p["investigations"]] == ["Laryngoscopy", "TSH"] and "Laryngoscopy" in p["_corroborated"]
    assert "chosen from a short list" in p["_text_scan"]["Laryngoscopy"]
    q = {"investigations": [{"text": "Lang?erseid?"}]}
    assert U.reread(Chooser("none of these"), IMAGE, BLOCKS, q) == {} and q["investigations"][0]["text"] == "Lang?erseid?"       # nothing settles: left as it was


def test_an_ent_doctors_unreadable_word_is_offered_laryngoscopy_whatever_its_letters():
    # the doctor is ENT: the usual ENT investigations are offered even when no letter of the reading fits (the model chooses by looking, or says none)
    assert "Laryngoscopy" in U.candidates(["Qzxv blah"], "ent")
    assert U.candidates(["Qzxv blah"], None) == []
    assert "Colonoscopy" not in U.candidates(["Qzxv blah"], "ent")


def test_the_department_the_front_desk_names_is_one_of_the_list_or_nothing():
    from cdi_adapter.extract import department as D

    assert D.normalise_hint("ENT") == "ent" and D.normalise_hint(" General  Medicine ") == "general medicine"
    assert D.normalise_hint("") is None and D.normalise_hint("not known") is None and D.normalise_hint("x; drop table") is None


def test_the_upload_page_asks_for_the_department_and_sends_it():
    from cdi_adapter.webapp.upload_page import ADMIN_PAGE

    assert 'id="dept"' in ADMIN_PAGE and '<option value="ent">ENT</option>' in ADMIN_PAGE and 'fd.append("department",dept)' in ADMIN_PAGE


def test_a_covered_entry_in_the_advice_list_is_not_made_a_test_again():
    from cdi_adapter.extract import service as X

    class Ctx:
        def __init__(self):
            self.added, self.med_resolved = [], {}

        def add(self, **kw):
            self.added.append(kw)
            return "id"

    payload = {"advice": [{"text": "Br Av TSH FT4"}], "investigations": [{"text": "TSH", "source": "list_context"}, {"text": "FT4", "source": "list_context"}]}
    before, after = Ctx(), Ctx()
    X._facts_prescription(before, payload)
    X._facts_prescription(after, {**payload, "_covered": ["Br Av TSH FT4"]})
    assert ("investigation_order", "Br Av TSH FT4") in [(a["fact_type"], a["local_text"]) for a in before.added]       # why the guard exists
    kinds = [(a["fact_type"], a["local_text"]) for a in after.added]
    assert ("advice", "Br Av TSH FT4") in kinds and ("investigation_order", "Br Av TSH FT4") not in kinds
    assert ("investigation_order", "TSH") in kinds and ("investigation_order", "FT4") in kinds
