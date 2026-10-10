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
