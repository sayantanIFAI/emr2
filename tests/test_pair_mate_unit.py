import re

import cv2
import numpy as np

from cdi_adapter.extract import pair_mate as P


def blocks():
    return [{"text": "CBC Creatinine ?APT/SGOT CRP", "bbox": [660, 1178, 896, 1322]}, {"text": "Platelet-2.09", "bbox": [52, 992, 433, 1035]}]


def test_the_unreadable_half_of_a_pair_is_found_only_when_it_is_alike_the_other_half():
    got = P.slips(blocks(), set())
    assert [(r, p, m) for _, r, p, m in got] == [("?APT", "SGOT", "SGPT")]
    assert P.slips(blocks(), {"sgpt"}) == []                                  # already listed
    assert P.slips([{"text": "Xyzqw/SGOT"}], set()) == []                     # nothing like SGPT
    assert P.slips([{"text": "SGPT/SGOT"}], set()) == []                      # both read: nothing to do


class Picks:
    """A model that always picks the option with this name, whatever order the options come in."""

    def __init__(self, want):
        self.want = want

    def vlm_json_ex(self, crop, prompt, schema, **k):
        options = dict((name, int(n)) for n, name in ((m.group(1), m.group(2)) for m in re.finditer(r"(\d)\) ([A-Za-z]+(?: of these)?)", prompt)))
        return {"choice": options.get(self.want, 0)}, None


def test_the_pair_is_listed_only_when_most_readings_choose_it():
    ok, png = cv2.imencode(".png", np.full((1500, 1200, 3), 255, np.uint8))
    assert list(P.resolve(Picks("SGPT"), png.tobytes(), blocks(), set())) == ["SGPT"]
    assert P.resolve(Picks("APTT"), png.tobytes(), blocks(), set()) == {}
    assert P.resolve(Picks("none of these"), png.tobytes(), blocks(), set()) == {}
