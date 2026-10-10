"""A box of tests the doctor wrote by hand, read again as one enlarged picture (extract/test_box.py)."""
from __future__ import annotations

import cv2
import numpy as np

from cdi_adapter.extract import test_box as T

# the line reader's blocks of a real page: the box of tests cut into pieces and misread
BLOCKS = [
    {"text": "Bp 120/80", "bbox": [52, 147, 125, 193]},
    {"text": "cbc, er", "bbox": [928, 221, 1168, 273]},          # b2
    {"text": "TSU/RTU", "bbox": [640, 230, 895, 287]},            # b3
    {"text": "R ?Jc", "bbox": [831, 286, 977, 348]},              # b4
    {"text": "CRB", "bbox": [1004, 286, 1105, 323]},              # b5
    {"text": "TgE", "bbox": [647, 294, 761, 349]},                # b6
    {"text": "yRB", "bbox": [772, 306, 818, 342]},                # b7
    {"text": "Cap Beta 10", "bbox": [869, 417, 1064, 481]},       # b8: a medicine, far from the box
]


def payload():
    return {"investigations": [{"text": "TSU/RTU", "evidence": ["b3"]}, {"text": "CBC, ER", "evidence": ["b2"]}, {"text": "TgE", "evidence": ["b6"]},
                               {"text": "yRB", "evidence": ["b7"]}, {"text": "ESR", "source": "second_look", "evidence": []},
                               {"text": "RBC", "source": "second_look", "evidence": []}]}


PAGE = np.full((900, 1300, 3), 255, np.uint8)
ok, _png = cv2.imencode(".png", PAGE)
IMAGE = _png.tobytes()


class Reader:
    """Answers like the real model did on the box: the plain transcription is bad, the request to list the tests is right at two sizes of three."""

    AS_MEASURED = ["TSU/FTU, CBSI, ERI\nTE, RBT, CRG", "TSH, FT4, CBC, ESR\nTgE, RBC, CRP", "TSH, FT4, CBC, ESR\nT3E, RBC, CRP"]     # the plain readings at the three sizes

    def __init__(self, listed, plain=None):
        self.listed, self.plain, self.n, self.m = list(listed), plain if plain is not None else self.AS_MEASURED, 0, 0

    def vlm_generate_ex(self, png, prompt, max_tokens=100):
        if prompt == T.PLAIN:
            i, self.m = self.m, self.m + 1
            return (self.plain[i % len(self.plain)] if isinstance(self.plain, list) else self.plain), "m"
        i, self.n = self.n, self.n + 1
        return self.listed[i % len(self.listed)], "m"


def test_the_box_the_page_answer_cited_is_found_and_the_medicine_line_is_not_in_it():
    boxes = T._boxes(BLOCKS, payload())
    assert len(boxes) == 1
    x0, y0, x1, y1 = boxes[0]
    assert x0 == 640 and y0 == 221 and x1 == 1168 and y1 == 349


def test_the_box_is_read_again_and_the_misreadings_of_it_go():
    p = payload()
    r = Reader(["TSH, FT4, CBC, ESR\nIgE, RBS, CRP", "TSH\nFT4\nCBC, ESR\nIgE\nRBS\nCRP", "TSH, FT4, CBC, ESR\nIgE, RBS, CRP"])
    info = T.reread(r, IMAGE, BLOCKS, p)
    texts = [i["text"] if isinstance(i, dict) else i for i in p["investigations"]]
    assert info and set(info["accepted"]) == {"TSH", "FT4", "CBC", "ESR", "RBS", "CRP", "IgE"}
    got = {t.casefold() for t in texts}
    assert {"tsh", "ft4", "cbc", "esr", "rbs", "crp", "ige"} <= got
    assert not ({"tsu/rtu", "tge", "yrb", "rbc", "er"} & got)               # the line misreadings and "RBC" (one letter from RBS) are gone
    assert "TSH" in p["_corroborated"] and "box of tests" in p["_text_scan"]["TSH"]


def test_a_test_the_hints_could_have_suggested_but_no_plain_reading_supports_is_not_accepted():
    p = {"investigations": [{"text": "CBC", "evidence": ["b2"]}]}
    r = Reader(["CBC, HbA1c", "CBC, HbA1c", "CBC, HbA1c"], plain="CBC, ESR")                   # HbA1c is in the hint list; nothing written looks like it
    info = T.reread(r, IMAGE, BLOCKS, p)
    assert info["accepted"] == ["CBC"]
    assert "HbA1c" not in [i["text"] if isinstance(i, dict) else i for i in p["investigations"]]


def test_a_test_listed_at_only_one_size_is_not_accepted():
    p = {"investigations": [{"text": "CBC", "evidence": ["b2"]}]}
    r = Reader(["CBC, ESR", "CBC", "CBC"], plain="CBC, ESR")
    info = T.reread(r, IMAGE, BLOCKS, p)
    assert info["accepted"] == ["CBC"]


def test_a_page_with_no_cited_box_or_an_unreadable_one_is_left_alone():
    assert T.reread(Reader(["CBC"]), IMAGE, BLOCKS, {"investigations": [{"text": "Hemogram profile", "evidence": []}]}) is None       # no line says that: no box
    p = payload()
    before = [dict(i) for i in p["investigations"]]
    assert T.reread(Reader(["", "", ""], plain=""), IMAGE, BLOCKS, p) is None
    assert p["investigations"] == before


# the OCR lines of a real page (the doctor ticked two lines: "USG (whole Abdomen)" and "B/F -> Ca2+, Vit D3, Uric acid"); the answer cited no line for either
USG_BLOCKS = [
    {"text": "Epigastric pain", "bbox": [93, 909, 327, 977]}, {"text": "Cap.", "bbox": [385, 949, 452, 988]}, {"text": "?SGT[whole]", "bbox": [38, 1013, 234, 1050]},
    {"text": "udiliv(300)", "bbox": [496, 1011, 684, 1049]}, {"text": "Abdocmem", "bbox": [160, 1046, 312, 1088]}, {"text": "√ B/F →", "bbox": [44, 1081, 165, 1138]},
    {"text": "Hepamerz", "bbox": [512, 1103, 672, 1142]}, {"text": "Ca??", "bbox": [181, 1110, 270, 1139]}, {"text": "un?c ac?d", "bbox": [263, 1145, 373, 1169]},
    {"text": "?it D3", "bbox": [148, 1150, 257, 1182]}, {"text": "1 month", "bbox": [740, 1156, 893, 1187]},
]


def usg_payload():
    return {"investigations": ["VSGT (whole Abdomen)", "B/F → +, Vit D3, Uric acid"]}


def test_a_line_of_tests_the_answer_did_not_cite_is_found_by_what_it_says():
    boxes = T._boxes(USG_BLOCKS, usg_payload())
    assert len(boxes) == 1
    x0, y0, x1, y1 = boxes[0]
    assert x0 <= 38 and y0 <= 1013 and x1 >= 373 and y1 >= 1182 and x1 < 600           # both ticked lines, and not the medicine column to the right


def test_a_tick_joined_to_the_first_letter_and_a_superscript_lost_are_read_from_the_enlarged_box():
    p = usg_payload()
    plain = ["VSGT (whole Abdomen)\n√ B/F → Ca??, Vit D3, Uric acid", "USG (whole Abdomen)\n√ B/F → Ca2+, Vit D3, Uric acid", "USG (whole Abdomen)\n√ B/F → Ca2+, Vit D3, Unic acid"]
    listed = ["USG (Whole Abdomen)\nCa2+, Vit D3, Uric acid"] * 3
    ok2, big = cv2.imencode(".png", np.full((1300, 1000, 3), 255, np.uint8))
    info = T.reread(Reader(listed, plain=plain), big.tobytes(), USG_BLOCKS, p)
    texts = [i["text"] if isinstance(i, dict) else i for i in p["investigations"]]
    assert info and {"Calcium", "Ca2+"} & set(info["accepted"]) and any("USG" in a for a in info["accepted"]) and any("Uric" in a for a in info["accepted"])
    assert not any("VSGT" in t or "B/F" in t for t in texts)                                # the misread pieces are gone
    assert any("Ca2" in t or "Calcium" in t for t in texts) and any("USG" in t for t in texts) and any("Vit D3" in t for t in texts)


def test_superscript_and_subscript_forms_are_the_plain_test_names():
    # MEASURED: the enlarged box was read "Ca²⁺" and "Vit D₃"; neither was placed
    assert T._names("USG (whole Abdomen)\nB/F → Ca²⁺,\nVit D₃, Uric acid") >= {"Uric acid"}
    names = {n.casefold() for n in T._names("Ca²⁺, Vit D₃")}
    assert any(n.startswith("ca2") for n in names) and any(n.startswith("vit d3") for n in names)
    from cdi_adapter.extract.test_cluster import placed_text
    assert placed_text("Ca²⁺") is not None and placed_text("Vit D₃") is not None
