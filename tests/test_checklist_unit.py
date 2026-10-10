"""A printed checklist of tests, ordered by marking names (extract/checklist.py)."""
from __future__ import annotations

import cv2
import numpy as np

from cdi_adapter.extract import checklist as C

PLACED = {"cbc", "alt", "ast", "bilirubin", "ft4", "tsh", "crp", "eeg"}


def placed(name: str) -> bool:
    return name.strip().casefold() in PLACED


MENU = [("CBC -", 0), ("Bilirubin / ALT / AST -", 1), ("Liver Function Test", 2), ("FT4 / TSH-", 3), ("CRP -", 4),
        ("Vitamin D (25 OH) -", 5), ("Mantoux Test (5tu) -", 6), ("EEG -", 7), ("Urine C/S -", 8), ("X-Ray Chest PA/ AP View -", 9)]
X0, Y0, PITCH, FONT, SCALE = 150, 120, 44, cv2.FONT_HERSHEY_SIMPLEX, 0.8


def form(strikes=(), ticks=(), above_frame_scribble=False):
    """A white page with a ruled frame and a one-column menu; returns the image and the OCR-like blocks."""
    img = np.full((700, 900, 3), 255, np.uint8)
    cv2.rectangle(img, (100, 90), (700, 90 + PITCH * (len(MENU) + 1)), (0, 0, 0), 2)
    blocks = []
    for text, i in MENU:
        y = Y0 + i * PITCH
        (w, h), base = cv2.getTextSize(text, FONT, SCALE, 2)
        cv2.putText(img, text, (X0, y + h), FONT, SCALE, (0, 0, 0), 2, cv2.LINE_AA)
        blocks.append({"text": text, "bbox": [X0, y - 2, X0 + w, y + h + base + 2]})
    for i, kind in strikes:                                  # a strike through the name, running past it on the left
        y = Y0 + i * PITCH + 14
        w = blocks[i]["bbox"][2] - X0
        cv2.line(img, (X0 - 25, y + 2), (X0 + w // 2, y - 2), (0, 0, 0), 3)
    for i in ticks:                                          # a tick in the margin after the name
        x = blocks[i]["bbox"][2] + 20
        y = Y0 + i * PITCH + 14
        cv2.line(img, (x, y), (x + 8, y + 12), (0, 0, 0), 3)
        cv2.line(img, (x + 8, y + 12), (x + 24, y - 14), (0, 0, 0), 3)
    if above_frame_scribble:                                 # handwriting that crosses the frame's top edge ("15.3 kg" written above the menu)
        cv2.line(img, (X0 + 60, 60), (X0 + 62, 97), (0, 0, 0), 3)
    return img, blocks


def test_a_name_is_split_only_when_every_part_is_a_test():
    assert [n for n, _a, _b in C.names_in("Bilirubin / ALT / AST -", placed)] == ["Bilirubin", "ALT", "AST"]
    assert [n for n, _a, _b in C.names_in("FT4 / TSH-", placed)] == ["FT4", "TSH"]
    assert [n for n, _a, _b in C.names_in("X-Ray Chest PA/ AP View -", placed)] == ["X-Ray Chest PA/ AP View"]
    assert [n for n, _a, _b in C.names_in("Urine C/S -", placed)] == ["Urine C/S"]
    assert C.names_in("- .", placed) == []


def test_a_menu_is_found_from_its_text_lines_and_an_ordinary_page_is_not_one():
    _img, blocks = form()
    menu = C.rows(blocks, placed)
    assert len(menu) == len(MENU)                                           # the line with no dash ("Liver Function Test") is a row of the column too
    handwritten = [{"text": t, "bbox": [50, 60 + 40 * i, 300, 90 + 40 * i]} for i, t in enumerate(["Tab Telma 40 1-0-1", "Tab Met 500 BD", "Review after 2 weeks"] * 4)]
    assert C.rows(handwritten, placed) == []
    assert C.rows([{"text": "CBC -", "bbox": [1, 1, 50, 20]}], placed) == []


def test_a_strike_a_tick_and_nothing_else_are_marks():
    img, blocks = form(strikes=[(7, "strike")], ticks=[3], above_frame_scribble=True)
    menu = C.rows(blocks, placed)
    got = {n for n, _t, _k in C.marked_names(menu, C.mark_rows(img, menu))}
    assert got == {"FT4", "TSH", "EEG"}                                     # the tick after "FT4 / TSH -" orders both of its names; the scribble above the frame orders nothing


def test_a_strike_over_one_name_of_a_row_of_several_orders_only_that_name():
    img, blocks = form()
    i = 1
    x0, y0, x1, y1 = blocks[i]["bbox"]
    # across the middle name ("ALT") only
    names = C.names_in(blocks[i]["text"], placed)
    a, b = names[1][1], names[1][2]
    y = (y0 + y1) // 2
    cv2.line(img, (x0 + int((x1 - x0) * a) - 3, y), (x0 + int((x1 - x0) * b) + 3, y), (0, 0, 0), 3)
    menu = C.rows(blocks, placed)
    assert [n for n, _t, _k in C.marked_names(menu, C.mark_rows(img, menu))] == ["ALT"]


def test_an_unmarked_menu_orders_nothing():
    img, blocks = form()
    menu = C.rows(blocks, placed)
    assert C.mark_rows(img, menu) == []


def test_apply_lists_the_marked_names_drops_the_unmarked_menu_names_and_keeps_what_is_written_elsewhere():
    img, blocks = form(strikes=[(7, "strike")])
    blocks.append({"text": "Adv: CRP after a week", "bbox": [150, 640, 600, 680]})          # CRP written by hand elsewhere: it stays
    ok, png = cv2.imencode(".png", img)
    payload = {"investigations": [{"text": "CBC", "evidence": []}, {"text": "CRP", "evidence": []}, {"text": "Bilirubin / ALT / AST"}, "EEG"]}
    info = C.apply(payload, blocks, png.tobytes(), placed)
    texts = [t if isinstance(t, str) else t["text"] for t in payload["investigations"]]
    assert info and info["marked"] == ["EEG"] and info["rows"] == len(MENU)
    assert "CBC" not in texts and "Bilirubin / ALT / AST" not in texts and "CRP" in texts and "EEG" in texts
    assert "EEG" in payload["_corroborated"] and "struck" not in payload["_text_scan"]["EEG"] and "marked on the printed checklist" in payload["_text_scan"]["EEG"]


def test_a_menu_with_no_mark_says_so_and_lists_nothing_from_it():
    img, blocks = form()
    ok, png = cv2.imencode(".png", img)
    payload = {"investigations": [{"text": "CBC"}, {"text": "EEG"}]}
    info = C.apply(payload, blocks, png.tobytes(), placed)
    assert info["marked"] == [] and payload["investigations"] == [] and payload["_checklist"]["rows"] == len(MENU)


def test_an_ordinary_page_is_left_alone_without_decoding_the_image():
    payload = {"investigations": [{"text": "CBC"}]}
    assert C.apply(payload, [{"text": "Tab Telma 40", "bbox": [1, 1, 100, 20]}], b"not an image", placed) is None
    assert payload == {"investigations": [{"text": "CBC"}]}


def test_boxes_outside_the_picture_are_not_looked_at():
    img, blocks = form(strikes=[(7, "strike")])
    small = cv2.resize(img, (300, 233))
    ok, png = cv2.imencode(".png", small)
    assert C.apply({"investigations": []}, blocks, png.tobytes(), placed) is None
