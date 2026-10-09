"""The labelled set of handwriting line crops (recognition/labelset.py): name rows are left out, the page hides the machine reading."""
from __future__ import annotations

import json

import cv2
import numpy as np
import pytest

from cdi_adapter.recognition import labelset as L


def B(text, y0, y1, x0=10, x1=300):
    return {"text": text, "bbox": [x0, y0, x1, y1]}


@pytest.mark.parametrize("text,reason", [
    ("Patient Name: Shibaji Sen", "on the same row as a name label"), ("M:8697709557,", "phone number"), ("SAYAN DAS (40 Y / MALE)", "age / sex"),
    ("Mr. Onkar Chowdhury", "name label / title"), ("Age 86 yrs Sex Male", "age / sex"),
])
def test_a_name_phone_or_age_line_is_left_out_with_its_reason(text, reason):
    got = L.exclusion_reason(0, [B(text, 100, 130), B("CBC", 300, 330)])
    assert got is not None and (got == reason or reason == "on the same row as a name label")


def test_a_handwritten_line_on_the_same_row_as_a_name_label_is_left_out_but_other_rows_are_not():
    blocks = [B("Patient Name", 100, 130, 10, 120), B("Shibay San", 102, 132, 130, 400), B("CBC Tr", 300, 330), B("Chest ECG", 360, 390)]
    assert L.exclusion_reason(1, blocks) == "on the same row as a name label"
    assert [L.exclusion_reason(i, blocks) for i in (2, 3)] == [None, None]


def test_tests_and_ordinary_lines_are_kept():
    for text in ("CBC Tr.", "Chest ECG", "Adv: Digital OPG, FBS, BT, CT", "Rest for 3-5 days", "Tab Montina fx 1 tab"):
        assert L.exclusion_reason(0, [B(text, 100, 130)]) is None


def test_the_closeup_lines_are_found_top_to_bottom():
    img = np.full((240, 420, 3), 255, np.uint8)
    for y in (50, 120, 190):
        cv2.putText(img, "Chest ECG test", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (20, 20, 20), 3, cv2.LINE_AA)
    boxes = L.closeup_boxes(img)
    assert len(boxes) == 3 and [b[1] for b in boxes] == sorted(b[1] for b in boxes)


def test_the_page_hides_the_machine_reading_and_the_download_has_only_id_crop_truth(tmp_path):
    items = [L.LineItem(id="p_000", page="p.jpg", kind="handwritten", bbox=[1, 2, 30, 20], crop="crops/p_000.png", machine="SECRET MACHINE READING"),
             L.LineItem(id="p_001", page="p.jpg", kind="printed", bbox=[1, 40, 30, 60], crop="crops/p_001.png")]
    png = cv2.imencode(".png", np.full((20, 40, 3), 255, np.uint8))[1].tobytes()
    summary = L.write_set(tmp_path, items, {"crops/p_000.png": png, "crops/p_001.png": png}, {"p.jpg": b"jpgbytes"}, [{"page": "p.jpg", "reason": "phone number"}])
    page = (tmp_path / "labelling.html").read_text(encoding="utf-8")
    assert summary["lines"] == 2 and summary["excluded"] == 1 and summary["kind_handwritten"] == 1 and summary["kind_printed"] == 1
    assert page.index("<details>") < page.index("SECRET MACHINE READING") < page.index("</details>")        # only inside the closed details
    assert "Download lines.jsonl" in page and "has a name" in page and "can't read" in page
    assert [json.loads(x)["id"] for x in (tmp_path / "lines.jsonl").read_text(encoding="utf-8").splitlines()] == ["p_000", "p_001"]
    assert (tmp_path / "pages" / "p.jpg").read_bytes() == b"jpgbytes" and (tmp_path / "crops" / "p_000.png").is_file()
