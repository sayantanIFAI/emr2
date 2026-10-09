"""Straighten and crop (IM-S2): upright pages, pages cut out of photos, the stored transform and its inverse.

Synthetic pages only. The thresholds are PLACEHOLDERS; real handwriting and real photos are not measured.
"""
from __future__ import annotations

import io
import json

import cv2
import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFont

from cdi_adapter.config import settings
from cdi_adapter.ingest import geometry as G
from cdi_adapter.ingest import pages
from cdi_adapter.recognition import quality as q

LINES = ["Tab Metformin 500 mg 1-0-1 after food", "HbA1c 7.8 % (ref 4.0 - 5.6)",
         "Creatinine 0.9 mg/dL  Urea 28 mg/dL", "BP 130/80 mmHg  Pulse 76 /min  SpO2 98 %",
         "Review after 2 weeks with fasting sugar", "Syp Cough Relief 5 ml TDS x 5 days",
         "Patient: Anil Mehra  Age 54  Sex M", "Dr A Sen  MD  Reg No 12345  Medicine"]


def _page(lines=LINES, px=34, w=1700, h=1400) -> np.ndarray:
    im = Image.new("L", (w, h), 255)
    d = ImageDraw.Draw(im)
    f, y, i = ImageFont.load_default(size=px), 100, 0
    while y + px * 1.9 < h * 0.9:
        d.text((100, y), lines[i % len(lines)], fill=20, font=f)
        y, i = y + int(px * 1.9), i + 1
    return cv2.cvtColor(np.array(im), cv2.COLOR_GRAY2BGR)


def _png(arr) -> bytes:
    ok, enc = cv2.imencode(".png", arr)
    assert ok
    return enc.tobytes()


def _norm(arr):
    """normalize_with_source on a BGR array: (straightened colour copy, meta)."""
    _n, src, meta = pages.normalize_with_source(_png(arr))
    return cv2.imdecode(np.frombuffer(src, np.uint8), cv2.IMREAD_COLOR), meta


@pytest.fixture(autouse=True)
def _fast(monkeypatch):
    monkeypatch.setattr(settings, "denoise_enabled", False)           # ~10 s a page and not under test


def _residual_tilt(gray) -> float:
    """Tilt left in a page by the projection-profile method (independent of the code under test)."""
    small = cv2.resize(gray, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    ink = (cv2.bitwise_not(small) > 100).astype(np.uint8)
    h, w = ink.shape
    best, best_s = 0.0, -1.0
    for a in np.arange(-10, 10.01, 0.25):
        r = cv2.warpAffine(ink, cv2.getRotationMatrix2D((w / 2, h / 2), a, 1.0), (w, h), flags=cv2.INTER_NEAREST)
        v = float(np.var(r.sum(axis=1)))
        if v > best_s:
            best, best_s = float(a), v
    return best


def _upright(gray) -> bool:
    s, _n = G.upright_score(gray)
    return s is not None and s > 0.015


def _photo(page: np.ndarray, corners, size=(2400, 1900), bg=(60, 70, 80), seed=3):
    """The page photographed from an angle: warped onto a dark desk with texture."""
    h, w = page.shape[:2]
    src = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], np.float32)
    dst = np.array(corners, np.float32)
    m = cv2.getPerspectiveTransform(src, dst)
    desk = np.full((size[1], size[0], 3), bg, np.uint8)
    noise = np.random.default_rng(seed).normal(0, 7, desk.shape)
    desk = np.clip(desk + noise, 0, 255).astype(np.uint8)
    warped = cv2.warpPerspective(page, m, size, flags=cv2.INTER_CUBIC, borderValue=(0, 0, 0))
    mask = cv2.warpPerspective(np.full(page.shape[:2], 255, np.uint8), m, size)
    out = np.where(mask[..., None] > 0, warped, desk)
    return out, m.astype(float)


# ------------------------------------------------------------------ the matrices


@pytest.mark.parametrize("k", [1, 2, 3])
def test_rot90_matrix_is_exactly_what_numpy_does(k):
    rng = np.random.default_rng(0)
    img = rng.integers(0, 255, (40, 70), dtype=np.uint8)
    out = np.rot90(img, k)
    m = G.rot90_matrix(70, 40, k)
    for x, y in [(0, 0), (69, 0), (69, 39), (0, 39), (12, 7), (50, 31)]:
        nx, ny = G.apply(m, np.array([[x, y]]))[0]
        assert out[round(ny), round(nx)] == img[y, x]


def test_a_box_on_the_straightened_copy_maps_back_through_a_rotation():
    t = G.new_transform(70, 40)
    G.add_step(t, {"op": "rotate90", "k": 1}, G.rot90_matrix(70, 40, 1), (40, 70))
    img = np.zeros((40, 70), np.uint8)
    img[10:20, 30:60] = 255                                           # a bar in the original
    out = np.rot90(img, 1)
    ys, xs = np.where(out > 0)
    box = [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]
    back = G.map_box_to_original(box, t)
    assert back == [30, 10, 60, 20]                                    # exactly the original bar


def test_the_transform_is_plain_json():
    t = G.new_transform(100, 80)
    G.add_step(t, {"op": "deskew", "angle_deg": 1.5}, G.affine3(cv2.getRotationMatrix2D((50, 40), 1.5, 1.0)), (100, 80))
    json.dumps(t)
    assert t["src_wh"] == [100, 80] and t["steps"][0]["op"] == "deskew"


# ------------------------------------------------------------------ AC2 / AC5: a sideways page is turned upright and recorded


@pytest.mark.parametrize("k", [1, 3])
def test_ac2_a_sideways_page_is_turned_upright_and_the_rotation_is_recorded(k):
    page = _page()
    sideways = np.ascontiguousarray(np.rot90(page, k))
    out, meta = _norm(sideways)
    gray = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
    assert "rotate90" in meta["steps"] and _upright(gray)
    assert q.orientation_ratio(gray) < 0.5                                 # lines run along the page again
    step = next(s for s in meta["transform"]["steps"] if s["op"] == "rotate90")
    assert step["k"] == (4 - k) % 4 and meta["quality"]["passed"] is True
    assert out.shape[:2] == page.shape[:2]


def test_a_sideways_page_whose_way_up_cannot_be_told_is_held_not_guessed():
    caps = _page([t.upper() for t in LINES])                              # all capitals are symmetric top to bottom
    _out, meta = _norm(np.ascontiguousarray(np.rot90(caps, 1)))
    assert "rotate90" not in meta["steps"]
    assert meta["quality"]["passed"] is False and meta["quality"]["reason_codes"] == ["orientation_uncertain"]
    assert any(s["op"] == "sideways_undecided" for s in meta["transform"]["steps"])


def test_an_upright_page_is_left_alone():
    _out, meta = _norm(_page())
    assert not {"rotate90", "rotate180", "perspective"} & set(meta["steps"]) and meta["quality"]["passed"] is True
    assert meta["transform"]["out_wh"] == meta["transform"]["src_wh"]


def test_ac5_an_upside_down_page_is_turned_upright_when_enabled(monkeypatch):
    monkeypatch.setattr(settings, "orient_upside_down", True)
    upside = np.ascontiguousarray(np.rot90(_page(), 2))
    out, meta = _norm(upside)
    assert "rotate180" in meta["steps"] and _upright(cv2.cvtColor(out, cv2.COLOR_BGR2GRAY))
    assert next(s for s in meta["transform"]["steps"] if s["op"] == "rotate180")["upright_score"] < 0


def test_upside_down_is_not_touched_by_default():
    """OFF by default: the signal is measured on synthetic text only (see config)."""
    _out, meta = _norm(np.ascontiguousarray(np.rot90(_page(), 2)))
    assert "rotate180" not in meta["steps"]


def test_enabling_upside_down_never_turns_an_upright_page(monkeypatch):
    monkeypatch.setattr(settings, "orient_upside_down", True)
    for px in (24, 34, 52):
        _out, meta = _norm(_page(px=px))
        assert "rotate180" not in meta["steps"], px


def test_each_page_of_one_prescription_is_turned_on_its_own():
    a, b = _page(), np.ascontiguousarray(np.rot90(_page(), 1))
    (_oa, ma), (ob, mb) = _norm(a), _norm(b)
    assert "rotate90" not in ma["steps"] and "rotate90" in mb["steps"] and _upright(cv2.cvtColor(ob, cv2.COLOR_BGR2GRAY))


def test_the_orientation_step_can_be_switched_off(monkeypatch):
    monkeypatch.setattr(settings, "orient_enabled", False)
    _out, meta = _norm(np.ascontiguousarray(np.rot90(_page(), 1)))
    assert "rotate90" not in meta["steps"]


# ------------------------------------------------------------------ a page photographed at an angle


CORNERS = [[330, 260], [2050, 380], [1950, 1660], [250, 1500]]            # tilted, in perspective


def test_a_page_in_a_photo_is_cut_out_and_flattened():
    page = _page()
    photo, _h = _photo(page, CORNERS)
    out, meta = _norm(photo)
    assert "perspective" in meta["steps"] and meta["quality"]["passed"] is True
    h, w = out.shape[:2]
    assert abs(w / h - page.shape[1] / page.shape[0]) < 0.25 and w > 1300         # about the page's own shape and size
    # flat: the lines are evenly spaced top to bottom (a photo at an angle squeezes one end) and level
    from cdi_adapter.recognition.regions import detect_lines

    def pitch_ratio(arr):
        ys = [(bx[1] + bx[3]) / 2 for bx in detect_lines(cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)) if bx[2] > 300]
        gaps = np.diff(ys)
        third = len(gaps) // 3
        return float(np.mean(gaps[-third:]) / np.mean(gaps[:third]))

    assert 0.9 < pitch_ratio(out) < 1.1
    assert abs(_residual_tilt(cv2.cvtColor(out, cv2.COLOR_BGR2GRAY))) <= 0.5
    step = meta["transform"]["steps"][0]
    assert step["op"] == "perspective" and len(step["quad"]) == 4


def test_ac3_a_box_on_the_straightened_page_lands_on_the_same_ink_in_the_original():
    page = _page()
    photo, _h = _photo(page, CORNERS)
    out, meta = _norm(photo)
    gray = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
    from cdi_adapter.recognition.regions import detect_lines

    boxes = [b for b in detect_lines(gray) if b[2] <= 1450]               # text lines (the right margin is blank)
    assert len(boxes) >= 6
    pg = cv2.cvtColor(photo, cv2.COLOR_BGR2GRAY)

    def ink(box):
        x0, y0, x1, y1 = G.map_box_to_original(box, meta["transform"])
        assert 0 <= x0 < x1 <= photo.shape[1] and 0 <= y0 < y1 <= photo.shape[0]
        return float((pg[y0:y1, x0:x1] < 110).mean())

    on_text = [ink(b) for b in boxes[:6]]
    margin = [ink([1480, b[1], 1640, b[3]]) for b in boxes[:6]]           # same rows, in the blank right margin
    assert min(on_text) > 0.04, on_text                                     # writing is where the box maps back
    assert float(np.median(margin)) < 0.005, margin                         # and nothing where there is none


def test_the_recovered_page_corners_are_close_to_the_true_ones():
    page = _page()
    photo, _h = _photo(page, CORNERS)
    _out, meta = _norm(photo)
    quad = np.array(meta["transform"]["steps"][0]["quad"], float)
    truth = np.array(CORNERS, float)
    assert float(np.max(np.linalg.norm(quad - truth, axis=1))) < 25, quad


def _clutter_with_text():
    rng = np.random.default_rng(9)
    clutter = np.clip(rng.normal(90, 40, (1900, 2400, 3)), 0, 255).astype(np.uint8)      # no page, just a busy desk
    clutter[700:1200, 900:1500] = 20
    text = _page()[:, :1500]
    clutter[300:300 + text.shape[0] // 2, 200:200 + text.shape[1] // 2] = cv2.resize(
        text, (text.shape[1] // 2, text.shape[0] // 2))
    return clutter


def test_ac4_a_photo_whose_page_edges_cannot_be_found_is_read_and_flagged_not_held():
    out, meta = _norm(_clutter_with_text())
    assert "perspective" not in meta["steps"]
    assert meta["quality"]["passed"] is True and meta["quality"]["reasons"] == []        # never refused
    assert meta["quality"]["warning_codes"] == ["page_edges_not_found"]                   # but flagged: needs a check
    assert "page_edges_not_found" not in meta["quality"]["reason_codes"]


def test_the_content_box_wraps_the_writing_with_a_margin_and_cuts_the_background_away():
    rng = np.random.default_rng(4)
    bg = cv2.GaussianBlur(rng.normal(110, 60, (1900, 2400)).astype(np.float32), (0, 0), 60)   # soft blotches, no strokes
    gray = np.clip(bg, 0, 255).astype(np.uint8)
    text = _page()[:, :1500]
    th, tw = text.shape[:2]
    patch = cv2.resize(cv2.cvtColor(text, cv2.COLOR_BGR2GRAY), (tw // 2, th // 2))
    gray[500:500 + th // 2, 700:700 + tw // 2] = patch
    ys, xs = np.nonzero(patch < 100)                                                      # where the writing really is
    wx0, wx1, wy0, wy1 = 700 + xs.min(), 700 + xs.max(), 500 + ys.min(), 500 + ys.max()
    x0, y0, x1, y1 = G.find_content_box(gray)
    assert (x1 - x0) * (y1 - y0) < 0.5 * gray.size                                        # the background is cut away
    assert x0 <= wx0 and y0 <= wy0 and x1 > wx1 and y1 > wy1                              # ... never into the writing


def test_the_content_box_is_none_when_there_is_nothing_to_crop():
    assert G.find_content_box(np.full((1000, 800), 200, np.uint8)) is None               # blank: no writing found
    rng = np.random.default_rng(2)
    assert G.find_content_box(rng.integers(0, 255, (1000, 800)).astype(np.uint8)) is None  # strokes everywhere: no block


def test_a_crop_step_is_a_translation_so_boxes_map_back_to_the_original_pixels():
    t = G.new_transform(2000, 1500)
    G.add_step(t, {"op": "crop_to_content", "box": [300, 200, 1300, 1100]},
               G.affine3(np.array([[1, 0, -300], [0, 1, -200]], float)), (1000, 900))
    assert G.map_box_to_original([10, 20, 60, 50], t) == [310, 220, 360, 250]             # shifted by the crop's corner


def test_a_page_that_fills_the_frame_is_not_flagged_even_with_a_dark_strip_beside_it():
    page = _page()
    framed = page.copy()
    framed[:, :90] = 25                                                                   # a hand / dark edge at one side
    _out, meta = _norm(framed)
    assert meta["quality"]["passed"] is True
    assert "page_edges_not_found" not in (meta["quality"].get("warning_codes") or [])


def test_the_loose_search_never_returns_a_quad_that_cuts_through_the_writing():
    # a bright page whose lower half sits in shadow, on a bright busy background: the bright part alone
    # is "a page-shaped region", but writing continues below it, so it must not be taken as the page
    photo, _h = _photo(_page(), CORNERS)
    gray = cv2.cvtColor(photo, cv2.COLOR_BGR2GRAY)
    shaded = gray.copy()
    shaded[1000:, :] = (shaded[1000:, :] * 0.55).astype(np.uint8)
    quad = G.find_page_quad_loose(shaded)
    if quad is not None:
        assert float(quad[:, 1].max()) > 1500, quad                                       # reaches the page's bottom


def test_a_flat_scan_that_fills_the_frame_is_not_treated_as_a_photo():
    _out, meta = _norm(_page())
    assert "perspective" not in meta["steps"] and meta["quality"]["passed"] is True


def test_a_page_already_filling_the_frame_is_not_cropped():
    page = _page()
    framed = cv2.copyMakeBorder(page, 12, 12, 12, 12, cv2.BORDER_CONSTANT, value=(40, 40, 40))
    _out, meta = _norm(framed)
    assert "perspective" not in meta["steps"]


def test_the_perspective_step_can_be_switched_off(monkeypatch):
    monkeypatch.setattr(settings, "perspective_enabled", False)
    photo, _h = _photo(_page(), CORNERS)
    _out, meta = _norm(photo)
    assert "perspective" not in meta["steps"] and meta["quality"]["passed"] is True


# ------------------------------------------------------------------ the deskew keeps its record too


def test_the_deskew_is_recorded_as_a_step_of_the_stored_transform():
    tilted = np.array(Image.fromarray(cv2.cvtColor(_page(), cv2.COLOR_BGR2RGB)).rotate(7, fillcolor=(255, 255, 255),
                                                                                         resample=Image.BICUBIC))
    _out, meta = _norm(cv2.cvtColor(tilted, cv2.COLOR_RGB2BGR))
    assert "deskew" in meta["steps"]
    ops = [s["op"] for s in meta["transform"]["steps"]]
    assert ops[-1] == "deskew" and meta["transform"]["steps"][-1]["angle_deg"] == pytest.approx(meta["skew_deg"])
    # a point on the straightened page maps back near where it was in the original
    t = meta["transform"]
    box = [600, 400, 900, 440]
    back = G.map_box_to_original(box, t)
    assert back[0] < back[2] and back[1] < back[3]


def test_the_original_bytes_are_never_touched_and_the_transform_is_stored_with_the_page():
    arr = _page()
    raw = _png(arr)
    snapshot = bytes(raw)
    _n, _s, meta = pages.normalize_with_source(raw)
    assert raw == snapshot and set(meta["transform"]) == {"matrix", "src_wh", "out_wh", "steps"}
    json.dumps(meta["transform"])
    assert np.asarray(meta["transform"]["matrix"]).shape == (3, 3)


def test_a_held_page_still_goes_through_the_normal_hold_path():
    """The ingest service holds on ``meta['quality']['passed']`` and ``reasons`` in enforce mode."""
    from cdi_adapter.ingest.service import quality_hold_reasons

    class P:
        page_no = 1

    caps = _page([t.upper() for t in LINES])
    _o, _s, meta = pages.normalize_with_source(_png(np.ascontiguousarray(np.rot90(caps, 1))))
    p = P()
    p.preproc = meta
    held = quality_hold_reasons([p])
    assert held and "which way is up" in held[0]


def test_the_page_image_roundtrip_through_pillow_is_unchanged_by_this_step():
    out, _meta = _norm(_page())
    assert out.shape == _page().shape and isinstance(Image.open(io.BytesIO(_png(out))), Image.Image)


# ------------------------------------------------------------------ guards, and the deskew that a page edge used to fool


def test_a_tiny_picture_is_never_looked_at_for_a_page_edge():
    img = np.zeros((60, 120, 3), np.uint8)
    img[:, 60:] = 255
    assert G.find_page_quad(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)) is None


def test_a_bright_shape_with_no_writing_is_not_a_page():
    img = np.full((1500, 2000, 3), 40, np.uint8)
    img[300:1200, 400:1500] = 235                                          # a blank bright rectangle
    assert G.find_page_quad(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)) is None


def test_a_dark_scanner_border_does_not_fool_the_deskew():
    """The old fit of a rectangle to all ink turned a level page when a dark edge was present."""
    page = _page()
    framed = cv2.copyMakeBorder(page, 40, 40, 40, 40, cv2.BORDER_CONSTANT, value=(25, 25, 25))
    _out, meta = _norm(framed)
    assert abs(meta["skew_deg"]) <= 0.15 and "deskew" not in meta["steps"]


@pytest.mark.parametrize("tilt", [-12, -5, 5, 12])
def test_the_deskew_estimate_matches_the_real_tilt(tilt):
    tilted = np.array(Image.fromarray(cv2.cvtColor(_page(), cv2.COLOR_BGR2RGB)).rotate(
        tilt, fillcolor=(255, 255, 255), resample=Image.BICUBIC))
    est = pages._estimate_skew_deg(cv2.cvtColor(tilted, cv2.COLOR_RGB2GRAY))
    assert est == pytest.approx(-tilt, abs=0.4)


def test_a_page_with_no_text_is_not_rotated():
    assert pages._estimate_skew_deg(np.full((1000, 800), 255, np.uint8)) == 0.0


def test_an_undecided_ink_rule_hands_the_decision_to_the_printed_text_vote(monkeypatch):
    """MEASURED on a real sideways photo: ink-shape ratio 1.2 (neither upright nor sideways), page not judged a photo: the printed-text vote was never
    asked and the page was read sideways. Now the vote decides whenever the ink rule is not decisive."""
    from cdi_adapter.ingest import orient_ocr, pages

    asked = []

    def vote(arr, host=None, candidates=(0, 1, 2, 3)):
        asked.append(candidates)
        return 1, {"scores": {"0": 148.2, "1": 278.2, "2": 23.0, "3": 23.0}, "best": 1, "decided": True}

    monkeypatch.setattr(settings, "orient_enabled", True)
    monkeypatch.setattr(settings, "orient_ocr_check", True)
    monkeypatch.setattr(pages, "_orientation_ratio", lambda gray: 1.2)
    monkeypatch.setattr(orient_ocr, "vote", vote)
    out, meta = _norm(_page())
    assert asked and "rotate90" in meta["steps"]


def test_a_decisively_upright_page_does_not_ask_the_vote(monkeypatch):
    from cdi_adapter.ingest import orient_ocr, pages

    asked = []
    monkeypatch.setattr(settings, "orient_enabled", True)
    monkeypatch.setattr(settings, "orient_ocr_check", True)
    monkeypatch.setattr(pages, "_orientation_ratio", lambda gray: 0.15)
    monkeypatch.setattr(orient_ocr, "vote", lambda *a, **k: asked.append(1) or (None, {}))
    _out, meta = _norm(_page())
    assert not asked and "rotate90" not in meta["steps"]
