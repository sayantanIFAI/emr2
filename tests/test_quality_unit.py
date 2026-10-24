"""Picture check (IM-S1): measurements, stable reason codes, tiny text, sideways pages, edge cases.

All pages here are SYNTHETIC. Thresholds are PLACEHOLDERS to be tuned on real pictures (IM-S1 NFR 2);
these tests pin the behaviour the story asks for, not the right numbers.
"""
from __future__ import annotations

import io

import cv2
import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFont

from cdi_adapter.config import settings
from cdi_adapter.ingest import pages
from cdi_adapter.recognition import quality as q

LINES = ["Tab Metformin 500 mg 1-0-1 after food", "HbA1c 7.8 % (ref 4.0 - 5.6)",
         "Creatinine 0.9 mg/dL  Urea 28 mg/dL", "BP 130/80 mmHg  Pulse 76 /min  SpO2 98 %",
         "Review after 2 weeks with fasting sugar", "Syp Cough Relief 5 ml TDS x 5 days"]


def _font(px: int) -> ImageFont.FreeTypeFont:
    return ImageFont.load_default(size=px)


def _page(px: int = 34, w: int = 1700, h: int = 2200, *, left: float = 0.06, paper: int = 255) -> np.ndarray:
    """Grey page of text lines, text height ~px."""
    im = Image.new("L", (w, h), paper)
    d = ImageDraw.Draw(im)
    f, pitch, y, i = _font(px), int(px * 2.0), int(h * 0.05), 0
    while y + pitch < h * 0.95:
        d.text((int(w * left), y), LINES[i % 6] + f" {i}", fill=20, font=f)
        y, i = y + pitch, i + 1
    return np.array(im)


def _bgr(gray: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def _png(arr: np.ndarray) -> bytes:
    ok, enc = cv2.imencode(".png", arr)
    assert ok
    return enc.tobytes()


# ------------------------------------------------------------------ AC1 / AC2 and the recorded measurements


METRIC_KEYS = {"width_px", "height_px", "short_side_px", "blur_var", "glare_frac", "dark_frac",
               "ink_frac", "text_height_px", "text_glyphs", "orientation_ratio", "rotated_90_suspected"}


def test_ac1_a_sharp_page_passes_and_every_measurement_is_recorded():
    rep = q.assess(_bgr(_page()))
    assert rep.passed and rep.reasons == [] and rep.reason_codes == []
    assert METRIC_KEYS <= set(rep.metrics)
    assert rep.metrics["text_height_px"] and rep.metrics["text_height_px"] > 15
    d = rep.as_dict()
    assert set(d) == {"passed", "reasons", "reason_codes", "warnings", "warning_codes", "metrics"}


def test_ac2_a_blurred_page_is_held_with_the_blur_code_and_still_records_everything():
    rep = q.assess(_bgr(cv2.GaussianBlur(_page(), (0, 0), 7)))
    assert not rep.passed and q.BLURRED in rep.reason_codes
    assert any("blurred" in r for r in rep.reasons)
    assert METRIC_KEYS <= set(rep.metrics)             # recorded for a FAILED page too


def test_a_failed_page_keeps_its_measurements_in_the_page_record(monkeypatch):
    """What ingest stores in ``document_page.preproc['quality']`` for pass AND fail."""
    monkeypatch.setattr(settings, "denoise_enabled", False)       # not under test, and ~10 s a page
    for arr, passed in ((_bgr(_page()), True), (_bgr(cv2.GaussianBlur(_page(), (0, 0), 7)), False)):
        _norm, _src, meta = pages.normalize_with_source(_png(arr))
        qual = meta["quality"]
        assert qual["passed"] is passed and METRIC_KEYS <= set(qual["metrics"])
        assert len(qual["reason_codes"]) == len(qual["reasons"])


def test_the_quality_check_can_be_switched_off_by_setting(monkeypatch):
    monkeypatch.setattr(settings, "denoise_enabled", False)
    monkeypatch.setattr(settings, "quality_gate_mode", "off")
    _n, _s, meta = pages.normalize_with_source(_png(_bgr(_page())))
    assert "quality" not in meta


# ------------------------------------------------------------------ reason codes (used by UP-S2)


def _glare_page() -> np.ndarray:
    g = _page(paper=200)
    g[:, : int(g.shape[1] * 0.30)] = 255      # a blown-out patch on grey paper (a photo, not a white scan)
    return g


def _dark_page() -> np.ndarray:
    g = _page()
    g[: int(g.shape[0] * 0.7), :] = 8
    return g


@pytest.mark.parametrize("make,code", [
    (lambda: cv2.GaussianBlur(_page(), (0, 0), 7), q.BLURRED),
    (_glare_page, q.GLARE),
    (_dark_page, q.TOO_DARK),
    (lambda: _page(24, 250, 350), q.RESOLUTION_LOW),                      # below the 300 px short side (a 538 / 577 px screenshot is allowed now)
    (lambda: _page(9, 4000, 5600), q.TEXT_TOO_SMALL),
])
def test_every_hold_has_a_stable_code_one_per_message(make, code):
    rep = q.assess(_bgr(make()))
    assert code in rep.reason_codes and not rep.passed
    assert len(rep.reason_codes) == len(rep.reasons) and set(rep.reason_codes) <= set(q.REASON_CODES)


def test_the_code_names_are_the_contract():
    assert q.REASON_CODES == ("resolution_low", "blurred", "glare", "too_dark", "text_too_small",
                              "orientation_uncertain", "page_edges_not_found")
    assert (q.BLANK_PAGE, q.SIDEWAYS) == ("blank_page", "sideways_suspected")


def test_a_blank_page_is_a_warning_not_a_hold():
    rep = q.assess(_bgr(np.full((2200, 1700), 255, np.uint8)))
    assert rep.warning_codes == [q.BLANK_PAGE] and rep.reason_codes == []


def test_an_undecodable_picture_has_a_code():
    rep = q.assess_png(b"not an image")
    assert not rep.passed and rep.reason_codes == ["undecodable"]


# ------------------------------------------------------------------ AC4: far away, tiny text, big file


def test_ac4_a_huge_photo_with_tiny_text_is_held_move_closer_even_though_the_file_is_large():
    arr = _page(9, 4000, 5600)
    rep = q.assess(_bgr(arr))
    assert rep.metrics["short_side_px"] == 4000 and rep.metrics["blur_var"] > settings.quality_min_blur_var
    assert rep.reason_codes == [q.TEXT_TOO_SMALL]
    assert "move closer" in rep.reasons[0]


def test_the_same_huge_photo_with_readable_text_passes():
    assert q.assess(_bgr(_page(60, 4000, 5600))).passed


def test_text_height_follows_the_font_size():
    sizes = {px: q.text_height_px(_page(px))[0] for px in (16, 24, 44)}
    assert 0.55 * 16 < sizes[16] < 0.95 * 16 and sizes[16] < sizes[24] < sizes[44]
    assert q.text_height_px(_page(24, 4000, 5600))[0] == pytest.approx(q.text_height_px(_page(24))[0], abs=3)


def test_too_little_text_is_not_judged():
    sparse = np.full((2200, 1700), 255, np.uint8)
    cv2.putText(sparse, "Rx", (200, 300), cv2.FONT_HERSHEY_SIMPLEX, 3, 0, 4)
    height, glyphs = q.text_height_px(sparse)
    assert height is None and glyphs < 150
    assert q.TEXT_TOO_SMALL not in q.assess(_bgr(sparse)).reason_codes


def test_the_limits_are_settings_not_code(monkeypatch):
    tiny = _bgr(_page(9, 4000, 5600))
    assert q.TEXT_TOO_SMALL in q.assess(tiny).reason_codes
    monkeypatch.setattr(settings, "quality_min_text_height_px", 0)           # 0 = check off
    assert q.TEXT_TOO_SMALL not in q.assess(tiny).reason_codes
    monkeypatch.setattr(settings, "quality_min_text_height_px", 30)
    assert q.TEXT_TOO_SMALL in q.assess(_bgr(_page(24))).reason_codes
    monkeypatch.setattr(settings, "quality_min_blur_var", 1e9)
    assert q.BLURRED in q.assess(_bgr(_page())).reason_codes


# ------------------------------------------------------------------ sideways pages ("looks sideways")


def _table(w=1700, h=2200) -> np.ndarray:
    im = Image.new("L", (w, h), 255)
    d = ImageDraw.Draw(im)
    f = _font(30)
    for r in range(12):
        y = 200 + r * 90
        d.line((100, y, w - 100, y), fill=0, width=3)
        for c in range(3):
            d.text((130 + c * 520, y + 25), LINES[(r + c) % 6][:18], fill=20, font=f)
    for c in range(4):
        d.line((100 + c * 520, 200, 100 + c * 520, 200 + 12 * 90), fill=0, width=3)
    return np.array(im)


def _narrow() -> np.ndarray:
    g = np.full((2200, 1700), 255, np.uint8)
    g[:, 700:1400] = _page(30, 700, 2200)
    return g


LAYOUTS = {"full_width": lambda: _page(34), "narrow_column": _narrow,
           "sparse": lambda: _page(40, 1700, 700), "table": _table}


@pytest.mark.parametrize("name", sorted(LAYOUTS))
def test_an_upright_page_is_not_flagged_and_a_turned_one_is(name):
    page = LAYOUTS[name]()
    assert not q.assess(_bgr(page)).metrics["rotated_90_suspected"], name
    for k in (1, 3):                                                         # 90 and 270 degrees
        rep = q.assess(_bgr(np.ascontiguousarray(np.rot90(page, k))))
        assert rep.metrics["rotated_90_suspected"] and q.SIDEWAYS in rep.warning_codes, (name, k)
        assert rep.passed                                                    # a warning, never a hold


def test_the_orientation_numbers_leave_a_wide_margin():
    ratios = {name: q.orientation_ratio(LAYOUTS[name]()) for name in LAYOUTS}
    turned = {name: q.orientation_ratio(np.ascontiguousarray(np.rot90(LAYOUTS[name](), 1))) for name in LAYOUTS}
    assert max(ratios.values()) < 0.5 and min(turned.values()) > 2.5     # threshold is 2.0


def test_upside_down_is_not_claimed_it_needs_the_separate_check_of_im_s2():
    assert not q.assess(_bgr(np.rot90(_page(), 2).copy())).metrics["rotated_90_suspected"]


# ------------------------------------------------------------------ a readable picture must not be rejected


def _variants() -> list[tuple[str, np.ndarray]]:
    rng = np.random.default_rng(7)
    out = []
    for px in (24, 34, 50):                                           # ~8-14 pt at 150-300 dpi
        base = _page(px)
        out.append((f"clean {px}px", base))
        noisy = np.clip(base.astype(np.int16) + rng.normal(0, 8, base.shape), 0, 255).astype(np.uint8)
        out.append((f"noise {px}px", noisy))
        buf = io.BytesIO()
        Image.fromarray(base).save(buf, format="JPEG", quality=60)
        out.append((f"jpeg60 {px}px", np.asarray(Image.open(io.BytesIO(buf.getvalue())))))
        out.append((f"light blur {px}px", cv2.GaussianBlur(base, (0, 0), 0.8)))
        tilted = Image.fromarray(base).rotate(6, fillcolor=255, resample=Image.BICUBIC)
        out.append((f"tilt6 {px}px", np.asarray(tilted)))
        out.append((f"grey paper {px}px", np.clip(base * 0.8 + 20, 0, 255).astype(np.uint8)))
    return out


def test_no_readable_synthetic_picture_is_rejected():
    """False-reject count on this SYNTHETIC set (not a rate on real pictures: that needs the
    answer-key set, scripts/quality_eval.py)."""
    reports = [(name, q.assess(_bgr(arr))) for name, arr in _variants()]
    assert [(n, r.reason_codes) for n, r in reports if not r.passed] == []


# ------------------------------------------------------------------ edge / adversarial: never crash, always record


@pytest.mark.parametrize("make", [
    lambda: np.full((1000, 800), 255, np.uint8),                                # blank
    lambda: np.zeros((1000, 800), np.uint8),                                     # all black
    lambda: np.random.default_rng(1).integers(0, 255, (1200, 900)).astype(np.uint8),   # pure noise
    lambda: _page(34)[: 600, :],                                                  # only the top strip (partial page)
    lambda: _page(34)[:, :200],                                                   # a thin slice
    lambda: np.full((40, 40), 128, np.uint8),                                     # tiny
])
def test_odd_pictures_never_crash_and_always_report_measurements(make):
    rep = q.assess(_bgr(make()))
    assert {"short_side_px", "blur_var", "ink_frac"} <= set(rep.metrics)
    assert len(rep.reason_codes) == len(rep.reasons)


def test_an_over_compressed_jpeg_still_gets_a_verdict_with_codes():
    buf = io.BytesIO()
    Image.fromarray(_page()).save(buf, format="JPEG", quality=4)
    rep = q.assess_png(_png(np.asarray(Image.open(io.BytesIO(buf.getvalue())).convert("RGB"))[:, :, ::-1].copy()))
    assert len(rep.reason_codes) == len(rep.reasons) and "blur_var" in rep.metrics


# ------------------------------------------------------------------ cost


def test_the_check_stays_fast_on_a_large_photo():
    import time

    arr = _bgr(_page(60, 3406, 4750))
    q.assess(arr)                                           # warm up
    t = time.perf_counter()
    q.assess(arr)
    assert time.perf_counter() - t < 4.0                    # generous: MEASURED ~0.8 s on a dev PC


# ------------------------------------------------------------------ deskew (existing, IM-S2 AC1 evidence)


def _residual_tilt(gray: np.ndarray) -> float:
    """Tilt left in a page, by the projection-profile method (independent of the code under test)."""
    small = cv2.resize(gray, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    ink = (cv2.bitwise_not(small) > 100).astype(np.uint8)
    h, w = ink.shape
    best, best_s = 0.0, -1.0
    for a in np.arange(-20, 20.01, 0.25):
        r = cv2.warpAffine(ink, cv2.getRotationMatrix2D((w / 2, h / 2), a, 1.0), (w, h), flags=cv2.INTER_NEAREST)
        s = float(np.var(r.sum(axis=1)))
        if s > best_s:
            best, best_s = float(a), s
    return best


@pytest.mark.parametrize("tilt", [-12, -6, -3, 3, 6, 12])
def test_a_tilted_page_comes_out_with_horizontal_lines(tilt, monkeypatch):
    monkeypatch.setattr(settings, "denoise_enabled", False)       # not under test, and ~10 s a page
    base = Image.fromarray(_page(34))
    tilted = np.asarray(base.rotate(tilt, fillcolor=255, resample=Image.BICUBIC))
    assert abs(_residual_tilt(tilted)) > 2                                            # it really is tilted
    _norm, src, meta = pages.normalize_with_source(_png(_bgr(tilted)))
    straightened = cv2.imdecode(np.frombuffer(src, np.uint8), cv2.IMREAD_GRAYSCALE)
    assert "deskew" in meta["steps"] and abs(_residual_tilt(straightened)) <= 0.3


# ------------------------------------------------------------------ AC3: the false-reject report


def _save(folder, name, arr):
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / name
    cv2.imwrite(str(path), arr)
    return path


def test_ac3_the_report_counts_false_rejects_and_false_accepts_with_the_thresholds_used(tmp_path):
    from cdi_adapter.recognition import quality_eval as qe

    good, bad = tmp_path / "good", tmp_path / "bad"
    _save(good, "a.png", _page(34))
    _save(good, "b.png", _page(34, left=0.2))
    _save(good, "c_blurry_but_labelled_readable.png", cv2.GaussianBlur(_page(), (0, 0), 7))   # a false reject
    _save(bad, "d_blurred.png", cv2.GaussianBlur(_page(), (0, 0), 7))
    _save(bad, "e_looks_fine_but_labelled_unreadable.png", _page(34))                          # a false accept
    report = qe.run(qe.load_items(good, bad, None))
    o = report["overall"]
    assert (o["readable"], o["false_rejects"], o["false_reject_rate"]) == (3, 1, round(1 / 3, 4))
    assert (o["unreadable"], o["false_accepts"], o["false_accept_rate"]) == (2, 1, 0.5)
    assert o["false_reject_reasons"] == {"blurred": 1}
    assert report["thresholds"]["quality_min_blur_var"] == settings.quality_min_blur_var
    assert "quality_min_text_height_px" in report["thresholds"]


def test_the_report_splits_by_document_type_from_a_manifest(tmp_path):
    from cdi_adapter.recognition import quality_eval as qe

    _save(tmp_path, "p1.png", _page(34))
    _save(tmp_path, "p2.png", cv2.GaussianBlur(_page(), (0, 0), 7))
    _save(tmp_path, "l1.png", _page(34))
    (tmp_path / "m.csv").write_text(
        "path,label,doc_type\np1.png,readable,prescription\np2.png,readable,prescription\n"
        "l1.png,readable,lab_report\n", encoding="utf-8")
    report = qe.run(qe.load_items(None, None, tmp_path / "m.csv"))
    assert report["by_doc_type"]["prescription"]["false_rejects"] == 1
    assert report["by_doc_type"]["prescription"]["false_reject_rate"] == 0.5
    assert report["by_doc_type"]["lab_report"]["false_reject_rate"] == 0.0


def test_a_bad_label_in_the_manifest_is_refused_and_an_undecodable_file_is_counted(tmp_path):
    from cdi_adapter.recognition import quality_eval as qe

    (tmp_path / "m.csv").write_text("path,label\nx.png,fine\n", encoding="utf-8")
    with pytest.raises(ValueError):
        qe.load_items(None, None, tmp_path / "m.csv")
    broken = tmp_path / "broken"
    broken.mkdir()
    (broken / "z.png").write_bytes(b"not a png")
    report = qe.run(qe.load_items(broken, None, None))
    assert report["overall"]["false_rejects"] == 1 and report["overall"]["false_reject_reasons"] == {"undecodable": 1}
