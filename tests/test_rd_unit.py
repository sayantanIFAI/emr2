"""Reading engines working together (RD-S3 self-consistency and drift alarm). No database, no GPU."""
from __future__ import annotations

from typing import ClassVar

import cv2
import numpy as np
import pytest

from cdi_adapter.config import settings
from cdi_adapter.recognition import drift, pipeline
from cdi_adapter.recognition.disagreement import (
    AGREE,
    DISAGREE,
    SINGLE,
    compare_engines,
    self_consistency,
)
from cdi_adapter.recognition.engines import Reading


def R(engine, text, conf=None, error=None):
    return Reading(engine, "v1", text, conf, error=error)


# ------------------------------------------------------------------ self-consistency (RD-S3 FR6, AC5)


def test_ac5_two_reads_of_the_same_crop_that_differ_send_an_agreeing_line_to_review():
    ra, q1, q2 = R("reader-a", "Telma 40"), R("qwen2.5-vl", "Telma 40"), R("qwen2.5-vl", "Telma 80")
    base = compare_engines([ra, q1])
    assert base.state == AGREE
    out = self_consistency(base, q1, q2)
    assert out.state == DISAGREE and out.display_text == "Telma 40"                 # the readings are untouched
    sc = out.detail["self_consistency"]
    assert sc["consistent"] is False and sc["texts"] == ["Telma 40", "Telma 80"] and sc["numbers_equal"] is False


def test_two_consistent_reads_change_nothing_but_record_the_check():
    ra, q1, q2 = R("reader-a", "Telma 40"), R("qwen2.5-vl", "Telma 40"), R("qwen2.5-vl", "Telma 4O")
    out = self_consistency(compare_engines([ra, q1]), q1, q2)
    assert out.state == AGREE and out.detail["self_consistency"]["consistent"] is True    # O vs 0 is the same number


def test_a_failed_or_empty_second_read_changes_nothing():
    base = compare_engines([R("reader-a", "Telma 40"), R("qwen2.5-vl", "Telma 40")])
    assert self_consistency(base, R("qwen2.5-vl", "Telma 40"), R("qwen2.5-vl", "", error="timeout")) is base
    assert self_consistency(base, R("qwen2.5-vl", "Telma 40"), R("qwen2.5-vl", "   ")) is base


def test_a_line_that_already_needs_a_person_is_not_changed_by_the_check():
    single = compare_engines([R("reader-a", "", error="host down"), R("qwen2.5-vl", "Telma 40")])
    assert single.state == SINGLE
    out = self_consistency(single, R("qwen2.5-vl", "Telma 40"), R("qwen2.5-vl", "Telma 90"))
    assert out.state == SINGLE and out.detail["self_consistency"]["consistent"] is False
    disagree = compare_engines([R("reader-a", "Telma 40"), R("qwen2.5-vl", "Telmikind 40")])
    assert self_consistency(disagree, R("qwen2.5-vl", "a"), R("qwen2.5-vl", "b")).state == DISAGREE


# ------------------------------------------------------------------ drift alarm (RD-S3 FR7)


def test_disagreement_rate_counts_only_handwritten_lines():
    assert drift.disagreement_rate({"printed": 40, "agree": 6, "disagree": 2, "single_engine": 1, "no_reading": 1}) == 0.2
    assert drift.disagreement_rate({"printed": 12}) is None and drift.disagreement_rate({}) is None


def test_no_alarm_until_there_is_a_usual_rate():
    out = drift.drift_alarm([0.1] * 5, 0.9, min_runs=20)
    assert out["alarm"] is False and "only 5 earlier runs" in out["reason"]


def test_a_sudden_jump_raises_the_alarm_and_a_normal_wobble_does_not():
    history = [0.10, 0.12, 0.08, 0.11, 0.09] * 6                        # 30 runs around 10 %
    up = drift.drift_alarm(history, 0.55, min_runs=20)
    assert up["alarm"] is True and up["direction"] == "up" and up["runs"] == 30
    assert drift.drift_alarm(history, 0.13, min_runs=20)["alarm"] is False
    down = drift.drift_alarm([0.5] * 30, 0.1, min_runs=20)
    assert down["alarm"] is True and down["direction"] == "down"


def test_a_perfectly_steady_history_still_needs_a_real_move_to_alarm():
    steady = [0.1] * 30                                                  # spread 0: the absolute tolerance rules
    assert drift.drift_alarm(steady, 0.12, min_runs=20, abs_tol=0.15)["alarm"] is False
    assert drift.drift_alarm(steady, 0.30, min_runs=20, abs_tol=0.15)["alarm"] is True


def test_the_limits_are_settings(monkeypatch):
    monkeypatch.setattr(settings, "drift_min_runs", 3)
    monkeypatch.setattr(settings, "drift_abs_tolerance", 0.01)
    assert drift.drift_alarm([0.1, 0.1, 0.1], 0.2)["alarm"] is True


def test_recent_rates_reads_the_stored_states_of_earlier_runs():
    class Rows:
        def all(self):
            return [({"agree": 8, "disagree": 2},), ({"printed": 9},), ({"agree": 1, "disagree": 1},), (None,)]

    class Sess:
        def execute(self, stmt, params):
            assert params["rid"] == "run-now" and "recognition-v2" in str(stmt)
            return Rows()

    assert drift.recent_rates(Sess(), "run-now") == [0.2, 0.5]


# ------------------------------------------------------------------ the pipeline uses it, and only when asked


def _handwritten_page() -> bytes:
    img = np.full((600, 1000, 3), 255, np.uint8)
    pts = np.array([[80 + i * 12, 300 + int(25 * np.sin(i / 2.0))] for i in range(70)], np.int32)
    cv2.polylines(img, [pts], False, (0, 0, 0), 6)
    ok, enc = cv2.imencode(".png", img)
    assert ok
    return enc.tobytes()


class _Qwen:
    calls: ClassVar[list[int]] = []
    scripts: ClassVar[list[str]] = ["Telma 40", "Telma 80"]

    def recognize(self, crops):
        k = len(_Qwen.calls)
        _Qwen.calls.append(len(crops))
        return [R("qwen2.5-vl", _Qwen.scripts[min(k, len(_Qwen.scripts) - 1)]) for _ in crops]


def _run(monkeypatch, self_consistency_on: bool):
    _Qwen.calls = []
    monkeypatch.setattr(pipeline.storage, "get_bytes", lambda key: _handwritten_page())
    monkeypatch.setattr(pipeline, "QwenLineEngine", _Qwen)
    monkeypatch.setattr(settings, "qwen_self_consistency", self_consistency_on)
    monkeypatch.setattr(settings, "qwen_adjudication_enabled", False)
    page = {"id": "pg1", "page_no": 1, "image_uri": "s3://b/x.png", "preproc": {"src_uri": "s3://b/x.png"}}
    return pipeline.recognize_page(document_id="d1", page=page, rapid_lines=[], run_id="r1", lang="en")


def test_with_the_check_off_there_is_one_qwen_read_per_crop(monkeypatch):
    blocks, obs = _run(monkeypatch, False)
    assert _Qwen.calls and len(_Qwen.calls) == 1
    assert {o["engine"] for o in obs} == {"qwen2.5-vl"}
    assert blocks[0]["recognition"]["state"] == SINGLE                    # one reader: a person looks at every line


def test_with_the_check_on_an_unstable_line_is_recorded_and_the_second_read_is_kept_as_evidence(monkeypatch):
    blocks, obs = _run(monkeypatch, True)
    assert len(_Qwen.calls) == 2                                          # a second read, different padding
    rec = blocks[0]["recognition"]
    assert rec["state"] == SINGLE and rec["detail"]["self_consistency"]["consistent"] is False
    by_engine = {o["engine"]: o for o in obs}
    assert set(by_engine) == {"qwen2.5-vl", "qwen2.5-vl-b"}
    assert by_engine["qwen2.5-vl"]["raw_text"] == "Telma 40" and by_engine["qwen2.5-vl-b"]["raw_text"] == "Telma 80"
    assert by_engine["qwen2.5-vl-b"]["crop_hash"] != by_engine["qwen2.5-vl"]["crop_hash"]   # a different crop


def test_with_the_check_on_a_stable_line_is_unchanged(monkeypatch):
    monkeypatch.setattr(_Qwen, "scripts", ["Telma 40", "Telma 40"])
    blocks, _obs = _run(monkeypatch, True)
    assert blocks[0]["recognition"]["state"] == SINGLE
    assert blocks[0]["recognition"]["detail"]["self_consistency"]["consistent"] is True
    monkeypatch.setattr(_Qwen, "scripts", ["Telma 40", "Telma 80"])


def test_the_crop_standard_is_recorded_on_every_handwritten_line(monkeypatch):
    blocks, _obs = _run(monkeypatch, False)
    crop = blocks[0]["recognition"]["crop"]
    assert set(crop) == {"bbox", "pad", "cut_wh", "scale", "out_wh", "upscaled", "below_standard"}


@pytest.mark.parametrize("states,expected", [({"disagree": 1, "agree": 3}, 0.25)])
def test_the_rate_is_what_the_run_stores(states, expected):
    assert drift.disagreement_rate(states) == expected
