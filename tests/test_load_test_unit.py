"""The load-test tool (scripts/load_test.py) against a pretend upload service: no network, no model."""
import importlib.util
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

spec = importlib.util.spec_from_file_location("load_test", Path(__file__).resolve().parents[1] / "scripts" / "load_test.py")
LT = importlib.util.module_from_spec(spec)
sys.modules["load_test"] = LT
spec.loader.exec_module(LT)


def fake_service(refuse_every: int = 0, fail_index: int = -1, tests_for=lambda token: ["tsh", "cbc"]):
    """A service that accepts an upload (refusing every Nth first try with 429), says 'running' once, then 'done'."""
    state = {"tries": 0, "polls": {}}
    lock = threading.Lock()

    def handler(req: httpx.Request) -> httpx.Response:
        path = req.url.path
        if req.method == "POST" and path == "/api/jobs":
            body = req.read().decode("latin-1")
            token = next(t for t in body.split() if t.startswith("LT") and "-" in t).strip('"')
            with lock:
                state["tries"] += 1
                if refuse_every and state["tries"] % refuse_every == 0:
                    return httpx.Response(429, json={"detail": "10 prescriptions are already being read or waiting."})
            return httpx.Response(202, json={"job_id": token, "documents": 1})
        if req.method == "GET" and path.startswith("/api/jobs/") and path.endswith("/result.json"):
            token = path.split("/")[3]
            return httpx.Response(200, json={"results": [{"lab_tests": [{"standard_name": t, "status": "needs_check"} for t in tests_for(token)]
                                                          + [{"standard_name": "junk", "status": "rejected"}]}]})
        if req.method == "GET" and path.startswith("/api/jobs/"):
            job = path.split("/")[3]
            with lock:
                state["polls"][job] = state["polls"].get(job, 0) + 1
                n = state["polls"][job]
            if job.endswith(f"-{fail_index}"):
                return httpx.Response(200, json={"state": "error", "error": "page too small", "documents": [{"seconds": None}]})
            return httpx.Response(200, json={"state": "running" if n < 2 else "done", "documents": [{"seconds": 1.5}]})
        return httpx.Response(404)

    return httpx.Client(base_url="http://test", transport=httpx.MockTransport(handler)), state


ARGS = SimpleNamespace(department="", refused_wait=0.0, max_refusals=5, poll=0.0, job_timeout=5.0, no_tests=False)
IMAGES = [("a.jpg", b"aaa"), ("b.jpg", b"bbb")]


def test_percentiles():
    assert LT.percentile([], 50) is None
    assert LT.percentile([5.0], 95) == 5.0
    assert LT.percentile([1, 2, 3, 4, 5], 50) == 3
    assert LT.percentile([10, 20], 50) == 15
    assert LT.percentile(list(range(1, 101)), 95) == pytest.approx(95.05)


def test_identities_are_valid_and_unique():
    seen = set()
    for run in (7, 1234, 9999):
        for n in (0, 1, 99999):
            token, phone = LT.identity(run, n)
            assert len(token) <= 20 and len(phone) == 10 and phone.isdigit() and phone[0] in "6789"
            seen.add((token, phone))
    assert len(seen) == 9


def test_a_step_finishes_every_upload_and_counts_them():
    client, _ = fake_service()
    outs, wall = LT.run_step(client, IMAGES, 3, 9, run=1, start=0, args=ARGS)
    assert len(outs) == 9 and all(o.state == "done" for o in outs)
    assert {o.image for o in outs} == {"a.jpg", "b.jpg"}
    assert all(o.tests == ["cbc", "tsh"] for o in outs)                  # the rejected one is not a test
    s = LT.summarise(3, outs, wall)
    assert s["done"] == 9 and s["failed"] == 0 and s["pages_per_minute"] > 0 and s["seconds_p95"] >= s["seconds_median"]


def test_a_refused_send_is_retried_and_counted():
    client, state = fake_service(refuse_every=2)
    outs, _ = LT.run_step(client, IMAGES, 1, 4, run=1, start=0, args=ARGS)
    assert all(o.state == "done" for o in outs) and sum(o.refused for o in outs) >= 1
    assert state["tries"] > 4


def test_a_failed_page_is_reported_not_hidden():
    client, _ = fake_service(fail_index=2)
    outs, wall = LT.run_step(client, IMAGES, 2, 4, run=1, start=0, args=ARGS)
    bad = [o for o in outs if o.state != "done"]
    assert len(bad) == 1 and bad[0].state == "error" and "too small" in bad[0].error
    assert LT.summarise(2, outs, wall)["failed"] == 1


def test_a_page_whose_tests_change_is_flagged():
    flip = {"n": 0}

    def tests_for(token):
        flip["n"] += 1
        return ["tsh", "cbc"] if flip["n"] % 2 else ["tsh"]

    client, _ = fake_service(tests_for=tests_for)
    outs, _ = LT.run_step(client, [("a.jpg", b"x")], 1, 4, run=1, start=0, args=ARGS)
    drift = LT.test_drift(outs)
    assert list(drift) == ["a.jpg"] and len(drift["a.jpg"]) == 2
    assert LT.test_drift([o for o in outs if o.tests == ["cbc", "tsh"]]) == {}


def test_main_writes_a_report(tmp_path, monkeypatch):
    (tmp_path / "imgs").mkdir()
    (tmp_path / "imgs" / "p.jpg").write_bytes(b"jpg")
    client, _ = fake_service()
    monkeypatch.setattr(LT.httpx, "Client", lambda **kw: client)
    out = tmp_path / "run.json"
    code = LT.main(["--base-url", "http://test", "--images", str(tmp_path / "imgs"), "--steps", "1,2", "--per-step", "2", "--warmup", "1",
                    "--poll", "0", "--password", "secret-not-saved", "--out", str(out)])
    assert code == 0
    text = out.read_text(encoding="utf-8")
    assert '"summary"' in text and "secret-not-saved" not in text
