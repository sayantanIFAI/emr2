import pytest

from cdi_adapter.config import settings
from cdi_adapter.webapp import jobs


def test_five_are_read_at_a_time_and_the_rest_wait_in_the_order_sent(monkeypatch):
    assert settings.job_max_concurrent == 5 and settings.job_queue_max == 10
    assert jobs._job_pool._max_workers == 5


def test_the_eleventh_is_refused_with_a_plain_message(monkeypatch):
    saved = dict(jobs._jobs)
    try:
        jobs._jobs.clear()
        for i in range(10):
            jobs._jobs[f"j{i}"] = jobs.Job(id=f"j{i}", abha=None, state="queued" if i >= 5 else "running")
        assert jobs.in_flight() == 10
        with pytest.raises(jobs.QueueFull) as e:
            jobs.create_job(None, [("a.jpg", b"x")])
        assert "10 prescriptions" in str(e.value)
        # the sixth sits behind five that are being read: nobody ahead of it except the other waiting ones
        waiting = [jobs._jobs[f"j{i}"] for i in range(5, 10)]
        for j in waiting:
            j.created = 1000.0 + int(j.id[1:])
        assert [jobs.queue_position(j) for j in waiting] == [0, 1, 2, 3, 4]
        assert jobs.queue_position(jobs._jobs["j0"]) is None
    finally:
        jobs._jobs.clear()
        jobs._jobs.update(saved)
