"""Operational numbers in the Prometheus text format (ENT-S4): no client library, no bundled dashboard.

``render(sess)`` returns the exposition text for ``GET /metrics``. Every number comes from the database or from
a live probe at the moment it is asked for; nothing is cached or invented. A probe that fails reports ``0``
(down) rather than disappearing, so an alert can fire on it.
"""
from __future__ import annotations

import os
import time
from typing import Any

from sqlalchemy import text

from ..config import settings


def _esc(v: Any) -> str:
    return str(v).replace("\\", "\\\\").replace('"', '\\"').replace("\n", " ")


class _Out:
    def __init__(self) -> None:
        self.lines: list[str] = []
        self._seen: set[str] = set()

    def metric(self, name: str, value: float | int | None, help_: str, labels: dict[str, Any] | None = None,
               kind: str = "gauge") -> None:
        if value is None:
            return
        if name not in self._seen:
            self._seen.add(name)
            self.lines += [f"# HELP {name} {help_}", f"# TYPE {name} {kind}"]
        lab = "{" + ",".join(f'{k}="{_esc(v)}"' for k, v in (labels or {}).items()) + "}" if labels else ""
        self.lines.append(f"{name}{lab} {value}")

    def text(self) -> str:
        return "\n".join(self.lines) + "\n"


def _probe(fn: Any) -> int:
    try:
        return 1 if fn() else 0
    except Exception:  # noqa: BLE001
        return 0


def render(sess: Any) -> str:
    from .. import storage, swap
    from ..db import ping as db_ping

    o = _Out()
    o.metric("cdi_up", 1, "The web app is answering")
    o.metric("cdi_db_up", _probe(db_ping), "PostgreSQL answers")
    o.metric("cdi_object_store_up", _probe(storage.ping), "The object store answers with our credentials")
    o.metric("cdi_queue_up", _probe(lambda: swap.resolve("job_queue").ping()), "The job queue (Redis / Valkey) answers")

    def gateway() -> bool:
        from ..ml.client import get_client

        return get_client().healthz().get("status") == "ok"

    o.metric("cdi_model_gateway_up", _probe(gateway), "The model gateway answers")

    def ocrhost() -> bool:
        import httpx

        return bool(settings.ocrhost_url) and httpx.get(settings.ocrhost_url.rstrip("/") + "/healthz", timeout=3).status_code == 200

    o.metric("cdi_ocr_host_up", _probe(ocrhost), "The OCR host answers")

    for status, n in sess.execute(text("SELECT status, count(*) FROM source_document GROUP BY status")).all():
        o.metric("cdi_documents", n, "Documents by status", {"status": status})
    o.metric("cdi_documents_unfinished_oldest_seconds", float(sess.execute(text(
        "SELECT coalesce(extract(epoch FROM now() - min(ingested_at)), 0) FROM source_document "
        "WHERE status NOT IN ('validated','projected','normalized','error','quality_hold')")).scalar_one()),
        "Age of the oldest document that has not reached a final state (stuck work)")
    o.metric("cdi_documents_last_hour", sess.execute(text(
        "SELECT count(*) FROM source_document WHERE ingested_at > now() - interval '1 hour'")).scalar_one(),
        "Documents received in the last hour")
    for stage, n, mean in sess.execute(text(
            "SELECT stage, count(*), avg(extract(epoch FROM ended_at - started_at)) FROM pipeline_run "
            "WHERE status = 'ok' AND ended_at IS NOT NULL AND started_at > now() - interval '1 day' GROUP BY stage")).all():
        o.metric("cdi_stage_runs_24h", n, "Successful stage runs in the last 24 hours", {"stage": stage})
        o.metric("cdi_stage_seconds_mean_24h", round(float(mean or 0), 3), "Mean seconds per successful run, last 24 hours",
                 {"stage": stage})
    o.metric("cdi_stage_failures_24h", sess.execute(text(
        "SELECT count(*) FROM pipeline_run WHERE status = 'failed' AND started_at > now() - interval '1 day'")).scalar_one(),
        "Failed stage runs in the last 24 hours")

    # listener (LS-S9)
    try:
        from ..listener.health import health

        h = health(sess)
        for state in ("waiting", "processing", "in_error", "in_quarantine", "completed", "ignored"):
            o.metric("cdi_listener_files", h[f"files_{state}"], "Listener files by state", {"state": state})
        o.metric("cdi_listener_seconds_since_good_poll", h["seconds_since_last_good_poll"],
                 "Seconds since the listener last polled the drive successfully")
        o.metric("cdi_listener_oldest_waiting_seconds", h["oldest_waiting_age_seconds"],
                 "Age of the oldest file waiting to be read")
        o.metric("cdi_listener_stalled", 1 if h["stalled"] else 0, "1 when the listener has not polled well within the stall limit or has stopped")
    except Exception:  # noqa: BLE001 - a database without the listener tables
        pass

    # drift (RD-S3): the share of handwriting lines on which the two readers disagree, last 50 runs
    try:
        from ..recognition.drift import recent_rates

        rates = recent_rates(sess, "00000000-0000-0000-0000-000000000000", limit=50)
        if rates:
            o.metric("cdi_reader_disagreement_rate", round(sum(rates) / len(rates), 4),
                     "Mean share of handwriting lines on which the two reads of the same line disagree (last 50 recognition runs)")
    except Exception:  # noqa: BLE001
        pass

    # backups (a restart rehearsal is only as good as the last dump)
    dump = os.environ.get("CDI_BACKUP_PATH", "/workspace/backup/cdi.dump")
    if os.path.exists(dump):
        o.metric("cdi_backup_age_seconds", round(time.time() - os.path.getmtime(dump)), "Age of the latest database dump")
    return o.text()
