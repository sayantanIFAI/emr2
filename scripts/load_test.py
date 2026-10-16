"""Load test for a running upload service (the admin screen's API).

Uploads real pages with a fixed number in flight, waits for each job to finish and reports how long a page takes (median / 95th / slowest), how many
pages a minute the service finishes, how often it refused a send, and whether the lab tests found for the SAME page change when the load changes.

    python scripts/load_test.py --base-url http://127.0.0.1:8888 --password-file /workspace/secrets/admin_password \\
        --images /workspace/loadtest/images --steps 1,3,5,8 --per-step 20 --out /workspace/loadtest/run1.json

* Every upload gets its own token number and mobile number, so no two uploads are the same document (the service tells documents apart by the hash of
  the bytes plus phone and token) and a repeated page is read again, not short-circuited.
* A page's time is counted from the moment the service ACCEPTED the upload to the moment its job is done or failed. A refusal (HTTP 429: too many pages in
  flight, CDI_JOB_QUEUE_MAX, or too many sends a minute, CDI_UPLOAD_RATE_PER_MINUTE) is counted and retried after ``--refused-wait`` seconds; it is not
  part of the page's time. Set CDI_UPLOAD_RATE_PER_MINUTE=0 on the service for a capacity test, or the limit is what is measured.
* The first ``--warmup`` uploads are not counted.
* Nothing here is a clinical measure: it measures time and throughput, and lists any change in the tests found (``--no-tests`` skips that).
"""
from __future__ import annotations

import argparse
import itertools
import json
import mimetypes
import os
import sys
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import httpx

EXTS = {".jpg", ".jpeg", ".png", ".pdf", ".tif", ".tiff"}


def percentile(values: list[float], q: float) -> float | None:
    """The q-th percentile (0-100) by linear interpolation; None for no values."""
    if not values:
        return None
    s = sorted(values)
    k = (len(s) - 1) * q / 100.0
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def identity(run: int, n: int) -> tuple[str, str]:
    """A token number and a 10-digit mobile number that no other upload of this run (or of another run) shares."""
    return f"LT{run % 100000}-{n}", "9" + f"{run % 10000:04d}" + f"{n % 100000:05d}"


@dataclass
class Outcome:
    index: int
    image: str
    state: str = "?"                  # done | error | timeout | refused | rejected
    seconds: float | None = None      # accepted -> job finished
    server_seconds: float | None = None
    refused: int = 0                  # how many times the send was refused (429) before it was accepted
    error: str = ""
    tests: list[str] = field(default_factory=list)


def submit(client: httpx.Client, n: int, name: str, data: bytes, run: int, args: Any) -> tuple[httpx.Response | None, int]:
    token, phone = identity(run, n)
    form = {"token_no": token, "phone": phone}
    if args.department:
        form["department"] = args.department
    mime = mimetypes.guess_type(name)[0] or "application/octet-stream"
    refused = 0
    while True:
        r = client.post("/api/jobs", data=form, files={"files": (name, data, mime)})
        if r.status_code == 429 and refused < args.max_refusals:
            refused += 1
            time.sleep(args.refused_wait)
            continue
        return r, refused


def wait_done(client: httpx.Client, job_id: str, poll_s: float, timeout_s: float) -> dict[str, Any]:
    t0 = time.monotonic()
    while True:
        try:
            r = client.get(f"/api/jobs/{job_id}")
            if r.status_code == 200:
                d = r.json()
                if d.get("state") in ("done", "error"):
                    return d
        except (httpx.HTTPError, ValueError):
            pass                                        # one lost poll is not a failed page
        if time.monotonic() - t0 > timeout_s:
            return {"state": "timeout"}
        time.sleep(poll_s)


def lab_tests(client: httpx.Client, job_id: str) -> list[str]:
    """The names of the lab tests the job found (a rejected one is not a test), sorted, for comparing the same page across steps."""
    try:
        r = client.get(f"/api/jobs/{job_id}/result.json")
        results = r.json().get("results") or []
    except (httpx.HTTPError, ValueError):
        return []
    names = {str(t.get("standard_name") or t.get("as_written") or "").strip().lower()
             for res in results for t in res.get("lab_tests") or [] if t.get("status") != "rejected"}
    return sorted(n for n in names if n)


def one(client: httpx.Client, index: int, name: str, data: bytes, run: int, args: Any) -> Outcome:
    out = Outcome(index=index, image=name)
    try:
        r, out.refused = submit(client, index, name, data, run, args)
        if r is None or r.status_code not in (200, 201, 202):
            out.state = "refused" if r is not None and r.status_code == 429 else "rejected"
            out.error = f"HTTP {getattr(r, 'status_code', '?')}: {(getattr(r, 'text', '') or '')[:160]}"
            return out
        job_id = r.json()["job_id"]
        t0 = time.monotonic()
        d = wait_done(client, job_id, args.poll, args.job_timeout)
        out.seconds = time.monotonic() - t0
        out.state = d.get("state", "?")
        docs = d.get("documents") or []
        out.server_seconds = docs[0].get("seconds") if docs else None
        if out.state == "error":
            out.error = str(d.get("error") or "")[:160]
        if out.state == "done" and not args.no_tests:
            out.tests = lab_tests(client, job_id)
    except Exception as exc:  # noqa: BLE001 - one failed page must not stop the run
        out.state, out.error = "rejected", f"{type(exc).__name__}: {exc}"[:160]
    return out


def run_step(client: httpx.Client, images: list[tuple[str, bytes]], concurrency: int, total: int, run: int, start: int, args: Any) -> tuple[list[Outcome], float]:
    """``total`` uploads with ``concurrency`` in flight; returns the outcomes and the step's wall-clock seconds."""
    counter = itertools.count(start)
    end = start + total
    lock = threading.Lock()
    outcomes: list[Outcome] = []

    def worker() -> None:
        while True:
            with lock:
                i = next(counter)
            if i >= end:
                return
            name, data = images[i % len(images)]
            o = one(client, i, name, data, run, args)
            with lock:
                outcomes.append(o)

    t0 = time.monotonic()
    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        for f in [ex.submit(worker) for _ in range(concurrency)]:
            f.result()
    return outcomes, time.monotonic() - t0


def summarise(concurrency: int, outcomes: list[Outcome], wall: float) -> dict[str, Any]:
    done = [o for o in outcomes if o.state == "done"]
    lat = [o.seconds for o in done if o.seconds is not None]
    thr = len(done) / wall * 60.0 if wall > 0 else 0.0
    mean = sum(lat) / len(lat) if lat else None
    return {
        "in_flight": concurrency, "uploads": len(outcomes), "done": len(done),
        "failed": sum(1 for o in outcomes if o.state in ("error", "timeout", "rejected", "refused")),
        "refused_sends": sum(o.refused for o in outcomes),
        "seconds_median": percentile(lat, 50), "seconds_p90": percentile(lat, 90), "seconds_p95": percentile(lat, 95),
        "seconds_p99": percentile(lat, 99), "seconds_max": max(lat) if lat else None, "seconds_mean": mean,
        "pages_per_minute": round(thr, 2), "pages_per_hour": round(thr * 60, 1),
        "average_in_flight_check": round(thr / 60.0 * mean, 2) if mean else None,       # Little's law: should be close to in_flight
        "wall_seconds": round(wall, 1),
    }


def test_drift(all_outcomes: list[Outcome]) -> dict[str, list[list[str]]]:
    """Pages whose list of lab tests was not the same every time they were read: ``{image: [the different lists]}``."""
    seen: dict[str, set[tuple[str, ...]]] = defaultdict(set)
    for o in all_outcomes:
        if o.state == "done":
            seen[o.image].add(tuple(o.tests))
    return {img: [list(t) for t in sorted(v)] for img, v in seen.items() if len(v) > 1}


def load_images(folder: Path) -> list[tuple[str, bytes]]:
    files = sorted(p for p in folder.iterdir() if p.suffix.lower() in EXTS)
    if not files:
        raise SystemExit(f"no images (jpg, png, pdf, tif) in {folder}")
    return [(p.name, p.read_bytes()) for p in files]


def fmt(v: Any, width: int = 7) -> str:
    return f"{v:{width}.1f}" if isinstance(v, (int, float)) else f"{'-':>{width}}"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-url", required=True, help="e.g. http://127.0.0.1:8888 (on the pod) or http://127.0.0.1:8080 (laptop)")
    ap.add_argument("--user", default="admin")
    ap.add_argument("--password", default=os.environ.get("LOAD_TEST_PASSWORD", ""), help="or env LOAD_TEST_PASSWORD, or --password-file")
    ap.add_argument("--password-file", help="a file holding the admin password (keeps it out of the shell history)")
    ap.add_argument("--images", required=True, help="a folder of pages; they are used in turn")
    ap.add_argument("--steps", default="1,3,5,8", help="pages in flight, one step per value")
    ap.add_argument("--per-step", type=int, default=20, help="uploads in each step")
    ap.add_argument("--warmup", type=int, default=3, help="uploads before the first step that are not counted")
    ap.add_argument("--poll", type=float, default=1.0, help="seconds between status checks")
    ap.add_argument("--job-timeout", type=float, default=900.0, help="seconds before a page counts as timed out")
    ap.add_argument("--timeout", type=float, default=120.0, help="seconds for one HTTP call")
    ap.add_argument("--refused-wait", type=float, default=3.0)
    ap.add_argument("--max-refusals", type=int, default=60)
    ap.add_argument("--department", default="")
    ap.add_argument("--no-tests", action="store_true", help="do not fetch the result of each page")
    ap.add_argument("--run-id", type=int, default=int(time.time()) % 10000, help="makes the token and phone numbers of this run unique")
    ap.add_argument("--out", help="write every outcome and the summary here (JSON)")
    args = ap.parse_args(argv)

    password = Path(args.password_file).read_text(encoding="utf-8").strip() if args.password_file else args.password
    images = load_images(Path(args.images))
    steps = [int(x) for x in args.steps.split(",") if x.strip()]
    print(f"{len(images)} page(s), steps {steps}, {args.per_step} uploads each, run id {args.run_id}")
    auth = (args.user, password) if password else None
    summary: list[dict[str, Any]] = []
    everything: list[Outcome] = []
    with httpx.Client(base_url=args.base_url, auth=auth, timeout=args.timeout, trust_env=False) as client:
        h = client.get("/healthz")
        print("healthz:", h.status_code, h.text[:160])
        index = 0
        if args.warmup:
            print(f"warm-up: {args.warmup} upload(s), not counted")
            run_step(client, images, 1, args.warmup, args.run_id, index, args)
            index += args.warmup
        for c in steps:
            outs, wall = run_step(client, images, c, args.per_step, args.run_id, index, args)
            index += args.per_step
            everything += outs
            s = summarise(c, outs, wall)
            summary.append(s)
            print(f"in flight {c}: done {s['done']}/{s['uploads']}  median {fmt(s['seconds_median'])}s  p95 {fmt(s['seconds_p95'])}s  "
                  f"slowest {fmt(s['seconds_max'])}s  {s['pages_per_minute']} pages/min  refused sends {s['refused_sends']}")
    print("\nin flight |  done | failed | median s |   p95 s |   max s | pages/min | pages/hour | check (avg in flight)")
    for s in summary:
        print(f"{s['in_flight']:9d} | {s['done']:5d} | {s['failed']:6d} | {fmt(s['seconds_median'], 8)} | {fmt(s['seconds_p95'], 7)} | {fmt(s['seconds_max'], 7)} | "
              f"{s['pages_per_minute']:9.2f} | {s['pages_per_hour']:10.1f} | {s['average_in_flight_check']}")
    drift = test_drift(everything) if not args.no_tests else {}
    if not args.no_tests:
        print(f"\nPages whose lab tests changed between reads: {len(drift)} of {len({o.image for o in everything})}")
        for img, variants in drift.items():
            print(f"  {img}: {variants}")
    failures = [o for o in everything if o.state != "done"]
    for o in failures[:10]:
        print(f"  FAILED #{o.index} {o.image}: {o.state} {o.error}")
    if args.out:
        Path(args.out).write_text(json.dumps({"args": {k: v for k, v in vars(args).items() if k != "password"}, "summary": summary, "test_drift": drift,
                                              "outcomes": [asdict(o) for o in everything]}, indent=1), encoding="utf-8")
        print("written", args.out)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
