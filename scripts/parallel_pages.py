"""Send every page in a folder to a running upload service AT THE SAME INSTANT and report, for each page, how long it took and what was read.

    python scripts/parallel_pages.py --base-url http://127.0.0.1:8888 --password-file /workspace/secrets/admin_password --images /workspace/loadtest/batch --out batch.json

All uploads are released together (a barrier), so N pages are in flight at once. The service holds at most ``CDI_JOB_QUEUE_MAX`` pages in flight (default 10) and accepts
``CDI_UPLOAD_RATE_PER_MINUTE`` sends a minute (default 30): for a batch larger than that, raise both on the service first or the rest are refused and retried.
Every upload gets its own token number and mobile number. A page's time runs from the moment the service ACCEPTED it to the moment its job finished.
Nothing here judges accuracy: it prints what the service returned."""
from __future__ import annotations

import argparse
import json
import mimetypes
import threading
import time
from pathlib import Path
from typing import Any

import httpx

EXTS = {".jpg", ".jpeg", ".png", ".pdf", ".tif", ".tiff"}


def val(x: Any) -> Any:
    return x.get("value") if isinstance(x, dict) else x


def one(client: httpx.Client, barrier: threading.Barrier, i: int, path: Path, run: int, t_zero: list[float], args: argparse.Namespace) -> dict[str, Any]:
    out: dict[str, Any] = {"index": i, "image": path.name, "state": "?"}
    token, phone = f"PB{run}-{i}", "9" + f"{run % 10000:04d}" + f"{i:05d}"
    data = path.read_bytes()
    mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    barrier.wait()
    if not t_zero[0]:
        t_zero[0] = time.monotonic()
    refused = 0
    while True:
        r = client.post("/api/jobs", data={"token_no": token, "phone": phone}, files={"files": (path.name, data, mime)})
        if r.status_code == 429 and refused < 60:
            refused += 1
            time.sleep(3)
            continue
        break
    out["refused"] = refused
    if r.status_code not in (200, 201, 202):
        out.update(state="rejected", error=f"HTTP {r.status_code}: {r.text[:160]}")
        return out
    job = r.json()["job_id"]
    t_acc = time.monotonic()
    out["accepted_at_s"] = round(t_acc - t_zero[0], 1)
    while True:
        d = client.get(f"/api/jobs/{job}").json()
        if d.get("state") in ("done", "error"):
            break
        if time.monotonic() - t_acc > args.job_timeout:
            d = {"state": "timeout"}
            break
        time.sleep(args.poll)
    out["state"] = d.get("state")
    out["seconds"] = round(time.monotonic() - t_acc, 1)
    out["finished_at_s"] = round(time.monotonic() - t_zero[0], 1)
    docs = d.get("documents") or []
    out["server_seconds"] = docs[0].get("seconds") if docs else None
    if d.get("state") == "error":
        out["error"] = str(d.get("error") or "")[:200]
    if d.get("state") == "done":
        res = (client.get(f"/api/jobs/{job}/result.json").json().get("results") or [{}])[0]
        pat, doc = res.get("patient") or {}, res.get("doctor") or {}
        out.update({
            "job_id": job, "doc_type": res.get("doc_type"), "handwritten": res.get("is_handwritten"),
            "patient_name": val(pat.get("name")), "age": val(pat.get("age_text")), "sex": val(pat.get("sex")),
            "doctor": val(doc.get("name")), "department": val(doc.get("department")),
            "tests": [{"as_written": t.get("as_written"), "standard": t.get("standard_name"), "status": t.get("status")} for t in res.get("lab_tests") or []],
            "n_medications": len(res.get("medications") or []), "status": res.get("status"), "flags": res.get("flags"),
        })
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--user", default="admin")
    ap.add_argument("--password-file", required=True)
    ap.add_argument("--images", required=True)
    ap.add_argument("--poll", type=float, default=1.0)
    ap.add_argument("--job-timeout", type=float, default=900.0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    files = sorted(p for p in Path(args.images).iterdir() if p.suffix.lower() in EXTS)
    run = int(time.time()) % 10000
    pw = Path(args.password_file).read_text(encoding="utf-8").strip()
    barrier = threading.Barrier(len(files))
    t_zero = [0.0]
    results: list[dict[str, Any]] = [{} for _ in files]
    with httpx.Client(base_url=args.base_url, auth=(args.user, pw), timeout=120.0, trust_env=False) as client:
        print("healthz:", client.get("/healthz").text[:120])

        def work(i: int) -> None:
            try:
                results[i] = one(client, barrier, i, files[i], run, t_zero, args)
            except Exception as exc:  # noqa: BLE001 - one failed page must not stop the batch
                results[i] = {"index": i, "image": files[i].name, "state": "rejected", "error": f"{type(exc).__name__}: {exc}"[:200]}

        threads = [threading.Thread(target=work, args=(i,)) for i in range(len(files))]
        t0 = time.monotonic()
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        wall = time.monotonic() - t0
    done = [r for r in results if r.get("state") == "done"]
    secs = sorted(r["seconds"] for r in done)
    print(f"{len(files)} pages released together: {len(done)} done, {len(files) - len(done)} not done; the whole batch took {wall:.1f} s")
    if secs:
        print(f"per page: fastest {secs[0]} s, median {secs[len(secs) // 2]} s, slowest {secs[-1]} s")
    for r in results:
        print(f"  {r['image']}: {r['state']} {r.get('seconds')} s  name={r.get('patient_name')!r}  tests={[t['as_written'] for t in r.get('tests', [])]}")
    Path(args.out).write_text(json.dumps({"wall_seconds": round(wall, 1), "pages": results}, indent=1), encoding="utf-8")
    print("written", args.out)
    return 0 if len(done) == len(files) else 1


if __name__ == "__main__":
    raise SystemExit(main())
