"""Model registry and the gate in front of every model swap (SW-S2, SW-S7).

``models.json`` says, for each role, which model is the champion, which is a candidate or the OOM fallback,
the exact revision, the checksum of every file, the licence (with a copy of its text and the date it was
checked), how it is served and what it needs. The gate:

* refuses a model id that is not registered (the champion is chosen by a setting, never by editing code);
* refuses a champion with no licence text, a licence the policy bans, or no exact 40-character revision;
* loads models only by that exact revision (never "latest"), only from safetensors, never with remote code;
* checks every file against the registered sha256 before the model is used, and refuses on a mismatch.

Not legal advice: counsel signs off the licence policy (``licence_policy.json``).
"""
from __future__ import annotations

import hashlib
import json
import re
import threading
from functools import lru_cache
from pathlib import Path
from typing import Any

from ..config import Settings, settings
from . import licences as L

HERE = Path(__file__).resolve().parent
REGISTRY_PATH = HERE / "models.json"
_REV = re.compile(r"^[0-9a-f]{40}$")


class ModelRefused(RuntimeError):
    """The registry or the licence gate refuses this model; the message names the setting."""


class ModelIntegrityError(ModelRefused):
    """A model file does not match the registered checksum."""


@lru_cache(maxsize=4)
def _load(path: str) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_registry(path: Path | None = None) -> dict[str, Any]:
    return _load(str(path or REGISTRY_PATH))


def entry(model_id: str, reg: dict[str, Any] | None = None) -> dict[str, Any] | None:
    reg = reg or load_registry()
    return next((m for m in reg["models"] if m["model_id"] == model_id), None)


def champion(role: str, reg: dict[str, Any] | None = None) -> dict[str, Any] | None:
    reg = reg or load_registry()
    return next((m for m in reg["models"] if m["role"] == role and m["status"] == "champion"), None)


def pinned_revision(model_id: str) -> str | None:
    """The exact revision to load for ``model_id`` (None = not registered, so the caller must refuse)."""
    e = entry(model_id)
    return e["revision"] if e else None


def licence_problem(m: dict[str, Any], *, root: Path | None = None, policy: dict[str, Any] | None = None) -> str | None:
    """Why this model's licence does not allow it to be used (None = fine)."""
    lic = m.get("licence") or {}
    name = lic.get("name") or ""
    if not name:
        return "no licence is recorded"
    text_file = (root or HERE) / (lic.get("text_file") or "")
    if not lic.get("text_file") or not text_file.is_file() or not text_file.read_text(encoding="utf-8", errors="replace").strip():
        return f"no copy of the licence text is saved ({lic.get('text_file') or 'no file named'})"
    if lic.get("commercial_use") is not True:
        return "the licence does not allow commercial use"
    state, _fam, why = L.classify(name, [], policy or L.load_policy())
    if state == "banned":
        return f"licence {name} is banned: {why}"
    if state != "allowed":
        return f"licence {name} is not on the allowed list ({state}): a recorded decision is required"
    return None


def check_model(model_id: str, setting_name: str, *, require_champion: bool = True, reg: dict[str, Any] | None = None,
                root: Path | None = None) -> dict[str, Any]:
    reg = reg or load_registry()
    m = entry(model_id, reg)
    if m is None:
        raise ModelRefused(f"model {model_id!r} is not registered ({setting_name}); add it to compliance/models.json "
                           "with its revision, checksums and licence first")
    if require_champion and m["status"] not in ("champion", "fallback"):
        raise ModelRefused(f"model {model_id!r} has status {m['status']!r}, not champion ({setting_name})")
    if not _REV.match(m.get("revision") or ""):
        raise ModelRefused(f"model {model_id!r} is not pinned to an exact revision ({setting_name})")
    if m.get("weights_format") != "safetensors":
        raise ModelRefused(f"model {model_id!r} is not in the safetensors format")
    if m.get("trust_remote_code"):
        raise ModelRefused(f"model {model_id!r} needs remote code, which is off unless a person has reviewed it")
    problem = licence_problem(m, root=root)
    if problem:
        raise ModelRefused(f"model {model_id!r} cannot be used ({setting_name}): {problem}")
    return m


def check_settings(s: Settings | None = None, reg: dict[str, Any] | None = None) -> list[str]:
    """Every model the settings select, checked. Returns what was checked; raises on the first refusal."""
    s = s or settings
    checked = []
    if s.mlserve_backend in ("hf", "vllm"):
        check_model(s.vlm_model_id, "CDI_VLM_MODEL_ID", reg=reg)
        checked.append(s.vlm_model_id)
        if s.vlm_fallback_model_id:
            check_model(s.vlm_fallback_model_id, "CDI_VLM_FALLBACK_MODEL_ID", reg=reg)
            checked.append(s.vlm_fallback_model_id)
    return checked


# ---------------------------------------------------------------- checksums
_hash_lock = threading.Lock()
_verified: dict[tuple[str, int, int], str] = {}


def sha256_file(path: Path, chunk: int = 1 << 24) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def verify_files(model_id: str, snapshot_dir: Path, *, reg: dict[str, Any] | None = None) -> int:
    """Check every registered file that is present in ``snapshot_dir`` against its sha256. Raises
    :class:`ModelIntegrityError` naming the file on a mismatch or a missing registered weight file. Returns the
    number of files checked. Hashes are remembered per (path, size, mtime) so a restart of one process does not
    rehash 16 GB twice."""
    e = entry(model_id, reg)
    if e is None:
        raise ModelRefused(f"model {model_id!r} is not registered")
    n = 0
    for name, meta in e["files"].items():
        want = meta.get("sha256")
        if not want:
            continue
        p = snapshot_dir / name
        if not p.exists():
            if name.endswith(".safetensors"):
                raise ModelIntegrityError(f"{model_id}: registered weight file {name} is missing")
            continue
        real = p.resolve()
        st = real.stat()
        key = (str(real), st.st_size, st.st_mtime_ns)
        with _hash_lock:
            got = _verified.get(key)
        if got is None:
            got = sha256_file(real)
            with _hash_lock:
                _verified[key] = got
        if got != want:
            raise ModelIntegrityError(f"{model_id}: {name} does not match the registered checksum "
                                      f"(registered {want[:12]}..., found {got[:12]}...)")
        n += 1
    return n


def snapshot_dir(model_id: str, revision: str) -> Path | None:
    """Where the hub cache keeps this exact revision (None = not downloaded yet)."""
    import os

    home = Path(os.environ.get("HF_HOME") or (Path.home() / ".cache" / "huggingface"))
    for base in (home / "hub", home):
        p = base / ("models--" + model_id.replace("/", "--")) / "snapshots" / revision
        if p.is_dir():
            return p
    return None


def prepare_load(model_id: str, *, setting_name: str, verify: bool | None = None) -> str:
    """The gate every model load passes: registered, licensed, pinned, and (when downloaded) checksum-verified.
    Returns the exact revision to load. A host that does not enforce the registry returns the id's pinned
    revision when it has one and ``'main'`` otherwise."""
    if not settings.model_registry_enforce:
        return pinned_revision(model_id) or "main"
    m = check_model(model_id, setting_name, require_champion=False)
    do_verify = settings.model_verify_checksums if verify is None else verify
    if do_verify:
        snap = snapshot_dir(model_id, m["revision"])
        if snap is not None:
            verify_files(model_id, snap)          # raises ModelIntegrityError on a mismatch
    return m["revision"]


def require_registered(s: Settings | None = None) -> list[str]:
    """Start-up check: every model the settings select must pass the gate. A refusal names the setting. A host
    with ``CDI_MODEL_REGISTRY_ENFORCE=false`` skips it (development only)."""
    s = s or settings
    if not s.model_registry_enforce:
        return []
    return check_settings(s)


def versions_for(model_id: str | None) -> dict[str, Any]:
    e = entry(model_id) if model_id else None
    return {"model_id": model_id, "revision": e["revision"] if e else None,
            "licence": (e or {}).get("licence", {}).get("name"), "registered": e is not None}
