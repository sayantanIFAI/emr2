"""Add a model to the registry (compliance/models.json) from its measured file hashes.

    # on the pod, after the model is downloaded (hashes are computed from the files that will really be loaded):
    python scripts/register_model.py --model-id Qwen/Qwen2.5-VL-32B-Instruct-AWQ --revision <40-hex> \
        --files files.json --role vlm --status champion --demote-champion \
        --licence Apache-2.0 --served-by "mlserve vllm backend, AWQ 4-bit"

``files.json`` is ``{name: {"sha256": ..., "bytes": ...}}``. With ``--demote-champion`` the current champion of the
role becomes a ``candidate`` (it stays registered, so going back is a settings change). The licence text must already
be in compliance/licences/ (the gate refuses a model without a saved copy of its licence). Not legal advice: counsel
signs off the licence policy.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import date
from pathlib import Path

REGISTRY = Path(__file__).resolve().parents[1] / "src" / "cdi_adapter" / "compliance" / "models.json"
LICENCE_FILES = {"Apache-2.0": "licences/Apache-2.0.txt", "MIT": "licences/MIT.txt"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", required=True)
    ap.add_argument("--revision", required=True)
    ap.add_argument("--files", required=True)
    ap.add_argument("--role", default="vlm")
    ap.add_argument("--status", default="candidate", choices=["champion", "candidate", "fallback"])
    ap.add_argument("--licence", default="Apache-2.0")
    ap.add_argument("--licence-source", default="hub model card tag 'apache-2.0'; the repo ships no LICENSE file, so licences/ holds the standard text")
    ap.add_argument("--served-by", required=True)
    ap.add_argument("--notes", default="")
    ap.add_argument("--demote-champion", action="store_true")
    ap.add_argument("--registry", default=str(REGISTRY))
    a = ap.parse_args()
    if not re.fullmatch(r"[0-9a-f]{40}", a.revision):
        sys.exit("the revision must be the exact 40-character commit")
    if a.licence not in LICENCE_FILES:
        sys.exit(f"no saved licence text for {a.licence}: add it to compliance/licences/ first")
    raw = Path(a.registry).read_text(encoding="utf-8")
    crlf = "\r\n" in raw
    reg = json.loads(raw)
    files = json.loads(Path(a.files).read_text(encoding="utf-8"))
    if a.demote_champion:
        for m in reg["models"]:
            if m["role"] == a.role and m["status"] == "champion" and m["model_id"] != a.model_id:
                m["status"] = "candidate"
    entry = {
        "role": a.role, "model_id": a.model_id, "revision": a.revision, "status": a.status,
        "weights_format": "safetensors", "trust_remote_code": False, "files": files,
        "checksum_basis": "sha256 computed on the deployed pod from the files that are loaded",
        "licence": {"name": a.licence, "spdx": a.licence, "text_file": LICENCE_FILES[a.licence],
                    "checked_on": date.today().isoformat(), "source": a.licence_source, "commercial_use": True,
                    "notice_line": None},
        "trained_on": f"as stated by the maker on the model card (not restated here): https://huggingface.co/{a.model_id}",
        "served_by": a.served_by,
        "gpu_memory_gb": {"value": None, "label": "NOT MEASURED yet"},
        "prompt_version": "computed at run time (provenance.prompt_version)", "schema_version": "v3",
        "notes": a.notes,
    }
    reg["models"] = [m for m in reg["models"] if m["model_id"] != a.model_id] + [entry]
    text = json.dumps(reg, indent=1, ensure_ascii=False) + "\n"
    Path(a.registry).write_text(text.replace("\n", "\r\n") if crlf else text, encoding="utf-8", newline="")
    print(f"registered {a.model_id} as {a.status} ({len(files)} files)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
