from __future__ import annotations

import base64
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from ..config import settings
from ..logging import get_logger

log = get_logger(__name__)
app = FastAPI(title="CDI-Adapter CPU OCR host", version="0.1.0")


class RapidReq(BaseModel):
    image_b64: str
    use_cls: bool = True        # False: no per-line "upside down" correction (the page-orientation vote needs print read as it lies)


@app.get("/healthz")
def healthz() -> dict[str, Any]:
    try:
        from ..ocr import rapid  # noqa: F401
        rapid_ok = True
    except Exception:  # noqa: BLE001
        rapid_ok = False
    return {"status": "ok", "device": "cuda" if settings.rapidocr_use_cuda else "cpu", "rapidocr": rapid_ok,
            "rapidocr_wants_cuda": settings.rapidocr_use_cuda}


@app.post("/ocr/rapid")
def ocr_rapid(req: RapidReq) -> dict[str, Any]:
    from ..ocr.rapid import run_rapidocr

    try:
        lines = run_rapidocr(base64.b64decode(req.image_b64), use_cls=req.use_cls)
    except Exception as exc:  # noqa: BLE001
        log.error("ocrhost_rapid_failed", error=str(exc)[:200])
        raise HTTPException(500, str(exc)[:300]) from exc
    return {"engine": "rapidocr", "lines": [
        {"text": ln.text, "bbox": ln.bbox, "conf": ln.conf, "polygon": ln.polygon}
        for ln in lines]}


def main() -> None:
    import uvicorn

    from ..compliance.models import require_registered

    require_registered()
    uvicorn.run(app, host="127.0.0.1", port=settings.ocrhost_port,
                log_level=settings.log_level.lower())


if __name__ == "__main__":
    main()
