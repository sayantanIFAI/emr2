"""One interface to the CPU OCR engines: over HTTP (``CDI_OCRHOST_URL`` set) or in-process."""
from __future__ import annotations

import base64
from typing import Protocol

import httpx

from ..config import settings
from ..ocr.rapid import OcrLine


class OcrHost(Protocol):
    def rapid(self, png: bytes, use_cls: bool = True) -> list[OcrLine]: ...


class LocalOcrHost:
    def rapid(self, png: bytes, use_cls: bool = True) -> list[OcrLine]:
        from ..ocr.rapid import run_rapidocr

        return run_rapidocr(png, use_cls=use_cls)


class HttpOcrHost:
    def __init__(self, base_url: str) -> None:
        self._c = httpx.Client(base_url=base_url.rstrip("/"), timeout=settings.ocrhost_timeout_s)

    def rapid(self, png: bytes, use_cls: bool = True) -> list[OcrLine]:
        r = self._c.post("/ocr/rapid", json={"image_b64": base64.b64encode(png).decode(), "use_cls": use_cls})
        r.raise_for_status()
        return [OcrLine(text=x["text"], bbox=list(x["bbox"]), conf=float(x["conf"]),
                        polygon=x.get("polygon") or []) for x in r.json()["lines"]]


_host: OcrHost | None = None


def get_ocr_host() -> OcrHost:
    global _host
    if _host is None:
        _host = HttpOcrHost(settings.ocrhost_url) if settings.ocrhost_url else LocalOcrHost()
    return _host


def set_ocr_host(host: OcrHost | None) -> None:
    """Test hook."""
    global _host
    _host = host
