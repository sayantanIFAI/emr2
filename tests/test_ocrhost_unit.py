"""CPU OCR host HTTP contract (RapidOCR, printed text)."""
from __future__ import annotations

import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image, ImageDraw, ImageFont

from cdi_adapter.ocrhost.app import app
from cdi_adapter.recognition.ocrhost_client import HttpOcrHost


def _png(text: str) -> bytes:
    im = Image.new("RGB", (900, 120), "white")
    try:
        f = ImageFont.truetype("arial.ttf", 44)
    except OSError:
        f = ImageFont.load_default()
    ImageDraw.Draw(im).text((20, 30), text, fill="black", font=f)
    b = io.BytesIO()
    im.save(b, "PNG")
    return b.getvalue()


@pytest.fixture()
def host():
    h = HttpOcrHost("http://ocrhost.test")
    h._c = TestClient(app, base_url="http://ocrhost.test")
    return h


def test_healthz():
    r = TestClient(app).get("/healthz").json()
    assert r["status"] == "ok" and r["device"] == "cpu"


def test_rapid_over_http(host):
    pytest.importorskip("rapidocr_onnxruntime")
    lines = host.rapid(_png("Telma 40 mg 1-0-1"))
    assert lines and "40" in " ".join(ln.text for ln in lines)
