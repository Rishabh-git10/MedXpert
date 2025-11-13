"""Tests for the POST /diagnose/ endpoint.

Validates MedXpert’s image diagnosis API for various inputs and edge cases,
including valid uploads, format handling, corrupted files, and probability sorting.
"""

import base64
from io import BytesIO

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from src.api import app


@pytest.mark.api
class TestDiagnoseEndpoint:
    """Integration tests for the /diagnose/ FastAPI route."""
    def test_diagnose_with_valid_image(self, sample_xray_file):
        """Return 200 OK and structured findings for a valid JPEG image."""    
        client = TestClient(app)
        resp = client.post(
            "/diagnose/", files={"file": ("test.jpg", sample_xray_file, "image/jpeg")}
        )
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data.get("findings"), list)
        assert isinstance(data.get("heatmaps"), dict)

    def test_findings_structure(self, sample_xray_file):
        """Each finding should have valid name and probability fields."""
        client = TestClient(app)
        data = client.post(
            "/diagnose/", files={"file": ("test.jpg", sample_xray_file, "image/jpeg")}
        ).json()
        if data["findings"]:
            f = data["findings"][0]
            assert isinstance(f["name"], str)
            assert isinstance(f["probability"], float)
            assert 0.0 <= f["probability"] <= 1.0

    def test_png_upload(self):
        """Accept PNG uploads and return 200 OK."""
        client = TestClient(app)
        img = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        _, buf = cv2.imencode(".png", img)
        resp = client.post(
            "/diagnose/", files={"file": ("x.png", BytesIO(buf.tobytes()), "image/png")}
        )
        assert resp.status_code == 200

    def test_missing_file(self):
        """Reject requests without file uploads (422 validation error)."""
        client = TestClient(app)
        resp = client.post("/diagnose/")
        assert resp.status_code == 422

    def test_invalid_bytes(self):
        """Return 500 error when file content is not an image."""
        client = TestClient(app)
        resp = client.post(
            "/diagnose/",
            files={"file": ("bad.jpg", BytesIO(b"not_an_image"), "image/jpeg")},
        )
        assert resp.status_code == 500
        assert "error" in resp.json()

    def test_corrupted_bytes(self):
        """Handle corrupted JPEG bytes gracefully."""
        client = TestClient(app)
        corrupted = b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01corrupted"
        resp = client.post(
            "/diagnose/", files={"file": ("bad.jpg", BytesIO(corrupted), "image/jpeg")}
        )
        assert resp.status_code == 500

    def test_heatmap_base64_images(self, sample_xray_file):
        """Ensure heatmaps are valid base64-encoded images."""
        client = TestClient(app)
        data = client.post(
            "/diagnose/", files={"file": ("x.jpg", sample_xray_file, "image/jpeg")}
        ).json()
        for name, b64s in data["heatmaps"].items():
            decoded = base64.b64decode(b64s)
            assert len(decoded) > 0
            Image.open(BytesIO(decoded))

    def test_large_image(self):
        """Accept large input images without server errors."""
        client = TestClient(app)
        img = np.random.randint(0, 255, (3840, 2160, 3), dtype=np.uint8)
        _, buf = cv2.imencode(".jpg", img)
        resp = client.post(
            "/diagnose/",
            files={"file": ("large.jpg", BytesIO(buf.tobytes()), "image/jpeg")},
        )
        assert resp.status_code == 200

    def test_small_image(self):
        """Accept small input images without server errors."""
        client = TestClient(app)
        img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        _, buf = cv2.imencode(".jpg", img)
        resp = client.post(
            "/diagnose/",
            files={"file": ("small.jpg", BytesIO(buf.tobytes()), "image/jpeg")},
        )
        assert resp.status_code == 200

    def test_sorted_by_probability(self, sample_xray_file):
        """Ensure findings are sorted in descending probability order."""
        client = TestClient(app)
        findings = client.post(
            "/diagnose/", files={"file": ("x.jpg", sample_xray_file, "image/jpeg")}
        ).json()["findings"]
        if len(findings) > 1:
            for i in range(len(findings) - 1):
                assert findings[i]["probability"] >= findings[i + 1]["probability"]
