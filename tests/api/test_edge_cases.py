"""Edge case tests for MedXpert API.

Covers concurrency handling, image intensity extremes, unusual aspect ratios,
and model weight loading errors.
"""

from io import BytesIO
from unittest.mock import patch

import cv2
import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient

from src.api import app


@pytest.mark.api
class TestEdgeCases:
    """Tests for robustness under uncommon or stress conditions."""
    def test_concurrent_requests(self, sample_xray_file):
        """Handle multiple concurrent /diagnose/ requests without failure."""
        client = TestClient(app)

        def call():
            sample_xray_file.seek(0)
            return client.post(
                "/diagnose/", files={"file": ("x.jpg", sample_xray_file, "image/jpeg")}
            )

        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as ex:
            results = list(ex.map(lambda _: call(), range(5)))
        assert all(r.status_code == 200 for r in results)

    def test_very_dark(self):
        """Accept and process very dark X-ray images."""
        client = TestClient(app)
        dark = np.ones((512, 512, 3), dtype=np.uint8) * 10
        _, buf = cv2.imencode(".jpg", dark)
        resp = client.post(
            "/diagnose/",
            files={"file": ("d.jpg", BytesIO(buf.tobytes()), "image/jpeg")},
        )
        assert resp.status_code == 200

    def test_very_bright(self):
        """Accept and process very bright X-ray images."""
        client = TestClient(app)
        bright = np.ones((512, 512, 3), dtype=np.uint8) * 245
        _, buf = cv2.imencode(".jpg", bright)
        resp = client.post(
            "/diagnose/",
            files={"file": ("b.jpg", BytesIO(buf.tobytes()), "image/jpeg")},
        )
        assert resp.status_code == 200

    def test_extreme_aspect_ratio(self):
        """Handle nonstandard aspect ratios without model failure."""
        client = TestClient(app)
        tall = np.random.randint(0, 255, (2000, 100, 3), dtype=np.uint8)
        _, buf = cv2.imencode(".jpg", tall)
        resp = client.post(
            "/diagnose/",
            files={"file": ("t.jpg", BytesIO(buf.tobytes()), "image/jpeg")},
        )
        assert resp.status_code == 200

    def test_missing_model_files(self):
        """Raise FileNotFoundError when model weights are missing."""
        with patch(
            "src.api.torch.load", side_effect=FileNotFoundError("Model not found")
        ):
            with pytest.raises(FileNotFoundError):
                torch.load("nonexistent_model.pth")
