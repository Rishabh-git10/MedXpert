"""End-to-end integration tests for MedXpert API workflows.

Validates the complete interaction between /diagnose/ and /get_summary/
endpoints, ensuring smooth data flow and consistent outputs across requests.
"""

from io import BytesIO

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.api import app


@pytest.mark.integration
class TestIntegration:
    """Integration tests for end-to-end diagnostic and summary workflows."""
    def test_full_workflow(self, sample_xray_file, mock_llm_chain):
        """Run diagnosis followed by summary retrieval for detected findings."""
        client = TestClient(app)
        diag = client.post(
            "/diagnose/", files={"file": ("x.jpg", sample_xray_file, "image/jpeg")}
        )
        assert diag.status_code == 200
        findings = diag.json()["findings"]
        for f in findings:
            if f["name"] != "No Finding":
                s = client.post("/get_summary/", json={"pathology": f["name"]})
                assert s.status_code == 200
                assert "summary" in s.json()

    def test_multiple_images(self):
        """Process multiple image uploads and verify consistent API responses."""
        client = TestClient(app)
        images = []
        for _ in range(3):
            img = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
            _, buf = cv2.imencode(".jpg", img)
            images.append(BytesIO(buf.tobytes()))
        results = [
            client.post("/diagnose/", files={"file": ("x.jpg", f, "image/jpeg")}).json()
            for f in images
        ]
        assert all("findings" in r for r in results)
        assert all("heatmaps" in r for r in results)
