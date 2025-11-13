"""Performance and latency tests for MedXpert API.

Ensures that /diagnose/ and /get_summary/ endpoints respond within
acceptable time limits under normal conditions.
"""

import time

import pytest
from fastapi.testclient import TestClient

from src.api import app


@pytest.mark.performance
class TestPerformance:
    """Latency guard tests for MedXpert API endpoints."""
    def test_diagnosis_response_time(self, sample_xray_file):
        """Ensure /diagnose/ completes within 30 seconds."""
        client = TestClient(app)
        start = time.time()
        resp = client.post(
            "/diagnose/", files={"file": ("x.jpg", sample_xray_file, "image/jpeg")}
        )
        elapsed = time.time() - start
        assert resp.status_code == 200
        assert elapsed < 30.0

    def test_summary_response_time(self, mock_llm_chain):
        """Ensure /get_summary/ completes within 10 seconds."""
        client = TestClient(app)
        start = time.time()
        resp = client.post("/get_summary/", json={"pathology": "Pneumonia"})
        elapsed = time.time() - start
        assert resp.status_code == 200
        assert elapsed < 10.0
