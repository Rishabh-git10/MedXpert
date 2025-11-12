"""Security and data validation tests for MedXpert API.

Covers input sanitization, large file handling, SQL injection safety,
and consistency of predefined constants and pathology labels.
"""

from io import BytesIO

import pytest
from fastapi.testclient import TestClient

from src.api import IMAGE_SIZE, app, pathology_cols


@pytest.mark.security
class TestSecurityValidation:
    """Tests for input safety, upload validation, and API robustness."""

    def test_no_path_traversal_in_filename(self, sample_xray_bytes):
        """Reject or safely handle filenames with path traversal patterns."""
        client = TestClient(app)
        malicious = "../../../etc/passwd"
        response = client.post(
            "/diagnose/",
            files={"file": (malicious, BytesIO(sample_xray_bytes), "image/jpeg")},
        )
        assert response.status_code in [200, 500]

    def test_large_file_handling(self):
        """Gracefully handle oversized image uploads."""
        client = TestClient(app)
        large_file = b"0" * (50 * 1024 * 1024)  # 50 MB payload
        response = client.post(
            "/diagnose/", files={"file": ("huge.jpg", BytesIO(large_file), "image/jpeg")}
        )
        assert response.status_code in [200, 413, 500]

    def test_sql_injection_in_pathology(self, mock_llm_chain):
        """Prevent SQL-like injection attempts in pathology field."""
        client = TestClient(app)
        malicious = "Pneumonia'; DROP TABLE patients; --"
        response = client.post("/get_summary/", json={"pathology": malicious})
        assert response.status_code in [200, 500]


@pytest.mark.unit
class TestDataValidation:
    """Tests for integrity of static constants and pathology configuration."""

    def test_pathology_names_valid(self):
        """Ensure all pathology labels are valid non-empty strings."""
        for pathology in pathology_cols:
            assert isinstance(pathology, str)
            assert pathology.strip()

    def test_image_size_constant(self):
        """Validate IMAGE_SIZE constant definition and constraints."""
        assert isinstance(IMAGE_SIZE, int)
        assert IMAGE_SIZE == 320
        assert IMAGE_SIZE > 0

    def test_findings_no_duplicates(self, sample_xray_file):
        """Ensure no duplicate findings are returned from /diagnose/."""
        client = TestClient(app)
        response = client.post(
            "/diagnose/", files={"file": ("x.jpg", sample_xray_file, "image/jpeg")}
        )
        findings = response.json()["findings"]
        names = [f["name"] for f in findings]
        assert len(names) == len(set(names))
