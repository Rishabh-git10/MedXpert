"""Health check tests for the MedXpert API root endpoint.

Verifies that the root ("/") endpoint responds successfully with the
expected status message and proper JSON content type.
"""

import pytest
from fastapi.testclient import TestClient

from src.api import app


@pytest.mark.api
class TestRootEndpoint:
    """Tests for the API root endpoint availability and response format."""
    def test_root_endpoint_success(self):
        """Return 200 OK and correct status message."""
        client = TestClient(app)
        resp = client.get("/")
        assert resp.status_code == 200
        assert resp.json() == {"message": "MedXpert API is running."}

    def test_root_endpoint_returns_json(self):
        """Ensure root endpoint returns JSON content type."""
        client = TestClient(app)
        resp = client.get("/")
        assert resp.headers["content-type"] == "application/json"
