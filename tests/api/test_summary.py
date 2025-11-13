"""Tests for the POST /get_summary/ endpoint.

Validates the MedXpert API's pathology summary generation, ensuring
proper responses for valid inputs, invalid payloads, and LLM failures.
"""

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from langchain_core.runnables.base import RunnableSequence

from src.api import app, pathology_cols


@pytest.mark.api
class TestSummaryEndpoint:
    """Integration tests for the /get_summary/ FastAPI route."""
    def test_get_summary_success(self, mock_llm_chain):
        """Return 200 OK and valid summary for a known pathology."""
        client = TestClient(app)
        resp = client.post("/get_summary/", json={"pathology": "Pneumonia"})
        assert resp.status_code == 200
        assert isinstance(resp.json().get("summary"), str)

    def test_all_pathologies(self, mock_llm_chain):
        """Ensure all known pathologies return valid summaries."""
        client = TestClient(app)
        for p in pathology_cols:
            assert (
                client.post("/get_summary/", json={"pathology": p}).status_code == 200
            )

    def test_missing_field(self):
        """Return 422 when pathology field is missing."""
        client = TestClient(app)
        assert client.post("/get_summary/", json={}).status_code == 422

    def test_invalid_json(self):
        """Reject invalid JSON payloads."""
        client = TestClient(app)
        resp = client.post(
            "/get_summary/",
            data="not json",
            headers={"Content-Type": "application/json"},
        )
        assert resp.status_code == 422

    def test_unknown_pathology(self, mock_llm_chain):
        """Handle unknown or unrecognized pathology names gracefully."""
        client = TestClient(app)
        assert (
            client.post(
                "/get_summary/", json={"pathology": "Unknown Disease XYZ"}
            ).status_code
            == 200
        )

    def test_empty_pathology(self, mock_llm_chain):
        """Handle empty pathology values without failure."""
        client = TestClient(app)
        assert client.post("/get_summary/", json={"pathology": ""}).status_code == 200

    def test_llm_failure(self):
        """Simulate and handle LLM/QA chain execution failures."""
        client = TestClient(app)

        with patch.object(
            RunnableSequence, "invoke", side_effect=Exception("LLM Error")
        ):
            response = client.post("/get_summary/", json={"pathology": "Pneumonia"})
            assert response.status_code == 500
            data = response.json()
            assert "error" in data
            assert "LLM Error" in data["error"]
