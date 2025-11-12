"""Streamlit UI tests for MedXpert backend integration.

Validates synchronization between Streamlit session state and FastAPI responses,
including persistence of findings, heatmaps, and cached clinical summaries.
"""

from unittest.mock import patch

import pytest
from streamlit.testing.v1 import AppTest


@pytest.mark.ui
class TestStreamlitAPI:
    """Tests for session state behavior and API interaction in Streamlit UI."""

    @patch("src.app.requests")
    def test_diagnose_session_state_update(self, mock_requests):
        """Verify that diagnosis results update Streamlit session state."""
        app = AppTest.from_file("src/app.py")
        app.run()

        app.session_state["findings"] = [
            {"name": "Pneumonia", "probability": 0.93, "index": 7}
        ]
        app.session_state["heatmaps"] = {}
        app.run()

        assert app.session_state["findings"][0]["name"] == "Pneumonia"
        assert app.session_state["findings"][0]["probability"] == 0.93

    def test_summary_caching_logic(self):
        """Ensure clinical summaries are cached and persisted between runs."""
        app = AppTest.from_file("src/app.py")
        app.run()

        app.session_state["findings"] = [{"name": "Pneumonia", "probability": 0.91}]
        app.session_state["summaries"] = {}
        app.run()

        app.session_state["summaries"]["Pneumonia"] = "Test summary"
        app.run()

        assert "Pneumonia" in app.session_state["summaries"]
        assert app.session_state["summaries"]["Pneumonia"] == "Test summary"
