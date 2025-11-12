"""Summary logic validation tests for Streamlit frontend.

Ensures that clinical summaries are correctly stored, persisted, and
associated with their corresponding findings in session state.
"""

import pytest
from streamlit.testing.v1 import AppTest


@pytest.mark.ui
class TestStreamlitSummary:
    """Tests for session state storage and handling of clinical summaries."""

    def test_summary_storage(self):
        """Verify a single summary is stored and retrieved correctly."""
        app = AppTest.from_file("src/app.py")
        app.run()

        app.session_state["findings"] = [{"name": "Pneumonia", "probability": 0.91}]
        app.session_state["summaries"] = {}
        app.run()

        app.session_state["summaries"]["Pneumonia"] = "Clinical information about pneumonia"
        app.run()

        assert "Pneumonia" in app.session_state["summaries"]
        assert len(app.exception) == 0

    def test_multiple_summaries(self):
        """Ensure multiple pathology summaries are handled correctly."""
        app = AppTest.from_file("src/app.py")
        app.run()

        app.session_state["findings"] = [
            {"name": "Pneumonia", "probability": 0.85},
            {"name": "Edema", "probability": 0.72},
        ]
        app.session_state["summaries"] = {
            "Pneumonia": "Summary 1",
            "Edema": "Summary 2",
        }
        app.run()

        assert len(app.session_state["summaries"]) == 2
        assert "Pneumonia" in app.session_state["summaries"]
        assert "Edema" in app.session_state["summaries"]
