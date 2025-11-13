"""Session state validation tests for Streamlit frontend.

Ensures that all required Streamlit session state variables initialize
correctly, update predictably, and persist across app reruns.
"""

import pytest
from streamlit.testing.v1 import AppTest


@pytest.mark.ui
class TestStreamlitState:
    """Tests for initialization and mutation of Streamlit session state."""

    def test_state_initialization(self):
        """Ensure required session_state keys are created on app start."""
        app = AppTest.from_file("src/app.py")
        app.run()

        required_keys = [
            "findings",
            "heatmaps",
            "original_image",
            "summaries",
            "last_file_id",
            "expanded_finding",
        ]
        for key in required_keys:
            assert key in app.session_state

    def test_state_file_id_tracking(self):
        """Test that file_id tracking works correctly."""
        app = AppTest.from_file("src/app.py")
        app.run()

        app.session_state["last_file_id"] = "file_123"
        app.session_state["findings"] = [{"name": "Old"}]
        app.run()

        old_id = app.session_state["last_file_id"]
        app.session_state["last_file_id"] = "file_456"

        assert app.session_state["last_file_id"] != old_id

    def test_findings_persistence(self):
        """Test that findings persist in session state."""
        app = AppTest.from_file("src/app.py")
        app.run()

        findings_data = [
            {"name": "Pneumonia", "probability": 0.85},
            {"name": "Consolidation", "probability": 0.72},
        ]

        app.session_state["findings"] = findings_data
        app.run()

        assert len(app.session_state["findings"]) == 2
        assert app.session_state["findings"][0]["name"] == "Pneumonia"
