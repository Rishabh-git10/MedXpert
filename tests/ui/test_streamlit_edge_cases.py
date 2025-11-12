"""Edge case tests for Streamlit frontend behavior.

Validates that the Streamlit UI handles missing uploads, absent summaries,
and unexpected state conditions gracefully without raising exceptions.
"""

import pytest
from streamlit.testing.v1 import AppTest


@pytest.mark.ui
class TestStreamlitEdgeCases:
    """Tests for resilience and graceful error handling in the Streamlit UI."""

    def test_no_upload_no_crash(self):
        """Ensure the app runs cleanly when no file is uploaded."""
        app = AppTest.from_file("src/app.py")
        app.run(timeout=30)
        assert len(app.exception) == 0

    def test_missing_summary_key(self):
        """Verify missing summary entries do not cause UI failures."""
        app = AppTest.from_file("src/app.py")
        app.run()

        app.session_state["findings"] = [{"name": "Unknown", "probability": 0.5}]
        app.session_state["summaries"] = {}
        app.run()

        assert "Unknown" in [f["name"] for f in app.session_state["findings"]]
        assert len(app.exception) == 0
