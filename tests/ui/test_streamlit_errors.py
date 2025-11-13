"""Error handling tests for Streamlit frontend.

Ensures the Streamlit interface remains stable during API failures,
missing data, or incomplete diagnostic outputs.
"""

import pytest
from streamlit.testing.v1 import AppTest


@pytest.mark.ui
class TestStreamlitErrors:
    """Tests for robustness of Streamlit UI under invalid or missing data."""

    def test_empty_findings_handling(self):
        """Test handling of empty findings list."""
        app = AppTest.from_file("src/app.py")
        app.run()

        app.session_state["findings"] = []
        app.run()

        assert len(app.exception) == 0

    def test_missing_heatmaps(self):
        """Test handling when heatmaps dictionary is empty."""
        app = AppTest.from_file("src/app.py")
        app.run()

        app.session_state["findings"] = [{"name": "Pneumonia", "probability": 0.8}]
        app.session_state["heatmaps"] = {}
        app.run()

        assert len(app.exception) == 0
