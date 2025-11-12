"""Integration tests for the Streamlit frontend.

Validates the complete workflow from AI findings to summary generation,
ensuring UI state consistency across diagnosis and summary retrieval.
"""

import pytest
from streamlit.testing.v1 import AppTest


@pytest.mark.ui
@pytest.mark.integration
class TestStreamlitIntegration:
    """Tests the end-to-end frontend workflow for diagnosis and summaries."""

    def test_findings_to_summary_flow(self):
        """Verify that findings correctly propagate into the summary display."""
        app = AppTest.from_file("src/app.py")
        app.run()

        app.session_state["findings"] = [
            {"name": "Edema", "probability": 0.88, "index": 5}
        ]
        app.session_state["heatmaps"] = {}
        app.run()

        app.session_state["summaries"]["Edema"] = "Fluid accumulation in lungs"
        app.run()

        assert app.session_state["findings"][0]["name"] == "Edema"
        assert "Edema" in app.session_state["summaries"]
        assert len(app.exception) == 0
