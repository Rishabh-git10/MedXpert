"""Performance regression tests for Streamlit frontend.

Measures rendering latency to ensure UI responsiveness remains within
acceptable limits across updates.
"""

import time

import pytest
from streamlit.testing.v1 import AppTest


@pytest.mark.ui
@pytest.mark.performance
class TestStreamlitPerformance:
    """Tests for Streamlit UI rendering speed and stability."""

    def test_app_renders_under_threshold(self):
        """Ensure initial load completes under 2 seconds."""
        start = time.time()
        app = AppTest.from_file("src/app.py")
        app.run(timeout=30)
        elapsed = time.time() - start
        assert elapsed < 2.0, f"UI took too long to render: {elapsed:.2f}s"
        assert len(app.exception) == 0
