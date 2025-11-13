"""Rendering validation tests for Streamlit frontend.

Ensures that the MedXpert interface loads successfully, displays the
expected headers and layout elements, and initializes without errors.
"""

import pytest
from streamlit.testing.v1 import AppTest


@pytest.mark.ui
class TestStreamlitRender:
    """Tests for Streamlit UI rendering and layout integrity."""

    def test_app_loads(self):
        """Ensure the MedXpert app initializes cleanly without runtime errors."""
        app = AppTest.from_file("src/app.py")
        app.run(timeout=30)
        assert len(app.exception) == 0
        assert app.session_state is not None

    def test_headers_present(self):
        """Check main header and subtitle render correctly."""
        app = AppTest.from_file("src/app.py")
        app.run()

        markdown_texts = [m.value for m in app.markdown]
        assert any("MedXpert" in m for m in markdown_texts)
        assert any("Diagnostic Assistant" in m for m in markdown_texts)

    def test_sidebar_instructions(self):
        """Sidebar must include step instructions."""
        app = AppTest.from_file("src/app.py")
        app.run()

        markdown_texts = [m.value for m in app.markdown]
        assert any("Upload" in t for t in markdown_texts)
