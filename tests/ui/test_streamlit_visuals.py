"""Visual state validation tests for Streamlit frontend.

Ensures that Grad-CAM heatmaps and uploaded images are correctly stored
and maintained in Streamlit session state without causing UI errors.
"""

import base64
from io import BytesIO

import pytest
from PIL import Image
from streamlit.testing.v1 import AppTest


@pytest.mark.ui
class TestStreamlitVisuals:
    """Tests for visual data handling and session persistence."""

    def test_heatmap_storage(self):
        """Test that heatmaps are stored in session state."""
        img = Image.new("RGB", (32, 32), color="gray")
        buf = BytesIO()
        img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        app = AppTest.from_file("src/app.py")
        app.run()

        app.session_state["heatmaps"] = {"Pneumonia": b64}
        app.session_state["findings"] = [{"name": "Pneumonia", "probability": 0.9}]
        app.run()

        assert "Pneumonia" in app.session_state["heatmaps"]
        assert len(app.exception) == 0

    def test_original_image_storage(self):
        """Test that original image is stored."""
        img = Image.new("RGB", (128, 128), color="gray")

        app = AppTest.from_file("src/app.py")
        app.run()

        app.session_state["original_image"] = img
        app.run()

        assert app.session_state["original_image"] is not None
        assert len(app.exception) == 0
