"""Unit tests for predict_image() in MedXpert API.

Validates prediction output structure, field integrity, probability ranges,
and handling of grayscale inputs in the image processing pipeline.
"""

import cv2
import numpy as np
import pytest

from src.api import IMAGE_SIZE, predict_image


@pytest.mark.model
class TestPredictImage:
    """Unit tests for the core image prediction workflow."""

    def test_returns_types(self, sample_xray_bytes):
        """Return findings list and heatmaps dictionary."""
        findings, heatmaps = predict_image(sample_xray_bytes)
        assert isinstance(findings, list)
        assert isinstance(heatmaps, dict)

    def test_required_fields(self, sample_xray_bytes):
        """Ensure each finding contains required fields."""
        findings, _ = predict_image(sample_xray_bytes)
        for finding in findings:
            assert {"name", "probability", "index"} <= set(finding.keys())

    def test_probability_range(self, sample_xray_bytes):
        """Validate all probabilities are within the 0–1 range."""
        findings, _ = predict_image(sample_xray_bytes)
        for finding in findings:
            assert 0.0 <= finding["probability"] <= 1.0

    def test_no_finding_logic(self, sample_xray_bytes):
        """Verify 'No Finding' does not appear with other pathologies."""
        findings, _ = predict_image(sample_xray_bytes)
        has_no_finding = any(f["name"] == "No Finding" for f in findings)
        has_others = any(f["name"] != "No Finding" for f in findings)
        if has_others:
            assert not (has_no_finding and len(findings) > 1)

    def test_grayscale_supported(self):
        """Handle grayscale images without raising errors."""
        gray = np.random.randint(0, 255, (512, 512), dtype=np.uint8)
        _, buffer = cv2.imencode(".jpg", gray)
        findings, heatmaps = predict_image(buffer.tobytes())
        assert isinstance(findings, list)
        assert isinstance(heatmaps, dict)

    def test_image_size_constant(self):
        """Ensure IMAGE_SIZE constant is correctly defined."""
        assert isinstance(IMAGE_SIZE, int)
        assert IMAGE_SIZE == 320
        assert IMAGE_SIZE > 0
