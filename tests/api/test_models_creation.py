"""Unit tests for model factory functions.

Validates initialization and configuration of DenseNet-121 and
MobileNetV3 view classifier architectures in src.api.
"""

from unittest.mock import MagicMock, patch

import pytest

from src.api import create_model_architecture, create_view_classifier


@pytest.mark.model
class TestModelCreation:
    """Tests for MedXpert model factory initialization and configuration."""

    @patch("src.api.timm.create_model")
    @patch("src.api.torch.load")
    def test_create_densenet(self, mock_load, mock_create):
        """Initialize DenseNet-121 with correct parameters and eval mode."""
        m = MagicMock()
        m.eval = MagicMock(return_value=m)
        m.to = MagicMock(return_value=m)
        mock_create.return_value = m
        mock_load.return_value = {}

        create_model_architecture()

        mock_create.assert_called_once_with(
            "densenet121", pretrained=False, num_classes=14
        )
        m.eval.assert_called_once()
        m.to.assert_called()

    @patch("src.api.timm.create_model")
    @patch("src.api.torch.load")
    def test_create_mobilenet_view(self, mock_load, mock_create):
        """Initialize MobileNetV3 view classifier with correct configuration."""
        m = MagicMock()
        m.eval = MagicMock(return_value=m)
        m.to = MagicMock(return_value=m)
        mock_create.return_value = m
        mock_load.return_value = {}

        create_view_classifier()

        mock_create.assert_called_once_with(
            "mobilenetv3_small_100", pretrained=False, num_classes=2
        )
        m.eval.assert_called_once()
        m.to.assert_called()
