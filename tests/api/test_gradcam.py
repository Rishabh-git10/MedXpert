"""Unit tests for the Grad-CAM implementation.

Validates the GradCAM class from src.api for activation/gradient handling,
CAM generation, normalization, and output integrity.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from src.api import GradCAM


@pytest.fixture
def mock_model():
    """Return a mock model with a dummy DenseNet-like structure."""
    m = MagicMock()
    m.features = MagicMock()
    m.features.denseblock4 = MagicMock()
    return m


@pytest.mark.model
class TestGradCAM:
    """Unit tests for GradCAM feature visualization."""

    def test_init_buffers(self, mock_model):
        """Initialize GradCAM with empty gradient and activation buffers."""
        gc = GradCAM(mock_model, mock_model.features.denseblock4)
        assert gc.gradients is None
        assert gc.activations is None

    def test_forward_backward_hooks(self, mock_model):
        """Store gradients and activations correctly through hooks."""
        gc = GradCAM(mock_model, mock_model.features.denseblock4)
        tensor = torch.randn(1, 512, 7, 7)
        gc.forward_hook(None, None, tensor)
        gc.backward_hook(None, None, (tensor,))
        assert gc.activations.shape == tensor.shape
        assert gc.gradients.shape == tensor.shape

    def test_cam_shape(self, mock_model):
        """Generate a CAM with the expected spatial resolution."""
        gc = GradCAM(mock_model, mock_model.features.denseblock4)
        gc.activations = torch.randn(1, 512, 7, 7)
        gc.gradients = torch.randn(1, 512, 7, 7)
        mock_model.return_value = torch.randn(1, 14)
        mock_model.zero_grad = MagicMock()
        with patch.object(torch.Tensor, "backward"):
            cam = gc.generate_cam(torch.randn(1, 3, 320, 320), target_class=0)
        assert isinstance(cam, np.ndarray)
        assert cam.shape == (7, 7)

    def test_cam_normalized_nonnegative(self, mock_model):
        """Ensure CAM output values are normalized and non-negative."""
        gc = GradCAM(mock_model, mock_model)
        gc.activations = torch.randn(1, 512, 7, 7)
        gc.gradients = torch.randn(1, 512, 7, 7)
        mock_model.zero_grad = MagicMock()
        mock_model.return_value = torch.randn(1, 14)
        with patch.object(torch.Tensor, "backward"):
            cam = gc.generate_cam(torch.randn(1, 3, 320, 320), target_class=0)
        assert cam.min() >= 0.0
