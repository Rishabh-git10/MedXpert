"""Pytest configuration and fixtures for the MedXpert test suite."""

import os
import sys
from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
import pytest
from streamlit.testing.v1 import AppTest

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def pytest_configure(config):
    """Register custom test markers and configure environment variables."""
    for name, desc in [
        ("slow", "marks tests as slow"),
        ("integration", "integration tests"),
        ("unit", "unit tests"),
        ("performance", "performance tests"),
        ("security", "security-related tests"),
        ("api", "API endpoint tests"),
        ("ui", "UI component tests"),
        ("model", "ML model tests"),
    ]:
        config.addinivalue_line("markers", f"{name}: {desc}")

    os.environ["TESTING"] = "true"


@pytest.fixture(scope="session", autouse=True)
def set_test_env():
    """Set consistent environment variables across all tests."""
    os.environ.update(
        {
            "STREAMLIT_HEADLESS": "true",
            "STREAMLIT_TEST_MODE": "true",
            "PYTHONWARNINGS": "ignore",
            "TESTING": "true",
        }
    )


@pytest.fixture
def sample_xray_image():
    """Return a random 512×512 RGB image simulating an X-ray."""
    return np.random.randint(50, 200, (512, 512, 3), dtype=np.uint8)


@pytest.fixture
def sample_xray_bytes(sample_xray_image):
    """Encode the sample image as JPEG bytes."""
    _, buffer = cv2.imencode(".jpg", sample_xray_image)
    return buffer.tobytes()


@pytest.fixture
def sample_xray_file(sample_xray_bytes):
    """Return a BytesIO file-like object from encoded image bytes."""
    return BytesIO(sample_xray_bytes)


@pytest.fixture
def api_client():
    """Return a FastAPI test client for the MedXpert API."""
    from fastapi.testclient import TestClient
    from src.api import app

    return TestClient(app)


@pytest.fixture
def create_test_image():
    """Factory fixture to create synthetic images with configurable properties."""
    def _create(width=512, height=512, channels=3, intensity_range=(50, 200)):
        return np.random.randint(
            intensity_range[0],
            intensity_range[1],
            (height, width, channels),
            dtype=np.uint8,
        )

    return _create


@pytest.fixture
def encode_image():
    """Factory fixture to encode a NumPy image into bytes."""
    def _encode(image, format=".jpg"):
        _, buffer = cv2.imencode(format, image)
        return buffer.tobytes()

    return _encode


@pytest.fixture
def timer():
    """Context manager for performance timing."""
    import time

    class Timer:
        def __enter__(self):
            self.start = time.time()
            return self

        def __exit__(self, *args):
            self.end = time.time()
            self.elapsed = self.end - self.start

    return Timer()


@pytest.fixture
def multiple_xray_images():
    """Generate multiple random X-ray images for batch testing."""
    return [
        np.random.randint(50 + i * 10, 200, (512, 512, 3), dtype=np.uint8)
        for i in range(3)
    ]


@pytest.fixture
def mock_llm_chain():
    """Mock the RAG LLM chain to avoid external API calls."""
    from unittest.mock import patch

    with patch("src.api.qa_chain") as mock_chain:
        mock_chain.invoke.return_value = "Mocked medical summary."
        yield mock_chain


@pytest.fixture(scope="function")
def streamlit_app():
    """Initialize the Streamlit frontend for integration testing."""
    app = AppTest.from_file("src/app.py")
    app.run(timeout=30)
    yield app


@pytest.fixture
def sample_findings_data():
    """Return sample findings for testing logical functions."""
    return [
        {"name": "Pneumonia", "probability": 0.85, "index": 7},
        {"name": "Consolidation", "probability": 0.72, "index": 6},
        {"name": "No Finding", "probability": 0.15, "index": 0},
    ]


@pytest.fixture
def sample_heatmaps_data():
    """Return base64-encoded dummy heatmap data."""
    import base64
    return {
        "Pneumonia": base64.b64encode(b"fake_heatmap_data_1").decode(),
        "Consolidation": base64.b64encode(b"fake_heatmap_data_2").decode(),
    }


@pytest.fixture
def mock_api_response_success():
    """Return a mock successful API response."""
    valid_base64 = "R0lGODlhAQABAIAAAAUEBAAAACwAAAAAAQABAAACAkQBADs="
    return {
        "findings": [
            {"name": "Pneumonia", "probability": 0.85, "index": 7},
            {"name": "Edema", "probability": 0.72, "index": 5},
        ],
        "heatmaps": {"Pneumonia": valid_base64, "Edema": valid_base64},
    }


@pytest.fixture
def mock_api_response_normal():
    """Return a mock API response for a normal (no finding) case."""
    return {
        "findings": [{"name": "No Finding", "probability": 0.92, "index": 0}],
        "heatmaps": {},
    }
