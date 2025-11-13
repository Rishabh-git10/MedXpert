"""Smoke tests for MedXpert.

Verifies that core modules import correctly, the FastAPI backend and Streamlit
frontend initialize, and the testing environment variables are properly set.
"""

import importlib
import os

import pytest


@pytest.mark.smoke
def test_core_imports():
    """Ensure core modules can be imported without ImportError."""
    for module in ["src.api", "src.app", "torch", "streamlit"]:
        importlib.import_module(module)


@pytest.mark.smoke
def test_fastapi_app_loads():
    """Check FastAPI app initializes correctly."""
    from src.api import app

    assert hasattr(app, "router")


@pytest.mark.smoke
def test_streamlit_script_loads():
    """Ensure Streamlit app script can be read and contains Streamlit calls."""
    path = "src/app.py"
    assert os.path.exists(path), f"{path} not found"
    with open(path, "r", encoding="utf-8") as f:
        code = f.read()
    assert "st." in code or "streamlit" in code.lower()


@pytest.mark.smoke
def test_environment_ready():
    """Confirm essential environment variables for test mode are active."""
    assert os.getenv("STREAMLIT_TEST_MODE") == "true"
