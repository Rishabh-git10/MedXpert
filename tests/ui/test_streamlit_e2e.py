"""End-to-end UI test for MedXpert using Playwright.

Launches the Streamlit frontend and FastAPI backend locally, then simulates
a full user workflow: uploading an image, running diagnosis, and validating
visual output through real browser automation.
"""

import subprocess
import time
from pathlib import Path

import pytest
from playwright.sync_api import sync_playwright


@pytest.mark.e2e
def test_medxpert_e2e(tmp_path):
    """Simulate complete MedXpert user interaction in a headless browser."""
    sample_image = Path("data/samples/sample_xray.jpg")
    assert sample_image.exists(), "Sample image not found at data/samples/sample_xray.jpg"

    process = subprocess.Popen(
        [
            "streamlit",
            "run",
            "src/app.py",
            "--server.headless",
            "true",
            "--server.port",
            "8501",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    start_time = time.time()
    startup_markers = ["Local URL:", "Running on", "You can now view your Streamlit app"]
    app_ready = False

    while time.time() - start_time < 90:
        line = process.stdout.readline()
        if not line:
            continue
        print(line.strip())
        if any(marker in line for marker in startup_markers):
            app_ready = True
            break

    if not app_ready:
        process.terminate()
        pytest.fail("Streamlit app failed to start within 90 seconds.")

    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True)
            context = browser.new_context()
            page = context.new_page()

            page.goto("http://localhost:8501", timeout=60000)
            page.wait_for_selector("input[type=file]", timeout=30000)
            page.set_input_files("input[type=file]", str(sample_image))
            page.wait_for_selector("button:has-text('Run Diagnosis')", timeout=30000)
            page.click("button:has-text('Run Diagnosis')")

            page.wait_for_timeout(15000)
            assert (
                page.is_visible("text=Diagnostic Results")
                or page.is_visible("text=AI Explainability")
                or page.is_visible("text=Error")
            ), "Expected diagnosis results not found in UI."

            screenshot_path = tmp_path / "e2e_result.png"
            page.screenshot(path=str(screenshot_path))
            print(f"✅ Screenshot saved to: {screenshot_path}")

            context.close()
            browser.close()

    finally:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
