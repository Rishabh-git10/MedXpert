"""User interaction flow tests for Streamlit frontend.

Simulates end-to-end user behavior including uploading X-rays, triggering
diagnosis, retrieving summaries, handling API errors, and verifying UI
state updates within Streamlit session.
"""

import io
from unittest.mock import patch

import pytest
from streamlit.testing.v1 import AppTest

pytestmark = pytest.mark.ui


class MockUploadedFile:
    """Mock Streamlit UploadedFile that works with PIL.Image.open()"""

    def __init__(self, file_bytes, filename="sample.jpg"):
        self.name = filename
        self.type = "image/jpeg"
        self.file_id = filename
        self._bytes = file_bytes
        self._buffer = io.BytesIO(file_bytes)

    def getvalue(self):
        return self._bytes

    def seek(self, pos):
        return self._buffer.seek(pos)

    def read(self, size=-1):
        return self._buffer.read(size)

    def tell(self):
        return self._buffer.tell()

    def __enter__(self):
        return self._buffer

    def __exit__(self, *args):
        pass


class MockResponse:
    """Helper class to mock requests.Response"""

    def __init__(self, json_data, status_code, text_data=None):
        self.json_data = json_data
        self.status_code = status_code
        self.text = text_data or str(json_data)

    def json(self):
        return self.json_data


@pytest.fixture(scope="function")
def app():
    """Load the app. Patch file_uploader to return None during initial load."""
    with patch("streamlit.file_uploader", return_value=None):
        app_test = AppTest.from_file("src/app.py")
        app_test.run()
        return app_test


class TestStreamlitUserFlow:
    """Simulates Streamlit UI user interactions with proper state changes."""

    @patch("requests.post")
    def test_run_diagnosis_button_triggers_analysis(
        self,
        mock_post,
        app,
        sample_xray_bytes,
        mock_api_response_success,
    ):
        """Ensure Run Diagnosis triggers inference and populates session state."""
        mock_post.return_value = MockResponse(mock_api_response_success, 200)

        mock_file = MockUploadedFile(sample_xray_bytes, "sample_xray.jpg")

        with patch("streamlit.file_uploader", return_value=mock_file):
            app.run()

            run_buttons = [b for b in app.button if "Run Diagnosis" in b.label]
            assert run_buttons, "Run Diagnosis button not found after file upload"

            run_buttons[0].click()
            app.run()

        assert app.session_state.findings is not None
        assert len(app.session_state.findings) > 0
        assert app.session_state.findings[0]["name"] == "Pneumonia"

        all_markdown = " ".join(m.value for m in app.markdown)
        assert "Diagnostic Results" in all_markdown
        assert "Pneumonia" in all_markdown

    @patch("requests.post")
    def test_api_failure_shows_error_message(self, mock_post, app, sample_xray_bytes):
        """Ensure app gracefully handles an API failure."""
        mock_post.return_value = MockResponse({}, 500, text_data="This is a test error")

        mock_file = MockUploadedFile(sample_xray_bytes, "sample.jpg")

        with patch("streamlit.file_uploader", return_value=mock_file):
            app.run()
            run_buttons = [b for b in app.button if "Run Diagnosis" in b.label]
            assert run_buttons, "Run Diagnosis button not found"

            run_buttons[0].click()
            app.run()

        assert len(app.error) > 0
        error_text = " ".join(e.value for e in app.error)
        assert "This is a test error" in error_text

    @patch("requests.post")
    def test_summary_load_button_fetches_summary(
        self,
        mock_post,
        app,
        sample_xray_bytes,
        mock_api_response_success,
    ):
        """Test the full flow: Diagnose -> Click 'Load Summary' -> See summary."""
        mock_diagnose_response = MockResponse(mock_api_response_success, 200)
        mock_summary_response = MockResponse(
            {"summary": "This is the mocked Pneumonia summary."}, 200
        )

        mock_post.side_effect = [mock_diagnose_response, mock_summary_response]

        mock_file = MockUploadedFile(sample_xray_bytes, "sample.jpg")

        with patch("streamlit.file_uploader", return_value=mock_file):
            app.run()

            run_buttons = [b for b in app.button if "Run Diagnosis" in b.label]
            assert run_buttons, "Run Diagnosis button not found"
            run_buttons[0].click()
            app.run()

            summary_buttons = [
                b for b in app.button if "Load Summary for Pneumonia" in b.label
            ]
            assert summary_buttons, "Summary button for Pneumonia not found"
            summary_buttons[0].click()
            app.run()

        assert "Pneumonia" in app.session_state.summaries
        assert (
            "This is the mocked Pneumonia summary."
            in app.session_state.summaries["Pneumonia"]
        )
        all_markdown = " ".join(m.value for m in app.markdown)
        assert "This is the mocked Pneumonia summary." in all_markdown

    @patch("requests.post")
    def test_no_finding_result_shows_normal_message(
        self, mock_post, app, sample_xray_bytes, mock_api_response_normal
    ):
        """Simulate a 'No Finding' case and check for the normal message."""
        mock_post.return_value = MockResponse(mock_api_response_normal, 200)

        mock_file = MockUploadedFile(sample_xray_bytes, "sample.jpg")

        with patch("streamlit.file_uploader", return_value=mock_file):
            app.run()
            run_buttons = [b for b in app.button if "Run Diagnosis" in b.label]
            assert run_buttons, "Run Diagnosis button not found"
            run_buttons[0].click()
            app.run()

        all_markdown = " ".join(m.value for m in app.markdown)
        assert "No Significant Findings" in all_markdown
        assert "finding-normal" in all_markdown

    @patch("requests.post")
    def test_summary_api_error_graceful_handling(
        self,
        mock_post,
        app,
        sample_xray_bytes,
        mock_api_response_success,
    ):
        """Ensure summary errors are handled gracefully in the expander."""
        mock_diagnose_response = MockResponse(mock_api_response_success, 200)
        mock_summary_response = MockResponse({}, 500, text_data="LLM summary failed")
        mock_post.side_effect = [mock_diagnose_response, mock_summary_response]

        mock_file = MockUploadedFile(sample_xray_bytes, "sample.jpg")

        with patch("streamlit.file_uploader", return_value=mock_file):
            app.run()

            run_buttons = [b for b in app.button if "Run Diagnosis" in b.label]
            assert run_buttons, "Run Diagnosis button not found"
            run_buttons[0].click()
            app.run()

            summary_buttons = [
                b for b in app.button if "Load Summary for Pneumonia" in b.label
            ]
            assert summary_buttons, "Summary button for Pneumonia not found"
            summary_buttons[0].click()
            app.run()

        assert len(app.error) > 0
        error_text = " ".join(e.value for e in app.error)
        assert "LLM summary failed" in error_text
