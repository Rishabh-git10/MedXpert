"""Unit and integration tests for app.py business logic.

Validates API calls, session management, findings filtering, heatmap handling,
formatting utilities, and integrated diagnostic-to-summary workflows without
requiring full Streamlit UI execution.
"""

import base64
from io import BytesIO
from unittest.mock import Mock, patch

import pytest
from PIL import Image


@pytest.mark.unit
class TestAPICallFunctions:
    """Tests for MedXpert API request helper functions."""

    @patch("src.app.requests.post")
    def test_call_diagnose_api_success(self, mock_post):
        """Return success when /diagnose/ responds with findings."""
        from src.app import call_diagnose_api

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "findings": [{"name": "Pneumonia", "probability": 0.85}],
            "heatmaps": {},
        }
        mock_post.return_value = mock_response

        result = call_diagnose_api(b"fake_image_data", "test.jpg", "image/jpeg")

        assert result["success"] is True
        assert "findings" in result["data"]
        assert result["data"]["findings"][0]["name"] == "Pneumonia"

    @patch("src.app.requests.post")
    def test_call_diagnose_api_failure(self, mock_post):
        """Return error when /diagnose/ API responds with failure."""
        from src.app import call_diagnose_api

        mock_response = Mock(status_code=500, text="Internal Server Error")
        mock_post.return_value = mock_response

        result = call_diagnose_api(b"fake_image_data", "test.jpg", "image/jpeg")

        assert result["success"] is False
        assert result["error"] == "Internal Server Error"

    @patch("src.app.requests.post")
    def test_call_summary_api_success(self, mock_post):
        """Return success when /get_summary/ responds with valid summary."""
        from src.app import call_summary_api

        mock_response = Mock(status_code=200)
        mock_response.json.return_value = {"summary": "Clinical information"}
        mock_post.return_value = mock_response

        result = call_summary_api("Pneumonia")

        assert result["success"] is True
        assert result["summary"] == "Clinical information"

    @patch("src.app.requests.post")
    def test_call_summary_api_failure(self, mock_post):
        """Return error when /get_summary/ API fails."""
        from src.app import call_summary_api

        mock_response = Mock(status_code=500, text="Server Error")
        mock_post.return_value = mock_response

        result = call_summary_api("Pneumonia")

        assert result["success"] is False
        assert result["error"] == "Server Error"


@pytest.mark.unit
class TestStateManagement:
    """Tests for session state handling and reset logic."""

    def test_should_reset_state_different_files(self):
        """Reset state when uploaded file IDs differ."""
        from src.app import should_reset_state
        assert should_reset_state("file_123", "file_456") is True

    def test_should_reset_state_same_file(self):
        """Do not reset state for same uploaded file."""
        from src.app import should_reset_state
        assert should_reset_state("file_123", "file_123") is False

    def test_should_reset_state_first_file(self):
        """Reset state on first file upload."""
        from src.app import should_reset_state
        assert should_reset_state("file_123", None) is True


@pytest.mark.unit
class TestFindingsLogic:
    """Tests for findings filtering and normal result detection."""

    def test_filter_findings_removes_no_finding_when_others_exist(self):
        """Remove 'No Finding' when other findings are present."""
        from src.app import filter_findings
        findings = [
            {"name": "Pneumonia", "probability": 0.85},
            {"name": "No Finding", "probability": 0.15},
        ]
        result = filter_findings(findings)
        assert len(result) == 1 and result[0]["name"] == "Pneumonia"

    def test_filter_findings_keeps_no_finding_alone(self):
        """Keep 'No Finding' when it's the only finding."""
        from src.app import filter_findings
        result = filter_findings([{"name": "No Finding", "probability": 0.92}])
        assert len(result) == 1 and result[0]["name"] == "No Finding"

    def test_is_normal_result_true(self):
        """Return True for single 'No Finding' result."""
        from src.app import is_normal_result
        findings = [{"name": "No Finding", "probability": 0.92}]
        assert is_normal_result(findings) is True

    def test_is_normal_result_false_with_findings(self):
        """Return False when abnormal findings exist."""
        from src.app import is_normal_result
        findings = [
            {"name": "Pneumonia", "probability": 0.85},
            {"name": "Edema", "probability": 0.72},
        ]
        assert is_normal_result(findings) is False


@pytest.mark.unit
class TestHeatmapLogic:
    """Tests for heatmap decoding and tab title logic."""

    def test_get_heatmap_tab_titles(self):
        """Return correct tab order: Original + pathology names."""
        from src.app import get_heatmap_tab_titles
        titles = get_heatmap_tab_titles({"Pneumonia": "data1", "Edema": "data2"})
        assert titles == ["Original", "Pneumonia", "Edema"]

    def test_decode_heatmap(self):
        """Decode valid base64 image string to PIL Image."""
        from src.app import decode_heatmap
        img = Image.new("RGB", (50, 50), color="blue")
        buf = BytesIO()
        img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode()
        decoded = decode_heatmap(b64)
        assert isinstance(decoded, Image.Image)
        assert decoded.size == (50, 50)


@pytest.mark.unit
class TestFormatting:
    """Tests for probability formatting and HTML card generation."""

    def test_format_probability(self):
        """Return percentage string formatted to one decimal place."""
        from src.app import format_probability
        assert format_probability(0.85) == "85.0%"
        assert format_probability(0.1) == "10.0%"

    def test_get_finding_card_html_normal(self):
        """Generate correct HTML for normal findings."""
        from src.app import get_finding_card_html
        html = get_finding_card_html({"name": "No Finding", "probability": 0.92}, True)
        assert "✅ No Significant Findings" in html and "92.0%" in html

    def test_get_finding_card_html_abnormal(self):
        """Generate correct HTML for abnormal findings."""
        from src.app import get_finding_card_html
        html = get_finding_card_html({"name": "Pneumonia", "probability": 0.85})
        assert "Pneumonia" in html and "85.0%" in html


@pytest.mark.integration
class TestIntegratedLogic:
    """Integration tests combining diagnosis and summary logic."""

    @patch("src.app.requests.post")
    def test_complete_diagnosis_flow(self, mock_post):
        """Simulate end-to-end diagnosis flow with abnormal findings."""
        from src.app import call_diagnose_api, filter_findings, should_show_summaries

        mock_response = Mock(status_code=200)
        mock_response.json.return_value = {
            "findings": [
                {"name": "Pneumonia", "probability": 0.85},
                {"name": "No Finding", "probability": 0.15},
            ],
            "heatmaps": {"Pneumonia": "base64data"},
        }
        mock_post.return_value = mock_response

        result = call_diagnose_api(b"img", "test.jpg", "image/jpeg")
        findings = filter_findings(result["data"]["findings"])

        assert result["success"] and len(findings) == 1
        assert findings[0]["name"] == "Pneumonia"
        assert should_show_summaries(findings) is True

    @patch("src.app.requests.post")
    def test_normal_result_flow(self, mock_post):
        """Simulate complete flow for a normal result."""
        from src.app import call_diagnose_api, filter_findings, is_normal_result, should_show_summaries

        mock_response = Mock(status_code=200)
        mock_response.json.return_value = {
            "findings": [{"name": "No Finding", "probability": 0.92}],
            "heatmaps": {},
        }
        mock_post.return_value = mock_response

        result = call_diagnose_api(b"img", "test.jpg", "image/jpeg")
        findings = filter_findings(result["data"]["findings"])

        assert is_normal_result(findings)
        assert should_show_summaries(findings) is False
