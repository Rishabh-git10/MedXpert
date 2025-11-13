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

        mock_response = Mock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"
        mock_post.return_value = mock_response

        result = call_diagnose_api(b"fake_image_data", "test.jpg", "image/jpeg")

        assert result["success"] is False
        assert "error" in result
        assert result["error"] == "Internal Server Error"

    @patch("src.app.requests.post")
    def test_call_summary_api_success(self, mock_post):
        """Return success when /get_summary/ responds with valid summary."""
        from src.app import call_summary_api

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"summary": "Clinical information"}
        mock_post.return_value = mock_response

        result = call_summary_api("Pneumonia")

        assert result["success"] is True
        assert result["summary"] == "Clinical information"

    @patch("src.app.requests.post")
    def test_call_summary_api_failure(self, mock_post):
        """Return error when /get_summary/ API fails."""
        from src.app import call_summary_api

        mock_response = Mock()
        mock_response.status_code = 500
        mock_response.text = "Server Error"
        mock_post.return_value = mock_response

        result = call_summary_api("Pneumonia")

        assert result["success"] is False
        assert "error" in result
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

        assert len(result) == 1
        assert result[0]["name"] == "Pneumonia"

    def test_filter_findings_keeps_no_finding_alone(self):
        """Keep 'No Finding' when it's the only finding."""
        from src.app import filter_findings

        findings = [{"name": "No Finding", "probability": 0.92}]

        result = filter_findings(findings)

        assert len(result) == 1
        assert result[0]["name"] == "No Finding"

    def test_filter_findings_empty_list(self):
        """Test filtering empty findings list."""
        from src.app import filter_findings

        result = filter_findings([])

        assert result == []

    def test_filter_findings_multiple_pathologies(self):
        """Test filtering with multiple pathologies."""
        from src.app import filter_findings

        findings = [
            {"name": "Pneumonia", "probability": 0.85},
            {"name": "Edema", "probability": 0.72},
            {"name": "Consolidation", "probability": 0.68},
        ]

        result = filter_findings(findings)

        assert len(result) == 3
        assert all(f["name"] != "No Finding" for f in result)

    def test_is_normal_result_true(self):
        """Test identification of normal result."""
        from src.app import is_normal_result

        findings = [{"name": "No Finding", "probability": 0.92}]

        assert is_normal_result(findings) is True

    def test_is_normal_result_false_with_findings(self):
        """Test that results with abnormal findings aren't normal."""
        from src.app import is_normal_result

        findings = [
            {"name": "Pneumonia", "probability": 0.85},
            {"name": "Edema", "probability": 0.72},
        ]

        assert is_normal_result(findings) is False

    def test_is_normal_result_false_empty(self):
        """Test that empty findings aren't considered normal."""
        from src.app import is_normal_result

        assert is_normal_result([]) is False

    def test_is_normal_result_false_multiple_findings(self):
        """Test that multiple findings including 'No Finding' aren't normal."""
        from src.app import is_normal_result

        findings = [
            {"name": "No Finding", "probability": 0.5},
            {"name": "Pneumonia", "probability": 0.5},
        ]

        assert is_normal_result(findings) is False


@pytest.mark.unit
class TestHeatmapLogic:
    """Tests for heatmap decoding and tab title logic."""

    def test_get_heatmap_tab_titles(self):
        """Test generation of heatmap tab titles."""
        from src.app import get_heatmap_tab_titles

        heatmaps = {"Pneumonia": "base64data1", "Edema": "base64data2"}

        titles = get_heatmap_tab_titles(heatmaps)

        assert titles == ["Original", "Pneumonia", "Edema"]

    def test_get_heatmap_tab_titles_empty(self):
        """Test tab titles with no heatmaps."""
        from src.app import get_heatmap_tab_titles

        titles = get_heatmap_tab_titles({})

        assert titles == ["Original"]

    def test_get_heatmap_tab_titles_single(self):
        """Test tab titles with single heatmap."""
        from src.app import get_heatmap_tab_titles

        heatmaps = {"Pneumonia": "base64data"}
        titles = get_heatmap_tab_titles(heatmaps)

        assert len(titles) == 2
        assert titles[0] == "Original"
        assert titles[1] == "Pneumonia"

    def test_decode_heatmap(self):
        """Test decoding base64 heatmap to image."""
        from src.app import decode_heatmap

        img = Image.new("RGB", (100, 100), color="red")
        buf = BytesIO()
        img.save(buf, format="PNG")
        b64_string = base64.b64encode(buf.getvalue()).decode("utf-8")

        decoded_img = decode_heatmap(b64_string)

        assert isinstance(decoded_img, Image.Image)
        assert decoded_img.size == (100, 100)

    def test_decode_heatmap_different_formats(self):
        """Test decoding heatmaps in different formats."""
        from src.app import decode_heatmap

        for fmt in ["PNG", "JPEG"]:
            img = Image.new("RGB", (50, 50), color="blue")
            buf = BytesIO()
            img.save(buf, format=fmt)
            b64_string = base64.b64encode(buf.getvalue()).decode("utf-8")

            decoded_img = decode_heatmap(b64_string)
            assert isinstance(decoded_img, Image.Image)


@pytest.mark.unit
class TestSummaryLogic:
    """Test summary-related logic."""

    def test_should_show_summaries_with_findings(self):
        """Test that summaries show for abnormal findings."""
        from src.app import should_show_summaries

        findings = [
            {"name": "Pneumonia", "probability": 0.85},
            {"name": "Edema", "probability": 0.72},
        ]

        assert should_show_summaries(findings) is True

    def test_should_show_summaries_normal_result(self):
        """Test that summaries don't show for normal results."""
        from src.app import should_show_summaries

        findings = [{"name": "No Finding", "probability": 0.92}]

        assert should_show_summaries(findings) is False

    def test_should_show_summaries_none(self):
        """Test that summaries don't show when findings is None."""
        from src.app import should_show_summaries

        assert should_show_summaries(None) is False

    def test_should_show_summaries_empty(self):
        """Test that summaries don't show for empty findings."""
        from src.app import should_show_summaries

        assert should_show_summaries([]) is False

    def test_should_show_summaries_single_abnormal(self):
        """Test summaries show for single abnormal finding."""
        from src.app import should_show_summaries

        findings = [{"name": "Pneumonia", "probability": 0.85}]

        assert should_show_summaries(findings) is True


@pytest.mark.unit
class TestFormatting:
    """Test formatting functions."""

    def test_format_probability(self):
        """Test probability formatting."""
        from src.app import format_probability

        assert format_probability(0.85) == "85.0%"
        assert format_probability(0.923) == "92.3%"
        assert format_probability(0.1) == "10.0%"

    def test_format_probability_edge_values(self):
        """Test probability formatting with edge values."""
        from src.app import format_probability

        assert format_probability(0.0) == "0.0%"
        assert format_probability(1.0) == "100.0%"
        assert format_probability(0.999) == "99.9%"

    def test_format_probability_precision(self):
        """Test probability formatting precision."""
        from src.app import format_probability

        assert format_probability(0.8888) == "88.9%"
        assert format_probability(0.8881) == "88.8%"

    def test_get_finding_card_html_normal(self):
        """Test HTML generation for normal finding."""
        from src.app import get_finding_card_html

        finding = {"name": "No Finding", "probability": 0.92}
        html = get_finding_card_html(finding, is_normal=True)

        assert "finding-normal" in html
        assert "✅ No Significant Findings" in html
        assert "92.0%" in html
        assert "prob-badge" in html

    def test_get_finding_card_html_abnormal(self):
        """Test HTML generation for abnormal finding."""
        from src.app import get_finding_card_html

        finding = {"name": "Pneumonia", "probability": 0.85}
        html = get_finding_card_html(finding, is_normal=False)

        assert "Pneumonia" in html
        assert "85.0%" in html
        assert "finding-normal" not in html
        assert "finding-card" in html

    def test_get_finding_card_html_structure(self):
        """Test that HTML cards have proper structure."""
        from src.app import get_finding_card_html

        finding = {"name": "Edema", "probability": 0.75}
        html = get_finding_card_html(finding)

        assert "<div" in html
        assert "</div>" in html
        assert "prob-badge" in html


@pytest.mark.unit
class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_filter_findings_all_no_finding(self):
        """Test filtering when all findings are 'No Finding'."""
        from src.app import filter_findings

        findings = [
            {"name": "No Finding", "probability": 0.6},
            {"name": "No Finding", "probability": 0.4},
        ]

        result = filter_findings(findings)

        assert len(result) == 2

    def test_decode_heatmap_invalid_base64(self):
        """Test that invalid base64 raises appropriate error."""
        from src.app import decode_heatmap

        with pytest.raises(Exception):
            decode_heatmap("not_valid_base64!!!")

    def test_filter_findings_with_duplicates(self):
        """Test filtering with duplicate pathology names."""
        from src.app import filter_findings

        findings = [
            {"name": "Pneumonia", "probability": 0.85},
            {"name": "Pneumonia", "probability": 0.80},
            {"name": "No Finding", "probability": 0.15},
        ]

        result = filter_findings(findings)

        assert len(result) == 2
        assert all(f["name"] == "Pneumonia" for f in result)

    def test_should_reset_state_with_empty_string(self):
        """Test state reset with empty string file ID."""
        from src.app import should_reset_state

        assert should_reset_state("", None) is True
        assert should_reset_state("", "") is False

    def test_get_heatmap_tab_titles_maintains_order(self):
        """Test that heatmap tab titles maintain dictionary order."""
        from src.app import get_heatmap_tab_titles

        heatmaps = {"Pneumonia": "data1", "Edema": "data2", "Consolidation": "data3"}

        titles = get_heatmap_tab_titles(heatmaps)

        assert titles[1] == "Pneumonia"
        assert titles[2] == "Edema"
        assert titles[3] == "Consolidation"


@pytest.mark.integration
class TestIntegratedLogic:
    """Test integrated logic flows."""

    @patch("src.app.requests.post")
    def test_complete_diagnosis_flow(self, mock_post):
        """Test complete diagnosis logic flow."""
        from src.app import (call_diagnose_api, filter_findings,
                             should_show_summaries)

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "findings": [
                {"name": "Pneumonia", "probability": 0.85},
                {"name": "No Finding", "probability": 0.15},
            ],
            "heatmaps": {"Pneumonia": "base64data"},
        }
        mock_post.return_value = mock_response

        result = call_diagnose_api(b"image_data", "test.jpg", "image/jpeg")
        assert result["success"] is True

        findings = filter_findings(result["data"]["findings"])
        assert len(findings) == 1
        assert findings[0]["name"] == "Pneumonia"

        assert should_show_summaries(findings) is True

    @patch("src.app.requests.post")
    def test_normal_result_flow(self, mock_post):
        """Test flow for normal/no-finding result."""
        from src.app import (call_diagnose_api, filter_findings,
                             is_normal_result, should_show_summaries)

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "findings": [{"name": "No Finding", "probability": 0.92}],
            "heatmaps": {},
        }
        mock_post.return_value = mock_response

        result = call_diagnose_api(b"image_data", "test.jpg", "image/jpeg")

        findings = filter_findings(result["data"]["findings"])

        assert is_normal_result(findings) is True

        assert should_show_summaries(findings) is False

    @patch("src.app.requests.post")
    def test_multiple_findings_with_heatmaps_flow(self, mock_post):
        """Test flow with multiple findings and heatmaps."""
        from src.app import (call_diagnose_api, filter_findings,
                             format_probability, get_heatmap_tab_titles,
                             should_show_summaries)

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "findings": [
                {"name": "Pneumonia", "probability": 0.85},
                {"name": "Edema", "probability": 0.72},
                {"name": "Consolidation", "probability": 0.68},
            ],
            "heatmaps": {
                "Pneumonia": "base64_1",
                "Edema": "base64_2",
                "Consolidation": "base64_3",
            },
        }
        mock_post.return_value = mock_response

        result = call_diagnose_api(b"image_data", "test.jpg", "image/jpeg")
        assert result["success"] is True

        findings = filter_findings(result["data"]["findings"])
        assert len(findings) == 3

        heatmaps = result["data"]["heatmaps"]
        tab_titles = get_heatmap_tab_titles(heatmaps)
        assert len(tab_titles) == 4

        assert should_show_summaries(findings) is True

        for finding in findings:
            prob_str = format_probability(finding["probability"])
            assert "%" in prob_str

    @patch("src.app.requests.post")
    def test_api_error_handling_flow(self, mock_post):
        """Test error handling in integrated flow."""
        from src.app import call_diagnose_api

        mock_response = Mock()
        mock_response.status_code = 500
        mock_response.text = "Server Error"
        mock_post.return_value = mock_response

        result = call_diagnose_api(b"image_data", "test.jpg", "image/jpeg")

        assert result["success"] is False
        assert "error" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--cov=src.app", "--cov-report=html"])
