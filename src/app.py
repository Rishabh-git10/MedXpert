"""
MedXpert Streamlit Application
==============================

Interactive frontend for the MedXpert system — an explainable AI diagnostic
assistant for chest X-rays.

The Streamlit interface communicates with the FastAPI backend to:
- Upload and analyze chest X-rays
- Display multi-pathology AI findings
- Visualize Grad-CAM heatmaps for interpretability
- Retrieve concise clinical summaries via RAG + Gemini

Design Notes
------------
- All business logic is decoupled from Streamlit rendering for testability.
- API endpoints are defined in the backend (`/diagnose/` and `/get_summary/`).
- Streamlit session state preserves findings, summaries, and UI state.
"""

import base64
import os
from io import BytesIO
from typing import Dict, List, Optional

import requests
import streamlit as st
from PIL import Image

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

# Default backend URL; can be overridden with environment variable.
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

# ---------------------------------------------------------------------
# Business Logic Functions (Testable)
# ---------------------------------------------------------------------

def call_diagnose_api(
    file_bytes: bytes, filename: str, file_type: str, api_url: str = API_URL
) -> Dict:
    """Call the `/diagnose/` API endpoint to perform image analysis.

    Args:
        file_bytes (bytes): Raw bytes of the uploaded image.
        filename (str): Name of the uploaded file.
        file_type (str): MIME type (e.g., 'image/jpeg').
        api_url (str, optional): API base URL.

    Returns:
        Dict: Response dict with either:
            - {"success": True, "data": {...}} on success
            - {"success": False, "error": "..."} on failure
    """
    multipart_data = {"file": (filename, file_bytes, file_type)}
    response = requests.post(f"{api_url}/diagnose/", files=multipart_data)

    if response.status_code == 200:
        return {"success": True, "data": response.json()}
    return {"success": False, "error": response.text}


def call_summary_api(pathology: str, api_url: str = API_URL) -> Dict:
    """Call the `/get_summary/` API endpoint to retrieve a clinical summary.

    Args:
        pathology (str): Pathology name (e.g., "Pneumonia").
        api_url (str, optional): API base URL.

    Returns:
        Dict: Response dict with either:
            - {"success": True, "summary": "..."} on success
            - {"success": False, "error": "..."} on failure
    """
    response = requests.post(f"{api_url}/get_summary/", json={"pathology": pathology})

    if response.status_code == 200:
        return {
            "success": True,
            "summary": response.json().get("summary", "No summary found."),
        }
    return {"success": False, "error": response.text}


def should_reset_state(current_file_id: str, last_file_id: Optional[str]) -> bool:
    """Check if a new file was uploaded (for session reset).

    Args:
        current_file_id (str): Unique identifier for the current upload.
        last_file_id (Optional[str]): File ID from previous session.

    Returns:
        bool: True if the uploaded file differs from the previous one.
    """
    return last_file_id != current_file_id


def filter_findings(findings: List[Dict]) -> List[Dict]:
    """Filter 'No Finding' if other pathologies exist.

    Args:
        findings (List[Dict]): List of model findings.

    Returns:
        List[Dict]: Cleaned findings list with 'No Finding' removed if redundant.
    """
    if not findings:
        return []
    has_other_findings = any(f["name"] != "No Finding" for f in findings)
    return [f for f in findings if f["name"] != "No Finding"] if has_other_findings else findings


def is_normal_result(findings: List[Dict]) -> bool:
    """Check if findings correspond to a normal (no pathology) result.

    Args:
        findings (List[Dict]): List of model findings.

    Returns:
        bool: True if the only finding is 'No Finding'.
    """
    return len(findings) == 1 and findings[0]["name"] == "No Finding"


def get_heatmap_tab_titles(heatmaps: Dict) -> List[str]:
    """Generate tab titles for Grad-CAM heatmap display.

    Args:
        heatmaps (Dict): Dictionary of pathology name → base64 image.

    Returns:
        List[str]: List of tab titles, starting with 'Original'.
    """
    return ["Original"] + list(heatmaps.keys())


def decode_heatmap(base64_string: str) -> Image.Image:
    """Convert a base64-encoded heatmap string into a PIL image.

    Args:
        base64_string (str): Base64-encoded JPEG image.

    Returns:
        Image.Image: Decoded PIL image for display.
    """
    heatmap_bytes = base64.b64decode(base64_string)
    return Image.open(BytesIO(heatmap_bytes))


def should_show_summaries(findings: Optional[List[Dict]]) -> bool:
    """Determine whether to show the summary section.

    Args:
        findings (Optional[List[Dict]]): Current findings list.

    Returns:
        bool: True if at least one non-normal finding exists.
    """
    return bool(findings) and not is_normal_result(findings)


def format_probability(prob: float) -> str:
    """Convert probability to percentage string.

    Args:
        prob (float): Probability value between 0 and 1.

    Returns:
        str: Formatted percentage string, e.g. "78.4%".
    """
    return f"{prob * 100:.1f}%"


def get_finding_card_html(finding: Dict, is_normal: bool = False) -> str:
    """Generate HTML snippet for a finding card.

    Args:
        finding (Dict): Finding entry with 'name' and 'probability'.
        is_normal (bool, optional): If True, uses special normal-state style.

    Returns:
        str: HTML markup for the card.
    """
    card_class = "finding-card finding-normal" if is_normal else "finding-card"
    if is_normal:
        return f"""
        <div class="{card_class}">
            <h3>✅ No Significant Findings</h3>
            <span class="prob-badge">Confidence: {format_probability(finding['probability'])}</span>
        </div>
        """
    return f"""
        <div class="{card_class}">
            <h4 style="margin:0;">{finding['name']}</h4>
            <span class="prob-badge">{format_probability(finding['probability'])}</span>
        </div>
        """

# ---------------------------------------------------------------------
# UI Rendering (Streamlit-specific)
# ---------------------------------------------------------------------

# Custom CSS styling for consistent branding and layout
st.markdown(
    """
    <style>
        .main-header { text-align:center; font-size:2.2rem; font-weight:700; margin-bottom:0.5rem; }
        .subtitle { text-align:center; font-size:1.1rem; color:gray; margin-bottom:2rem; }
        .finding-card { border-radius:1rem; padding:1rem; margin:0.5rem 0; text-align:center;
                        box-shadow:0 2px 6px rgba(0,0,0,0.1); background:#f9fafb; }
        .finding-normal { background:#eafbea; }
        .prob-badge { background:#2563eb; color:white; border-radius:0.5rem;
                      padding:0.2rem 0.5rem; font-size:0.85rem; }
        .summary-box { background:#f3f4f6; padding:1rem; border-radius:1rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# Application header and subtitle
st.markdown('<h1 class="main-header">🩺 MedXpert</h1>', unsafe_allow_html=True)
st.markdown(
    '<p class="subtitle">AI-Powered Chest X-ray Diagnostic Assistant with Explainable AI</p>',
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------
# Session State Initialization
# ---------------------------------------------------------------------

# Initialize all necessary Streamlit state variables
for key, default in {
    "findings": None,
    "heatmaps": None,
    "original_image": None,
    "summaries": {},
    "last_file_id": None,
    "expanded_finding": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ---------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------

with st.sidebar:
    st.markdown("### 📋 Instructions")
    st.markdown(
        """
        1. Upload a chest X-ray image  
        2. Click **Run Diagnosis**  
        3. Explore AI findings and heatmaps  
        4. View summaries for each finding  
        """
    )
    st.markdown("---")
    st.markdown("### ℹ️ About")
    st.markdown(
        """
        This system integrates:
        - **DenseNet121** for pathology classification  
        - **Grad-CAM** for explainability  
        - **RAG** for medical knowledge retrieval  
        - **Gemini AI** for concise summaries  
        """
    )

# ---------------------------------------------------------------------
# File Upload and Diagnosis Flow
# ---------------------------------------------------------------------

uploaded_file = st.file_uploader("📤 Upload Chest X-ray", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    current_file_id = uploaded_file.file_id

    # Reset session state when a new file is uploaded
    if should_reset_state(current_file_id, st.session_state.last_file_id):
        st.session_state.findings = None
        st.session_state.heatmaps = None
        st.session_state.summaries = {}
        st.session_state.last_file_id = current_file_id
        st.session_state.original_image = Image.open(uploaded_file)
        st.session_state.expanded_finding = None

    st.image(st.session_state.original_image, caption="📷 Uploaded X-ray", width=400)

    # -----------------------------------------------------------------
    # Run Diagnosis
    # -----------------------------------------------------------------

    if st.button("🔬 Run Diagnosis", type="primary"):
        with st.spinner("🧠 AI is analyzing the X-ray..."):
            uploaded_file.seek(0)
            try:
                result = call_diagnose_api(
                    uploaded_file.getvalue(), uploaded_file.name, uploaded_file.type
                )
                if result["success"]:
                    data = result["data"]
                    st.session_state.findings = filter_findings(data.get("findings", []))
                    st.session_state.heatmaps = data.get("heatmaps", {})
                    st.session_state.summaries = {}
                else:
                    st.error(f"❌ API Error: {result['error']}")
            except Exception as e:
                st.error(f"❌ Failed to connect to API. Ensure it's running. Error: {e}")

    # -----------------------------------------------------------------
    # Display Diagnostic Findings
    # -----------------------------------------------------------------

    if st.session_state.findings:
        st.markdown("---")
        st.markdown("## 🔬 Diagnostic Results")

        findings = st.session_state.findings
        if is_normal_result(findings):
            st.markdown(
                get_finding_card_html(findings[0], is_normal=True),
                unsafe_allow_html=True,
            )
        else:
            cols = st.columns(min(len(findings), 3))
            for idx, finding in enumerate(findings):
                with cols[idx % 3]:
                    st.markdown(get_finding_card_html(finding), unsafe_allow_html=True)

    # -----------------------------------------------------------------
    # Display Heatmaps (Explainability)
    # -----------------------------------------------------------------

    if st.session_state.heatmaps:
        st.markdown("---")
        st.markdown("### 💡 AI Explainability (Grad-CAM Heatmaps)")
        st.markdown("*Highlighted regions show areas the AI model focused on.*")

        tab_titles = get_heatmap_tab_titles(st.session_state.heatmaps)
        tabs = st.tabs(tab_titles)

        # Original image
        with tabs[0]:
            st.image(st.session_state.original_image, caption="Original X-ray")

        # Grad-CAM overlays
        for idx, (finding_name, heatmap_b64) in enumerate(
            st.session_state.heatmaps.items(), 1
        ):
            with tabs[idx]:
                heatmap_image = decode_heatmap(heatmap_b64)
                st.image(heatmap_image, caption=f"AI Focus: {finding_name}")

    # -----------------------------------------------------------------
    # Medical Knowledge Summaries
    # -----------------------------------------------------------------

    if should_show_summaries(st.session_state.findings):
        st.markdown("---")
        st.markdown("### 📚 Medical Knowledge Base")

        for finding in st.session_state.findings:
            name = finding["name"]
            expanded = st.session_state.expanded_finding == name
            with st.expander(f"📖 {name} - Clinical Summary", expanded=expanded):
                if name not in st.session_state.summaries:
                    if st.button(f"Load Summary for {name}", key=f"btn_{name}"):
                        st.session_state.expanded_finding = name
                        with st.spinner("Retrieving medical information..."):
                            try:
                                result = call_summary_api(name)
                                if result["success"]:
                                    st.session_state.summaries[name] = result["summary"]
                                    st.rerun()
                                else:
                                    st.error(f"Error: {result['error']}")
                            except Exception as e:
                                st.error(f"Failed to retrieve summary: {e}")
                else:
                    st.markdown('<div class="summary-box">', unsafe_allow_html=True)
                    st.markdown(st.session_state.summaries[name])
                    st.markdown("</div>", unsafe_allow_html=True)
