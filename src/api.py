"""
MedXpert API
============

FastAPI backend for the MedXpert system — an explainable AI (XAI) diagnostic
assistant for chest X-rays.

Endpoints
---------
- POST /diagnose/     : Multi-stage AI diagnosis with optional Grad-CAM heatmaps
- POST /get_summary/  : Context-aware medical summary via RAG + Gemini
- GET  /              : Health/status check

Notes
-----
- Models are loaded at import time to keep request latency low.
- The RAG system uses a local Chroma vector store; ensure the persisted DB exists.
- All generation relies on environment configuration (e.g., GEMINI_API_KEY).
"""

import base64
import os
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import timm
import torch
import uvicorn
from albumentations.pytorch import ToTensorV2
from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from pydantic import BaseModel

# ---------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------

load_dotenv()
print("--- MedXpert API Starting Up ---")

device = torch.device("cpu")
print(f"Using device: {device}")

PROJECT_ROOT = Path(__file__).parent.parent
FRONTAL_MODEL_PATH = PROJECT_ROOT / "models" / "frontal_model_best.pth"
LATERAL_MODEL_PATH = PROJECT_ROOT / "models" / "lateral_model_best.pth"
VIEW_MODEL_PATH = PROJECT_ROOT / "models" / "view_classifier.pth"
DB_PATH = PROJECT_ROOT / "chroma_db"

# Inference and visualization configuration
IMAGE_SIZE = 320
pathology_cols = [
    "No Finding",
    "Enlarged Cardiomediastinum",
    "Cardiomegaly",
    "Lung Opacity",
    "Lung Lesion",
    "Edema",
    "Consolidation",
    "Pneumonia",
    "Atelectasis",
    "Pneumothorax",
    "Pleural Effusion",
    "Pleural Other",
    "Fracture",
    "Support Devices",
]


# ---------------------------------------------------------------------
# Model Loading
# ---------------------------------------------------------------------

def create_model_architecture():
    """Create and load a DenseNet-121 classifier for 14-pathology prediction.

    Returns:
        torch.nn.Module: DenseNet-121 model in eval mode with loaded weights.
    """
    model = timm.create_model("densenet121", pretrained=False, num_classes=14)
    model.load_state_dict(
        torch.load(FRONTAL_MODEL_PATH, map_location=device, weights_only=True)
    )
    model.to(device).eval()
    return model


def create_view_classifier():
    """Create and load a MobileNetV3 view classifier (frontal/lateral).

    Returns:
        torch.nn.Module: MobileNetV3 model in eval mode with loaded weights.
    """
    model = timm.create_model("mobilenetv3_small_100", pretrained=False, num_classes=2)
    model.load_state_dict(
        torch.load(VIEW_MODEL_PATH, map_location=device, weights_only=True)
    )
    model.to(device).eval()
    return model


print("Loading models...")
model_frontal = create_model_architecture()
model_lateral = create_model_architecture()
model_lateral.load_state_dict(
    torch.load(LATERAL_MODEL_PATH, map_location=device, weights_only=True)
)
model_view = create_view_classifier()
print("All models loaded successfully.")

# ---------------------------------------------------------------------
# Image Preprocessing
# ---------------------------------------------------------------------

# Pathology model transforms (match training-time normalization/resolution)
img_transforms = A.Compose(
    [
        A.Resize(height=IMAGE_SIZE, width=IMAGE_SIZE),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ]
)

# View classifier transforms (fixed to 224 for MobileNetV3)
view_transforms = A.Compose(
    [
        A.Resize(height=224, width=224),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ]
)

# ---------------------------------------------------------------------
# RAG System Initialization
# ---------------------------------------------------------------------

embedding_model = HuggingFaceEmbeddings(model_name="all-mpnet-base-v2")
vectordb = Chroma(persist_directory=DB_PATH, embedding_function=embedding_model)

if not os.environ.get("GEMINI_API_KEY"):
    print("WARNING: GEMINI_API_KEY not found.")

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-pro",
    google_api_key=os.environ.get("GEMINI_API_KEY"),
    temperature=0.2,
)

prompt = PromptTemplate(
    template=(
        "You are a concise medical expert. A chest X-ray model has identified a finding.\n"
        "Provide a short, clinically useful explanation based on this context only.\n\n"
        "CONTEXT: {context}\n"
        "QUESTION: {question}\n"
        "SUMMARY:"
    ),
    input_variables=["context", "question"],
)

qa_chain = (
    {
        "context": vectordb.as_retriever()
        | (lambda docs: "\n\n".join(doc.page_content for doc in docs)),
        "question": RunnablePassthrough(),
    }
    | prompt
    | llm
    | StrOutputParser()
)

# ---------------------------------------------------------------------
# Grad-CAM Implementation
# ---------------------------------------------------------------------

class GradCAM:
    """Generate Grad-CAM heatmaps for feature attribution on CNNs.

    This class registers forward and backward hooks on a target layer to
    capture activations and gradients, then computes class-specific
    attribution maps (CAMs) for visualization.
    """

    def __init__(self, model, target_layer):
        """Init GradCAM.

        Args:
            model (torch.nn.Module): Model to explain (expects conv features).
            target_layer (torch.nn.Module): Layer whose activations/gradients
                are used to compute CAMs (e.g., DenseNet block).
        """
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

    def forward_hook(self, _, __, output):
        """Capture forward activations.

        Args:
            _ : Unused module reference (signature required by PyTorch).
            __: Unused input tuple (signature required by PyTorch).
            output (torch.Tensor): Layer output tensor saved for CAM.
        """
        self.activations = output.detach()

    def backward_hook(self, _, __, grad_output):
        """Capture gradients during backprop.

        Args:
            _ : Unused module reference.
            __: Unused grad_input tuple.
            grad_output (Tuple[torch.Tensor, ...]): Gradients wrt layer output.
        """
        self.gradients = grad_output[0].detach()

    def generate_cam(self, image_tensor, target_class):
        """Compute a normalized Grad-CAM heatmap for a target class index.

        The model is temporarily set to train mode to enable gradient flow
        without altering any weights (no optimizer steps are performed).

        Args:
            image_tensor (torch.Tensor): Preprocessed input tensor of shape
                (1, C, H, W).
            target_class (int): Class index for which attribution is computed.

        Returns:
            np.ndarray: CAM array in [0, 1] of shape (H', W') from the target
            layer’s spatial resolution. If hooks fail, returns zeros.
        """
        forward_handle = self.target_layer.register_forward_hook(self.forward_hook)
        backward_handle = self.target_layer.register_full_backward_hook(
            self.backward_hook
        )

        self.model.train()
        self.model.zero_grad()
        output = self.model(image_tensor)
        score = output[0, target_class]
        score.backward(retain_graph=True)

        forward_handle.remove()
        backward_handle.remove()

        if self.gradients is None or self.activations is None:
            return np.zeros((7, 7))

        # Global-average the gradients over spatial dims, then weight activations
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * self.activations, dim=1).squeeze(0)
        cam = torch.relu(cam)

        cam_min, cam_max = cam.min(), cam.max()
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min)

        return cam.cpu().numpy()


# ---------------------------------------------------------------------
# Core Prediction Pipeline
# ---------------------------------------------------------------------

def predict_image(image_bytes: bytes):
    """Run the complete diagnostic pipeline on an uploaded image.

    Steps:
        1) Decode bytes → OpenCV image.
        2) Predict view (frontal/lateral) and select the corresponding model.
        3) Preprocess image for pathology model and run inference.
        4) Post-process probabilities into findings list.
        5) Optionally compute Grad-CAM heatmaps for top findings.

    Args:
        image_bytes (bytes): Raw image bytes from upload.

    Returns:
        Tuple[List[Dict[str, Any]], Dict[str, str]]:
            findings: List of detected findings (name, probability, index).
            heatmaps: Dict mapping finding name → base64-encoded JPEG heatmap.

    Raises:
        ValueError: If the uploaded data cannot be decoded as an image.
    """
    nparr = np.frombuffer(image_bytes, np.uint8)
    image_cv = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if image_cv is None:
        raise ValueError("Invalid image data")

    # 1) Detect image view and route to the appropriate classifier
    view_tensor = view_transforms(image=image_cv)["image"].unsqueeze(0).to(device)
    with torch.no_grad():
        view_pred = model_view(view_tensor)
        _, view_idx = torch.max(view_pred, 1)
    model = model_frontal if view_idx.item() == 0 else model_lateral

    # 2) Preprocess image for pathology model
    image_tensor = img_transforms(image=image_cv)["image"].unsqueeze(0).to(device)

    # 3) Pathology prediction (multi-label, sigmoid)
    with torch.no_grad():
        outputs = model(image_tensor)
    probs = torch.sigmoid(outputs).squeeze().cpu().numpy()

    # 4) Post-process probabilities into human-readable findings
    findings = [
        {"name": pathology_cols[i], "probability": float(p), "index": i}
        for i, p in enumerate(probs)
        if p > 0.5
    ]
    findings.sort(key=lambda x: x["probability"], reverse=True)

    # Handle "No Finding" logic conservatively
    if not findings and probs[0] > 0.1:
        findings = [{"name": "No Finding", "probability": float(probs[0]), "index": 0}]
    elif findings and any(f["name"] == "No Finding" for f in findings):
        findings = [f for f in findings if f["name"] != "No Finding"]

    # 5) Optional: generate Grad-CAM overlays for top findings
    heatmaps = {}
    if findings and findings[0]["name"] != "No Finding":
        grad_cam = GradCAM(model, model.features.denseblock4)
        for finding in findings[:5]:
            idx = finding["index"]
            cam = grad_cam.generate_cam(image_tensor, idx)
            cam = np.power(cam, 0.7)  # slight sharpening without altering semantics
            cam_resized = cv2.resize(cam, (IMAGE_SIZE, IMAGE_SIZE))
            heatmap = np.uint8(255 * cam_resized)
            overlay = cv2.addWeighted(
                cv2.resize(image_cv, (IMAGE_SIZE, IMAGE_SIZE)),
                0.4,
                cv2.applyColorMap(heatmap, cv2.COLORMAP_JET),
                0.6,
                0,
            )
            _, buffer = cv2.imencode(".jpg", overlay)
            heatmaps[finding["name"]] = base64.b64encode(buffer).decode("utf-8")

    return findings, heatmaps


# ---------------------------------------------------------------------
# FastAPI Endpoints
# ---------------------------------------------------------------------

app = FastAPI(title="MedXpert API")


class SummaryRequest(BaseModel):
    """Pydantic model for summary requests."""

    pathology: str


@app.post("/diagnose/")
async def diagnose(file: UploadFile = File(...)):
    """Run AI-based chest X-ray diagnosis.

    Args:
        file (UploadFile): Uploaded image file (JPEG/PNG).

    Returns:
        JSONResponse: Findings and optional heatmaps as base64 strings.

    Notes:
        - Heatmaps are returned only when relevant (no "No Finding" case).
        - Returns HTTP 500 with an error message on failure.
    """
    try:
        image_bytes = await file.read()
        findings, heatmaps = predict_image(image_bytes)
        return JSONResponse(content={"findings": findings, "heatmaps": heatmaps})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


@app.post("/get_summary/")
async def get_summary(request: SummaryRequest):
    """Return a concise, context-aware medical summary for a pathology.

    Args:
        request (SummaryRequest): Pathology term to summarize.

    Returns:
        JSONResponse: Generated text under the "summary" key, or an error.
    """
    try:
        query = (
            f"What is the overview, diagnosis, and management for {request.pathology}?"
        )
        response = qa_chain.invoke(query)
        return JSONResponse(content={"summary": response})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


@app.get("/")
def read_root():
    """Health/status check for the API."""
    return {"message": "MedXpert API is running."}


# ---------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run("src.api:app", host="127.0.0.1", port=8000, reload=True)
