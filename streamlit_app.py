# app.py
import os
import io
import time
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn.functional as F
from PIL import Image
import cv2

import streamlit as st

# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Organoid Segmentation", layout="wide")

# ──────────────────────────────────────────────────────────────────────────────
# Utility functions
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model(model_path: str, device: str = "cpu"):
    """
    Load a PyTorch segmentation model.
    Adjust this to match how you saved your model.
    """
    mdl = torch.load(model_path, map_location=device)
    mdl.eval()
    return mdl, device

def pil_to_tensor(img: Image.Image) -> torch.Tensor:
    arr = np.array(img.convert("RGB"))  # H, W, 3
    t = torch.from_numpy(arr).float() / 255.0  # [0,1]
    t = t.permute(2, 0, 1).unsqueeze(0)        # 1,3,H,W
    return t

def tensor_to_mask(prob: torch.Tensor, threshold: float = 0.5) -> np.ndarray:
    """
    Convert model output to a binary mask. 
    Modify this if your model outputs multi-class masks:
    - For multi-class (C,H,W) logits: mask = probs.argmax(0)
    - For sigmoid map: mask = (prob > threshold)
    """
    if prob.ndim == 4 and prob.shape[1] == 1:
        # shape: 1,1,H,W (sigmoid)
        probs = torch.sigmoid(prob)[0, 0].cpu().numpy()
        mask = (probs >= threshold).astype(np.uint8)
    elif prob.ndim == 4 and prob.shape[1] > 1:
        # softmax multi-class
        probs = F.softmax(prob, dim=1)[0]                    # C,H,W
        mask = probs.argmax(0).cpu().numpy().astype(np.uint8)  # 0..C-1
    else:
        raise ValueError("Unexpected model output shape")
    return mask

def overlay_mask_on_image(image: np.ndarray, mask: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """
    Overlay a binary or multi-class mask on an RGB image.
    For multi-class masks, each class is given a distinct color.
    """
    h, w, _ = image.shape
    if mask.ndim == 2:
        mask_vis = np.zeros((h, w, 3), dtype=np.uint8)
        # binary: color = green
        mask_vis[mask == 1] = (0, 255, 0)
    else:
        raise ValueError("Mask must be 2D (H,W)")

    blended = cv2.addWeighted(image, 1.0, mask_vis, alpha, 0.0)
    return blended

def compute_basic_stats(mask: np.ndarray, pixel_size_um: float | None = None) -> dict:
    """
    Extract simple, useful metrics.
    - area_px: number of mask pixels == 1
    - area_um2 (optional): area in µm² if pixel size is provided
    - components: number of connected components
    """
    area_px = int(mask.sum())
    if pixel_size_um is not None:
        area_um2 = float(area_px * (pixel_size_um ** 2))
    else:
        area_um2 = None

    # Connected components (binary)
    num_labels, _ = cv2.connectedComponents(mask.astype(np.uint8))
    components = int(num_labels - 1)  # exclude background

    return {
        "area_px": area_px,
        "area_um2": area_um2,
        "components": components,
    }

def predict_mask(model, device: str, img: Image.Image):
    """
    Run the model on the image and return a 2D mask (H,W).
    Modify to match your model’s expected input pipeline.
    """
    with torch.no_grad():
        t = pil_to_tensor(img).to(device)
        out = model(t)            # expected shapes: (1,1,H,W) or (1,C,H,W)
    mask = tensor_to_mask(out)
    return mask

# ──────────────────────────────────────────────────────────────────────────────
# Sidebar controls
# ──────────────────────────────────────────────────────────────────────────────
st.sidebar.header("Settings")
model_path = st.sidebar.text_input(
    "Model path (.pt or .pth)", 
    value="/content/drive/MyDrive/Segmenting_Brain_Organoids/organoid_segmentation_model_pytorch_attention.pt"
)
device_choice = st.sidebar.selectbox("Device", ["cpu", "cuda"], index=0)
threshold = st.sidebar.slider("Mask threshold (if binary)", 0.0, 1.0, 0.5, 0.01)
pixel_size_um = st.sidebar.number_input("Pixel size (µm/pixel) – optional", value=0.0, min_value=0.0, step=0.01)
gallery_cols = st.sidebar.slider("Gallery columns", 2, 6, 4)

# ──────────────────────────────────────────────────────────────────────────────
# Main UI
# ──────────────────────────────────────────────────────────────────────────────
st.title("Organoid Segmentation – Web Uploader")

st.markdown(
    """
    Upload one or more images. The app will segment each image, show a gallery, 
    and generate a metrics table you can download as CSV.
    """
)

# File uploader
files = st.file_uploader(
    "Upload images (PNG/JPG/TIF)", type=["png", "jpg", "jpeg", "tif", "tiff"], accept_multiple_files=True
)

# Load model only when needed
model = None
if files and model_path:
    if not os.path.exists(model_path):
        st.error("Model file not found. Please check the path in the sidebar.")
    else:
        # safety if user chooses cuda but it is unavailable
        device = "cuda" if (device_choice == "cuda" and torch.cuda.is_available()) else "cpu"
        model, device = load_model(model_path, device)
        st.success(f"Model loaded on {device.upper()}")

# Process
results = []
if files and model is not None:
    st.subheader("Results")
    start_all = time.time()

    # Gallery
    cols = st.columns(gallery_cols, gap="small")
    i = 0

    for up in files:
        # Read
        image_bytes = up.read()
        pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img_np = np.array(pil)

        # Predict
        t0 = time.time()
        mask = predict_mask(model, device, pil)
        runtime_s = time.time() - t0

        # Postprocess + overlay
        overlay = overlay_mask_on_image(img_np, mask, alpha=0.45)

        # Compute stats
        px_size = pixel_size_um if pixel_size_um > 0 else None
        stats = compute_basic_stats(mask, pixel_size_um=px_size)

        # Save into results
        row = {
            "filename": up.name,
            "height_px": img_np.shape[0],
            "width_px": img_np.shape[1],
            "area_px": stats["area_px"],
            "area_um2": stats["area_um2"],
            "components": stats["components"],
            "runtime_s": round(runtime_s, 4),
        }
        results.append(row)

        # Show images side-by-side: original and overlay
        with cols[i % gallery_cols]:
            st.caption(up.name)
            st.image(img_np, caption="Original", use_column_width=True)
            st.image(overlay, caption="Segmentation overlay", use_column_width=True)
        i += 1

    total_time = time.time() - start_all
    st.info(f"Processed {len(files)} image(s) in {total_time:.2f} s")

# Metrics table + download
if results:
    st.subheader("Per-image metrics")
    df = pd.DataFrame(results)
    st.dataframe(df, use_container_width=True)

    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download CSV", data=csv, file_name="segmentation_metrics.csv", mime="text/csv"
    )

# Footer
st.markdown("---")
st.markdown(
    "Tip: if your model outputs multi-class masks, change `tensor_to_mask` and `overlay_mask_on_image` "
    "to use `argmax` and a color palette."
)
