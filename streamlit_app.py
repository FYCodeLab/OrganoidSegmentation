# streamlit_app.py
import os, io, time, tempfile
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from PIL import Image
import cv2
import streamlit as st

# ── Hard requirement: PyTorch must be installed (compatible with Python 3.13)
try:
    import torch
    import torch.nn.functional as F
except Exception as e:
    st.error(
        "PyTorch failed to import. Ensure requirements.txt pins a Python-3.13 compatible version, "
        "for example: torch==2.5.1 (and torchvision==0.20.1 only if you actually import it)."
    )
    st.stop()

# ── External downloader
try:
    import gdown
except Exception:
    st.error("Missing dependency 'gdown'. Add `gdown>=5` to requirements.txt.")
    st.stop()

# ──────────────────────────────────────────────────────────────────────────────
# App config
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Organoid Segmentation – Web Uploader (CPU)", layout="wide")
st.title("Organoid Segmentation – Web Uploader (CPU)")

# Force CPU/threading constraints suitable for Streamlit Cloud
os.environ["OMP_NUM_THREADS"] = "1"
torch.set_num_threads(1)
DEVICE = "cpu"

# ──────────────────────────────────────────────────────────────────────────────
# Model download & load (Google Drive)
# ──────────────────────────────────────────────────────────────────────────────
# Replace only this block if you move the model elsewhere
FILE_ID = "1ExZa5GLlRHdRermgnssqV7-TlqmUr0I5"
DRIVE_VIEW_URL = f"https://drive.google.com/file/d/{FILE_ID}/view"
DRIVE_DIRECT_URL = f"https://drive.google.com/uc?id={FILE_ID}"

@st.cache_resource(show_spinner=False)
def _download_model_to_tmp() -> str:
    """
    Downloads the model from Google Drive to a temp file (cached across reruns).
    Returns the local file path.
    """
    target = os.path.join(tempfile.gettempdir(), "organoid_model.pt")
    if not os.path.exists(target) or os.path.getsize(target) == 0:
        # Visible status to the user: we are downloading from Google Drive
        with st.status("Downloading model from Google Drive…", expanded=True) as status:
            st.write(f"Source: {DRIVE_VIEW_URL}")
            gdown.download(DRIVE_DIRECT_URL, target, quiet=False)
            # Basic sanity check
            if not os.path.exists(target) or os.path.getsize(target) == 0:
                status.update(label="Download failed.", state="error")
                raise RuntimeError("Model download failed or produced empty file.")
            status.update(label="Model downloaded.", state="complete")
    return target

@st.cache_resource(show_spinner=True)
def load_model() -> "torch.nn.Module":
    """
    Loads the PyTorch model (CPU) from the downloaded file and returns it in eval mode.
    """
    local_path = _download_model_to_tmp()
    # If you saved a full module: torch.load(local_path, map_location="cpu")
    # If you saved a state_dict: you must reconstruct your model class and call load_state_dict.
    model = torch.load(local_path, map_location="cpu")
    model.eval()
    return model

# ──────────────────────────────────────────────────────────────────────────────
# Inference utilities (adapt these to your model’s I/O if needed)
# ──────────────────────────────────────────────────────────────────────────────
def pil_to_tensor(img: Image.Image) -> torch.Tensor:
    arr = np.array(img.convert("RGB"))
    t = torch.from_numpy(arr).float() / 255.0  # [0,1]
    return t.permute(2, 0, 1).unsqueeze(0)     # 1,3,H,W

def tensor_to_mask(prob: torch.Tensor, threshold: float = 0.5) -> np.ndarray:
    """
    Supports either (1,1,H,W) -> sigmoid binary, or (1,C,H,W) -> softmax multi-class.
    Modify if your model’s output format differs.
    """
    if prob.ndim == 4 and prob.shape[1] == 1:
        probs = torch.sigmoid(prob)[0, 0].cpu().numpy()
        return (probs >= threshold).astype(np.uint8)
    elif prob.ndim == 4 and prob.shape[1] > 1:
        probs = F.softmax(prob, dim=1)[0]                  # C,H,W
        return probs.argmax(0).cpu().numpy().astype(np.uint8)
    raise ValueError(f"Unexpected model output shape: {tuple(prob.shape)}")

def overlay_mask_on_image(image: np.ndarray, mask: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    h, w, _ = image.shape
    if mask.ndim != 2:
        raise ValueError("Mask must be 2D (H,W).")
    mask_vis = np.zeros((h, w, 3), dtype=np.uint8)
    # Binary: green overlay for 1s; extend if multi-class with a palette.
    mask_vis[mask == 1] = (0, 255, 0)
    return cv2.addWeighted(image, 1.0, mask_vis, alpha, 0.0)

def compute_basic_stats(mask: np.ndarray, pixel_size_um: Optional[float]) -> dict:
    area_px = int(mask.sum())
    area_um2 = float(area_px * (pixel_size_um ** 2)) if pixel_size_um else None
    nlab, _ = cv2.connectedComponents(mask.astype(np.uint8))
    return {"area_px": area_px, "area_um2": area_um2, "components": int(nlab - 1)}

def predict_mask(model, img: Image.Image, threshold: float) -> np.ndarray:
    with torch.no_grad():
        t = pil_to_tensor(img).to(DEVICE)
        out = model(t)
    return tensor_to_mask(out, threshold=threshold)

# ──────────────────────────────────────────────────────────────────────────────
# Sidebar controls
# ──────────────────────────────────────────────────────────────────────────────
st.sidebar.header("Settings")
st.sidebar.info(f"Model source: Google Drive\n\n{DRIVE_VIEW_URL}")
threshold = st.sidebar.slider("Mask threshold (binary models)", 0.0, 1.0, 0.5, 0.01)
pixel_size_um = st.sidebar.number_input("Pixel size (µm/pixel, optional)", value=0.0, min_value=0.0, step=0.01)
gallery_cols = st.sidebar.slider("Gallery columns", 2, 6, 4)
redownload = st.sidebar.button("Force re-download model")

if redownload:
    # Clear caches for a fresh download/load
    _download_model_to_tmp.clear()
    load_model.clear()
    # Remove old file to guarantee re-fetch
    try:
        old = os.path.join(tempfile.gettempdir(), "organoid_model.pt")
        if os.path.exists(old):
            os.remove(old)
    except Exception:
        pass
    st.experimental_rerun()

# ──────────────────────────────────────────────────────────────────────────────
# Uploader + processing
# ──────────────────────────────────────────────────────────────────────────────
st.markdown(
    "Upload one or more images. The app will segment each image, display a gallery, "
    "and produce a per-image metrics table (CSV downloadable)."
)

files = st.file_uploader(
    "Upload images (PNG/JPG/TIF)", type=["png", "jpg", "jpeg", "tif", "tiff"], accept_multiple_files=True
)

# Load model (with visible status message)
with st.status("Preparing model (CPU)…", expanded=True) as s:
    s.write("Loading from Google Drive cache and initializing weights on CPU.")
    model = load_model()
    s.update(label="Model ready.", state="complete")

results = []
if files and model is not None:
    st.subheader("Results")
    cols = st.columns(gallery_cols, gap="small")
    i = 0
    t_all = time.time()

    px_size = pixel_size_um if pixel_size_um > 0 else None

    for up in files:
        image_bytes = up.read()
        pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img_np = np.array(pil)

        t0 = time.time()
        try:
            mask = predict_mask(model, pil, threshold)
        except Exception as e:
            st.error(f"Inference failed for {up.name}: {e}")
            continue
        dt = time.time() - t0

        overlay = overlay_mask_on_image(img_np, mask, alpha=0.45)
        stats = compute_basic_stats(mask, px_size)

        row = {
            "filename": up.name,
            "height_px": img_np.shape[0],
            "width_px": img_np.shape[1],
            "area_px": stats["area_px"],
            "area_um2": stats["area_um2"],
            "components": stats["components"],
            "runtime_s": round(dt, 4),
        }
        results.append(row)

        with cols[i % gallery_cols]:
            st.caption(up.name)
            st.image(img_np, caption="Original", use_column_width=True)
            st.image(overlay, caption="Segmentation overlay", use_column_width=True)
        i += 1

    st.info(f"Processed {len(results)} image(s) in {time.time() - t_all:.2f} s")

if results:
    st.subheader("Per-image metrics")
    df = pd.DataFrame(results)
    st.dataframe(df, use_container_width=True)
    st.download_button(
        "Download CSV",
        df.to_csv(index=False).encode("utf-8"),
        file_name="segmentation_metrics.csv",
        mime="text/csv",
    )

st.markdown("---")
st.caption("Notes: CPU-only. Model is fetched from Google Drive and cached. Update the FILE_ID to change the model.")
