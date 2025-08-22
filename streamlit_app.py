# streamlit_app.py
import os, io, time, tempfile
from typing import Optional
import numpy as np
import pandas as pd
from PIL import Image
import cv2
import streamlit as st

# ──────────────────────────────────────────────────────────────────────────────
# Hard requirements and environment setup
# ──────────────────────────────────────────────────────────────────────────────
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except Exception:
    st.error(
        "PyTorch failed to import. Ensure requirements.txt pins a Python-3.13 compatible version, "
        "for example: torch==2.5.1 (and torchvision==0.20.1 only if you import it)."
    )
    st.stop()

try:
    import gdown
except Exception:
    st.error("Missing dependency 'gdown'. Add `gdown>=5` to requirements.txt.")
    st.stop()

# CPU-only for Streamlit Community Cloud
os.environ["OMP_NUM_THREADS"] = "1"
torch.set_num_threads(1)
DEVICE = "cpu"

# Default pixel size (µm/pixel), user can change in sidebar
ORIGINAL_PIXEL_SIZE_UM = 1.41

# ──────────────────────────────────────────────────────────────────────────────
# Streamlit page config
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Organoid Segmentation – Web Uploader (CPU)", layout="wide")
st.title("Organoid Segmentation – Web Uploader (CPU)")

# ──────────────────────────────────────────────────────────────────────────────
# Google Drive model download
# ──────────────────────────────────────────────────────────────────────────────
FILE_ID = "1ExZa5GLlRHdRermgnssqV7-TlqmUr0I5"
DRIVE_VIEW_URL = f"https://drive.google.com/file/d/{FILE_ID}/view"
DRIVE_DIRECT_URL = f"https://drive.google.com/uc?id={FILE_ID}"

@st.cache_resource(show_spinner=False)
def _download_model_to_tmp() -> str:
    """
    Download the model from Google Drive into a temp file, cached across reruns.
    """
    target = os.path.join(tempfile.gettempdir(), "organoid_model.pt")
    if not os.path.exists(target) or os.path.getsize(target) == 0:
        with st.status("Downloading model from Google Drive…", expanded=True) as status:
            st.write(f"Source: {DRIVE_VIEW_URL}")
            gdown.download(DRIVE_DIRECT_URL, target, quiet=False)
            if not os.path.exists(target) or os.path.getsize(target) == 0:
                status.update(label="Download failed.", state="error")
                raise RuntimeError("Model download failed or produced empty file.")
            status.update(label="Model downloaded.", state="complete")
    return target

# ──────────────────────────────────────────────────────────────────────────────
# Attention U-Net reconstruction (to load a state_dict)
# ──────────────────────────────────────────────────────────────────────────────
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels),
        )
    def forward(self, x):
        return self.maxpool_conv(x)

class AttentionBlock(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, bias=True),
            nn.BatchNorm2d(F_int),
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, bias=True),
            nn.BatchNorm2d(F_int),
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        # g: decoder feature (gate), x: encoder skip feature
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi

class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=False):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels)
        else:
            # transposed conv expects half the in_channels because of concatenation
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)
        # attention acts on skip and gate features; here we align dimensions using out_channels
        self.attention = AttentionBlock(F_g=out_channels, F_l=out_channels, F_int=out_channels // 2)

    def forward(self, x1, x2):
        # x1: decoder feature, x2: encoder skip
        x1 = self.up(x1)
        # pad to handle odd shapes
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        # attention refine on skip
        x2 = self.attention(x1, x2)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    """U-Net with Attention Gates (default: 3→1 channels)"""
    def __init__(self, n_channels=3, n_classes=1, bilinear=False):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x,  x3)
        x = self.up3(x,  x2)
        x = self.up4(x,  x1)
        return self.outc(x)

def _build_exact_model() -> nn.Module:
    # Adjust if your training used different channels/classes/bilinear
    return UNet(n_channels=3, n_classes=1, bilinear=False)

@st.cache_resource(show_spinner=True)
def load_model() -> "torch.nn.Module":
    """
    Load either a state_dict or a fully serialized model, and return it in eval mode (CPU).
    """
    local_path = _download_model_to_tmp()
    checkpoint = torch.load(local_path, map_location="cpu")

    # If it is a state_dict (OrderedDict of parameter tensors)
    if isinstance(checkpoint, dict) and not hasattr(checkpoint, "forward"):
        model = _build_exact_model()
        missing, unexpected = model.load_state_dict(checkpoint, strict=False)
        if missing or unexpected:
            st.warning(f"State dict loaded with key differences. Missing: {len(missing)}, Unexpected: {len(unexpected)}")
            if missing:
                st.write(missing[:10])
            if unexpected:
                st.write(unexpected[:10])
        model.eval()
        return model

    # Else assume it is a full serialized model object
    model = checkpoint
    if hasattr(model, "to"):
        model.to("cpu")
    if hasattr(model, "eval"):
        model.eval()
    return model

# ──────────────────────────────────────────────────────────────────────────────
# Pre/post-processing helpers
# ──────────────────────────────────────────────────────────────────────────────
def pil_to_tensor(img: Image.Image) -> torch.Tensor:
    arr = np.array(img.convert("RGB"))
    t = torch.from_numpy(arr).float() / 255.0
    return t.permute(2, 0, 1).unsqueeze(0)  # 1,3,H,W

def tensor_to_mask(prob: torch.Tensor, threshold: float = 0.5) -> np.ndarray:
    """
    Binary: (1,1,H,W) via sigmoid threshold.
    Multi-class: (1,C,H,W) via softmax argmax.
    """
    if prob.ndim == 4 and prob.shape[1] == 1:
        probs = torch.sigmoid(prob)[0, 0].cpu().numpy()
        return (probs >= threshold).astype(np.uint8)
    elif prob.ndim == 4 and prob.shape[1] > 1:
        probs = F.softmax(prob, dim=1)[0]
        return probs.argmax(0).cpu().numpy().astype(np.uint8)
    raise ValueError(f"Unexpected model output shape: {tuple(prob.shape)}")

def overlay_mask_on_image(image: np.ndarray, mask: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    h, w, _ = image.shape
    if mask.ndim != 2:
        raise ValueError("Mask must be 2D (H,W).")
    mask_vis = np.zeros((h, w, 3), dtype=np.uint8)
    # Binary mask overlay in green
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
pixel_size_um = st.sidebar.number_input(
    "Pixel size (µm/pixel)", value=float(ORIGINAL_PIXEL_SIZE_UM), min_value=0.0, step=0.01
)
gallery_cols = st.sidebar.slider("Gallery columns", 2, 6, 4)
redownload = st.sidebar.button("Force re-download model")

if redownload:
    # Clear caches and file to force a fresh download
    _download_model_to_tmp.clear()
    load_model.clear()
    try:
        old = os.path.join(tempfile.gettempdir(), "organoid_model.pt")
        if os.path.exists(old):
            os.remove(old)
    except Exception:
        pass
    st.experimental_rerun()

# ──────────────────────────────────────────────────────────────────────────────
# Main UI
# ──────────────────────────────────────────────────────────────────────────────
st.markdown(
    "Upload one or more images. The app will segment each image, display a gallery, "
    "and produce a per-image metrics table that you can download as CSV."
)

files = st.file_uploader(
    "Upload images (PNG/JPG/TIF)", type=["png", "jpg", "jpeg", "tif", "tiff"], accept_multiple_files=True
)

# Load model with visible status
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
st.caption("CPU-only. Model is downloaded from Google Drive and cached. Update FILE_ID to change the model.")
