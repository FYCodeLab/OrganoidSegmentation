# streamlit_app.py
import os, io, time, tempfile
from typing import Optional, Dict, Tuple
import numpy as np
import pandas as pd
from PIL import Image
import cv2
import streamlit as st

# ──────────────────────────────────────────────────────────────────────────────
# Runtime prerequisites
# ──────────────────────────────────────────────────────────────────────────────
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except Exception:
    st.error(
        "PyTorch failed to import. Ensure requirements.txt pins a Python-3.13 wheel, e.g.: torch==2.5.1"
    )
    st.stop()

try:
    import gdown
except Exception:
    st.error("Missing dependency 'gdown'. Add `gdown>=5` to requirements.txt.")
    st.stop()

# CPU-only in Streamlit Cloud
os.environ["OMP_NUM_THREADS"] = "1"
torch.set_num_threads(1)
DEVICE = "cpu"

# Default pixel size (µm/pixel), editable in sidebar
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
# Attention U-Net (parameterizable widths) – close to your Colab topology
# ──────────────────────────────────────────────────────────────────────────────
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.double_conv(x)

class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.seq = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_ch, out_ch))
    def forward(self, x): return self.seq(x)

class AttentionBlock(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(nn.Conv2d(F_g, F_int, 1, bias=True), nn.BatchNorm2d(F_int))
        self.W_x = nn.Sequential(nn.Conv2d(F_l, F_int, 1, bias=True), nn.BatchNorm2d(F_int))
        self.psi = nn.Sequential(nn.Conv2d(F_int, 1, 1, bias=True), nn.BatchNorm2d(1), nn.Sigmoid())
        self.relu = nn.ReLU(inplace=True)
    def forward(self, g, x):
        psi = self.relu(self.W_g(g) + self.W_x(x))
        psi = self.psi(psi)
        return x * psi

class Up(nn.Module):
    def __init__(self, in_ch_cat, out_ch, bilinear=False):
        """
        in_ch_cat is the channel count AFTER concatenation (skip + upsampled).
        We implement upsampling expecting decoder half before concat:
          - if transposed conv: we upsample channels = out_ch, then concat with skip (out_ch), so in_ch_cat ~ 2*out_ch
        """
        super().__init__()
        self.bilinear = bilinear
        # We assume decoder channel count equals out_ch before concat
        dec_ch = out_ch
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(dec_ch, dec_ch, kernel_size=2, stride=2)
        # attention over skip feature (we align dims to out_ch)
        self.att = AttentionBlock(F_g=dec_ch, F_l=dec_ch, F_int=dec_ch // 2 if dec_ch >= 2 else 1)
        self.conv = DoubleConv(in_ch_cat, out_ch)

    def forward(self, x_dec, x_skip):
        x_dec = self.up(x_dec)
        # pad if needed
        dy = x_skip.size(2) - x_dec.size(2)
        dx = x_skip.size(3) - x_dec.size(3)
        x_dec = F.pad(x_dec, [dx // 2, dx - dx // 2, dy // 2, dy - dy // 2])
        x_skip = self.att(x_dec, x_skip)
        x = torch.cat([x_skip, x_dec], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_ch, out_ch): super().__init__(); self.conv = nn.Conv2d(in_ch, out_ch, 1)
    def forward(self, x): return self.conv(x)

class UNetAttn(nn.Module):
    """
    Parameterizable Attention U-Net.
    widths: tuple of encoder widths, e.g. (64,128,256,512,1024)
    """
    def __init__(self, n_channels=3, n_classes=1, widths=(64,128,256,512,1024), bilinear=False):
        super().__init__()
        w0, w1, w2, w3, w4 = widths
        factor = 2 if bilinear else 1
        self.inc   = DoubleConv(n_channels, w0)
        self.down1 = Down(w0, w1)
        self.down2 = Down(w1, w2)
        self.down3 = Down(w2, w3)
        self.down4 = Down(w3, w4 // factor)
        # Decoder expects concatenation: (skip + upsampled) → in_ch_cat
        self.up1 = Up(in_ch_cat=(w4 // factor + w3), out_ch=w3 // factor, bilinear=bilinear)
        self.up2 = Up(in_ch_cat=(w3 // factor + w2), out_ch=w2 // factor, bilinear=bilinear)
        self.up3 = Up(in_ch_cat=(w2 // factor + w1), out_ch=w1 // factor, bilinear=bilinear)
        self.up4 = Up(in_ch_cat=(w1 // factor + w0), out_ch=w0,           bilinear=bilinear)
        self.outc = OutConv(w0, n_classes)

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

# ──────────────────────────────────────────────────────────────────────────────
# Safe state-dict loader (skips mismatched shapes, reports summary)
# ──────────────────────────────────────────────────────────────────────────────
def load_state_dict_safely(model: nn.Module, state: Dict[str, torch.Tensor]) -> Tuple[int, int, int]:
    model_state = model.state_dict()
    loaded, skipped_shape, skipped_missing = 0, 0, 0
    to_load = {}
    for k, v in state.items():
        if k in model_state:
            if model_state[k].shape == v.shape:
                to_load[k] = v
                loaded += 1
            else:
                skipped_shape += 1
        else:
            skipped_missing += 1
    model_state.update(to_load)
    model.load_state_dict(model_state, strict=False)
    return loaded, skipped_shape, skipped_missing

# ──────────────────────────────────────────────────────────────────────────────
# Sidebar controls (let you match the training config)
# ──────────────────────────────────────────────────────────────────────────────
st.sidebar.header("Settings")
st.sidebar.info(f"Model source: Google Drive\n\n{DRIVE_VIEW_URL}")

threshold = st.sidebar.slider("Mask threshold (binary)", 0.0, 1.0, 0.5, 0.01)
pixel_size_um = st.sidebar.number_input("Pixel size (µm/pixel)", value=float(ORIGINAL_PIXEL_SIZE_UM), min_value=0.0, step=0.01)

# Architecture knobs—set these to what you trained
n_classes = st.sidebar.number_input("n_classes", min_value=1, max_value=8, value=1, step=1)
base = st.sidebar.selectbox("Base width", options=[32, 48, 64, 96, 128], index=2)  # 64 default
bilinear = st.sidebar.checkbox("Bilinear upsampling (instead of transposed conv)", value=False)
widths = (base, base*2, base*4, base*8, base*16)

show_ckpt_shapes = st.sidebar.checkbox("Debug: show checkpoint key shapes", value=False)
redownload = st.sidebar.button("Force re-download model")
reload_model = st.sidebar.button("Reload model with current settings")

if redownload:
    _download_model_to_tmp.clear()
    try:
        p = os.path.join(tempfile.gettempdir(), "organoid_model.pt")
        if os.path.exists(p): os.remove(p)
    except Exception: pass
    st.experimental_rerun()

# ──────────────────────────────────────────────────────────────────────────────
# Load model (download + construct arch + safe load)
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=True)
def load_model(widths, n_classes, bilinear):
    local_path = _download_model_to_tmp()
    checkpoint = torch.load(local_path, map_location="cpu")

    # Build architecture with current knobs
    model = UNetAttn(n_channels=3, n_classes=int(n_classes), widths=tuple(int(w) for w in widths), bilinear=bool(bilinear))

    # state_dict path
    if isinstance(checkpoint, dict) and not hasattr(checkpoint, "forward"):
        loaded, skipped_shape, skipped_missing = load_state_dict_safely(model, checkpoint)
        st.session_state["_load_report"] = (loaded, skipped_shape, skipped_missing)
    else:
        # full serialized model; try to use as-is, but adapt head if needed
        if hasattr(checkpoint, "to"): checkpoint.to("cpu")
        model = checkpoint
        st.session_state["_load_report"] = None

    if hasattr(model, "eval"): model.eval()
    return model

if reload_model:
    load_model.clear()

with st.status("Preparing model (CPU)…", expanded=True) as s:
    s.write("Loading from Google Drive cache and initializing weights on CPU.")
    model = load_model(widths, n_classes, bilinear)
    rep = st.session_state.get("_load_report", None)
    if rep is not None:
        loaded, skipped_shape, skipped_missing = rep
        s.write(f"State-dict load summary → loaded: {loaded}, skipped (shape mismatch): {skipped_shape}, skipped (no key): {skipped_missing}")
        if skipped_shape > 0:
            s.write("Hint: adjust 'Base width', 'n_classes', or 'Bilinear' to match the training config, then click 'Reload model'.")
    s.update(label="Model ready.", state="complete")

# Optional: show checkpoint key shapes for diagnosis
if show_ckpt_shapes:
    try:
        local_path = _download_model_to_tmp()
        state = torch.load(local_path, map_location="cpu")
        if isinstance(state, dict) and not hasattr(state, "forward"):
            import itertools
            st.subheader("Checkpoint parameter shapes")
            rows = [{"key": k, "shape": tuple(v.shape)} for k, v in itertools.islice(state.items(), 0, 2000)]
            dfk = pd.DataFrame(rows)
            st.dataframe(dfk, use_container_width=True, height=300)
        else:
            st.info("Checkpoint appears to be a serialized model object, not a state_dict.")
    except Exception as e:
        st.warning(f"Could not introspect checkpoint: {e}")

# ──────────────────────────────────────────────────────────────────────────────
# Inference helpers
# ──────────────────────────────────────────────────────────────────────────────
def pil_to_tensor(img: Image.Image) -> torch.Tensor:
    arr = np.array(img.convert("RGB"))
    t = torch.from_numpy(arr).float() / 255.0
    return t.permute(2, 0, 1).unsqueeze(0)  # 1,3,H,W

def tensor_to_mask(prob: torch.Tensor, threshold: float = 0.5) -> np.ndarray:
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
# Main UI
# ──────────────────────────────────────────────────────────────────────────────
st.markdown(
    "Upload one or more images. The app will segment each image, display a gallery, "
    "and produce a per-image metrics table (CSV downloadable)."
)

files = st.file_uploader(
    "Upload images (PNG/JPG/TIF)", type=["png", "jpg", "jpeg", "tif", "tiff"], accept_multiple_files=True
)

results = []
if files:
    st.subheader("Results")
    gallery_cols = st.sidebar.slider("Gallery columns", 2, 6, 4)
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
st.caption("CPU-only. The model is downloaded from Google Drive and cached. "
           "Use sidebar knobs to match your training config; then ‘Reload model’.")
