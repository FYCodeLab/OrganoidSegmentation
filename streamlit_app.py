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
