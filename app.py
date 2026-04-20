import streamlit as st
import torch
import torch.nn as nn
import cv2
import numpy as np
from PIL import Image
import io

st.set_page_config(page_title="Retinal Vessel Segmentation", page_icon="👁️", layout="wide")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500&display=swap');
    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
    .main { background-color: #0a0e1a; color: #e0e6f0; }
    .block-container { padding: 2rem 3rem; }
    h1, h2, h3 { font-family: 'Space Mono', monospace !important; color: #00e5ff !important; }
    .title-box { background: linear-gradient(135deg, #0d1b2a, #1a2a3a); border: 1px solid #00e5ff33; border-radius: 12px; padding: 2rem; margin-bottom: 2rem; text-align: center; }
    .title-box h1 { font-size: 2.2rem; margin: 0; letter-spacing: -1px; }
    .title-box p  { color: #7a9bbf; margin: 0.5rem 0 0; font-size: 1rem; }
    .metric-card { background: #0d1b2a; border: 1px solid #00e5ff22; border-radius: 10px; padding: 1.2rem; text-align: center; }
    .metric-card .val   { font-family: 'Space Mono', monospace; font-size: 1.8rem; color: #00e5ff; }
    .metric-card .label { font-size: 0.8rem; color: #7a9bbf; text-transform: uppercase; letter-spacing: 1px; }
    .stButton > button { background: linear-gradient(90deg, #00e5ff, #0080ff); color: #0a0e1a; font-family: 'Space Mono', monospace; font-weight: 700; border: none; border-radius: 8px; padding: 0.6rem 2rem; font-size: 1rem; width: 100%; }
    .img-label { font-family: 'Space Mono', monospace; font-size: 0.75rem; color: #00e5ff; text-transform: uppercase; letter-spacing: 2px; text-align: center; margin-bottom: 0.4rem; }
    div[data-testid="stImage"] img { border-radius: 10px; border: 1px solid #00e5ff22; }
    .info-box { background: #0d1b2a; border-left: 3px solid #00e5ff; border-radius: 0 8px 8px 0; padding: 1rem 1.2rem; margin: 1rem 0; font-size: 0.9rem; color: #7a9bbf; }
    .upload-area { border: 2px dashed #00e5ff44; border-radius: 12px; padding: 2rem; text-align: center; background: #0d1b2a; }
</style>
""", unsafe_allow_html=True)


# U-Net
class conv_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, 3, padding=1)
        self.bn1   = nn.BatchNorm2d(out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, 3, padding=1)
        self.bn2   = nn.BatchNorm2d(out_c)
        self.relu  = nn.ReLU()
    def forward(self, x):
        return self.relu(self.bn2(self.conv2(self.relu(self.bn1(self.conv1(x))))))

class encoder_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = conv_block(in_c, out_c)
        self.pool = nn.MaxPool2d((2, 2))
    def forward(self, x):
        s = self.conv(x); return s, self.pool(s)

class decoder_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.up   = nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=2)
        self.conv = conv_block(out_c + out_c, out_c)
    def forward(self, x, skip):
        return self.conv(torch.cat([self.up(x), skip], axis=1))

class build_unet(nn.Module):
    def __init__(self):
        super().__init__()
        self.en1 = encoder_block(3, 32);   self.en2 = encoder_block(32, 64)
        self.en3 = encoder_block(64, 128); self.en4 = encoder_block(128, 256)
        self.bottleneck = conv_block(256, 512)
        self.d1 = decoder_block(512, 256); self.d2 = decoder_block(256, 128)
        self.d3 = decoder_block(128, 64);  self.d4 = decoder_block(64, 32)
        self.outputs = nn.Conv2d(32, 1, kernel_size=1)
    def forward(self, x):
        s1,p1=self.en1(x); s2,p2=self.en2(p1); s3,p3=self.en3(p2); s4,p4=self.en4(p3)
        b=self.bottleneck(p4)
        return self.outputs(self.d4(self.d3(self.d2(self.d1(b,s4),s3),s2),s1))


@st.cache_resource
def load_model(path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model  = build_unet().to(device)
    ckpt   = torch.load(path, map_location=device)
    model.load_state_dict(ckpt['model_state'] if 'model_state' in ckpt else ckpt)
    model.eval()
    return model, device

def predict(model, device, image_np, threshold=0.5):
    img    = cv2.resize(image_np, (256, 256))
    tensor = torch.from_numpy(np.transpose(img / 255.0, (2, 0, 1))).float().unsqueeze(0).to(device)
    with torch.no_grad():
        return (torch.sigmoid(model(tensor)) > threshold).float().squeeze().cpu().numpy()

def overlay_mask(image_np, mask_np, color=(0, 229, 255), alpha=0.45):
    img  = cv2.resize(image_np, (256, 256)).copy()
    mask = mask_np.astype(bool)
    img[mask] = ((1 - alpha) * img[mask] + alpha * np.array(color)).astype(np.uint8)
    return img


# UI
st.markdown("""
<div class="title-box">
    <h1>👁️ Retinal Vessel Segmentation</h1>
    <p>Upload a retinal fundus image — U-Net will segment the blood vessels</p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### ⚙️ Settings")
    checkpoint_path = st.text_input("Model checkpoint path", value="checkpoint.pth")
    threshold       = st.slider("Prediction threshold", 0.1, 0.9, 0.5, 0.05,
                                help="Lower = more vessels detected")
    show_overlay    = st.checkbox("Show vessel overlay", value=True)
    st.markdown("---")
    st.markdown("""<div class="info-box">
        <b>Supported formats</b><br>PNG · JPG · TIF · BMP<br><br>
        <b>Model</b><br>U-Net trained on DRIVE dataset<br><br>
        <b>Input size</b><br>Auto-resized to 256×256
    </div>""", unsafe_allow_html=True)

try:
    model, device = load_model(checkpoint_path)
    st.sidebar.success(f"✅ Model loaded  |  **{device}**")
except Exception as e:
    st.sidebar.error(f"❌ Cannot load model:\n{e}")
    st.stop()

uploaded = st.file_uploader("Drop your retinal image here", type=["png","jpg","jpeg","tif","tiff","bmp"])

if uploaded:
    file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
    image_bgr  = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    image_rgb  = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    with st.spinner("🔬 Segmenting vessels..."):
        pred_mask = predict(model, device, image_rgb, threshold)

    vessel_pct = pred_mask.mean() * 100
    vessel_px  = int(pred_mask.sum())

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f'<div class="metric-card"><div class="val">{vessel_pct:.1f}%</div><div class="label">Vessel Coverage</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="metric-card"><div class="val">{vessel_px:,}</div><div class="label">Vessel Pixels</div></div>', unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="metric-card"><div class="val">{threshold}</div><div class="label">Threshold Used</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    cols = st.columns(3) if show_overlay else st.columns(2)

    with cols[0]:
        st.markdown('<p class="img-label">Input Image</p>', unsafe_allow_html=True)
        st.image(image_rgb, use_container_width=True)
    with cols[1]:
        st.markdown('<p class="img-label">Predicted Mask</p>', unsafe_allow_html=True)
        st.image(pred_mask, use_container_width=True, clamp=True)
    if show_overlay:
        with cols[2]:
            st.markdown('<p class="img-label">Vessel Overlay</p>', unsafe_allow_html=True)
            st.image(overlay_mask(image_rgb, pred_mask), use_container_width=True)

    st.markdown("---")
    buf = io.BytesIO()
    Image.fromarray((pred_mask * 255).astype(np.uint8)).save(buf, format="PNG")
    st.download_button("⬇️ Download Predicted Mask", buf.getvalue(),
                       file_name=f"mask_{uploaded.name}", mime="image/png")
else:
    st.markdown("""
    <div class="upload-area">
        <h3 style="color:#00e5ff;font-family:'Space Mono',monospace;">Upload an Image to Begin</h3>
        <p style="color:#7a9bbf;">Supports PNG · JPG · TIF · BMP</p>
    </div>""", unsafe_allow_html=True)