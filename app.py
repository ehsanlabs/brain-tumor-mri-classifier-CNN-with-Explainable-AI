import os, io, gc
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import cv2
import tensorflow as tf
from tensorflow import keras
import streamlit as st
from PIL import Image

# ─── CONFIG ───────────────────────────────────────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(__file__), "guaranteed_90plus.keras")
CLASSES    = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]
IMG_SIZE   = 128
CAM_LAYER  = "last_conv"

CLASS_INFO = {
    "Glioma": {
        "color": "#e74c3c",
        "desc":  "Gliomas are tumors that arise from glial cells in the brain or spine. "
                 "They are the most common primary brain tumors and can be benign or malignant.",
        "icon":  "🔴",
    },
    "Meningioma": {
        "color": "#f39c12",
        "desc":  "Meningiomas arise from the meninges — the membranes surrounding the brain "
                 "and spinal cord. Most are benign and slow-growing.",
        "icon":  "🟡",
    },
    "No Tumor": {
        "color": "#2ecc71",
        "desc":  "No tumor detected. The MRI scan appears normal with no signs of a brain tumor.",
        "icon":  "🟢",
    },
    "Pituitary": {
        "color": "#9b59b6",
        "desc":  "Pituitary tumors form in the pituitary gland at the base of the brain. "
                 "Most are benign (adenomas) and respond well to treatment.",
        "icon":  "🟣",
    },
}

# ─── PAGE SETUP ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Brain Tumor MRI Classifier",
    page_icon="🧠",
    layout="wide",
)

st.markdown("""
<style>
    .stApp { background-color: #0e1117; }
    .main-title {
        font-size: 2.4rem; font-weight: 800; text-align: center;
        background: linear-gradient(90deg, #00c6ff, #0072ff);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        text-align: center; color: #888; font-size: 1rem; margin-bottom: 2rem;
    }
    .result-card {
        border-radius: 12px; padding: 1.5rem;
        background: #1a1f2e; border: 2px solid;
        margin-top: 1rem;
    }
    .confidence-bar { border-radius: 8px; height: 10px; margin: 4px 0; }
    .section-header {
        font-size: 1.1rem; font-weight: 700;
        color: #00c6ff; border-bottom: 1px solid #333; padding-bottom: 6px;
        margin-bottom: 12px;
    }
</style>
""", unsafe_allow_html=True)

# ─── HEADER ───────────────────────────────────────────────────────────────────
st.markdown('<div class="main-title">🧠 Brain Tumor MRI Classifier</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Final Year Project — AI-Powered Brain Tumor Detection using Deep Learning</div>', unsafe_allow_html=True)

# ─── LOAD MODEL (cached) ──────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model...")
def load_model():
    m = keras.models.load_model(MODEL_PATH)
    grad_model = keras.Model(inputs=m.inputs,
                             outputs=[m.get_layer(CAM_LAYER).output, m.output])
    return m, grad_model

model, grad_model = load_model()

# ─── GRADCAM FUNCTION ─────────────────────────────────────────────────────────
def compute_gradcam(pil_img):
    """Returns (pred_class, confidence_list, cam_overlay_pil, tumor_overlay_pil)"""
    # Preprocess
    rgb128 = np.array(pil_img.resize((IMG_SIZE, IMG_SIZE)).convert("RGB"))
    arr    = np.expand_dims(rgb128.astype("float32") / 255.0, axis=0)

    # GradCAM
    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(tf.cast(arr, tf.float32))
        pred_idx = int(np.argmax(preds[0].numpy()))
        loss = preds[:, pred_idx]

    grads   = tape.gradient(loss, conv_out)[0].numpy()
    weights = np.mean(grads, axis=(0, 1))
    cam     = np.sum(weights * conv_out[0].numpy(), axis=-1)
    cam     = np.maximum(cam, 0)
    if cam.max() > 0:
        cam = (cam - cam.min()) / (cam.max() - cam.min())

    cam_up = cv2.resize(cam.astype("float32"), (IMG_SIZE, IMG_SIZE))

    # GradCAM mask
    gradcam_mask = (cam_up > 0.35).astype(np.uint8) * 255

    # Otsu on CLAHE gray
    bgr = cv2.cvtColor(rgb128, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray_eq = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4,4)).apply(gray)
    _, otsu = cv2.threshold(gray_eq, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    combined = cv2.bitwise_and(otsu, gradcam_mask)
    k1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    k2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, k1)
    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, k2)
    combined = cv2.dilate(combined, k1, iterations=1)

    n, labels, stats, _ = cv2.connectedComponentsWithStats(combined)
    if n > 1:
        largest  = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        combined = ((labels == largest) * 255).astype(np.uint8)

    # Build display images at 320x320
    SIZE = 320
    disp   = np.array(pil_img.resize((SIZE, SIZE)).convert("RGB"))
    cam320 = cv2.resize(cam_up, (SIZE, SIZE))

    heat_bgr = cv2.applyColorMap(np.uint8(255 * cam320), cv2.COLORMAP_JET)
    heat_rgb = cv2.cvtColor(heat_bgr, cv2.COLOR_BGR2RGB)
    cam_viz  = cv2.addWeighted(disp, 0.45, heat_rgb, 0.55, 0)

    mask320 = cv2.resize(combined, (SIZE, SIZE))
    if CLASSES[pred_idx] != "No Tumor" and mask320.max() > 0:
        tumor_overlay = disp.copy()
        tumor_overlay[mask320 > 0] = [220, 50, 50]
        final_viz = cv2.addWeighted(disp, 0.5, tumor_overlay, 0.5, 0)
        contours, _ = cv2.findContours(mask320, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(final_viz, contours, -1, (0, 255, 80), 2)
    else:
        final_viz = disp.copy()

    confidences = (preds[0].numpy() * 100).tolist()
    gc.collect()

    return (
        pred_idx,
        confidences,
        Image.fromarray(cam_viz),
        Image.fromarray(final_viz),
        Image.fromarray(disp),
    )

# ─── SIDEBAR ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ℹ️ About")
    st.markdown("""
This system uses a **Convolutional Neural Network (CNN)** trained on Brain MRI scans
to classify tumors into 4 categories.

**Model Accuracy:** 93%+

**Tumor Types:**
- 🔴 Glioma
- 🟡 Meningioma
- 🟢 No Tumor
- 🟣 Pituitary

**Visualization:**
- GradCAM heatmap shows which brain region the model focused on.
- Tumor mask uses GradCAM + Otsu thresholding for boundary detection.
    """)
    st.markdown("---")
    st.markdown("**FYP — Muhammad S.**  \n*Deep Learning, Medical Imaging*")

# ─── MAIN UPLOAD ──────────────────────────────────────────────────────────────
col_up, col_gap = st.columns([2, 1])
with col_up:
    uploaded = st.file_uploader(
        "Upload a Brain MRI image (JPG / PNG)",
        type=["jpg", "jpeg", "png"],
        help="Upload an axial MRI scan for tumor classification",
    )

if uploaded is None:
    st.markdown("""
<div style='text-align:center; color:#444; padding: 3rem 0;'>
    <div style='font-size:4rem;'>🧠</div>
    <div style='margin-top:1rem; font-size:1.1rem;'>Upload an MRI image to get started</div>
</div>
""", unsafe_allow_html=True)
    st.stop()

# ─── PROCESS ──────────────────────────────────────────────────────────────────
pil_img = Image.open(uploaded)

with st.spinner("Analyzing MRI scan..."):
    pred_idx, confidences, cam_pil, tumor_pil, orig_pil = compute_gradcam(pil_img)

pred_class = CLASSES[pred_idx]
pred_conf  = confidences[pred_idx]
info       = CLASS_INFO[pred_class]

# ─── RESULT CARD ──────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="result-card" style="border-color: {info['color']};">
    <span style="font-size:2rem;">{info['icon']}</span>
    <span style="font-size:1.8rem; font-weight:800; color:{info['color']}; margin-left:12px;">
        {pred_class.upper()}
    </span>
    <span style="font-size:1.2rem; color:#aaa; margin-left:16px;">
        {pred_conf:.1f}% confidence
    </span>
    <p style="color:#ccc; margin-top:10px; margin-bottom:0;">{info['desc']}</p>
</div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ─── IMAGES ROW ───────────────────────────────────────────────────────────────
c1, c2, c3 = st.columns(3)
with c1:
    st.markdown('<div class="section-header">Original MRI</div>', unsafe_allow_html=True)
    st.image(orig_pil, width='stretch')

with c2:
    st.markdown('<div class="section-header">GradCAM Heatmap</div>', unsafe_allow_html=True)
    st.image(cam_pil, width='stretch')
    st.caption("Red/warm areas = model's focus region")

with c3:
    if pred_class == "No Tumor":
        st.markdown('<div class="section-header">Tumor Region</div>', unsafe_allow_html=True)
        st.image(orig_pil, width='stretch')
        st.caption("No tumor detected — scan appears normal")
    else:
        st.markdown('<div class="section-header">Tumor Region Highlighted</div>', unsafe_allow_html=True)
        st.image(tumor_pil, width='stretch')
        st.caption("Green contour = detected tumor boundary")

# ─── CONFIDENCE BARS ──────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
st.markdown('<div class="section-header">Confidence Scores</div>', unsafe_allow_html=True)

bar_cols = st.columns(4)
colors   = ["#e74c3c", "#f39c12", "#2ecc71", "#9b59b6"]
for i, (cls, conf, col, bcol) in enumerate(zip(CLASSES, confidences, bar_cols, colors)):
    with col:
        bold = "**" if i == pred_idx else ""
        st.markdown(f"{bold}{CLASS_INFO[cls]['icon']} {cls}{bold}")
        st.progress(int(conf))
        st.markdown(f"<span style='color:{bcol}; font-weight:700;'>{conf:.1f}%</span>",
                    unsafe_allow_html=True)
