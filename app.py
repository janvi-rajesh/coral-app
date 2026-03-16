import os
import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import time
import gdown

# ─────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Coral Health Classifier",
    page_icon="🪸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────────────────────
# ABSOLUTE PATH FIX
# ─────────────────────────────────────────────────────────────────

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ─────────────────────────────────────────────────────────────────
# CUSTOM CSS — ocean-dark theme
# ─────────────────────────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

.stApp {
    background: linear-gradient(135deg, #020c14 0%, #041824 50%, #052030 100%);
    color: #e0f4f8;
}

.main-header {
    text-align: center;
    padding: 2.5rem 0 1rem 0;
}
.main-header h1 {
    font-family: 'Syne', sans-serif;
    font-size: 3rem;
    font-weight: 800;
    background: linear-gradient(90deg, #00d4aa, #0099cc, #00d4aa);
    background-size: 200%;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    animation: shimmer 3s infinite linear;
    margin: 0;
}
@keyframes shimmer {
    0% { background-position: 0% }
    100% { background-position: 200% }
}
.main-header p {
    color: #7ab8cc;
    font-size: 1.05rem;
    font-weight: 300;
    margin-top: 0.4rem;
}

.result-box {
    background: rgba(0, 212, 170, 0.08);
    border: 1px solid rgba(0, 212, 170, 0.3);
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    text-align: center;
}
.result-box .label {
    font-family: 'Syne', sans-serif;
    font-size: 1.4rem;
    font-weight: 700;
}
.result-box .sublabel {
    font-size: 0.8rem;
    color: #7ab8cc;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-top: 0.2rem;
}
.result-box .confidence {
    font-size: 2rem;
    font-weight: 800;
    font-family: 'Syne', sans-serif;
    color: #ffffff;
    margin-top: 0.3rem;
}

[data-testid="stSidebar"] {
    background: rgba(2, 18, 28, 0.95) !important;
    border-right: 1px solid rgba(0,212,170,0.1);
}
[data-testid="stSidebar"] .stRadio label,
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] p {
    color: #b0d8e8 !important;
    font-family: 'DM Sans', sans-serif;
}

[data-testid="stFileUploader"] {
    border: 2px dashed rgba(0,212,170,0.3) !important;
    border-radius: 14px !important;
    background: rgba(0,212,170,0.04) !important;
}

hr { border-color: rgba(0,212,170,0.15) !important; }

.section-label {
    font-family: 'Syne', sans-serif;
    font-size: 1.25rem;
    font-weight: 700;
    color: #e0f4f8;
    margin-bottom: 0.8rem;
    margin-top: 1rem;
}

.info-row {
    display: flex;
    gap: 1rem;
    flex-wrap: wrap;
    margin-top: 0.5rem;
}
.info-chip {
    background: rgba(0,212,170,0.1);
    border: 1px solid rgba(0,212,170,0.2);
    border-radius: 20px;
    padding: 0.25rem 0.75rem;
    font-size: 0.82rem;
    color: #7ab8cc;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────

CLASS_NAMES = ["bleached_coral", "dead_coral", "healthy_coral"]

CLASS_COLORS = {
    "healthy_coral":  "#00e676",
    "bleached_coral": "#ffb300",
    "dead_coral":     "#ef5350",
}

CLASS_EMOJI = {
    "healthy_coral":  "🟢",
    "bleached_coral": "🟡",
    "dead_coral":     "🔴",
}

IMG_SIZE = (224, 224)

# ─────────────────────────────────────────────────────────────────
# LOAD MODELS
# ─────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────
# LOAD MODELS
# ─────────────────────────────────────────────────────────────────



@st.cache_resource
def load_densenet():
    path = "DenseNet121_model.h5"
    if not os.path.exists(path):
        gdown.download(
            "https://drive.google.com/uc?id=14UlJawF4oT4IHgWo4wegC4xizg6oF0gG",
            path, quiet=False
        )
    return tf.keras.models.load_model(path)

@st.cache_resource
def load_resnet():
    path = "ResNet.h5"
    if not os.path.exists(path):
        gdown.download(
            "https://drive.google.com/uc?id=1zo3QEEGrUrEGYWYFfH8IeEFodRzVC8pC",
            path, quiet=False
        )
    return tf.keras.models.load_model(path)

# ─────────────────────────────────────────────────────────────────
# PREPROCESSING — different per model
# ─────────────────────────────────────────────────────────────────

def preprocess_densenet(image: Image.Image):
    from tensorflow.keras.applications.densenet import preprocess_input
    img = image.convert("RGB").resize(IMG_SIZE)
    arr = tf.keras.preprocessing.image.img_to_array(img)
    arr = np.expand_dims(arr, axis=0)
    return preprocess_input(arr)

def preprocess_resnet(image: Image.Image):
    from tensorflow.keras.applications.resnet import preprocess_input
    img = image.convert("RGB").resize(IMG_SIZE)
    arr = tf.keras.preprocessing.image.img_to_array(img)
    arr = np.expand_dims(arr, axis=0)
    return preprocess_input(arr)

# ─────────────────────────────────────────────────────────────────
# PREDICT
# ─────────────────────────────────────────────────────────────────

def predict(model, image: Image.Image, model_name: str):
    start = time.time()
    arr   = preprocess_densenet(image) if model_name == "DenseNet121" else preprocess_resnet(image)
    preds = model.predict(arr, verbose=0)
    elapsed    = time.time() - start
    class_idx  = int(np.argmax(preds[0]))
    confidence = float(np.max(preds[0]))
    return CLASS_NAMES[class_idx], confidence, elapsed

# ─────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────

st.markdown("""
<div class="main-header">
    <h1>🪸 Coral Health Classifier</h1>
    <p> Coral reef health detection using DenseNet121 & ResNet152</p>
</div>
""", unsafe_allow_html=True)

st.divider()

# ─────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    st.markdown("---")

    model_choice = st.selectbox(
        " Select Model",
        ["DenseNet121", "ResNet152"],
        help="Choose which model to run inference with"
    )

    st.markdown("---")
    st.markdown(" Classes ")
    for cls in CLASS_NAMES:
        color = CLASS_COLORS[cls]
        emoji = CLASS_EMOJI[cls]
        st.markdown(
            f"<span style='color:{color}; font-weight:600'>{emoji} {cls.replace('_',' ').title()}</span>",
            unsafe_allow_html=True
        )

# ─────────────────────────────────────────────────────────────────
# UPLOAD
# ─────────────────────────────────────────────────────────────────

col_upload, _ = st.columns([2, 1])
with col_upload:
    uploaded_file = st.file_uploader(
        "📤 Upload a coral image",
        type=["jpg", "jpeg", "png"],
        help="Supports JPG and PNG. Input resized to 224×224."
    )

if not uploaded_file:
    st.markdown("""
    <div style="text-align:center; padding: 3rem; color: #7ab8cc;">
        <div style="font-size: 4rem;">🌊</div>
        <div style="font-family:'Syne',sans-serif; font-size:1.2rem; color:#b0d8e8; margin-top:1rem;">
            Upload a coral image to begin classification
        </div>
        <div style="font-size:0.9rem; margin-top:0.5rem;">
            Detects: Healthy Coral · Bleached Coral · Dead Coral
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ─────────────────────────────────────────────────────────────────
# IMAGE INFO
# ─────────────────────────────────────────────────────────────────

image = Image.open(uploaded_file)

col_img, col_meta = st.columns([1, 2])
with col_img:
    st.image(image, caption="Uploaded Image", use_container_width=True)
with col_meta:
    st.markdown('<div class="section-label">📋 Image Info</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="info-row">
        <span class="info-chip">📁 {uploaded_file.name}</span>
        <span class="info-chip">📐 {image.size[0]} × {image.size[1]} px</span>
        <span class="info-chip">🎨 {image.mode}</span>
        <span class="info-chip">📦 {uploaded_file.size // 1024} KB</span>
    </div>
    """, unsafe_allow_html=True)

st.divider()

# ─────────────────────────────────────────────────────────────────
# PREDICTION
# ─────────────────────────────────────────────────────────────────

st.markdown(f'<div class="section-label"> Prediction — {model_choice}</div>',
            unsafe_allow_html=True)

with st.spinner(f"Loading {model_choice}..."):
    model = load_densenet() if model_choice == "DenseNet121" else load_resnet()

with st.spinner("Running inference..."):
    label, confidence, elapsed = predict(model, image, model_choice)

color  = CLASS_COLORS[label]
emoji  = CLASS_EMOJI[label]
pretty = label.replace("_", " ").title()

c1, c2, c3 = st.columns(3)
with c1:
    st.markdown(f"""
    <div class="result-box">
        <div class="sublabel">Predicted Class</div>
        <div class="label" style="color:{color}">{emoji} {pretty}</div>
    </div>
    """, unsafe_allow_html=True)
with c2:
    st.markdown(f"""
    <div class="result-box">
        <div class="sublabel">Confidence</div>
        <div class="confidence">{confidence*100:.2f}%</div>
    </div>
    """, unsafe_allow_html=True)
with c3:
    st.markdown(f"""
    <div class="result-box">
        <div class="sublabel">Inference Time</div>
        <div class="confidence">{elapsed:.3f}s</div>
    </div>
    """, unsafe_allow_html=True)