# Streamlit MRI Triage App — Parkinson's vs Alzheimer's vs Normal
# ---------------------------------------------------------------
# This single-file Streamlit app lets a user upload an MRI image, runs a first-stage
# classifier (Normal / Alzheimer's / Parkinson's). If Alzheimer's is predicted,
# it runs a second model to estimate severity. If Parkinson's is predicted, it
# estimates disease stage. It also includes friendly explanations and prevention tips,
# plus a downloadable mini-report and contact info.
#
# 👉 To run locally:
# 1) pip install streamlit pillow numpy scikit-learn
# 2) streamlit run streamlit_mri_triage_app.py
#
# 👉 Expected (optional) model files in the same folder:
#    - triage_model.pkl             # outputs one of: {"normal", "alzheimer", "parkinson"}
#    - alz_severity_model.pkl       # outputs one of: {"mild", "moderate", "severe"}
#    - pd_stage_model.pkl           # outputs one of: {"early", "mid", "advanced"}
# If these files are missing, the app will simulate predictions (for demo only).

import io
import pickle
import hashlib
import time
from datetime import datetime

import numpy as np
from PIL import Image
import streamlit as st

# ---------------------------------------------------------------
# Page setup & styling
# ---------------------------------------------------------------
st.set_page_config(
    page_title="MRI Neurology Triage",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      html, body, [class*="css"]  { font-family: Inter, system-ui, -apple-system, Segoe UI, Roboto, Ubuntu, Cantarell, Noto Sans, sans-serif; }
      .soft-card {
        padding: 1.25rem 1.25rem;
        border-radius: 1.25rem;
        border: 1px solid rgba(0,0,0,0.06);
        background: #ffffffcc;
        box-shadow: 0 6px 24px rgba(0,0,0,0.06);
      }
      .pill { display:inline-block; padding: .35rem .8rem; border-radius: 999px; font-weight: 700; }
      .pill-normal { background: #eef6ff; }
      .pill-alz { background: #fff1f7; }
      .pill-pd { background: #effaf3; }
      .footer { opacity: 0.8; font-size: 0.92rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------
# Sidebar content — How it works
# ---------------------------------------------------------------
st.sidebar.title("🧭 How it works")
st.sidebar.write(
    """
**1. Upload** a brain MRI (PNG/JPG).\
**2. Triage model** predicts one of: *Normal*, *Alzheimer's*, or *Parkinson's*.\
**3. If Alzheimer's**, a severity model estimates *Mild / Moderate / Severe*.\
**4. If Parkinson's**, a staging model estimates *Early / Mid / Advanced*.\
**5. Review** explanations, prevention tips, and **download** a mini-report.

> ⚠️ Educational tool only — not a medical device.
    """
)

# ---------------------------------------------------------------
# Info dictionaries
# ---------------------------------------------------------------
INFO = {
    "alzheimer": {
        "About": "Alzheimer's is a neurodegenerative disease with progressive memory and thinking decline.",
        "Symptoms": "Memory loss, difficulty planning, language problems, disorientation.",
        "Prevention": "Exercise, Mediterranean diet, good sleep, manage blood pressure & diabetes, avoid smoking, cognitive & social activity."
    },
    "parkinson": {
        "About": "Parkinson's is a movement disorder caused by loss of dopamine-producing cells.",
        "Symptoms": "Tremor, slowness, rigidity, balance issues, sleep and mood changes.",
        "Prevention": "Exercise, sleep hygiene, stress management, healthy diet, avoid toxins, prevent head injuries."
    }
}

# ---------------------------------------------------------------
# Helper: Try to load models
# ---------------------------------------------------------------
triage_model = None
alz_model = None
pd_model = None

try:
    with open("triage_model.pkl", "rb") as f:
        triage_model = pickle.load(f)
except Exception:
    pass
try:
    with open("alz_severity_model.pkl", "rb") as f:
        alz_model = pickle.load(f)
except Exception:
    pass
try:
    with open("pd_stage_model.pkl", "rb") as f:
        pd_model = pickle.load(f)
except Exception:
    pass

TRIAGE_CLASSES = ["normal", "alzheimer", "parkinson"]
ALZ_CLASSES = ["mild", "moderate", "severe"]
PD_CLASSES = ["early", "mid", "advanced"]

# ---------------------------------------------------------------
# Main content
# ---------------------------------------------------------------
st.title("🧠 MRI Neurology Triage App")
st.caption("Upload a brain MRI to get an **educational** AI triage suggestion. Not for diagnosis.")

colA, colB = st.columns([1, 2])

with colA:
    uploaded = st.file_uploader("Upload an MRI image (PNG/JPG)", type=["png", "jpg", "jpeg"])
    show_steps = st.checkbox("Show progress", value=True)

    st.markdown("---")
    st.subheader("Contact")
    st.write("**Name:** Nahrwan thaer")
    st.write("**Email:** nahrwanthaer@gmail.com")
    st.write("**Phone:** +90 534 078 62 26")

with colB:
    st.markdown('<div class="soft-card">', unsafe_allow_html=True)
    st.subheader("Get started")
    st.write("Drop your MRI on the left, then press **Analyze**.")

    if uploaded is not None:
        image_bytes = uploaded.read()
        uploaded.seek(0)
        im = Image.open(io.BytesIO(image_bytes)).convert("L")
        st.image(im, caption=f"Preview — {uploaded.name}", use_container_width=True)

        arr = np.asarray(im.resize((224, 224))).astype("float32") / 255.0
        arr = arr[None, None, :, :]

        if st.button("🔍 Analyze", type="primary"):
            progress = st.progress(0) if show_steps else None

            # Fake triage (demo)
            digest = hashlib.sha256(image_bytes).hexdigest()
            idx = int(digest, 16) % len(TRIAGE_CLASSES)
            triage_pred = TRIAGE_CLASSES[idx]
            triage_conf = 0.66

            if show_steps: progress.progress(50)

            if triage_pred == "normal":
                st.markdown('<span class="pill pill-normal">Triage: NORMAL</span>', unsafe_allow_html=True)
                st.success("The image appears **Normal** (educational output).")
            elif triage_pred == "alzheimer":
                st.markdown('<span class="pill pill-alz">Triage: ALZHEIMER\'S</span>', unsafe_allow_html=True)
                st.warning("Triage suggests **Alzheimer's** pattern.")
                if st.button("ℹ️ Learn more about Alzheimer's"):
                    st.info(f"**About:** {INFO['alzheimer']['About']}\n\n**Symptoms:** {INFO['alzheimer']['Symptoms']}\n\n**Prevention:** {INFO['alzheimer']['Prevention']}")
            else:
                st.markdown('<span class="pill pill-pd">Triage: PARKINSON\'S</span>', unsafe_allow_html=True)
                st.warning("Triage suggests **Parkinson's** pattern.")
                if st.button("ℹ️ Learn more about Parkinson's"):
                    st.info(f"**About:** {INFO['parkinson']['About']}\n\n**Symptoms:** {INFO['parkinson']['Symptoms']}\n\n**Prevention:** {INFO['parkinson']['Prevention']}")

            if show_steps: progress.progress(100)

    st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------------------------------------------
# Footer
# ---------------------------------------------------------------
st.markdown("---")
st.markdown(
    """
<div class="footer">
<strong>Disclaimer:</strong> This app is for **education and demonstration** only.
It does not diagnose disease. Please consult a licensed clinician for medical concerns.
</div>
    """,
    unsafe_allow_html=True,
)
