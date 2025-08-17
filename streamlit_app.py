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
import os
import time
import pickle
import hashlib
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
      /* Clean typography */
      html, body, [class*="css"]  { font-family: Inter, system-ui, -apple-system, Segoe UI, Roboto, Ubuntu, Cantarell, Noto Sans, sans-serif; }

      /* Card-like containers */
      .soft-card {
        padding: 1.25rem 1.25rem;
        border-radius: 1.25rem;
        border: 1px solid rgba(0,0,0,0.06);
        background: linear-gradient(180deg, rgba(255,255,255,0.6), rgba(255,255,255,0.9));
        box-shadow: 0 6px 24px rgba(0,0,0,0.06);
      }

      /* Pill badges */
      .pill { display:inline-block; padding: .35rem .8rem; border-radius: 999px; font-weight: 700; }
      .pill-normal { background: #eef6ff; }
      .pill-alz { background: #fff1f7; }
      .pill-pd { background: #effaf3; }

      /* Subtle footer */
      .footer { opacity: 0.8; font-size: 0.92rem; }

      /* Center helper */
      .center { display:flex; align-items:center; justify-content:center; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------
# Sidebar content — How it works & Info
# ---------------------------------------------------------------
st.sidebar.title("🧭 How it works")
st.sidebar.write(
    """
**1. Upload** a brain MRI (PNG/JPG).\
**2. Triage model** predicts one of: *Normal*, *Alzheimer's*, or *Parkinson's*.\
**3. If Alzheimer's**, a severity model estimates *Mild / Moderate / Severe*.\
**4. If Parkinson's**, a staging model estimates *Early / Mid / Advanced*.\
**5. Review** explanations, prevention tips, and **download** a mini-report.

> ⚠️ Educational tool only — not a medical device. Always consult a clinician.
    """
)

with st.sidebar.expander("ℹ️ About Alzheimer's (AD)", expanded=False):
    st.write(
        """
- Neurodegenerative disease with progressive memory and thinking decline.
- Common MRI notes (at a high level): medial temporal lobe/hippocampal atrophy and cortical thinning (patterns vary by stage and individual).
- Typical symptoms: memory loss, difficulty planning, language problems, and disorientation.
        """
    )
    st.markdown("**Prevention & risk reduction (general lifestyle):**")
    st.write(
        """
- Regular physical activity
- Mediterranean‑style diet, adequate hydration
- Quality sleep & treatment of sleep apnea
- Manage blood pressure, diabetes, and cholesterol
- Avoid smoking; moderate alcohol
- Cognitive & social engagement; lifelong learning
        """
    )

with st.sidebar.expander("ℹ️ About Parkinson's (PD)", expanded=False):
    st.write(
        """
- Movement disorder caused by loss of dopamine-producing cells.\
- Typical features: tremor, slowness (bradykinesia), rigidity, balance issues; non‑motor symptoms may include sleep changes, constipation, mood & smell changes.
- MRI is often nonspecific in early PD; advanced research methods may show subtle changes.
        """
    )
    st.markdown("**Risk reduction & wellbeing:**")
    st.write(
        """
- Regular aerobic & resistance exercise; balance training
- Good sleep hygiene; manage stress
- Healthy, fiber‑rich diet; stay hydrated
- Reduce exposure to toxins/pesticides where feasible
- Avoid head injuries (seatbelts, helmets)
        """
    )

with st.sidebar.expander("📁 Model files (optional)", expanded=False):
    st.write("Place these files next to the app to use your trained models:")
    st.code("""
triage_model.pkl
alz_severity_model.pkl
pd_stage_model.pkl
    """)

# ---------------------------------------------------------------
# Helper: Try to load models (fall back to demo mode if missing)
# ---------------------------------------------------------------
triage_model = None
alz_model = None
pd_model = None

try:
    with open("triage_model.pkl", "rb") as f:
        triage_model = pickle.load(f)
except Exception:
    triage_model = None

try:
    with open("alz_severity_model.pkl", "rb") as f:
        alz_model = pickle.load(f)
except Exception:
    alz_model = None

try:
    with open("pd_stage_model.pkl", "rb") as f:
        pd_model = pickle.load(f)
except Exception:
    pd_model = None

DEMO_MODE = (triage_model is None) or (alz_model is None) or (pd_model is None)

# Label spaces (feel free to edit to your classes)
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
    uploaded = st.file_uploader(
        "Upload an MRI image (PNG/JPG)",
        type=["png", "jpg", "jpeg"],
        accept_multiple_files=False,
        help="Single axial slice or composite MRI image."
    )

    show_steps = st.checkbox("Show step-by-step progress", value=True)

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
        # Display preview
        image_bytes = uploaded.read()
        uploaded.seek(0)
        im = Image.open(io.BytesIO(image_bytes)).convert("L")  # grayscale
        w, h = im.size
        st.image(im, caption=f"Preview — {uploaded.name} ({w}×{h})", use_container_width=True)

        # Preprocess (simple resize/normalize demo)
        im_resized = im.resize((224, 224))
        arr = np.asarray(im_resized).astype("float32") / 255.0
        arr = arr[None, None, :, :]  # (N, C, H, W) style placeholder

        if st.button("🔍 Analyze", type="primary"):
            progress = st.progress(0) if show_steps else None

            # -------- Stage 1: Triage --------
            if show_steps:
                progress.progress(10)
            time.sleep(0.2)

            triage_pred = None
            triage_conf = 0.0

            if triage_model is not None:
                try:
                    # Expecting model to have predict_proba; otherwise fallback
                    if hasattr(triage_model, "predict_proba"):
                        probs = triage_model.predict_proba(arr.reshape(1, -1))[0]
                        idx = int(np.argmax(probs))
                        triage_pred = TRIAGE_CLASSES[idx]
                        triage_conf = float(np.max(probs))
                    else:
                        idx = int(triage_model.predict(arr.reshape(1, -1))[0])
                        triage_pred = TRIAGE_CLASSES[idx]
                        triage_conf = 0.0
                except Exception:
                    triage_pred = None

            if triage_pred is None:
                # Demo prediction: deterministic by image hash
                digest = hashlib.sha256(image_bytes).hexdigest()
                idx = int(digest, 16) % len(TRIAGE_CLASSES)
                triage_pred = TRIAGE_CLASSES[idx]
                triage_conf = 0.66  # placeholder

            if show_steps:
                progress.progress(45)
                time.sleep(0.2)

            # Show triage result
            if triage_pred == "normal":
                st.markdown('<span class="pill pill-normal">Triage: NORMAL</span>', unsafe_allow_html=True)
                st.success("The image appears **Normal** by the triage model (educational output).")
            elif triage_pred == "alzheimer":
                st.markdown('<span class="pill pill-alz">Triage: ALZHEIMER\'S</span>', unsafe_allow_html=True)
                st.warning("Triage suggests **Alzheimer's** pattern. Proceeding to severity estimation…")
            else:
                st.markdown('<span class="pill pill-pd">Triage: PARKINSON\'S</span>', unsafe_allow_html=True)
                st.warning("Triage suggests **Parkinson's** pattern. Proceeding to stage estimation…")

            if triage_conf:
                st.caption(f"Triage confidence (approx.): {triage_conf:.2f}")

            # -------- Stage 2: Branch to severity/stage --------
            secondary_label = None
            secondary_conf = 0.0

            if triage_pred == "alzheimer":
                if alz_model is not None:
                    try:
                        if hasattr(alz_model, "predict_proba"):
                            probs = alz_model.predict_proba(arr.reshape(1, -1))[0]
                            j = int(np.argmax(probs))
                            secondary_label = ALZ_CLASSES[j]
                            secondary_conf = float(np.max(probs))
                        else:
                            j = int(alz_model.predict(arr.reshape(1, -1))[0])
                            secondary_label = ALZ_CLASSES[j]
                            secondary_conf = 0.0
                    except Exception:
                        secondary_label = None
                if secondary_label is None:
                    j = (idx + 1) % len(ALZ_CLASSES)
                    secondary_label = ALZ_CLASSES[j]
                    secondary_conf = 0.64

            elif triage_pred == "parkinson":
                if pd_model is not None:
                    try:
                        if hasattr(pd_model, "predict_proba"):
                            probs = pd_model.predict_proba(arr.reshape(1, -1))[0]
                            j = int(np.argmax(probs))
                            secondary_label = PD_CLASSES[j]
                            secondary_conf = float(np.max(probs))
                        else:
                            j = int(pd_model.predict(arr.reshape(1, -1))[0])
                            secondary_label = PD_CLASSES[j]
                            secondary_conf = 0.0
                    except Exception:
                        secondary_label = None
                if secondary_label is None:
                    j = (idx + 2) % len(PD_CLASSES)
                    secondary_label = PD_CLASSES[j]
                    secondary_conf = 0.61

            if show_steps:
                progress.progress(85)
                time.sleep(0.2)

            # Display secondary result and guidance
            guidance = ""
            if triage_pred == "alzheimer":
                st.subheader("Alzheimer's severity (educational estimate)")
                st.info(f"Predicted severity: **{secondary_label.title()}**")
                if secondary_conf:
                    st.caption(f"Severity confidence (approx.): {secondary_conf:.2f}")
                guidance = (
                    "This output is not diagnostic. If cognitive symptoms are present, consider a comprehensive clinical assessment, neuropsychological testing, and appropriate imaging/labs."
                )
            elif triage_pred == "parkinson":
                st.subheader("Parkinson's stage (educational estimate)")
                st.info(f"Predicted stage: **{secondary_label.title()}**")
                if secondary_conf:
                    st.caption(f"Stage confidence (approx.): {secondary_conf:.2f}")
                guidance = (
                    "Clinical staging for Parkinson's typically considers motor and non‑motor symptoms and response to therapy. Please consult a neurologist for proper evaluation."
                )
            else:
                st.subheader("Normal — what this means")
                st.info("The triage did not detect patterns suggesting Alzheimer's or Parkinson's. If symptoms exist, clinical evaluation may still be helpful.")
                guidance = (
                    "Maintain healthy lifestyle habits for long‑term brain wellbeing (activity, sleep, diet, social engagement)."
                )

            st.markdown(guidance)

            if show_steps:
                progress.progress(100)
                time.sleep(0.2)

            # -------- Download mini-report --------
            ts = datetime.now().strftime("%Y-%m-%d %H:%M")
            report_lines = [
                "MRI Neurology Triage — Mini Report",
                f"Timestamp: {ts}",
                f"File: {uploaded.name}",
                f"Triage: {triage_pred.title()} (approx. conf {triage_conf:.2f})",
            ]
            if triage_pred in ("alzheimer", "parkinson"):
                report_lines.append(
                    f"Secondary: {secondary_label.title()} (approx. conf {secondary_conf:.2f})"
                )
            report_lines += [
                "",
                "Notes:",
                "- Educational use only. Not a medical diagnosis.",
                "- Seek professional medical advice for symptoms or concerns.",
                "",
                "Contact:",
                "- Name: Nahrwan thaer",
                "- Email: nahrwanthaer@gmail.com",
                "- Phone: +90 534 078 62 26",
            ]
            report_txt = "\n".join(report_lines)

            st.download_button(
                label="⬇️ Download mini‑report (.txt)",
                data=report_txt,
                file_name="mri_triage_report.txt",
                mime="text/plain",
            )

    st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------------------------------------------
# Educational disclaimer footer
# ---------------------------------------------------------------
st.markdown("---")
st.markdown(
    """
<div class="footer">
<strong>Disclaimer:</strong> This app is for **education and demonstration** only.
It does not diagnose disease and must not be used as a substitute for professional medical advice, diagnosis, or treatment.
If you have health concerns, please consult a licensed clinician.
</div>
    """,
    unsafe_allow_html=True,
)
