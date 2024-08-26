import streamlit as st
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler

# Load the trained model
model = joblib.load('diabetes_model.pkl')

# Set up the Streamlit app
st.set_page_config(page_title="Diabetes Prediction App", layout="centered")

# Title and description
st.title("Diabetes Prediction App")
st.write("Fill in the details below to predict the likelihood of diabetes.")

# User-friendly input form with side-by-side layout
col1, col2 = st.columns(2)

with col1:
    pregnancies = st.number_input("Number of Pregnancies", min_value=0, max_value=20, step=1, value=1)
    glucose = st.number_input("Glucose Level", min_value=0, max_value=200, step=1, value=100)
    blood_pressure = st.number_input("Blood Pressure (mm Hg)", min_value=0, max_value=122, step=1, value=70)
    skin_thickness = st.number_input("Skin Thickness (mm)"), min_value=0,
