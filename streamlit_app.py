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
    skin_thickness = st.number_input("Skin Thickness (mm)", min_value=0, max_value=100, step=1, value=20)

with col2:
    insulin = st.number_input("Insulin Level (mu U/ml)", min_value=0, max_value=846, step=1, value=30)
    bmi = st.number_input("BMI (Body Mass Index)", min_value=0.0, max_value=67.1, step=0.1, value=25.0)
    diabetes_pedigree_function = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=2.5, step=0.01, value=0.5)
    age = st.number_input("Age", min_value=21, max_value=81, step=1, value=30)

# Prediction button
if st.button("Predict"):
    # Collect user input into a numpy array
    user_input = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree_function, age]])

    # Scale the input data using StandardScaler
    scaler = StandardScaler()
    user_input_scaled = scaler.fit_transform(user_input)

    # Make a prediction and get the probability
    prediction = model.predict(user_input_scaled)
    prediction_proba = model.predict_proba(user_input_scaled)

    # Display the result
    if prediction[0] == 1:
        st.error(f"The model predicts that you **have diabetes** with a probability of {prediction_proba[0][1]:.2f}. Please consult a healthcare professional.")
    else:
        st.success(f"The model predicts that you **do not have diabetes** with a probability of {prediction_proba[0][0]:.2f}. Keep up the healthy lifestyle!")

# Footer
st.markdown("""
    <hr>
    <p style='text-align: center;'>
    This app is for educational purposes only. Always consult a healthcare professional for medical advice.
    </p>
    """, unsafe_allow_html=True)
