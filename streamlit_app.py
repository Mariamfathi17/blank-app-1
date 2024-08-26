import streamlit as st
st.title("Diabetes Prediction  ")
# User input fields
pregnancies = st.number_input("Number of Pregnancies", 0, 20, step=1)
glucose = st.number_input("Glucose Level", 0, 200)
blood_pressure = st.number_input("Blood Pressure (mm Hg)", 0, 122)
skin_thickness = st.number_input("Skin Thickness (mm)", 0, 100)
insulin = st.number_input("Insulin Level (mu U/ml)", 0, 846)
bmi = st.number_input("BMI (Body Mass Index)", 0.0, 67.1)
dpf = st.number_input("Diabetes Pedigree Function", 0.0, 2.5)
age = st.number_input("Age", 21, 81)

# Prediction
user_data = pd.DataFrame({
    'Pregnancies': [pregnancies],
    'Glucose': [glucose],
    'BloodPressure': [blood_pressure],
    'SkinThickness': [skin_thickness],
    'Insulin': [insulin],
    'BMI': [bmi],
    'DiabetesPedigreeFunction': [dpf],
    'Age': [age]
})

user_data_scaled = scaler.transform(user_data)
prediction = model.predict(user_data_scaled)

# Display the result
if st.button('Predict'):
    if prediction[0] == 1:
        st.write("The model predicts that you **have diabetes**.")
    else:
        st.write("The model predicts that you **do not have diabetes**.")


