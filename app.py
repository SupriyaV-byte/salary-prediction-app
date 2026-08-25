import streamlit as st
import joblib
import numpy as np

st.title("💼 Employee Salary Prediction")

st.write("Enter your details to predict an estimated salary.")

experience = st.number_input(
    "Years of Experience",
    min_value=0.0,
    max_value=40.0,
    value=2.0
)

education = st.selectbox(
    "Education Level",
    ["High School", "Bachelor's", "Master's", "PhD"]
)

education_map = {
    "High School": 0,
    "Bachelor's": 1,
    "Master's": 2,
    "PhD": 3
}

if st.button("Predict Salary"):

    model = joblib.load("salary_model.pkl")

    x = np.array([
        [experience, education_map[education]]
    ])

    prediction = model.predict(x)[0]

    st.success(
        f"Estimated Salary: ₹{prediction:,.0f}"
    )