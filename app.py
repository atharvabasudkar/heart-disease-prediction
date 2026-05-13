import streamlit as st
import numpy as np
import pandas as pd
import joblib

# =========================
# LOAD FILES
# =========================

model = joblib.load("KNN_heart.pkl")
scaler = joblib.load("heart_scaler.pkl")
expected_columns = joblib.load("heart_column.pkl")

# =========================
# TITLE
# =========================

st.title("❤️ Heart Stroke Prediction By Atharva Basudkar")

st.write("Enter patient details below")

# =========================
# USER INPUTS
# =========================

age = st.number_input("Age", 1, 120)

sex = st.selectbox("Sex", ["Male", "Female"])

resting_bp = st.number_input("Resting Blood Pressure")

chol = st.number_input("Cholesterol")

fbs = st.selectbox("Fasting Blood Sugar", [0, 1])

max_hr = st.number_input("Maximum Heart Rate")

oldpeak = st.number_input("Oldpeak", 0.0, 10.0)

# =========================
# CHEST PAIN TYPE
# =========================

cp = st.selectbox(
    "Chest Pain Type",
    ["ATA", "NAP", "TA", "ASY"]
)

# =========================
# REST ECG
# =========================

restecg = st.selectbox(
    "Rest ECG",
    ["Normal", "ST", "LVH"]
)

# =========================
# EXERCISE ANGINA
# =========================

exang = st.selectbox(
    "Exercise Induced Angina",
    ["Yes", "No"]
)

# =========================
# ST SLOPE
# =========================

slope = st.selectbox(
    "ST Slope",
    ["Flat", "Up", "Down"]
)

# =========================
# PREDICTION BUTTON
# =========================

if st.button("Predict"):

    # Initialize all columns with 0
    input_data = dict.fromkeys(expected_columns, 0)

    # Numerical Features
    input_data['Age'] = age
    input_data['RestingBP'] = resting_bp
    input_data['Cholesterol'] = chol
    input_data['FastingBS'] = fbs
    input_data['MaxHR'] = max_hr
    input_data['Oldpeak'] = oldpeak

    # Sex
    if sex == "Male":
        input_data['Sex_M'] = 1

    # Chest Pain Type
    if cp == "ATA":
        input_data['ChestPainType_ATA'] = 1

    elif cp == "NAP":
        input_data['ChestPainType_NAP'] = 1

    elif cp == "TA":
        input_data['ChestPainType_TA'] = 1

    # ASY becomes all zeros automatically

    # Rest ECG
    if restecg == "Normal":
        input_data['RestingECG_Normal'] = 1

    elif restecg == "ST":
        input_data['RestingECG_ST'] = 1

    # LVH becomes all zeros automatically

    # Exercise Angina
    if exang == "Yes":
        input_data['ExerciseAngina_Y'] = 1

    # ST Slope
    if slope == "Flat":
        input_data['ST_Slope_Flat'] = 1

    elif slope == "Up":
        input_data['ST_Slope_Up'] = 1

    # Down becomes all zeros automatically

    # =========================
    # CONVERT TO DATAFRAME
    # =========================

    input_df = pd.DataFrame([input_data])

    # Remove target column if present
    if 'HeartDisease' in input_df.columns:
        input_df = input_df.drop('HeartDisease', axis=1)

    # =========================
    # SCALE INPUT
    # =========================

    input_scaled = scaler.transform(input_df)

    # =========================
    # PREDICTION
    # =========================

    prediction = model.predict(input_scaled)

    # =========================
    # RESULT
    # =========================

    if prediction[0] == 1:
        st.error("⚠️ High Chance of Heart Disease")

    else:
        st.success("✅ Low Chance of Heart Disease")