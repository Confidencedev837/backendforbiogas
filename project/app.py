from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import streamlit as st
import joblib
import numpy as np

# Load models
model_yield = joblib.load("yield_model.pkl")
model_ch4 = joblib.load("ch4_model.pkl")

# Streamlit UI
st.title("🌱 Biogas AI Predictor")
st.write("Enter temperature and pH to predict biogas yield & methane %")

# User inputs
temp = st.number_input("Temperature (°C)", min_value=0.0, max_value=100.0, step=0.1)
ph = st.number_input("pH", min_value=0.0, max_value=14.0, step=0.1)

if st.button("Predict"):
    X_new = np.array([[temp, ph]])
    yield_pred = model_yield.predict(X_new)[0]
    ch4_pred = model_ch4.predict(X_new)[0]

    st.success(f"Predicted Biogas Yield: {yield_pred:.2f} L/g VS")
    st.success(f"Predicted CH₄ Content: {ch4_pred:.2f}%")


# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],  # allow all origins (or use specific: ["http://127.0.0.1:5500"])
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# import pickle

# with open("yield_model.pkl") as f:
#     data=pickle.load(f);
# print(data)