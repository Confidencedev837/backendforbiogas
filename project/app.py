from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import joblib
import numpy as np


model_yield = joblib.load("project/yield_model.pkl")
model_ch4 = joblib.load("project/ch4_model.pkl")

app = FastAPI()

@app.get("/")
def root():
    return {"message": "Biogas AI API is running 🚀"}

@app.get("/predict")
def predict(temp: float, pH: float):
    X_new = np.array([[temp, pH]])
    yield_pred = model_yield.predict(X_new)[0]
    ch4_pred = model_ch4.predict(X_new)[0]
    return {
        "biogas_yield": float(yield_pred),
        "ch4_percent": float(ch4_pred)
    }


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # allow all origins (or use specific: ["http://127.0.0.1:5500"])
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# import pickle

# with open("yield_model.pkl") as f:
#     data=pickle.load(f);

# print(data)


