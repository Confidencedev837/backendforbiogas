import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import joblib


data = pd.read_csv("your_data.csv")


X = data[["Temperature (°C)", "pH"]]
y_yield = data["Biogas Yield (m³/Kg)"]
y_ch4 = data["CH4 (%)"]


X_train, X_test, y_yield_train, y_yield_test = train_test_split(
    X, y_yield, test_size=0.2, random_state=42
)
_, _, y_ch4_train, y_ch4_test = train_test_split(
    X, y_ch4, test_size=0.2, random_state=42
)


model_yield = RandomForestRegressor(n_estimators=100, random_state=42)
model_yield.fit(X_train, y_yield_train)

model_ch4 = RandomForestRegressor(n_estimators=100, random_state=42)
model_ch4.fit(X_train, y_ch4_train)


joblib.dump(model_yield, "yield_model.pkl")
joblib.dump(model_ch4, "ch4_model.pkl")

print(" Models have been  trained and saved!")
