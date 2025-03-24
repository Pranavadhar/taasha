import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import requests

firebase_url = st.secrets["firebase"]["url"]
firebase_auth_token = st.secrets["firebase"]["auth_token"]

# Function to fetch data from Firebase using REST API
def fetch_firebase_data():
    try:
        response = requests.get(f'{firebase_url}/parameters.json?auth={firebase_auth_token}')
        if response.ok:
            entries = response.json()
            return entries
        else:
            st.error("Error fetching data from Firebase.")
            return None
    except requests.exceptions.RequestException as e:
        st.error(f"Network error: {e}")
        return None

# Fetch Firebase data
entries = fetch_firebase_data()
DATA_PATH = "real_BABY.csv"  
try:
    data = pd.read_csv(DATA_PATH)
    data.dropna(inplace=True)
except FileNotFoundError:
    st.error("Dataset file not found. Please check the path and file name.")
    st.stop()

input_features = ["Timestamp", "volData", "currentData"]
output_features = ["batTempData", "socData", "sohData", "motTempData", "speedData"]

x = data[input_features]
y = data[output_features]

scaler_x = StandardScaler()
scaler_y = StandardScaler()
x_scaled = scaler_x.fit_transform(x)
y_scaled = scaler_y.fit_transform(y)

x_train, x_test, y_train, y_test = train_test_split(x_scaled, y_scaled, test_size=0.2, random_state=42)

models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "Random Forest": RandomForestRegressor(random_state=42, n_estimators=25),
}

for model in models.values():
    model.fit(x_train, y_train)

def detect_faults(predictions):
    faults = []
    if predictions[0] > 32:  # Battery temperature
        faults.append("Battery Temperature > 32°C")
    if predictions[1] < 85:  # SOC
        faults.append("SOC < 85%")
    if predictions[2] < 85:  # SOH
        faults.append("SOH < 85%")
    if predictions[3] > 32:  # Motor temperature
        faults.append("Motor Temperature > 32°C")
    if predictions[4] < 57:  # Motor speed
        faults.append("Motor Speed < 57")
    return faults

st.sidebar.header("Model and Input Selection")
selected_model_name = st.sidebar.selectbox("Select a model", list(models.keys()))
selected_model = models[selected_model_name]

timestamp = st.sidebar.number_input("Timestamp", min_value=0.0, value=10.0, step=0.1)

# Ensure fetched values are correctly formatted as float
vol_data = float(entries.get("systemVoltage", 0.0))
current_data = float(entries.get("current", 0.0))  # Keeping in mA

st.write("### Fetched Data from Firebase:")
st.write(f"**Voltage:** {vol_data} V")
st.write(f"**Current:** {current_data}")  # Display as mA

# Predict current and future states
sample_input = np.array([[timestamp, vol_data, current_data]], dtype=float)
sample_input_scaled = scaler_x.transform(sample_input)
pred_scaled = selected_model.predict(sample_input_scaled)
pred = scaler_y.inverse_transform(pred_scaled).flatten()

next_timestamp = timestamp + 10
next_input = np.array([[next_timestamp, vol_data, current_data]], dtype=float)
next_input_scaled = scaler_x.transform(next_input)
next_pred_scaled = selected_model.predict(next_input_scaled)
next_pred = scaler_y.inverse_transform(next_pred_scaled).flatten()

st.subheader("Predictions")
st.write(f"**Model Used:** {selected_model_name}")
st.write(f"**Predicted Outputs for Timestamp {timestamp}:**")
st.write(dict(zip(output_features, pred)))

st.write(f"**Predicted Outputs for Timestamp {next_timestamp}:**")
st.write(dict(zip(output_features, next_pred)))

st.subheader("Fault Detection")
faults = detect_faults(pred)
if faults:
    st.error("Faults Detected:")
    for fault in faults:
        st.write(f"- {fault}")
else:
    st.success("No faults detected.")

st.header("Prediction Visualization")
fig, axs = plt.subplots(2, 1, figsize=(10, 8))

# Current predictions
axs[0].bar(output_features, pred, color="blue", alpha=0.7)
axs[0].set_title(f"Predicted Outputs for Timestamp {timestamp}")
axs[0].set_ylabel("Values")
axs[0].grid(True)

# Future predictions
axs[1].bar(output_features, next_pred, color="green", alpha=0.7)
axs[1].set_title(f"Predicted Outputs for Timestamp {next_timestamp}")
axs[1].set_ylabel("Values")
axs[1].grid(True)

plt.tight_layout()
st.pyplot(fig)
