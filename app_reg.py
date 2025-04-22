import streamlit as st
import pandas as pd
import numpy as np
import re
import requests
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

firebase_url = st.secrets["firebase"]["url"]
firebase_auth_token = st.secrets["firebase"]["auth_token"]

def fetch_firebase_data():
    try:
        response = requests.get(f'{firebase_url}/parameters.json?auth={firebase_auth_token}')
        if response.ok:
            return response.json()
        else:
            st.error("Error fetching data from Firebase.")
            return None
    except requests.exceptions.RequestException as e:
        st.error(f"Network error: {e}")
        return None

entries = fetch_firebase_data()
DATA_PATH = "new_Baby.csv"  # Updated path as requested
try:
    data = pd.read_csv(DATA_PATH)
    data.dropna(inplace=True)
except FileNotFoundError:
    st.error("Dataset file not found. Please check the path and file name.")
    st.stop()

input_features = ["volData", "currentData"]
output_features = ["batTempData", "socData", "sohData", "motTempData"]

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

st.sidebar.header("Model Selection")
selected_model_name = st.sidebar.selectbox("Select a model", list(models.keys()))
selected_model = models[selected_model_name]

vol_data = float(entries.get("systemVoltage", 0.0))
current_str = str(entries.get("current", "0.0 mA"))
current_match = re.search(r"[\d\.]+", current_str)
current_data = float(current_match.group()) if current_match else 0.0

def detect_faults(predictions):
    faults = []
    if predictions[0] > 35:
        faults.append("Battery Temperature > 35°C - SYSTEM COOLING ACTIVELY")
    if predictions[0] > 45:
        faults.append("Battery Temperature > 45°C - SYSTEM OVER HEATING : COOLING SYSTEM CHECK UP RECOMMENDED")
    if predictions[1] < 30:
        faults.append("SOC < 30% - LOW BATTERY PLUG IN CHARGE")
    if predictions[2] < 89:
        faults.append("SOH < 89% - BATTERY SERVICE RECOMMENDED")
    if predictions[3] > 35:
        faults.append("Motor Temperature > 35°C - SYSTEM COOLING ACTIVELY")
    if predictions[3] > 45:
        faults.append("Motor Temperature > 45°C - SYSTEM OVER HEATING : COOLING SYSTEM CHECK UP RECOMMENDED")
    return faults

st.title("PMEV - REG MODELS")

st.write("### Fetched Data from Firebase:")
st.write(f"**Voltage:** {vol_data} V")
st.write(f"**Current:** {current_data} mA")

# Check if the vehicle is in motion or not
if 0 <= current_data <= 1:
    st.info("Vehicle is not in motion - refrain from referring to the prediction.")

else:
    # Prediction section
    sample_input = np.array([[vol_data, current_data]], dtype=float)
    sample_input_scaled = scaler_x.transform(sample_input)
    pred_scaled = selected_model.predict(sample_input_scaled)
    pred = scaler_y.inverse_transform(pred_scaled).flatten()

    st.subheader("Predicted Values for Fetched Data")
    st.write(f"**Model Used:** {selected_model_name}")
    st.write(f"**Battery Temperature:** {pred[0]:.2f} °C")
    st.write(f"**SOC (State of Charge):** {pred[1]:.2f} %")
    st.write(f"**SOH (State of Health):** {pred[2]:.2f} %")
    st.write(f"**Motor Temperature:** {pred[3]:.2f} °C")

    st.subheader("Predicted Values Bar Chart")
    fig_bar, ax_bar = plt.subplots(figsize=(8, 5))
    ax_bar.bar(output_features, pred, color=['blue', 'blue', 'blue', 'blue'])
    ax_bar.set_ylabel("Predicted Values")
    ax_bar.set_title("Predicted Output for Fetched Data")
    st.pyplot(fig_bar)

    # Fault Detection
    st.subheader("Fault Detection")
    faults = detect_faults(pred)
    if faults:
        st.error("Faults Detected:")
        for fault in faults:
            st.write(f"- {fault}")
    else:
        st.success("No faults detected.")

    # Plus-Minus Analysis
    st.subheader("Plus-Minus Analysis")
    currents = list(range(int(current_data - 5), int(current_data + 6)))
    voltages = list(np.linspace(vol_data - 5, vol_data + 5, 11))

    plus_minus_results = []
    for c, v in zip(currents, voltages):
        test_input = np.array([[v, c]], dtype=float)
        test_input_scaled = scaler_x.transform(test_input)
        test_pred_scaled = selected_model.predict(test_input_scaled)
        test_pred = scaler_y.inverse_transform(test_pred_scaled).flatten()
        plus_minus_results.append([c, v, *test_pred])

    plus_minus_df = pd.DataFrame(plus_minus_results, columns=["Current (mA)", "Voltage (V)", "Battery Temp (°C)", "SOC (%)", "SOH (%)", "Motor Temp (°C)"])
    st.write(plus_minus_df)

    st.header("Trend of Plus-Minus Predictions")
    fig, ax = plt.subplots(figsize=(12, 6))
    for feature in ["Battery Temp (°C)", "SOC (%)", "SOH (%)", "Motor Temp (°C)"]:
        ax.plot(plus_minus_df["Current (mA)"], plus_minus_df[feature], label=feature)

    ax.set_xlabel("Current (mA)")
    ax.set_ylabel("Predicted Values")
    ax.set_title("Predictions for Plus-Minus Analysis")
    ax.legend()
    st.pyplot(fig)

    # Plot for Voltage vs Predicted Values
    fig, ax = plt.subplots(figsize=(12, 6))
    for feature in ["Battery Temp (°C)", "SOC (%)", "SOH (%)", "Motor Temp (°C)"]:
        ax.plot(plus_minus_df["Voltage (V)"], plus_minus_df[feature], label=f"{feature} vs Voltage")

    ax.set_xlabel("Voltage (V)")
    ax.set_ylabel("Predicted Values")
    ax.set_title("Predictions for Plus-Minus Analysis (Voltage-Based)")
    ax.legend()
    st.pyplot(fig)
