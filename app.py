import streamlit as st
import pandas as pd
import numpy as np
import re
import requests
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler

# Firebase credentials
firebase_url = st.secrets["firebase"]["url"]
firebase_auth_token = st.secrets["firebase"]["auth_token"]

# Function to fetch data from Firebase
def fetch_firebase_data():
    try:
        response = requests.get(f'{firebase_url}/parameters.json?auth={firebase_auth_token}')
        if response.ok:
            return response.json() or None
        else:
            st.error("Error fetching data from Firebase.")
            return None
    except requests.exceptions.RequestException as e:
        st.error(f"Network error: {e}")
        return None

# Function to extract numeric values from strings
def extract_numeric(value):
    match = re.search(r"[-+]?\d*\.\d+|\d+", str(value))
    return float(match.group()) if match else 0.0

# Function to retrieve Firebase values
def get_firebase_values():
    entries = fetch_firebase_data()
    if entries:
        voltage = extract_numeric(entries.get("systemVoltage", "0.0 V"))
        current = extract_numeric(entries.get("current", "0.0 mA"))
        timestamp = extract_numeric(entries.get("timestamp", 0.0))
        return timestamp, voltage, current
    return 0.0, 0.0, 0.0

# Load model and dataset
MODEL_PATH = "LSTM_final_newmodel_230425.h5"
DATA_PATH = "new_Born.csv"

try:
    model = load_model(MODEL_PATH, compile=False)
    df = pd.read_csv(DATA_PATH).dropna()
except Exception as e:
    st.error(f"Error loading model or dataset: {e}")
    st.stop()

# Define input and output features
input_features = ['Timestamp', 'volData', 'currentData']
output_features = ['batTempData', 'socData', 'sohData', 'motTempData']

# Separate scalers for input and output
input_scaler = StandardScaler()
output_scaler = StandardScaler()

# Fit the scalers
input_scaler.fit(df[input_features])
output_scaler.fit(df[output_features])

# Prediction function
def predict(input_data):
    try:
        scaled_input = input_scaler.transform([input_data])
        reshaped_input = scaled_input.reshape(1, 1, len(input_features))
        predictions = model.predict(reshaped_input)
        rescaled_output = output_scaler.inverse_transform(predictions)
        return rescaled_output[0]
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return np.zeros(len(output_features))

# Fault detection logic
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

# Streamlit UI
st.title("PMEV - LSTM")

if st.button("Predict & Analyze"):
    timestamp, voltage, current = get_firebase_values()
    st.subheader("Fetched Data from Firebase")
    st.write(f"**Current:** {current} mA")
    st.write(f"**Voltage:** {voltage} V")

    if 0 <= current <= 1:
        st.info("Vehicle is not in motion - refrain from referring to the prediction")
    else:
        input_data = [timestamp, voltage, current]
        predicted_values = predict(input_data)

        # Extract predicted values
        predicted_batTemp, predicted_soc, predicted_soh, predicted_motTemp = predicted_values

        # Display results
        results = {
            "Battery Temperature (°C)": float(predicted_batTemp),
            "State of Charge (SOC %)": float(predicted_soc),
            "State of Health (SOH %)": float(predicted_soh),
            "Motor Temperature (°C)": float(predicted_motTemp),
        }

        st.subheader("Predicted Values for Fetched Data")
        st.write(results)

        # Show faults
        faults = detect_faults(predicted_values)
        if faults:
            st.subheader("Detected Faults")
            for fault in faults:
                st.warning(fault)

        # Bar chart visualization
        st.subheader("Predicted Values Bar Chart")
        fig, ax = plt.subplots()
        ax.bar(["batTempData", "socData", "sohData", "motTempData"], results.values(), color='blue')
        ax.set_ylabel("Value")
        ax.set_title("Predicted Values")
        ax.grid()
        st.pyplot(fig)

        # Plus-minus analysis
        st.subheader("Plus Minus Analysis")
        current_values = [current + i for i in range(-5, 6)]
        voltage_values = [voltage + i for i in range(-5, 6)]

        analysis_results = []
        for c, v in zip(current_values, voltage_values):
            pred = predict([timestamp, v, c])
            analysis_results.append([c, v, pred[0], pred[1], pred[2], pred[3]])

        plus_minus_df = pd.DataFrame(analysis_results, columns=["Current (mA)", "Voltage (V)", "Battery Temp (°C)", "SOC (%)", "SOH (%)", "Motor Temp (°C)"])
        st.write(plus_minus_df)

        # Trend Analysis Visualization
        st.subheader("Trend of Plus Minus Predictions")
        fig, axs = plt.subplots(2, 1, figsize=(10, 8))

        for feature in ["Battery Temp (°C)", "SOC (%)", "SOH (%)", "Motor Temp (°C)"]:
            axs[0].plot(plus_minus_df["Current (mA)"], plus_minus_df[feature], label=feature)
        axs[0].set_title("Trend Analysis for Current")
        axs[0].legend()
        axs[0].grid()

        for feature in ["Battery Temp (°C)", "SOC (%)", "SOH (%)", "Motor Temp (°C)"]:
            axs[1].plot(plus_minus_df["Voltage (V)"], plus_minus_df[feature], label=feature)
        axs[1].set_title("Trend Analysis for Voltage")
        axs[1].legend()
        axs[1].grid()

        plt.tight_layout()
        st.pyplot(fig)
