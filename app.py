import streamlit as st
import pandas as pd
import numpy as np
import re
import requests
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler

firebase_url = st.secrets["firebase"]["url"]
firebase_auth_token = st.secrets["firebase"]["auth_token"]

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

def extract_numeric(value):
    match = re.search(r"[-+]?\d*\.\d+|\d+", str(value))
    return float(match.group()) if match else 0.0

def get_firebase_values():
    entries = fetch_firebase_data()
    if entries:
        voltage = extract_numeric(entries.get("systemVoltage", "0.0 V"))
        current = extract_numeric(entries.get("current", "0.0 mA"))
        timestamp = extract_numeric(entries.get("timestamp", 0.0))
        return timestamp, voltage, current
    return 0.0, 0.0, 0.0

MODEL_PATH = "LSTM_final_model_upt.h5"
DATA_PATH = "real_updated_BABY.csv"

try:
    model = load_model(MODEL_PATH, compile=False)
    df = pd.read_csv(DATA_PATH).dropna()
    st.success("Model and dataset loaded successfully!")
except Exception as e:
    st.error(f"Error loading model or dataset: {e}")
    st.stop()

input_features = ['Timestamp', 'volData', 'currentData']
output_features = ['batTempData', 'socData', 'sohData', 'motTempData']

scaler = StandardScaler()
scaler.fit(df[input_features + output_features])

def predict(input_data):
    try:
        scaled_input = scaler.transform([input_data + [0] * len(output_features)])
        reshaped_input = scaled_input[:, :-len(output_features)].reshape(1, 1, len(input_features))
        predictions = model.predict(reshaped_input)
        rescaled_output = scaler.inverse_transform(np.concatenate((reshaped_input[:, 0], predictions), axis=1))[:, -len(output_features):]
        return rescaled_output[0]
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return np.zeros(len(output_features))

st.title("PMEV - LSTM")

if st.button("Predict & Analyze"):
    timestamp, voltage, current = get_firebase_values()
    st.subheader("Fetched Data from Firebase")
    st.write(f"**Current:** {current} mA")
    st.write(f"**Voltage:** {voltage} V")

    input_data = [timestamp, voltage, current]
    predicted_values = predict(input_data)
    predicted_batTemp, predicted_soc, predicted_soh, predicted_motTemp = predicted_values

    results = {
        "Battery Temperature (°C)": predicted_batTemp,
        "State of Charge (SOC %)": predicted_soc,
        "State of Health (SOH %)": predicted_soh,
        "Motor Temperature (°C)": predicted_motTemp,
    }

    st.subheader("Predicted Values for Fetched Data")
    st.write(results)

    st.subheader("Predicted Values Bar Chart")
    fig, ax = plt.subplots()
    ax.bar(["batTempData", "socData", "sohData", "motTempData"], results.values(), color='blue')
    ax.set_ylabel("Value")
    ax.set_title("Predicted Values")
    ax.grid()
    st.pyplot(fig)

    st.subheader("Plus Minus Analysis")
    current_values = [current + i for i in range(-5, 6)]
    voltage_values = [voltage + i for i in range(-5, 6)]

    analysis_results = []
    for c, v in zip(current_values, voltage_values):
        pred = predict([timestamp, v, c])
        analysis_results.append([c, v] + list(pred))

    plus_minus_df = pd.DataFrame(analysis_results, columns=["Current (mA)", "Voltage (V)"] + output_features)
    st.write(plus_minus_df)

    st.subheader("Trend of Plus Minus Predictions")
    fig, axs = plt.subplots(2, 1, figsize=(10, 8))
    
    for i, feature in enumerate(output_features):
        axs[0].plot(plus_minus_df["Current (mA)"], plus_minus_df[feature], label=feature)
    axs[0].set_title("Trend Analysis for Current")
    axs[0].legend()
    axs[0].grid()
    
    for i, feature in enumerate(output_features):
        axs[1].plot(plus_minus_df["Voltage (V)"], plus_minus_df[feature], label=feature)
    axs[1].set_title("Trend Analysis for Voltage")
    axs[1].legend()
    axs[1].grid()
    
    plt.tight_layout()
    st.pyplot(fig)
