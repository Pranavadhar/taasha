import streamlit as st
import pandas as pd
import numpy as np
import re
import requests
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.metrics import MeanSquaredError
from sklearn.preprocessing import StandardScaler

firebase_url = st.secrets["firebase"]["url"]
firebase_auth_token = st.secrets["firebase"]["auth_token"]

def fetch_firebase_data():
    try:
        response = requests.get(f'{firebase_url}/parameters.json?auth={firebase_auth_token}')
        if response.ok:
            entries = response.json()
            return entries if entries else None
        else:
            st.error("Error fetching data from Firebase.")
            return None
    except requests.exceptions.RequestException as e:
        st.error(f"Network error: {e}")
        return None

def extract_numeric(value):
    if isinstance(value, str):
        match = re.search(r"[-+]?\d*\.\d+|\d+", value)
        return float(match.group()) if match else 0.0
    return float(value)

def get_firebase_values():
    entries = fetch_firebase_data()
    if entries:
        voltage = extract_numeric(entries.get("systemVoltage", "0.0 V"))
        current = extract_numeric(entries.get("current", "0.0 mA"))
        timestamp = extract_numeric(entries.get("timestamp", 0.0))
        st.write(f"**Voltage:** {voltage} V")
        st.write(f"**Current:** {current} mA")
        return timestamp, voltage, current
    else:
        return 0.0, 0.0, 0.0

MODEL_PATH = "LSTM_final_model_upt.h5"
DATA_PATH = "real_updated_BABY.csv"

try:
    model = load_model(MODEL_PATH, compile=False)
    st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

try:
    df = pd.read_csv(DATA_PATH)
    df.dropna(inplace=True)
except FileNotFoundError:
    st.error("Dataset file not found.")
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
        rescaled_output = scaler.inverse_transform(
            np.concatenate((reshaped_input[:, 0], predictions), axis=1)
        )[:, -len(output_features):]
        return rescaled_output[0]
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return np.zeros(len(output_features))

def detect_faults(predictions):
    faults = []
    if predictions[0] > 35:
        faults.append("Battery Temperature > 35°C - SYSTEM COOLING ACTIVELY")
    if predictions[0] > 45:
        faults.append("Battery Temperature > 45°C - SYSTEM OVER HEATING  : COOLING SYSTEM CHECK UP RECOMMENDED ")
    if predictions[1] < 30:
        faults.append("SOC < 30% - LOW BATTERY PLUG IN CHARGE")
    if predictions[2] < 89:
        faults.append("SOH < 89% - BATTERY SERVICE RECOMMENDED")
    if predictions[3] > 35:
        faults.append("Motor Temperature > 35°C - SYSTEM COLLING ACTIVELY")
    if predictions[3] > 45:
        faults.append("Motor Temperature > 45°C - SYSTEM OVER HEATING  : COOLING SYSTEM CHECK UP RECOMMENDED ")
    return faults

def calculate_soc(voltage, current, timestamp, lastVoltage, lastTime, lastVoltageUpdate, batteryPercent, isInitialized, minVoltage=3.0, maxVoltage=12.6):
    voltagebatteryPercentFactor = 100.0 / (maxVoltage - minVoltage)

    if not isInitialized:
        batteryPercent = (voltage - minVoltage) * voltagebatteryPercentFactor
        batteryPercent = np.clip(batteryPercent, 0, 100)
        isInitialized = True

    elapsedTime = timestamp - lastTime
    if elapsedTime >= 1:
        lastTime = timestamp
        if current != 0:
            deltabatteryPercent = (current / 1000.0) * (elapsedTime / 3600.0)
            batteryPercent -= deltabatteryPercent * 2.0
            batteryPercent = np.clip(batteryPercent, 0, 100)

    if timestamp - lastVoltageUpdate >= 10:
        lastVoltageUpdate = timestamp
        if voltage < lastVoltage:
            batteryPercent -= 1.0
            batteryPercent = np.clip(batteryPercent, 0, 100)

    return batteryPercent, lastVoltage, lastTime, lastVoltageUpdate

st.title("Battery & Motor Health Prediction")

if st.button("Predict & Forecast"):
    timestamp, voltage, current = get_firebase_values()

    lastVoltage = voltage
    lastTime = timestamp
    lastVoltageUpdate = timestamp
    batteryPercent = 0.0
    isInitialized = False

    soc_mapped, lastVoltage, lastTime, lastVoltageUpdate = calculate_soc(
        voltage, current, timestamp, lastVoltage, lastTime, lastVoltageUpdate, batteryPercent, isInitialized
    )

    input_data = [timestamp, voltage, current]
    current_predictions = predict(input_data)

    predicted_batTemp, predicted_soc, predicted_soh, predicted_motTemp = current_predictions

    results = {
        "batTempData": predicted_batTemp,
        "socData": soc_mapped,
        "sohData": predicted_soh,
        "motTempData": predicted_motTemp,
    }

    future_timestamps = [timestamp + i for i in range(1, 151)]
    future_predictions = []

    for t in future_timestamps:
        future_pred = predict([t, voltage, current])
        future_batTemp, future_soc, future_soh, future_motTemp = future_pred

        future_soc_mapped, lastVoltage, lastTime, lastVoltageUpdate = calculate_soc(
            voltage, current, t, lastVoltage, lastTime, lastVoltageUpdate, soc_mapped, isInitialized
        )

        future_predictions.append([future_batTemp, future_soc_mapped, future_soh, future_motTemp])

    future_df = pd.DataFrame(future_predictions, columns=output_features, index=future_timestamps)
    future_df.index.name = "Future Timestamp"

    st.header(f"Predictions for Timestamp: {timestamp}")
    st.write("### Predicted Values:", results)

    faults = detect_faults(current_predictions)
    if faults:
        st.error(", ".join(faults))
    else:
        st.success("No faults detected.")

    st.subheader("Future Predictions for 150 Time Steps")
    st.write(future_df)

    st.header("Future Prediction Visualization")
    fig, axs = plt.subplots(2, 1, figsize=(10, 8))

    future_predictions = np.array(future_predictions)
    for i, feature in enumerate(output_features):
        axs[0].plot(future_timestamps, future_predictions[:, i], label=feature)
    axs[0].set_title("Future Predictions (150 Timestamps)")
    axs[0].legend()
    axs[0].grid()

    axs[1].bar(output_features, results.values(), color='blue', alpha=0.7)
    axs[1].set_title("Predicted Values Bar Chart")
    axs[1].grid()

    plt.tight_layout()
    st.pyplot(fig)
