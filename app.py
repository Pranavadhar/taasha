import streamlit as st
import pandas as pd
import numpy as np
import re
import requests
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.metrics import MeanSquaredError
from sklearn.preprocessing import StandardScaler

# Firebase configuration
firebase_url = st.secrets["firebase"]["url"]
firebase_auth_token = st.secrets["firebase"]["auth_token"]

# Function to fetch data from Firebase
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

# Extract numeric values from strings
def extract_numeric(value):
    if isinstance(value, str):
        match = re.search(r"[-+]?\d*\.\d+|\d+", value)
        return float(match.group()) if match else 0.0
    return float(value)

# Get Firebase values
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

# File paths
MODEL_PATH = "LSTM_final_model.h5"
DATA_PATH = "real_BABY.csv"

# Load model
try:
    model = load_model(MODEL_PATH, compile=False)
    st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Load dataset
try:
    df = pd.read_csv(DATA_PATH)
    df.dropna(inplace=True)
except FileNotFoundError:
    st.error("Dataset file not found.")
    st.stop()

# Define input and output features
input_features = ['Timestamp', 'volData', 'currentData']
output_features = ['batTempData', 'socData', 'sohData', 'motTempData', 'speedData']

# Standardize data
scaler = StandardScaler()
scaler.fit(df[input_features + output_features])

# Prediction function
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

# Fault detection function
def detect_faults(predictions):
    faults = []
    if predictions[0] > 32:
        faults.append("Battery Temperature > 32°C")
    if predictions[1] < 85:
        faults.append("SOC < 85%")
    if predictions[2] < 85:
        faults.append("SOH < 85%")
    if predictions[3] > 32:
        faults.append("Motor Temperature > 32°C")
    if predictions[4] < 57:
        faults.append("Motor Speed < 57")
    return faults

# Streamlit UI
st.title("Battery & Motor Health Prediction")

if st.button("Predict & Forecast"):
    timestamp, voltage, current = get_firebase_values()
    input_data = [timestamp, voltage, current]
    current_predictions = predict(input_data)

    # Generate future predictions for 150 timestamps
    future_timestamps = [timestamp + i for i in range(1, 151)]
    future_predictions = [predict([t, voltage, current]) for t in future_timestamps]
    
    # Store predictions in DataFrame
    future_df = pd.DataFrame(future_predictions, columns=output_features)
    future_df.index.name = "Future Timestamp"

    # Display current predictions
    st.header(f"Predictions for Timestamp: {timestamp}")
    results = dict(zip(output_features, current_predictions))
    st.write("### Predicted Values:", results)
    faults = detect_faults(current_predictions)
    if faults:
        st.error(", ".join(faults))
    else:
        st.success("No faults detected.")

    # Display future predictions
    st.subheader("Future Predictions for 150 Time Steps")
    st.write(future_df)

    # Visualization
    st.header("Future Prediction Visualization")
    future_predictions = np.array(future_predictions)
    fig, axs = plt.subplots(2, 1, figsize=(10, 8))
    
    for i, feature in enumerate(output_features):
        axs[0].plot(future_timestamps, future_predictions[:, i], label=feature)
    axs[0].set_title("Future Predictions (150 Timestamps)")
    axs[0].legend()
    axs[0].grid()
    
    axs[1].bar(output_features, current_predictions, color='blue', alpha=0.7)
    axs[1].set_title(f"Predicted Values for Timestamp: {timestamp}")
    axs[1].grid()
    
    plt.tight_layout()
    st.pyplot(fig)
