import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.metrics import MeanSquaredError
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import requests

# Firebase configuration
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

MODEL_PATH = "LSTM_nn.h5" 
DATA_PATH = "randsam_BABIES.csv"

# Load model and dataset
try:
    model = load_model(MODEL_PATH, custom_objects={'mse': MeanSquaredError()})
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

try:
    df = pd.read_csv(DATA_PATH)
except FileNotFoundError:
    st.error("Dataset file not found. Please check the path and file name.")
    st.stop()

input_features = ['Timestamp', 'volData', 'currentData']
output_features = ['batTempData', 'socData', 'sohData', 'motTempData', 'speedData']

scaler = StandardScaler()
scaler.fit(df[input_features + output_features])

# Fetch and display data
def get_firebase_values():
    voltage = entries.get("voltage", 0.0)
    current = entries.get("current", 0.0)
    st.write("### Fetched Data from Firebase:")
    st.write(f"**Voltage:** {voltage} V")
    st.write(f"**Current:** {current} A")
    return voltage, current

# Prediction function
def predict(input_data):
    scaled_input = scaler.transform([input_data + [0] * len(output_features)])  # Dummy target values
    reshaped_input = scaled_input[:, :-len(output_features)].reshape(1, 1, len(input_features))
    predictions = model.predict(reshaped_input)
    rescaled_output = scaler.inverse_transform(
        np.concatenate((reshaped_input[:, 0], predictions), axis=1)
    )[:, -len(output_features):]
    return rescaled_output[0]

# Fault detection function
def detect_faults(predictions):
    faults = []
    if predictions[0] > 32:  # Battery temperature
        faults.append("Battery Temperature > 32°C")
    if predictions[1] < 85:  # SOC
        faults.append("SOC < 70%")
    if predictions[2] < 85:  # SOH
        faults.append("SOH < 70%")
    if predictions[3] > 32:  # Motor temperature
        faults.append("Motor Temperature > 32°C")
    if predictions[4] < 57:  # Motor speed
        faults.append("Motor Speed < 57")
    return faults

# Streamlit UI
timestamp_input = st.number_input("Timestamp", min_value=0.0, step=0.0001, format="%.5f")
if st.button("Predict"):
    voltage_input, current_input = get_firebase_values()
    input_data = [timestamp_input, voltage_input, current_input]
    current_predictions = predict(input_data)

    future_timestamp = timestamp_input + 10
    future_data = [future_timestamp, voltage_input, current_input]
    future_predictions = predict(future_data)

    # Display current predictions
    st.header(f"Predictions for Timestamp: {timestamp_input}")
    results = dict(zip(output_features, current_predictions))
    st.write("### Predicted Values:", results)
    st.write("### Fault Conditions:")
    faults = detect_faults(current_predictions)
    if faults:
        st.error(", ".join(faults))
    else:
        st.success("No faults detected.")

    # Display future predictions
    st.header(f"Predictions for Timestamp: {future_timestamp}")
    future_results = dict(zip(output_features, future_predictions))
    st.write("### Predicted Values:", future_results)
    st.write("### Fault Conditions:")
    future_faults = detect_faults(future_predictions)
    if future_faults:
        st.error(", ".join(future_faults))
    else:
        st.success("No faults detected.")

    # Visualization
    st.header("Prediction Visualization")
    fig, axs = plt.subplots(2, 1, figsize=(10, 8))
    axs[0].bar(output_features, current_predictions, color='blue', alpha=0.7)
    axs[0].set_title(f"Predicted Values for Timestamp: {timestamp_input}")
    axs[0].set_ylabel("Value")
    axs[0].grid(True)

    axs[1].bar(output_features, future_predictions, color='orange', alpha=0.7)
    axs[1].set_title(f"Predicted Values for Timestamp: {future_timestamp}")
    axs[1].set_ylabel("Value")
    axs[1].grid(True)

    plt.tight_layout()
    st.pyplot(fig)
