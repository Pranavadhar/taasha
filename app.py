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

# Firebase credentials
firebase_url = st.secrets["firebase"]["url"]
firebase_auth_token = st.secrets["firebase"]["auth_token"]

# Function to fetch data from Firebase
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

# Fetch Firebase data
entries = fetch_firebase_data()

# Load dataset
DATA_PATH = "real_BABY.csv"
try:
    data = pd.read_csv(DATA_PATH).dropna()
except FileNotFoundError:
    st.error("Dataset file not found. Please check the path and file name.")
    st.stop()

# Define input and output features
input_features = ["volData", "currentData"]
output_features = ["batTempData", "socData", "sohData", "motTempData", "speedData"]

# Prepare dataset
x = data[input_features]
y = data[output_features]

# Standardization
scaler_x, scaler_y = StandardScaler(), StandardScaler()
x_scaled, y_scaled = scaler_x.fit_transform(x), scaler_y.fit_transform(y)

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(x_scaled, y_scaled, test_size=0.2, random_state=42)

# Models
models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "Random Forest": RandomForestRegressor(random_state=42, n_estimators=25),
}

# Train models
for model in models.values():
    model.fit(x_train, y_train)

# Fault detection function
def detect_faults(predictions):
    conditions = [
        (predictions[0] > 35, "Battery Temperature > 35°C"),
        (predictions[1] < 30, "SOC < 30%"),
        (predictions[2] < 89, "SOH < 89%"),
        (predictions[3] > 35, "Motor Temperature > 35°C"),
        (predictions[4] < 60, "Motor Speed < 60")
    ]
    return [msg for cond, msg in conditions if cond]

# Sidebar for model selection
st.sidebar.header("Model Selection")
selected_model_name = st.sidebar.selectbox("Select a model", list(models.keys()))
selected_model = models[selected_model_name]

# Fetch voltage and current data from Firebase
vol_data = float(entries.get("systemVoltage", 0.0))
current_str = str(entries.get("current", "0.0 mA"))
current_data = float(re.search(r"[\d\.]+", current_str).group()) if re.search(r"[\d\.]+", current_str) else 0.0

# Display fetched values
st.write("### Fetched Data from Firebase:")
st.write(f"**Voltage:** {vol_data} V")
st.write(f"**Current:** {current_data} mA")

# Predict for current values
sample_input_scaled = scaler_x.transform([[vol_data, current_data]])
pred_scaled = selected_model.predict(sample_input_scaled)
pred = scaler_y.inverse_transform(pred_scaled).flatten()

# Predict for 150 future timestamps
future_inputs_scaled = scaler_x.transform(np.tile([vol_data, current_data], (150, 1)))
future_preds = scaler_y.inverse_transform(selected_model.predict(future_inputs_scaled))

# Store predictions in DataFrame
future_df = pd.DataFrame(future_preds, columns=output_features)
future_df.index.name = "Future Timestamp"

# Display predictions
st.subheader("Predictions")
st.write(f"**Model Used:** {selected_model_name}")
st.write(f"**Predicted Outputs for Current Values:**", dict(zip(output_features, pred)))

# Bar plot for predicted values
st.subheader("Predicted Values Bar Chart")
fig_bar, ax_bar = plt.subplots(figsize=(8, 5))
ax_bar.bar(output_features, pred, color=['blue', 'green', 'red', 'orange', 'purple'])
ax_bar.set_ylabel("Predicted Values")
ax_bar.set_title("Predicted Output for Current Values")
st.pyplot(fig_bar)

st.subheader("Future Predictions for 150 Time Steps")
st.write(future_df)

# Fault detection
st.subheader("Fault Detection")
faults = detect_faults(pred)
if faults:
    st.error("Faults Detected:")
    for fault in faults:
        st.write(f"- {fault}")
else:
    st.success("No faults detected.")

# Visualization
st.header("Future Prediction Trend")
fig, ax = plt.subplots(figsize=(12, 6))
for feature in output_features:
    ax.plot(range(1, 151), future_df[feature], label=feature)
ax.set_xlabel("Future Timestamp")
ax.set_ylabel("Predicted Values")
ax.set_title("Predictions Over 150 Future Time Steps")
ax.legend()
st.pyplot(fig)
