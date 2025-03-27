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
DATA_PATH = "real_updated_BABY.csv"
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

def voltage_to_soc(voltage):
    return np.clip(((voltage - 3) / (12.6 - 3)) * 100, 0, 100)

st.sidebar.header("Model Selection")
selected_model_name = st.sidebar.selectbox("Select a model", list(models.keys()))
selected_model = models[selected_model_name]

vol_data = float(entries.get("systemVoltage", 0.0))
current_str = str(entries.get("current", "0.0 mA"))
current_match = re.search(r"[\d\.]+", current_str)
current_data = float(current_match.group()) if current_match else 0.0
mapped_soc = voltage_to_soc(vol_data)

st.write("### Fetched Data from Firebase:")
st.write(f"**Voltage:** {vol_data} V")
st.write(f"**Current:** {current_data} mA")

sample_input = np.array([[vol_data, current_data]], dtype=float)
sample_input_scaled = scaler_x.transform(sample_input)
pred_scaled = selected_model.predict(sample_input_scaled)
pred = scaler_y.inverse_transform(pred_scaled).flatten()

pred[1] = mapped_soc

future_inputs = np.tile([vol_data, current_data], (150, 1))
future_inputs_scaled = scaler_x.transform(future_inputs)
future_preds_scaled = selected_model.predict(future_inputs_scaled)
future_preds = scaler_y.inverse_transform(future_preds_scaled)

future_preds[:, 1] = voltage_to_soc(vol_data)
future_df = pd.DataFrame(future_preds, columns=output_features)
future_df.index.name = "Future Timestamp"

st.subheader("Predictions")
st.write(f"**Model Used:** {selected_model_name}")
st.write(f"**Predicted Outputs for Current Values (with SOC Mapped):**")
st.write(dict(zip(output_features, pred)))

st.subheader("Predicted Values Bar Chart")
fig_bar, ax_bar = plt.subplots(figsize=(8, 5))
ax_bar.bar(output_features, pred, color=['blue', 'green', 'red', 'orange', 'purple'])
ax_bar.set_ylabel("Predicted Values")
ax_bar.set_title("Predicted Output for Current Values")
st.pyplot(fig_bar)

st.subheader("Future Predictions for 150 Time Steps")
st.write(future_df)

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

st.subheader("Fault Detection")
faults = detect_faults(pred)
if faults:
    st.error("Faults Detected:")
    for fault in faults:
        st.write(f"- {fault}")
else:
    st.success("No faults detected.")

st.header("Future Prediction Trend")
fig, ax = plt.subplots(figsize=(12, 6))
for feature in output_features:
    ax.plot(range(1, 151), future_df[feature], label=feature)
ax.set_xlabel("Future Timestamp")
ax.set_ylabel("Predicted Values")
ax.set_title("Predictions Over 150 Future Time Steps")
ax.legend()
st.pyplot(fig)
