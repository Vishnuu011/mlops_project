import streamlit as st
import requests

st.set_page_config(page_title="Power Consumption Predictor", page_icon="⚡")
st.title("⚡ Power Consumption Predictor")

# Input fields
Temperature = st.number_input("🌡️ Temperature (°C)", value=25.0)
Humidity = st.number_input("💧 Humidity (%)", value=50.0)
WindSpeed = st.number_input("💨 Wind Speed (m/s)", value=5.0)
GeneralDiffuseFlows = st.number_input("🌤️ General Diffuse Flows", value=100.0)
DiffuseFlows = st.number_input("☁️ Diffuse Flows", value=50.0)

# Prediction
if st.button("🔮 Predict Power Consumption"):
    payload = {
        "Temperature": Temperature,
        "Humidity": Humidity,
        "WindSpeed": WindSpeed,
        "GeneralDiffuseFlows": GeneralDiffuseFlows,
        "DiffuseFlows": DiffuseFlows
    }

    try:
        response = requests.post("http://localhost:5000/predict", json=payload)
        if response.status_code == 200:
            prediction = response.json().get("prediction")
            st.success(f"⚡ Predicted Power Consumption: `{prediction:.2f}` kW")
        else:
            st.error(f"❌ Error from server: {response.status_code}")
    except requests.exceptions.ConnectionError:
        st.error("⚠️ Could not connect to backend.")


st.markdown("---")
st.header("🔁 Retrain Model Engine")

if st.button("🛠️ Start Training Engine"):
    log_placeholder = st.empty()
    logs = ""

    try:
        with st.spinner("Training Engine in progress Please Wait ‼️..."):
            response = requests.get("http://localhost:5000/train", stream=True)

            for line in response.iter_lines(decode_unicode=True):
                if line:
                    logs += line + "\n"
                    log_placeholder.text_area("📜 Real-Time Training Logs", logs, height=400)

        st.success("✅ Training Completed.")

    except requests.exceptions.ConnectionError:
        st.error("⚠️ Backend server not running.")