
import streamlit as st
from utils import load_model_artifacts, create_features_for_prediction, load_or_create_shap_explainer
from datetime import datetime
import pandas as pd

st.set_page_config(page_title="Mumbai Vendor Forecast - Home", layout="wide", page_icon="📊")
st.title("Mumbai Vendor Demand Forecast — Home")
st.markdown("Welcome to the multipage vendor demand forecasting dashboard. Use the pages on the left to explore forecasts and explainability.")

# load artifacts (lazy)
try:
    model, label_encoders, feature_columns, metadata, model_type = load_model_artifacts()
    st.sidebar.success(f"\Loaded model: {model_type}")
except Exception as e:
    st.sidebar.error(f"Model load error: {e}")
    model, label_encoders, feature_columns, metadata, model_type = (None, {}, None, {}, None)

# Quick scenario controls
st.sidebar.header("Quick Scenario")
vendor = st.sidebar.selectbox("Vendor", ['vendor_01','vendor_02','vendor_03','vendor_04','vendor_05'])
date = st.sidebar.date_input("Date", value=datetime(2025,8,15))
hour = st.sidebar.slider("Hour", 0, 23, 12)

temp = st.sidebar.slider("Temperature (°C)", 10.0, 45.0, 28.0)
rain = st.sidebar.slider("Rainfall (mm)", 0.0, 200.0, 0.0)
humidity = st.sidebar.slider("Humidity (%)", 10.0, 100.0, 80.0)
traffic = st.sidebar.selectbox("Traffic", ['low','medium','high'])
competitors = st.sidebar.slider("Competitors", 0, 30, 6)
festival = st.sidebar.selectbox("Festival", ['None', 'Independence Day', 'Janmashtami', 'Ganesh Chaturthi'])

if st.button("🔮 Predict"):
    if model is None:
        st.error("Model not loaded. Ensure rf_model.pkl or xgb_model.pkl is present.")
    else:
        X, vinfo = create_features_for_prediction(vendor, date, hour, temp, rain, humidity, traffic, False, competitors, festival, label_encoders, feature_columns)
        try:
            pred = model.predict(X)[0]
            st.success(f"Predicted units: {pred:.0f}")
            st.write("Vendor info:", vinfo)
        except Exception as e:
            st.error(f"Prediction failed: {e}")

st.markdown("---")
st.header("Model Information")
st.write({ 'Model Type': model_type, 'Metadata': metadata })