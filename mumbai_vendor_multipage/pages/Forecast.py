
import streamlit as st
from utils import load_model_artifacts, create_features_for_prediction
import plotly.graph_objects as go
import pandas as pd

st.set_page_config(page_title="Forecast", layout="wide")
st.title("Forecast • Hourly & 24-Hour")

model, label_encoders, feature_columns, metadata, model_type = load_model_artifacts()

# Controls
col1, col2 = st.columns([2,1])
with col2:
    st.sidebar.header("Forecast Controls (page)")
with col1:
    vendor = st.selectbox("Vendor", ['vendor_01','vendor_02','vendor_03','vendor_04','vendor_05'])
    date = st.date_input("Date", value=pd.to_datetime("2025-08-15"))
    hour = st.slider("Hour", 0, 23, 12)
    temp = st.slider("Temperature (°C)", 10.0, 45.0, 28.0)
    rain = st.slider("Rainfall (mm)", 0.0, 200.0, 0.0)
    humidity = st.slider("Humidity (%)", 10.0, 100.0, 80.0)
    traffic = st.selectbox("Traffic", ['low','medium','high'])
    competitors = st.slider("Competitors", 0, 30, 6)
    festival = st.selectbox("Festival", ['None','Independence Day','Janmashtami'])

if st.button("Predict & Generate 24h Forecast"):
    X, vinfo = create_features_for_prediction(vendor, date, hour, temp, rain, humidity, traffic, False, competitors, festival, label_encoders, feature_columns)
    try:
        single = model.predict(X)[0]
        st.metric("Predicted Units (single point)", f"{single:.0f}")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        single = None

    # generate 24h
    if single is not None:
        rows = []
        for i in range(24):
            new_h = (hour + i) % 24
            X_i, _ = create_features_for_prediction(vendor, date, new_h, temp, rain, humidity, traffic, False, competitors, festival, label_encoders, feature_columns)
            try:
                p = model.predict(X_i)[0]
            except Exception:
                p = None
            rows.append({'hour': new_h, 'predicted_units': p, 'label': f\"{new_h:02d}:00\"})
        df = pd.DataFrame(rows)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df['label'], y=df['predicted_units'], mode='lines+markers'))
        fig.update_layout(title='24-Hour Forecast', template='plotly_dark')
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(df)
        csv = df.to_csv(index=False)
        st.download_button('Download Forecast CSV', data=csv, file_name='24h_forecast.csv', mime='text/csv')
