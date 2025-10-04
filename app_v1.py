import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import plotly.graph_objects as go
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ----------------------------
# Page configuration
st.set_page_config(
    page_title="Mumbai Street Vendor Demand Forecasting",
    page_icon="🍛",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------------------
# Load model and artifacts
@st.cache_resource
def load_model_artifacts():
    try:
        try:
            model = joblib.load('xgb_model.pkl')
            model_type = "XGBoost"
        except:
            model = joblib.load('rf_model.pkl')
            model_type = "Random Forest"
        
        label_encoders = joblib.load('label_encoders.pkl')
        feature_columns = joblib.load('feature_columns.pkl')

        with open('model_metadata.json', 'r') as f:
            metadata = json.load(f)

        return model, label_encoders, feature_columns, metadata, model_type
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None, None, None, None

model, label_encoders, feature_columns, metadata, model_type = load_model_artifacts()
if model is None:
    st.stop()

# ----------------------------
# Load historical data
@st.cache_data
def load_historical_data():
    try:
        df = pd.read_csv('mumbai_vendors_hourly_20250701_20250930.csv')
        # Parse datetime and keep existing timezone (if any)
        df['datetime_parsed'] = pd.to_datetime(df['datetime'], errors='coerce')
        # Ensure it’s in Asia/Kolkata timezone
        if df['datetime_parsed'].dt.tz is None:
            df['datetime_parsed'] = df['datetime_parsed'].dt.tz_localize('Asia/Kolkata')
        else:
            df['datetime_parsed'] = df['datetime_parsed'].dt.tz_convert('Asia/Kolkata')
        return df
    except Exception as e:
        st.error(f"Error loading historical data: {e}")
        return None

historical_data = load_historical_data()

# ----------------------------
# App header
st.markdown("<h1 style='text-align:center'>🍛 Mumbai Street Vendor Demand Forecasting</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center'>Predict hourly demand and explore scenario simulations with interactive charts.</p>", unsafe_allow_html=True)

# ----------------------------
# Sidebar controls
st.sidebar.header("📊 Prediction Controls")

vendor_options = {
    'vendor_01': 'BKC_Pavbhaji (Business District)',
    'vendor_02': 'Churchgate_Chai (Near College)', 
    'vendor_03': 'Dadar_Chaat (Market)',
    'vendor_04': 'Andheri_Juice (Metro Station)',
    'vendor_05': 'Powai_Dessert (Office Area)'
}

selected_vendor = st.sidebar.selectbox(
    "Select Vendor:",
    options=list(vendor_options.keys()),
    format_func=lambda x: vendor_options[x]
)

st.sidebar.subheader("📅 Date & Time")
selected_date = st.sidebar.date_input(
    "Select Date:",
    value=datetime(2025, 8, 15),
    min_value=datetime(2025, 7, 1),
    max_value=datetime(2025, 12, 31)
)
selected_hour = st.sidebar.slider("Select Hour:", 0, 23, 12)

st.sidebar.subheader("🌤️ Weather Conditions")
temperature = st.sidebar.slider("Temperature (°C):", 22.0, 35.0, 28.0, 0.5)
rainfall = st.sidebar.slider("Rainfall (mm):", 0.0, 50.0, 0.0, 0.5)
humidity = st.sidebar.slider("Humidity (%):", 50.0, 95.0, 80.0, 1.0)

st.sidebar.subheader("🚦 Environment")
traffic_density = st.sidebar.selectbox("Traffic Density:", ['low', 'medium', 'high'], index=1)
event_nearby = st.sidebar.checkbox("Special Event Nearby", value=False)
competitor_count = st.sidebar.slider("Nearby Competitors:", 0, 15, 6)

st.sidebar.subheader("🎉 Festival Options")
festival_options = ['None', 'Independence Day', 'Janmashtami', 'Dahi Handi', 
                   'Ganesh Chaturthi', 'Ganesh Visarjan', 'Eid-e-Milad']
selected_festival = st.sidebar.selectbox("Simulate Festival:", festival_options)

# ----------------------------
# Vendor details
vendor_details = {
    'vendor_01': {'location_type': 'business_district', 'cuisine_type': 'main_course', 'avg_price': 85.0, 'menu_diversity': 12, 'peak_hours': [12, 13, 19, 20]},
    'vendor_02': {'location_type': 'near_college', 'cuisine_type': 'chai_snacks', 'avg_price': 25.0, 'menu_diversity': 8, 'peak_hours': [7, 8, 16, 17, 18]},
    'vendor_03': {'location_type': 'market', 'cuisine_type': 'chaat', 'avg_price': 45.0, 'menu_diversity': 15, 'peak_hours': [17, 18, 19, 20, 21]},
    'vendor_04': {'location_type': 'metro_station', 'cuisine_type': 'beverages', 'avg_price': 35.0, 'menu_diversity': 10, 'peak_hours': [7, 8, 9, 17, 18, 19]},
    'vendor_05': {'location_type': 'office', 'cuisine_type': 'dessert', 'avg_price': 65.0, 'menu_diversity': 7, 'peak_hours': [15, 16, 21, 22]}
}

# ----------------------------
# Safe encoding
def safe_transform(encoder, value):
    return encoder.transform([value])[0] if value in encoder.classes_ else -1

# ----------------------------
# Feature generation
def create_prediction_features(vendor_id, date, hour, temp, rain, humid, traffic, 
                               event, competitors, festival):
    vendor_info = vendor_details[vendor_id]
    dt = pd.Timestamp(datetime.combine(date, datetime.min.time()) + timedelta(hours=hour)).tz_localize('Asia/Kolkata')
    day_of_week = dt.weekday()
    is_weekend = int(day_of_week >= 5)
    is_festival = int(festival != 'None')
    is_holiday = int(festival in ['Independence Day'])
    wind_speed = 15.0 + (5.0 if rain > 0 else 0.0)
    is_peak_hour = int(hour in vendor_info['peak_hours'])
    is_evening = int(17 <= hour <= 21)
    is_morning = int(6 <= hour <= 10)
    base_demand = 15 if vendor_id == 'vendor_01' else 20
    lag_1h = max(1, int(base_demand * (0.8 + 0.4 * np.random.random())))
    lag_24h = max(1, int(base_demand * (0.9 + 0.2 * np.random.random())))
    rolling_avg = (lag_1h + lag_24h) / 2

    features = {
        'vendor_id_encoded': safe_transform(label_encoders['vendor_id'], vendor_id),
        'location_type_encoded': safe_transform(label_encoders['location_type'], vendor_info['location_type']),
        'cuisine_type_encoded': safe_transform(label_encoders['cuisine_type'], vendor_info['cuisine_type']),
        'avg_price': vendor_info['avg_price'],
        'menu_diversity': vendor_info['menu_diversity'],
        'hour_of_day': hour,
        'day_of_week': day_of_week,
        'is_weekend': is_weekend,
        'is_holiday': is_holiday,
        'is_festival': is_festival,
        'temperature_c': temp,
        'rainfall_mm': rain,
        'humidity_pct': humid,
        'wind_speed_kmh': wind_speed,
        'event_nearby': int(event),
        'traffic_density_encoded': safe_transform(label_encoders['traffic_density'], traffic),
        'competitor_count': competitors,
        'lag_1h_units': lag_1h,
        'lag_24h_units': lag_24h,
        'rolling_avg_24h': rolling_avg,
        'is_peak_hour': is_peak_hour,
        'is_evening': is_evening,
        'is_morning': is_morning
    }
    return pd.DataFrame([features]), vendor_info

# ----------------------------
# Prediction explanation
def generate_explanation(prediction, features_df, vendor_info, festival):
    """Generate natural language explanation for prediction"""
    hour = features_df.iloc[0]['hour_of_day']
    is_peak = features_df.iloc[0]['is_peak_hour']
    rainfall = features_df.iloc[0]['rainfall_mm']
    traffic = features_df.iloc[0]['traffic_density_encoded']
    is_festival_day = features_df.iloc[0]['is_festival']

    explanation = f"**Predicted demand: {prediction:.0f} units**\n\n"

    # Time factor
    if is_peak:
        explanation += "✅ **Peak hour boost**: High demand expected during busy hours\n"
    else:
        explanation += "⚠️ **Off-peak hour**: Lower baseline demand expected\n"

    # Weather impact
    if rainfall > 5:
        if vendor_info['cuisine_type'] == 'chai_snacks':
            explanation += "☔ **Rain advantage**: Hot snacks/tea more popular in rain (+40%)\n"
        elif vendor_info['cuisine_type'] == 'beverages':
            explanation += "☔ **Rain impact**: Cold beverages less popular in rain (-40%)\n"
        else:
            explanation += "☔ **Weather impact**: Rain slightly reduces foot traffic (-10%)\n"
    elif rainfall > 0:
        explanation += "🌦️ **Light rain**: Minor impact on demand\n"

    # Festival impact
    if is_festival_day:
        if festival == 'Ganesh Chaturthi':
            explanation += "🎉 **Major festival boost**: Ganesh Chaturthi brings huge crowds (+150%)\n"
        elif festival in ['Independence Day', 'Janmashtami']:
            explanation += "🎊 **Festival boost**: Holiday increases demand (+60%)\n"
        else:
            explanation += "🎈 **Festival effect**: Celebration brings more customers (+30%)\n"

    # Traffic impact
    traffic_names = {0: 'Low', 1: 'Medium', 2: 'High'}
    traffic_name = traffic_names.get(int(traffic), 'Medium')
    if traffic == 2:  # High traffic
        explanation += "🚗 **High traffic boost**: Busy area increases sales (+40%)\n"
    elif traffic == 0:  # Low traffic
        explanation += "🚶 **Low traffic**: Fewer people around reduces demand (-30%)\n"

    # Location-specific insights
    location_insights = {
        'business_district': 'Office workers drive weekday lunch/dinner demand',
        'market': 'Market location benefits from continuous foot traffic',
        'metro_station': 'Commuters create morning and evening rush patterns',
        'near_college': 'Students provide consistent demand with price sensitivity',
        'office': 'Corporate area with break-time and after-work peaks'
    }

    explanation += f"📍 **Location insight**: {location_insights[vendor_info['location_type']]}"

    return explanation

# ----------------------------
# Layout columns
col1, col2= st.columns([1,1])

with col1:
    st.subheader("🎯 Demand Prediction")
    if st.button("🔮 Predict Units", type="primary"):
        features_df, vendor_info = create_prediction_features(
            selected_vendor, selected_date, selected_hour,
            temperature, rainfall, humidity, traffic_density,
            event_nearby, competitor_count, selected_festival
        )
        prediction = model.predict(features_df)[0]
        uncertainty = float(metadata.get('prediction_uncertainty_std', 3.0))
        st.success(f"**Predicted Units Sold: {prediction:.0f}**")
        #st.info(f"**95% CI: {prediction-1.96*uncertainty:.0f} - {prediction+1.96*uncertainty:.0f} units**")
        st.markdown("### 📝 Prediction Explanation")
        st.markdown(generate_explanation(prediction, features_df, vendor_info, selected_festival))
        st.session_state.last_prediction = prediction
        st.session_state.last_features = features_df
        st.session_state.last_vendor = selected_vendor


with col2:
    st.subheader("🛠️ Scenario Parameters")
    st.markdown(f"""
    - **Vendor:** {vendor_options[selected_vendor]}
    - **Date/Time:** {selected_date} at {selected_hour}:00
    - **Temperature:** {temperature} °C
    - **Rainfall:** {rainfall} mm
    - **Humidity:** {humidity} %
    - **Traffic Density:** {traffic_density.capitalize()}
    - **Special Event Nearby:** {"Yes" if event_nearby else "No"}
    - **Nearby Competitors:** {competitor_count}
    - **Simulated Festival:** {selected_festival if selected_festival != 'None' else 'No Festival'}
    """)



st.subheader("📈 Historical Context")
if historical_data is not None:
        vendor_hist = historical_data[historical_data['vendor_id'] == selected_vendor].copy()
        if not vendor_hist.empty:
            tab1, tab2 = st.tabs([ "🕒 Hourly Pattern","📅 48-Hour Context"])
            with tab1:
                hourly_avg = vendor_hist.groupby('hour_of_day')['units_sold'].agg(['mean','std']).reset_index()
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=hourly_avg['hour_of_day'], y=hourly_avg['mean'], mode='lines+markers', line=dict(color='green', width=3)))
                fig.add_trace(go.Scatter(x=hourly_avg['hour_of_day'], y=hourly_avg['mean']+hourly_avg['std'], mode='lines', line=dict(width=0), showlegend=False))
                fig.add_trace(go.Scatter(x=hourly_avg['hour_of_day'], y=hourly_avg['mean']-hourly_avg['std'], mode='lines', line=dict(width=0), fill='tonexty', fillcolor='rgba(0,200,100,0.2)', showlegend=False))
                fig.add_vline(x=selected_hour, line_dash="dash", line_color="red", annotation_text=f"Selected: {selected_hour}:00")
                fig.update_layout(title="Average Hourly Demand Pattern", xaxis_title="Hour", yaxis_title="Avg Units Sold", height=400)
                st.plotly_chart(fig, use_container_width=True)
            with tab2:
                target_dt = pd.Timestamp(datetime.combine(selected_date, datetime.min.time()) + timedelta(hours=selected_hour)).tz_localize('Asia/Kolkata')
                start_dt = target_dt - timedelta(hours=24)
                end_dt = target_dt + timedelta(hours=24)
                context_data = vendor_hist[
                    (vendor_hist['datetime_parsed'] >= start_dt) &
                    (vendor_hist['datetime_parsed'] <= end_dt)
                ].copy()
                if not context_data.empty:
                    fig1 = go.Figure()
                    fig1.add_trace(go.Scatter(
                        x=context_data['datetime_parsed'].dt.tz_convert(None),
                        y=context_data['units_sold'],
                        mode='lines+markers',
                        line=dict(color='blue', width=2)
                    ))
                    fig1.update_layout(title="48-Hour Demand Context", xaxis_title="Date/Time", yaxis_title="Units Sold", height=400)
                    st.plotly_chart(fig1, use_container_width=True)
st.subheader("📆 Daily Average Demand")
daily_avg = vendor_hist.groupby(vendor_hist['datetime_parsed'].dt.date)['units_sold'].mean().reset_index()
fig_daily = go.Figure()
fig_daily.add_trace(go.Bar(
    x=daily_avg['datetime_parsed'], 
    y=daily_avg['units_sold'], 
    marker_color='orange'
))
fig_daily.update_layout(
    title="Average Daily Units Sold",
    xaxis_title="Date",
    yaxis_title="Avg Units Sold",
    height=400
)
st.plotly_chart(fig_daily, use_container_width=True)


st.subheader("📊 Average Demand by Weekday")
vendor_hist['weekday'] = vendor_hist['datetime_parsed'].dt.day_name()
weekday_avg = vendor_hist.groupby('weekday')['units_sold'].mean().reindex(
    ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
).reset_index()
fig_weekday = go.Figure()
fig_weekday.add_trace(go.Bar(
    x=weekday_avg['weekday'],
    y=weekday_avg['units_sold'],
    marker_color='green'
))
fig_weekday.update_layout(
    title="Average Units Sold by Weekday",
    xaxis_title="Weekday",
    yaxis_title="Avg Units Sold",
    height=400
)
st.plotly_chart(fig_weekday, use_container_width=True)


st.subheader("🌡️ Temperature vs Demand")
fig_temp = go.Figure()
fig_temp.add_trace(go.Scatter(
    x=vendor_hist['temperature_c'],
    y=vendor_hist['units_sold'],
    mode='markers',
    marker=dict(size=8, color='red', opacity=0.6)
))
fig_temp.update_layout(
    title="Temperature vs Units Sold",
    xaxis_title="Temperature (°C)",
    yaxis_title="Units Sold",
    height=400
)
st.plotly_chart(fig_temp, use_container_width=True)


st.subheader("☔ Rainfall Impact on Demand")
fig_rain = go.Figure()
fig_rain.add_trace(go.Box(
    x=vendor_hist['rainfall_mm'].round(0),
    y=vendor_hist['units_sold'],
    marker_color='blue'
))
fig_rain.update_layout(
    title="Units Sold vs Rainfall",
    xaxis_title="Rainfall (mm)",
    yaxis_title="Units Sold",
    height=400
)
st.plotly_chart(fig_rain, use_container_width=True)


st.subheader("🔥 Hourly Demand Heatmap by Weekday")
heatmap_data = vendor_hist.groupby(['weekday','hour_of_day'])['units_sold'].mean().reset_index()
heatmap_data['weekday'] = pd.Categorical(heatmap_data['weekday'], 
                                         categories=['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'], 
                                         ordered=True)
heatmap_pivot = heatmap_data.pivot(index='hour_of_day', columns='weekday', values='units_sold')
fig_heat = go.Figure(data=go.Heatmap(
    z=heatmap_pivot.values,
    x=heatmap_pivot.columns,
    y=heatmap_pivot.index,
    colorscale='Viridis'
))
fig_heat.update_layout(
    title="Heatmap of Average Units Sold by Hour and Weekday",
    xaxis_title="Weekday",
    yaxis_title="Hour of Day",
    height=500
)
st.plotly_chart(fig_heat, use_container_width=True)



st.subheader("🌦️ Temperature, Rainfall & Demand (3D)")
fig_3d = go.Figure()
fig_3d.add_trace(go.Scatter3d(
    x=vendor_hist['temperature_c'],
    y=vendor_hist['rainfall_mm'],
    z=vendor_hist['units_sold'],
    mode='markers',
    marker=dict(size=5, color=vendor_hist['units_sold'], colorscale='Rainbow', opacity=0.8)
))
fig_3d.update_layout(
    scene=dict(
        xaxis_title='Temperature (°C)',
        yaxis_title='Rainfall (mm)',
        zaxis_title='Units Sold'
    ),
    height=600,
    title="3D Scatter: Temperature & Rainfall vs Demand"
)
st.plotly_chart(fig_3d, use_container_width=True)




# ----------------------------
# Footer
st.markdown("---")
st.markdown("**Mumbai Street Vendor Demand Forecasting System** | Streamlit & ML-powered predictions")
