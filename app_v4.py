import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import plotly.graph_objects as go
from datetime import datetime, timedelta
import requests
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
        df['datetime_parsed'] = pd.to_datetime(df['datetime'], errors='coerce')
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
# Weather Alerts using WeatherAPI
@st.cache_data(ttl=600)
def get_weather_alerts():
    API_KEY = "e9323a9f57e6466e91d180135211911"
    location = "Mumbai"
    url = f"http://api.weatherapi.com/v1/alerts.json?key={API_KEY}&q={location}"
    try:
        response = requests.get(url)
        data = response.json()
        if "alerts" in data and data["alerts"]["alert"]:
            alerts = data["alerts"]["alert"]
            alert_list = []
            for a in alerts:
                alert_list.append({
                    "headline": a.get("headline", "No title"),
                    "severity": a.get("severity", "Unknown"),
                    "event": a.get("event", "N/A"),
                    "effective": a.get("effective", "N/A"),
                    "expires": a.get("expires", "N/A"),
                    "desc": a.get("desc", "No details available.")
                })
            return alert_list
        else:
            return []
    except Exception as e:
        st.warning(f"Error fetching weather alerts: {e}")
        return []

# ----------------------------
# Fetch current weather for Mumbai using WeatherAPI
@st.cache_data(ttl=600)
def get_current_weather():
    API_KEY = "e9323a9f57e6466e91d180135211911"
    location = "Mumbai"
    url = f"http://api.weatherapi.com/v1/current.json?key={API_KEY}&q={location}&aqi=no"
    try:
        response = requests.get(url)
        data = response.json()
        if data:
            location_str=f"{data['location']['name']} {data['location']['region']} {data['location']['country']}"
            last_updated = data['current']["last_updated"]
            temperature=data['current']["temp_c"]
            humidity=data['current']["humidity"]
            rainfall= data['current']["precip_mm"]
            condition=  data['current']["condition"]["text"]
            return temperature, humidity, rainfall , condition,last_updated,location_str
        else:
            st.warning("Failed to fetch weather, using default values.")
            return 28.0, 80.0, 0.0, "Clear", datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "Mumbai"
    except:
        st.warning("Error fetching weather, using default values.")
        return 28.0, 80.0, 0.0, "Clear", datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "Mumbai"

st.toast("Current Weather Data Extracted!", icon="🎉")  
temperature, humidity, rainfall , condition,last_updated,location = get_current_weather()
current_date = datetime.now().date()
current_hour = datetime.now().hour

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
selected_date = st.sidebar.date_input("Select Date:", value=current_date)
selected_hour = st.sidebar.slider("Select Hour:", 0, 23, current_hour)

st.sidebar.subheader("🌤️ Weather Conditions (Auto-Filled)")
st.sidebar.text(f"☁️ Condition: {condition} ")
st.sidebar.text(f"📍 Location: {location} ")
st.sidebar.text(f"🕒 Last Updated: {last_updated}")
st.sidebar.text(f"🌡️ Temperature: { temperature:.1f} °C" )
st.sidebar.text(f"💧 Humidity: {humidity}%")
st.sidebar.text(f"🌧️ Rainfall: {rainfall}")

weather_alerts = get_weather_alerts()
st.sidebar.subheader("⚠️ Weather Alerts")
if weather_alerts:
    for alert in weather_alerts:
        st.sidebar.markdown(f"""
        **{alert['event']} ({alert['severity']})**
        🕒 {alert['effective']} → {alert['expires']}  
        📢 {alert['headline']}  
        💬 _{alert['desc'][:120]}..._
        """)
else:
    st.sidebar.info("No active weather alerts for Mumbai.")

# Optional manual override
temperature = st.sidebar.slider("Temperature (°C) Override:", 22.0, 35.0, float(temperature), 0.5)
rainfall = st.sidebar.slider("Rainfall (mm) Override:", 0.0, 50.0, float(rainfall), 0.5)
humidity = st.sidebar.slider("Humidity (%) Override:", 50.0, 95.0, float(humidity), 1.0)

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
    hour = features_df.iloc[0]['hour_of_day']
    is_peak = features_df.iloc[0]['is_peak_hour']
    rainfall = features_df.iloc[0]['rainfall_mm']
    traffic = features_df.iloc[0]['traffic_density_encoded']
    is_festival_day = features_df.iloc[0]['is_festival']

    explanation = f"**Predicted demand: {prediction:.0f} units**\n\n"
    explanation += "✅ **Peak hour boost**\n" if is_peak else "⚠️ **Off-peak hour**\n"
    if rainfall > 5:
        explanation += "☔ **Rain effect**\n"
    if is_festival_day:
        explanation += "🎉 **Festival effect**\n"
    traffic_names = {0: 'Low', 1: 'Medium', 2: 'High'}
    traffic_name = traffic_names.get(int(traffic), 'Medium')
    if traffic == 2:
        explanation += "🚗 **High traffic boost**\n"
    elif traffic == 0:
        explanation += "🚶 **Low traffic reduction**\n"

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
    - **Temperature:** {temperature:.1f} °C
    - **Rainfall:** {rainfall:.1f} mm
    - **Humidity:** {humidity:.1f} %
    - **Traffic Density:** {traffic_density.capitalize()}
    - **Special Event Nearby:** {"Yes" if event_nearby else "No"}
    - **Nearby Competitors:** {competitor_count}
    - **Simulated Festival:** {selected_festival if selected_festival != 'None' else 'No Festival'}
    """)

# ----------------------------
# Historical & chart visualization
# Keep your existing charts unchanged
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
# 📊 Additional Visualizations
st.markdown("---")
st.header("🧠 Advanced Insights & Comparisons")

# 1️⃣ Feature Correlation Heatmap
st.subheader("📊 Feature Correlation Heatmap (Numeric Variables)")
try:
    numeric_cols = ['units_sold', 'temperature_c', 'rainfall_mm', 'humidity_pct', 'avg_price', 'competitor_count']
    corr_data = vendor_hist[numeric_cols].corr()
    fig_corr = go.Figure(data=go.Heatmap(
        z=corr_data.values,
        x=corr_data.columns,
        y=corr_data.columns,
        colorscale="RdBu",
        zmid=0
    ))
    fig_corr.update_layout(
        title="Correlation Matrix Between Key Variables",
        height=500
    )
    st.plotly_chart(fig_corr, use_container_width=True)
except Exception as e:
    st.warning(f"Could not plot correlation heatmap: {e}")

# 2️⃣ 7-Day Rolling Average Trend
st.subheader("📈 7-Day Rolling Average Demand Trend")
try:
    daily_avg['rolling_7d'] = daily_avg['units_sold'].rolling(7, min_periods=1).mean()
    fig_trend = go.Figure()
    fig_trend.add_trace(go.Scatter(
        x=daily_avg['datetime_parsed'],
        y=daily_avg['rolling_7d'],
        mode='lines+markers',
        line=dict(color='purple', width=3)
    ))
    fig_trend.update_layout(
        title="7-Day Smoothed Demand Trend",
        xaxis_title="Date",
        yaxis_title="7-Day Rolling Avg Units Sold",
        height=400
    )
    st.plotly_chart(fig_trend, use_container_width=True)
except Exception as e:
    st.warning(f"Could not plot rolling trend: {e}")

# 3️⃣ Vendor Comparison Chart
st.subheader("🏪 Vendor Comparison (Avg Daily Sales)")
try:
    vendor_comparison = historical_data.groupby('vendor_id')['units_sold'].mean().reset_index()
    vendor_comparison['vendor_name'] = vendor_comparison['vendor_id'].map(vendor_options)
    fig_comp = go.Figure()
    fig_comp.add_trace(go.Bar(
        x=vendor_comparison['vendor_name'],
        y=vendor_comparison['units_sold'],
        marker_color='teal'
    ))
    fig_comp.update_layout(
        title="Average Daily Sales Comparison Across Vendors",
        xaxis_title="Vendor",
        yaxis_title="Average Units Sold",
        height=400
    )
    st.plotly_chart(fig_comp, use_container_width=True)
except Exception as e:
    st.warning(f"Could not plot vendor comparison: {e}")

# ----------------------------
# 📥 Download Prediction Option
st.markdown("---")
st.header("📥 Export Prediction Result")

if 'last_prediction' in st.session_state and 'last_features' in st.session_state:
    result_df = st.session_state.last_features.copy()
    result_df['predicted_units'] = st.session_state.last_prediction
    result_df['vendor_name'] = vendor_options[st.session_state.last_vendor]
    result_df['prediction_time'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    csv_data = result_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="⬇️ Download Latest Prediction as CSV",
        data=csv_data,
        file_name=f"{st.session_state.last_vendor}_prediction.csv",
        mime="text/csv",
    )
else:
    st.info("Run a prediction first to enable download.")




# app_complete.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ----------------------------
# Page config
st.set_page_config(
    page_title="Mumbai Street Vendor Demand Forecasting",
    page_icon="🍛",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------------------
# ----------------------------
# Translation dictionary
translations = {
    "English": {
        "app_title": "Mumbai Street Vendor Demand Forecasting",
        "app_subtitle": "Predict hourly demand and explore scenario simulations with interactive charts.",
        "select_language": "Select Language",
        "predict_units": "Predict Units",
        "temperature": "Temperature",
        "rainfall": "Rainfall",
        "humidity": "Humidity",
        "weather_alerts": "Weather Alerts",
        "no_alerts": "No active weather alerts for Mumbai.",
        "vendor_select": "Select Vendor:",
        "date": "Select Date:",
        "hour": "Select Hour:",
        "temperature_override": "Temperature (°C) Override:",
        "rainfall_override": "Rainfall (mm) Override:",
        "humidity_override": "Humidity (%) Override:",
        "traffic_density": "Traffic Density:",
        "special_event": "Special Event Nearby",
        "nearby_competitors": "Nearby Competitors:",
        "simulate_festival": "Simulate Festival:",
        "demand_prediction": "Demand Prediction",
        "scenario_parameters": "Scenario Parameters",
        "historical_context": "Historical Context",
        "daily_avg_demand": "Daily Average Demand",
        "avg_demand_weekday": "Average Demand by Weekday",
        "temp_vs_demand": "Temperature vs Demand",
        "rainfall_vs_demand": "Rainfall Impact on Demand",
        "hourly_heatmap": "Hourly Demand Heatmap by Weekday",
        "temp_rain_demand_3d": "Temperature, Rainfall & Demand (3D)",
        "feature_corr_heatmap": "Feature Correlation Heatmap (Numeric Variables)",
        "rolling_trend": "7-Day Rolling Average Demand Trend",
        "vendor_comparison": "Vendor Comparison (Avg Daily Sales)",
        "export_prediction": "Export Prediction Result",
        "run_prediction_first": "Run a prediction first to enable download.",
        "prediction_success": "Predicted Units Sold:",
        "prediction_explanation": "Prediction Explanation",
        "prediction_controls": "Prediction Controls",
        "weather_section": "Weather Conditions (Auto-Filled)",
        "environment": "Environment",
        "events": "Events",
        "footer": "Mumbai Street Vendor Demand Forecasting System | Streamlit & ML-powered predictions"
    },
    "Hindi": {
        "app_title": "मुंबई स्ट्रीट विक्रेता मांग पूर्वानुमान",
        "app_subtitle": "घंटा-घंटा मांग का पूर्वानुमान लगाएँ और इंटरैक्टिव चार्ट्स के साथ परिदृश्य सिमुलेशन देखें।",
        "select_language": "भाषा चुनें",
        "predict_units": "यूनिट्स का पूर्वानुमान लगाएँ",
        "temperature": "तापमान",
        "rainfall": "वर्षा",
        "humidity": "आर्द्रता",
        "weather_alerts": "मौसम अलर्ट",
        "no_alerts": "मुंबई के लिए कोई सक्रिय मौसम अलर्ट नहीं।",
        "vendor_select": "विक्रेता चुनें:",
        "date": "तारीख चुनें:",
        "hour": "घंटा चुनें:",
        "temperature_override": "तापमान (°C) ओवरराइड:",
        "rainfall_override": "वर्षा (mm) ओवरराइड:",
        "humidity_override": "आर्द्रता (%) ओवरराइड:",
        "traffic_density": "ट्रैफिक घनत्व:",
        "special_event": "पास में विशेष कार्यक्रम",
        "nearby_competitors": "पास के प्रतियोगी:",
        "simulate_festival": "त्योहार सिमुलेट करें:",
        "demand_prediction": "मांग पूर्वानुमान",
        "scenario_parameters": "परिदृश्य पैरामीटर",
        "historical_context": "ऐतिहासिक संदर्भ",
        "daily_avg_demand": "दैनिक औसत मांग",
        "avg_demand_weekday": "सप्ताह के दिन के अनुसार औसत मांग",
        "temp_vs_demand": "तापमान बनाम मांग",
        "rainfall_vs_demand": "वर्षा का प्रभाव",
        "hourly_heatmap": "सप्ताह के दिन और घंटे के अनुसार मांग हीटमैप",
        "temp_rain_demand_3d": "तापमान, वर्षा और मांग (3D)",
        "feature_corr_heatmap": "मुख्य विशेषताओं का सहसंबंध हीटमैप",
        "rolling_trend": "7-दिन की औसत मांग रुझान",
        "vendor_comparison": "विक्रेता तुलना (औसत दैनिक बिक्री)",
        "export_prediction": "पूर्वानुमान परिणाम निर्यात करें",
        "run_prediction_first": "डाउनलोड सक्षम करने के लिए पहले पूर्वानुमान चलाएँ।",
        "prediction_success": "पूर्वानुमानित यूनिट्स:",
        "prediction_explanation": "पूर्वानुमान व्याख्या",
        "prediction_controls": "पूर्वानुमान नियंत्रण",
        "weather_section": "मौसम स्थितियाँ (ऑटो-भरी हुई)",
        "environment": "पर्यावरण",
        "events": "इवेंट्स",
        "footer": "मुंबई स्ट्रीट विक्रेता मांग पूर्वानुमान सिस्टम | Streamlit & ML द्वारा संचालित"
    },
    "Marathi": {
        "app_title": "मुंबई स्ट्रीट विक्रेता मागणी अंदाज",
        "app_subtitle": "तासांनुसार मागणीचे अंदाज लावा आणि इंटरॅक्टिव चार्टसह सिम्युलेशन पहा.",
        "select_language": "भाषा निवडा",
        "predict_units": "युनिट्सचा अंदाज लावा",
        "temperature": "तापमान",
        "rainfall": "पाऊस",
        "humidity": "आर्द्रता",
        "weather_alerts": "हवामान अलर्ट",
        "no_alerts": "मुंबईसाठी कोणताही सक्रिय हवामान अलर्ट नाही.",
        "vendor_select": "विक्रेता निवडा:",
        "date": "तारीख निवडा:",
        "hour": "तास निवडा:",
        "temperature_override": "तापमान (°C) ओव्हरराईड:",
        "rainfall_override": "पाऊस (mm) ओव्हरराईड:",
        "humidity_override": "आर्द्रता (%) ओव्हरराईड:",
        "traffic_density": "वाहन दाटी:",
        "special_event": "शेजारील विशेष कार्यक्रम",
        "nearby_competitors": "शेजारील स्पर्धक:",
        "simulate_festival": "सण सिम्युलेट करा:",
        "demand_prediction": "मागणी अंदाज",
        "scenario_parameters": "परिस्थिती पॅरामीटर्स",
        "historical_context": "ऐतिहासिक संदर्भ",
        "daily_avg_demand": "दररोजची सरासरी मागणी",
        "avg_demand_weekday": "सप्ताहाच्या दिवशी सरासरी मागणी",
        "temp_vs_demand": "तापमान विरुद्ध मागणी",
        "rainfall_vs_demand": "पावसाचा परिणाम",
        "hourly_heatmap": "सप्ताहाच्या दिवसांनुसार तासानुसार मागणी हीटमॅप",
        "temp_rain_demand_3d": "तापमान, पाऊस आणि मागणी (3D)",
        "feature_corr_heatmap": "मुख्य वैशिष्ट्ये सहसंबंध हीटमैप",
        "rolling_trend": "7-दिवसांची सरासरी मागणी प्रवाह",
        "vendor_comparison": "विक्रेता तुलना (सरासरी दैनिक विक्री)",
        "export_prediction": "अंदाज परिणाम निर्यात करा",
        "run_prediction_first": "डाउनलोड सक्षम करण्यासाठी प्रथम अंदाज चालवा.",
        "prediction_success": "अंदाजित युनिट्स:",
        "prediction_explanation": "अंदाज स्पष्टीकरण",
        "prediction_controls": "अंदाज नियंत्रण",
        "weather_section": "हवामान स्थिती (ऑटो-भरलेले)",
        "environment": "पर्यावरण",
        "events": "कार्यक्रम",
        "footer": "मुंबई स्ट्रीट विक्रेता मागणी अंदाज प्रणाली | Streamlit & ML-शक्तीने"
    }
}

st.sidebar.subheader("Select Language")
language = st.sidebar.selectbox("", ["English", "Hindi", "Marathi"], index=0)
lang = translations[language]
def t(key): return lang.get(key, key)

# ----------------------------
# Load model & metadata
@st.cache_resource
def load_model_artifacts():
    model = joblib.load('rf_model.pkl')
    label_encoders = joblib.load('label_encoders.pkl')
    feature_columns = joblib.load('feature_columns.pkl')
    with open('model_metadata.json') as f:
        metadata = json.load(f)
    return model, label_encoders, feature_columns, metadata

model, label_encoders, feature_columns, metadata = load_model_artifacts()

# ----------------------------
# Load historical data
@st.cache_data
def load_historical_data():
    df = pd.read_csv('mumbai_vendors_hourly_20250701_20250930.csv')
    df['datetime_parsed'] = pd.to_datetime(df['datetime'], errors='coerce')
    if df['datetime_parsed'].dt.tz is None:
        df['datetime_parsed'] = df['datetime_parsed'].dt.tz_localize('Asia/Kolkata')
    else:
        df['datetime_parsed'] = df['datetime_parsed'].dt.tz_convert('Asia/Kolkata')
    return df

historical_data = load_historical_data()

# ----------------------------
# Feature preparation
def safe_transform(encoder, value):
    return encoder.transform([value])[0] if value in encoder.classes_ else -1

def create_prediction_features(vendor_id, date, hour, temp, rain, humid, traffic,
                               event, competitors, festival):
    vendor_info = {
        'vendor_01': {'location_type': 'business_district', 'cuisine_type': 'main_course', 'avg_price': 85.0, 'menu_diversity': 12, 'peak_hours': [12, 13, 19, 20]},
        'vendor_02': {'location_type': 'near_college', 'cuisine_type': 'chai_snacks', 'avg_price': 25.0, 'menu_diversity': 8, 'peak_hours': [7,8,16,17,18]},
        'vendor_03': {'location_type': 'market', 'cuisine_type': 'main_course', 'avg_price': 75.0, 'menu_diversity': 15, 'peak_hours': [11,12,19,20]}
    }
    info = vendor_info[vendor_id]
    df = pd.DataFrame({
        'vendor_id_encoded': [safe_transform(label_encoders['vendor_id'], vendor_id)],
        'location_type_encoded': [safe_transform(label_encoders['location_type'], info['location_type'])],
        'cuisine_type_encoded': [safe_transform(label_encoders['cuisine_type'], info['cuisine_type'])],
        'avg_price': [info['avg_price']],
        'menu_diversity': [info['menu_diversity']],
        'hour_of_day': [hour],
        'is_weekend': [1 if date.weekday()>=5 else 0],
        'is_holiday': [0],
        'is_festival': [festival],
        'temperature_c': [temp],
        'rainfall_mm': [rain],
        'humidity_pct': [humid],
        'wind_speed_kmh': [0],
        'event_nearby': [event],
        'traffic_density_encoded': [traffic],
        'competitor_count': [competitors],
        'lag_1h_units': [0],
        'lag_24h_units': [0],
        'rolling_avg_24h': [0],
        'is_peak_hour': [1 if hour in info['peak_hours'] else 0],
        'is_evening': [1 if 17<=hour<=20 else 0],
        'is_morning': [1 if 6<=hour<=11 else 0]
    })
    return df, info

# ----------------------------
# Sidebar controls
st.sidebar.subheader(t("prediction_controls"))
vendor_options = historical_data['vendor_id'].unique()
selected_vendor = st.sidebar.selectbox(t("vendor_select"), vendor_options)
selected_date = st.sidebar.date_input(t("date"), datetime.today())
selected_hour = st.sidebar.slider(t("hour"), 0, 23, datetime.now().hour)
temperature_override = st.sidebar.number_input(t("temperature_override"), value=30)
rainfall_override = st.sidebar.number_input(t("rainfall_override"), value=0)
humidity_override = st.sidebar.number_input(t("humidity_override"), value=70)
traffic_density = st.sidebar.slider(t("traffic_density"), 0, 10, 5)
special_event = st.sidebar.checkbox(t("special_event"))
nearby_competitors = st.sidebar.number_input(t("nearby_competitors"), value=0)
simulate_festival = st.sidebar.checkbox(t("simulate_festival"))

# ----------------------------
# Prediction
if st.sidebar.button(t("predict_units")):
    features_df, vendor_info = create_prediction_features(
        selected_vendor, selected_date, selected_hour,
        temperature_override, rainfall_override, humidity_override,
        traffic_density, special_event, nearby_competitors, simulate_festival
    )
    X = features_df[feature_columns]
    prediction = model.predict(X)[0]
    st.subheader(t("prediction_success"))
    st.write(prediction)

# ----------------------------
# Historical visualizations
st.subheader(t("historical_context"))

# Daily avg demand
daily_avg = historical_data.groupby(historical_data['datetime_parsed'].dt.date)['units_sold'].mean().reset_index()
fig_daily = px.line(daily_avg, x='datetime_parsed', y='units_sold', title=t("daily_avg_demand"))
st.plotly_chart(fig_daily, use_container_width=True)

# Hourly heatmap
historical_data['hour'] = historical_data['datetime_parsed'].dt.hour
historical_data['weekday'] = historical_data['datetime_parsed'].dt.day_name()
heatmap_data = historical_data.groupby(['weekday','hour'])['units_sold'].mean().reset_index()
heatmap_pivot = heatmap_data.pivot('weekday','hour','units_sold')
fig_heatmap = px.imshow(heatmap_pivot, aspect='auto', title=t("hourly_heatmap"))
st.plotly_chart(fig_heatmap, use_container_width=True)

# Temp vs Demand
fig_temp = px.scatter(historical_data, x='temperature_c', y='units_sold', color='vendor_id', trendline='ols', title=t("temp_vs_demand"))
st.plotly_chart(fig_temp, use_container_width=True)

# Rainfall vs Demand
fig_rain = px.scatter(historical_data, x='rainfall_mm', y='units_sold', color='vendor_id', trendline='ols', title=t("rainfall_vs_demand"))
st.plotly_chart(fig_rain, use_container_width=True)

# 3D plot: Temp, Rainfall, Demand
fig_3d = px.scatter_3d(historical_data, x='temperature_c', y='rainfall_mm', z='units_sold',
                       color='vendor_id', size='units_sold', title=t("temp_rain_demand_3d"))
st.plotly_chart(fig_3d, use_container_width=True)

# ----------------------------
# Footer
st.markdown("---")
st.markdown(t("footer"))
