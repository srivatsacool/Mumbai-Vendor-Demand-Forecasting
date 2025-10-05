# app_multilang_simplified.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import plotly.graph_objects as go
from datetime import datetime, timedelta
from deep_translator import GoogleTranslator
import warnings
import requests
warnings.filterwarnings('ignore')

# ----------------------------
# NOTE:
# Requires: pip install streamlit pandas numpy plotly joblib deep-translator
# ----------------------------

# ----------------------------
# Translation helper (cached)
@st.cache_data(ttl=24 * 3600, show_spinner=False)
def translate_text(text: str, target_lang: str) -> str:
    """
    Translate `text` into `target_lang` using GoogleTranslator (deep-translator).
    If target_lang == 'en', returns original text.
    Cached for 24 hours to reduce repeated calls.
    """
    if not text or target_lang == "en":
        return text
    try:
        # auto-detect source
        return GoogleTranslator(source="auto", target=target_lang).translate(text)
    except Exception:
        # On failure, return English fallback
        return text

# Shortcut to use in code
def t(text: str, lang: str) -> str:
    return translate_text(text, lang)

# ----------------------------
# Page configuration
st.set_page_config(
    page_title="Mumbai Street Vendor Demand Forecasting",
    page_icon="🍛",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------------------
# Language selector
st.sidebar.markdown("### 🌐 Language Settings")
lang_choice = st.sidebar.selectbox(
    "Select Language / भाषा / भाषा निवडा",
    ["English", "हिन्दी", "मराठी"]
)
lang_map = {"English": "en", "हिन्दी": "hi", "मराठी": "mr"}
target_lang = lang_map[lang_choice]

# small helper to translate f-strings or multi-line text concisely
def TT(s: str) -> str:
    return t(s, target_lang)

# ----------------------------
# Header
st.markdown(f"<h1 style='text-align:center'>{TT('🍛 Mumbai Street Vendor Demand Forecasting')}</h1>", unsafe_allow_html=True)
st.markdown(f"<p style='text-align:center'>{TT('Predict hourly demand and explore scenario simulations with interactive charts.')}</p>", unsafe_allow_html=True)
st.markdown("---")

# ----------------------------
# Load model & artifacts (cached)
@st.cache_resource
def load_model_artifacts():
    try:
        try:
            model = joblib.load('xgb_model.pkl')
            model_type = "XGBoost"
        except Exception:
            # try RF
            try:
                model = joblib.load('rf_model.pkl')
                model_type = "Random Forest"
            except Exception:
                model = None
                model_type = "Dummy"
        
        # label encoders and feature columns optional
        try:
            label_encoders = joblib.load('label_encoders.pkl')
        except Exception:
            label_encoders = None
        try:
            feature_columns = joblib.load('feature_columns.pkl')
        except Exception:
            feature_columns = None

        try:
            with open('model_metadata.json', 'r') as f:
                metadata = json.load(f)
        except Exception:
            metadata = {}

        return model, label_encoders, feature_columns, metadata, model_type
    except Exception as e:
        st.error(TT("Error loading model artifacts:") + f" {e}")
        return None, None, None, {}, "Error"

model, label_encoders, feature_columns, metadata, model_type = load_model_artifacts()

# If no model available, use a safe dummy predictor function
def dummy_predict(X_df: pd.DataFrame) -> np.ndarray:
    # produce deterministic-ish values based on some inputs
    base = 30 + 2 * X_df['hour_of_day'].astype(float)
    price_factor = (100.0 - X_df.get('avg_price', 50)) / 100.0
    temp_factor = 1 + (X_df.get('temperature_c', 28) - 28) * 0.02
    competitor_factor = 1 - (X_df.get('competitor_count', 3) * 0.03)
    pred = base * price_factor * temp_factor * competitor_factor
    # ensure non-negative ints
    return np.maximum(1, np.round(pred)).astype(int)

# ----------------------------
# Load historical data (cached)
@st.cache_data
def load_historical():
    try:
        df = pd.read_csv('mumbai_vendors_hourly_20250701_20250930.csv')
        df['datetime_parsed'] = pd.to_datetime(df['datetime'], errors='coerce')
        if df['datetime_parsed'].dt.tz is None:
            df['datetime_parsed'] = df['datetime_parsed'].dt.tz_localize('Asia/Kolkata')
        else:
            df['datetime_parsed'] = df['datetime_parsed'].dt.tz_convert('Asia/Kolkata')
        return df
    except Exception:
        # fallback small synthetic dataset (keeps app functional)
        now = pd.Timestamp.now(tz='Asia/Kolkata')
        dates = pd.date_range(now - pd.Timedelta(days=14), now, freq='H', tz='Asia/Kolkata')
        df = pd.DataFrame({
            'datetime': dates.astype(str),
            'datetime_parsed': dates,
            'vendor_id': np.random.choice(['vendor_01','vendor_02','vendor_03'], size=len(dates)),
            'units_sold': np.random.randint(10, 200, size=len(dates)),
            'temperature_c': np.random.randint(24, 36, size=len(dates)),
            'rainfall_mm': np.random.choice([0,0,0,1,2,5,10], size=len(dates)),
            'humidity_pct': np.random.randint(50, 90, size=len(dates)),
            'hour_of_day': dates.hour
        })
        return df

historical_data = load_historical()
# ----------------------------
# Weather Alerts using WeatherAPI
@st.cache_data(ttl=600)
def get_weather_alerts():
    import requests 
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
    import requests 
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
# Sidebar: Scenario inputs (simplified, no weather API)
st.sidebar.header(TT("📊 Prediction Controls"))

vendor_options = {
    'vendor_01': TT('BKC_Pavbhaji (Business District)'),
    'vendor_02': TT('Churchgate_Chai (Near College)'),
    'vendor_03': TT('Dadar_Chaat (Market)'),
    'vendor_04': TT('Andheri_Juice (Metro Station)'),
    'vendor_05': TT('Powai_Dessert (Office Area)')
}

selected_vendor = st.sidebar.selectbox(
    TT("Select Vendor:"),
    options=list(vendor_options.keys()),
    format_func=lambda x: vendor_options[x]
)

st.sidebar.subheader(TT("📅 Date & Time"))
selected_date = st.sidebar.date_input(TT("Select Date:"), value=current_date)
selected_hour = st.sidebar.slider(TT("Select Hour:"), 0, 23, int(current_hour))

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


st.sidebar.subheader(TT("🚦 Environment"))
traffic_density = st.sidebar.selectbox(TT("Traffic Density:"), [TT('low'), TT('medium'), TT('high')], index=1)
event_nearby = st.sidebar.checkbox(TT("Special Event Nearby"), value=False)
competitor_count = st.sidebar.slider(TT("Nearby Competitors:"), 0, 20, 5)

st.sidebar.subheader(TT("🎉 Festival Options"))
festival_options = ['None', 'Independence Day', 'Janmashtami', 'Dahi Handi', 'Ganesh Chaturthi', 'Ganesh Visarjan', 'Eid-e-Milad']
selected_festival = st.sidebar.selectbox(TT("Simulate Festival:"), festival_options)

# ----------------------------
# Vendor static info (used by features)
vendor_details = {
    'vendor_01': {'location_type': 'business_district', 'cuisine_type': 'main_course', 'avg_price': 85.0, 'menu_diversity': 12, 'peak_hours': [12,13,19,20]},
    'vendor_02': {'location_type': 'near_college', 'cuisine_type': 'chai_snacks', 'avg_price': 25.0, 'menu_diversity': 8, 'peak_hours': [7,8,16,17,18]},
    'vendor_03': {'location_type': 'market', 'cuisine_type': 'chaat', 'avg_price': 45.0, 'menu_diversity': 15, 'peak_hours': [17,18,19,20,21]},
    'vendor_04': {'location_type': 'metro_station', 'cuisine_type': 'beverages', 'avg_price': 35.0, 'menu_diversity': 10, 'peak_hours': [7,8,9,17,18,19]},
    'vendor_05': {'location_type': 'office', 'cuisine_type': 'dessert', 'avg_price': 65.0, 'menu_diversity': 7, 'peak_hours': [15,16,21,22]}
}

# ----------------------------
# Safe label encoder wrapper (if available)
def safe_transform(encoder, value):
    try:
        return int(encoder.transform([value])[0])
    except Exception:
        return -1

# ----------------------------
# Feature generation
def create_prediction_features(vendor_id, date_obj, hour, temp, rain, humid, traffic, event, competitors, festival):
    vendor_info = vendor_details[vendor_id]
    dt = pd.Timestamp(datetime.combine(date_obj, datetime.min.time()) + timedelta(hours=int(hour))).tz_localize('Asia/Kolkata')
    day_of_week = int(dt.weekday())
    is_weekend = int(day_of_week >= 5)
    is_festival = int(festival != 'None')
    is_holiday = int(festival == 'Independence Day')
    wind_speed = 15.0 + (5.0 if rain > 0 else 0.0)
    is_peak_hour = int(int(hour) in vendor_info['peak_hours'])
    is_evening = int(17 <= int(hour) <= 21)
    is_morning = int(6 <= int(hour) <= 10)
    base_demand = 15 if vendor_id == 'vendor_01' else 20
    lag_1h = max(1, int(base_demand * (0.8 + 0.4 * np.random.random())))
    lag_24h = max(1, int(base_demand * (0.9 + 0.2 * np.random.random())))
    rolling_avg = (lag_1h + lag_24h) / 2.0

    features = {
        'vendor_id_encoded': vendor_id,  # model may expect encoded; fallback handled
        'location_type': vendor_info['location_type'],
        'cuisine_type': vendor_info['cuisine_type'],
        'avg_price': vendor_info['avg_price'],
        'menu_diversity': vendor_info['menu_diversity'],
        'hour_of_day': int(hour),
        'day_of_week': day_of_week,
        'is_weekend': is_weekend,
        'is_holiday': is_holiday,
        'is_festival': is_festival,
        'temperature_c': float(temp),
        'rainfall_mm': float(rain),
        'humidity_pct': float(humid),
        'wind_speed_kmh': float(wind_speed),
        'event_nearby': int(bool(event)),
        'traffic_density': traffic,
        'competitor_count': int(competitors),
        'lag_1h_units': int(lag_1h),
        'lag_24h_units': int(lag_24h),
        'rolling_avg_24h': float(rolling_avg),
        'is_peak_hour': is_peak_hour,
        'is_evening': is_evening,
        'is_morning': is_morning
    }

    # Make DataFrame with columns matching model expectations if available
    df_feat = pd.DataFrame([features])

    # If label_encoders exist and model expects encoded values, try to encode
    if label_encoders is not None:
        try:
            # Example safe encodings if encoders exist
            for key in ['vendor_id', 'location_type', 'cuisine_type', 'traffic_density']:
                enc_key = f"{key}"
                if enc_key in label_encoders:
                    # map original string to encoded
                    if key == 'vendor_id':
                        df_feat['vendor_id_encoded'] = safe_transform(label_encoders[enc_key], vendor_id)
                    else:
                        df_feat[f"{key}_encoded"] = safe_transform(label_encoders[enc_key], features[key])
        except Exception:
            pass

    return df_feat, vendor_info

# ----------------------------
# Prediction explanation generator
def generate_explanation(prediction, features_df, vendor_info, festival):
    f = features_df.iloc[0]
    hour = int(f['hour_of_day'])
    is_peak = int(f.get('is_peak_hour', 0))
    rainfall = float(f.get('rainfall_mm', 0.0))
    traffic = f.get('traffic_density', 'medium')
    is_festival_day = int(f.get('is_festival', 0))

    explanation_lines = []
    explanation_lines.append(TT(f"**Predicted demand: {prediction:.0f} units**"))
    if is_peak:
        explanation_lines.append(TT("✅ Peak hour boost"))
    else:
        explanation_lines.append(TT("⚠️ Off-peak hour"))

    if rainfall > 5:
        explanation_lines.append(TT("☔ Rain effect: demand may change"))

    if is_festival_day:
        explanation_lines.append(TT("🎉 Festival effect: higher demand possible"))

    if traffic == 'high' or traffic == TT('high'):
        explanation_lines.append(TT("🚗 High traffic boost"))
    elif traffic == 'low' or traffic == TT('low'):
        explanation_lines.append(TT("🚶 Low traffic — lower footfall"))

    location_insights = {
        'business_district': TT('Office workers drive weekday lunch/dinner demand'),
        'market': TT('Market location benefits from continuous foot traffic'),
        'metro_station': TT('Commuters create morning and evening rush patterns'),
        'near_college': TT('Students provide consistent demand with price sensitivity'),
        'office': TT('Corporate area with break-time and after-work peaks')
    }

    loc_text = location_insights.get(vendor_info['location_type'], "")
    if loc_text:
        explanation_lines.append(TT("📍 Location insight: ") + loc_text)

    return "\n\n".join(explanation_lines)

# ----------------------------
# Layout: two columns
col1, col2 = st.columns([1.5, 1])

with col1:
    st.subheader(TT("🎯 Demand Prediction"))
    st.caption(TT(f"🧠 Using model: {model_type}"))

    predict_button = st.button(TT("🔮 Predict Units"), type="primary")
    if predict_button:
        features_df, vendor_info = create_prediction_features(
            selected_vendor, selected_date, selected_hour,
            temperature, rainfall, humidity, traffic_density,
            event_nearby, competitor_count, selected_festival
        )

        # Choose model or dummy
        try:
            if model is not None:
                # If model expects specific feature ordering, try to select feature_columns
                if feature_columns is not None and isinstance(feature_columns, (list, tuple)):
                    X = features_df.reindex(columns=feature_columns, fill_value=0)
                else:
                    X = features_df
                prediction = model.predict(X)[0]
            else:
                prediction = int(dummy_predict(features_df)[0])
        except Exception:
            # fallback
            prediction = int(dummy_predict(features_df)[0])

        uncertainty = float(metadata.get('prediction_uncertainty_std', 3.0))
        st.success(TT(f"**Predicted Units Sold: {prediction:.0f}**"))

        st.markdown("### " + TT("📝 Prediction Explanation"))
        st.markdown(generate_explanation(prediction, features_df, vendor_info, selected_festival))

        # Save to session_state for download
        st.session_state['last_prediction'] = int(prediction)
        st.session_state['last_features'] = features_df
        st.session_state['last_vendor'] = selected_vendor

with col2:
    st.subheader(TT("🛠️ Scenario Parameters"))
    st.markdown(TT(f"""
    - **Vendor:** {vendor_options[selected_vendor]}
    - **Date/Time:** {selected_date} at {selected_hour}:00
    - **Temperature:** {temperature:.1f} °C
    - **Rainfall:** {rainfall:.1f} mm
    - **Humidity:** {humidity:.1f} %
    - **Traffic Density:** {traffic_density}
    - **Special Event Nearby:** {'Yes' if event_nearby else 'No'}
    - **Nearby Competitors:** {competitor_count}
    - **Simulated Festival:** {selected_festival if selected_festival != 'None' else 'No Festival'}
    """))

st.markdown("---")

# ----------------------------
# Historical visualizations (simplified)
st.subheader(TT("📈 Historical Context"))
if historical_data is not None:
    vendor_hist = historical_data[historical_data['vendor_id'] == selected_vendor].copy()
    if vendor_hist.empty:
        st.info(TT("No historical data available for this vendor. Showing aggregated view."))
        vendor_hist = historical_data.copy()
    # Hourly pattern
    try:
        hourly_avg = vendor_hist.groupby('hour_of_day')['units_sold'].agg(['mean', 'std']).reset_index()
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=hourly_avg['hour_of_day'],
            y=hourly_avg['mean'],
            mode='lines+markers',
            name=TT('Avg Units'),
            line=dict(width=3)
        ))
        fig.add_vline(x=int(selected_hour), line_dash="dash", line_color="red",
                      annotation_text=TT(f"Selected: {selected_hour}:00"))
        fig.update_layout(title=TT("Average Hourly Demand Pattern"), xaxis_title=TT("Hour"), yaxis_title=TT("Avg Units Sold"), height=380)
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.warning(TT("Could not compute hourly averages:") + f" {e}")

    # 48-hour context
    try:
        target_dt = pd.Timestamp(datetime.combine(selected_date, datetime.min.time()) + timedelta(hours=int(selected_hour))).tz_localize('Asia/Kolkata')
        start_dt = target_dt - timedelta(hours=24)
        end_dt = target_dt + timedelta(hours=24)
        context_data = vendor_hist[(vendor_hist['datetime_parsed'] >= start_dt) & (vendor_hist['datetime_parsed'] <= end_dt)].copy()
        if not context_data.empty:
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(
                x=context_data['datetime_parsed'].dt.tz_convert(None),
                y=context_data['units_sold'],
                mode='lines+markers',
                name=TT('Units Sold')
            ))
            fig2.update_layout(title=TT("48-Hour Demand Context"), xaxis_title=TT("Date/Time"), yaxis_title=TT("Units Sold"), height=380)
            st.plotly_chart(fig2, use_container_width=True)
    except Exception:
        pass

# Daily average
try:
    daily_avg = vendor_hist.groupby(vendor_hist['datetime_parsed'].dt.date)['units_sold'].mean().reset_index()
    daily_avg.columns = ['date', 'units_sold']
    fig_daily = go.Figure()
    fig_daily.add_trace(go.Bar(x=daily_avg['date'], y=daily_avg['units_sold'], name=TT('Avg Units')))
    fig_daily.update_layout(title=TT("Average Daily Units Sold"), xaxis_title=TT("Date"), yaxis_title=TT("Avg Units Sold"), height=380)
    st.plotly_chart(fig_daily, use_container_width=True)
except Exception:
    pass

# ----------------------------
# Extra visualizations
st.subheader(TT("📊 Average Demand by Weekday"))
try:
    vendor_hist['weekday'] = vendor_hist['datetime_parsed'].dt.day_name()
    weekday_order = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
    weekday_avg = vendor_hist.groupby('weekday')['units_sold'].mean().reindex(weekday_order).reset_index()
    fig_w = go.Figure()
    fig_w.add_trace(go.Bar(x=weekday_avg['weekday'], y=weekday_avg['units_sold']))
    fig_w.update_layout(title=TT("Average Units Sold by Weekday"), xaxis_title=TT("Weekday"), yaxis_title=TT("Avg Units Sold"), height=350)
    st.plotly_chart(fig_w, use_container_width=True)
except Exception:
    pass

# Temperature vs demand scatter
st.subheader(TT("🌡️ Temperature vs Demand"))
try:
    fig_temp = go.Figure()
    fig_temp.add_trace(go.Scatter(x=vendor_hist['temperature_c'], y=vendor_hist['units_sold'], mode='markers', marker=dict(size=7, opacity=0.6)))
    fig_temp.update_layout(title=TT("Temperature vs Units Sold"), xaxis_title=TT("Temperature (°C)"), yaxis_title=TT("Units Sold"), height=350)
    st.plotly_chart(fig_temp, use_container_width=True)
except Exception:
    pass

# Rainfall impact
st.subheader(TT("☔ Rainfall Impact on Demand"))
try:
    fig_rain = go.Figure()
    fig_rain.add_trace(go.Box(x=vendor_hist['rainfall_mm'].round(0), y=vendor_hist['units_sold']))
    fig_rain.update_layout(title=TT("Units Sold vs Rainfall"), xaxis_title=TT("Rainfall (mm)"), yaxis_title=TT("Units Sold"), height=350)
    st.plotly_chart(fig_rain, use_container_width=True)
except Exception:
    pass

# Heatmap
st.subheader(TT("🔥 Hourly Demand Heatmap by Weekday"))
try:
    heatmap_data = vendor_hist.groupby([vendor_hist['datetime_parsed'].dt.day_name(), 'hour_of_day'])['units_sold'].mean().reset_index()
    heatmap_data.columns = ['weekday', 'hour_of_day', 'units_sold']
    heatmap_data['weekday'] = pd.Categorical(heatmap_data['weekday'], categories=['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'], ordered=True)
    heatmap_pivot = heatmap_data.pivot(index='hour_of_day', columns='weekday', values='units_sold')
    fig_heat = go.Figure(data=go.Heatmap(z=heatmap_pivot.values, x=heatmap_pivot.columns, y=heatmap_pivot.index, colorscale='Viridis'))
    fig_heat.update_layout(title=TT("Heatmap of Average Units Sold by Hour and Weekday"), xaxis_title=TT("Weekday"), yaxis_title=TT("Hour of Day"), height=420)
    st.plotly_chart(fig_heat, use_container_width=True)
except Exception:
    pass

# ----------------------------
# Export / Download Prediction
st.markdown("---")
st.header(TT("📥 Export Prediction Result"))
if 'last_prediction' in st.session_state and 'last_features' in st.session_state:
    result_df = st.session_state['last_features'].copy()
    result_df['predicted_units'] = st.session_state['last_prediction']
    result_df['vendor_name'] = vendor_options[st.session_state['last_vendor']]
    result_df['prediction_time'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    csv_data = result_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label=TT("⬇️ Download Latest Prediction as CSV"),
        data=csv_data,
        file_name=f"{st.session_state['last_vendor']}_prediction.csv",
        mime="text/csv",
    )
else:
    st.info(TT("Run a prediction first to enable download."))

# ----------------------------
# Footer
st.caption(TT("Developed with ❤️ for Mumbai's vibrant street vendors."))
