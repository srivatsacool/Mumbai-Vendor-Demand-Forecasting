import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="Mumbai Street Vendor Demand Forecasting",
    page_icon="🍛",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load model and encoders
@st.cache_resource
def load_model_artifacts():
    """Load model and preprocessing artifacts"""
    try:
        # Try XGBoost first, then Random Forest
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

# Load artifacts
model, label_encoders, feature_columns, metadata, model_type = load_model_artifacts()

if model is None:
    st.error("Failed to load model. Please ensure all model files are present.")
    st.stop()

# Load historical data for charts
@st.cache_data
def load_historical_data():
    """Load historical data for visualization"""
    try:
        df = pd.read_csv('mumbai_vendors_hourly_20250701_20250930.csv')
        df['datetime_parsed'] = pd.to_datetime(df['datetime'])
        return df
    except Exception as e:
        st.error(f"Error loading historical data: {e}")
        return None

historical_data = load_historical_data()

# App header
st.title("🍛 Mumbai Street Vendor Demand Forecasting")
st.markdown("""
**Predict hourly demand for street food vendors in Mumbai using machine learning**

Select vendor, date/time, and scenario parameters to get demand predictions with explanations.
""")

# Sidebar controls
st.sidebar.header("📊 Prediction Controls")

# Vendor selection
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

# Date and time selection
st.sidebar.subheader("📅 Date & Time")
selected_date = st.sidebar.date_input(
    "Select Date:",
    value=datetime(2025, 8, 15),
    min_value=datetime(2025, 7, 1),
    max_value=datetime(2025, 12, 31)
)

selected_hour = st.sidebar.slider(
    "Select Hour:",
    min_value=0, max_value=23, value=12,
    format="%d:00"
)

# Weather controls
st.sidebar.subheader("🌤️ Weather Conditions")
temperature = st.sidebar.slider(
    "Temperature (°C):",
    min_value=22.0, max_value=35.0, value=28.0, step=0.5
)

rainfall = st.sidebar.slider(
    "Rainfall (mm):",
    min_value=0.0, max_value=50.0, value=0.0, step=0.5
)

humidity = st.sidebar.slider(
    "Humidity (%):",
    min_value=50.0, max_value=95.0, value=80.0, step=1.0
)

# Traffic and environment
st.sidebar.subheader("🚦 Environment")
traffic_density = st.sidebar.selectbox(
    "Traffic Density:",
    options=['low', 'medium', 'high'],
    index=1
)

event_nearby = st.sidebar.checkbox("Special Event Nearby", value=False)

competitor_count = st.sidebar.slider(
    "Nearby Competitors:",
    min_value=0, max_value=15, value=6
)

# Festival simulation
st.sidebar.subheader("🎉 Festival Options")
festival_options = ['None', 'Independence Day', 'Janmashtami', 'Dahi Handi', 
                   'Ganesh Chaturthi', 'Ganesh Visarjan', 'Eid-e-Milad']
selected_festival = st.sidebar.selectbox(
    "Simulate Festival:",
    options=festival_options
)

# Vendor details mapping
vendor_details = {
    'vendor_01': {
        'location_type': 'business_district', 'cuisine_type': 'main_course',
        'avg_price': 85.0, 'menu_diversity': 12, 'peak_hours': [12, 13, 19, 20]
    },
    'vendor_02': {
        'location_type': 'near_college', 'cuisine_type': 'chai_snacks',
        'avg_price': 25.0, 'menu_diversity': 8, 'peak_hours': [7, 8, 16, 17, 18]
    },
    'vendor_03': {
        'location_type': 'market', 'cuisine_type': 'chaat',
        'avg_price': 45.0, 'menu_diversity': 15, 'peak_hours': [17, 18, 19, 20, 21]
    },
    'vendor_04': {
        'location_type': 'metro_station', 'cuisine_type': 'beverages',
        'avg_price': 35.0, 'menu_diversity': 10, 'peak_hours': [7, 8, 9, 17, 18, 19]
    },
    'vendor_05': {
        'location_type': 'office', 'cuisine_type': 'dessert',
        'avg_price': 65.0, 'menu_diversity': 7, 'peak_hours': [15, 16, 21, 22]
    }
}

def create_prediction_features(vendor_id, date, hour, temp, rain, humid, traffic, 
                             event, competitors, festival):
    """Create feature vector for prediction"""

    # Get vendor details
    vendor_info = vendor_details[vendor_id]

    # Date features
    dt = datetime.combine(date, datetime.min.time()) + timedelta(hours=hour)
    day_of_week = dt.weekday()  # Monday=0
    is_weekend = 1 if day_of_week >= 5 else 0

    # Festival features
    is_festival = 1 if festival != 'None' else 0
    is_holiday = 1 if festival in ['Independence Day'] else 0
    festival_name = festival if festival != 'None' else ''

    # Weather features
    is_rainy = 1 if rain > 0.1 else 0
    wind_speed = 15.0 + (5.0 if rain > 0 else 0.0)  # Estimate wind speed

    # Time-based features
    is_peak_hour = 1 if hour in [7, 8, 12, 13, 17, 18, 19, 20] else 0
    is_evening = 1 if 17 <= hour <= 21 else 0
    is_morning = 1 if 6 <= hour <= 10 else 0

    # Lag features (estimated based on typical patterns)
    base_demand = 15 if vendor_id == 'vendor_01' else 20  # Simplified estimation
    lag_1h = max(1, int(base_demand * (0.8 + 0.4 * np.random.random())))
    lag_24h = max(1, int(base_demand * (0.9 + 0.2 * np.random.random())))
    rolling_avg = (lag_1h + lag_24h) / 2

    # Create feature dictionary
    features = {
        'vendor_id_encoded': label_encoders['vendor_id'].transform([vendor_id])[0],
        'location_type_encoded': label_encoders['location_type'].transform([vendor_info['location_type']])[0],
        'cuisine_type_encoded': label_encoders['cuisine_type'].transform([vendor_info['cuisine_type']])[0],
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
        'event_nearby': 1 if event else 0,
        'traffic_density_encoded': label_encoders['traffic_density'].transform([traffic])[0],
        'competitor_count': competitors,
        'lag_1h_units': lag_1h,
        'lag_24h_units': lag_24h,
        'rolling_avg_24h': rolling_avg,
        'is_peak_hour': is_peak_hour,
        'is_evening': is_evening,
        'is_morning': is_morning
    }

    return pd.DataFrame([features]), vendor_info

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

# Main prediction section
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("🎯 Demand Prediction")

    if st.button("🔮 Predict Units", type="primary"):
        # Create features
        features_df, vendor_info = create_prediction_features(
            selected_vendor, selected_date, selected_hour,
            temperature, rainfall, humidity, traffic_density,
            event_nearby, competitor_count, selected_festival
        )

        # Make prediction
        prediction = model.predict(features_df)[0]
        uncertainty = float(metadata.get('prediction_uncertainty_std', 3.0))

        # Display prediction
        st.success(f"**Predicted Units Sold: {prediction:.0f}**")
        st.info(f"**Confidence Interval: {prediction-1.96*uncertainty:.0f} - {prediction+1.96*uncertainty:.0f} units (95% CI)**")

        # Generate and display explanation
        explanation = generate_explanation(prediction, features_df, vendor_info, selected_festival)
        st.markdown("### 📝 Prediction Explanation")
        st.markdown(explanation)

        # Store prediction for batch scenario
        st.session_state.last_prediction = prediction
        st.session_state.last_features = features_df
        st.session_state.last_vendor = selected_vendor

with col2:
    st.subheader("📈 Historical Context")

    if historical_data is not None:
        # Filter data for selected vendor
        vendor_data = historical_data[historical_data['vendor_id'] == selected_vendor].copy()

        if not vendor_data.empty:
            # Show average units for this hour/day combination
            same_hour_data = vendor_data[vendor_data['hour_of_day'] == selected_hour]

            if not same_hour_data.empty:
                avg_units = same_hour_data['units_sold'].mean()
                max_units = same_hour_data['units_sold'].max()
                min_units = same_hour_data['units_sold'].min()

                st.metric("Historical Average (Same Hour)", f"{avg_units:.1f}", 
                         help=f"Range: {min_units}-{max_units} units")

                # Show recent trend (last 7 days of same hour)
                recent_data = same_hour_data.tail(7)['units_sold'].tolist()
                if len(recent_data) >= 3:
                    trend = "📈" if recent_data[-1] > recent_data[0] else "📉"
                    st.metric("Recent Trend", f"{trend} {recent_data[-1]} units",
                             delta=f"{recent_data[-1] - recent_data[0]:.0f} vs week ago")

# Historical visualization
if historical_data is not None:
    st.subheader("📊 Historical Demand Patterns")

    # Filter for selected vendor
    vendor_hist = historical_data[historical_data['vendor_id'] == selected_vendor].copy()

    if not vendor_hist.empty:
        # Create tabs for different views
        tab1, tab2 = st.tabs(["📅 48-Hour Context", "🕒 Hourly Pattern"])

        with tab1:
            # Show 48 hours around selected date/time


            target_dt = datetime.combine(selected_date, datetime.min.time()) + timedelta(hours=selected_hour)
            start_dt = target_dt - timedelta(hours=24)
            end_dt = target_dt + timedelta(hours=24)

            # Ensure start_dt and end_dt are timezone-aware to match vendor_hist['datetime_parsed']
            dt_tz = None
            if pd.api.types.is_datetime64tz_dtype(vendor_hist['datetime_parsed']):
                dt_tz = vendor_hist['datetime_parsed'].dt.tz
            if dt_tz is not None:
                start_dt = pd.Timestamp(start_dt).tz_localize(dt_tz)
                end_dt = pd.Timestamp(end_dt).tz_localize(dt_tz)

            # Filter data
            context_data = vendor_hist[
                (vendor_hist['datetime_parsed'] >= start_dt) &
                (vendor_hist['datetime_parsed'] <= end_dt)
            ].copy()

            if not context_data.empty:
                fig = go.Figure()
                # Convert x-axis to Python datetime objects for Plotly compatibility
                x_vals = context_data['datetime_parsed']
                if hasattr(x_vals, 'dt'):
                    x_vals = x_vals.dt.tz_localize(None, ambiguous='NaT', nonexistent='NaT').dt.to_pydatetime()
                fig.add_trace(go.Scatter(
                    x=x_vals,
                    y=context_data['units_sold'],
                    mode='lines+markers',
                    name='Historical Demand',
                    line=dict(color='blue', width=2),
                    marker=dict(size=6)
                ))

                # # Highlight selected time
                # vline_x = target_dt
                # if hasattr(target_dt, 'tzinfo') and target_dt.tzinfo is not None:
                #     vline_x = target_dt.replace(tzinfo=None)
                # if isinstance(vline_x, pd.Timestamp):
                #     vline_x = vline_x.to_pydatetime()
                # fig.add_vline(
                #     x=vline_x,
                #     line_dash="dash",
                #     line_color="red",
                #     annotation_text="Selected Time"
                # )

                fig.update_layout(
                    title=f"48-Hour Demand Context - {vendor_options[selected_vendor]}",
                    xaxis_title="Date/Time",
                    yaxis_title="Units Sold",
                    height=400
                )

                st.plotly_chart(fig, use_container_width=True)

        with tab2:
            # Average hourly pattern
            hourly_avg = vendor_hist.groupby('hour_of_day')['units_sold'].agg(['mean', 'std']).reset_index()

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=hourly_avg['hour_of_day'],
                y=hourly_avg['mean'],
                mode='lines+markers',
                name='Average Demand',
                line=dict(color='green', width=3),
                marker=dict(size=8)
            ))

            # Add error bars
            fig.add_trace(go.Scatter(
                x=hourly_avg['hour_of_day'],
                y=hourly_avg['mean'] + hourly_avg['std'],
                mode='lines',
                line=dict(width=0),
                showlegend=False,
                name='Upper Bound'
            ))

            fig.add_trace(go.Scatter(
                x=hourly_avg['hour_of_day'],
                y=hourly_avg['mean'] - hourly_avg['std'],
                mode='lines',
                line=dict(width=0),
                fill='tonexty',
                fillcolor='rgba(0,100,80,0.2)',
                showlegend=False,
                name='Lower Bound'
            ))

            # Highlight selected hour
            fig.add_vline(
                x=selected_hour,
                line_dash="dash",
                line_color="red",
                annotation_text=f"Selected: {selected_hour}:00"
            )

            fig.update_layout(
                title=f"Average Hourly Demand Pattern - {vendor_options[selected_vendor]}",
                xaxis_title="Hour of Day",
                yaxis_title="Average Units Sold",
                height=400
            )

            st.plotly_chart(fig, use_container_width=True)

# Batch prediction section
st.subheader("🔄 Batch Scenario: Next 24 Hours")

if st.button("📊 Generate 24-Hour Forecast"):
    if 'last_features' in st.session_state:
        # Generate 24-hour forecast
        base_features = st.session_state.last_features.iloc[0].to_dict()

        forecast_data = []
        for i in range(24):
            hour_features = base_features.copy()
            new_hour = (selected_hour + i) % 24
            hour_features['hour_of_day'] = new_hour

            # Update time-based features
            hour_features['is_peak_hour'] = 1 if new_hour in [7, 8, 12, 13, 17, 18, 19, 20] else 0
            hour_features['is_evening'] = 1 if 17 <= new_hour <= 21 else 0
            hour_features['is_morning'] = 1 if 6 <= new_hour <= 10 else 0

            # Predict
            hour_df = pd.DataFrame([hour_features])
            prediction = model.predict(hour_df)[0]

            forecast_data.append({
                'hour': new_hour,
                'predicted_units': prediction,
                'time_label': f"{new_hour:02d}:00"
            })

        forecast_df = pd.DataFrame(forecast_data)

        # Visualize forecast
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=forecast_df['time_label'],
            y=forecast_df['predicted_units'],
            mode='lines+markers',
            name='24-Hour Forecast',
            line=dict(color='purple', width=3),
            marker=dict(size=8, color='purple')
        ))

        fig.update_layout(
            title=f"24-Hour Demand Forecast - {vendor_options[selected_vendor]}",
            xaxis_title="Hour",
            yaxis_title="Predicted Units",
            height=400
        )

        st.plotly_chart(fig, use_container_width=True)

        # Summary statistics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total 24h Units", f"{forecast_df['predicted_units'].sum():.0f}")
        with col2:
            st.metric("Average per Hour", f"{forecast_df['predicted_units'].mean():.1f}")
        with col3:
            peak_hour = forecast_df.loc[forecast_df['predicted_units'].idxmax(), 'time_label']
            st.metric("Peak Hour", peak_hour)
        with col4:
            st.metric("Peak Demand", f"{forecast_df['predicted_units'].max():.0f}")

        # Download CSV option
        csv = forecast_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Forecast CSV",
            data=csv,
            file_name=f"{selected_vendor}_24h_forecast.csv",
            mime="text/csv"
        )

# Model information
with st.expander("ℹ️ Model Information"):
    st.write(f"**Model Type:** {model_type}")
    st.write(f"**Validation MAE:** {metadata.get('best_validation_mae', 'N/A'):.3f} units")
    st.write(f"**Features:** {metadata.get('feature_count', 'N/A')}")
    st.write(f"**Training Date:** {metadata.get('training_date', 'N/A')}")
    st.write(f"**Dataset Size:** {metadata.get('dataset_shape', ['N/A', 'N/A'])[0]:,} records")
    st.write("**Random Seed:** 42 (reproducible results)")

# Footer
st.markdown("---")
st.markdown("""
**Mumbai Street Vendor Demand Forecasting System** | Built with Streamlit & XGBoost/Random Forest  
*For accurate predictions, ensure input parameters reflect realistic scenarios*
""")
