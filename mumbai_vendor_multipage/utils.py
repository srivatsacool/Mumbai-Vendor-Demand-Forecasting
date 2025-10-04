
import os
import joblib
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import shap
import warnings
warnings.filterwarnings("ignore")

# ----------------------
# Model & artifact loader
# ----------------------
def load_model_artifacts():
    """Load model and related artifacts from disk. Expects files in working directory:"""
    model = None
    model_type = None
    if os.path.exists("xgb_model.pkl"):
        model = joblib.load("xgb_model.pkl")
        model_type = "XGBoost"
    elif os.path.exists("rf_model.pkl"):
        model = joblib.load("rf_model.pkl")
        model_type = "Random Forest"
    else:
        raise FileNotFoundError("No rf_model.pkl or xgb_model.pkl found in working directory.")

    label_encoders = {}
    if os.path.exists("label_encoders.pkl"):
        label_encoders = joblib.load("label_encoders.pkl")

    if not os.path.exists("feature_columns.pkl"):
        # fallback: try to infer numeric columns from sample CSV
        feature_columns = None
    else:
        feature_columns = joblib.load("feature_columns.pkl")

    metadata = {}
    if os.path.exists("model_metadata.json"):
        with open("model_metadata.json", "r") as f:
            metadata = json.load(f)

    return model, label_encoders, feature_columns, metadata, model_type

# ----------------------
# Feature builder
# ----------------------
def create_features_for_prediction(selected_vendor, selected_date, selected_hour,
                                   temperature, rainfall, humidity, traffic_density,
                                   event_nearby, competitor_count, selected_festival,
                                   label_encoders=None, feature_columns=None):
    vendor_details = {
        'vendor_01': {'location_type': 'business_district', 'cuisine_type': 'main_course', 'avg_price': 85.0, 'menu_diversity': 12},
        'vendor_02': {'location_type': 'near_college', 'cuisine_type': 'chai_snacks', 'avg_price': 25.0, 'menu_diversity': 8},
        'vendor_03': {'location_type': 'market', 'cuisine_type': 'chaat', 'avg_price': 45.0, 'menu_diversity': 15},
        'vendor_04': {'location_type': 'metro_station', 'cuisine_type': 'beverages', 'avg_price': 35.0, 'menu_diversity': 10},
        'vendor_05': {'location_type': 'office', 'cuisine_type': 'dessert', 'avg_price': 65.0, 'menu_diversity': 7}
    }
    vendor_info = vendor_details[selected_vendor]
    dt = datetime.combine(selected_date, datetime.min.time()) + timedelta(hours=selected_hour)
    day_of_week = dt.weekday()
    is_weekend = 1 if day_of_week >= 5 else 0
    is_festival = 1 if selected_festival != 'None' else 0
    is_holiday = 1 if selected_festival in ['Independence Day'] else 0
    is_rainy = 1 if rainfall > 0.1 else 0
    wind_speed = 10.0 + (5.0 if rainfall > 0 else 0.0)
    is_peak_hour = 1 if selected_hour in [7,8,12,13,17,18,19,20] else 0
    is_evening = 1 if 17 <= selected_hour <= 21 else 0
    is_morning = 1 if 6 <= selected_hour <= 10 else 0

    base_demand = 15 if selected_vendor == 'vendor_01' else 20
    lag_1h = max(1, int(base_demand * (0.8 + 0.4 * np.random.random())))
    lag_24h = max(1, int(base_demand * (0.9 + 0.2 * np.random.random())))
    rolling_avg = (lag_1h + lag_24h) / 2

    # safely encode categorical features if encoders provided
    def enc(key, value, fallback=0):
        if label_encoders and key in label_encoders:
            try:
                return label_encoders[key].transform([value])[0]
            except Exception:
                return fallback
        else:
            return fallback

    features = {
        'vendor_id_encoded': enc('vendor_id', selected_vendor),
        'location_type_encoded': enc('location_type', vendor_info['location_type']),
        'cuisine_type_encoded': enc('cuisine_type', vendor_info['cuisine_type']),
        'avg_price': vendor_info['avg_price'],
        'menu_diversity': vendor_info['menu_diversity'],
        'hour_of_day': selected_hour,
        'day_of_week': day_of_week,
        'is_weekend': is_weekend,
        'is_holiday': is_holiday,
        'is_festival': is_festival,
        'temperature_c': temperature,
        'rainfall_mm': rainfall,
        'humidity_pct': humidity,
        'wind_speed_kmh': wind_speed,
        'event_nearby': 1 if event_nearby else 0,
        'traffic_density_encoded': enc('traffic_density', traffic_density, fallback=(0 if traffic_density=='low' else (1 if traffic_density=='medium' else 2))),
        'competitor_count': competitor_count,
        'lag_1h_units': lag_1h,
        'lag_24h_units': lag_24h,
        'rolling_avg_24h': rolling_avg,
        'is_peak_hour': is_peak_hour,
        'is_evening': is_evening,
        'is_morning': is_morning
    }

    df = pd.DataFrame([features])
    if feature_columns:
        # try to reorder to the model's expected order
        try:
            df = df[feature_columns]
        except Exception:
            pass
    return df, vendor_info

# ----------------------
# SHAP explainer loader (cache-safe signature)
# ----------------------
def load_or_create_shap_explainer(_model, hist_df=None, feature_columns=None, explainer_path="shap_explainer.pkl"):
    """Create or load a shap.Explainer. Leading underscore avoids caching-hash issues in streamlit.
    Returns explainer instance and boolean indicating whether it was loaded from disk."""
    if os.path.exists(explainer_path):
        try:
            explainer = joblib.load(explainer_path)
            return explainer, True
        except Exception:
            pass
    # create using background sample if available
    background = None
    if hist_df is not None and feature_columns is not None:
        try:
            background = hist_df.sample(n=min(500, len(hist_df))).reset_index(drop=True)
            background = background[feature_columns].fillna(0)
        except Exception:
            background = None
    try:
        explainer = shap.Explainer(_model, background) if background is not None else shap.Explainer(_model)
    except Exception:
        explainer = shap.Explainer(_model)
    try:
        joblib.dump(explainer, explainer_path)
    except Exception:
        pass
    return explainer, False

