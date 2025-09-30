# Create a simple trained model without grid search to avoid timeout
print("Creating model artifacts (simplified version)...")

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pickle
import json

# Set random seed
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# Load data
df = pd.read_csv('mumbai_vendors_hourly_20250701_20250930.csv')
df['datetime_parsed'] = pd.to_datetime(df['datetime'])

# Quick preprocessing
model_df = df.copy()

# Handle missing values simply
model_df = model_df.fillna(method='ffill').fillna(0)

# Encode categoricals
label_encoders = {}
for col in ['vendor_id', 'location_type', 'cuisine_type', 'traffic_density']:
    le = LabelEncoder()
    model_df[col + '_encoded'] = le.fit_transform(model_df[col])
    label_encoders[col] = le

# Add time features
model_df['is_peak_hour'] = model_df['hour_of_day'].apply(lambda x: 1 if x in [7,8,12,13,17,18,19,20] else 0)
model_df['is_evening'] = model_df['hour_of_day'].apply(lambda x: 1 if 17 <= x <= 21 else 0)
model_df['is_morning'] = model_df['hour_of_day'].apply(lambda x: 1 if 6 <= x <= 10 else 0)

# Select features
feature_cols = [
    'vendor_id_encoded', 'location_type_encoded', 'cuisine_type_encoded',
    'avg_price', 'menu_diversity', 'hour_of_day', 'day_of_week', 'is_weekend',
    'is_holiday', 'is_festival', 'temperature_c', 'rainfall_mm', 'humidity_pct',
    'wind_speed_kmh', 'event_nearby', 'traffic_density_encoded', 'competitor_count',
    'lag_1h_units', 'lag_24h_units', 'rolling_avg_24h', 'is_peak_hour', 'is_evening', 'is_morning'
]

X = model_df[feature_cols]
y = model_df['units_sold']

# Simple train/val split
split_idx = int(0.8 * len(X))
X_train, X_val = X[:split_idx], X[split_idx:]
y_train, y_val = y[:split_idx], y[split_idx:]

# Train simple Random Forest
rf_model = RandomForestRegressor(
    n_estimators=100, 
    max_depth=12,
    random_state=RANDOM_SEED
)

rf_model.fit(X_train, y_train)
y_pred = rf_model.predict(X_val)

# Metrics
mae = mean_absolute_error(y_val, y_pred)
rmse = np.sqrt(mean_squared_error(y_val, y_pred))
residual_std = np.std(y_val - y_pred)

# Save model
with open('rf_model.pkl', 'wb') as f:
    pickle.dump(rf_model, f)

# Save encoders  
with open('label_encoders.pkl', 'wb') as f:
    pickle.dump(label_encoders, f)

# Save feature columns
with open('feature_columns.pkl', 'wb') as f:
    pickle.dump(feature_cols, f)

# Save metadata
metadata = {
    'model_type': 'RandomForestRegressor',
    'best_validation_mae': float(mae),
    'feature_count': len(feature_cols),
    'training_date': '2025-10-01', 
    'random_seed': RANDOM_SEED,
    'dataset_shape': list(df.shape),
    'prediction_uncertainty_std': float(residual_std)
}

with open('model_metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)

print(f"✅ Model artifacts created!")
print(f"Model: Random Forest (MAE: {mae:.3f})")
print(f"Files created: rf_model.pkl, label_encoders.pkl, feature_columns.pkl, model_metadata.json")

# Show feature importance
importance_df = pd.DataFrame({
    'feature': feature_cols,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

print(f"\nTop 5 Important Features:")
print(importance_df.head(5).to_string(index=False))