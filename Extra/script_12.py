# Create model artifacts using available libraries (Random Forest)
print("Creating model artifacts using Random Forest...")

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, TimeSeriesSplit, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib
import json
import warnings
warnings.filterwarnings('ignore')

# Set random seed
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# Load the dataset
df = pd.read_csv('mumbai_vendors_hourly_20250701_20250930.csv')
print(f"✅ Dataset loaded: {df.shape}")

# Prepare data
df['datetime_parsed'] = pd.to_datetime(df['datetime'])
model_df = df.copy()

# Handle missing values
print(f"Before imputation - Missing values: {model_df.isnull().sum().sum()}")

# Forward fill lag features
model_df['lag_1h_units'] = model_df.groupby('vendor_id')['lag_1h_units'].fillna(method='ffill')
model_df['lag_24h_units'] = model_df.groupby('vendor_id')['lag_24h_units'].fillna(method='ffill') 
model_df['rolling_avg_24h'] = model_df.groupby('vendor_id')['rolling_avg_24h'].fillna(method='ffill')

# Fill weather missing values
weather_cols = ['temperature_c', 'rainfall_mm', 'humidity_pct', 'wind_speed_kmh']
for col in weather_cols:
    if col in model_df.columns:
        model_df[col] = model_df[col].fillna(model_df[col].median())

# Fill remaining lag features with 0
lag_cols = ['lag_1h_units', 'lag_24h_units', 'rolling_avg_24h'] 
for col in lag_cols:
    if col in model_df.columns:
        model_df[col] = model_df[col].fillna(0)

print(f"✅ After imputation - Missing values: {model_df.isnull().sum().sum()}")

# Encode categorical variables
label_encoders = {}
categorical_cols = ['vendor_id', 'location_type', 'cuisine_type', 'traffic_density']

for col in categorical_cols:
    le = LabelEncoder()
    model_df[col + '_encoded'] = le.fit_transform(model_df[col])
    label_encoders[col] = le
    print(f"✅ Encoded {col}: {len(le.classes_)} categories")

# Create additional features
model_df['is_peak_hour'] = model_df['hour_of_day'].apply(
    lambda x: 1 if x in [7, 8, 12, 13, 17, 18, 19, 20] else 0
)
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

X = model_df[feature_cols].copy()
y = model_df['units_sold'].copy()

# Time-based split
split_date = model_df['datetime_parsed'].quantile(0.8)
train_mask = model_df['datetime_parsed'] < split_date
X_train, X_val = X[train_mask], X[~train_mask]
y_train, y_val = y[train_mask], y[~train_mask]

print(f"✅ Train/Val split: {len(X_train)}/{len(X_val)}")

# Train Random Forest with hyperparameter tuning
print("🤖 Training Random Forest...")

param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 15],
    'min_samples_split': [5, 10]
}

tscv = TimeSeriesSplit(n_splits=3)
rf_grid = GridSearchCV(
    RandomForestRegressor(random_state=RANDOM_SEED),
    param_grid,
    cv=tscv,
    scoring='neg_mean_absolute_error',
    n_jobs=-1
)

rf_grid.fit(X_train, y_train)
best_rf_model = rf_grid.best_estimator_

# Evaluate model
y_pred_val = best_rf_model.predict(X_val)
mae = mean_absolute_error(y_val, y_pred_val)
rmse = np.sqrt(mean_squared_error(y_val, y_pred_val))
mape = np.mean(np.abs((y_val - y_pred_val) / (y_val + 1e-8))) * 100

print(f"✅ Random Forest trained with MAE: {mae:.3f}, RMSE: {rmse:.3f}, MAPE: {mape:.2f}%")
print(f"Best parameters: {rf_grid.best_params_}")

# Calculate uncertainty
val_residuals = y_val - y_pred_val
residual_std = val_residuals.std()

# Save model artifacts
joblib.dump(best_rf_model, 'rf_model.pkl')
print("✅ Model saved: rf_model.pkl")

joblib.dump(label_encoders, 'label_encoders.pkl') 
print("✅ Label encoders saved")

joblib.dump(feature_cols, 'feature_columns.pkl')
print("✅ Feature columns saved")

# Save metadata
metadata = {
    'model_type': 'RandomForestRegressor',
    'best_validation_mae': float(mae),
    'validation_rmse': float(rmse),
    'validation_mape': float(mape),
    'feature_count': len(feature_cols),
    'training_date': '2025-10-01',
    'random_seed': RANDOM_SEED,
    'dataset_shape': list(df.shape),
    'prediction_uncertainty_std': float(residual_std),
    'best_params': rf_grid.best_params_
}

with open('model_metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)
print("✅ Metadata saved")

# Show feature importance
feature_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': best_rf_model.feature_importances_
}).sort_values('importance', ascending=False)

print(f"\n🎯 MODEL SUMMARY:")
print(f"Model: Random Forest")  
print(f"Validation MAE: {mae:.3f} units")
print(f"Validation RMSE: {rmse:.3f} units")
print(f"Validation MAPE: {mape:.2f}%")
print(f"Features: {len(feature_cols)}")

print(f"\nTop 10 Most Important Features:")
print(feature_importance.head(10)[['feature', 'importance']].to_string(index=False))

print(f"\n🎉 All model artifacts created successfully!")
print("Ready to run: streamlit run app_streamlit.py")