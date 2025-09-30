# Run the complete model training pipeline to generate all artifacts
print("Starting complete model training pipeline...")

# Import all required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, TimeSeriesSplit, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error
import xgboost as xgb
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

# Convert datetime and prepare data
df['datetime_parsed'] = pd.to_datetime(df['datetime'])

# Handle missing values
model_df = df.copy()
print(f"Before imputation - Missing values: {model_df.isnull().sum().sum()}")

# Forward fill lag features
model_df['lag_1h_units'] = model_df.groupby('vendor_id')['lag_1h_units'].fillna(method='ffill')
model_df['lag_24h_units'] = model_df.groupby('vendor_id')['lag_24h_units'].fillna(method='ffill')
model_df['rolling_avg_24h'] = model_df.groupby('vendor_id')['rolling_avg_24h'].fillna(method='ffill')

# Fill weather missing values with median
weather_cols = ['temperature_c', 'rainfall_mm', 'humidity_pct', 'wind_speed_kmh']
for col in weather_cols:
    if col in model_df.columns:
        model_df[col] = model_df[col].fillna(model_df[col].median())

# Fill remaining lag features with 0
lag_cols = ['lag_1h_units', 'lag_24h_units', 'rolling_avg_24h']
for col in lag_cols:
    if col in model_df.columns:
        model_df[col] = model_df[col].fillna(0)

print(f"After imputation - Missing values: {model_df.isnull().sum().sum()}")

# Encode categorical variables
label_encoders = {}
categorical_cols = ['vendor_id', 'location_type', 'cuisine_type', 'traffic_density']

for col in categorical_cols:
    le = LabelEncoder()
    model_df[col + '_encoded'] = le.fit_transform(model_df[col])
    label_encoders[col] = le
    print(f"✅ Encoded {col}: {len(le.classes_)} categories")

# Create additional time-based features
model_df['is_peak_hour'] = model_df['hour_of_day'].apply(
    lambda x: 1 if x in [7, 8, 12, 13, 17, 18, 19, 20] else 0
)
model_df['is_evening'] = model_df['hour_of_day'].apply(lambda x: 1 if 17 <= x <= 21 else 0)
model_df['is_morning'] = model_df['hour_of_day'].apply(lambda x: 1 if 6 <= x <= 10 else 0)

print("✅ Additional time features created")

# Select features for modeling
feature_cols = [
    'vendor_id_encoded', 'location_type_encoded', 'cuisine_type_encoded',
    'avg_price', 'menu_diversity', 'hour_of_day', 'day_of_week', 'is_weekend',
    'is_holiday', 'is_festival', 'temperature_c', 'rainfall_mm', 'humidity_pct',
    'wind_speed_kmh', 'event_nearby', 'traffic_density_encoded', 'competitor_count',
    'lag_1h_units', 'lag_24h_units', 'rolling_avg_24h', 'is_peak_hour', 'is_evening', 'is_morning'
]

target_col = 'units_sold'
X = model_df[feature_cols].copy()
y = model_df[target_col].copy()

print(f"✅ Feature matrix prepared: {X.shape}")

# Time-based split
split_date = model_df['datetime_parsed'].quantile(0.8)
train_mask = model_df['datetime_parsed'] < split_date
val_mask = model_df['datetime_parsed'] >= split_date

X_train = X[train_mask]
X_val = X[val_mask]
y_train = y[train_mask]
y_val = y[val_mask]

print(f"✅ Train/Val split: {X_train.shape[0]}/{X_val.shape[0]} ({100*len(X_train)/len(X):.1f}%/{100*len(X_val)/len(X):.1f}%)")

# Define evaluation functions
def calculate_mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100

def evaluate_model(y_true, y_pred, model_name):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = calculate_mape(y_true, y_pred)
    
    return {
        'Model': model_name,
        'MAE': mae,
        'RMSE': rmse,
        'MAPE': mape
    }

print("🤖 Starting model training...")

# Train XGBoost model
xgb_model = xgb.XGBRegressor(
    random_state=RANDOM_SEED,
    n_estimators=300,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=1.0
)

xgb_model.fit(X_train, y_train)
xgb_pred_val = xgb_model.predict(X_val)
xgb_results = evaluate_model(y_val, xgb_pred_val, 'XGBoost')
print(f"✅ XGBoost trained - Val MAE: {xgb_results['MAE']:.3f}")

# Train Random Forest model
rf_model = RandomForestRegressor(
    random_state=RANDOM_SEED,
    n_estimators=200,
    max_depth=12,
    min_samples_split=5,
    min_samples_leaf=2,
    max_features='sqrt',
    bootstrap=True
)

rf_model.fit(X_train, y_train)
rf_pred_val = rf_model.predict(X_val)
rf_results = evaluate_model(y_val, rf_pred_val, 'RandomForest')
print(f"✅ Random Forest trained - Val MAE: {rf_results['MAE']:.3f}")

# Simple hyperparameter tuning for XGBoost
tscv = TimeSeriesSplit(n_splits=3)
xgb_param_grid = {
    'n_estimators': [200, 300],
    'max_depth': [4, 6],
    'learning_rate': [0.05, 0.1]
}

print("🔧 Hyperparameter tuning...")
xgb_grid = GridSearchCV(
    xgb.XGBRegressor(random_state=RANDOM_SEED),
    xgb_param_grid,
    cv=tscv,
    scoring='neg_mean_absolute_error',
    n_jobs=-1
)

xgb_grid.fit(X_train, y_train)
best_xgb_model = xgb_grid.best_estimator_
best_xgb_pred_val = best_xgb_model.predict(X_val)
best_xgb_results = evaluate_model(y_val, best_xgb_pred_val, 'XGBoost_Tuned')

print(f"✅ Best XGBoost params: {xgb_grid.best_params_}")
print(f"✅ Tuned XGBoost - Val MAE: {best_xgb_results['MAE']:.3f}")

# Compare models and select best
results_df = pd.DataFrame([xgb_results, rf_results, best_xgb_results])
print("\n" + "="*50)
print("MODEL EVALUATION RESULTS")
print("="*50)
print(results_df.round(3))

# Select best model
best_mae = results_df['MAE'].min()
best_model_name = results_df.loc[results_df['MAE'].idxmin(), 'Model']

if 'Tuned' in best_model_name:
    final_model = best_xgb_model
elif 'XGBoost' in best_model_name:
    final_model = xgb_model
else:
    final_model = rf_model

print(f"\n🏆 Best model: {best_model_name} (MAE: {best_mae:.3f})")

# Calculate prediction uncertainty
val_residuals = y_val - (best_xgb_pred_val if 'XGB' in str(type(final_model)) else rf_pred_val)
residual_std = val_residuals.std()

# Save model artifacts
model_filename = 'xgb_model.pkl'
joblib.dump(final_model, model_filename)
print(f"✅ Model saved: {model_filename}")

joblib.dump(label_encoders, 'label_encoders.pkl')
print("✅ Label encoders saved")

joblib.dump(feature_cols, 'feature_columns.pkl')
print("✅ Feature columns saved")

# Save metadata
metadata = {
    'model_type': type(final_model).__name__,
    'best_validation_mae': float(best_mae),
    'feature_count': len(feature_cols),
    'training_date': '2025-10-01',
    'random_seed': RANDOM_SEED,
    'dataset_shape': list(df.shape),
    'prediction_uncertainty_std': float(residual_std),
    'best_params': xgb_grid.best_params_ if 'XGB' in str(type(final_model)) else None
}

with open('model_metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)
print("✅ Metadata saved")

print(f"\n🎉 Model training completed successfully!")
print(f"Final model: {metadata['model_type']} with MAE: {metadata['best_validation_mae']:.3f}")
print(f"All artifacts saved and ready for Streamlit app!")

# Show feature importance
if hasattr(final_model, 'feature_importances_'):
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': final_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"\nTop 10 Most Important Features:")
    print(feature_importance.head(10).to_string(index=False))