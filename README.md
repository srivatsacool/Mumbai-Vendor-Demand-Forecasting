# Mumbai Vendor Demand Forecasting 🍛

Predict hourly demand for street food vendors in Mumbai using machine learning. This project provides an end-to-end solution including data preprocessing, model training, evaluation, and a Streamlit web app for interactive demand prediction.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Data](#data)
- [Technical Stack](#technical-stack)
- [Installation](#installation)
- [Usage](#usage)
- [Model Training](#model-training)
- [Visualization](#visualization)
- [Screenshots](#screenshots)
- [Contributing](#contributing)
- [License](#license)

---

## Project Overview

Mumbai street vendors face highly variable demand due to factors like weather, location, time of day, and festivals. This project forecasts vendor demand on an hourly basis to help vendors optimize inventory, reduce wastage, and maximize sales.

The project includes:

- Data preprocessing and feature engineering
- Machine learning model training
- Model evaluation with metrics
- Interactive Streamlit web app for live predictions

---

## Features

- **Demand Forecasting:** Predict hourly sales for vendors
- **Feature Automation:** Weather, traffic, events, and holidays are automatically considered
- **Visualization:** Graphs and charts for historical and predicted demand
- **User-Friendly UI:** Streamlit web app for easy interaction

---

## Data

The dataset `mumbai_vendors_hourly_20250701_20250930.csv` contains historical sales data with the following features:

| Feature          | Description                               |
| ---------------- | ----------------------------------------- |
| vendor_id        | Unique identifier for each vendor         |
| location_type    | Location type (street, market, etc.)      |
| cuisine_type     | Type of cuisine sold                      |
| avg_price        | Average price per unit                    |
| menu_diversity   | Number of items offered                   |
| hour_of_day      | Hourly timestamp of sale                  |
| is_weekend       | Binary indicator for weekends             |
| is_holiday       | Binary indicator for public holidays      |
| is_festival      | Binary indicator for festivals            |
| temperature_c    | Temperature in Celsius                    |
| rainfall_mm      | Rainfall in mm                            |
| humidity_pct     | Humidity percentage                       |
| wind_speed_kmh   | Wind speed in km/h                        |
| event_nearby     | Indicator if an event is happening nearby |
| traffic_density  | Traffic conditions                        |
| competitor_count | Number of nearby vendors                  |
| lag_1h_units     | Sales 1 hour ago                          |
| lag_24h_units    | Sales 24 hours ago                        |

---

## Technical Stack

- **Python** – Core programming language
- **Pandas & NumPy** – Data manipulation and numerical operations
- **Scikit-learn** – Machine learning algorithms
- **Joblib** – Model serialization
- **Plotly & Plotly Express** – Interactive visualizations
- **Streamlit** – Web application interface
- **WeatherAPI** – Fetching weather and alerts

---

## Installation

1. Clone the repository:

```bash
git clone https://github.com/srivatsacool/Mumbai-Vendor-Demand-Forecasting.git
```

2. Navigate to the project folder:

```bash
cd Mumbai-Vendor-Demand-Forecasting
```

3. Install required dependencies:

```bash
pip install -r requirements.txt
```

---

## Usage

### Running the Streamlit App

```bash
streamlit run app_streamlit.py
```

This will launch a local web application where you can:

- Input vendor parameters
- View historical sales
- Get hourly demand predictions
- Visualize trends with interactive charts

---

## Model Training

- The model is a **Random Forest Regressor** trained on historical vendor sales.
- Training notebook: `demand_forecast_train.ipynb`
- Pre-trained model file: `rf_model.pkl`
- Preprocessing objects: `feature_columns.pkl` and `label_encoders.pkl`

### Steps to retrain the model:

1. Update the dataset with new vendor data.
2. Run `demand_forecast_train.ipynb`.
3. Save the updated model and encoders.

---

## Visualization

- Historical demand trends
- Predicted demand vs actual sales
- Weather impact on sales
- Vendor-specific performance charts

---

## Screenshots

### Streamlit Dashboard

![Dashboard](screenshots/dashboard.png)

### Predicted vs Actual Sales

![Prediction](screenshots/prediction_chart.png)

### Weather Impact Chart

![Weather](screenshots/weather_impact.png)

_Note: Replace the screenshots with actual images from your app._

---

## Contributing

Contributions are welcome! To contribute:

1. Fork the repository
2. Create a new branch: `git checkout -b feature/your-feature`
3. Commit changes: `git commit -m "Add your feature"`
4. Push to branch: `git push origin feature/your-feature`
5. Create a Pull Request

---

### Mumbai Vendor Demand Forecasting - App File Features

This document outlines the purpose and features of each file in the Mumbai Vendor Demand Forecasting project.

---

## 1. `app_streamlit.py` (Streamlit Web App)

**Purpose:** Interactive user interface to input vendor data, fetch predictions, and visualize results.

**Features:**

- **User Inputs:**

  - Vendor parameters: `vendor_id`, `location_type`, `cuisine_type`, `avg_price`, etc.
  - Optional: Date, time, environmental inputs (temperature, traffic).

- **Prediction Section:**

  - Uses pre-trained `rf_model.pkl` to predict hourly demand.
  - Displays forecast for selected vendor and time.

- **Visualizations:**

  - Historical demand trends.
  - Predicted vs actual demand comparison charts.
  - Weather and event impact on sales.

- **Integration:**

  - Loads `feature_columns.pkl` and `label_encoders.pkl` for preprocessing.
  - Uses Plotly and Plotly Express for interactive charts.

---

## 2. `demand_forecast_train.ipynb` (Model Training Notebook)

**Purpose:** Data preprocessing, feature engineering, and training the Random Forest model.

**Features:**

- **Data Loading and Cleaning:** Reads historical vendor CSV and handles missing values.
- **Feature Engineering:**

  - Time-based features: `hour_of_day`, `is_weekend`, etc.
  - Lag features: `lag_1h_units`, `lag_24h_units`.
  - Encodes categorical variables using label encoders.

- **Model Training:** Trains a Random Forest Regressor.
- **Evaluation:** Calculates RMSE, MAE, and R²; saves results to `model_evaluation_results.csv`.
- **Model Export:** Saves `rf_model.pkl` and preprocessing objects (`feature_columns.pkl`, `label_encoders.pkl`).

---

## 3. `requirements.txt`

**Purpose:** Lists Python dependencies needed to run the project.

**Key Packages:**

- `streamlit` – Web application framework
- `pandas`, `numpy` – Data manipulation
- `scikit-learn` – Machine learning
- `joblib` – Model serialization
- `plotly` – Visualization
- `requests` – Fetching API data
- `deep_translator` – Optional translation of fields

---

## 4. `mumbai_vendors_hourly_20250701_20250930.csv` (Dataset)

**Purpose:** Raw historical vendor sales data used for training and testing.

**Features:**

- Vendor identifiers: `vendor_id`, `location_type`, `cuisine_type`
- Sales metrics: `avg_price`, `menu_diversity`, `units_sold`
- Temporal features: `hour_of_day`, `is_weekend`, `is_holiday`, `is_festival`
- Environmental features: `temperature_c`, `rainfall_mm`, `humidity_pct`, `wind_speed_kmh`
- Event/traffic features: `event_nearby`, `traffic_density_encoded`
- Competitor information: `competitor_count`
- Lag features: `lag_1h_units`, `lag_24h_units`

---

## 5. `feature_columns.pkl` & `label_encoders.pkl`

**Purpose:** Preprocessing artifacts to ensure correct feature encoding and order.

**Features:**

- `feature_columns.pkl`: List of model features in the correct order
- `label_encoders.pkl`: Encoders for categorical variables like `vendor_id`, `location_type`, `cuisine_type`, `traffic_density`

---

## 6. `model_evaluation_results.csv`

**Purpose:** Stores model performance metrics.

**Features:**

- RMSE (Root Mean Squared Error)
- MAE (Mean Absolute Error)
- R² (Coefficient of Determination)
- Helps compare different model versions

## Contact

- GitHub: [srivatsacool](https://github.com/srivatsacool)
- Project Author: Srivatsa Cool
