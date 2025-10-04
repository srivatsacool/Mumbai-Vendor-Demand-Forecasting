# Mumbai Street Vendor Demand Forecasting

This project provides an end-to-end machine learning solution for predicting hourly demand for street food vendors in Mumbai. The system includes synthetic realistic data, trained models, and an interactive Streamlit application.

## 📁 Project Structure

```
mumbai_street_vendors_forecasting/
├── mumbai_vendors_hourly_20250701_20250930.csv    # Synthetic dataset
├── demand_forecast_train.ipynb                     # Training notebook
├── app_streamlit.py                                # Streamlit application
├── requirements.txt                                # Python dependencies
├── README.md                                       # This file
└── Model artifacts (generated after training):
    ├── xgb_model.pkl / rf_model.pkl               # Trained model
    ├── label_encoders.pkl                          # Categorical encoders
    ├── feature_columns.pkl                         # Feature list
    └── model_metadata.json                         # Model information
```

## 🎯 Dataset Overview

**Time Period:** July 1, 2025 - September 30, 2025 (hourly granularity)  
**Vendors:** 5 distinct street vendors across Mumbai  
**Total Records:** ~11,040 rows  
**Features:** 28 columns including weather, festivals, traffic, and lag features

### Vendor Locations
- **BKC_Pavbhaji**: Business district (main course, ₹85 avg price)
- **Churchgate_Chai**: Near college (chai & snacks, ₹25 avg price)
- **Dadar_Chaat**: Market area (chaat, ₹45 avg price)
- **Andheri_Juice**: Metro station (beverages, ₹35 avg price)
- **Powai_Dessert**: Office area (desserts, ₹65 avg price)

### Key Features
- **Weather**: Temperature, rainfall, humidity, wind speed
- **Time**: Hour, day of week, weekend indicators
- **Events**: Holidays, festivals, nearby events
- **Location**: Traffic density, competitor count, location type
- **Historical**: 1-hour lag, 24-hour lag, rolling averages

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Training Notebook
```bash
jupyter notebook demand_forecast_train.ipynb
```

Execute all cells to:
- Perform exploratory data analysis
- Train XGBoost and Random Forest models
- Generate feature importance analysis
- Save model artifacts

### 3. Launch the Streamlit App
```bash
streamlit run app_streamlit.py
```

The app provides:
- **Interactive Predictions**: Select vendor, date/time, and weather conditions
- **Festival Simulation**: Test impact of major Mumbai festivals
- **Historical Context**: View 48-hour demand patterns
- **Batch Forecasting**: Generate 24-hour demand forecasts
- **Downloadable Results**: Export predictions as CSV

## 📊 Model Performance

The system trains both XGBoost and Random Forest models with time-based validation:

- **Validation Split**: Last 20% of data (time-based)
- **Cross-Validation**: TimeSeriesSplit for hyperparameter tuning
- **Metrics**: MAE, RMSE, MAPE
- **Typical Performance**: MAE ~3-5 units, MAPE ~15-25%

## 🏆 Key Features & Realism

### Mumbai-Specific Elements
- **Monsoon Weather Patterns**: Heavy rainfall clustering in July-September
- **Festival Calendar**: Ganesh Chaturthi, Janmashtami, Independence Day, Eid
- **Location-Based Demand**: Business districts vs. markets vs. stations
- **Cuisine-Specific Weather Impact**: Tea sales up during rain, cold drinks down

### Realistic Behavioral Patterns
- **Peak Hours**: Vendor-specific (breakfast, lunch, evening snacks)
- **Weekend Effects**: Lower demand in business areas, higher in leisure areas
- **Festival Multipliers**: 50-150% demand increases during major festivals
- **Weather Correlations**: Rain affects different cuisines differently

## 📈 Usage Examples

### Prediction Scenarios
1. **Rainy Festival Evening**: High demand at market vendors during Ganesh Chaturthi
2. **Business Lunch**: Peak demand at BKC during weekday lunch hours
3. **Weekend Morning**: Lower office area demand, higher college area demand
4. **Hot Weather**: Increased beverage sales, decreased hot food sales

### Model Insights
- **Most Important Features**: Hour of day, lag values, traffic density
- **Weather Impact**: 20-40% demand changes based on rainfall
- **Festival Boost**: Major festivals can double demand
- **Location Matters**: Business districts peak at lunch, markets peak evenings

## 🔧 Technical Details

### Data Generation
- **Random Seed**: 42 (fully reproducible)
- **Weather Patterns**: Based on Mumbai monsoon climatology
- **Holiday Calendar**: Maharashtra government official holidays 2025
- **Realistic Noise**: 15% variance with outliers and missing values

### Model Training
- **Feature Engineering**: 23 engineered features from raw data
- **Encoding**: Label encoding for categorical variables
- **Validation**: Time-based split to prevent data leakage
- **Hyperparameter Tuning**: GridSearchCV with TimeSeriesSplit

### Streamlit App Features
- **Real-time Predictions**: Instant forecasts with parameter changes
- **Interactive Controls**: Sliders, dropdowns, date pickers
- **Visualizations**: Plotly charts for historical context
- **Explanations**: Natural language prediction reasoning
- **Export Options**: CSV download for batch predictions

## 🌟 Next Steps & Improvements

1. **Real Data Integration**: Replace synthetic data with actual vendor sales
2. **Advanced Models**: Try LSTM, Prophet, or ensemble methods
3. **External Data**: Weather APIs, event calendars, traffic data
4. **Vendor-Specific Models**: Individual models per vendor type
5. **Real-time Updates**: Live data feeds and model retraining

## 📞 Support

For questions or issues:
- Review the Jupyter notebook for detailed analysis
- Check model artifacts and metadata
- Examine the Streamlit app code for implementation details

## 📄 License

This project is created for educational and demonstration purposes. The synthetic dataset is generated and does not represent real vendor data.

---

**Created with**: Python 3.10+, XGBoost, Random Forest, Streamlit, Plotly  
**Random Seed**: 42 (reproducible results)  
**Generated**: October 1, 2025
