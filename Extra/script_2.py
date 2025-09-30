# Generate realistic weather patterns for Mumbai
def generate_mumbai_weather(datetime_obj):
    """Generate realistic hourly weather for Mumbai during monsoon season"""
    month = datetime_obj.month
    hour = datetime_obj.hour
    day_of_year = datetime_obj.timetuple().tm_yday
    
    # Base temperature ranges by month (Mumbai monsoon season)
    if month == 7:  # Peak monsoon
        base_temp_min, base_temp_max = 25.0, 29.8
        rainfall_prob = 0.65  # High probability of rain
        base_humidity = 82
    elif month == 8:  # Monsoon continues
        base_temp_min, base_temp_max = 24.5, 29.3  
        rainfall_prob = 0.60
        base_humidity = 80
    else:  # September - monsoon retreating
        base_temp_min, base_temp_max = 25.5, 30.0
        rainfall_prob = 0.45
        base_humidity = 75
    
    # Hourly temperature variation
    temp_variation = 2.5 * math.sin((hour - 6) * math.pi / 12)  # Peak at 2 PM, min at 6 AM
    temperature = (base_temp_min + base_temp_max) / 2 + temp_variation + np.random.normal(0, 1.2)
    temperature = max(22, min(35, temperature))  # Clamp to reasonable range
    
    # Rainfall generation with clustering (monsoon characteristic)
    if np.random.random() < rainfall_prob:
        # Heavy rain more likely during late afternoon/evening (4-8 PM)
        if 16 <= hour <= 20:
            rainfall = np.random.exponential(8) * np.random.choice([1, 2.5], p=[0.7, 0.3])  # Some heavy spells
        else:
            rainfall = np.random.exponential(3)
        rainfall = min(150, rainfall)  # Cap extreme values
    else:
        rainfall = 0.0
    
    # Humidity correlated with rainfall and base humidity
    humidity = base_humidity + np.random.normal(0, 8) + (10 if rainfall > 0 else 0)
    humidity = max(50, min(95, humidity))
    
    # Wind speed (higher during rain and in afternoon)
    base_wind = 15 + 8 * math.sin((hour - 12) * math.pi / 12)  # Peak in afternoon
    if rainfall > 5:
        base_wind *= 1.5  # Higher wind during rain
    wind_speed = max(5, base_wind + np.random.normal(0, 5))
    
    return {
        'temperature_c': round(temperature, 1),
        'rainfall_mm': round(rainfall, 2),
        'is_rainy': 1 if rainfall > 0.1 else 0,
        'humidity_pct': round(humidity, 1),
        'wind_speed_kmh': round(wind_speed, 1)
    }

# Test weather generation
test_date = datetime(2025, 8, 15, 14, 0, 0)  # Aug 15, 2 PM
test_weather = generate_mumbai_weather(test_date)
print("Sample weather generation:")
print(test_weather)