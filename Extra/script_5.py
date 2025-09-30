# Generate the complete dataset
print("Generating complete Mumbai street vendor dataset...")

# Initialize list to store all records
records = []

# Track progress
total_records = len(date_range) * len(vendors)
processed = 0

# Generate data for each hour and vendor
for dt in date_range:
    # Get date info
    date_str = dt.strftime('%Y-%m-%d')
    hour = dt.hour
    day_of_week = dt.weekday()
    is_weekend = 1 if day_of_week >= 5 else 0
    
    # Check if it's a holiday/festival
    is_holiday = holidays_festivals.get(date_str, {}).get('is_holiday', 0)
    is_festival = holidays_festivals.get(date_str, {}).get('is_festival', 0) 
    festival_name = holidays_festivals.get(date_str, {}).get('festival_name', '')
    
    # Generate weather for this hour (same for all vendors in same location)
    weather = generate_mumbai_weather(dt)
    
    # Generate data for each vendor
    for vendor in vendors:
        # Generate events and traffic for this vendor/datetime
        event_nearby, traffic_level, competitor_count = generate_events_traffic(dt, vendor)
        
        # Calculate units sold
        units_sold = calculate_units_sold(
            dt, vendor, weather, event_nearby, traffic_level, 
            is_holiday, is_festival, festival_name
        )
        
        # Create record
        record = {
            'datetime': dt.strftime('%Y-%m-%d %H:%M:%S%z'),
            'vendor_id': vendor['vendor_id'],
            'vendor_name': vendor['vendor_name'],
            'latitude': vendor['latitude'],
            'longitude': vendor['longitude'],
            'location_type': vendor['location_type'],
            'cuisine_type': vendor['cuisine_type'],
            'avg_price': vendor['avg_price'],
            'menu_diversity': vendor['menu_diversity'],
            'hour_of_day': hour,
            'day_of_week': day_of_week,
            'is_weekend': is_weekend,
            'is_holiday': is_holiday,
            'festival_name': festival_name,
            'is_festival': is_festival,
            'temperature_c': weather['temperature_c'],
            'rainfall_mm': weather['rainfall_mm'],
            'is_rainy': weather['is_rainy'],
            'humidity_pct': weather['humidity_pct'],
            'wind_speed_kmh': weather['wind_speed_kmh'],
            'event_nearby': event_nearby,
            'traffic_density': traffic_level,
            'competitor_count': competitor_count,
            'units_sold': units_sold,
            'lag_1h_units': np.nan,  # Will fill later
            'lag_24h_units': np.nan,  # Will fill later
            'rolling_avg_24h': np.nan,  # Will fill later
            'special_note': ''  # Will add some special cases later
        }
        
        records.append(record)
        processed += 1
        
        if processed % 1000 == 0:
            print(f"Processed {processed}/{total_records} records ({100*processed/total_records:.1f}%)")

print(f"Generated {len(records)} total records")

# Convert to DataFrame
df = pd.DataFrame(records)
print(f"DataFrame shape: {df.shape}")