# Generate units sold based on multiple factors
def calculate_units_sold(datetime_obj, vendor_info, weather, event_nearby, traffic_level, is_holiday, is_festival, festival_name):
    """Calculate realistic units sold based on all factors"""
    hour = datetime_obj.hour
    day_of_week = datetime_obj.weekday()
    is_weekend = 1 if day_of_week >= 5 else 0
    
    # Base demand pattern by hour for this vendor
    base_demand = vendor_info['base_demand']
    peak_hours = vendor_info['peak_hours']
    cuisine_type = vendor_info['cuisine_type']
    location_type = vendor_info['location_type']
    
    # Hour multiplier based on vendor's peak hours
    if hour in peak_hours:
        hour_multiplier = 2.0
    elif hour in [h-1 for h in peak_hours] or hour in [h+1 for h in peak_hours]:  # Adjacent hours
        hour_multiplier = 1.4
    elif 6 <= hour <= 22:  # Operating hours
        hour_multiplier = 0.8
    else:  # Very late night/early morning
        hour_multiplier = 0.2
    
    # Weekend multiplier
    weekend_multiplier = 1.0
    if location_type in ['business_district', 'office'] and is_weekend:
        weekend_multiplier = 0.4  # Much less demand in business areas on weekends
    elif location_type in ['market', 'near_college'] and is_weekend:
        weekend_multiplier = 1.3  # More demand in leisure areas on weekends
    
    # Weather multipliers
    weather_multiplier = 1.0
    if weather['is_rainy']:
        if cuisine_type == 'chai_snacks':
            weather_multiplier = 1.4  # Tea/snacks popular in rain
        elif cuisine_type == 'beverages':
            weather_multiplier = 0.6  # Cold drinks less popular in rain
        elif cuisine_type == 'main_course':
            weather_multiplier = 0.8  # Slightly less foot traffic
        else:
            weather_multiplier = 0.9
    
    if weather['temperature_c'] > 32:  # Hot weather
        if cuisine_type == 'beverages':
            weather_multiplier *= 1.3
        elif cuisine_type == 'chai_snacks':
            weather_multiplier *= 0.8
    
    # Traffic multiplier
    traffic_multiplier = {'low': 0.7, 'medium': 1.0, 'high': 1.4}[traffic_level]
    
    # Event multiplier
    event_multiplier = 1.5 if event_nearby else 1.0
    
    # Holiday and festival multipliers
    holiday_multiplier = 1.0
    if is_holiday:
        if location_type in ['business_district', 'office']:
            holiday_multiplier = 0.5  # Offices closed
        else:
            holiday_multiplier = 1.4  # More leisure time
    
    festival_multiplier = 1.0
    if is_festival:
        if festival_name in ['Ganesh Chaturthi', 'Ganesh Visarjan']:
            if location_type == 'market':
                festival_multiplier = 2.5  # Huge crowds during Ganesh festival
            else:
                festival_multiplier = 1.8
        elif festival_name in ['Janmashtami', 'Dahi Handi']:
            festival_multiplier = 1.6
        elif festival_name == 'Independence Day':
            festival_multiplier = 1.3
        else:
            festival_multiplier = 1.2
    
    # Calculate final units
    final_units = (base_demand * hour_multiplier * weekend_multiplier * 
                  weather_multiplier * traffic_multiplier * event_multiplier * 
                  holiday_multiplier * festival_multiplier)
    
    # Add noise and ensure non-negative integer
    noise_factor = np.random.normal(1.0, 0.15)  # 15% noise
    final_units = max(0, int(round(final_units * noise_factor)))
    
    return final_units

# Test units calculation
test_weather_data = generate_mumbai_weather(test_date)
test_units = calculate_units_sold(
    test_date, test_vendor, test_weather_data, 
    test_event, test_traffic, 1, 1, 'Independence Day'  # Holiday and festival
)
print(f"Sample units calculation: {test_units} units")
print(f"Date: {test_date}, Vendor: {test_vendor['vendor_name']}")
print(f"Weather: {test_weather_data}")
print(f"Event: {test_event}, Traffic: {test_traffic}")