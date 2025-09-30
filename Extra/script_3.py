# Generate events and traffic patterns
def generate_events_traffic(datetime_obj, vendor_info):
    """Generate events and traffic for given datetime and vendor"""
    hour = datetime_obj.hour
    day_of_week = datetime_obj.weekday()  # Monday=0
    location_type = vendor_info['location_type']
    
    # Event probability based on location and time
    event_prob = 0.02  # Base 2% chance
    if location_type in ['business_district', 'office'] and day_of_week < 5:  # Weekday business areas
        event_prob = 0.05
    elif location_type == 'market' and day_of_week in [5, 6]:  # Weekend markets
        event_prob = 0.08
    
    # Special events for specific dates (concerts, matches, etc.)
    date_str = datetime_obj.strftime('%Y-%m-%d')
    special_event_dates = {
        '2025-08-15': 0.4,  # Independence Day celebrations
        '2025-08-27': 0.6,  # Ganesh Chaturthi - major event
        '2025-09-06': 0.5,  # Ganesh Visarjan
    }
    
    if date_str in special_event_dates:
        event_prob = special_event_dates[date_str]
    
    event_nearby = 1 if np.random.random() < event_prob else 0
    
    # Traffic density based on hour, day, location, and events
    traffic_base = {
        'business_district': {'peak_hours': [8, 9, 18, 19], 'base_level': 'medium'},
        'near_college': {'peak_hours': [8, 9, 16, 17, 18], 'base_level': 'medium'},
        'market': {'peak_hours': [10, 11, 17, 18, 19], 'base_level': 'high'},
        'metro_station': {'peak_hours': [7, 8, 9, 17, 18, 19], 'base_level': 'high'},
        'office': {'peak_hours': [8, 9, 12, 13, 18, 19], 'base_level': 'medium'}
    }
    
    location_traffic = traffic_base.get(location_type, {'peak_hours': [12, 18], 'base_level': 'medium'})
    
    if hour in location_traffic['peak_hours'] and day_of_week < 5:  # Weekday peaks
        traffic_level = 'high'
    elif hour in location_traffic['peak_hours'] and day_of_week >= 5:  # Weekend peaks
        traffic_level = 'medium'
    elif event_nearby:
        traffic_level = 'high'
    elif location_traffic['base_level'] == 'high':
        traffic_level = 'medium'
    else:
        traffic_level = 'low'
    
    # Weekend traffic adjustments
    if day_of_week >= 5:  # Weekend
        if location_type in ['business_district', 'office']:
            traffic_level = 'low'  # Less office traffic on weekends
        elif location_type == 'market':
            traffic_level = 'high'  # Markets busy on weekends
    
    # Competitor count (relatively stable with some variation)
    base_competitors = {
        'business_district': 6, 'near_college': 8, 'market': 10, 
        'metro_station': 7, 'office': 5
    }
    competitor_count = max(0, base_competitors.get(location_type, 6) + np.random.randint(-2, 3))
    
    return event_nearby, traffic_level, competitor_count

# Test event/traffic generation
test_vendor = vendors[0]  # BKC vendor
test_event, test_traffic, test_competitors = generate_events_traffic(test_date, test_vendor)
print(f"Sample event/traffic generation for {test_vendor['vendor_name']}:")
print(f"Event nearby: {test_event}, Traffic: {test_traffic}, Competitors: {test_competitors}")