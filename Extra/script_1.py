# Generate the comprehensive synthetic dataset
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz
import random
import math

# Set random seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

# Mumbai timezone
mumbai_tz = pytz.timezone('Asia/Kolkata')

# Define time range
start_date = datetime(2025, 7, 1, 0, 0, 0)
end_date = datetime(2025, 9, 30, 23, 0, 0)

# Create hourly datetime range
date_range = pd.date_range(start=start_date, end=end_date, freq='H', tz=mumbai_tz)

print(f"Total hours in dataset: {len(date_range)}")
print(f"Expected rows (5 vendors): {len(date_range) * 5}")

# Define vendors with realistic Mumbai locations
vendors = [
    {
        'vendor_id': 'vendor_01',
        'vendor_name': 'BKC_Pavbhaji',
        'latitude': 19.0685, 'longitude': 72.8785,  # BKC area
        'location_type': 'business_district',
        'cuisine_type': 'main_course',
        'avg_price': 85.0,
        'menu_diversity': 12,
        'peak_hours': [12, 13, 19, 20],  # Lunch and dinner peaks
        'base_demand': 15
    },
    {
        'vendor_id': 'vendor_02', 
        'vendor_name': 'Churchgate_Chai',
        'latitude': 18.9322, 'longitude': 72.8264,  # Churchgate area
        'location_type': 'near_college',
        'cuisine_type': 'chai_snacks',
        'avg_price': 25.0,
        'menu_diversity': 8,
        'peak_hours': [7, 8, 16, 17, 18],  # Morning and evening peaks
        'base_demand': 25
    },
    {
        'vendor_id': 'vendor_03',
        'vendor_name': 'Dadar_Chaat',
        'latitude': 19.0178, 'longitude': 72.8478,  # Dadar market
        'location_type': 'market',
        'cuisine_type': 'chaat',
        'avg_price': 45.0,
        'menu_diversity': 15,
        'peak_hours': [17, 18, 19, 20, 21],  # Evening peaks
        'base_demand': 20
    },
    {
        'vendor_id': 'vendor_04',
        'vendor_name': 'Andheri_Juice',
        'latitude': 19.1136, 'longitude': 72.8697,  # Andheri station
        'location_type': 'metro_station',
        'cuisine_type': 'beverages',
        'avg_price': 35.0,
        'menu_diversity': 10,
        'peak_hours': [7, 8, 9, 17, 18, 19],  # Commute hours
        'base_demand': 18
    },
    {
        'vendor_id': 'vendor_05',
        'vendor_name': 'Powai_Dessert',
        'latitude': 19.1197, 'longitude': 72.9128,  # Powai IT area
        'location_type': 'office',
        'cuisine_type': 'dessert',
        'avg_price': 65.0,
        'menu_diversity': 7,
        'peak_hours': [15, 16, 21, 22],  # Afternoon break and post-dinner
        'base_demand': 12
    }
]

print(f"Vendors defined: {len(vendors)}")
for vendor in vendors:
    print(f"- {vendor['vendor_name']}: {vendor['cuisine_type']} at {vendor['location_type']}")