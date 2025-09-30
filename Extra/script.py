# First, let me create a comprehensive list of holidays and festivals for July-September 2025 based on research

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz
import json

# Set random seed for reproducibility
np.random.seed(42)

# Mumbai timezone
mumbai_tz = pytz.timezone('Asia/Kolkata')

# Define holidays and festivals for July-September 2025 based on research
holidays_festivals = {
    # July 2025
    '2025-07-06': {'name': 'Muharram', 'is_holiday': 1, 'is_festival': 1, 'festival_name': 'Muharram'},
    
    # August 2025
    '2025-08-15': {'name': 'Independence Day', 'is_holiday': 1, 'is_festival': 1, 'festival_name': 'Independence Day'},
    '2025-08-16': {'name': 'Janmashtami', 'is_holiday': 1, 'is_festival': 1, 'festival_name': 'Janmashtami'},
    '2025-08-17': {'name': 'Dahi Handi', 'is_holiday': 0, 'is_festival': 1, 'festival_name': 'Dahi Handi'},  # Day after Janmashtami
    '2025-08-27': {'name': 'Ganesh Chaturthi', 'is_holiday': 1, 'is_festival': 1, 'festival_name': 'Ganesh Chaturthi'},
    
    # September 2025
    '2025-09-05': {'name': 'Eid-e-Milad', 'is_holiday': 1, 'is_festival': 1, 'festival_name': 'Eid-e-Milad'},
    '2025-09-06': {'name': 'Ganesh Visarjan', 'is_holiday': 0, 'is_festival': 1, 'festival_name': 'Ganesh Visarjan'}, # 10 days after Ganesh Chaturthi
}

print("Holidays and Festivals identified for July-September 2025:")
for date, info in holidays_festivals.items():
    print(f"{date}: {info['name']}")