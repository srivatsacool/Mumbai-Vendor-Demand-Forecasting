# Generate lag features and rolling averages
print("Generating lag features and rolling averages...")

# Sort by vendor and datetime to ensure correct order
df = df.sort_values(['vendor_id', 'datetime']).reset_index(drop=True)

# Convert datetime column to datetime type for easier manipulation
df['datetime_parsed'] = pd.to_datetime(df['datetime'])

# Generate lag features for each vendor separately
for vendor_id in df['vendor_id'].unique():
    vendor_mask = df['vendor_id'] == vendor_id
    vendor_data = df[vendor_mask].copy().sort_values('datetime_parsed')
    
    # 1-hour lag
    df.loc[vendor_mask, 'lag_1h_units'] = vendor_data['units_sold'].shift(1)
    
    # 24-hour lag
    df.loc[vendor_mask, 'lag_24h_units'] = vendor_data['units_sold'].shift(24)
    
    # 24-hour rolling average (excluding current hour)
    df.loc[vendor_mask, 'rolling_avg_24h'] = vendor_data['units_sold'].shift(1).rolling(window=24, min_periods=12).mean()

# Add some special notes for realistic scenarios
print("Adding special scenarios...")

# Add supply shortage scenarios (random 2-6 hour periods)
supply_shortage_events = 3  # 3 supply shortage events
for _ in range(supply_shortage_events):
    vendor_id = np.random.choice(df['vendor_id'].unique())
    vendor_data = df[df['vendor_id'] == vendor_id]
    
    # Random start time
    start_idx = np.random.randint(100, len(vendor_data) - 50)  # Leave buffer
    duration = np.random.randint(2, 7)  # 2-6 hours
    
    affected_indices = vendor_data.iloc[start_idx:start_idx + duration].index
    df.loc[affected_indices, 'units_sold'] = df.loc[affected_indices, 'units_sold'] * 0.1  # Severe drop
    df.loc[affected_indices, 'special_note'] = 'supply_shortage'

# Add power outage scenarios
power_outage_events = 2
for _ in range(power_outage_events):
    vendor_id = np.random.choice(df['vendor_id'].unique())
    vendor_data = df[df['vendor_id'] == vendor_id]
    
    start_idx = np.random.randint(100, len(vendor_data) - 30)
    duration = np.random.randint(1, 4)  # 1-3 hours
    
    affected_indices = vendor_data.iloc[start_idx:start_idx + duration].index
    df.loc[affected_indices, 'units_sold'] = 0  # Complete stop
    df.loc[affected_indices, 'special_note'] = 'power_outage'

# Introduce some missing values (0.8% as requested)
missing_percentage = 0.008
n_missing = int(len(df) * missing_percentage)
missing_indices = np.random.choice(df.index, size=n_missing, replace=False)

# Make some weather data missing
weather_cols = ['temperature_c', 'rainfall_mm', 'humidity_pct', 'wind_speed_kmh']
for idx in missing_indices[:n_missing//2]:
    col = np.random.choice(weather_cols)
    df.loc[idx, col] = np.nan

# Make some lag features missing (in addition to natural NaN from first hours)
for idx in missing_indices[n_missing//2:]:
    col = np.random.choice(['lag_1h_units', 'rolling_avg_24h'])
    df.loc[idx, col] = np.nan

# Ensure units_sold is non-negative and integer
df['units_sold'] = df['units_sold'].clip(lower=0).astype(int)

# Drop the temporary datetime column
df = df.drop('datetime_parsed', axis=1)

print("Dataset generation complete!")
print(f"Final dataset shape: {df.shape}")
print(f"Missing values introduced: {n_missing}")
print(f"Supply shortage events: {supply_shortage_events}")
print(f"Power outage events: {power_outage_events}")

# Display basic stats
print("\nBasic dataset statistics:")
print(f"Date range: {df['datetime'].min()} to {df['datetime'].max()}")
print(f"Vendors: {df['vendor_id'].nunique()}")
print(f"Total hours per vendor: {len(df) // df['vendor_id'].nunique()}")
print(f"Average units sold: {df['units_sold'].mean():.1f}")
print(f"Holiday hours: {df['is_holiday'].sum()}")
print(f"Festival hours: {df['is_festival'].sum()}")
print(f"Rainy hours: {df['is_rainy'].sum()}")