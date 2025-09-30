# Save the dataset as CSV
csv_filename = 'mumbai_vendors_hourly_20250701_20250930.csv'
df.to_csv(csv_filename, index=False)

print(f"Dataset saved as: {csv_filename}")
print(f"File size: {len(df)} rows x {len(df.columns)} columns")

# Display first 8 rows as requested
print("\n" + "="*80)
print("PREVIEW: First 8 rows of the dataset")
print("="*80)
print(df.head(8).to_string())

# Show data types and missing values
print("\n" + "="*50)
print("DATA TYPES AND MISSING VALUES")
print("="*50)
print(df.dtypes)
print(f"\nMissing values per column:")
print(df.isnull().sum())

# Show unique values for categorical columns
print("\n" + "="*50)
print("CATEGORICAL VARIABLE DISTRIBUTIONS")
print("="*50)
categorical_cols = ['vendor_name', 'location_type', 'cuisine_type', 'traffic_density']
for col in categorical_cols:
    print(f"\n{col}:")
    print(df[col].value_counts())

# Show holiday/festival distribution
print(f"\nFestival distribution:")
print(df[df['is_festival'] == 1]['festival_name'].value_counts())

# Weather statistics
print(f"\n" + "="*50)
print("WEATHER STATISTICS")
print("="*50)
weather_stats = df[['temperature_c', 'rainfall_mm', 'humidity_pct', 'wind_speed_kmh']].describe()
print(weather_stats)