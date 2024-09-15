import pandas as pd

# Path to your dataset
dataset_path = "Combined_with_lat_long_trf_2.csv"

df = pd.read_csv(dataset_path, encoding='ISO-8859-1')

# Group by Location and calculate the necessary values
result = df.groupby('location').agg(
    Latitude=('Latitude', 'first'),           # Keep the first Latitude for each location
    Longitude=('Longitude', 'first'),         # Keep the first Longitude for each location
    Total=('label', 'count'),                 # Total number of labels
    Real=('label', 'sum'),                    # Sum of real labels
)

# Calculate Real-%
result['Real-%'] = round((result['Real'] / result['Total']) * 100)

# Sort by 'Total' in descending order
result = result.sort_values(by='Total', ascending=False)

# Save the result to a CSV file
output_path = "location_summary.csv"
result.to_csv(output_path)

print(f"Results saved to {output_path}")  # Print a confirmation message
