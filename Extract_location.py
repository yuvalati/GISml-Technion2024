import pandas as pd
import spacy
from collections import Counter
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut
import numpy as np

# Load the dataset
dataset_path = "Combined.csv"
df_test = pd.read_csv(dataset_path)

# Load spaCy model
nlp = spacy.load("en_core_web_trf")

# Function to extract the most common location from text
def extract_most_common_location(text):

    doc = nlp(text)
    locations = [ent.text for ent in doc.ents if ent.label_ == "GPE"]
    locations = ["United States" if "U.S" in loc or "America" in loc or "States" in loc or "US" in loc else loc for loc in locations]

    if locations:
        location_counts = Counter(locations)
        most_common_location = location_counts.most_common(1)[0][0]
        if "@" in most_common_location:
            return "No location"
        if most_common_location == "United States":
            if len(location_counts) > 1:
                # If there are other locations, take the second most common one
                most_common_location = location_counts.most_common(2)[1][0]
            else:
                # If no other locations are available, keep "United States"
                return most_common_location

        return  most_common_location

    return "No location"

# Apply function to text column of the first 10 rows
df_test['location'] = df_test['text'].apply(extract_most_common_location)

# Function to find geocode with caching
def find_Geocode(location, geolocator, cache):
    if location == "No location":
        return 0, 0

    if location in cache:
        return cache[location]
    try:
        loc = geolocator.geocode(location)
        if loc:
            cache[location] = (loc.latitude, loc.longitude)
            return loc.latitude, loc.longitude
    except GeocoderTimedOut:
        return None
    cache[location] = (np.nan, np.nan)
    return 0, 0

# Initialize geocoder and cache
geolocator = Nominatim(user_agent="Yuval", timeout=10)
cache = {}

# Geocode locations
latitude = []
longitude = []

for i, location in enumerate(df_test["location"]):
    lat, lon = find_Geocode(location, geolocator, cache)
    if lat == 0 or lon == 0:
        # Update the location in the DataFrame if geocoding fails
        df_test.at[i, "location"] = "No location"
    latitude.append(lat)
    longitude.append(lon)

# Add lat, long to dataframe
df_test["Latitude"] = latitude
df_test["Longitude"] = longitude

# Save the results to an Excel file
output_path = "Combined_with_lat_long.csv"
df_test.to_csv(output_path, index=False)

print(f"Results saved to {output_path}")