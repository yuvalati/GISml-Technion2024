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
nlp = spacy.load("en_core_web_sm")

# Function to extract the most common location from text
def extract_most_common_location(text):
    # Check if the text contains "(Reuters)"
    if "(Reuters)" in text:
        # Start the text just after "(Reuters)"
        text = text.split("(Reuters)", 1)[1]
    doc = nlp(text)
    locations = [ent.text for ent in doc.ents if ent.label_ == "GPE"]
    locations = ["United States" if loc == "U.S." else loc for loc in locations]
    if locations:
        most_common_location = Counter(locations).most_common(1)[0][0]
        return most_common_location
    return None

# Apply function to text column of the first 10 rows
df_test['location'] = df_test['text'].apply(extract_most_common_location)

# Function to find geocode with caching
def find_Geocode(location, geolocator, cache):
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
    return np.nan, np.nan

# Initialize geocoder and cache
geolocator = Nominatim(user_agent="Yuval", timeout=10)
cache = {}

# Geocode locations
latitude = []
longitude = []

for location in df_test["location"]:
    if location is None:
        latitude.append("")
        longitude.append("")
    else:
        lat, lon = find_Geocode(location, geolocator, cache)
        latitude.append(lat)
        longitude.append(lon)

# Add lat, long to dataframe
df_test["Latitude"] = latitude
df_test["Longitude"] = longitude

# Save the results to an Excel file
output_path = "Combined_with_lat_long.csv"
df_test.to_csv(output_path, index=False)

print(f"Results saved to {output_path}")