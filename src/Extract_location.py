import pandas as pd
import spacy  # For natural language processing and extracting locations from text
from collections import Counter  # For counting occurrences of extracted locations
from geopy.geocoders import Nominatim  # For geocoding locations to latitude and longitude
from geopy.exc import GeocoderTimedOut  # To handle timeouts during geocoding
import numpy as np  # For handling NaN and numerical operations

###############################################################
dataset_path = "Combined.csv"
df_test = pd.read_csv(dataset_path)

# Load spaCy model
# spaCy's pre-trained model 'en_core_web_trf' is used for extracting entities from text.
nlp = spacy.load("en_core_web_trf")

# Define synonyms for "United States" to unify different variations under a single label
United_states_synonyms = ["U.S", "America", "States", "US", "U.S.", "the United States"]


###############################################################
# Function to extract the most common location from text
# This function takes a text input, processes it using spaCy to extract location entities (GPE),
# and returns the most frequently mentioned location. If "United States" is most common, it tries
# to return the second most common location if possible.

def extract_most_common_location(text):
    # Process the text using spaCy's NLP model
    doc = nlp(text)

    # Extract entities labeled as "GPE" (geopolitical entities, e.g., countries, cities)
    locations = [ent.text for ent in doc.ents if ent.label_ == "GPE"]

    # Replace any synonyms for "United States" with the standard "United States"
    locations = ["United States" if loc in United_states_synonyms else loc for loc in locations]

    # If locations were found, count their occurrences and find the most common one
    if locations:
        location_counts = Counter(locations)  # Count the frequency of each location
        most_common_location = location_counts.most_common(1)[0][0]  # Find the most common location

        # Handle cases where the most common location is invalid (e.g., contains "@")
        if "@" in most_common_location:
            return "No location"

        # If the most common location is "United States", try to return the second most common location
        if most_common_location == "United States":
            if len(location_counts) > 1:  # Check if there's another location available
                most_common_location = location_counts.most_common(2)[1][0]  # Return second most common location
            else:
                # If no other locations are available, keep "United States"
                return most_common_location

        return most_common_location  # Return the most common location

    # If no locations were found in the text, return "No location"
    return "No location"


###############################################################
# Apply the function to extract the most common location from the 'text' column of the dataset
# This will process each row of the dataset and store the extracted location in a new 'location' column.
df_test['location'] = df_test['text'].apply(extract_most_common_location)


###############################################################
# Function to geocode locations using a geocoder (Nominatim)
# This function checks a cache to avoid repeated geocoding requests for the same location.
# If a location is found in the cache, the cached latitude and longitude are returned.
# Otherwise, the location is geocoded using the Nominatim service, and the result is cached.

def find_Geocode(location, geolocator, cache):
    if location == "No location":
        return 0, 0

    # Check if the location is already in the cache to avoid repeated API requests
    if location in cache:
        return cache[location]

    # Try to geocode the location using Nominatim
    try:
        loc = geolocator.geocode(location)
        if loc:  # If geocoding is successful, return the coordinates and cache them
            cache[location] = (loc.latitude, loc.longitude)
            return loc.latitude, loc.longitude
    except GeocoderTimedOut:  # Handle geocoding timeouts gracefully
        return None

    # If geocoding fails, cache the result as (NaN, NaN) and return default coordinates
    cache[location] = (np.nan, np.nan)
    return 0, 0


###############################################################
# Initialize the geocoder (Nominatim) and the cache for storing geocode results
geolocator = Nominatim(user_agent="Yuval", timeout=10)  # Initialize geolocator with a user agent and timeout
cache = {}  # Dictionary to store geocode results and avoid redundant API calls

# Initialize lists to store the latitude and longitude results
latitude = []
longitude = []

# Loop over the 'location' column in the DataFrame to geocode each location
for i, location in enumerate(df_test["location"]):
    lat, lon = find_Geocode(location, geolocator, cache)  # Geocode the location
    if lat == 0 or lon == 0:  # If geocoding fails, update the location to "No location"
        df_test.at[i, "location"] = "No location"
    latitude.append(lat)  # Store the latitude
    longitude.append(lon)  # Store the longitude

###############################################################
# Add the geocoded latitude and longitude as new columns in the DataFrame
df_test["Latitude"] = latitude
df_test["Longitude"] = longitude

output_path = "Combined_with_lat_long.csv"
df_test.to_csv(output_path, index=False)

print(f"Results saved to {output_path}")  # Print a confirmation message
