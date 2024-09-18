import pandas as pd
import folium
import matplotlib.colors as mcolors
import numpy as np

# Load the summary CSV
data = pd.read_csv("location_summary.csv")


# Function to normalize Real-% to a color between red (0%) and green (100%)
def get_color(real_percent):
    # Create a color map that transitions from red to green
    cmap = mcolors.LinearSegmentedColormap.from_list("red_green", ["red", "green"])
    # Normalize the percentage value (0 to 1 scale)
    normalized_value = real_percent / 100
    return mcolors.to_hex(cmap(normalized_value))


# keep only the entry with the highest Total for each unique location, so the locations will not be on top of each other
data = data.sort_values('Total', ascending=False).drop_duplicates(subset=['Latitude', 'Longitude'], keep='first')

mymap = folium.Map(location=[0, 0], zoom_start=2)

# Add circles to the map for each location
for index, row in data.iterrows():

    if row['location'] != "No location":

        # Get color based on Real-% (0% is red, 100% is green)
        color = get_color(row['Real-%'])

        # Apply logarithmic scaling to the circle size
        size = np.log(row['Total'] + 1) * 3  # Log scale with a constant multiplier for visibility

        # Set the opacity
        opacity = 0.5

        # Add a circle marker for each location
        folium.CircleMarker(
            location=(row['Latitude'], row['Longitude']),
            radius=size,  # Log-scaled size for the circles
            color=None,  # No border color
            fill=True,
            fill_color=color,  # Fill color
            fill_opacity=opacity,
            popup=f"Location: {row['location']}. Total news: {row['Total']}, {row['Real-%']}% Real"
        ).add_to(mymap)

# Save the heatmap to an HTML file
mymap.save("heatmap_location_summary.html")

print("Heatmap saved as heatmap_location_summary.html")
