import pandas as pd
import folium
from folium.plugins import HeatMap

# Load your CSV with the geocoded data (Combined_with_lat_long.csv)
df = pd.read_csv('Combined_with_lat_long.csv')

# Initialize a base map
m = folium.Map(location=[20, 0], zoom_start=2)  # You can center the map according to your data


# Function to assign colors to markers based on the 'label' (1 = Real, 0 = Fake)
def get_color(label):
    if label == 1:
        return 'green'  # Green for real news
    else:
        return 'red'  # Red for fake news


# Add markers to the map
for i, row in df.iterrows():
    # Skip rows where there are no valid locations
    if row['Latitude'] == 0 or row['Longitude'] == 0:
        continue

    # Create a marker for each location
    folium.Marker(
        location=[row['Latitude'], row['Longitude']],
        popup=f"{row['title']}: {row['label']}",  # Show title and label on click
        icon=folium.Icon(color=get_color(row['label']))
    ).add_to(m)

# Save the map to an HTML file
m.save("map_with_news.html")

print("Map saved to 'map_with_news.html'")
