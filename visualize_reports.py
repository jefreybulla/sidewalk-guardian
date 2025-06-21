# This script analyzes NYC 311 service requests for sidewalk blockages by creating a buffer
# around each complaint location and finding their union. It generates a bounding box around
# all buffered points to identify the overall area affected by sidewalk issues, then
# visualizes this on an interactive map along with sample complaint locations.

import pandas as pd, geopandas as gpd
from shapely.geometry import Point
import folium
from shapely.geometry import box

# Load the 311 service requests data
df = pd.read_csv('311_Service_Requests_from_2010_to_Present_20250621.csv')

# Filter out rows with missing coordinates
df = df.dropna(subset=['Longitude', 'Latitude'])

gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.Longitude, df.Latitude), crs="EPSG:4326")
hot = gdf.to_crs(2263).buffer(25).union_all()  # 25 m buffer - using union_all() instead of unary_union
hot_bounds = hot.envelope.bounds  # minx, miny, maxx, maxy
print(hot_bounds)
print('end of clustering')

# Visualization: Create a map showing the bounding box
# Convert the bounding box back to WGS84 for mapping
bbox_geom = box(*hot_bounds)  # Create rectangle from bounds
bbox_gdf = gpd.GeoDataFrame([1], geometry=[bbox_geom], crs="EPSG:2263")
bbox_wgs84 = bbox_gdf.to_crs("EPSG:4326")  # Convert to WGS84

# Get the center point for the map
center_lat = (bbox_wgs84.bounds.iloc[0]['miny'] + bbox_wgs84.bounds.iloc[0]['maxy']) / 2
center_lon = (bbox_wgs84.bounds.iloc[0]['minx'] + bbox_wgs84.bounds.iloc[0]['maxx']) / 2

# Create the map
m = folium.Map(location=[center_lat, center_lon], zoom_start=12)

# Add the bounding box rectangle
folium.GeoJson(
    bbox_wgs84,
    style_function=lambda x: {
        'fillColor': 'red',
        'color': 'red',
        'weight': 2,
        'fillOpacity': 0.1
    }
).add_to(m)

# Add some sample points (first 100 points to avoid overcrowding)
sample_points = gdf.head(3000)
for idx, row in sample_points.iterrows():
    # Double-check for NaN values before plotting
    if not pd.isna(row.geometry.y) and not pd.isna(row.geometry.x):
        folium.CircleMarker(
            location=[row.geometry.y, row.geometry.x],
            radius=3,
            color='blue',
            fill=True,
            fillColor='blue',
            fillOpacity=0.7
        ).add_to(m)

# Save the map
m.save('bounding_box_map.html')
print("Map saved as 'bounding_box_map.html'") 