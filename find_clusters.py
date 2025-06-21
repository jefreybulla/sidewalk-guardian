# This script analyzes NYC 311 service requests for sidewalk blockages using DBSCAN clustering
# to identify areas with high concentrations of complaints. It processes coordinate data,
# finds the top 5 densest clusters of reports, and visualizes them on an interactive map.

import pandas as pd, geopandas as gpd
from shapely.geometry import Point
from sklearn.cluster import DBSCAN
import folium
import numpy as np

df = pd.read_csv("311_Service_Requests_from_2010_to_Present_20250621.csv")

# Clean the data by removing rows with missing coordinates
df = df.dropna(subset=['Longitude', 'Latitude'])

gdf = gpd.GeoDataFrame(df,
        geometry=gpd.points_from_xy(df.Longitude, df.Latitude),
        crs="EPSG:4326").to_crs(2263)          # NY-Long Island ft

coords = gdf.geometry.apply(lambda p: [p.x, p.y]).tolist()
db = DBSCAN(eps=30, min_samples=5).fit(coords)  # 30 ft = ~9 m

gdf["cluster"] = db.labels_
hot = (gdf[gdf.cluster != -1]
       .groupby("cluster")
       .size()
       .sort_values(ascending=False)
       .head(5))                      # top 5 dense clusters

print(hot)

# Visualization: Create an interactive map of the clusters
# Convert back to WGS84 for mapping
gdf_wgs84 = gdf.to_crs("EPSG:4326")

# Calculate center point for the map
center_lat = gdf_wgs84.geometry.y.mean()
center_lon = gdf_wgs84.geometry.x.mean()

# Create the map
m = folium.Map(location=[center_lat, center_lon], zoom_start=12)

# Define colors for clusters
colors = ['red', 'blue', 'green', 'purple', 'orange', 'darkred', 'lightred', 'beige', 'darkblue', 'darkgreen']

# Add points to the map
for idx, row in gdf_wgs84.iterrows():
    if row.cluster == -1:
        # Noise points (not in any cluster)
        folium.CircleMarker(
            location=[row.geometry.y, row.geometry.x],
            radius=2,
            color='gray',
            fill=True,
            fillColor='gray',
            fillOpacity=0.3,
            popup=f'Noise Point<br>Cluster: None'
        ).add_to(m)
    else:
        # Clustered points
        color_idx = row.cluster % len(colors)
        cluster_size = len(gdf_wgs84[gdf_wgs84.cluster == row.cluster])
        
        folium.CircleMarker(
            location=[row.geometry.y, row.geometry.x],
            radius=4,
            color=colors[color_idx],
            fill=True,
            fillColor=colors[color_idx],
            fillOpacity=0.7,
            popup=f'Cluster {row.cluster}<br>Size: {cluster_size} points'
        ).add_to(m)

# Add a legend
legend_html = '''
<div style="position: fixed; 
            bottom: 50px; left: 50px; width: 200px; height: 200px; 
            background-color: white; border:2px solid grey; z-index:9999; 
            font-size:14px; padding: 10px">
            <p><b>DBSCAN Clusters</b></p>
            <p><i class="fa fa-circle" style="color:red"></i> Cluster 0</p>
            <p><i class="fa fa-circle" style="color:blue"></i> Cluster 1</p>
            <p><i class="fa fa-circle" style="color:green"></i> Cluster 2</p>
            <p><i class="fa fa-circle" style="color:purple"></i> Cluster 3</p>
            <p><i class="fa fa-circle" style="color:orange"></i> Cluster 4</p>
            <p><i class="fa fa-circle" style="color:gray"></i> Noise Points</p>
            <p><b>Parameters:</b><br>
            eps=30ft (~9m)<br>
            min_samples=5</p>
</div>
'''
m.get_root().html.add_child(folium.Element(legend_html))

# Save the map
m.save('dbscan_clusters_map.html')
print("Interactive cluster map saved as 'dbscan_clusters_map.html'")

# Print cluster statistics
print(f"\nCluster Statistics:")
print(f"Total points: {len(gdf)}")
print(f"Points in clusters: {len(gdf[gdf.cluster != -1])}")
print(f"Noise points: {len(gdf[gdf.cluster == -1])}")
print(f"Number of clusters: {len(gdf[gdf.cluster != -1]['cluster'].unique())}")
