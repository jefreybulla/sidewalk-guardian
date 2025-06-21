# -----------------------------------------------------------
# 1.  Load 311-cluster results and pick the top N clusters
# -----------------------------------------------------------
# This script processes NYC 311 service request data to identify hotspots of sidewalk blockages.
# It loads clustered complaint data, selects the top N most concentrated areas, and calculates
# bounding boxes around them for further analysis. The bounding boxes are used to fetch street-level
# imagery of these problematic locations.
import pandas as pd, geopandas as gpd
from shapely.geometry import Point, box
from sklearn.cluster import DBSCAN
import folium

CLUSTERS_CSV = "311_Service_Requests_from_2010_to_Present_20250621.csv"
TOP_N = 3                              # download imagery for the 3 busiest blocks
BUFFER_FT = 60                         # meters: enlarge each block bbox a tiny bit

df = pd.read_csv(CLUSTERS_CSV)

# Clean the data by removing rows with missing coordinates
df = df.dropna(subset=['Longitude', 'Latitude'])

gdf = gpd.GeoDataFrame(
        df, geometry=gpd.points_from_xy(df.Longitude, df.Latitude),
        crs="EPSG:4326").to_crs(2263)   # NY State Plane (ft)

# Perform DBSCAN clustering
coords = gdf.geometry.apply(lambda p: [p.x, p.y]).tolist()
db = DBSCAN(eps=30, min_samples=5).fit(coords)  # 30 ft = ~9 m
gdf["cluster"] = db.labels_

# group by cluster id, descending size
top_clusters = (gdf[gdf.cluster!=-1]
                .groupby("cluster")
                .size()
                .sort_values(ascending=False)
                .head(TOP_N)
                .index)

hot_bboxes = []
for cid in top_clusters:
    geom = gdf[gdf.cluster==cid].geometry
    env  = geom.unary_union.envelope.buffer(BUFFER_FT)  # add margin
    minx, miny, maxx, maxy = env.bounds
    # re-project back to lat/lon for Mapillary
    env_latlon = gpd.GeoSeries(box(minx, miny, maxx, maxy), crs=2263).to_crs(4326)[0]
    w, s, e, n = env_latlon.bounds
    hot_bboxes.append(dict(west=w, south=s, east=e, north=n, cid=int(cid)))

print(f"Selected {len(hot_bboxes)} hot blocks:")
for bb in hot_bboxes:
    print(f"  cluster {bb['cid']:>3}: ({bb['south']:.6f},{bb['west']:.6f}) – "
          f"({bb['north']:.6f},{bb['east']:.6f})")

# Print cluster statistics
print(f"\nCluster Statistics:")
print(f"Total points: {len(gdf)}")
print(f"Points in clusters: {len(gdf[gdf.cluster != -1])}")
print(f"Noise points: {len(gdf[gdf.cluster == -1])}")
print(f"Number of clusters: {len(gdf[gdf.cluster != -1]['cluster'].unique())}")

# Visualization: Create interactive map of top 3 clusters
# Convert back to WGS84 for mapping
gdf_wgs84 = gdf.to_crs("EPSG:4326")

# Calculate center point for the map
center_lat = gdf_wgs84.geometry.y.mean()
center_lon = gdf_wgs84.geometry.x.mean()

# Create the map
m = folium.Map(location=[center_lat, center_lon], zoom_start=12)

# Define colors for top clusters
top_colors = ['red', 'blue', 'green']

# Add bounding boxes for top clusters
for i, bb in enumerate(hot_bboxes):
    color = top_colors[i] if i < len(top_colors) else 'purple'
    
    # Create bounding box polygon
    bbox_coords = [
        [bb['south'], bb['west']],
        [bb['south'], bb['east']],
        [bb['north'], bb['east']],
        [bb['north'], bb['west']],
        [bb['south'], bb['west']]
    ]
    
    # Add bounding box to map
    folium.Polygon(
        locations=bbox_coords,
        color=color,
        weight=3,
        fill=True,
        fillColor=color,
        fillOpacity=0.1,
        popup=f'Cluster {bb["cid"]} Bounding Box<br>Size: {len(gdf_wgs84[gdf_wgs84.cluster == bb["cid"]])} points'
    ).add_to(m)

# Add points for top clusters
for i, cid in enumerate(top_clusters):
    color = top_colors[i] if i < len(top_colors) else 'purple'
    cluster_points = gdf_wgs84[gdf_wgs84.cluster == cid]
    
    for idx, row in cluster_points.iterrows():
        folium.CircleMarker(
            location=[row.geometry.y, row.geometry.x],
            radius=4,
            color=color,
            fill=True,
            fillColor=color,
            fillOpacity=0.7,
            popup=f'Cluster {cid}<br>Point {idx}'
        ).add_to(m)

# Add noise points (smaller and gray)
noise_points = gdf_wgs84[gdf_wgs84.cluster == -1]
for idx, row in noise_points.iterrows():
    folium.CircleMarker(
        location=[row.geometry.y, row.geometry.x],
        radius=2,
        color='gray',
        fill=True,
        fillColor='gray',
        fillOpacity=0.3,
        popup=f'Noise Point<br>Point {idx}'
    ).add_to(m)

# Add legend
legend_html = '''
<div style="position: fixed; 
            bottom: 50px; left: 50px; width: 250px; height: 200px; 
            background-color: white; border:2px solid grey; z-index:9999; 
            font-size:14px; padding: 10px">
            <p><b>Top 3 Clusters</b></p>
            <p><i class="fa fa-square" style="color:red"></i> Cluster 0 (Largest)</p>
            <p><i class="fa fa-square" style="color:blue"></i> Cluster 1 (2nd Largest)</p>
            <p><i class="fa fa-square" style="color:green"></i> Cluster 2 (3rd Largest)</p>
            <p><i class="fa fa-circle" style="color:gray"></i> Noise Points</p>
            <p><b>Red/Blue/Green boxes:</b><br>
            Bounding boxes with 60ft buffer</p>
</div>
'''
m.get_root().html.add_child(folium.Element(legend_html))

# Save the map
m.save('top3_clusters_map.html')
print("\nInteractive map saved as 'top3_clusters_map.html'")
print("Open this file in your web browser to view the visualization")
