import json
import random
import matplotlib.pyplot as plt
from shapely.geometry import shape, Point
import geopandas as gpd
import pandas as pd
import numpy as np

# ==============
# 1. LOAD THE GEOJSON (IZMIR BOUNDARY)
# ==============
geojson_file = "izmir.json"

with open(geojson_file, "r", encoding="utf-8") as f:
    izmir_geojson = json.load(f)

# Convert the "geometry" part of the GeoJSON into a Shapely geometry
izmir_polygon = shape(izmir_geojson["geometry"])

# ==============
# 2. LOAD NEIGHBORHOOD DATA
# ==============
neighborhoods_df = pd.read_csv("neighborhood_population.csv")
total_population = neighborhoods_df['Population'].sum()

# Calculate the number of points for each neighborhood based on population
num_total_points = 50  # Total points to generate
neighborhoods_df['num_points'] = (neighborhoods_df['Population'] / total_population * num_total_points).astype(int)

# ==============
# 3. GENERATE POPULATION-WEIGHTED RANDOM POINTS
# ==============
valid_points = []

# Parameters for the normal distribution around neighborhood centers
std_dev = 0.05  # Standard deviation in degrees (adjust this to control spread)

for _, neighborhood in neighborhoods_df.iterrows():
    points_to_generate = neighborhood['num_points']
    center_lat = neighborhood['Latitude']
    center_lon = neighborhood['Longitude']
    
    neighborhood_points = []
    while len(neighborhood_points) < points_to_generate:
        # Generate points using normal distribution around the neighborhood center
        rand_x = random.gauss(center_lon, std_dev)
        rand_y = random.gauss(center_lat, std_dev)
        pt = Point(rand_x, rand_y)
        
        # Only keep points that fall within Izmir boundaries
        if izmir_polygon.contains(pt):
            neighborhood_points.append(pt)
    
    valid_points.extend(neighborhood_points)

print(f"Generated {len(valid_points)} points distributed by population density.")

# ==============
# 4. SAVE TO CSV
# ==============
csv_file = "random_izmir_points.csv"
with open(csv_file, "w", encoding="utf-8") as f:
    f.write("longitude,latitude\n")
    for p in valid_points:
        f.write(f"{p.x},{p.y}\n")

print(f"Points saved to '{csv_file}'")

# ==============
# 5. VISUALIZE WITH GEOPANDAS
# ==============
# Create a GeoDataFrame for the polygon
gdf_polygon = gpd.GeoDataFrame(
    {"name": ["Izmir"]},
    geometry=[izmir_polygon],
    crs="EPSG:4326"
)

# Create a GeoDataFrame for the points
gdf_points = gpd.GeoDataFrame(
    geometry=valid_points,
    crs="EPSG:4326"
)

# Create a GeoDataFrame for neighborhood centers
gdf_centers = gpd.GeoDataFrame(
    neighborhoods_df,
    geometry=gpd.points_from_xy(neighborhoods_df.Longitude, neighborhoods_df.Latitude),
    crs="EPSG:4326"
)

# Plot
fig, ax = plt.subplots(figsize=(12, 12))
gdf_polygon.plot(ax=ax, color="white", edgecolor="black")
gdf_points.plot(ax=ax, color="red", markersize=2, alpha=0.5)
gdf_centers.plot(ax=ax, color="blue", markersize=50, alpha=0.5, label="Neighborhood Centers")

plt.title("Population-Based Random Points Within Izmir")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.legend()
plt.show()
