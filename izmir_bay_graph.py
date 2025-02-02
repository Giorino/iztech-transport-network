import json
import random
import matplotlib.pyplot as plt
from shapely.geometry import shape, Point
import geopandas as gpd

# ==============
# 1. LOAD THE GEOJSON (IZMIR BOUNDARY)
# ==============
geojson_file = "izmir.json"  # Adjust if your file name or path differs

with open(geojson_file, "r", encoding="utf-8") as f:
    izmir_geojson = json.load(f)

# Convert the "geometry" part of the GeoJSON into a Shapely geometry
izmir_polygon = shape(izmir_geojson["geometry"])  # returns a Polygon

# ==============
# 2. RANDOM POINTS INSIDE THE POLYGON
# ==============
num_points = 3000
valid_points = []

# Get bounding box (minx, miny, maxx, maxy)
minx, miny, maxx, maxy = izmir_polygon.bounds

while len(valid_points) < num_points:
    rand_x = random.uniform(minx, maxx)
    rand_y = random.uniform(miny, maxy)
    pt = Point(rand_x, rand_y)

    if izmir_polygon.contains(pt):
        valid_points.append(pt)

print(f"Generated {len(valid_points)} points inside the Izmir polygon.")

# ==============
# 3. SAVE TO CSV (OPTION A)
# ==============
csv_file = "random_izmir_points.csv"
with open(csv_file, "w", encoding="utf-8") as f:
    f.write("longitude,latitude\n")
    for p in valid_points:
        f.write(f"{p.x},{p.y}\n")

print(f"Points saved to '{csv_file}'")

# ==============
# 4. VISUALIZE WITH GEOPANDAS (OPTION C)
# ==============
# Create a GeoDataFrame for the polygon
gdf_polygon = gpd.GeoDataFrame(
    {"name": ["Izmir"]},
    geometry=[izmir_polygon],
    crs="EPSG:4326"  # assuming WGS84 lat-lon
)

# Create a GeoDataFrame for the points
gdf_points = gpd.GeoDataFrame(
    geometry=valid_points,
    crs="EPSG:4326"
)

# Plot
ax = gdf_polygon.plot(color="white", edgecolor="black", figsize=(8,8))
gdf_points.plot(ax=ax, color="red", markersize=2)

plt.title("Random Points Within Izmir")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.show()
