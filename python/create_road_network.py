import osmnx as ox
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from shapely.geometry import Point, LineString, box
import geopandas as gpd
import numpy as np

# Read the random points
points_df = pd.read_csv('random_izmir_points.csv')

# Define the bounding box for Izmir (approximately)
north, south = points_df['latitude'].max(), points_df['latitude'].min()
east, west = points_df['longitude'].max(), points_df['longitude'].min()

# Add some padding to the bounding box
padding = 0.02  # degrees
north += padding
south -= padding
east += padding
west -= padding

print("Downloading road network for Izmir...")
# Create a bounding box polygon
bbox = box(west, south, east, north)
bbox_gdf = gpd.GeoDataFrame({'geometry': [bbox]}, crs="EPSG:4326")

# Download the street network for the bounding box
G = ox.graph.graph_from_polygon(bbox_gdf.geometry.iloc[0], network_type='drive')

# Project the graph to UTM
G_proj = ox.project_graph(G)

# Convert points to a GeoDataFrame
geometry = [Point(xy) for xy in zip(points_df['longitude'], points_df['latitude'])]
points_gdf = gpd.GeoDataFrame(points_df, geometry=geometry, crs="EPSG:4326")
points_gdf_proj = points_gdf.to_crs(G_proj.graph['crs'])

print("Snapping points to nearest network nodes...")
# Find the nearest network nodes for each point
nearest_nodes = []
for point in points_gdf_proj.geometry:
    nearest_node = ox.distance.nearest_nodes(G_proj, point.x, point.y)
    nearest_nodes.append(nearest_node)

# Create edges between points using shortest paths
print("Creating edges using shortest paths...")
edges = []
node_pairs = []

# Create edges between each point and its k nearest neighbors
k = 3  # number of nearest neighbors to connect
for i, node1 in enumerate(nearest_nodes):
    # Calculate distances to all other points
    point1 = points_gdf_proj.geometry[i]
    distances = []
    for j, node2 in enumerate(nearest_nodes):
        if i != j:
            point2 = points_gdf_proj.geometry[j]
            distance = point1.distance(point2)
            distances.append((j, distance))
    
    # Sort by distance and get k nearest neighbors
    distances.sort(key=lambda x: x[1])
    for j, _ in distances[:k]:
        if {i, j} not in node_pairs:  # Avoid duplicate edges
            node_pairs.append({i, j})
            try:
                path = nx.shortest_path(G_proj, nearest_nodes[i], nearest_nodes[j], weight='length')
                edges.append(path)
            except nx.NetworkXNoPath:
                print(f"No path found between points {i} and {j}")

# Plot the results
fig, ax = plt.subplots(figsize=(15, 15))

# Plot the base road network
ox.plot_graph(G_proj, ax=ax, node_size=0, edge_color='gray', edge_alpha=0.2)

# Plot the points
points_gdf_proj.plot(ax=ax, color='red', markersize=50, alpha=0.6, zorder=3)

# Plot the shortest paths
for path in edges:
    path_coords = []
    for node in path:
        path_coords.append((G_proj.nodes[node]['x'], G_proj.nodes[node]['y']))
    line = LineString(path_coords)
    gpd.GeoSeries([line], crs=G_proj.graph['crs']).plot(ax=ax, color='blue', linewidth=2, alpha=0.4, zorder=2)

plt.title('Izmir Transportation Network\nRed dots: Nodes, Blue lines: Shortest paths along roads')
plt.tight_layout()
plt.show()

# Save the network data
print("Saving network data...")
# Convert the graph to GeoDataFrame for easier saving
edges_gdf = ox.graph_to_gdfs(G_proj, nodes=False, edges=True)
edges_gdf.to_file("izmir_road_network.gpkg", driver='GPKG')

# Save the node locations
nodes_df = pd.DataFrame({
    'node_id': range(len(nearest_nodes)),
    'network_node_id': nearest_nodes,
    'longitude': points_df['longitude'],
    'latitude': points_df['latitude']
})
nodes_df.to_csv('izmir_network_nodes.csv', index=False)

print("Done! Network data has been saved.")