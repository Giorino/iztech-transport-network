# Transportation Network Analysis of Izmir Institute of Technology

This project analyzes the transportation network of Izmir Institute of Technology using graph theory and various clustering algorithms. It allows for the construction of transportation graphs, application of community detection algorithms, and analysis of transportation costs.

## Features

- Generation of random vertices based on population centers
- Construction of road networks using different graph strategies (Complete, K-Nearest Neighbors, Gabriel, Delaunay)
- Application of community detection algorithms (Leiden, Spectral Clustering)
- Transportation cost analysis for optimized bus routing
- Visualization of graphs and communities
- Persistence of constructed graphs for reuse

## Project Structure

The project is organized into several modules:

- **Graph Construction**: Builds transportation graphs using different strategies
- **Community Detection**: Applies clustering algorithms to identify transportation zones
- **Cost Analysis**: Analyzes transportation costs for each community
- **Persistence**: Saves and loads constructed graphs to avoid rebuilding

## Configuration

The main application parameters can be configured in the `App.java` file:

```java
// Number of nodes to generate
private static final int NODE_COUNT = 1000; 

// Graph construction strategy (COMPLETE, K_NEAREST_NEIGHBORS, GABRIEL, DELAUNAY)
private static final GraphConstructionService.GraphStrategy GRAPH_STRATEGY = 
        GraphConstructionService.GraphStrategy.COMPLETE;

// K value for K-nearest neighbors strategy
private static final int K_VALUE = 5; 

// Clustering algorithm (LEIDEN, SPECTRAL)
private static final ClusteringService.ClusteringAlgorithm CLUSTERING_ALGORITHM = 
        ClusteringService.ClusteringAlgorithm.LEIDEN;

// Whether to use parallel processing
private static final boolean USE_PARALLEL = true;

// Whether to visualize the graph
private static final boolean VISUALIZE_GRAPH = true;

// Whether to visualize clusters
private static final boolean VISUALIZE_CLUSTERS = true;

// Whether to save the graph for future use
private static final boolean SAVE_GRAPH = true;
```

## Running the Application

To run the application:

1. Build the project: `mvn clean package`
2. Run the application: `java -jar target/transportation-analysis-1.0.jar`

The application will:
1. Generate random points based on population centers
2. Check if a saved graph exists with the specified parameters
3. If a saved graph exists, load it; otherwise, construct a new graph
4. Apply the specified clustering algorithm to the graph
5. Perform transportation cost analysis on the detected communities
6. Visualize the results and save them to files

## Extending the Project

### Adding a New Graph Construction Strategy

1. Create a new class implementing `GraphConnectivityStrategy` in the `src/main/java/com/izmir/transportation/helper/strategy` package
2. Add the new strategy to the `GraphStrategy` enum in `GraphConstructionService.java`
3. Update the `createStrategy` method in `GraphConstructionService.java` to handle the new strategy

### Adding a New Clustering Algorithm

1. Create a new class implementing `GraphClusteringAlgorithm` in the `src/main/java/com/izmir/transportation/helper/clustering` package
2. Add the new algorithm to the `ClusteringAlgorithm` enum in `ClusteringService.java`
3. Update the `performClustering` method in `ClusteringService.java` to handle the new algorithm

## Performance Considerations

For large graphs (e.g., 3000+ nodes), the graph construction process can be time-consuming. The persistence mechanism allows you to save constructed graphs and reuse them for different clustering algorithms or analyses. 

Key tips:
- Set `SAVE_GRAPH = true` to persist graphs
- Graphs are saved with filenames based on node count and strategy (e.g., `graph_3000_complete.json`)
- When running with the same node count and strategy, the application will automatically load the saved graph

# Node Removal Script

This repository contains a shell script (`remove_node.sh`) that removes a specific node with coordinates "26.643221256222105, 38.319147501994145" from a transportation graph JSON file, along with all edges connected to that node.

## Prerequisites

- Bash shell
- jq (JSON processor) - Install with `brew install jq` on macOS

## Usage

```bash
./remove_node.sh <path_to_json_file>
```

For example:
```bash
./remove_node.sh graph_data/graph_100_gabriel.json
```

## What the Script Does

1. Checks if jq is installed
2. Verifies the input file exists
3. Finds the node with coordinates (26.643221256222105, 38.319147501994145)
4. Creates a new JSON file with timestamp in the name (original_filename_without_node_YYYYMMDD_HHMMSS.json)
5. Removes the node from the nodes array
6. Removes all edges connected to the node
7. Updates the nodeCount property
8. Outputs detailed logging information

## Output

The script will create a new JSON file with all changes applied. The original file remains untouched.

Sample output:
```
Starting to process file: graph_data/graph_100_gabriel.json
Will create new file: graph_data/graph_100_gabriel_without_node_20250409_214328.json
Found node with ID: -630016627 to remove
Operation completed successfully.
Original node count: 100
New node count: 99
Original edge count: 169
New edge count: 168
Removed 1 edges connected to node -630016627
New file created: graph_data/graph_100_gabriel_without_node_20250409_214328.json
Done!
```

# Transportation Graph Generator

This project analyzes transportation networks in Izmir Institute of Technology using graph theory and algorithms.

## Graph Generator Tool

The `GraphGenerator` class is a standalone tool that:

1. Generates 25 random nodes based on population centers in Izmir Bay
2. Saves these nodes to a JSON file (`nodes.json`)
3. Creates four different transportation graphs using distinct connectivity strategies:
   - Complete Graph: Every node is connected to every other node
   - Gabriel Graph: Two points are connected if their diametric circle contains no other points
   - Delaunay Triangulation: Creates a triangulation that maximizes the minimum angle of all triangles
   - K-Nearest Neighbors: Each node is connected to its 5 closest neighbors

## Clustering Example Tool

The `ClusteringExample` class is a standalone tool that:

1. Loads the previously generated graphs (Complete, Gabriel, Delaunay, K-Nearest Neighbors)
2. Applies three different clustering algorithms to each graph:
   - Leiden: A community detection algorithm optimized for modularity
   - Spectral: Uses eigenvectors of the graph Laplacian to find communities
   - MVAGC: Multi-view adaptive graph clustering algorithm
3. Visualizes the clustering results for each combination (12 visualizations total)
4. Performs transportation cost analysis and saves detailed metrics
5. Generates histograms of key metrics for each clustering

## How to Run

### Graph Generation

```bash
# Compile the project
mvn compile

# Run the GraphGenerator class
mvn exec:java -Dexec.mainClass="com.izmir.GraphGenerator"
```

### Clustering Analysis

```bash
# Run the ClusteringExample class (after running GraphGenerator)
mvn exec:java -Dexec.mainClass="com.izmir.ClusteringExample"
```

## Output

The tools generate:

1. `nodes.json` - A JSON file containing the generated nodes
2. Visualization of each graph type and clustering result
3. Persistent storage of graphs and analysis results
4. Transportation cost analysis with metrics saved as CSV files
5. Histograms of key performance metrics for each clustering solution

## Configuration

You can modify various parameters in both classes to customize the analysis:

- `NODE_COUNT` - Number of nodes to generate (default: 25)
- `MAX_CLUSTERS` - Maximum number of clusters to detect (default: 5 for clustering)
- `MIN_CLUSTER_SIZE` - Minimum size for a valid cluster (default: 3)
- `MAX_CLUSTER_SIZE` - Maximum size for a valid cluster (default: 10)

## Note

These tools are independent of the main application (`App.java`) and can be run separately to generate, analyze, and visualize transportation graphs and clustering solutions.
