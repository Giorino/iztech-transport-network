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
