# Workspace Guide for Transportation Network Analysis

This guide provides an overview of the project structure and guidance on where to make changes when extending the codebase.

## Key Files and Their Purposes

| File | Purpose | When to Modify |
|------|---------|----------------|
| `src/main/java/com/izmir/App.java` | Main application entry point and configuration | Modify to change configuration parameters or execution flow |
| `src/main/java/com/izmir/transportation/GraphConstructionService.java` | Handles graph construction using different strategies | Modify when adding new graph construction strategies |
| `src/main/java/com/izmir/transportation/ClusteringService.java` | Applies clustering algorithms to detect communities | Modify when adding new clustering algorithms |
| `src/main/java/com/izmir/transportation/IzmirBayGraph.java` | Generates random points based on population centers | Modify to change point generation logic |
| `src/main/java/com/izmir/transportation/TransportationGraph.java` | Core graph structure with visualization capabilities | Modify when changing core graph functionality |
| `src/main/java/com/izmir/transportation/persistence/` | Contains classes for graph persistence | Modify when changing persistence mechanism |
| `src/main/java/com/izmir/transportation/helper/strategy/` | Contains graph construction strategies | Add new strategies here |
| `src/main/java/com/izmir/transportation/helper/clustering/` | Contains clustering algorithms | Add new clustering algorithms here |
| `src/main/java/com/izmir/transportation/cost/` | Contains transportation cost analysis | Modify to change cost analysis logic |

## Common Tasks

### Adding a New Graph Construction Strategy

1. Create a new class in `src/main/java/com/izmir/transportation/helper/strategy/` that implements `GraphConnectivityStrategy`
2. Add your strategy to the `GraphStrategy` enum in `GraphConstructionService.java`:
   ```java
   public enum GraphStrategy {
       COMPLETE("complete"),
       K_NEAREST_NEIGHBORS("k_nearest_neighbors"),
       GABRIEL("gabriel"),
       DELAUNAY("delaunay"),
       YOUR_STRATEGY("your_strategy_name");
       // ...
   }
   ```
3. Update the `createStrategy` method in `GraphConstructionService.java` to handle your new strategy:
   ```java
   private GraphConnectivityStrategy createStrategy(GraphStrategy strategyType, int kValue) {
       switch (strategyType) {
           // ...
           case YOUR_STRATEGY:
               return new YourStrategyClass(parameters);
           // ...
       }
   }
   ```

### Adding a New Clustering Algorithm

1. Create a new class in `src/main/java/com/izmir/transportation/helper/clustering/` that extends `GraphClusteringAlgorithm`
2. Implement the `findCommunities` method to detect communities in the graph
3. Add your algorithm to the `ClusteringAlgorithm` enum in `ClusteringService.java`:
   ```java
   public enum ClusteringAlgorithm {
       LEIDEN("leiden"),
       SPECTRAL("spectral"),
       YOUR_ALGORITHM("your_algorithm_name");
       // ...
   }
   ```
4. Update the `performClustering` method in `ClusteringService.java` to handle your new algorithm:
   ```java
   public Map<Integer, List<Node>> performClustering(
           TransportationGraph graph, 
           ClusteringAlgorithm algorithm,
           boolean visualize) {
       // ...
       if (algorithm == ClusteringAlgorithm.YOUR_ALGORITHM) {
           YourAlgorithm yourAlgorithm = new YourAlgorithm(graph);
           communities = yourAlgorithm.detectCommunities();
       }
       // ...
   }
   ```

### Modifying Graph Persistence

If you need to change how graphs are saved or loaded:

1. Modify the `GraphPersistenceService.java` class in the `src/main/java/com/izmir/transportation/persistence/` package
2. Key methods to consider:
   - `saveGraph`: Controls how graphs are saved to files
   - `loadGraph`: Controls how graphs are loaded from files
   - `generateFilename`: Controls how filenames are generated

### Changing Configuration

To change application configuration:

1. Open `src/main/java/com/izmir/App.java`
2. Modify the configuration constants at the top of the class:
   ```java
   private static final int NODE_COUNT = 1000; // Change number of nodes
   private static final GraphConstructionService.GraphStrategy GRAPH_STRATEGY = 
           GraphConstructionService.GraphStrategy.COMPLETE; // Change strategy
   private static final ClusteringService.ClusteringAlgorithm CLUSTERING_ALGORITHM = 
           ClusteringService.ClusteringAlgorithm.LEIDEN; // Change algorithm
   ```

## Visualization and Output

The application generates several types of visualizations:

1. **Graph Visualization**: Shows the constructed graph with nodes and edges
   - Controlled by `TransportationGraph.visualizeGraph()`
   - Modify this method to change graph visualization

2. **Community Visualization**: Shows detected communities with different colors
   - Controlled by `TransportationGraph.visualizeCommunities()`
   - Modify this method to change community visualization

3. **Cost Analysis Output**: CSV files with transportation cost analysis
   - Controlled by classes in the `cost` package
   - Modify `OptimizedTransportationCostAnalyzer` to change cost analysis output

## Project Flow

The application follows this execution flow:

1. `App.main()` -> Entry point
2. -> `IzmirBayGraph.generatePoints()` -> Generates random points
3. -> `GraphConstructionService.createGraph()` -> Constructs/loads the graph
   - Checks if a saved graph exists and loads it if available
   - Otherwise constructs a new graph using the specified strategy
4. -> `ClusteringService.performClustering()` -> Applies clustering algorithm
5. -> `TransportationCostAnalysis.analyzeCosts()` -> Analyzes transportation costs

When extending the project, ensure your changes fit into this flow appropriately. 