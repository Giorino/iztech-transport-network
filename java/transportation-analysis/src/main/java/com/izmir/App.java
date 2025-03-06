package com.izmir;

import com.izmir.transportation.ClusteringService;
import com.izmir.transportation.GraphConstructionService;
import com.izmir.transportation.IzmirBayGraph;
import com.izmir.transportation.TransportationGraph;

import java.util.List;
import java.util.logging.Logger;

import org.locationtech.jts.geom.Point;

/**
 * Main application class for the Iztech Transportation Analysis project.
 * 1. Generation of random vertices based on population centers (IzmirBayGraph)
 * 2. Construction of a road network connecting these vertices using a specified strategy
 * 3. Analysis and visualization of the transportation network graph using specified clustering algorithm
 * 
 * @author yagizugurveren
 */
public class App 
{
    static Logger LOGGER = Logger.getLogger(App.class.getName());
    
    // Configuration properties
    private static final int NODE_COUNT = 3000; // Number of nodes to generate
    private static final GraphConstructionService.GraphStrategy GRAPH_STRATEGY = 
            GraphConstructionService.GraphStrategy.COMPLETE; // Graph construction strategy
    private static final int K_VALUE = 5; // K value for K-nearest neighbors strategy
    private static final ClusteringService.ClusteringAlgorithm CLUSTERING_ALGORITHM = 
            ClusteringService.ClusteringAlgorithm.LEIDEN; // Clustering algorithm
    private static final boolean USE_PARALLEL = true; // Whether to use parallel processing
    private static final boolean VISUALIZE_GRAPH = true; // Whether to visualize the graph
    private static final boolean VISUALIZE_CLUSTERS = true; // Whether to visualize clusters
    private static final boolean SAVE_GRAPH = true; // Whether to save the graph for future use

    public static void main( String[] args )
    {
        try {
            LOGGER.info("Starting Iztech Transportation Analysis...");
            
            // Step 1: Generate random points using IzmirBayGraph
            LOGGER.info("Step 1: Generating random vertices...");
            List<Point> points = IzmirBayGraph.generatePoints(NODE_COUNT);
            LOGGER.info("Generated " + points.size() + " points.");
            
            // Step 2: Create transportation graph using specified strategy
            LOGGER.info("Step 2: Creating transportation graph using " + GRAPH_STRATEGY + " strategy...");
            GraphConstructionService graphService = new GraphConstructionService();
            TransportationGraph graph = graphService.createGraph(
                points, GRAPH_STRATEGY, K_VALUE, USE_PARALLEL, VISUALIZE_GRAPH, SAVE_GRAPH);
            
            // Step 3: Perform clustering using specified algorithm
            LOGGER.info("Step 3: Performing clustering using " + CLUSTERING_ALGORITHM + " algorithm...");
            ClusteringService clusteringService = new ClusteringService();
            clusteringService.performClustering(graph, CLUSTERING_ALGORITHM, VISUALIZE_CLUSTERS);
            
            LOGGER.info("Iztech Transportation Analysis completed successfully.");
        } catch (Exception e) {
            LOGGER.severe("Error during analysis: " + e.getMessage());
            e.printStackTrace();
        }
    }
}
