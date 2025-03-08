package com.izmir;

import java.util.List;
import java.util.logging.Logger;

import org.locationtech.jts.geom.Point;

import com.izmir.transportation.ClusteringService;
import com.izmir.transportation.GraphConstructionService;
import com.izmir.transportation.IzmirBayGraph;
import com.izmir.transportation.TransportationGraph;
import com.izmir.transportation.helper.clustering.SpectralClusteringConfig;

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
    private static final int NODE_COUNT = 2000; // Number of nodes to generate
    private static final GraphConstructionService.GraphStrategy GRAPH_STRATEGY = 
            GraphConstructionService.GraphStrategy.COMPLETE; // Graph construction strategy
    private static final int K_VALUE = 50; // K value for K-nearest neighbors strategy
    private static final ClusteringService.ClusteringAlgorithm CLUSTERING_ALGORITHM = 
            ClusteringService.ClusteringAlgorithm.LEIDEN; // Clustering algorithm
    private static final boolean USE_PARALLEL = true; // Whether to use parallel processing
    private static final boolean VISUALIZE_GRAPH = true; // Whether to visualize the graph
    private static final boolean VISUALIZE_CLUSTERS = true; // Whether to visualize clusters
    private static final boolean SAVE_GRAPH = true; // Whether to save the graph for future use
    
    // Clustering configuration (increased from defaults to detect more communities)
    private static final int MAX_CLUSTERS = 30; // Maximum number of clusters to detect
    private static final double COMMUNITY_SCALING_FACTOR = 0.1; // Higher value = more communities
    private static final boolean USE_ADAPTIVE_RESOLUTION = true; // Adaptive resolution based on network size
    private static final int MIN_COMMUNITY_SIZE = 5; // Minimum size for communities (eliminates singletons)
    
    // Spectral clustering specific configuration
    private static final SpectralClusteringConfig SPECTRAL_CONFIG = new SpectralClusteringConfig()
            .setNumberOfClusters(MAX_CLUSTERS)
            .setMinCommunitySize(MIN_COMMUNITY_SIZE)
            .setPreventSingletons(true)
            .setSigma(100)  // Higher value for better similarity detection
            .setGeographicWeight(0.3); // Higher weight for geographic proximity

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
            if (CLUSTERING_ALGORITHM == ClusteringService.ClusteringAlgorithm.SPECTRAL) {
                LOGGER.info("Step 3: Performing Spectral Clustering with advanced configuration...");
                LOGGER.info("Max Clusters: " + SPECTRAL_CONFIG.getNumberOfClusters() + 
                          ", Min Size: " + SPECTRAL_CONFIG.getMinCommunitySize() + 
                          ", Prevent Singletons: " + SPECTRAL_CONFIG.isPreventSingletons());
                
                ClusteringService clusteringService = new ClusteringService();
                clusteringService.setSpectralConfig(SPECTRAL_CONFIG);
                clusteringService.performClustering(graph, CLUSTERING_ALGORITHM, VISUALIZE_CLUSTERS);
            } else {
                // Leiden algorithm
                LOGGER.info("Step 3: Performing clustering using " + CLUSTERING_ALGORITHM);
                
                ClusteringService clusteringService = new ClusteringService();
                clusteringService.setMaxClusters(MAX_CLUSTERS)
                                .setCommunityScalingFactor(COMMUNITY_SCALING_FACTOR)
                                .setAdaptiveResolution(USE_ADAPTIVE_RESOLUTION)
                                .setMinCommunitySize(MIN_COMMUNITY_SIZE);
                
                clusteringService.performClustering(graph, CLUSTERING_ALGORITHM, VISUALIZE_CLUSTERS);
            }
            
            LOGGER.info("Iztech Transportation Analysis completed successfully.");
        } catch (Exception e) {
            LOGGER.severe("Error during analysis: " + e.getMessage());
            e.printStackTrace();
        }
    }
}
