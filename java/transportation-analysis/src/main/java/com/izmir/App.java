package com.izmir;

import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Properties;
import java.util.logging.Logger;

import org.locationtech.jts.geom.Point;

import com.izmir.transportation.ClusteringService;
import com.izmir.transportation.GraphConstructionService;
import com.izmir.transportation.IzmirBayGraph;
import com.izmir.transportation.TransportationGraph;
import com.izmir.transportation.helper.Node;
import com.izmir.transportation.helper.clustering.GirvanNewmanClustering;
import com.izmir.transportation.helper.clustering.InfomapCommunityDetection;
import com.izmir.transportation.helper.clustering.MvAGCClustering;
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
            GraphConstructionService.GraphStrategy.COMPLETE; // Using Gabriel graph
    private static final int K_VALUE = 50; // K value for K-nearest neighbors strategy
    private static final ClusteringService.ClusteringAlgorithm CLUSTERING_ALGORITHM = 
            ClusteringService.ClusteringAlgorithm.LEIDEN; // Using Infomap algorithm
            // Options: LEIDEN, SPECTRAL, GIRVAN_NEWMAN, INFOMAP, MVAGC
    private static final boolean USE_PARALLEL = true; // Whether to use parallel processing
    private static final boolean VISUALIZE_GRAPH = true; // Whether to visualize the graph
    private static final boolean VISUALIZE_CLUSTERS = true; // Whether to visualize clusters
    private static final boolean SAVE_GRAPH = true; // Whether to save the graph for future use
    
    // Clustering configuration - now loaded from properties file
    private static int MAX_CLUSTERS = 40; // Increased from 12 back to 40 to ensure each community can be within 50-node limit
    private static final double COMMUNITY_SCALING_FACTOR = 0.5; // Increased to create more balanced communities
    private static final boolean USE_ADAPTIVE_RESOLUTION = true; // Adaptive resolution based on network size
    private static int MIN_CLUSTER_SIZE = 40; // Increased from 10 to 40 for efficient bus utilization (minimum passengers)
    private static int MAX_CLUSTER_SIZE = 50; // Strict maximum - bus capacity
    private static final double GEOGRAPHIC_WEIGHT = 0.7; // Higher weight for geographic proximity
    private static final double MAX_COMMUNITY_DIAMETER = 5000.0; // Maximum allowed diameter for a community in meters
   
    // Spectral clustering specific configuration
    private static SpectralClusteringConfig SPECTRAL_CONFIG; // Will be initialized with properties
    
    // Girvan-Newman specific configuration
    private static final boolean USE_MODULARITY_MAXIMIZATION = true; // Disable modularity maximization to force finding more communities
    
    // Additional Girvan-Newman specific configuration
    private static final int GN_MAX_ITERATIONS = 100; // Increased max iterations to ensure enough communities
    private static final boolean GN_EARLY_STOP = true; // Disable early stopping to force more iterations
    
    // Infomap specific configuration
    private static final int INFOMAP_MAX_ITERATIONS = 1000; // Increased for better convergence
    private static final double INFOMAP_TOLERANCE = 1e-8; // Tighter convergence tolerance
    private static final boolean INFOMAP_FORCE_MAX_CLUSTERS = true; // Force merging to max clusters
    private static final double INFOMAP_RESOLUTION_PARAMETER = 1.5; // Higher value for more compact communities
    private static final double INFOMAP_GEOGRAPHIC_IMPORTANCE = 0.6; // 60% weight to geographic proximity
    private static final double INFOMAP_MAX_GEOGRAPHIC_DISTANCE = 8000.0; // Max distance in meters
    
    // MvAGC specific configuration
    private static final int MVAGC_NUM_ANCHORS = 500; // Further increased number of anchor nodes for better representation
    private static final int MVAGC_FILTER_ORDER = 4; // Increased filter order for more effective smoothing
    private static final double MVAGC_ALPHA = 8.0; // Balanced value for alpha - too high can over-emphasize structure
    private static final double MVAGC_SAMPLING_POWER = 0.3; // Further reduced to improve anchor diversity in complete graphs

    public static void main( String[] args )
    {
        try {
            LOGGER.info("Starting Iztech Transportation Analysis...");
            
            // Load properties from application.properties
            loadProperties();
            
            LOGGER.info("Configuration loaded: MAX_CLUSTERS=" + MAX_CLUSTERS + 
                      ", MIN_CLUSTER_SIZE=" + MIN_CLUSTER_SIZE + 
                      ", MAX_CLUSTER_SIZE=" + MAX_CLUSTER_SIZE);
            
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
            } else if (CLUSTERING_ALGORITHM == ClusteringService.ClusteringAlgorithm.GIRVAN_NEWMAN) {
                // Girvan-Newman algorithm
                LOGGER.info("Step 3: Performing Girvan-Newman clustering");
                LOGGER.info("Target Clusters: " + MAX_CLUSTERS + ", Min Cluster Size: " + MIN_CLUSTER_SIZE + ", Max Cluster Size: " + MAX_CLUSTER_SIZE);
                LOGGER.info("Max Iterations: " + GN_MAX_ITERATIONS + ", Early Stop: " + GN_EARLY_STOP);
                LOGGER.info("Using Modularity Maximization: " + USE_MODULARITY_MAXIMIZATION);
                
                // Create and configure Girvan-Newman algorithm directly
                GirvanNewmanClustering gnAlgorithm = new GirvanNewmanClustering(graph);
                gnAlgorithm.setTargetCommunityCount(MAX_CLUSTERS)
                         .setMaxIterations(GN_MAX_ITERATIONS)
                         .setEarlyStop(GN_EARLY_STOP)
                         .setMinCommunitySize(MIN_CLUSTER_SIZE)
                         .setUseModularityMaximization(USE_MODULARITY_MAXIMIZATION);
                
                // Run the algorithm
                Map<Integer, List<Node>> communities = gnAlgorithm.detectCommunities();
                
                // Visualize if requested
                if (VISUALIZE_CLUSTERS && !communities.isEmpty()) {
                    List<List<Node>> communityList = new ArrayList<>(communities.values());
                    graph.visualizeCommunities(communityList, CLUSTERING_ALGORITHM.toString(), false);
                    //graph.saveCommunityData(communities, CLUSTERING_ALGORITHM.toString());
                }
                
                // Perform transportation cost analysis
                LOGGER.info("Performing transportation cost analysis...");
                new com.izmir.transportation.cost.TransportationCostAnalysis().analyzeCosts(graph, communities);
            } else if (CLUSTERING_ALGORITHM == ClusteringService.ClusteringAlgorithm.INFOMAP) {
                // Infomap algorithm
                LOGGER.info("Step 3: Performing Infomap clustering");
                LOGGER.info("Target Clusters: " + MAX_CLUSTERS + ", Min Cluster Size: " + MIN_CLUSTER_SIZE + ", Max Cluster Size: " + MAX_CLUSTER_SIZE);
                LOGGER.info("Max Iterations: " + INFOMAP_MAX_ITERATIONS + ", Tolerance: " + INFOMAP_TOLERANCE);
                LOGGER.info("Force Max Clusters: " + INFOMAP_FORCE_MAX_CLUSTERS + " (Letting algorithm find natural communities)");
                
                ClusteringService clusteringService = new ClusteringService();
                clusteringService.setMaxClusters(MAX_CLUSTERS)
                                .setMinCommunitySize(MIN_CLUSTER_SIZE);
                
                // Use Infomap-specific configuration
                InfomapCommunityDetection infomapAlgorithm = new InfomapCommunityDetection(graph);
                infomapAlgorithm.setMaxClusters(MAX_CLUSTERS)
                               .setMinClusterSize(MIN_CLUSTER_SIZE)
                               .setMaxIterations(INFOMAP_MAX_ITERATIONS)
                               .setTolerance((float)INFOMAP_TOLERANCE)
                               .setForceMaxClusters(INFOMAP_FORCE_MAX_CLUSTERS)
                               .setUseHierarchicalRefinement(false) // Disable hierarchical refinement to prevent fragmentation
                               .setTeleportationProbability(0.05) // Significantly reduced to prevent "jumping" across the network
                               .setGeographicImportance(INFOMAP_GEOGRAPHIC_IMPORTANCE) // Add geographic importance
                               .setMaxGeographicDistance(INFOMAP_MAX_GEOGRAPHIC_DISTANCE); // Set max geographic distance
                
                Map<Integer, List<Node>> communities = infomapAlgorithm.detectCommunities();
                
                // Visualize the communities
                if (VISUALIZE_CLUSTERS && !communities.isEmpty()) {
                    List<List<Node>> communityList = new ArrayList<>(communities.values());
                    graph.visualizeCommunities(communityList, CLUSTERING_ALGORITHM.toString(), false); // Hide community 0
                    //graph.saveCommunityData(communities, CLUSTERING_ALGORITHM.toString());
                }
                
                // Perform transportation cost analysis
                LOGGER.info("Performing transportation cost analysis...");
                new com.izmir.transportation.cost.TransportationCostAnalysis().analyzeCosts(graph, communities);
            } else if (CLUSTERING_ALGORITHM == ClusteringService.ClusteringAlgorithm.MVAGC) {
                // MvAGC algorithm
                LOGGER.info("Step 3: Performing MvAGC clustering");
                LOGGER.info("Target Clusters: " + MAX_CLUSTERS + ", Min Cluster Size: " + MIN_CLUSTER_SIZE + ", Max Cluster Size: " + MAX_CLUSTER_SIZE);
                LOGGER.info("Anchor Nodes: " + MVAGC_NUM_ANCHORS);
                LOGGER.info("Filter Order: " + MVAGC_FILTER_ORDER + ", Alpha: " + MVAGC_ALPHA);
                LOGGER.info("Importance Sampling Power: " + MVAGC_SAMPLING_POWER);
                
                // Create and configure MvAGC algorithm directly
                MvAGCClustering mvagcAlgorithm = new MvAGCClustering(graph);
                mvagcAlgorithm.setNumClusters(MAX_CLUSTERS)
                             .setNumAnchors(MVAGC_NUM_ANCHORS)
                             .setFilterOrder(MVAGC_FILTER_ORDER)
                             .setAlpha(MVAGC_ALPHA)
                             .setImportanceSamplingPower(MVAGC_SAMPLING_POWER);
                
                // Run the algorithm
                Map<Integer, List<Node>> communities = mvagcAlgorithm.detectCommunities();
                
                // Visualize if requested
                if (VISUALIZE_CLUSTERS && !communities.isEmpty()) {
                    List<List<Node>> communityList = new ArrayList<>(communities.values());
                    graph.visualizeCommunities(communityList, CLUSTERING_ALGORITHM.toString(), false);
                    //graph.saveCommunityData(communities, CLUSTERING_ALGORITHM.toString());
                }
                
                // Perform transportation cost analysis
                LOGGER.info("Performing transportation cost analysis...");
                new com.izmir.transportation.cost.TransportationCostAnalysis().analyzeCosts(graph, communities);
            } else {
                // Leiden algorithm
                LOGGER.info("Step 3: Performing clustering using " + CLUSTERING_ALGORITHM);
                LOGGER.info("Target Clusters: " + MAX_CLUSTERS + ", Min Cluster Size: " + MIN_CLUSTER_SIZE + ", Max Cluster Size: " + MAX_CLUSTER_SIZE);
                
                ClusteringService clusteringService = new ClusteringService();
                clusteringService.setMaxClusters(MAX_CLUSTERS)
                                .setCommunityScalingFactor(COMMUNITY_SCALING_FACTOR)
                                .setAdaptiveResolution(USE_ADAPTIVE_RESOLUTION)
                                .setMinCommunitySize(MIN_CLUSTER_SIZE);
                
                clusteringService.performClustering(graph, CLUSTERING_ALGORITHM, VISUALIZE_CLUSTERS);
            }
            
            LOGGER.info("Iztech Transportation Analysis completed successfully.");
        } catch (Exception e) {
            LOGGER.severe("Error during analysis: " + e.getMessage());
            e.printStackTrace();
        }
    }
    
    /**
     * Loads configuration from application.properties file
     */
    private static void loadProperties() {
        Properties properties = new Properties();
        try (InputStream input = App.class.getClassLoader().getResourceAsStream("application.properties")) {
            if (input == null) {
                LOGGER.warning("Unable to find application.properties, using default values");
                return;
            }
            
            // Load properties file
            properties.load(input);
            
            // Load clustering configuration
            MAX_CLUSTERS = Integer.parseInt(properties.getProperty("transportation.services.count", "40"));
            MIN_CLUSTER_SIZE = Integer.parseInt(properties.getProperty("transportation.bus.min.seats", "10"));
            MAX_CLUSTER_SIZE = Integer.parseInt(properties.getProperty("transportation.bus.max.seats", 
                                                properties.getProperty("transportation.bus.capacity", "50")));
            
            // Initialize spectral clustering config
            SPECTRAL_CONFIG = new SpectralClusteringConfig()
                    .setNumberOfClusters(MAX_CLUSTERS)
                    .setMinCommunitySize(MIN_CLUSTER_SIZE)
                    .setPreventSingletons(true)
                    .setSigma(300)
                    .setGeographicWeight(0.8);
            
        } catch (IOException ex) {
            LOGGER.warning("Error loading properties: " + ex.getMessage());
        } catch (NumberFormatException ex) {
            LOGGER.warning("Error parsing property value: " + ex.getMessage());
        }
    }
}
