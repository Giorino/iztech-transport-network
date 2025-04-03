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
import com.izmir.transportation.cost.ClusterMetrics;
import com.izmir.transportation.cost.TransportationCostAnalysis;
import com.izmir.transportation.helper.Node;
import com.izmir.transportation.helper.clustering.GirvanNewmanClustering;
import com.izmir.transportation.helper.clustering.InfomapCommunityDetection;
import com.izmir.transportation.helper.clustering.MvAGCClustering;
import com.izmir.transportation.helper.clustering.SpectralClusteringConfig;
import com.izmir.visualization.HistogramService;

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
            GraphConstructionService.GraphStrategy.COMPLETE; // Using Greedy Spanner graph
    private static final int K_VALUE = 3; // K value for spanner's stretch factor (2k-1)
    private static final ClusteringService.ClusteringAlgorithm CLUSTERING_ALGORITHM = 
            ClusteringService.ClusteringAlgorithm.MVAGC; // Using SPECTRAL algorithm
            // Options: LEIDEN, SPECTRAL, GIRVAN_NEWMAN, INFOMAP, MVAGC
    private static final boolean USE_PARALLEL = true; // Whether to use parallel processing
    private static final boolean VISUALIZE_GRAPH = true; // Whether to visualize the graph
    private static final boolean VISUALIZE_CLUSTERS = true; // Whether to visualize clusters
    private static final boolean SAVE_GRAPH = true; // Whether to save the graph for future use
    
    // Clustering configuration - now loaded from properties file
    private static int MAX_CLUSTERS = 55; // Increased from 12 back to 40 to ensure each community can be within 50-node limit
    private static final double COMMUNITY_SCALING_FACTOR = 0.5; // Increased to create more balanced communities
    private static final boolean USE_ADAPTIVE_RESOLUTION = true; // Adaptive resolution based on network size
    private static int MIN_CLUSTER_SIZE = 30; // Minimum efficient bus occupancy
    private static int MAX_CLUSTER_SIZE = 45; // Maximum bus capacity (reduced from 50 for better balance)
    private static final double GEOGRAPHIC_WEIGHT = 0.9; // Increased from 0.7 for more geographically cohesive communities
    private static final double MAX_COMMUNITY_DIAMETER = 20000.0; // Maximum allowed diameter for a community in meters (Increased significantly from 2500.0m to disable problematic splitting)
   
    // Spectral clustering specific configuration - Initialize with default values to prevent NullPointerException
    private static SpectralClusteringConfig SPECTRAL_CONFIG = new SpectralClusteringConfig()
            .setNumberOfClusters(MAX_CLUSTERS)
            .setMinCommunitySize(MIN_CLUSTER_SIZE)
            .setPreventSingletons(true)
            .setSigma(100)
            .setGeographicWeight(0.9)
            .setMaxClusterSize(MAX_CLUSTER_SIZE)
            .setForceNumClusters(false)
            .setMaxCommunityDiameter(MAX_COMMUNITY_DIAMETER);
    
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
            
            TransportationGraph graph;
            
            // If using Greedy Spanner, first create a complete graph for optimization
            if (GRAPH_STRATEGY == GraphConstructionService.GraphStrategy.GREEDY_SPANNER) {
                LOGGER.info("Creating complete graph first for Greedy Spanner optimization...");
                // Create a complete graph but don't visualize or save it
                TransportationGraph completeGraph = graphService.createGraph(
                    points, GraphConstructionService.GraphStrategy.COMPLETE, 
                    K_VALUE, USE_PARALLEL, false, false);
                
                int expectedEdges = (points.size() * (points.size() - 1)) / 2;
                int actualEdges = completeGraph.getEdgeCount();
                LOGGER.info("Complete graph created with " + actualEdges + " edges out of expected " + expectedEdges);
                
                // Check if the complete graph was successfully created with all expected edges
                if (actualEdges > 0 && completeGraph.isCompleteGraph()) {
                    LOGGER.info("Now creating optimized Greedy Spanner...");
                    
                    // Now create the spanner using the complete graph
                    graph = graphService.createGraph(
                        points, GRAPH_STRATEGY, K_VALUE, USE_PARALLEL, VISUALIZE_GRAPH, SAVE_GRAPH, completeGraph);
                } else {
                    LOGGER.warning("Complete graph is incomplete or empty. Falling back to standard approach.");
                    graph = graphService.createGraph(
                        points, GRAPH_STRATEGY, K_VALUE, USE_PARALLEL, VISUALIZE_GRAPH, SAVE_GRAPH);
                }
            } else {
                // For other strategies, create the graph directly
                graph = graphService.createGraph(
                    points, GRAPH_STRATEGY, K_VALUE, USE_PARALLEL, VISUALIZE_GRAPH, SAVE_GRAPH);
            }
            
            // Remove isolated nodes before clustering
            graph.removeIsolatedNodes();
            
            // Instantiate Histogram Service
            HistogramService histogramService = new HistogramService();
            Map<Integer, ClusterMetrics> clusterMetrics = null;
            
            // Step 3: Perform clustering using specified algorithm
            if (CLUSTERING_ALGORITHM == ClusteringService.ClusteringAlgorithm.SPECTRAL) {
                LOGGER.info("Step 3: Performing Spectral Clustering with advanced configuration...");
                LOGGER.info("Max Clusters: " + SPECTRAL_CONFIG.getNumberOfClusters() + 
                          ", Min Size: " + SPECTRAL_CONFIG.getMinCommunitySize() + 
                          ", Prevent Singletons: " + SPECTRAL_CONFIG.isPreventSingletons());
                
                ClusteringService clusteringService = new ClusteringService();
                clusteringService.setSpectralConfig(SPECTRAL_CONFIG);
                
                // Capture the communities map from performClustering
                Map<Integer, List<Node>> communities = clusteringService.performClustering(
                    graph, 
                    CLUSTERING_ALGORITHM, 
                    VISUALIZE_CLUSTERS
                );
                
                // Perform transportation cost analysis after getting communities
                LOGGER.info("Performing transportation cost analysis for Spectral...");
                clusterMetrics = TransportationCostAnalysis.analyzeCosts(graph, communities); // Capture metrics
                
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
                clusterMetrics = TransportationCostAnalysis.analyzeCosts(graph, communities);
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
                clusterMetrics = TransportationCostAnalysis.analyzeCosts(graph, communities);
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
                             .setImportanceSamplingPower(MVAGC_SAMPLING_POWER)
                             .setMinClusterSize(MIN_CLUSTER_SIZE)
                             .setMaxClusterSize(MAX_CLUSTER_SIZE)
                             .setForceMinClusters(true);
                
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
                clusterMetrics = TransportationCostAnalysis.analyzeCosts(graph, communities);
            } else {
                // Leiden algorithm
                LOGGER.info("Step 3: Performing clustering using " + CLUSTERING_ALGORITHM);
                LOGGER.info("Target Clusters: " + MAX_CLUSTERS + ", Min Cluster Size: " + MIN_CLUSTER_SIZE + ", Max Cluster Size: " + MAX_CLUSTER_SIZE);
                
                ClusteringService clusteringService = new ClusteringService();
                clusteringService.setMaxClusters(MAX_CLUSTERS)
                                .setCommunityScalingFactor(COMMUNITY_SCALING_FACTOR)
                                .setAdaptiveResolution(USE_ADAPTIVE_RESOLUTION)
                                .setMinCommunitySize(MIN_CLUSTER_SIZE);
                
                // Capture the communities map from performClustering
                Map<Integer, List<Node>> communities = clusteringService.performClustering(
                    graph, 
                    CLUSTERING_ALGORITHM, 
                    VISUALIZE_CLUSTERS
                );
                
                // Perform transportation cost analysis after getting communities
                LOGGER.info("Performing transportation cost analysis for Leiden...");
                clusterMetrics = TransportationCostAnalysis.analyzeCosts(graph, communities); // Capture metrics
            }
            
            // Step 4: Generate Histograms if metrics were calculated
            if (clusterMetrics != null) {
                histogramService.generateHistograms(clusterMetrics, CLUSTERING_ALGORITHM.toString());
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
            MAX_CLUSTERS = Integer.parseInt(properties.getProperty("transportation.services.count", "55"));
            MIN_CLUSTER_SIZE = Integer.parseInt(properties.getProperty("transportation.bus.min.seats", "30"));
            MAX_CLUSTER_SIZE = Integer.parseInt(properties.getProperty("transportation.bus.max.seats", 
                                                properties.getProperty("transportation.bus.capacity", "45")));
            
            // Update spectral clustering config with loaded properties
            SPECTRAL_CONFIG = new SpectralClusteringConfig()
                    .setNumberOfClusters(MAX_CLUSTERS)
                    .setMinCommunitySize(MIN_CLUSTER_SIZE)
                    .setPreventSingletons(true)
                    .setSigma(100)
                    .setGeographicWeight(0.9)
                    .setMaxClusterSize(MAX_CLUSTER_SIZE)
                    .setForceNumClusters(false)
                    .setMaxCommunityDiameter(MAX_COMMUNITY_DIAMETER);
            
        } catch (IOException ex) {
            LOGGER.warning("Error loading properties: " + ex.getMessage());
        } catch (NumberFormatException ex) {
            LOGGER.warning("Error parsing property value: " + ex.getMessage());
        }
    }
}
