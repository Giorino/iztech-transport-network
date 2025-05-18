package com.izmir;

import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Properties;
import java.util.Set;
import java.util.logging.Logger;

import org.jgrapht.graph.DefaultWeightedEdge;
import org.locationtech.jts.geom.Point;

import com.izmir.transportation.ClusteringService;
import com.izmir.transportation.GraphConstructionService;
import com.izmir.transportation.IzmirBayGraph;
import com.izmir.transportation.OutlierDetectionService;
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
            GraphConstructionService.GraphStrategy.GABRIEL; // Using Greedy Spanner graph
    private static final int K_VALUE = 30; // K value for spanner's stretch factor (2k-1)
    private static final ClusteringService.ClusteringAlgorithm CLUSTERING_ALGORITHM = 
            ClusteringService.ClusteringAlgorithm.LEIDEN; // Using SPECTRAL algorithm
            // Options: LEIDEN, SPECTRAL, GIRVAN_NEWMAN, INFOMAP, MVAGC
    private static final boolean USE_PARALLEL = true; // Whether to use parallel processing
    private static final boolean VISUALIZE_GRAPH = true; // Whether to visualize the graph
    private static final boolean VISUALIZE_CLUSTERS = true; // Whether to visualize clusters
    private static final boolean SAVE_GRAPH = true; // Whether to save the graph for future use
    
    // Outlier detection configuration
    private static final boolean APPLY_OUTLIER_DETECTION = true; // Whether to apply outlier detection
    private static final OutlierDetectionService.OutlierAlgorithm OUTLIER_ALGORITHM = 
            OutlierDetectionService.OutlierAlgorithm.KNN_DISTANCE; // Default outlier detection algorithm
    private static final double OUTLIER_THRESHOLD = 3; // Number of standard deviations to consider as outlier
    private static final int OUTLIER_K_VALUE = 5; // K value for KNN_DISTANCE algorithm
    private static final boolean VISUALIZE_OUTLIERS = true; // Whether to visualize outliers
    private static final boolean VISUALIZE_OUTLIER_LEGEND = true; // Whether to show the legend in outlier visualization
    private static final int OUTLIER_MAX_THREADS = Runtime.getRuntime().availableProcessors(); // Use all available processors
    
    // Clustering configuration - now loaded from properties file
    private static int MAX_CLUSTERS = 55; // Increased from 12 back to 40 to ensure each community can be within 50-node limit
    private static final double COMMUNITY_SCALING_FACTOR = 0.5; // Increased to create more balanced communities
    private static final boolean USE_ADAPTIVE_RESOLUTION = true; // Adaptive resolution based on network size
    private static int MIN_CLUSTER_SIZE = 10; // Minimum efficient bus occupancy
    private static int MAX_CLUSTER_SIZE = 45; // Maximum bus capacity (reduced from 50 for better balance)
    private static final double GEOGRAPHIC_WEIGHT = 0.9; // Increased from 0.7 for more geographically cohesive communities
    private static final double MAX_COMMUNITY_DIAMETER = 20000.0; // Maximum allowed diameter for a community in meters (Increased significantly from 2500.0m to disable problematic splitting)
   
    // Whether to use minibuses for small communities (< 25 nodes) or only use buses for all communities
    private static final boolean USE_MINIBUS = false; // Set to false to use only buses
   
    // Spectral clustering specific configuration - Initialize with default values to prevent NullPointerException
    private static SpectralClusteringConfig SPECTRAL_CONFIG = new SpectralClusteringConfig()
            .setNumberOfClusters(MAX_CLUSTERS)
            .setMinCommunitySize(MIN_CLUSTER_SIZE)
            .setPreventSingletons(true)
            .setSigma(100)
            .setGeographicWeight(0.9)
            .setMaxClusterSize(MAX_CLUSTER_SIZE)
            .setForceNumClusters(false) // Changed to false to preserve natural community structure
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
                
                // Check if the complete graph was successfully created with a sufficient number of edges
                // Allow graphs with at least 90% of the expected edges to be considered "complete enough"
                if (actualEdges > 0 && (actualEdges >= 0.9 * expectedEdges)) {
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
            
            // Log edge count before removing isolated nodes
            LOGGER.info("Graph created with " + graph.getEdgeCount() + " edges.");
            
            // Remove isolated nodes before clustering
            int nodesBeforeRemoval = 0; // We don't have direct access to count
            try {
                graph.removeIsolatedNodes();
                
                // Check if the graph has edges after removing isolated nodes
                if (graph.getEdgeCount() == 0) {
                    LOGGER.severe("Error: Graph has no edges after removing isolated nodes. This suggests all nodes were isolated.");
                    LOGGER.info("Attempting to recover by using a different graph strategy...");
                    
                    // Try with a fallback strategy (Delaunay triangulation is usually reliable)
                    LOGGER.info("Creating fallback graph using DELAUNAY triangulation...");
                    graph = graphService.createGraph(
                        points, GraphConstructionService.GraphStrategy.DELAUNAY, 
                        K_VALUE, USE_PARALLEL, VISUALIZE_GRAPH, SAVE_GRAPH);
                    
                    // Check if the fallback graph has edges
                    if (graph.getEdgeCount() == 0) {
                        LOGGER.severe("Error: Fallback graph creation also failed. Cannot proceed with analysis.");
                        throw new IllegalStateException("Graph creation failed - fallback also failed");
                    }
                    
                    LOGGER.info("Fallback graph created successfully with " + graph.getEdgeCount() + " edges.");
                    
                    // Make sure to remove any isolated nodes in the fallback graph
                    graph.removeIsolatedNodes();
                }
            } catch (Exception e) {
                LOGGER.severe("Error during graph preparation: " + e.getMessage());
                throw e;
            }
            
            // Instantiate Histogram Service
            HistogramService histogramService = new HistogramService();
            Map<Integer, ClusterMetrics> clusterMetrics = null;
            Map<Integer, List<Node>> communities = null; // Declare communities map here
            
            // Step 2.5: Apply outlier detection if enabled
            if (APPLY_OUTLIER_DETECTION) {
                LOGGER.info("Step 2.5: Detecting outliers using " + OUTLIER_ALGORITHM + " algorithm");
                OutlierDetectionService outlierService = new OutlierDetectionService();
                outlierService.setThreshold(OUTLIER_THRESHOLD)
                             .setKValue(OUTLIER_K_VALUE)
                             .setUseParallel(true)
                             .setNumThreads(OUTLIER_MAX_THREADS);
                
                LOGGER.info("Using " + OUTLIER_MAX_THREADS + " threads for parallel outlier detection");
                
                Set<Node> outliers = outlierService.detectOutliers(
                    graph, 
                    OUTLIER_ALGORITHM,
                    VISUALIZE_OUTLIERS,
                    VISUALIZE_OUTLIER_LEGEND
                );
                
                LOGGER.info("Detected " + outliers.size() + " outliers that will be excluded from clustering");
                
                // Create a filtered graph that explicitly excludes outliers for clustering
                TransportationGraph filteredGraph = createFilteredGraph(graph, outliers);
                
                // Use the filtered graph for clustering
                if (CLUSTERING_ALGORITHM == ClusteringService.ClusteringAlgorithm.SPECTRAL) {
                    LOGGER.info("Step 3: Performing Spectral Clustering with advanced configuration...");
                    LOGGER.info("Max Clusters: " + SPECTRAL_CONFIG.getNumberOfClusters() + 
                              ", Min Size: " + SPECTRAL_CONFIG.getMinCommunitySize() + 
                              ", Prevent Singletons: " + SPECTRAL_CONFIG.isPreventSingletons());
                    
                    ClusteringService clusteringService = new ClusteringService();
                    clusteringService.setSpectralConfig(SPECTRAL_CONFIG);
                    
                    // Capture the communities map from performClustering
                    communities = clusteringService.performClustering(
                        filteredGraph, 
                        CLUSTERING_ALGORITHM, 
                        VISUALIZE_CLUSTERS
                    );
                    
                    // Perform advanced transportation cost analysis with both vehicle options
                    LOGGER.info("Performing transportation cost analysis with vehicle options comparison...");
                    TransportationCostAnalysis.analyzeAndCompareVehicleOptions(
                        filteredGraph, communities, CLUSTERING_ALGORITHM.toString(), 
                        GRAPH_STRATEGY.toString(), OUTLIER_ALGORITHM.toString(), APPLY_OUTLIER_DETECTION);
                    
                    // Regular analysis for compatibility
                    clusterMetrics = TransportationCostAnalysis.analyzeCosts(filteredGraph, communities, USE_MINIBUS);
                    
                    // Save the cost analysis with metadata
                    LOGGER.info("Saving transportation cost analysis with metadata...");
                    TransportationCostAnalysis.saveAnalysisWithMetadata(
                        filteredGraph, communities, CLUSTERING_ALGORITHM.toString(), GRAPH_STRATEGY.toString(), K_VALUE, USE_MINIBUS);
                } else if (CLUSTERING_ALGORITHM == ClusteringService.ClusteringAlgorithm.GIRVAN_NEWMAN) {
                    // Girvan-Newman algorithm
                    LOGGER.info("Step 3: Performing Girvan-Newman clustering");
                    LOGGER.info("Target Clusters: " + MAX_CLUSTERS + ", Min Cluster Size: " + MIN_CLUSTER_SIZE + ", Max Cluster Size: " + MAX_CLUSTER_SIZE);
                    LOGGER.info("Max Iterations: " + GN_MAX_ITERATIONS + ", Early Stop: " + GN_EARLY_STOP);
                    LOGGER.info("Using Modularity Maximization: " + USE_MODULARITY_MAXIMIZATION);
                    
                    // Create and configure Girvan-Newman algorithm directly
                    GirvanNewmanClustering gnAlgorithm = new GirvanNewmanClustering(filteredGraph);
                    gnAlgorithm.setTargetCommunityCount(MAX_CLUSTERS)
                             .setMaxIterations(GN_MAX_ITERATIONS)
                             .setEarlyStop(GN_EARLY_STOP)
                             .setMinCommunitySize(MIN_CLUSTER_SIZE)
                             .setMaxCommunitySize(MAX_CLUSTER_SIZE)
                             .setUseModularityMaximization(USE_MODULARITY_MAXIMIZATION);
                    
                    // Run the algorithm
                    communities = gnAlgorithm.detectCommunities();
                    
                    // ADDITIONAL POST-PROCESSING: Enforce maximum community size
                    LOGGER.info("Performing post-processing to enforce community size constraints");
                    communities = enforceCommunityConstraints(communities, filteredGraph, MIN_CLUSTER_SIZE, MAX_CLUSTER_SIZE);
                    LOGGER.info("After post-processing: " + communities.size() + " communities");
                    
                    // Visualize if requested
                    if (VISUALIZE_CLUSTERS && !communities.isEmpty()) {
                        List<List<Node>> communityList = new ArrayList<>(communities.values());
                        filteredGraph.visualizeCommunities(communityList, CLUSTERING_ALGORITHM.toString(), false);
                        //filteredGraph.saveCommunityData(communities, CLUSTERING_ALGORITHM.toString());
                    }
                    
                    // Perform advanced transportation cost analysis with both vehicle options
                    LOGGER.info("Performing transportation cost analysis with vehicle options comparison...");
                    TransportationCostAnalysis.analyzeAndCompareVehicleOptions(
                        filteredGraph, communities, CLUSTERING_ALGORITHM.toString(), 
                        GRAPH_STRATEGY.toString(), OUTLIER_ALGORITHM.toString(), APPLY_OUTLIER_DETECTION);
                    
                    // Regular analysis for compatibility
                    clusterMetrics = TransportationCostAnalysis.analyzeCosts(filteredGraph, communities, USE_MINIBUS);
                    
                    // Save the cost analysis with metadata
                    LOGGER.info("Saving transportation cost analysis with metadata...");
                    TransportationCostAnalysis.saveAnalysisWithMetadata(
                        filteredGraph, communities, CLUSTERING_ALGORITHM.toString(), GRAPH_STRATEGY.toString(), K_VALUE, USE_MINIBUS);
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
                    InfomapCommunityDetection infomapAlgorithm = new InfomapCommunityDetection(filteredGraph);
                    infomapAlgorithm.setMaxClusters(MAX_CLUSTERS)
                                   .setMinClusterSize(MIN_CLUSTER_SIZE)
                                   .setMaxIterations(INFOMAP_MAX_ITERATIONS)
                                   .setTolerance((float)INFOMAP_TOLERANCE)
                                   .setForceMaxClusters(INFOMAP_FORCE_MAX_CLUSTERS)
                                   .setUseHierarchicalRefinement(false) // Disable hierarchical refinement to prevent fragmentation
                                   .setTeleportationProbability(0.05) // Significantly reduced to prevent "jumping" across the network
                                   .setGeographicImportance(INFOMAP_GEOGRAPHIC_IMPORTANCE) // Add geographic importance
                                   .setMaxGeographicDistance(INFOMAP_MAX_GEOGRAPHIC_DISTANCE); // Set max geographic distance
                    
                    communities = infomapAlgorithm.detectCommunities();
                    
                    // Visualize the communities
                    if (VISUALIZE_CLUSTERS && !communities.isEmpty()) {
                        List<List<Node>> communityList = new ArrayList<>(communities.values());
                        filteredGraph.visualizeCommunities(communityList, CLUSTERING_ALGORITHM.toString(), false); // Hide community 0
                        //filteredGraph.saveCommunityData(communities, CLUSTERING_ALGORITHM.toString());
                    }
                    
                    // Perform advanced transportation cost analysis with both vehicle options
                    LOGGER.info("Performing transportation cost analysis with vehicle options comparison...");
                    TransportationCostAnalysis.analyzeAndCompareVehicleOptions(
                        filteredGraph, communities, CLUSTERING_ALGORITHM.toString(), 
                        GRAPH_STRATEGY.toString(), OUTLIER_ALGORITHM.toString(), APPLY_OUTLIER_DETECTION);
                    
                    // Regular analysis for compatibility
                    clusterMetrics = TransportationCostAnalysis.analyzeCosts(filteredGraph, communities, USE_MINIBUS);
                    
                    // Save the cost analysis with metadata
                    LOGGER.info("Saving transportation cost analysis with metadata...");
                    TransportationCostAnalysis.saveAnalysisWithMetadata(
                        filteredGraph, communities, CLUSTERING_ALGORITHM.toString(), GRAPH_STRATEGY.toString(), K_VALUE, USE_MINIBUS);
                } else if (CLUSTERING_ALGORITHM == ClusteringService.ClusteringAlgorithm.MVAGC) {
                    // MvAGC algorithm
                    LOGGER.info("Step 3: Performing MvAGC clustering");
                    LOGGER.info("Target Clusters: " + MAX_CLUSTERS + ", Min Cluster Size: " + MIN_CLUSTER_SIZE + ", Max Cluster Size: " + MAX_CLUSTER_SIZE);
                    LOGGER.info("Anchor Nodes: " + MVAGC_NUM_ANCHORS);
                    LOGGER.info("Filter Order: " + MVAGC_FILTER_ORDER + ", Alpha: " + MVAGC_ALPHA);
                    LOGGER.info("Importance Sampling Power: " + MVAGC_SAMPLING_POWER);
                    
                    // Create and configure MvAGC algorithm directly
                    MvAGCClustering mvagcAlgorithm = new MvAGCClustering(filteredGraph);
                    mvagcAlgorithm.setNumClusters(MAX_CLUSTERS)
                                 .setNumAnchors(MVAGC_NUM_ANCHORS)
                                 .setFilterOrder(MVAGC_FILTER_ORDER)
                                 .setAlpha(MVAGC_ALPHA)
                                 .setImportanceSamplingPower(MVAGC_SAMPLING_POWER)
                                 .setMinClusterSize(MIN_CLUSTER_SIZE)
                                 .setMaxClusterSize(MAX_CLUSTER_SIZE)
                                 .setForceMinClusters(true);
                    
                    // Log the configuration
                    LOGGER.info("MvAGC configured with minClusterSize=" + MIN_CLUSTER_SIZE + ", loaded from application.properties");
                    
                    // Run the algorithm
                    communities = mvagcAlgorithm.detectCommunities();
                    
                    // Visualize if requested
                    if (VISUALIZE_CLUSTERS && !communities.isEmpty()) {
                        List<List<Node>> communityList = new ArrayList<>(communities.values());
                        filteredGraph.visualizeCommunities(communityList, CLUSTERING_ALGORITHM.toString(), false);
                        //filteredGraph.saveCommunityData(communities, CLUSTERING_ALGORITHM.toString());
                    }
                    
                    // Perform advanced transportation cost analysis with both vehicle options
                    LOGGER.info("Performing transportation cost analysis with vehicle options comparison...");
                    TransportationCostAnalysis.analyzeAndCompareVehicleOptions(
                        filteredGraph, communities, CLUSTERING_ALGORITHM.toString(), 
                        GRAPH_STRATEGY.toString(), OUTLIER_ALGORITHM.toString(), APPLY_OUTLIER_DETECTION);
                    
                    // Regular analysis for compatibility
                    clusterMetrics = TransportationCostAnalysis.analyzeCosts(filteredGraph, communities, USE_MINIBUS);
                    
                    // Save the cost analysis with metadata
                    LOGGER.info("Saving transportation cost analysis with metadata...");
                    TransportationCostAnalysis.saveAnalysisWithMetadata(
                        filteredGraph, communities, CLUSTERING_ALGORITHM.toString(), GRAPH_STRATEGY.toString(), K_VALUE, USE_MINIBUS);
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
                    communities = clusteringService.performClustering(
                        filteredGraph, 
                        CLUSTERING_ALGORITHM, 
                        VISUALIZE_CLUSTERS
                    );
                    
                    // Perform advanced transportation cost analysis with both vehicle options
                    LOGGER.info("Performing transportation cost analysis with vehicle options comparison...");
                    TransportationCostAnalysis.analyzeAndCompareVehicleOptions(
                        filteredGraph, communities, CLUSTERING_ALGORITHM.toString(), 
                        GRAPH_STRATEGY.toString(), OUTLIER_ALGORITHM.toString(), APPLY_OUTLIER_DETECTION);
                    
                    // Regular analysis for compatibility
                    clusterMetrics = TransportationCostAnalysis.analyzeCosts(filteredGraph, communities, USE_MINIBUS);
                    
                    // Save the cost analysis with metadata
                    LOGGER.info("Saving transportation cost analysis with metadata...");
                    TransportationCostAnalysis.saveAnalysisWithMetadata(
                        filteredGraph, communities, CLUSTERING_ALGORITHM.toString(), GRAPH_STRATEGY.toString(), K_VALUE, USE_MINIBUS);
                }
            } else {
                LOGGER.info("Outlier detection is disabled, proceeding directly to clustering");
                
                // Use the original graph for clustering
                if (CLUSTERING_ALGORITHM == ClusteringService.ClusteringAlgorithm.SPECTRAL) {
                    LOGGER.info("Step 3: Performing Spectral Clustering with advanced configuration...");
                    LOGGER.info("Max Clusters: " + SPECTRAL_CONFIG.getNumberOfClusters() + 
                              ", Min Size: " + SPECTRAL_CONFIG.getMinCommunitySize() + 
                              ", Prevent Singletons: " + SPECTRAL_CONFIG.isPreventSingletons());
                    
                    ClusteringService clusteringService = new ClusteringService();
                    clusteringService.setSpectralConfig(SPECTRAL_CONFIG);
                    
                    // Capture the communities map from performClustering
                    communities = clusteringService.performClustering(
                        graph, 
                        CLUSTERING_ALGORITHM, 
                        VISUALIZE_CLUSTERS
                    );
                    
                    // Perform advanced transportation cost analysis with both vehicle options
                    LOGGER.info("Performing transportation cost analysis with vehicle options comparison...");
                    TransportationCostAnalysis.analyzeAndCompareVehicleOptions(
                        graph, communities, CLUSTERING_ALGORITHM.toString(), 
                        GRAPH_STRATEGY.toString(), "NO_OUTLIER", false);
                    
                    // Regular analysis for compatibility
                    clusterMetrics = TransportationCostAnalysis.analyzeCosts(graph, communities, USE_MINIBUS);
                    
                    // Save the cost analysis with metadata
                    LOGGER.info("Saving transportation cost analysis with metadata...");
                    TransportationCostAnalysis.saveAnalysisWithMetadata(
                        graph, communities, CLUSTERING_ALGORITHM.toString(), GRAPH_STRATEGY.toString(), K_VALUE, USE_MINIBUS);
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
                             .setMaxCommunitySize(MAX_CLUSTER_SIZE)
                             .setUseModularityMaximization(USE_MODULARITY_MAXIMIZATION);
                    
                    // Run the algorithm
                    communities = gnAlgorithm.detectCommunities();
                    
                    // ADDITIONAL POST-PROCESSING: Enforce maximum community size
                    LOGGER.info("Performing post-processing to enforce community size constraints");
                    communities = enforceCommunityConstraints(communities, graph, MIN_CLUSTER_SIZE, MAX_CLUSTER_SIZE);
                    LOGGER.info("After post-processing: " + communities.size() + " communities");
                    
                    // Visualize if requested
                    if (VISUALIZE_CLUSTERS && !communities.isEmpty()) {
                        List<List<Node>> communityList = new ArrayList<>(communities.values());
                        graph.visualizeCommunities(communityList, CLUSTERING_ALGORITHM.toString(), false);
                        //graph.saveCommunityData(communities, CLUSTERING_ALGORITHM.toString());
                    }
                    
                    // Perform advanced transportation cost analysis with both vehicle options
                    LOGGER.info("Performing transportation cost analysis with vehicle options comparison...");
                    TransportationCostAnalysis.analyzeAndCompareVehicleOptions(
                        graph, communities, CLUSTERING_ALGORITHM.toString(), 
                        GRAPH_STRATEGY.toString(), "NO_OUTLIER", false);
                    
                    // Regular analysis for compatibility
                    clusterMetrics = TransportationCostAnalysis.analyzeCosts(graph, communities, USE_MINIBUS);
                    
                    // Save the cost analysis with metadata
                    LOGGER.info("Saving transportation cost analysis with metadata...");
                    TransportationCostAnalysis.saveAnalysisWithMetadata(
                        graph, communities, CLUSTERING_ALGORITHM.toString(), GRAPH_STRATEGY.toString(), K_VALUE, USE_MINIBUS);
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
                    
                    communities = infomapAlgorithm.detectCommunities();
                    
                    // Visualize the communities
                    if (VISUALIZE_CLUSTERS && !communities.isEmpty()) {
                        List<List<Node>> communityList = new ArrayList<>(communities.values());
                        graph.visualizeCommunities(communityList, CLUSTERING_ALGORITHM.toString(), false); // Hide community 0
                        //graph.saveCommunityData(communities, CLUSTERING_ALGORITHM.toString());
                    }
                    
                    // Perform advanced transportation cost analysis with both vehicle options
                    LOGGER.info("Performing transportation cost analysis with vehicle options comparison...");
                    TransportationCostAnalysis.analyzeAndCompareVehicleOptions(
                        graph, communities, CLUSTERING_ALGORITHM.toString(), 
                        GRAPH_STRATEGY.toString(), "NO_OUTLIER", false);
                    
                    // Regular analysis for compatibility
                    clusterMetrics = TransportationCostAnalysis.analyzeCosts(graph, communities, USE_MINIBUS);
                    
                    // Save the cost analysis with metadata
                    LOGGER.info("Saving transportation cost analysis with metadata...");
                    TransportationCostAnalysis.saveAnalysisWithMetadata(
                        graph, communities, CLUSTERING_ALGORITHM.toString(), GRAPH_STRATEGY.toString(), K_VALUE, USE_MINIBUS);
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
                    
                    // Log the configuration
                    LOGGER.info("MvAGC configured with minClusterSize=" + MIN_CLUSTER_SIZE + ", loaded from application.properties");
                    
                    // Run the algorithm
                    communities = mvagcAlgorithm.detectCommunities();
                    
                    // Visualize if requested
                    if (VISUALIZE_CLUSTERS && !communities.isEmpty()) {
                        List<List<Node>> communityList = new ArrayList<>(communities.values());
                        graph.visualizeCommunities(communityList, CLUSTERING_ALGORITHM.toString(), false);
                        //graph.saveCommunityData(communities, CLUSTERING_ALGORITHM.toString());
                    }
                    
                    // Perform advanced transportation cost analysis with both vehicle options
                    LOGGER.info("Performing transportation cost analysis with vehicle options comparison...");
                    TransportationCostAnalysis.analyzeAndCompareVehicleOptions(
                        graph, communities, CLUSTERING_ALGORITHM.toString(), 
                        GRAPH_STRATEGY.toString(), "NO_OUTLIER", false);
                    
                    // Regular analysis for compatibility
                    clusterMetrics = TransportationCostAnalysis.analyzeCosts(graph, communities, USE_MINIBUS);
                    
                    // Save the cost analysis with metadata
                    LOGGER.info("Saving transportation cost analysis with metadata...");
                    TransportationCostAnalysis.saveAnalysisWithMetadata(
                        graph, communities, CLUSTERING_ALGORITHM.toString(), GRAPH_STRATEGY.toString(), K_VALUE, USE_MINIBUS);
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
                    communities = clusteringService.performClustering(
                        graph, 
                        CLUSTERING_ALGORITHM, 
                        VISUALIZE_CLUSTERS
                    );
                    
                    // Perform advanced transportation cost analysis with both vehicle options
                    LOGGER.info("Performing transportation cost analysis with vehicle options comparison...");
                    TransportationCostAnalysis.analyzeAndCompareVehicleOptions(
                        graph, communities, CLUSTERING_ALGORITHM.toString(), 
                        GRAPH_STRATEGY.toString(), "NO_OUTLIER", false);
                    
                    // Regular analysis for compatibility
                    clusterMetrics = TransportationCostAnalysis.analyzeCosts(graph, communities, USE_MINIBUS);
                    
                    // Save the cost analysis with metadata
                    LOGGER.info("Saving transportation cost analysis with metadata...");
                    TransportationCostAnalysis.saveAnalysisWithMetadata(
                        graph, communities, CLUSTERING_ALGORITHM.toString(), GRAPH_STRATEGY.toString(), K_VALUE, USE_MINIBUS);
                }
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
            MIN_CLUSTER_SIZE = Integer.parseInt(properties.getProperty("transportation.bus.min.seats", "10"));
            MAX_CLUSTER_SIZE = Integer.parseInt(properties.getProperty("transportation.bus.max.seats", 
                                                properties.getProperty("transportation.bus.capacity", "45")));
            
            LOGGER.info("Loaded configuration from properties: MAX_CLUSTERS=" + MAX_CLUSTERS + 
                       ", MIN_CLUSTER_SIZE=" + MIN_CLUSTER_SIZE + 
                       ", MAX_CLUSTER_SIZE=" + MAX_CLUSTER_SIZE);
            
            // Update spectral clustering config with loaded properties
            SPECTRAL_CONFIG = new SpectralClusteringConfig()
                    .setNumberOfClusters(MAX_CLUSTERS)
                    .setMinCommunitySize(MIN_CLUSTER_SIZE)
                    .setPreventSingletons(true)
                    .setSigma(100)
                    .setGeographicWeight(0.9)
                    .setMaxClusterSize(MAX_CLUSTER_SIZE)
                    .setForceNumClusters(false) // Changed to false to preserve natural community structure
                    .setMaxCommunityDiameter(MAX_COMMUNITY_DIAMETER);
            
        } catch (IOException ex) {
            LOGGER.warning("Error loading properties: " + ex.getMessage());
        } catch (NumberFormatException ex) {
            LOGGER.warning("Error parsing property value: " + ex.getMessage());
        }
    }

    /**
     * Enforces minimum and maximum size constraints on communities.
     * Large communities (exceeding maxSize) are split, and small communities (below minSize) are merged.
     * 
     * @param communities The original communities map
     * @param graph The transportation graph
     * @param minSize Minimum allowed community size
     * @param maxSize Maximum allowed community size
     * @return A new map of communities respecting the size constraints
     */
    private static Map<Integer, List<Node>> enforceCommunityConstraints(
            Map<Integer, List<Node>> communities, 
            TransportationGraph graph,
            int minSize, 
            int maxSize) {
        
        LOGGER.info("Enforcing community constraints: min=" + minSize + ", max=" + maxSize);
        
        // Create a new map to hold the processed communities
        Map<Integer, List<Node>> result = new HashMap<>();
        int nextId = communities.size();
        
        // Find communities exceeding the maximum size
        List<Map.Entry<Integer, List<Node>>> toProcess = new ArrayList<>();
        for (Map.Entry<Integer, List<Node>> entry : communities.entrySet()) {
            if (entry.getValue().size() > maxSize) {
                LOGGER.info("Community " + entry.getKey() + " exceeds max size: " + entry.getValue().size());
                toProcess.add(entry);
            } else {
                // Keep communities that are already within bounds
                result.put(entry.getKey(), entry.getValue());
            }
        }
        
        // Process each oversized community
        for (Map.Entry<Integer, List<Node>> entry : toProcess) {
            List<Node> community = entry.getValue();
            int size = community.size();
            
            // Calculate how many subcommunities we need
            int numSubCommunities = (int) Math.ceil((double) size / maxSize);
            LOGGER.info("Splitting community " + entry.getKey() + " into " + numSubCommunities + " parts");
            
            // Use a simple geographic division strategy
            List<List<Node>> subCommunities = splitByGeography(community, numSubCommunities);
            
            // Add the subcommunities to the result
            for (List<Node> subCommunity : subCommunities) {
                if (subCommunity.size() >= minSize) {
                    result.put(nextId++, subCommunity);
                    LOGGER.info("Created subcommunity with size " + subCommunity.size());
                } else {
                    // For small subcommunities, find the closest community to merge with
                    mergeWithClosestCommunity(subCommunity, result, maxSize);
                }
            }
        }
        
        // Check for any remaining communities below min size
        toProcess = new ArrayList<>();
        for (Map.Entry<Integer, List<Node>> entry : result.entrySet()) {
            if (entry.getValue().size() < minSize) {
                toProcess.add(entry);
            }
        }
        
        // Merge small communities
        for (Map.Entry<Integer, List<Node>> entry : toProcess) {
            result.remove(entry.getKey());
            mergeWithClosestCommunity(entry.getValue(), result, maxSize);
        }
        
        LOGGER.info("Final community count after enforcing constraints: " + result.size());
        return result;
    }
    
    /**
     * Merges a list of nodes with the geographically closest community.
     * 
     * @param nodes The nodes to merge
     * @param communities The map of communities
     * @param maxSize Maximum allowed community size
     */
    private static void mergeWithClosestCommunity(
            List<Node> nodes, 
            Map<Integer, List<Node>> communities,
            int maxSize) {
        
        if (nodes.isEmpty() || communities.isEmpty()) {
            return;
        }
        
        // Calculate centroid of the nodes to merge
        double avgLat = 0, avgLon = 0;
        for (Node node : nodes) {
            avgLat += node.getLocation().getY();
            avgLon += node.getLocation().getX();
        }
        avgLat /= nodes.size();
        avgLon /= nodes.size();
        
        // Find the closest community that has room
        Integer closestCommunity = null;
        double minDistance = Double.MAX_VALUE;
        
        for (Map.Entry<Integer, List<Node>> entry : communities.entrySet()) {
            List<Node> community = entry.getValue();
            
            // Skip if adding would exceed max size
            if (community.size() + nodes.size() > maxSize) {
                continue;
            }
            
            // Calculate centroid of this community
            double comLat = 0, comLon = 0;
            for (Node node : community) {
                comLat += node.getLocation().getY();
                comLon += node.getLocation().getX();
            }
            comLat /= community.size();
            comLon /= community.size();
            
            // Calculate distance between centroids
            double distance = Math.sqrt(
                Math.pow(avgLat - comLat, 2) + 
                Math.pow(avgLon - comLon, 2)
            );
            
            if (distance < minDistance) {
                minDistance = distance;
                closestCommunity = entry.getKey();
            }
        }
        
        // Merge with the closest community, or create a new one if none found
        if (closestCommunity != null) {
            communities.get(closestCommunity).addAll(nodes);
            LOGGER.info("Merged " + nodes.size() + " nodes with community " + closestCommunity);
        } else {
            // If no suitable community found, create a new one
            int newId = communities.keySet().stream().max(Integer::compare).orElse(0) + 1;
            communities.put(newId, new ArrayList<>(nodes));
            LOGGER.info("Created new community " + newId + " with " + nodes.size() + " nodes");
        }
    }
    
    /**
     * Splits a community into subcommunities based on geographical position.
     * 
     * @param community The community to split
     * @param numParts The number of parts to split into
     * @return A list of subcommunities
     */
    private static List<List<Node>> splitByGeography(List<Node> community, int numParts) {
        // Sort nodes by latitude (north to south)
        List<Node> sortedNodes = new ArrayList<>(community);
        sortedNodes.sort((n1, n2) -> Double.compare(n1.getLocation().getY(), n2.getLocation().getY()));
        
        // Calculate part size
        int partSize = (int) Math.ceil((double) community.size() / numParts);
        
        // Create parts
        List<List<Node>> parts = new ArrayList<>();
        List<Node> currentPart = new ArrayList<>();
        
        for (Node node : sortedNodes) {
            currentPart.add(node);
            
            if (currentPart.size() >= partSize && parts.size() < numParts - 1) {
                parts.add(currentPart);
                currentPart = new ArrayList<>();
            }
        }
        
        // Add the last part if it's not empty
        if (!currentPart.isEmpty()) {
            parts.add(currentPart);
        }
        
        return parts;
    }

    /**
     * Creates a filtered graph that excludes outlier nodes
     * 
     * @param originalGraph The original graph with all nodes
     * @param outliers The set of outlier nodes to exclude
     * @return A new graph without the outlier nodes
     */
    private static TransportationGraph createFilteredGraph(TransportationGraph originalGraph, Set<Node> outliers) {
        LOGGER.info("Creating filtered graph without outliers for clustering...");
        
        // Create a new graph with only non-outlier nodes
        List<Point> nonOutlierPoints = new ArrayList<>();
        Set<Node> nodesToKeep = new HashSet<>();
        
        // Collect all non-outlier nodes
        for (Node node : originalGraph.getGraph().vertexSet()) {
            if (!outliers.contains(node) && !node.isOutlier()) {
                nodesToKeep.add(node);
                nonOutlierPoints.add(node.getLocation());
            }
        }
        
        // Create a new graph with only the non-outlier points
        TransportationGraph filteredGraph = new TransportationGraph(nonOutlierPoints);
        filteredGraph.setGraphConstructionMethod(originalGraph.getGraphConstructionMethod());
        
        // Copy edges between non-outlier nodes
        for (Node source : nodesToKeep) {
            for (Node target : nodesToKeep) {
                if (source.equals(target)) continue;
                
                DefaultWeightedEdge existingEdge = originalGraph.getGraph().getEdge(source, target);
                if (existingEdge != null) {
                    double weight = originalGraph.getGraph().getEdgeWeight(existingEdge);
                    filteredGraph.addConnection(source.getLocation(), target.getLocation(), weight);
                }
            }
        }
        
        LOGGER.info("Created filtered graph with " + nodesToKeep.size() + " nodes (excluded " + 
                    outliers.size() + " outliers)");
        
        return filteredGraph;
    }

    // Helper method for SPECTRAL clustering with the appropriate graph
    private static void spectral(TransportationGraph graph, boolean visualize) {
        // Existing SPECTRAL clustering code but with the passed graph
        ClusteringService clusteringService = new ClusteringService();
        clusteringService.setMaxClusters(MAX_CLUSTERS)
                       .setSpectralConfig(SPECTRAL_CONFIG);
        clusteringService.performClustering(graph, 
                                            ClusteringService.ClusteringAlgorithm.SPECTRAL,
                                            visualize);
    }

    // Helper method for GIRVAN_NEWMAN clustering with the appropriate graph
    private static void girvanNewman(TransportationGraph graph, boolean visualize) {
        // Existing GIRVAN_NEWMAN clustering code but with the passed graph
        ClusteringService clusteringService = new ClusteringService();
        clusteringService.setMaxClusters(MAX_CLUSTERS)
                       .setMinCommunitySize(MIN_CLUSTER_SIZE)
                       .setUseModularityMaximization(USE_MODULARITY_MAXIMIZATION);
        clusteringService.performClustering(graph, 
                                            ClusteringService.ClusteringAlgorithm.GIRVAN_NEWMAN,
                                            visualize);
    }

    // Helper method for INFOMAP clustering with the appropriate graph
    private static void infomap(TransportationGraph graph, boolean visualize) {
        // Existing INFOMAP clustering code but with the passed graph
        ClusteringService clusteringService = new ClusteringService();
        clusteringService.setMaxClusters(MAX_CLUSTERS)
                       .setMinCommunitySize(MIN_CLUSTER_SIZE);
        clusteringService.performClustering(graph, 
                                            ClusteringService.ClusteringAlgorithm.INFOMAP,
                                            visualize);
    }

    // Helper method for MVAGC clustering with the appropriate graph
    private static void mvagc(TransportationGraph graph, boolean visualize) {
        // Existing MVAGC clustering code but with the passed graph
        ClusteringService clusteringService = new ClusteringService();
        clusteringService.setMaxClusters(MAX_CLUSTERS)
                       .setMinCommunitySize(MIN_CLUSTER_SIZE);
        clusteringService.performClustering(graph, 
                                            ClusteringService.ClusteringAlgorithm.MVAGC,
                                            visualize);
    }

    // Helper method for LEIDEN clustering with the appropriate graph
    private static void leiden(TransportationGraph graph, boolean visualize) {
        // Existing LEIDEN clustering code but with the passed graph
        ClusteringService clusteringService = new ClusteringService();
        clusteringService.setMaxClusters(MAX_CLUSTERS)
                       .setMinCommunitySize(MIN_CLUSTER_SIZE)
                       .setCommunityScalingFactor(COMMUNITY_SCALING_FACTOR)
                       .setAdaptiveResolution(USE_ADAPTIVE_RESOLUTION);
        clusteringService.performClustering(graph, 
                                            ClusteringService.ClusteringAlgorithm.LEIDEN,
                                            visualize);
    }
}
