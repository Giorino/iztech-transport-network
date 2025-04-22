package com.izmir;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.logging.Level;
import java.util.logging.Logger;

import com.izmir.transportation.ClusteringService;
import com.izmir.transportation.TransportationGraph;
import com.izmir.transportation.cost.ClusterMetrics;
import com.izmir.transportation.cost.TransportationCostAnalysis;
import com.izmir.transportation.helper.Node;
import com.izmir.transportation.helper.clustering.MvAGCClustering;
import com.izmir.transportation.helper.clustering.SpectralClusteringConfig;
import com.izmir.transportation.persistence.GraphPersistenceService;
import com.izmir.visualization.HistogramService;

/**
 * Standalone class for applying various clustering algorithms to the transportation graphs.
 * This class:
 * 1. Loads previously created graphs (Complete, Gabriel, Delaunay, K-Nearest Neighbors)
 * 2. Applies three clustering algorithms to each graph:
 *    - Leiden
 *    - Spectral
 *    - MVAGC
 * 3. Visualizes the clustering results and performs cost analysis
 * 
 * @author davondeveloper
 */
public class ClusteringExample {
    private static final Logger LOGGER = Logger.getLogger(ClusteringExample.class.getName());
    
    // Configuration properties
    private static final int NODE_COUNT = 25;
    private static final boolean VISUALIZE_CLUSTERS = true;
    
    // Clustering configuration
    private static final int MAX_CLUSTERS = 5; // Fewer max clusters since we have fewer nodes (25)
    private static final int MIN_CLUSTER_SIZE = 3; // Minimum cluster size
    private static final int MAX_CLUSTER_SIZE = 10; // Maximum cluster size
    private static final double COMMUNITY_SCALING_FACTOR = 0.5;
    private static final boolean USE_ADAPTIVE_RESOLUTION = true;
    private static final double GEOGRAPHIC_WEIGHT = 0.9;
    private static final double MAX_COMMUNITY_DIAMETER = 20000.0;
    
    // Spectral clustering specific configuration
    private static final SpectralClusteringConfig SPECTRAL_CONFIG = new SpectralClusteringConfig()
            .setNumberOfClusters(MAX_CLUSTERS)
            .setMinCommunitySize(MIN_CLUSTER_SIZE)
            .setPreventSingletons(true)
            .setSigma(100)
            .setGeographicWeight(0.9)
            .setMaxClusterSize(MAX_CLUSTER_SIZE)
            .setForceNumClusters(false)
            .setMaxCommunityDiameter(MAX_COMMUNITY_DIAMETER);
    
    // MVAGC specific configuration
    private static final int MVAGC_NUM_ANCHORS = 200;
    private static final int MVAGC_FILTER_ORDER = 4;
    private static final double MVAGC_ALPHA = 8.0;
    private static final double MVAGC_SAMPLING_POWER = 0.3;
    
    // Graph types to analyze
    private static final String[] GRAPH_TYPES = {
            "complete", 
            "gabriel", 
            "delaunay", 
            "k_nearest_neighbors"
    };
    
    // Clustering algorithms to apply
    private static final ClusteringService.ClusteringAlgorithm[] CLUSTERING_ALGORITHMS = {
            ClusteringService.ClusteringAlgorithm.LEIDEN,
            ClusteringService.ClusteringAlgorithm.SPECTRAL,
            ClusteringService.ClusteringAlgorithm.MVAGC
    };
    
    public static void main(String[] args) {
        try {
            LOGGER.info("Starting Clustering Example...");
            
            GraphPersistenceService persistenceService = new GraphPersistenceService();
            HistogramService histogramService = new HistogramService();
            
            // Process each graph type
            for (String graphType : GRAPH_TYPES) {
                LOGGER.info("Loading " + graphType + " graph...");
                
                // Load the graph from persistent storage
                TransportationGraph graph = persistenceService.loadGraph(NODE_COUNT, graphType);
                
                if (graph == null) {
                    LOGGER.warning("Graph not found: " + NODE_COUNT + " nodes, type " + graphType);
                    continue;
                }
                
                LOGGER.info("Loaded " + graphType + " graph with " + graph.getEdgeCount() + " edges.");
                
                // Apply each clustering algorithm
                for (ClusteringService.ClusteringAlgorithm algorithm : CLUSTERING_ALGORITHMS) {
                    LOGGER.info("Applying " + algorithm + " clustering to " + graphType + " graph...");
                    
                    // Create a titled version to show which algorithm and graph type
                    String clusteringTitle = algorithm + " on " + graphType;
                    
                    // Clustering and visualization
                    Map<Integer, List<Node>> communities = applyClustering(graph, algorithm, clusteringTitle);
                    
                    // Skip if clustering failed
                    if (communities == null || communities.isEmpty()) {
                        LOGGER.warning("Clustering failed for " + algorithm + " on " + graphType);
                        continue;
                    }
                    
                    // Perform transportation cost analysis
                    LOGGER.info("Performing transportation cost analysis...");
                    Map<Integer, ClusterMetrics> clusterMetrics = 
                            TransportationCostAnalysis.analyzeCosts(graph, communities);
                    
                    // Save the cost analysis with metadata
                    LOGGER.info("Saving transportation cost analysis...");
                    TransportationCostAnalysis.saveAnalysisWithMetadata(
                            graph, communities, algorithm.toString(), graphType, 0);
                    
                    // Generate histograms
                    histogramService.generateHistograms(clusterMetrics, clusteringTitle);
                    
                    // Add a delay to allow visualization to complete
                    Thread.sleep(3000);
                }
            }
            
            LOGGER.info("Clustering analysis completed successfully.");
            
        } catch (Exception e) {
            LOGGER.log(Level.SEVERE, "Error during clustering analysis: " + e.getMessage(), e);
            e.printStackTrace();
        }
    }
    
    /**
     * Applies a specific clustering algorithm to the transportation graph.
     * 
     * @param graph The transportation graph to cluster
     * @param algorithm The clustering algorithm to apply
     * @param clusteringTitle Title for the clustering visualization
     * @return A map of communities detected
     * @throws Exception If an error occurs during clustering
     */
    private static Map<Integer, List<Node>> applyClustering(
            TransportationGraph graph, 
            ClusteringService.ClusteringAlgorithm algorithm,
            String clusteringTitle) throws Exception {
        
        Map<Integer, List<Node>> communities = null;
        
        if (algorithm == ClusteringService.ClusteringAlgorithm.SPECTRAL) {
            // Spectral clustering
            LOGGER.info("Performing Spectral clustering with advanced configuration...");
            
            ClusteringService clusteringService = new ClusteringService();
            clusteringService.setSpectralConfig(SPECTRAL_CONFIG);
            
            communities = clusteringService.performClustering(
                graph, 
                algorithm, 
                VISUALIZE_CLUSTERS
            );
            
        } else if (algorithm == ClusteringService.ClusteringAlgorithm.MVAGC) {
            // MVAGC algorithm
            LOGGER.info("Performing MVAGC clustering...");
            
            // Create and configure MVAGC algorithm directly
            MvAGCClustering mvagcAlgorithm = new MvAGCClustering(graph);
            mvagcAlgorithm.setNumClusters(MAX_CLUSTERS)
                         .setNumAnchors(MVAGC_NUM_ANCHORS)
                         .setFilterOrder(MVAGC_FILTER_ORDER)
                         .setAlpha(MVAGC_ALPHA)
                         .setImportanceSamplingPower(MVAGC_SAMPLING_POWER)
                         .setMinClusterSize(MIN_CLUSTER_SIZE)
                         .setMaxClusterSize(MAX_CLUSTER_SIZE)
                         .setForceMinClusters(true);
            
            communities = mvagcAlgorithm.detectCommunities();
            
            // Visualize if requested
            if (VISUALIZE_CLUSTERS && !communities.isEmpty()) {
                List<List<Node>> communityList = new ArrayList<>(communities.values());
                graph.visualizeCommunities(communityList, clusteringTitle, false);
            }
            
        } else {
            // Leiden algorithm
            LOGGER.info("Performing Leiden clustering...");
            
            ClusteringService clusteringService = new ClusteringService();
            clusteringService.setMaxClusters(MAX_CLUSTERS)
                            .setCommunityScalingFactor(COMMUNITY_SCALING_FACTOR)
                            .setAdaptiveResolution(USE_ADAPTIVE_RESOLUTION)
                            .setMinCommunitySize(MIN_CLUSTER_SIZE);
            
            communities = clusteringService.performClustering(
                graph, 
                algorithm, 
                VISUALIZE_CLUSTERS
            );
        }
        
        return communities;
    }
} 