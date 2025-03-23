package com.izmir.transportation;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.jgrapht.Graph;
import org.jgrapht.graph.DefaultWeightedEdge;
import org.locationtech.jts.geom.Point;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.izmir.transportation.cost.TransportationCostAnalysis;
import com.izmir.transportation.helper.Node;
import com.izmir.transportation.helper.clustering.GirvanNewmanClustering;
import com.izmir.transportation.helper.clustering.InfomapCommunityDetection;
import com.izmir.transportation.helper.clustering.LeidenCommunityDetection;
import com.izmir.transportation.helper.clustering.MvAGCClustering;
import com.izmir.transportation.helper.clustering.SpectralClustering;
import com.izmir.transportation.helper.clustering.SpectralClusteringConfig;

/**
 * Service for performing clustering operations on the transportation graph.
 * This allows for separation of graph construction and clustering operations.
 */
public class ClusteringService {
    private static final Logger LOGGER = LoggerFactory.getLogger(ClusteringService.class);
    
    // Configuration parameters for clustering
    private int maxClusters = 25; // Default maximum number of clusters to detect
    private double communityScalingFactor = 0.75; // Controls community detection sensitivity (higher = more communities)
    private boolean adaptiveResolution = true; // Whether to use adaptive resolution for Leiden algorithm
    private int minCommunitySize = 5; // Minimum size for a community, smaller ones will be merged
    private int maxCommunitySize = 50; // Maximum community size (corresponds to bus capacity)
    private boolean useModularityMaximization = true; // Whether to use modularity maximization for Girvan-Newman
    
    // Specific spectral clustering configuration
    private SpectralClusteringConfig spectralConfig;
    
    /**
     * Constructor that initializes the default configuration
     */
    public ClusteringService() {
        // Initialize default spectral clustering config
        spectralConfig = new SpectralClusteringConfig()
            .setNumberOfClusters(maxClusters)
            .setMinCommunitySize(minCommunitySize)
            .setPreventSingletons(true);
    }
    
    /**
     * Available clustering algorithms
     */
    public enum ClusteringAlgorithm {
        LEIDEN("leiden"),
        SPECTRAL("spectral"),
        GIRVAN_NEWMAN("girvan_newman"),
        INFOMAP("infomap"),
        MVAGC("mvagc");
        
        private final String code;
        
        ClusteringAlgorithm(String code) {
            this.code = code;
        }
        
        public String getCode() {
            return code;
        }
        
        public static ClusteringAlgorithm fromCode(String code) {
            for (ClusteringAlgorithm algorithm : values()) {
                if (algorithm.getCode().equals(code)) {
                    return algorithm;
                }
            }
            throw new IllegalArgumentException("Unknown algorithm code: " + code);
        }
    }
    
    /**
     * Sets the maximum number of clusters to detect
     * 
     * @param maxClusters The maximum number of clusters to allow (must be at least 2)
     * @return This ClusteringService instance for method chaining
     */
    public ClusteringService setMaxClusters(int maxClusters) {
        this.maxClusters = Math.max(2, maxClusters);
        LOGGER.info("Maximum number of clusters set to {}", this.maxClusters);
        return this;
    }
    
    /**
     * Sets the community scaling factor which controls sensitivity of community detection
     * Higher values result in more communities being detected
     * 
     * @param factor Scaling factor between 0.1 and 2.0
     * @return This ClusteringService instance for method chaining
     */
    public ClusteringService setCommunityScalingFactor(double factor) {
        this.communityScalingFactor = Math.max(0.1, Math.min(2.0, factor));
        LOGGER.info("Community scaling factor set to {}", this.communityScalingFactor);
        return this;
    }
    
    /**
     * Sets whether to use adaptive resolution based on network size
     * 
     * @param adaptive True to use adaptive resolution, false for fixed resolution
     * @return This ClusteringService instance for method chaining
     */
    public ClusteringService setAdaptiveResolution(boolean adaptive) {
        this.adaptiveResolution = adaptive;
        LOGGER.info("Adaptive resolution set to {}", this.adaptiveResolution);
        return this;
    }
    
    /**
     * Sets the minimum size for communities. Communities smaller than this will be merged
     * with their nearest neighbor community. Helps eliminate singletons and very small communities.
     * 
     * @param minSize Minimum number of nodes in a community (default is 5)
     * @return This ClusteringService instance for method chaining
     */
    public ClusteringService setMinCommunitySize(int minSize) {
        this.minCommunitySize = Math.max(1, minSize);
        LOGGER.info("Minimum community size set to {}", this.minCommunitySize);
        
        // Update spectral config as well
        if (spectralConfig != null) {
            spectralConfig.setMinCommunitySize(this.minCommunitySize);
        }
        
        return this;
    }
    
    /**
     * Sets whether to use modularity maximization for Girvan-Newman algorithm.
     * When enabled, the algorithm will find the community division that maximizes modularity.
     * 
     * @param useModularity True to use modularity maximization
     * @return This ClusteringService instance for method chaining
     */
    public ClusteringService setUseModularityMaximization(boolean useModularity) {
        this.useModularityMaximization = useModularity;
        LOGGER.info("Modularity maximization set to {}", this.useModularityMaximization);
        return this;
    }
    
    /**
     * Gets the SpectralClusteringConfig object for detailed configuration of spectral clustering.
     * This allows for fine-tuning the spectral clustering algorithm beyond the basic parameters.
     * 
     * @return The SpectralClusteringConfig instance
     */
    public SpectralClusteringConfig getSpectralConfig() {
        return spectralConfig;
    }
    
    /**
     * Sets the SpectralClusteringConfig object for detailed configuration of spectral clustering.
     * 
     * @param config The configuration object to use
     * @return This ClusteringService instance for method chaining
     */
    public ClusteringService setSpectralConfig(SpectralClusteringConfig config) {
        this.spectralConfig = config;
        
        // Sync the maxClusters value with what's in the config
        if (config != null) {
            this.maxClusters = config.getNumberOfClusters();
            this.minCommunitySize = config.getMinCommunitySize();
        }
        
        LOGGER.info("Set custom spectral clustering configuration");
        return this;
    }
    
    /**
     * Apply a clustering algorithm to the transportation graph
     *
     * @param graph The transportation graph to cluster
     * @param algorithm The clustering algorithm to use
     * @param visualize Whether to visualize the clusters
     * @return Map of community IDs to lists of nodes
     */
    public Map<Integer, List<Node>> performClustering(
            TransportationGraph graph, 
            ClusteringAlgorithm algorithm,
            boolean visualize) {
        
        LOGGER.info("Performing clustering using {} algorithm with max clusters set to {}", algorithm, maxClusters);
        LOGGER.info("Community scaling factor: {}, Adaptive resolution: {}, Min community size: {}", 
                    communityScalingFactor, adaptiveResolution, minCommunitySize);
        
        // Create the affinity matrix if it doesn't exist
        if (graph.getAffinityMatrix() == null) {
            graph.createAffinityMatrix();
        }
        
        // Apply the appropriate clustering algorithm
        Map<Integer, List<Node>> communities;
        
        if (algorithm == ClusteringAlgorithm.LEIDEN) {
            // Use Leiden algorithm
            LeidenCommunityDetection leidenAlgorithm = new LeidenCommunityDetection(graph);
            leidenAlgorithm.setCommunityCountLimits(2, maxClusters); // Set the max clusters
            leidenAlgorithm.setCommunityScalingFactor(communityScalingFactor); // Set sensitivity
            leidenAlgorithm.setAdaptiveResolution(adaptiveResolution); // Configure resolution approach
            
            // Set the maximum size constraint for bus capacity
            leidenAlgorithm.setMaxCommunitySize(maxCommunitySize);
            leidenAlgorithm.setEnforceMaxCommunitySize(true);
            
            // Set the minimum size constraint for efficient bus utilization
            leidenAlgorithm.setMinCommunitySize(minCommunitySize);
            leidenAlgorithm.setEnforceMinCommunitySize(true);
            
            communities = leidenAlgorithm.detectCommunities();
        } else if (algorithm == ClusteringAlgorithm.SPECTRAL) {
            // Use Spectral algorithm with SpectralClusteringConfig
            LOGGER.info("Using spectral clustering with minimum community size: {}, prevent singletons: {}", 
                       spectralConfig.getMinCommunitySize(), spectralConfig.isPreventSingletons());
            
            // Ensure spectral config has current values
            spectralConfig.setNumberOfClusters(maxClusters);
            
            SpectralClustering spectralAlgorithm = new SpectralClustering(graph, spectralConfig);
            communities = spectralAlgorithm.detectCommunities();
        } else if (algorithm == ClusteringAlgorithm.GIRVAN_NEWMAN) {
            // Use Girvan-Newman algorithm
            LOGGER.info("Using Girvan-Newman algorithm with target communities: {}, min community size: {}",
                       maxClusters, minCommunitySize);
            LOGGER.info("Modularity maximization: {}", useModularityMaximization);
            
            GirvanNewmanClustering girvanNewmanAlgorithm = new GirvanNewmanClustering(graph);
            girvanNewmanAlgorithm.setTargetCommunityCount(maxClusters)
                                .setMinCommunitySize(minCommunitySize)
                                .setUseModularityMaximization(useModularityMaximization);
            communities = girvanNewmanAlgorithm.detectCommunities();
        } else if (algorithm == ClusteringAlgorithm.INFOMAP) {
            // Use Infomap algorithm
            LOGGER.info("Using Infomap algorithm with natural community discovery");
            LOGGER.info("Min community size: {}, Max iterations: {}", minCommunitySize, 200);
            
            InfomapCommunityDetection infomapAlgorithm = new InfomapCommunityDetection(graph);
            infomapAlgorithm.setMinClusterSize(minCommunitySize)
                           .setMaxIterations(200)  // Increase iterations for better convergence
                           .setTolerance(1e-5)     // Lower tolerance for more precise results
                           .setForceMaxClusters(false);  // Don't force merging, let algorithm find natural communities
            communities = infomapAlgorithm.detectCommunities();
        } else if (algorithm == ClusteringAlgorithm.MVAGC) {
            // Use MvAGC algorithm
            LOGGER.info("Using MvAGC algorithm with {} clusters and {} anchor nodes", maxClusters, Math.min(200, graph.getGraph().vertexSet().size()/2));
            
            MvAGCClustering mvagcAlgorithm = new MvAGCClustering(graph);
            mvagcAlgorithm.setNumClusters(maxClusters)
                         .setNumAnchors(Math.min(200, graph.getGraph().vertexSet().size()/2)) // Use at most 200 anchor nodes or half the nodes
                         .setFilterOrder(2)
                         .setAlpha(5.0)
                         .setImportanceSamplingPower(1.0);
            communities = mvagcAlgorithm.detectCommunities();
        } else {
            throw new IllegalArgumentException("Unsupported algorithm: " + algorithm);
        }
        
        LOGGER.info("Found {} communities before post-processing", communities.size());
        
        // Post-process to handle small communities - only for Leiden or if we're not using spectral config
        if ((algorithm == ClusteringAlgorithm.LEIDEN || algorithm == ClusteringAlgorithm.GIRVAN_NEWMAN) && minCommunitySize > 1) {
            communities = mergeSmallCommunities(communities, graph, minCommunitySize);
            LOGGER.info("After merging small communities: {} communities remain", communities.size());
        }
        
        // Visualize the communities if requested
        if (visualize && !communities.isEmpty()) {
            List<List<Node>> communityList = new ArrayList<>(communities.values());
            graph.visualizeCommunities(communityList, algorithm.toString(), true); // Hide community 0
            
            // Save the community data
            //graph.saveCommunityData(communities, algorithm.toString());
        }
        
        // Perform transportation cost analysis
        LOGGER.info("Performing transportation cost analysis...");
        performCostAnalysis(graph, communities);
        
        return communities;
    }
    
    /**
     * Merges communities smaller than the specified minimum size with their nearest neighboring community
     * 
     * @param communities The initial community assignments
     * @param graph The transportation graph
     * @param minSize The minimum allowed community size
     * @return A new map with small communities merged
     */
    private Map<Integer, List<Node>> mergeSmallCommunities(
            Map<Integer, List<Node>> communities, 
            TransportationGraph graph, 
            int minSize) {
        
        // Count how many small communities we have
        List<Integer> smallCommunityIds = new ArrayList<>();
        for (Map.Entry<Integer, List<Node>> entry : communities.entrySet()) {
            if (entry.getValue().size() < minSize) {
                smallCommunityIds.add(entry.getKey());
            }
        }
        
        LOGGER.info("Found {} communities smaller than minimum size {}", 
                   smallCommunityIds.size(), minSize);
        
        if (smallCommunityIds.isEmpty()) {
            return communities; // Nothing to merge
        }
        
        // Create a copy of the communities map
        Map<Integer, List<Node>> mergedCommunities = new HashMap<>();
        for (Map.Entry<Integer, List<Node>> entry : communities.entrySet()) {
            mergedCommunities.put(entry.getKey(), new ArrayList<>(entry.getValue()));
        }
        
        // Get the main graph
        Graph<Node, DefaultWeightedEdge> graphObj = graph.getGraph();
        
        // Calculate centroids for larger communities
        Map<Integer, double[]> centroids = new HashMap<>();
        for (Map.Entry<Integer, List<Node>> entry : mergedCommunities.entrySet()) {
            if (!smallCommunityIds.contains(entry.getKey())) {
                centroids.put(entry.getKey(), calculateCentroid(entry.getValue()));
            }
        }
        
        // Process each small community
        for (Integer smallCommunityId : smallCommunityIds) {
            List<Node> smallCommunityNodes = mergedCommunities.get(smallCommunityId);
            
            if (smallCommunityNodes == null || smallCommunityNodes.isEmpty()) {
                continue; // Skip if already processed
            }
            
            // Find the best community to merge with
            int bestCommunityId = -1;
            double bestScore = Double.MAX_VALUE;
            
            for (Map.Entry<Integer, double[]> centroidEntry : centroids.entrySet()) {
                int targetCommunityId = centroidEntry.getKey();
                double[] targetCentroid = centroidEntry.getValue();
                
                // Skip if it's the same community or another small community
                if (targetCommunityId == smallCommunityId || smallCommunityIds.contains(targetCommunityId)) {
                    continue;
                }
                
                // Calculate average distance to centroid
                double totalDistance = 0.0;
                for (Node node : smallCommunityNodes) {
                    totalDistance += calculateDistance(getNodeCoordinates(node), targetCentroid);
                }
                double avgDistance = totalDistance / smallCommunityNodes.size();
                
                // Consider connections in the graph
                double connectionScore = calculateConnectionScore(smallCommunityNodes, 
                                                              mergedCommunities.get(targetCommunityId),
                                                              graphObj);
                
                // Combined score (weight distance more heavily for singletons)
                double score = (smallCommunityNodes.size() == 1 ? 0.8 : 0.5) * avgDistance + 
                               (1.0 - (smallCommunityNodes.size() == 1 ? 0.8 : 0.5)) * (1.0 - connectionScore);
                
                if (score < bestScore) {
                    bestScore = score;
                    bestCommunityId = targetCommunityId;
                }
            }
            
            // If no suitable community found (unlikely), use the largest community
            if (bestCommunityId == -1) {
                int maxSize = 0;
                for (Map.Entry<Integer, List<Node>> entry : mergedCommunities.entrySet()) {
                    if (!smallCommunityIds.contains(entry.getKey()) && entry.getValue().size() > maxSize) {
                        maxSize = entry.getValue().size();
                        bestCommunityId = entry.getKey();
                    }
                }
            }
            
            // Merge the small community into the best community
            if (bestCommunityId != -1) {
                List<Node> targetCommunity = mergedCommunities.get(bestCommunityId);
                targetCommunity.addAll(smallCommunityNodes);
                mergedCommunities.remove(smallCommunityId);
                
                // Update the centroid
                centroids.put(bestCommunityId, calculateCentroid(targetCommunity));
                
                LOGGER.debug("Merged community {} (size {}) into community {} (new size {})",
                           smallCommunityId, smallCommunityNodes.size(), 
                           bestCommunityId, targetCommunity.size());
            }
        }
        
        return mergedCommunities;
    }
    
    /**
     * Calculate the centroid (average position) of nodes in a community
     * 
     * @param nodes List of nodes
     * @return Centroid coordinates [x, y]
     */
    private double[] calculateCentroid(List<Node> nodes) {
        double sumX = 0.0;
        double sumY = 0.0;
        
        for (Node node : nodes) {
            double[] coords = getNodeCoordinates(node);
            sumX += coords[0];
            sumY += coords[1];
        }
        
        return new double[] { sumX / nodes.size(), sumY / nodes.size() };
    }
    
    /**
     * Get coordinates for a node
     * 
     * @param node The node
     * @return Coordinates [x, y]
     */
    private double[] getNodeCoordinates(Node node) {
        Point point = node.getLocation();
        return new double[] { point.getX(), point.getY() };
    }
    
    /**
     * Calculate Euclidean distance between two points
     * 
     * @param point1 First point
     * @param point2 Second point
     * @return Euclidean distance
     */
    private double calculateDistance(double[] point1, double[] point2) {
        double dx = point1[0] - point2[0];
        double dy = point1[1] - point2[1];
        return Math.sqrt(dx * dx + dy * dy);
    }
    
    /**
     * Calculate connection score between two communities
     * 
     * @param community1 First community
     * @param community2 Second community
     * @param graph The graph
     * @return Connection score (0 to 1, higher means more connected)
     */
    private double calculateConnectionScore(List<Node> community1, List<Node> community2, 
                                      Graph<Node, DefaultWeightedEdge> graph) {
        int connections = 0;
        int possibleConnections = community1.size() * community2.size();
        
        if (possibleConnections == 0) {
            return 0.0;
        }
        
        for (Node node1 : community1) {
            for (Node node2 : community2) {
                if (graph.containsEdge(node1, node2) || graph.containsEdge(node2, node1)) {
                    connections++;
                }
            }
        }
        
        return (double) connections / possibleConnections;
    }
    
    /**
     * Performs transportation cost analysis on the detected communities
     *
     * @param graph The transportation graph
     * @param communities The detected communities
     */
    private void performCostAnalysis(TransportationGraph graph, Map<Integer, List<Node>> communities) {
        try {
            LOGGER.info("Starting transportation cost analysis...");
            TransportationCostAnalysis.analyzeCosts(graph, communities);
            LOGGER.info("Transportation cost analysis completed.");
        } catch (Exception e) {
            LOGGER.error("Error performing transportation cost analysis", e);
        }
    }
} 