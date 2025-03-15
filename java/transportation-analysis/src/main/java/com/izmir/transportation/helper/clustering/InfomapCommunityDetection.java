package com.izmir.transportation.helper.clustering;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;

import org.jgrapht.Graph;
import org.jgrapht.graph.DefaultWeightedEdge;
import org.jgrapht.graph.SimpleWeightedGraph;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.izmir.transportation.TransportationGraph;
import com.izmir.transportation.helper.Node;

/**
 * Implementation of the Infomap algorithm for community detection in transportation networks.
 * 
 * The Infomap algorithm is based on information flow in networks and uses the minimum description length
 * principle to identify communities. It simulates random walks on the network and tries to minimize
 * the description length of these walks to find meaningful communities.
 *
 * This implementation is simplified for educational purposes and clarity.
 * 
 * @author yagizugurveren
 */
public class InfomapCommunityDetection implements GraphClusteringAlgorithm {
    private static final Logger LOGGER = LoggerFactory.getLogger(InfomapCommunityDetection.class);
    
    private TransportationGraph transportationGraph;
    private int maxIterations = 200;
    private double tolerance = 1e-5;
    private double teleportationProbability = 0.15; // Similar to PageRank's damping factor
    private int randomSeed = 42; // For reproducibility
    private int minClusterSize = 3; // Minimum size for a community
    private int maxClusters = 20; // Maximum number of clusters
    private boolean forceMaxClusters = false; // Don't force merging to maxClusters by default
    private boolean useHierarchicalRefinement = true; // Use hierarchical refinement to improve communities
    private double geographicImportance = 0.0; // Weight given to geographic distance (0.0 = topology only, 1.0 = geography only)
    private double maxGeographicDistance = 10000.0; // Maximum geographic distance to consider (meters)
    
    // Random number generator
    private final Random random;
    
    // Current community assignments
    private Map<Node, Integer> nodeCommunities;
    
    // Current codeLengths
    private double currentCodeLength;
    
    /**
     * Constructor that initializes the Infomap algorithm with a transportation graph
     * 
     * @param transportationGraph The transportation graph to cluster
     */
    public InfomapCommunityDetection(TransportationGraph transportationGraph) {
        this.transportationGraph = transportationGraph;
        this.random = new Random(randomSeed);
        this.nodeCommunities = new HashMap<>();
    }
    
    /**
     * Sets the maximum number of iterations for the algorithm
     * 
     * @param maxIterations Maximum number of iterations
     * @return This InfomapCommunityDetection instance for method chaining
     */
    public InfomapCommunityDetection setMaxIterations(int maxIterations) {
        this.maxIterations = Math.max(10, maxIterations);
        return this;
    }
    
    /**
     * Sets the convergence tolerance
     * 
     * @param tolerance Convergence tolerance (smaller means more precise)
     * @return This InfomapCommunityDetection instance for method chaining
     */
    public InfomapCommunityDetection setTolerance(double tolerance) {
        this.tolerance = Math.max(1e-6, Math.min(1e-2, tolerance));
        return this;
    }
    
    /**
     * Sets the teleportation probability for random walks
     * 
     * @param probability Probability of random teleportation (between 0 and 1)
     * @return This InfomapCommunityDetection instance for method chaining
     */
    public InfomapCommunityDetection setTeleportationProbability(double probability) {
        this.teleportationProbability = Math.max(0.05, Math.min(0.5, probability));
        return this;
    }
    
    /**
     * Sets the random seed for reproducibility
     * 
     * @param seed Random seed
     * @return This InfomapCommunityDetection instance for method chaining
     */
    public InfomapCommunityDetection setRandomSeed(int seed) {
        this.randomSeed = seed;
        return this;
    }
    
    /**
     * Sets the minimum size for a community
     * 
     * @param minSize Minimum number of nodes in a community
     * @return This InfomapCommunityDetection instance for method chaining
     */
    public InfomapCommunityDetection setMinClusterSize(int minSize) {
        this.minClusterSize = Math.max(1, minSize);
        return this;
    }
    
    /**
     * Sets the maximum number of clusters to detect
     * 
     * @param maxClusters Maximum number of clusters
     * @return This InfomapCommunityDetection instance for method chaining
     */
    public InfomapCommunityDetection setMaxClusters(int maxClusters) {
        this.maxClusters = Math.max(2, maxClusters);
        return this;
    }
    
    /**
     * Sets whether to force merging down to maximum number of clusters
     * 
     * @param force True to force merging down to maxClusters
     * @return This InfomapCommunityDetection instance for method chaining
     */
    public InfomapCommunityDetection setForceMaxClusters(boolean force) {
        this.forceMaxClusters = force;
        return this;
    }
    
    /**
     * Sets whether to use hierarchical refinement approach
     * 
     * @param use True to use hierarchical refinement to get better communities
     * @return This InfomapCommunityDetection instance for method chaining
     */
    public InfomapCommunityDetection setUseHierarchicalRefinement(boolean use) {
        this.useHierarchicalRefinement = use;
        return this;
    }
    
    /**
     * Sets the importance of geographic proximity in community assignment
     * 
     * @param weight Weight from 0.0 (topology only) to 1.0 (geography only)
     * @return This InfomapCommunityDetection instance for method chaining
     */
    public InfomapCommunityDetection setGeographicImportance(double weight) {
        this.geographicImportance = Math.max(0.0, Math.min(1.0, weight));
        return this;
    }
    
    /**
     * Sets the maximum geographic distance to consider for community cohesion
     * 
     * @param distance Maximum distance in meters
     * @return This InfomapCommunityDetection instance for method chaining
     */
    public InfomapCommunityDetection setMaxGeographicDistance(double distance) {
        this.maxGeographicDistance = Math.max(1000.0, distance);
        return this;
    }
    
    /**
     * Detects communities in the transportation graph using the Infomap algorithm
     * 
     * @return Map of community IDs to lists of nodes
     */
    public Map<Integer, List<Node>> detectCommunities() {
        Graph<Node, DefaultWeightedEdge> graph = transportationGraph.getGraph();
        List<Node> nodes = new ArrayList<>(graph.vertexSet());
        
        if (nodes.isEmpty()) {
            LOGGER.warn("Empty graph - no communities to detect");
            return Collections.emptyMap();
        }
        
        LOGGER.info("Starting Infomap community detection on graph with {} nodes", nodes.size());
        LOGGER.info("Using natural community discovery without forcing cluster count");
        
        // Initialize each node to its own community
        initializeCommunities(nodes);
        
        // Calculate initial flow and transition probabilities (PageRank-like)
        Map<Node, Double> stationaryDistribution = calculateStationaryDistribution(graph);
        Map<DefaultWeightedEdge, Double> flowMatrix = calculateFlowMatrix(graph, stationaryDistribution);
        
        // Calculate initial code length
        this.currentCodeLength = calculateMapEquation(graph, nodeCommunities, stationaryDistribution, flowMatrix);
        LOGGER.info("Initial code length: {}", currentCodeLength);
        
        // Main optimization loop
        boolean improved = true;
        int iteration = 0;
        
        while (improved && iteration < maxIterations) {
            iteration++;
            improved = false;
            
            // Shuffle nodes to avoid order bias
            Collections.shuffle(nodes, random);
            
            double totalImprovement = 0.0;
            int moveCount = 0;
            
            // Try to move each node to a better community
            for (Node node : nodes) {
                int currentCommunity = nodeCommunities.get(node);
                
                // Find the best community for this node
                int bestCommunity = findBestCommunity(node, graph, stationaryDistribution, flowMatrix);
                
                // If the best community is different, move the node
                if (bestCommunity != currentCommunity) {
                    // Calculate old code length
                    double oldCodeLength = currentCodeLength;
                    
                    // Move node to new community
                    nodeCommunities.put(node, bestCommunity);
                    
                    // Recalculate code length
                    double newCodeLength = calculateMapEquation(graph, nodeCommunities, stationaryDistribution, flowMatrix);
                    double improvement = oldCodeLength - newCodeLength;
                    
                    if (improvement > 0) {
                        // Keep the move
                        currentCodeLength = newCodeLength;
                        improved = true;
                        totalImprovement += improvement;
                        moveCount++;
                    } else {
                        // Revert the move
                        nodeCommunities.put(node, currentCommunity);
                    }
                }
            }
            
            // Log improvement details
            if (moveCount > 0) {
                double avgImprovement = totalImprovement / moveCount;
                LOGGER.info("Iteration {}: Moved {} nodes, Avg improvement = {}, Total improvement = {}", 
                           iteration, moveCount, avgImprovement, totalImprovement);
                
                // Count current communities
                Set<Integer> distinctCommunities = new HashSet<>(nodeCommunities.values());
                LOGGER.info("Current community count: {}", distinctCommunities.size());
            } else {
                LOGGER.info("Iteration {}: No improvements found", iteration);
            }
            
            // Check if we're below the tolerance threshold
            if (totalImprovement / nodes.size() < tolerance && iteration > 10) {
                LOGGER.info("Stopping early - improvement below tolerance");
                break;
            }
        }
        
        // Apply hierarchical refinement if enabled
        if (useHierarchicalRefinement) {
            LOGGER.info("Applying hierarchical refinement to improve communities");
            applyHierarchicalRefinement(graph, stationaryDistribution, flowMatrix);
        }
        
        // Post-process communities
        Map<Integer, List<Node>> communities = postProcessCommunities(nodes);
        
        LOGGER.info("Infomap detected {} communities after {} iterations", 
                   communities.size(), iteration);
        
        return communities;
    }
    
    /**
     * Apply hierarchical refinement to further improve the community structure
     * This is a key component of the Infomap algorithm - after finding a partition,
     * try to find substructure within each community.
     */
    private void applyHierarchicalRefinement(Graph<Node, DefaultWeightedEdge> graph, 
                                            Map<Node, Double> stationaryDistribution,
                                            Map<DefaultWeightedEdge, Double> flowMatrix) {
        // Get the current communities
        Map<Integer, List<Node>> currentCommunities = new HashMap<>();
        for (Node node : graph.vertexSet()) {
            int communityId = nodeCommunities.get(node);
            if (!currentCommunities.containsKey(communityId)) {
                currentCommunities.put(communityId, new ArrayList<>());
            }
            currentCommunities.get(communityId).add(node);
        }
        
        LOGGER.info("Refining {} communities hierarchically", currentCommunities.size());
        
        // Try to subdivide each community (if it's large enough)
        for (Map.Entry<Integer, List<Node>> entry : currentCommunities.entrySet()) {
            int communityId = entry.getKey();
            List<Node> communityNodes = entry.getValue();
            
            // Only try to subdivide communities with more than minClusterSize*3 nodes
            if (communityNodes.size() <= minClusterSize * 3) {
                continue;
            }
            
            LOGGER.debug("Attempting to refine community {} with {} nodes", communityId, communityNodes.size());
            
            // Create a subgraph for this community
            Set<Node> communityNodeSet = new HashSet<>(communityNodes);
            Graph<Node, DefaultWeightedEdge> subgraph = createSubgraph(graph, communityNodeSet);
            
            // Skip if subgraph is too small or has no edges
            if (subgraph.vertexSet().size() <= minClusterSize * 3 || subgraph.edgeSet().isEmpty()) {
                continue;
            }
            
            // Recalculate flow for the subgraph
            Map<Node, Double> subStationaryDistribution = calculateSubgraphStationaryDistribution(
                    subgraph, stationaryDistribution);
            Map<DefaultWeightedEdge, Double> subFlowMatrix = calculateSubgraphFlowMatrix(
                    subgraph, subStationaryDistribution);
            
            // Initialize subgraph nodes to a single community
            Map<Node, Integer> subCommunities = new HashMap<>();
            for (Node node : subgraph.vertexSet()) {
                subCommunities.put(node, 0); // All in same initial community
            }
            
            // Calculate initial subgraph code length
            double subCodeLength = calculateMapEquation(subgraph, subCommunities, 
                                                      subStationaryDistribution, subFlowMatrix);
            
            // Try to find subcommunities
            boolean subImproved = true;
            int subIteration = 0;
            
            while (subImproved && subIteration < maxIterations / 2) {
                subIteration++;
                subImproved = false;
                
                List<Node> subNodes = new ArrayList<>(subgraph.vertexSet());
                Collections.shuffle(subNodes, random);
                
                int moveCount = 0;
                
                // Try to move each node to a better subcommunity
                for (Node node : subNodes) {
                    int currentSubCommunity = subCommunities.get(node);
                    
                    // Find neighboring subcommunities
                    Set<Integer> neighborSubCommunities = new HashSet<>();
                    neighborSubCommunities.add(currentSubCommunity);
                    
                    for (DefaultWeightedEdge edge : subgraph.edgesOf(node)) {
                        Node neighbor = subgraph.getEdgeSource(edge);
                        if (neighbor.equals(node)) {
                            neighbor = subgraph.getEdgeTarget(edge);
                        }
                        neighborSubCommunities.add(subCommunities.get(neighbor));
                    }
                    
                    // Try each neighboring subcommunity
                    int bestSubCommunity = currentSubCommunity;
                    double bestSubCodeLength = subCodeLength;
                    
                    for (int candidateSubCommunity : neighborSubCommunities) {
                        if (candidateSubCommunity == currentSubCommunity) {
                            continue;
                        }
                        
                        // Try moving to this subcommunity
                        subCommunities.put(node, candidateSubCommunity);
                        double candidateCodeLength = calculateMapEquation(subgraph, subCommunities, 
                                                                        subStationaryDistribution, subFlowMatrix);
                        
                        if (candidateCodeLength < bestSubCodeLength) {
                            bestSubCommunity = candidateSubCommunity;
                            bestSubCodeLength = candidateCodeLength;
                        }
                        
                        // Revert
                        subCommunities.put(node, currentSubCommunity);
                    }
                    
                    // If found a better subcommunity, move there
                    if (bestSubCommunity != currentSubCommunity) {
                        subCommunities.put(node, bestSubCommunity);
                        subCodeLength = bestSubCodeLength;
                        subImproved = true;
                        moveCount++;
                    }
                }
                
                if (moveCount == 0) {
                    break;
                }
            }
            
            // Check if we found meaningful subcommunities
            Set<Integer> distinctSubCommunities = new HashSet<>(subCommunities.values());
            
            if (distinctSubCommunities.size() > 1) {
                LOGGER.info("Found {} subcommunities in community {}", 
                           distinctSubCommunities.size(), communityId);
                
                // Assign new community IDs to the subcommunities
                // First, get the highest current community ID
                int highestCommunityId = nodeCommunities.values().stream()
                        .mapToInt(Integer::intValue)
                        .max()
                        .orElse(0);
                
                // Create mapping from subcommunity ID to new community ID
                Map<Integer, Integer> subToGlobalCommunity = new HashMap<>();
                for (int subCommunityId : distinctSubCommunities) {
                    // Original community keeps its ID, others get new IDs
                    if (subCommunityId == 0) {
                        subToGlobalCommunity.put(subCommunityId, communityId);
                    } else {
                        subToGlobalCommunity.put(subCommunityId, ++highestCommunityId);
                    }
                }
                
                // Update node communities in the main graph
                for (Node node : subgraph.vertexSet()) {
                    int subCommunityId = subCommunities.get(node);
                    int newGlobalCommunityId = subToGlobalCommunity.get(subCommunityId);
                    nodeCommunities.put(node, newGlobalCommunityId);
                }
            }
        }
    }
    
    /**
     * Create a subgraph containing only the specified nodes and their interconnections
     */
    private Graph<Node, DefaultWeightedEdge> createSubgraph(
            Graph<Node, DefaultWeightedEdge> parentGraph, Set<Node> nodeSubset) {
        
        Graph<Node, DefaultWeightedEdge> subgraph = 
                new SimpleWeightedGraph<>(DefaultWeightedEdge.class);
        
        // Add all nodes from the subset
        for (Node node : nodeSubset) {
            subgraph.addVertex(node);
        }
        
        // Add all edges between nodes in the subset
        for (Node source : nodeSubset) {
            for (DefaultWeightedEdge edge : parentGraph.edgesOf(source)) {
                Node target = parentGraph.getEdgeTarget(edge);
                if (source.equals(target)) {
                    target = parentGraph.getEdgeSource(edge);
                }
                
                if (nodeSubset.contains(target) && !source.equals(target)) {
                    if (subgraph.containsEdge(source, target)) {
                        continue; // Edge already added
                    }
                    
                    DefaultWeightedEdge newEdge = subgraph.addEdge(source, target);
                    if (newEdge != null) {
                        subgraph.setEdgeWeight(newEdge, parentGraph.getEdgeWeight(edge));
                    }
                }
            }
        }
        
        return subgraph;
    }
    
    /**
     * Calculate stationary distribution for a subgraph using values from parent graph
     */
    private Map<Node, Double> calculateSubgraphStationaryDistribution(
            Graph<Node, DefaultWeightedEdge> subgraph, Map<Node, Double> parentDistribution) {
        
        Map<Node, Double> subDistribution = new HashMap<>();
        
        // Extract the values for nodes in subgraph
        double totalWeight = 0.0;
        for (Node node : subgraph.vertexSet()) {
            double weight = parentDistribution.getOrDefault(node, 0.0);
            subDistribution.put(node, weight);
            totalWeight += weight;
        }
        
        // Normalize if needed
        if (totalWeight > 0) {
            for (Map.Entry<Node, Double> entry : subDistribution.entrySet()) {
                entry.setValue(entry.getValue() / totalWeight);
            }
        } else {
            // Fallback to uniform distribution
            double uniformValue = 1.0 / subgraph.vertexSet().size();
            for (Node node : subgraph.vertexSet()) {
                subDistribution.put(node, uniformValue);
            }
        }
        
        return subDistribution;
    }
    
    /**
     * Calculate flow matrix for a subgraph
     */
    private Map<DefaultWeightedEdge, Double> calculateSubgraphFlowMatrix(
            Graph<Node, DefaultWeightedEdge> subgraph, Map<Node, Double> stationaryDistribution) {
        
        Map<DefaultWeightedEdge, Double> flowMatrix = new HashMap<>();
        
        // Calculate flow on each edge
        for (DefaultWeightedEdge edge : subgraph.edgeSet()) {
            Node source = subgraph.getEdgeSource(edge);
            Node target = subgraph.getEdgeTarget(edge);
            
            double sourceProb = stationaryDistribution.getOrDefault(source, 0.0);
            double edgeWeight = subgraph.getEdgeWeight(edge);
            
            // Calculate total weight of edges from source
            double totalSourceWeight = 0.0;
            for (DefaultWeightedEdge outEdge : subgraph.edgesOf(source)) {
                if (subgraph.getEdgeSource(outEdge).equals(source) || 
                    subgraph.getEdgeTarget(outEdge).equals(source)) {
                    totalSourceWeight += subgraph.getEdgeWeight(outEdge);
                }
            }
            
            // Calculate transition probability
            double transitionProb = totalSourceWeight > 0 ? edgeWeight / totalSourceWeight : 0.0;
            
            // Calculate flow
            double flow = sourceProb * transitionProb;
            flowMatrix.put(edge, flow);
        }
        
        return flowMatrix;
    }
    
    /**
     * Calculate the stationary distribution (PageRank) of nodes in the graph
     */
    private Map<Node, Double> calculateStationaryDistribution(Graph<Node, DefaultWeightedEdge> graph) {
        Map<Node, Double> distribution = new HashMap<>();
        int n = graph.vertexSet().size();
        
        // Initialize uniformly
        for (Node node : graph.vertexSet()) {
            distribution.put(node, 1.0 / n);
        }
        
        // Power iteration to calculate PageRank
        for (int i = 0; i < 50; i++) { // 50 iterations should be enough for convergence
            Map<Node, Double> newDistribution = new HashMap<>();
            
            // Initialize with teleportation probability
            for (Node node : graph.vertexSet()) {
                newDistribution.put(node, teleportationProbability / n);
            }
            
            // Add contributions from neighboring nodes
            for (Node source : graph.vertexSet()) {
                double sourceRank = distribution.get(source);
                
                // Get total weight of outgoing edges
                double totalWeight = 0.0;
                for (DefaultWeightedEdge edge : graph.edgesOf(source)) {
                    totalWeight += graph.getEdgeWeight(edge);
                }
                
                if (totalWeight > 0) {
                    // Distribute rank to neighbors
                    for (DefaultWeightedEdge edge : graph.edgesOf(source)) {
                        Node target = graph.getEdgeTarget(edge);
                        if (target.equals(source)) {
                            target = graph.getEdgeSource(edge);
                        }
                        
                        double edgeWeight = graph.getEdgeWeight(edge);
                        double flow = sourceRank * (1.0 - teleportationProbability) * (edgeWeight / totalWeight);
                        newDistribution.put(target, newDistribution.get(target) + flow);
                    }
                } else {
                    // If no outgoing edges, distribute evenly (teleport)
                    double teleportFlow = sourceRank * (1.0 - teleportationProbability) / n;
                    for (Node target : graph.vertexSet()) {
                        newDistribution.put(target, newDistribution.get(target) + teleportFlow);
                    }
                }
            }
            
            // Normalize
            double sum = newDistribution.values().stream().mapToDouble(Double::doubleValue).sum();
            for (Map.Entry<Node, Double> entry : newDistribution.entrySet()) {
                entry.setValue(entry.getValue() / sum);
            }
            
            // Update distribution
            distribution = newDistribution;
        }
        
        return distribution;
    }
    
    /**
     * Calculate flow matrix for the graph
     */
    private Map<DefaultWeightedEdge, Double> calculateFlowMatrix(
            Graph<Node, DefaultWeightedEdge> graph, Map<Node, Double> stationaryDistribution) {
        
        Map<DefaultWeightedEdge, Double> flowMatrix = new HashMap<>();
        
        // Calculate flow on each edge
        for (DefaultWeightedEdge edge : graph.edgeSet()) {
            Node source = graph.getEdgeSource(edge);
            Node target = graph.getEdgeTarget(edge);
            
            double sourceProb = stationaryDistribution.getOrDefault(source, 0.0);
            double edgeWeight = graph.getEdgeWeight(edge);
            
            // Calculate total weight of edges from source
            double totalSourceWeight = 0.0;
            for (DefaultWeightedEdge outEdge : graph.edgesOf(source)) {
                if (graph.getEdgeSource(outEdge).equals(source)) {
                    totalSourceWeight += graph.getEdgeWeight(outEdge);
                }
            }
            
            // Calculate transition probability with teleportation
            double transitionProb = (1.0 - teleportationProbability) * (edgeWeight / totalSourceWeight);
            
            // Add teleportation probability
            transitionProb += teleportationProbability / graph.vertexSet().size();
            
            // Calculate flow
            double flow = sourceProb * transitionProb;
            flowMatrix.put(edge, flow);
        }
        
        return flowMatrix;
    }
    
    /**
     * Calculate the Map Equation (code length) for the current community assignment
     */
    private double calculateMapEquation(
            Graph<Node, DefaultWeightedEdge> graph, 
            Map<Node, Integer> communities,
            Map<Node, Double> stationaryDistribution,
            Map<DefaultWeightedEdge, Double> flowMatrix) {
        
        // Calculate exit and entry flow probabilities for each community
        Map<Integer, Double> exitFlows = new HashMap<>();
        Map<Integer, Double> entryFlows = new HashMap<>();
        Map<Integer, Double> internalFlows = new HashMap<>();
        Map<Integer, Double> totalVisitFlows = new HashMap<>();
        
        // Calculate visit rates for each community
        for (Node node : graph.vertexSet()) {
            int communityId = communities.get(node);
            double visitRate = stationaryDistribution.getOrDefault(node, 0.0);
            totalVisitFlows.put(communityId, totalVisitFlows.getOrDefault(communityId, 0.0) + visitRate);
        }
        
        // Calculate entry, exit, and internal flows
        for (DefaultWeightedEdge edge : graph.edgeSet()) {
            Node source = graph.getEdgeSource(edge);
            Node target = graph.getEdgeTarget(edge);
            double flow = flowMatrix.getOrDefault(edge, 0.0);
            
            int sourceCommunity = communities.get(source);
            int targetCommunity = communities.get(target);
            
            if (sourceCommunity == targetCommunity) {
                // Internal flow
                internalFlows.put(sourceCommunity, internalFlows.getOrDefault(sourceCommunity, 0.0) + flow);
            } else {
                // Exit flow
                exitFlows.put(sourceCommunity, exitFlows.getOrDefault(sourceCommunity, 0.0) + flow);
                // Entry flow
                entryFlows.put(targetCommunity, entryFlows.getOrDefault(targetCommunity, 0.0) + flow);
            }
        }
        
        // Calculate total flow (should be 1.0)
        double totalFlow = totalVisitFlows.values().stream().mapToDouble(Double::doubleValue).sum();
        
        // Index codebook length (encodes which community to enter)
        double indexLength = 0.0;
        
        // Module codebook lengths (encodes movements within communities)
        double moduleLength = 0.0;
        
        // Calculate index codebook length
        for (int community : totalVisitFlows.keySet()) {
            double pExit = exitFlows.getOrDefault(community, 0.0);
            moduleLength += pExit * Math.log(pExit) / Math.log(2);
            
            double pModule = totalVisitFlows.get(community);
            if (pModule > 0) {
                indexLength -= (pModule / totalFlow) * Math.log(pModule / totalFlow) / Math.log(2);
            }
        }
        
        // Calculate module codebook lengths
        for (int community : totalVisitFlows.keySet()) {
            double pModule = totalVisitFlows.get(community);
            double exitProb = exitFlows.getOrDefault(community, 0.0);
            
            if (pModule > 0) {
                // Normalize internal flow by total visit rate
                double internalProb = internalFlows.getOrDefault(community, 0.0);
                
                // Add entry probability (from index)
                double entryProb = entryFlows.getOrDefault(community, 0.0);
                
                // Total flow within module
                double totalModuleFlow = internalProb + exitProb;
                
                if (totalModuleFlow > 0) {
                    // Calculate internal entropy of module
                    double moduleEntropy = 0.0;
                    
                    // Node visit rates within module contribute to entropy
                    for (Node node : graph.vertexSet()) {
                        if (communities.get(node) == community) {
                            double nodeProb = stationaryDistribution.get(node) / pModule;
                            if (nodeProb > 0) {
                                moduleEntropy -= nodeProb * Math.log(nodeProb) / Math.log(2);
                            }
                        }
                    }
                    
                    // Exit probability contributes to entropy
                    if (exitProb > 0) {
                        double exitNodeProb = exitProb / totalModuleFlow;
                        moduleEntropy -= exitNodeProb * Math.log(exitNodeProb) / Math.log(2);
                    }
                    
                    // Weight by module usage probability
                    moduleLength += pModule * moduleEntropy;
                }
            }
        }
        
        // Total description length
        double L = indexLength + moduleLength;
        return L;
    }
    
    /**
     * Find the best community for a node based on both topology and geography
     */
    private int findBestCommunity(Node node, Graph<Node, DefaultWeightedEdge> graph, 
                                 Map<Node, Double> stationaryDistribution,
                                 Map<DefaultWeightedEdge, Double> flowMatrix) {
        
        int currentCommunity = nodeCommunities.get(node);
        int bestCommunity = currentCommunity;
        double bestCodeLength = currentCodeLength;
        
        // Get neighboring communities
        Set<Integer> neighborCommunities = new HashSet<>();
        neighborCommunities.add(currentCommunity); // Consider staying in current community
        
        for (DefaultWeightedEdge edge : graph.edgesOf(node)) {
            Node neighbor = graph.getEdgeSource(edge);
            if (neighbor.equals(node)) {
                neighbor = graph.getEdgeTarget(edge);
            }
            neighborCommunities.add(nodeCommunities.get(neighbor));
        }
        
        // If geographic importance is set, consider all communities within max distance
        if (geographicImportance > 0.0) {
            for (Node otherNode : graph.vertexSet()) {
                // Skip if already considering this community
                int otherCommunity = nodeCommunities.get(otherNode);
                if (neighborCommunities.contains(otherCommunity)) {
                    continue;
                }
                
                // Check geographic distance
                double distance = Math.sqrt(
                    Math.pow(node.getLocation().getX() - otherNode.getLocation().getX(), 2) + 
                    Math.pow(node.getLocation().getY() - otherNode.getLocation().getY(), 2)
                );
                
                // If within range, consider this community too
                if (distance < maxGeographicDistance * 0.5) {
                    neighborCommunities.add(otherCommunity);
                }
            }
        }
        
        // Evaluate each neighboring community
        for (int candidateCommunity : neighborCommunities) {
            if (candidateCommunity == currentCommunity) {
                continue; // Skip current community
            }
            
            // Temporarily move the node to candidate community
            nodeCommunities.put(node, candidateCommunity);
            
            // Calculate new code length
            double newCodeLength = calculateMapEquation(graph, nodeCommunities, 
                                                      stationaryDistribution, flowMatrix);
            
            // Add geographic penalty if enabled
            if (geographicImportance > 0.0) {
                // Get center of candidate community
                List<Node> communityNodes = new ArrayList<>();
                for (Node n : graph.vertexSet()) {
                    if (nodeCommunities.get(n) == candidateCommunity && !n.equals(node)) {
                        communityNodes.add(n);
                    }
                }
                
                // Apply geographic penalty
                if (!communityNodes.isEmpty()) {
                    double geographicPenalty = calculateGeographicPenalty(node, communityNodes);
                    newCodeLength += geographicImportance * geographicPenalty;
                }
            }
            
            // If this improves the code length, remember it
            if (newCodeLength < bestCodeLength) {
                bestCodeLength = newCodeLength;
                bestCommunity = candidateCommunity;
            }
            
            // Restore node to its original community
            nodeCommunities.put(node, currentCommunity);
        }
        
        return bestCommunity;
    }
    
    /**
     * Calculate geographic penalty for placing a node in a community
     * Higher distance = higher penalty
     */
    private double calculateGeographicPenalty(Node node, List<Node> communityNodes) {
        // Find community center
        double centerX = 0.0, centerY = 0.0;
        for (Node n : communityNodes) {
            centerX += n.getLocation().getX();
            centerY += n.getLocation().getY();
        }
        centerX /= communityNodes.size();
        centerY /= communityNodes.size();
        
        // Calculate distance from node to center
        double distance = Math.sqrt(
            Math.pow(node.getLocation().getX() - centerX, 2) + 
            Math.pow(node.getLocation().getY() - centerY, 2)
        );
        
        // Convert to penalty (larger distance = larger penalty)
        return Math.min(1.0, distance / maxGeographicDistance);
    }
    
    /**
     * Initialize each node to its own community
     * 
     * @param nodes List of nodes in the graph
     */
    private void initializeCommunities(List<Node> nodes) {
        nodeCommunities.clear();
        
        // Each node starts in its own community
        for (int i = 0; i < nodes.size(); i++) {
            nodeCommunities.put(nodes.get(i), i);
        }
    }
    
    /**
     * Post-process communities to handle small communities and renumber them
     * 
     * @param nodes List of nodes in the graph
     * @return Map of community IDs to lists of nodes
     */
    private Map<Integer, List<Node>> postProcessCommunities(List<Node> nodes) {
        // Collect communities
        Map<Integer, List<Node>> communities = new HashMap<>();
        
        for (Node node : nodes) {
            int communityId = nodeCommunities.get(node);
            
            if (!communities.containsKey(communityId)) {
                communities.put(communityId, new ArrayList<>());
            }
            
            communities.get(communityId).add(node);
        }
        
        LOGGER.info("Before post-processing: {} communities", communities.size());
        
        // Handle small communities - merge them with the closest community
        // only if they're smaller than minClusterSize
        boolean merged;
        do {
            merged = false;
            
            for (Integer communityId : new ArrayList<>(communities.keySet())) {
                List<Node> community = communities.get(communityId);
                
                // Check if this community is too small
                if (community.size() < minClusterSize && communities.size() > 1) {
                    // Find the nearest neighboring community
                    int nearestCommunity = findNearestCommunity(communityId, communities, 
                                                               transportationGraph.getGraph());
                    
                    // Merge with the nearest community
                    if (nearestCommunity != communityId) {
                        List<Node> nearestNodes = communities.get(nearestCommunity);
                        nearestNodes.addAll(community);
                        
                        // Update node community assignments
                        for (Node node : community) {
                            nodeCommunities.put(node, nearestCommunity);
                        }
                        
                        // Remove merged community
                        communities.remove(communityId);
                        merged = true;
                        break;
                    }
                }
            }
        } while (merged);
        
        LOGGER.info("After merging small communities: {} communities", communities.size());
        
        // Only force merging if explicitly requested
        if (forceMaxClusters && communities.size() > maxClusters) {
            LOGGER.info("Forcing merge down to {} communities from {}", maxClusters, communities.size());
            
            // Keep merging the smallest communities until we're under maxClusters
            while (communities.size() > maxClusters) {
                // Find smallest community
                int smallestCommunityId = -1;
                int smallestSize = Integer.MAX_VALUE;
                
                for (Map.Entry<Integer, List<Node>> entry : communities.entrySet()) {
                    if (entry.getValue().size() < smallestSize) {
                        smallestSize = entry.getValue().size();
                        smallestCommunityId = entry.getKey();
                    }
                }
                
                // Find nearest community to the smallest
                int nearestCommunity = findNearestCommunity(smallestCommunityId, communities, 
                                                           transportationGraph.getGraph());
                
                // If can't find a different community, pick any other one
                if (nearestCommunity == smallestCommunityId && communities.size() > 1) {
                    for (int otherId : communities.keySet()) {
                        if (otherId != smallestCommunityId) {
                            nearestCommunity = otherId;
                            break;
                        }
                    }
                }
                
                // Merge with nearest
                if (nearestCommunity != smallestCommunityId) {
                    List<Node> smallestNodes = communities.get(smallestCommunityId);
                    List<Node> nearestNodes = communities.get(nearestCommunity);
                    
                    LOGGER.debug("Merging community {} (size {}) into {} (size {})", 
                               smallestCommunityId, smallestNodes.size(), 
                               nearestCommunity, nearestNodes.size());
                    
                    nearestNodes.addAll(smallestNodes);
                    
                    // Update node community assignments
                    for (Node node : smallestNodes) {
                        nodeCommunities.put(node, nearestCommunity);
                    }
                    
                    // Remove merged community
                    communities.remove(smallestCommunityId);
                } else {
                    // No more merging possible
                    LOGGER.warn("Cannot merge further, stopping at {} communities", communities.size());
                    break;
                }
            }
        }
        
        // Renumber communities from 0 to N-1
        Map<Integer, Integer> communityMapping = new HashMap<>();
        Map<Integer, List<Node>> renumberedCommunities = new HashMap<>();
        int newId = 0;
        
        for (int oldId : communities.keySet()) {
            communityMapping.put(oldId, newId);
            renumberedCommunities.put(newId, communities.get(oldId));
            newId++;
        }
        
        // Update node community assignments with new IDs
        for (Node node : nodes) {
            int oldCommunityId = nodeCommunities.get(node);
            int newCommunityId = communityMapping.get(oldCommunityId);
            nodeCommunities.put(node, newCommunityId);
        }
        
        // Print community sizes
        LOGGER.info("Final community distribution:");
        for (Map.Entry<Integer, List<Node>> entry : renumberedCommunities.entrySet()) {
            LOGGER.info("Community {}: {} nodes", entry.getKey(), entry.getValue().size());
        }
        
        return renumberedCommunities;
    }
    
    /**
     * Find the nearest community to a given community
     * Modified to consider geographic proximity
     * 
     * @param communityId The ID of the community to find neighbors for
     * @param communities Map of community IDs to lists of nodes
     * @param graph The graph
     * @return The ID of the nearest community
     */
    private int findNearestCommunity(int communityId, Map<Integer, List<Node>> communities,
                                    Graph<Node, DefaultWeightedEdge> graph) {
        List<Node> sourceCommunity = communities.get(communityId);
        
        int nearestCommunity = communityId;
        double bestScore = Double.NEGATIVE_INFINITY;
        
        // Check connections to other communities
        for (Map.Entry<Integer, List<Node>> entry : communities.entrySet()) {
            int targetCommunityId = entry.getKey();
            
            // Skip self
            if (targetCommunityId == communityId) continue;
            
            List<Node> targetCommunity = entry.getValue();
            
            // Calculate connectivity score (network topology)
            double connections = countConnections(sourceCommunity, targetCommunity, graph);
            
            // Calculate geographic proximity score if enabled
            double geographicScore = 0.0;
            if (geographicImportance > 0.0) {
                geographicScore = calculateGeographicProximity(sourceCommunity, targetCommunity);
            }
            
            // Combine scores based on importance weight
            double combinedScore = (1.0 - geographicImportance) * connections + 
                                  geographicImportance * geographicScore;
            
            if (combinedScore > bestScore) {
                bestScore = combinedScore;
                nearestCommunity = targetCommunityId;
            }
        }
        
        return nearestCommunity;
    }
    
    /**
     * Calculate geographic proximity between two communities
     * Higher value means communities are closer together
     * 
     * @param community1 First community
     * @param community2 Second community
     * @return Proximity score (higher = closer)
     */
    private double calculateGeographicProximity(List<Node> community1, List<Node> community2) {
        // Find geographic centers of each community
        double c1X = 0.0, c1Y = 0.0, c2X = 0.0, c2Y = 0.0;
        
        for (Node node : community1) {
            c1X += node.getLocation().getX();
            c1Y += node.getLocation().getY();
        }
        c1X /= community1.size();
        c1Y /= community1.size();
        
        for (Node node : community2) {
            c2X += node.getLocation().getX();
            c2Y += node.getLocation().getY();
        }
        c2X /= community2.size();
        c2Y /= community2.size();
        
        // Calculate Euclidean distance between centers
        double distance = Math.sqrt(Math.pow(c1X - c2X, 2) + Math.pow(c1Y - c2Y, 2));
        
        // Convert distance to proximity score (closer = higher score)
        double proximity = Math.max(0.0, 1.0 - (distance / maxGeographicDistance));
        return proximity;
    }
    
    /**
     * Count the number of connections between two communities
     * 
     * @param community1 First community
     * @param community2 Second community
     * @param graph The graph
     * @return The number of connections weighted by edge weights
     */
    private double countConnections(List<Node> community1, List<Node> community2,
                                   Graph<Node, DefaultWeightedEdge> graph) {
        double connections = 0.0;
        
        for (Node node1 : community1) {
            for (Node node2 : community2) {
                DefaultWeightedEdge edge = graph.getEdge(node1, node2);
                
                if (edge != null) {
                    connections += graph.getEdgeWeight(edge);
                }
            }
        }
        
        return connections;
    }
    
    @Override
    public List<List<Node>> findCommunities(TransportationGraph graph) {
        this.transportationGraph = graph;
        Map<Integer, List<Node>> communities = detectCommunities();
        return new ArrayList<>(communities.values());
    }
    
    /**
     * Calculate the description length (code length) of the current community assignment
     * 
     * @param graph The graph
     * @param communities Current community assignments
     * @return The description length
     */
    private double calculateCodeLength(Graph<Node, DefaultWeightedEdge> graph, 
                                      Map<Node, Integer> communities) {
        // Count communities and their sizes
        Map<Integer, Integer> communitySizes = new HashMap<>();
        for (Integer communityId : communities.values()) {
            communitySizes.put(communityId, communitySizes.getOrDefault(communityId, 0) + 1);
        }
        
        // Calculate total edge weight in the graph
        double totalEdgeWeight = 0.0;
        for (DefaultWeightedEdge edge : graph.edgeSet()) {
            totalEdgeWeight += graph.getEdgeWeight(edge);
        }
        
        if (totalEdgeWeight == 0) {
            // No edges, just return number of communities
            return communitySizes.size();
        }
        
        // For each community, calculate internal and external edge weights
        Map<Integer, Double> internalEdgeWeights = new HashMap<>();
        Map<Integer, Double> externalEdgeWeights = new HashMap<>();
        Map<Integer, Double> totalCommunityEdgeWeights = new HashMap<>();
        
        for (DefaultWeightedEdge edge : graph.edgeSet()) {
            Node source = graph.getEdgeSource(edge);
            Node target = graph.getEdgeTarget(edge);
            
            int sourceCommunity = communities.get(source);
            int targetCommunity = communities.get(target);
            
            double weight = graph.getEdgeWeight(edge);
            
            // Update total weights for each community
            totalCommunityEdgeWeights.put(sourceCommunity, 
                    totalCommunityEdgeWeights.getOrDefault(sourceCommunity, 0.0) + weight / 2);
            totalCommunityEdgeWeights.put(targetCommunity, 
                    totalCommunityEdgeWeights.getOrDefault(targetCommunity, 0.0) + weight / 2);
            
            if (sourceCommunity == targetCommunity) {
                // Internal edge
                internalEdgeWeights.put(sourceCommunity, 
                                      internalEdgeWeights.getOrDefault(sourceCommunity, 0.0) + weight);
            } else {
                // External edge (count for both communities)
                externalEdgeWeights.put(sourceCommunity, 
                                      externalEdgeWeights.getOrDefault(sourceCommunity, 0.0) + weight / 2);
                externalEdgeWeights.put(targetCommunity, 
                                      externalEdgeWeights.getOrDefault(targetCommunity, 0.0) + weight / 2);
            }
        }
        
        // Calculate modularity score - higher is better
        double modularity = 0.0;
        
        // Implement a proper modularity calculation
        for (int communityId : communitySizes.keySet()) {
            double internal = internalEdgeWeights.getOrDefault(communityId, 0.0);
            double total = totalCommunityEdgeWeights.getOrDefault(communityId, 0.0);
            double expected = (total * total) / totalEdgeWeight;
            
            modularity += (internal / totalEdgeWeight) - (expected / totalEdgeWeight);
        }
        
        // Calculate information-theoretic complexity penalty
        // This implements a simplified Map Equation concept - fewer communities with good internal connectivity
        double complexity = 0.0;
        for (int communityId : communitySizes.keySet()) {
            int size = communitySizes.get(communityId);
            double fragmentationPenalty = 1.0 / Math.sqrt(size + 1);
            complexity += fragmentationPenalty; 
        }
        
        // Balance between modularity (higher is better) and complexity (lower is better)
        // Note that we return a value where lower is better (to fit with the algorithm)
        double mapEquation = -modularity + 0.05 * complexity;
        
        return mapEquation;
    }
} 