package com.izmir.transportation.helper.clustering;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Queue;
import java.util.Set;

import org.jgrapht.Graph;
import org.jgrapht.alg.connectivity.ConnectivityInspector;
import org.jgrapht.graph.DefaultWeightedEdge;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.izmir.transportation.TransportationGraph;
import com.izmir.transportation.helper.Node;

/**
 * Implementation of the Girvan-Newman community detection algorithm for transportation networks.
 * 
 * This algorithm works by progressively removing edges with high "edge betweenness" to reveal
 * community structures in the network. Edge betweenness is defined as the number of shortest
 * paths between all pairs of vertices that pass through an edge.
 * 
 * The key steps of the algorithm are:
 * 1. Calculate edge betweenness for all edges
 * 2. Remove the edge with highest betweenness
 * 3. Recalculate betweenness for all remaining edges
 * 4. Repeat steps 2-3 until a desired number of communities is reached
 * 
 * @author yagizugurveren
 */
public class GirvanNewmanClustering implements GraphClusteringAlgorithm {
    
    private static final Logger LOGGER = LoggerFactory.getLogger(GirvanNewmanClustering.class);
    
    private final TransportationGraph transportationGraph;
    private int targetCommunityCount = 8; // Default target community count
    private int maxIterations = 100; // Maximum number of edge removals
    private boolean earlyStop = true; // Whether to stop when target communities reached
    private int minCommunitySize = 3; // Minimum size of a community
    private int maxCommunitySize = 45; // Maximum size of a community
    private boolean useModularityMaximization = false; // Whether to use modularity maximization
    
    // Variables to track the best community division based on modularity
    private Map<Integer, List<Node>> bestCommunities;
    private double bestModularity = Double.NEGATIVE_INFINITY;
    
    /**
     * Creates a new GirvanNewmanClustering object with the specified transportation graph.
     * 
     * @param transportationGraph The transportation graph to analyze
     */
    public GirvanNewmanClustering(TransportationGraph transportationGraph) {
        this.transportationGraph = transportationGraph;
    }
    
    /**
     * Sets the target number of communities to detect.
     * 
     * @param count The desired number of communities (must be at least 2)
     * @return This GirvanNewmanClustering instance for method chaining
     */
    public GirvanNewmanClustering setTargetCommunityCount(int count) {
        this.targetCommunityCount = Math.max(2, count);
        LOGGER.info("Target community count set to {}", this.targetCommunityCount);
        return this;
    }
    
    /**
     * Sets the maximum number of edge removal iterations.
     * 
     * @param iterations Maximum iterations (must be at least 1)
     * @return This GirvanNewmanClustering instance for method chaining
     */
    public GirvanNewmanClustering setMaxIterations(int iterations) {
        this.maxIterations = Math.max(1, iterations);
        LOGGER.info("Maximum iterations set to {}", this.maxIterations);
        return this;
    }
    
    /**
     * Sets whether to stop early when the target community count is reached.
     * 
     * @param earlyStop True to stop when target community count is reached, false to always run max iterations
     * @return This GirvanNewmanClustering instance for method chaining
     */
    public GirvanNewmanClustering setEarlyStop(boolean earlyStop) {
        this.earlyStop = earlyStop;
        LOGGER.info("Early stopping set to {}", this.earlyStop);
        return this;
    }
    
    /**
     * Sets the minimum community size. Communities smaller than this will be merged with their nearest neighbor.
     * 
     * @param size Minimum community size (must be at least 1)
     * @return This GirvanNewmanClustering instance for method chaining
     */
    public GirvanNewmanClustering setMinCommunitySize(int size) {
        this.minCommunitySize = Math.max(1, size);
        LOGGER.info("Minimum community size set to {}", this.minCommunitySize);
        return this;
    }
    
    /**
     * Sets the maximum community size. Communities larger than this will be split.
     * 
     * @param size Maximum community size (must be at least minCommunitySize)
     * @return This GirvanNewmanClustering instance for method chaining
     */
    public GirvanNewmanClustering setMaxCommunitySize(int size) {
        this.maxCommunitySize = Math.max(this.minCommunitySize, size);
        LOGGER.info("Maximum community size set to {}", this.maxCommunitySize);
        return this;
    }
    
    /**
     * Sets whether to use modularity maximization to find the optimal number of communities.
     * When enabled, the algorithm will track community divisions that maximize modularity.
     * 
     * @param useModularity True to use modularity maximization
     * @return This GirvanNewmanClustering instance for method chaining
     */
    public GirvanNewmanClustering setUseModularityMaximization(boolean useModularity) {
        this.useModularityMaximization = useModularity;
        LOGGER.info("Modularity maximization set to {}", this.useModularityMaximization);
        return this;
    }
    
    /**
     * Detects communities in the transportation graph using the Girvan-Newman algorithm.
     * 
     * @return A map of community IDs to lists of nodes in each community
     */
    public Map<Integer, List<Node>> detectCommunities() {
        LOGGER.info("Starting Girvan-Newman community detection with target count: {}", targetCommunityCount);
        
        // Get a copy of the graph to work with (so we can remove edges)
        Graph<Node, DefaultWeightedEdge> workingGraph = 
                (Graph<Node, DefaultWeightedEdge>) transportationGraph.getGraph();
        
        int nodeCount = workingGraph.vertexSet().size();
        int edgeCount = workingGraph.edgeSet().size();
        LOGGER.info("Graph has {} nodes and {} edges", nodeCount, edgeCount);
        
        // Calculate edge-to-node ratio to determine graph sparsity
        double edgeToNodeRatio = (double) edgeCount / nodeCount;
        LOGGER.info("Edge-to-node ratio: {}", String.format("%.2f", edgeToNodeRatio));
        
        // For Gabriel graphs with many nodes, use the optimized approach
        // Gabriel graphs typically have edge-to-node ratio around 2.5-3.0
        if (edgeToNodeRatio < 10.0 && nodeCount > 1000) {
            LOGGER.info("Detected sparse graph structure (likely Gabriel). Using specialized approach.");
            return detectCommunitiesForGabrielGraph(workingGraph);
        }
        
        // For very large graphs with many edges, use the faster approach
        if (nodeCount > 1000) {
            LOGGER.warn("Girvan-Newman is computationally intensive for large graphs (O(m²n) complexity)");
            LOGGER.warn("With {} nodes and {} edges, this may take a very long time", nodeCount, edgeCount);
            
            // For very large graphs, we'll use a sampling approach
            if (nodeCount > 1500 && !useModularityMaximization) {
                LOGGER.info("Using faster edge removal strategy for large graph");
                return detectCommunitiesForLargeGraph(workingGraph);
            }
        }
        
        // Create a mutable copy of the graph that we can modify
        Graph<Node, DefaultWeightedEdge> graphCopy = cloneGraph(workingGraph);
        
        // Keep track of edges removed for potential visualization
        List<DefaultWeightedEdge> removedEdges = new ArrayList<>();
        
        // Keep track of the original graph for modularity calculation
        Graph<Node, DefaultWeightedEdge> originalGraph = cloneGraph(workingGraph);
        
        // Initialize best communities tracking
        bestModularity = Double.NEGATIVE_INFINITY;
        bestCommunities = null;
        
        // Main Girvan-Newman loop
        int iteration = 0;
        long startTime = System.currentTimeMillis();
        long lastLogTime = startTime;
        
        while (iteration < maxIterations) {
            iteration++;
            long currentTime = System.currentTimeMillis();
            
            // Log progress every 30 seconds to show the algorithm is still running
            if (currentTime - lastLogTime > 30000) {
                LOGGER.info("Still working on iteration {} (elapsed time: {} seconds)", 
                         iteration, (currentTime - startTime) / 1000);
                lastLogTime = currentTime;
            } else {
                LOGGER.info("Iteration {}/{}", iteration, maxIterations);
            }
            
            // Calculate edge betweenness for all edges
            LOGGER.info("Calculating edge betweenness for {} edges", graphCopy.edgeSet().size());
            long betweennessStart = System.currentTimeMillis();
            
            Map<DefaultWeightedEdge, Double> edgeBetweenness = calculateEdgeBetweenness(graphCopy);
            
            long betweennessEnd = System.currentTimeMillis();
            LOGGER.info("Edge betweenness calculation took {} ms", betweennessEnd - betweennessStart);
            
            // Find edge with maximum betweenness
            DefaultWeightedEdge maxEdge = findMaxBetweennessEdge(edgeBetweenness);
            
            if (maxEdge == null) {
                LOGGER.info("No more edges to remove, stopping algorithm");
                break;
            }
            
            // Remove the edge with highest betweenness
            Node source = graphCopy.getEdgeSource(maxEdge);
            Node target = graphCopy.getEdgeTarget(maxEdge);
            LOGGER.info("Removing edge between {} and {} with betweenness {}", 
                      source.getId(), target.getId(), edgeBetweenness.get(maxEdge));
            
            graphCopy.removeEdge(maxEdge);
            removedEdges.add(maxEdge);
            
            // Check if we've reached the target number of communities
            ConnectivityInspector<Node, DefaultWeightedEdge> inspector = 
                    new ConnectivityInspector<>(graphCopy);
            List<Set<Node>> connectedSets = inspector.connectedSets();
            
            LOGGER.info("Current number of communities: {}", connectedSets.size());
            
            // Calculate modularity if needed
            if (useModularityMaximization) {
                // Convert to community map format
                Map<Integer, List<Node>> currentCommunities = new HashMap<>();
                int communityId = 0;
                for (Set<Node> community : connectedSets) {
                    currentCommunities.put(communityId, new ArrayList<>(community));
                    communityId++;
                }
                
                // Calculate modularity
                double modularity = calculateModularity(currentCommunities, originalGraph);
                LOGGER.info("Current modularity: {}", modularity);
                
                // Update best communities if this is the highest modularity so far
                if (modularity > bestModularity) {
                    bestModularity = modularity;
                    bestCommunities = new HashMap<>(currentCommunities);
                    LOGGER.info("Found new best modularity: {} with {} communities", 
                              bestModularity, bestCommunities.size());
                }
            }
            
            // Stop if we've reached the target community count and early stopping is enabled
            if (earlyStop && connectedSets.size() >= targetCommunityCount) {
                LOGGER.info("Reached target community count of {}, stopping algorithm", targetCommunityCount);
                break;
            }
        }
        
        long endTime = System.currentTimeMillis();
        LOGGER.info("Girvan-Newman algorithm completed in {} seconds", (endTime - startTime) / 1000);
        
        // Get the final communities
        Map<Integer, List<Node>> communities;
        
        if (useModularityMaximization && bestCommunities != null) {
            // Use the communities with best modularity
            LOGGER.info("Using community division with best modularity: {}", bestModularity);
            communities = bestCommunities;
        } else {
            // Use the final state of the algorithm
            ConnectivityInspector<Node, DefaultWeightedEdge> inspector = 
                    new ConnectivityInspector<>(graphCopy);
            List<Set<Node>> connectedSets = inspector.connectedSets();
            
            // Convert to the expected return format
            communities = new HashMap<>();
            int communityId = 0;
            for (Set<Node> community : connectedSets) {
                communities.put(communityId, new ArrayList<>(community));
                communityId++;
            }
        }
        
        LOGGER.info("Detected {} communities using Girvan-Newman algorithm", communities.size());
        
        // Handle small communities if needed
        if (minCommunitySize > 1) {
            communities = mergeSmallCommunities(communities, workingGraph);
            LOGGER.info("After merging small communities: {} communities remain", communities.size());
        }
        
        // Handle large communities
        communities = splitLargeCommunities(communities, workingGraph);
        LOGGER.info("After splitting large communities: {} communities remain", communities.size());
        
        return communities;
    }
    
    /**
     * Specialized method for detecting communities in Gabriel graphs.
     * Gabriel graphs have a specific structure where edges are present only if there 
     * are no other vertices in the circular area between two vertices.
     * 
     * @param workingGraph The graph to analyze
     * @return A map of community IDs to lists of nodes in each community
     */
    private Map<Integer, List<Node>> detectCommunitiesForGabrielGraph(Graph<Node, DefaultWeightedEdge> workingGraph) {
        LOGGER.info("Using specialized algorithm for Gabriel graph");
        
        // Get the total number of nodes in the graph
        int nodeCount = workingGraph.vertexSet().size();
        
        // Create a mutable copy of the graph that we can modify
        Graph<Node, DefaultWeightedEdge> graphCopy = cloneGraph(workingGraph);
        
        // Precompute geographical distances for all edges
        Map<DefaultWeightedEdge, Double> edgeDistances = new HashMap<>();
        for (DefaultWeightedEdge edge : graphCopy.edgeSet()) {
            Node source = graphCopy.getEdgeSource(edge);
            Node target = graphCopy.getEdgeTarget(edge);
            double distance = calculateGeographicalDistance(source, target);
            edgeDistances.put(edge, distance);
        }
        
        // Calculate average and standard deviation of edge distances
        double totalDistance = 0;
        for (double distance : edgeDistances.values()) {
            totalDistance += distance;
        }
        double averageDistance = totalDistance / edgeDistances.size();
        
        double sumSquaredDiff = 0;
        for (double distance : edgeDistances.values()) {
            sumSquaredDiff += Math.pow(distance - averageDistance, 2);
        }
        double stdDevDistance = Math.sqrt(sumSquaredDiff / edgeDistances.size());
        
        LOGGER.info("Average edge distance: {} km, Standard deviation: {} km", 
                 String.format("%.2f", averageDistance), 
                 String.format("%.2f", stdDevDistance));
        
        // We'll focus on removing edges that are:
        // 1. Significantly longer than average (potential bridges between communities)
        // 2. Have high "bridge coefficient" (connect different regions)
        
        // Track removed edges to avoid processing them again
        Set<DefaultWeightedEdge> removedEdges = new HashSet<>();
        
        // Process edges in batches, starting with the most likely bridge edges
        List<DefaultWeightedEdge> allEdges = new ArrayList<>(graphCopy.edgeSet());
        
        // Sort edges by a combination of distance and bridge coefficient
        allEdges.sort((e1, e2) -> {
            double score1 = calculateBridgeScore(graphCopy, e1, edgeDistances);
            double score2 = calculateBridgeScore(graphCopy, e2, edgeDistances);
            return Double.compare(score2, score1); // Higher scores first
        });
        
        // Prioritize edges with high bridge scores - start with more aggressive removal
        int iteration = 0;
        double percentageToRemove = 0.03; // Start by removing 3% of edges per iteration (increased from 1%)
        double maxRemovalRate = 0.15; // Maximum removal rate of 15% per iteration (increased from 10%)
        
        while (iteration < maxIterations) {
            iteration++;
            LOGGER.info("Iteration {}/{}", iteration, maxIterations);
            
            // Determine how many edges to remove this iteration
            int edgesToRemove = Math.max(1, (int)(allEdges.size() * percentageToRemove));
            int removed = 0;
            
            // Remove edges with highest bridge scores first
            for (int i = 0; i < allEdges.size() && removed < edgesToRemove; i++) {
                DefaultWeightedEdge edge = allEdges.get(i);
                
                // Skip if already removed
                if (removedEdges.contains(edge) || !graphCopy.containsEdge(edge)) {
                    continue;
                }
                
                // Get the nodes connected by this edge
                Node source = graphCopy.getEdgeSource(edge);
                Node target = graphCopy.getEdgeTarget(edge);
                
                // More permissive removal - only prevent completely isolated nodes
                if (graphCopy.degreeOf(source) <= 1 || graphCopy.degreeOf(target) <= 1) {
                    continue;
                }
                
                // Calculate bridge score - if it's high enough, remove regardless of other factors
                double bridgeScore = calculateBridgeScore(graphCopy, edge, edgeDistances);
                boolean isLikelyBridge = bridgeScore > (averageDistance * 2);
                
                // If the edge distance is more than 1.5 standard deviations above average, it's likely a bridge
                double edgeDistance = edgeDistances.getOrDefault(edge, 0.0);
                boolean isLongEdge = edgeDistance > (averageDistance + (1.5 * stdDevDistance));
                
                // If this is likely a bridge edge, we'll remove it regardless of other factors
                if (isLikelyBridge || isLongEdge) {
                    // Remove the edge
                    graphCopy.removeEdge(edge);
                    removedEdges.add(edge);
                    removed++;
                    
                    LOGGER.debug("Removed edge with bridge score {} between nodes {} and {}", 
                               bridgeScore, source.getId(), target.getId());
                    continue;
                }
                
                // Remove the edge if it doesn't disconnect the graph too much
                graphCopy.removeEdge(edge);
                removedEdges.add(edge);
                removed++;
                
                LOGGER.debug("Removed edge with bridge score {} between nodes {} and {}", 
                           bridgeScore, source.getId(), target.getId());
            }
            
            LOGGER.info("Removed {} edges in iteration {}", removed, iteration);
            
            // If we couldn't remove any edges, increase percentage or break
            if (removed == 0) {
                percentageToRemove *= 1.8; // Increase by 80% (more aggressive than before)
                if (percentageToRemove > maxRemovalRate) { // Cap at maximum rate
                    LOGGER.info("Reached maximum edge removal rate, stopping");
                    break;
                }
                LOGGER.info("Increasing edge removal rate to {}%", percentageToRemove * 100);
                continue;
            }
            
            // Check if we've reached the target number of communities
            ConnectivityInspector<Node, DefaultWeightedEdge> inspector = 
                    new ConnectivityInspector<>(graphCopy);
            List<Set<Node>> connectedSets = inspector.connectedSets();
            
            LOGGER.info("Current number of communities: {}", connectedSets.size());
            
            // Get the distribution of community sizes
            List<Integer> communitySizes = new ArrayList<>();
            for (Set<Node> community : connectedSets) {
                communitySizes.add(community.size());
            }
            communitySizes.sort((a, b) -> Integer.compare(b, a)); // Sort descending
            
            // Log the sizes of the top 5 communities (or fewer if there are less than 5)
            StringBuilder sizesStr = new StringBuilder("Community sizes: ");
            for (int i = 0; i < Math.min(5, communitySizes.size()); i++) {
                sizesStr.append(communitySizes.get(i));
                if (i < Math.min(5, communitySizes.size()) - 1) {
                    sizesStr.append(", ");
                }
            }
            if (communitySizes.size() > 5) {
                sizesStr.append(", ...");
            }
            LOGGER.info(sizesStr.toString());
            
            // Don't stop early if there's a dominant community that's too large
            boolean hasLargeDominantCommunity = false;
            if (!communitySizes.isEmpty()) {
                // If the largest community contains more than 75% of nodes
                int largestSize = communitySizes.get(0);
                if (largestSize > nodeCount * 0.75) {
                    hasLargeDominantCommunity = true;
                    LOGGER.info("Largest community still has {}% of nodes, continuing...", 
                             String.format("%.1f", (100.0 * largestSize / nodeCount)));
                }
            }
            
            // Stop if we've reached the target community count AND we don't have a dominant community
            if (connectedSets.size() >= targetCommunityCount && !hasLargeDominantCommunity) {
                LOGGER.info("Reached target community count of {}, stopping algorithm", targetCommunityCount);
                break;
            }
            
            // If we're close to the target but there's a dominant community, continue more aggressively
            if (connectedSets.size() >= targetCommunityCount / 2 && hasLargeDominantCommunity) {
                LOGGER.info("Close to target but still have dominant community, increasing removal rate");
                percentageToRemove = Math.min(maxRemovalRate, percentageToRemove * 1.5);
            }
            
            // Reorganize remaining edges more frequently
            if (iteration % 3 == 0) { // Every 3 iterations instead of 5
                allEdges = new ArrayList<>(graphCopy.edgeSet());
                
                // Recalculate bridge scores
                allEdges.sort((e1, e2) -> {
                    double score1 = calculateBridgeScore(graphCopy, e1, edgeDistances);
                    double score2 = calculateBridgeScore(graphCopy, e2, edgeDistances);
                    return Double.compare(score2, score1); // Higher scores first
                });
                
                // Increase removal rate gradually
                percentageToRemove = Math.min(maxRemovalRate, percentageToRemove * 1.3);
                LOGGER.info("Updated edge list and increased removal rate to {}%", percentageToRemove * 100);
            }
        }
        
        // Get the final communities
        ConnectivityInspector<Node, DefaultWeightedEdge> inspector = 
                new ConnectivityInspector<>(graphCopy);
        List<Set<Node>> connectedSets = inspector.connectedSets();
        
        // Convert to the expected return format
        Map<Integer, List<Node>> communities = new HashMap<>();
        int communityId = 0;
        for (Set<Node> community : connectedSets) {
            communities.put(communityId, new ArrayList<>(community));
            communityId++;
        }
        
        LOGGER.info("Detected {} communities using Gabriel-optimized approach", communities.size());
        
        // If we still have a single dominant community and need to partition further,
        // apply a recursive subdivision approach to the large community
        List<Integer> communitySizes = new ArrayList<>();
        int largestCommunityId = -1;
        int largestCommunitySize = 0;
        
        for (Map.Entry<Integer, List<Node>> entry : communities.entrySet()) {
            int size = entry.getValue().size();
            communitySizes.add(size);
            
            if (size > largestCommunitySize) {
                largestCommunitySize = size;
                largestCommunityId = entry.getKey();
            }
        }
        
        // If the largest community is more than 70% of the total and we have fewer than half the target communities
        if (largestCommunitySize > nodeCount * 0.7 && communities.size() < targetCommunityCount / 2) {
            LOGGER.info("Largest community has {}% of nodes - applying recursive subdivision", 
                     String.format("%.1f", (100.0 * largestCommunitySize / nodeCount)));
            
            // Extract the largest community for further subdivision
            List<Node> largestCommunity = communities.get(largestCommunityId);
            
            // Create a subgraph of just this community
            Graph<Node, DefaultWeightedEdge> subgraph = new org.jgrapht.graph.SimpleWeightedGraph<>(DefaultWeightedEdge.class);
            
            // Add all nodes from the community
            for (Node node : largestCommunity) {
                subgraph.addVertex(node);
            }
            
            // Add edges between nodes in this community
            for (Node source : largestCommunity) {
                for (Node target : largestCommunity) {
                    if (source.equals(target)) continue;
                    
                    DefaultWeightedEdge edge = workingGraph.getEdge(source, target);
                    if (edge != null) {
                        DefaultWeightedEdge newEdge = subgraph.addEdge(source, target);
                        if (newEdge != null) {
                            subgraph.setEdgeWeight(newEdge, workingGraph.getEdgeWeight(edge));
                        }
                    }
                }
            }
            
            // Apply a more aggressive subdivision to this community
            int targetSubCommunities = Math.max(targetCommunityCount - communities.size() + 1, 
                                            targetCommunityCount / 2);
            
            LOGGER.info("Attempting to subdivide largest community into {} subcommunities", targetSubCommunities);
            
            // Initialize subcommunities with nodes sorted by geographical position
            List<List<Node>> initialGroups = createGeographicalGroups(largestCommunity, targetSubCommunities);
            
            // Use these initial groups to help split the large community
            int subCommunityId = communities.size();
            for (List<Node> group : initialGroups) {
                if (!group.isEmpty()) {
                    communities.put(subCommunityId++, group);
                }
            }
            
            // Remove the original large community
            communities.remove(largestCommunityId);
            
            LOGGER.info("After subdivision: {} communities", communities.size());
        }
        
        // Handle small communities if needed
        if (minCommunitySize > 1) {
            communities = mergeSmallCommunities(communities, workingGraph);
            LOGGER.info("After merging small communities: {} communities remain", communities.size());
        }
        
        return communities;
    }
    
    /**
     * Creates geographical groups of nodes for initial community division.
     * This helps provide a better starting point for large community subdivision.
     * 
     * @param nodes List of nodes to group
     * @param numGroups Number of groups to create
     * @return List of node groups divided by geographical position
     */
    private List<List<Node>> createGeographicalGroups(List<Node> nodes, int numGroups) {
        List<List<Node>> groups = new ArrayList<>();
        
        // First, create empty groups
        for (int i = 0; i < numGroups; i++) {
            groups.add(new ArrayList<>());
        }
        
        // Sort nodes by latitude (north to south)
        List<Node> sortedNodes = new ArrayList<>(nodes);
        sortedNodes.sort((n1, n2) -> {
            double lat1 = n1.getLocation().getY();
            double lat2 = n2.getLocation().getY();
            return Double.compare(lat2, lat1); // North to south (higher latitude first)
        });
        
        // Distribute nodes evenly among groups
        for (int i = 0; i < sortedNodes.size(); i++) {
            int groupIndex = i % numGroups;
            groups.get(groupIndex).add(sortedNodes.get(i));
        }
        
        return groups;
    }
    
    /**
     * Calculates a "bridge score" for an edge indicating how likely it is to be a bridge between communities.
     * Higher scores indicate edges that are more likely to be between different communities.
     * 
     * @param graph The graph
     * @param edge The edge to evaluate
     * @param edgeDistances Map of edges to their geographical distances
     * @return A score indicating how likely the edge is to be a bridge
     */
    private double calculateBridgeScore(Graph<Node, DefaultWeightedEdge> graph, 
                                      DefaultWeightedEdge edge,
                                      Map<DefaultWeightedEdge, Double> edgeDistances) {
        if (!graph.containsEdge(edge)) {
            return 0.0; // Edge not in graph
        }
        
        Node source = graph.getEdgeSource(edge);
        Node target = graph.getEdgeTarget(edge);
        
        // Factor 1: Distance - longer edges are more likely to be between communities
        double distance = edgeDistances.getOrDefault(edge, 0.0);
        
        // Factor 2: Local connectivity - edges that connect different dense regions
        Set<Node> sourceNeighbors = new HashSet<>();
        Set<Node> targetNeighbors = new HashSet<>();
        
        for (DefaultWeightedEdge e : graph.edgesOf(source)) {
            if (!e.equals(edge)) {
                sourceNeighbors.add(getOppositeNode(graph, e, source));
            }
        }
        
        for (DefaultWeightedEdge e : graph.edgesOf(target)) {
            if (!e.equals(edge)) {
                targetNeighbors.add(getOppositeNode(graph, e, target));
            }
        }
        
        // Calculate overlap between neighborhoods
        Set<Node> commonNeighbors = new HashSet<>(sourceNeighbors);
        commonNeighbors.retainAll(targetNeighbors);
        int neighborOverlap = commonNeighbors.size();
        
        // The less overlap in neighborhoods, the more likely this is a bridge edge
        // (nodes in the same community tend to share many neighbors)
        double connectivityFactor = 1.0 - (neighborOverlap / (double)Math.max(1, Math.min(sourceNeighbors.size(), targetNeighbors.size())));
        
        // Factor 3: Clustering coefficient difference
        double sourceClusteringCoeff = calculateLocalClusteringCoefficient(graph, source);
        double targetClusteringCoeff = calculateLocalClusteringCoefficient(graph, target);
        double clusteringDiff = Math.abs(sourceClusteringCoeff - targetClusteringCoeff);
        
        // Factor 4: Geographical direction - edges that cross perpendicular to the main population flow
        // can often be bridges between communities
        double directionFactor = 1.0;
        
        // Calculate the angle this edge makes with the north-south axis
        double lat1 = source.getLocation().getY();
        double lon1 = source.getLocation().getX();
        double lat2 = target.getLocation().getY();
        double lon2 = target.getLocation().getX();
        
        // This factor will be higher for east-west edges compared to north-south edges
        // (assuming most transportation flows along the north-south axis)
        double latDiff = Math.abs(lat1 - lat2);
        double lonDiff = Math.abs(lon1 - lon2);
        directionFactor = 1.0 + (lonDiff / (latDiff + 0.0001)); // Avoid division by zero
        
        // Combine factors - bridge score is higher when:
        // - Distance is greater
        // - Neighborhoods have less overlap
        // - Clustering coefficients are different
        // - Edge runs perpendicular to main population flow
        return distance * (1.0 + connectivityFactor) * (1.0 + clusteringDiff) * directionFactor;
    }
    
    /**
     * Calculates the local clustering coefficient for a node.
     * This measures how close the node's neighbors are to being a complete graph.
     * 
     * @param graph The graph
     * @param node The node to calculate for
     * @return The local clustering coefficient
     */
    private double calculateLocalClusteringCoefficient(Graph<Node, DefaultWeightedEdge> graph, Node node) {
        Set<Node> neighbors = new HashSet<>();
        
        // Get all neighbors of the node
        for (DefaultWeightedEdge edge : graph.edgesOf(node)) {
            neighbors.add(getOppositeNode(graph, edge, node));
        }
        
        // If the node has less than 2 neighbors, clustering coefficient is 0
        if (neighbors.size() < 2) {
            return 0.0;
        }
        
        // Count connections between neighbors
        int connections = 0;
        for (Node n1 : neighbors) {
            for (Node n2 : neighbors) {
                if (n1.equals(n2)) continue;
                
                if (graph.getEdge(n1, n2) != null) {
                    connections++;
                }
            }
        }
        
        // Each connection is counted twice (once from each end)
        connections /= 2;
        
        // Maximum possible connections between n neighbors is n*(n-1)/2
        int maxConnections = (neighbors.size() * (neighbors.size() - 1)) / 2;
        
        // Clustering coefficient is the ratio of actual connections to possible connections
        return (maxConnections > 0) ? connections / (double)maxConnections : 0.0;
    }
    
    /**
     * Calculates edge betweenness for all edges in the graph.
     * 
     * Edge betweenness is defined as the number of shortest paths between all pairs of vertices
     * that pass through an edge. This implementation uses Brandes' algorithm adapted for edge betweenness.
     * 
     * @param graph The graph to analyze
     * @return A map of edges to their betweenness values
     */
    private Map<DefaultWeightedEdge, Double> calculateEdgeBetweenness(Graph<Node, DefaultWeightedEdge> graph) {
        Map<DefaultWeightedEdge, Double> betweenness = new HashMap<>();
        
        // Initialize all edges with betweenness of 0
        for (DefaultWeightedEdge edge : graph.edgeSet()) {
            betweenness.put(edge, 0.0);
        }
        
        // Get all nodes
        List<Node> nodes = new ArrayList<>(graph.vertexSet());
        int nodeCount = nodes.size();
        
        // Log progress periodically
        long startTime = System.currentTimeMillis();
        int logInterval = Math.max(1, nodeCount / 20); // Log every 5%
        
        // For each vertex, calculate its contribution to edge betweenness
        for (int i = 0; i < nodeCount; i++) {
            Node source = nodes.get(i);
            
            // Log progress periodically
            if (i % logInterval == 0 || i == nodeCount - 1) {
                LOGGER.debug("Processing node {} of {} ({}%)", 
                           i + 1, nodeCount, (int)((i + 1) * 100.0 / nodeCount));
            }
            
            // Part 1: Perform BFS from the source node to calculate shortest paths
            Map<Node, Integer> distance = new HashMap<>(); // Distance from source
            Map<Node, Double> pathCount = new HashMap<>(); // Number of shortest paths
            Map<Node, Double> dependency = new HashMap<>(); // Dependency of source on vertex
            Map<Node, List<Node>> predecessors = new HashMap<>(); // Predecessors in shortest paths
            
            // Initialize maps
            for (Node v : graph.vertexSet()) {
                distance.put(v, Integer.MAX_VALUE);
                pathCount.put(v, 0.0);
                dependency.put(v, 0.0);
                predecessors.put(v, new ArrayList<>());
            }
            
            // BFS to calculate shortest paths
            distance.put(source, 0);
            pathCount.put(source, 1.0);
            
            Queue<Node> queue = new LinkedList<>();
            queue.add(source);
            
            List<Node> stack = new ArrayList<>(); // Stack for vertices in order of non-increasing distance
            
            while (!queue.isEmpty()) {
                Node v = queue.poll();
                stack.add(v);
                
                for (DefaultWeightedEdge edge : graph.edgesOf(v)) {
                    Node w = getOppositeNode(graph, edge, v);
                    
                    // If discovering w for the first time
                    if (distance.get(w) == Integer.MAX_VALUE) {
                        distance.put(w, distance.get(v) + 1);
                        queue.add(w);
                    }
                    
                    // If w is on a shortest path from s through v
                    if (distance.get(w) == distance.get(v) + 1) {
                        pathCount.put(w, pathCount.get(w) + pathCount.get(v));
                        predecessors.get(w).add(v);
                    }
                }
            }
            
            // Part 2: Calculate edge betweenness using accumulated dependencies
            while (!stack.isEmpty()) {
                Node w = stack.remove(stack.size() - 1);
                
                for (Node v : predecessors.get(w)) {
                    // The amount of dependency that w adds to v
                    double edgeDependency = (pathCount.get(v) / pathCount.get(w)) * (1.0 + dependency.get(w));
                    dependency.put(v, dependency.get(v) + edgeDependency);
                    
                    // Find edge between v and w
                    DefaultWeightedEdge edge = graph.getEdge(v, w);
                    if (edge == null) {
                        edge = graph.getEdge(w, v);
                    }
                    
                    if (edge != null) {
                        // Add to edge betweenness
                        betweenness.put(edge, betweenness.get(edge) + edgeDependency);
                    }
                }
            }
        }
        
        long endTime = System.currentTimeMillis();
        LOGGER.debug("Betweenness calculation completed in {} ms for {} nodes", 
                   endTime - startTime, nodeCount);
        
        return betweenness;
    }
    
    /**
     * Finds the edge with the maximum betweenness value.
     * 
     * @param betweenness Map of edges to their betweenness values
     * @return The edge with the highest betweenness, or null if the map is empty
     */
    private DefaultWeightedEdge findMaxBetweennessEdge(Map<DefaultWeightedEdge, Double> betweenness) {
        DefaultWeightedEdge maxEdge = null;
        double maxValue = Double.NEGATIVE_INFINITY;
        
        for (Map.Entry<DefaultWeightedEdge, Double> entry : betweenness.entrySet()) {
            if (entry.getValue() > maxValue) {
                maxValue = entry.getValue();
                maxEdge = entry.getKey();
            }
        }
        
        return maxEdge;
    }
    
    /**
     * Gets the node at the other end of an edge.
     * 
     * @param graph The graph containing the edge
     * @param edge The edge to get the opposite node from
     * @param node The known node at one end of the edge
     * @return The node at the other end of the edge
     */
    private Node getOppositeNode(Graph<Node, DefaultWeightedEdge> graph, DefaultWeightedEdge edge, Node node) {
        Node source = graph.getEdgeSource(edge);
        Node target = graph.getEdgeTarget(edge);
        
        if (source.equals(node)) {
            return target;
        } else {
            return source;
        }
    }
    
    /**
     * Creates a deep copy of the graph that we can modify without affecting the original.
     * 
     * @param original The original graph
     * @return A new graph with the same vertices and edges
     */
    private Graph<Node, DefaultWeightedEdge> cloneGraph(Graph<Node, DefaultWeightedEdge> original) {
        // Create a new graph of the same type
        Graph<Node, DefaultWeightedEdge> clone = new org.jgrapht.graph.SimpleWeightedGraph<>(DefaultWeightedEdge.class);
        
        // Add all vertices
        for (Node node : original.vertexSet()) {
            clone.addVertex(node);
        }
        
        // Add all edges with their weights
        for (DefaultWeightedEdge edge : original.edgeSet()) {
            Node source = original.getEdgeSource(edge);
            Node target = original.getEdgeTarget(edge);
            DefaultWeightedEdge newEdge = clone.addEdge(source, target);
            
            if (newEdge != null) {
                clone.setEdgeWeight(newEdge, original.getEdgeWeight(edge));
            }
        }
        
        return clone;
    }
    
    /**
     * Merges communities smaller than the minimum size with their nearest neighboring community.
     * 
     * @param communities The initial community assignments
     * @param graph The original graph (used to calculate connections between communities)
     * @return A new map with small communities merged
     */
    private Map<Integer, List<Node>> mergeSmallCommunities(
            Map<Integer, List<Node>> communities, 
            Graph<Node, DefaultWeightedEdge> graph) {
        
        // Identify small communities
        List<Integer> smallCommunityIds = new ArrayList<>();
        for (Map.Entry<Integer, List<Node>> entry : communities.entrySet()) {
            if (entry.getValue().size() < minCommunitySize) {
                smallCommunityIds.add(entry.getKey());
            }
        }
        
        LOGGER.info("Found {} communities smaller than minimum size {}", 
                smallCommunityIds.size(), minCommunitySize);
        
        if (smallCommunityIds.isEmpty()) {
            return communities;
        }
        
        // Create a new map to hold the merged communities
        Map<Integer, List<Node>> mergedCommunities = new HashMap<>(communities);
        
        // Process each small community
        for (Integer smallCommunityId : smallCommunityIds) {
            List<Node> smallCommunity = mergedCommunities.get(smallCommunityId);
            
            if (smallCommunity == null || smallCommunity.isEmpty()) {
                continue; // Skip if already merged
            }
            
            // Find the best community to merge with
            int bestCommunityId = -1;
            double bestScore = Double.NEGATIVE_INFINITY;
            
            for (Map.Entry<Integer, List<Node>> entry : mergedCommunities.entrySet()) {
                int communityId = entry.getKey();
                List<Node> community = entry.getValue();
                
                // Skip the community itself and other small communities
                if (communityId == smallCommunityId || community.size() < minCommunitySize) {
                    continue;
                }
                
                // Calculate connection score between communities
                double score = calculateConnectionScore(smallCommunity, community, graph);
                
                if (score > bestScore) {
                    bestScore = score;
                    bestCommunityId = communityId;
                }
            }
            
            // If we found a community to merge with
            if (bestCommunityId != -1) {
                LOGGER.debug("Merging community {} (size {}) with community {} (score {})",
                          smallCommunityId, smallCommunity.size(), bestCommunityId, bestScore);
                
                // Add nodes from small community to the best community
                mergedCommunities.get(bestCommunityId).addAll(smallCommunity);
                
                // Remove the small community
                mergedCommunities.remove(smallCommunityId);
            }
        }
        
        return mergedCommunities;
    }
    
    /**
     * Calculates a connection score between two communities based on their connectivity in the graph.
     * 
     * @param community1 First community
     * @param community2 Second community
     * @param graph The graph containing the communities
     * @return A score representing the strength of connection between communities
     */
    private double calculateConnectionScore(List<Node> community1, List<Node> community2, 
                                         Graph<Node, DefaultWeightedEdge> graph) {
        double totalConnections = 0;
        
        for (Node node1 : community1) {
            for (Node node2 : community2) {
                DefaultWeightedEdge edge = graph.getEdge(node1, node2);
                
                if (edge != null) {
                    // Add the inverse of the edge weight (shorter/lighter edges are stronger connections)
                    double weight = graph.getEdgeWeight(edge);
                    totalConnections += (weight > 0) ? 1.0 / weight : 1.0;
                }
            }
        }
        
        // Normalize by the number of possible connections
        int possibleConnections = community1.size() * community2.size();
        return (possibleConnections > 0) ? totalConnections / possibleConnections : 0;
    }
    
    /**
     * Calculates the modularity of a given community division.
     * Modularity measures the strength of division of a network into communities.
     * Higher values indicate better community structure.
     * 
     * @param communities Map of community IDs to lists of nodes
     * @param graph The original graph
     * @return Modularity value
     */
    private double calculateModularity(Map<Integer, List<Node>> communities, Graph<Node, DefaultWeightedEdge> graph) {
        double modularity = 0.0;
        double totalWeight = calculateTotalEdgeWeight(graph);
        
        if (totalWeight == 0) {
            return 0.0;
        }
        
        // Calculate modularity based on the weighted formula
        for (List<Node> community : communities.values()) {
            for (Node node1 : community) {
                for (Node node2 : community) {
                    // Skip self-loops
                    if (node1.equals(node2)) {
                        continue;
                    }
                    
                    // Calculate actual connection
                    DefaultWeightedEdge edge = graph.getEdge(node1, node2);
                    double actualWeight = (edge != null) ? graph.getEdgeWeight(edge) : 0.0;
                    
                    // Calculate expected connection
                    double node1Strength = calculateNodeStrength(node1, graph);
                    double node2Strength = calculateNodeStrength(node2, graph);
                    double expectedWeight = (node1Strength * node2Strength) / (2 * totalWeight);
                    
                    // Add to modularity
                    modularity += (actualWeight - expectedWeight) / totalWeight;
                }
            }
        }
        
        // Divide by 2 because each edge is counted twice in undirected graphs
        return modularity / 2.0;
    }
    
    /**
     * Calculates the total weight of all edges in the graph.
     * 
     * @param graph The graph
     * @return Total edge weight
     */
    private double calculateTotalEdgeWeight(Graph<Node, DefaultWeightedEdge> graph) {
        double totalWeight = 0.0;
        
        for (DefaultWeightedEdge edge : graph.edgeSet()) {
            totalWeight += graph.getEdgeWeight(edge);
        }
        
        return totalWeight;
    }
    
    /**
     * Calculates the strength of a node, which is the sum of weights of all its edges.
     * 
     * @param node The node
     * @param graph The graph
     * @return Node strength
     */
    private double calculateNodeStrength(Node node, Graph<Node, DefaultWeightedEdge> graph) {
        double strength = 0.0;
        
        for (DefaultWeightedEdge edge : graph.edgesOf(node)) {
            strength += graph.getEdgeWeight(edge);
        }
        
        return strength;
    }
    
    /**
     * Calculates geographical distance between two nodes in kilometers.
     * 
     * @param node1 First node
     * @param node2 Second node
     * @return Distance in kilometers
     */
    private double calculateGeographicalDistance(Node node1, Node node2) {
        // Extract coordinates
        double lon1 = node1.getLocation().getX();
        double lat1 = node1.getLocation().getY();
        double lon2 = node2.getLocation().getX();
        double lat2 = node2.getLocation().getY();
        
        // Convert degrees to radians
        double lat1Rad = Math.toRadians(lat1);
        double lon1Rad = Math.toRadians(lon1);
        double lat2Rad = Math.toRadians(lat2);
        double lon2Rad = Math.toRadians(lon2);
        
        // Haversine formula for distance between two points on a sphere
        double dlon = lon2Rad - lon1Rad;
        double dlat = lat2Rad - lat1Rad;
        double a = Math.pow(Math.sin(dlat / 2), 2) + 
                  Math.cos(lat1Rad) * Math.cos(lat2Rad) * Math.pow(Math.sin(dlon / 2), 2);
        double c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
        
        // Earth radius in kilometers
        double radius = 6371.0;
        
        // Distance in kilometers
        return radius * c;
    }
    
    /**
     * Modified version of the Girvan-Newman algorithm for large graphs.
     * This uses a more efficient approach by considering edge weight and
     * removing multiple edges per iteration to speed up the process.
     * 
     * @param workingGraph The graph to analyze
     * @return A map of community IDs to lists of nodes in each community
     */
    private Map<Integer, List<Node>> detectCommunitiesForLargeGraph(Graph<Node, DefaultWeightedEdge> workingGraph) {
        LOGGER.info("Using optimized algorithm for large graph");
        
        // Create a mutable copy of the graph that we can modify
        Graph<Node, DefaultWeightedEdge> graphCopy = cloneGraph(workingGraph);
        
        // For very large graphs, we'll use edge weight as a proxy for betweenness
        // and remove multiple edges per iteration to speed things up
        int iteration = 0;
        
        // For large dense graphs, we need to be more aggressive in edge removal
        double edgeRemovalFactor = 0.005; // 0.5% of edges per iteration
        int totalEdges = graphCopy.edgeSet().size();
        int edgesToRemovePerIteration = Math.max(1, (int)(totalEdges * edgeRemovalFactor));
        
        LOGGER.info("Will remove up to {} edges per iteration ({} of graph edges)", 
                  edgesToRemovePerIteration, edgeRemovalFactor * 100);
        
        // Initialize centrality measures based on edge weights and node degrees
        Map<DefaultWeightedEdge, Double> edgeCentrality = new HashMap<>();
        
        // Track removed edges
        Set<DefaultWeightedEdge> removedEdges = new HashSet<>();
        
        while (iteration < maxIterations) {
            iteration++;
            LOGGER.info("Iteration {}/{}", iteration, maxIterations);
            
            // Get current graph size
            int currentEdgeCount = graphCopy.edgeSet().size();
            if (currentEdgeCount == 0) break;
            
            // Calculate a simplified form of edge centrality based on edge weights and node degrees
            edgeCentrality.clear();
            for (DefaultWeightedEdge edge : graphCopy.edgeSet()) {
                Node source = graphCopy.getEdgeSource(edge);
                Node target = graphCopy.getEdgeTarget(edge);
                
                // Use a combination of edge weight and degree as a proxy for betweenness
                double weight = graphCopy.getEdgeWeight(edge);
                double sourceDegree = graphCopy.degreeOf(source);
                double targetDegree = graphCopy.degreeOf(target);
                
                // Edges with high weight between high-degree nodes are likely to have high betweenness
                double centrality = (sourceDegree + targetDegree) * weight;
                edgeCentrality.put(edge, centrality);
            }
            
            // Sort edges by centrality
            List<DefaultWeightedEdge> sortedEdges = new ArrayList<>(edgeCentrality.keySet());
            sortedEdges.sort((e1, e2) -> Double.compare(edgeCentrality.get(e2), edgeCentrality.get(e1)));
            
            // Remove top N edges with highest centrality
            int edgesToRemove = Math.min(edgesToRemovePerIteration, sortedEdges.size());
            int edgesRemoved = 0;
            
            for (int i = 0; i < edgesToRemove; i++) {
                if (i >= sortedEdges.size()) break;
                
                DefaultWeightedEdge edge = sortedEdges.get(i);
                
                // Skip if already removed
                if (removedEdges.contains(edge) || !graphCopy.containsEdge(edge)) {
                    continue;
                }
                
                Node source = graphCopy.getEdgeSource(edge);
                Node target = graphCopy.getEdgeTarget(edge);
                
                // Skip removing if it would create isolated nodes
                if (graphCopy.degreeOf(source) <= 1 || graphCopy.degreeOf(target) <= 1) {
                    continue;
                }
                
                // Remove the edge
                graphCopy.removeEdge(edge);
                removedEdges.add(edge);
                edgesRemoved++;
            }
            
            LOGGER.info("Removed {} edges in iteration {}", edgesRemoved, iteration);
            
            // If we couldn't remove any edges, try to increase the edge removal rate
            if (edgesRemoved == 0) {
                edgeRemovalFactor *= 1.5;
                if (edgeRemovalFactor > 0.1) { // Cap at 10%
                    LOGGER.info("Reached maximum edge removal rate, stopping");
                    break;
                }
                edgesToRemovePerIteration = Math.max(1, (int)(totalEdges * edgeRemovalFactor));
                LOGGER.info("Increased edge removal rate to {}% ({} edges per iteration)", 
                         edgeRemovalFactor * 100, edgesToRemovePerIteration);
                continue;
            }
            
            // Check if we've reached the target number of communities
            ConnectivityInspector<Node, DefaultWeightedEdge> inspector = 
                    new ConnectivityInspector<>(graphCopy);
            List<Set<Node>> connectedSets = inspector.connectedSets();
            
            LOGGER.info("Current number of communities: {}", connectedSets.size());
            
            // Stop if we've reached the target community count
            if (connectedSets.size() >= targetCommunityCount) {
                LOGGER.info("Reached target community count of {}, stopping algorithm", targetCommunityCount);
                break;
            }
        }
        
        // Get the final communities
        ConnectivityInspector<Node, DefaultWeightedEdge> inspector = 
                new ConnectivityInspector<>(graphCopy);
        List<Set<Node>> connectedSets = inspector.connectedSets();
        
        // Convert to the expected return format
        Map<Integer, List<Node>> communities = new HashMap<>();
        int communityId = 0;
        for (Set<Node> community : connectedSets) {
            communities.put(communityId, new ArrayList<>(community));
            communityId++;
        }
        
        LOGGER.info("Detected {} communities using optimized approach", communities.size());
        
        // Handle small communities if needed
        if (minCommunitySize > 1) {
            communities = mergeSmallCommunities(communities, workingGraph);
            LOGGER.info("After merging small communities: {} communities remain", communities.size());
        }
        
        return communities;
    }
    
    /**
     * Splits communities that exceed the maximum size into smaller communities.
     * 
     * @param communities The initial community assignments
     * @param graph The original graph (used to create subgraphs for large communities)
     * @return A new map with large communities split into smaller ones
     */
    private Map<Integer, List<Node>> splitLargeCommunities(
            Map<Integer, List<Node>> communities, 
            Graph<Node, DefaultWeightedEdge> graph) {
        
        // Identify large communities
        List<Integer> largeCommunityIds = new ArrayList<>();
        for (Map.Entry<Integer, List<Node>> entry : communities.entrySet()) {
            if (entry.getValue().size() > maxCommunitySize) {
                largeCommunityIds.add(entry.getKey());
            }
        }
        
        LOGGER.info("Found {} communities larger than maximum size {}", 
                largeCommunityIds.size(), maxCommunitySize);
        
        if (largeCommunityIds.isEmpty()) {
            return communities;
        }
        
        // Create a new map to hold the split communities
        Map<Integer, List<Node>> splitCommunities = new HashMap<>(communities);
        int nextCommunityId = communities.size();
        
        // Process each large community
        for (Integer largeCommunityId : largeCommunityIds) {
            List<Node> largeCommunity = splitCommunities.get(largeCommunityId);
            
            if (largeCommunity == null || largeCommunity.isEmpty()) {
                continue; // Skip if already processed
            }
            
            int communitySize = largeCommunity.size();
            LOGGER.info("Splitting community {} with size {}", largeCommunityId, communitySize);
            
            // For very large communities (> 5x max size), use hierarchical splitting
            if (communitySize > maxCommunitySize * 5) {
                LOGGER.info("Very large community detected ({}x max size). Using hierarchical splitting.", 
                         communitySize / maxCommunitySize);
                
                // Remove the original large community
                splitCommunities.remove(largeCommunityId);
                
                // Use hierarchical splitting approach
                List<List<Node>> subCommunities = splitLargeCommunityHierarchically(
                        largeCommunity, graph, maxCommunitySize, minCommunitySize);
                
                // Add all valid subcommunities to the result
                for (List<Node> subCommunity : subCommunities) {
                    if (subCommunity.size() >= minCommunitySize) {
                        splitCommunities.put(nextCommunityId++, subCommunity);
                        LOGGER.info("Created subcommunity with size {}", subCommunity.size());
                    }
                }
                
                continue; // Skip to next large community
            }
            
            // For moderately large communities, use standard splitting
            
            // Calculate how many subcommunities we need
            int numSubCommunities = (int) Math.ceil((double) communitySize / maxCommunitySize);
            LOGGER.info("Creating {} subcommunities", numSubCommunities);
            
            // Create a subgraph of just this community
            Graph<Node, DefaultWeightedEdge> subgraph = createSubgraph(largeCommunity, graph);
            
            // If community is at least twice the minimum size, we can safely split it
            if (communitySize >= minCommunitySize * 2) {
                List<List<Node>> subCommunities;
                
                // First try to use edge betweenness for better community preservation
                LOGGER.info("Attempting to split community using edge betweenness");
                subCommunities = splitByEdgeBetweenness(subgraph, numSubCommunities, minCommunitySize);
                
                // Check if the split produced valid subcommunities and none exceed max size
                boolean validSplit = true;
                for (List<Node> subCommunity : subCommunities) {
                    if (subCommunity.size() < minCommunitySize || subCommunity.size() > maxCommunitySize) {
                        validSplit = false;
                        break;
                    }
                }
                
                // If edge betweenness didn't work well, try spatial clustering
                if (!validSplit) {
                    LOGGER.info("Edge betweenness produced invalid subcommunities, trying spatial clustering");
                    subCommunities = splitBySpatialClustering(largeCommunity, numSubCommunities);
                    
                    // Verify spatial clusters are valid
                    validSplit = true;
                    for (List<Node> subCommunity : subCommunities) {
                        if (subCommunity.size() < minCommunitySize || subCommunity.size() > maxCommunitySize) {
                            validSplit = false;
                            break;
                        }
                    }
                    
                    // If still not valid, try direct partitioning as last resort
                    if (!validSplit) {
                        LOGGER.info("Spatial clustering failed. Using direct partitioning as last resort.");
                        subCommunities = partitionCommunityDirectly(largeCommunity, maxCommunitySize, minCommunitySize);
                    }
                }
                
                // Add the new subcommunities to our map
                splitCommunities.remove(largeCommunityId);
                
                for (List<Node> subCommunity : subCommunities) {
                    if (subCommunity.size() >= minCommunitySize) {
                        splitCommunities.put(nextCommunityId++, subCommunity);
                        LOGGER.info("Created subcommunity with size {}", subCommunity.size());
                    } else {
                        LOGGER.info("Discarding subcommunity with size {} < minimum {}", 
                                  subCommunity.size(), minCommunitySize);
                    }
                }
            } else {
                // If the community is too small to split safely, leave it as is
                LOGGER.info("Community {} is too small to split while respecting minimum size, keeping as is", 
                          largeCommunityId);
            }
        }
        
        // Check if we still have communities exceeding maximum size
        boolean hasLargeCommunities = false;
        for (List<Node> community : splitCommunities.values()) {
            if (community.size() > maxCommunitySize) {
                hasLargeCommunities = true;
                LOGGER.info("Still have community with size {} > max size {}", 
                         community.size(), maxCommunitySize);
                break;
            }
        }
        
        // If we still have large communities, recursively split them
        if (hasLargeCommunities) {
            LOGGER.info("Some communities still exceed maximum size. Performing another round of splitting.");
            return splitLargeCommunities(splitCommunities, graph);
        }
        
        return splitCommunities;
    }
    
    /**
     * Hierarchically splits a very large community into subcommunities.
     * This uses a divide-and-conquer approach for very large communities.
     * 
     * @param community The large community to split
     * @param graph The original graph for edge information
     * @param maxSize Maximum community size
     * @param minSize Minimum community size
     * @return A list of subcommunities
     */
    private List<List<Node>> splitLargeCommunityHierarchically(
            List<Node> community, 
            Graph<Node, DefaultWeightedEdge> graph, 
            int maxSize, 
            int minSize) {
        
        int communitySize = community.size();
        LOGGER.info("Hierarchically splitting community with {} nodes", communitySize);
        
        // Base case: If community is small enough, return it as is
        if (communitySize <= maxSize) {
            return List.of(community);
        }
        
        // Create a subgraph for this community
        Graph<Node, DefaultWeightedEdge> subgraph = createSubgraph(community, graph);
        
        // For very large communities, first split into a small number of large subcommunities
        int initialDivisions = Math.min(10, (int)Math.ceil(communitySize / (double)(maxSize * 3)));
        LOGGER.info("First dividing into {} intermediate communities", initialDivisions);
        
        // Use edge betweenness to split into initial divisions (better preserves structure)
        List<List<Node>> intermediateCommunities = splitByEdgeBetweenness(
                subgraph, initialDivisions, minSize);
        
        // If edge betweenness fails or creates poor divisions, try spatial clustering
        if (intermediateCommunities.size() < 2) {
            LOGGER.info("Edge betweenness failed, trying spatial clustering for initial division");
            intermediateCommunities = new ArrayList<>(splitBySpatialClustering(community, initialDivisions));
            
            // Check if spatial clustering produced valid results
            boolean hasInvalidCommunities = false;
            for (List<Node> comm : intermediateCommunities) {
                if (comm.size() < minSize) {
                    hasInvalidCommunities = true;
                    break;
                }
            }
            
            // If spatial clustering fails or creates too few communities, use geographical partitioning
            if (hasInvalidCommunities || intermediateCommunities.size() < 2) {
                LOGGER.info("Spatial clustering failed for initial division, using geographical partitioning");
                intermediateCommunities = new ArrayList<>(partitionByGeography(community, initialDivisions));
            }
        }
        
        // Recursively split each intermediate community if it's still too large
        List<List<Node>> finalCommunities = new ArrayList<>();
        
        for (List<Node> intermediateComm : intermediateCommunities) {
            if (intermediateComm.size() <= maxSize) {
                // If this community is already within size limits, add it directly
                finalCommunities.add(intermediateComm);
            } else {
                // Recursively split this intermediate community
                List<List<Node>> splitResults = splitLargeCommunityHierarchically(
                        intermediateComm, graph, maxSize, minSize);
                finalCommunities.addAll(splitResults);
            }
        }
        
        // Ensure no tiny communities are created
        List<List<Node>> validCommunities = new ArrayList<>();
        List<Node> nodesInTinyCommunities = new ArrayList<>();
        
        for (List<Node> comm : finalCommunities) {
            if (comm.size() >= minSize) {
                validCommunities.add(comm);
            } else {
                nodesInTinyCommunities.addAll(comm);
            }
        }
        
        // Handle any leftover nodes from tiny communities
        if (!nodesInTinyCommunities.isEmpty()) {
            LOGGER.info("Redistributing {} nodes from tiny communities", nodesInTinyCommunities.size());
            
            if (validCommunities.isEmpty()) {
                // If we have no valid communities, create one from the tiny community nodes
                if (nodesInTinyCommunities.size() >= minSize) {
                    validCommunities.add(nodesInTinyCommunities);
                } else {
                    // If we can't even form one valid community, return the original (this shouldn't happen)
                    LOGGER.warn("Failed to create valid communities. Returning original community.");
                    return List.of(community);
                }
            } else {
                // Add leftover nodes to the smallest valid communities
                redistributeLeftoverNodes(nodesInTinyCommunities, validCommunities, maxSize);
            }
        }
        
        LOGGER.info("Hierarchical splitting completed. Created {} subcommunities", validCommunities.size());
        return validCommunities;
    }
    
    /**
     * Redistributes leftover nodes to the smallest valid communities.
     * 
     * @param leftoverNodes Nodes to redistribute
     * @param validCommunities List of valid communities
     * @param maxSize Maximum community size
     */
    private void redistributeLeftoverNodes(
            List<Node> leftoverNodes, 
            List<List<Node>> validCommunities,
            int maxSize) {
        
        // Sort communities by size (smallest first)
        validCommunities.sort(Comparator.comparingInt(List::size));
        
        // Add leftover nodes to the smallest communities
        for (Node node : leftoverNodes) {
            // Find the smallest community that has room
            for (List<Node> community : validCommunities) {
                if (community.size() < maxSize) {
                    community.add(node);
                    break;
                }
            }
        }
    }
    
    /**
     * Partitions a community directly into approximately equal-sized subcommunities.
     * This is a last resort when other clustering methods fail.
     * 
     * @param community The community to partition
     * @param maxSize Maximum allowed size for a subcommunity
     * @param minSize Minimum allowed size for a subcommunity
     * @return A list of subcommunities
     */
    private List<List<Node>> partitionCommunityDirectly(
            List<Node> community, 
            int maxSize, 
            int minSize) {
        
        LOGGER.info("Directly partitioning community with {} nodes", community.size());
        
        int numPartitions = (int) Math.ceil(community.size() / (double)maxSize);
        int targetSize = community.size() / numPartitions;
        
        LOGGER.info("Creating {} partitions with target size ~{}", numPartitions, targetSize);
        
        // Sort by geographic location for more sensible partitioning
        List<Node> sortedNodes = new ArrayList<>(community);
        sortedNodes.sort((n1, n2) -> {
            double lat1 = n1.getLocation().getY();
            double lat2 = n2.getLocation().getY();
            if (Math.abs(lat1 - lat2) < 0.01) {
                // If latitudes are close, sort by longitude
                return Double.compare(n1.getLocation().getX(), n2.getLocation().getX());
            }
            return Double.compare(lat1, lat2);
        });
        
        // Create partitions
        List<List<Node>> partitions = new ArrayList<>();
        List<Node> currentPartition = new ArrayList<>();
        
        for (Node node : sortedNodes) {
            currentPartition.add(node);
            
            // When partition reaches target size, start a new one
            if (currentPartition.size() >= targetSize && partitions.size() < numPartitions - 1) {
                partitions.add(currentPartition);
                currentPartition = new ArrayList<>();
            }
        }
        
        // Add the last partition
        if (!currentPartition.isEmpty()) {
            partitions.add(currentPartition);
        }
        
        // Validate partitions
        List<List<Node>> validPartitions = new ArrayList<>();
        List<Node> nodesInInvalidPartitions = new ArrayList<>();
        
        for (List<Node> partition : partitions) {
            if (partition.size() >= minSize) {
                validPartitions.add(partition);
            } else {
                nodesInInvalidPartitions.addAll(partition);
            }
        }
        
        // Redistribute nodes from invalid partitions
        if (!nodesInInvalidPartitions.isEmpty() && !validPartitions.isEmpty()) {
            redistributeLeftoverNodes(nodesInInvalidPartitions, validPartitions, maxSize);
        } else if (!nodesInInvalidPartitions.isEmpty()) {
            // If all partitions are invalid, create one partition with all nodes
            validPartitions.add(new ArrayList<>(community));
        }
        
        return validPartitions;
    }
    
    /**
     * Partitions a community based on geographical areas.
     * Divides the area into a grid and assigns nodes to cells.
     * 
     * @param community The community to partition
     * @param numPartitions Approximate number of partitions to create
     * @return A list of subcommunities
     */
    private List<List<Node>> partitionByGeography(List<Node> community, int numPartitions) {
        LOGGER.info("Partitioning community by geography into ~{} partitions", numPartitions);
        
        // Find the geographical bounds of the community
        double minLat = Double.MAX_VALUE;
        double maxLat = Double.MIN_VALUE;
        double minLon = Double.MAX_VALUE;
        double maxLon = Double.MIN_VALUE;
        
        for (Node node : community) {
            double lat = node.getLocation().getY();
            double lon = node.getLocation().getX();
            
            minLat = Math.min(minLat, lat);
            maxLat = Math.max(maxLat, lat);
            minLon = Math.min(minLon, lon);
            maxLon = Math.max(maxLon, lon);
        }
        
        // Calculate grid dimensions
        // We'll create a grid with cells that are roughly square in shape
        double latRange = maxLat - minLat;
        double lonRange = maxLon - minLon;
        
        // Adjust for Earth's coordinate system (longitude degrees vary in distance by latitude)
        // Rough approximation: at latitude φ, 1° longitude = cos(φ) * 111.32 km
        double midLat = (minLat + maxLat) / 2;
        double lonAdjustment = Math.cos(Math.toRadians(midLat));
        double adjustedLonRange = lonRange * lonAdjustment;
        
        // Calculate the number of rows and columns for our grid
        int rows, cols;
        if (latRange >= adjustedLonRange) {
            rows = (int) Math.ceil(Math.sqrt(numPartitions * latRange / adjustedLonRange));
            cols = (int) Math.ceil(numPartitions / (double) rows);
        } else {
            cols = (int) Math.ceil(Math.sqrt(numPartitions * adjustedLonRange / latRange));
            rows = (int) Math.ceil(numPartitions / (double) cols);
        }
        
        LOGGER.info("Created geographical grid with {} rows and {} columns", rows, cols);
        
        // Create partitions grid
        Map<String, List<Node>> grid = new HashMap<>();
        
        // Assign nodes to grid cells
        for (Node node : community) {
            double lat = node.getLocation().getY();
            double lon = node.getLocation().getX();
            
            // Calculate grid cell
            int row = Math.min(rows - 1, (int) Math.floor((lat - minLat) / latRange * rows));
            int col = Math.min(cols - 1, (int) Math.floor((lon - minLon) / lonRange * cols));
            
            String cellId = row + "," + col;
            grid.computeIfAbsent(cellId, k -> new ArrayList<>()).add(node);
        }
        
        // Convert grid cells to partitions
        List<List<Node>> partitions = new ArrayList<>(grid.values());
        
        // Filter out empty partitions
        partitions.removeIf(List::isEmpty);
        
        LOGGER.info("Created {} geographical partitions", partitions.size());
        
        // Merge very small partitions with neighbors
        if (minCommunitySize > 1) {
            partitions = mergeSmallPartitions(partitions, community, minCommunitySize);
        }
        
        return partitions;
    }
    
    /**
     * Merges small partitions with their geographical neighbors.
     * 
     * @param partitions List of partitions
     * @param originalCommunity The original community (for centroid calculation)
     * @param minSize Minimum allowed size for a partition
     * @return List of merged partitions
     */
    private List<List<Node>> mergeSmallPartitions(
            List<List<Node>> partitions, 
            List<Node> originalCommunity, 
            int minSize) {
        
        // Filter partitions
        List<List<Node>> validPartitions = new ArrayList<>();
        List<List<Node>> smallPartitions = new ArrayList<>();
        
        for (List<Node> partition : partitions) {
            if (partition.size() >= minSize) {
                validPartitions.add(partition);
            } else {
                smallPartitions.add(partition);
            }
        }
        
        LOGGER.info("Found {} partitions below minimum size {}", smallPartitions.size(), minSize);
        
        // If all partitions are small, combine them all
        if (validPartitions.isEmpty() && !smallPartitions.isEmpty()) {
            List<Node> combined = new ArrayList<>();
            for (List<Node> partition : smallPartitions) {
                combined.addAll(partition);
            }
            return List.of(combined);
        }
        
        // For each small partition, merge with the closest valid partition
        for (List<Node> smallPartition : smallPartitions) {
            if (smallPartition.isEmpty()) continue;
            
            // Find the closest valid partition
            double bestDistance = Double.MAX_VALUE;
            List<Node> closestPartition = null;
            
            // Calculate centroid of small partition
            double smallLat = 0, smallLon = 0;
            for (Node node : smallPartition) {
                smallLat += node.getLocation().getY();
                smallLon += node.getLocation().getX();
            }
            smallLat /= smallPartition.size();
            smallLon /= smallPartition.size();
            
            // Find closest valid partition
            for (List<Node> validPartition : validPartitions) {
                // Calculate centroid of valid partition
                double validLat = 0, validLon = 0;
                for (Node node : validPartition) {
                    validLat += node.getLocation().getY();
                    validLon += node.getLocation().getX();
                }
                validLat /= validPartition.size();
                validLon /= validPartition.size();
                
                // Calculate distance between centroids
                double distance = Math.sqrt(
                    Math.pow(smallLat - validLat, 2) + 
                    Math.pow(smallLon - validLon, 2)
                );
                
                if (distance < bestDistance) {
                    bestDistance = distance;
                    closestPartition = validPartition;
                }
            }
            
            // Merge with closest partition
            if (closestPartition != null) {
                closestPartition.addAll(smallPartition);
            }
        }
        
        LOGGER.info("After merging small partitions: {} partitions remain", validPartitions.size());
        return validPartitions;
    }
    
    /**
     * Creates a subgraph containing only the specified nodes and edges between them.
     * 
     * @param nodes The nodes to include in the subgraph
     * @param graph The original graph
     * @return A new graph containing only the specified nodes and edges between them
     */
    private Graph<Node, DefaultWeightedEdge> createSubgraph(List<Node> nodes, Graph<Node, DefaultWeightedEdge> graph) {
        Graph<Node, DefaultWeightedEdge> subgraph = new org.jgrapht.graph.SimpleWeightedGraph<>(DefaultWeightedEdge.class);
        
        // Add all nodes
        for (Node node : nodes) {
            subgraph.addVertex(node);
        }
        
        // Add edges between nodes
        for (int i = 0; i < nodes.size(); i++) {
            Node source = nodes.get(i);
            for (int j = i + 1; j < nodes.size(); j++) {
                Node target = nodes.get(j);
                DefaultWeightedEdge edge = graph.getEdge(source, target);
                if (edge != null) {
                    DefaultWeightedEdge newEdge = subgraph.addEdge(source, target);
                    if (newEdge != null) {
                        subgraph.setEdgeWeight(newEdge, graph.getEdgeWeight(edge));
                    }
                }
            }
        }
        
        return subgraph;
    }
    
    /**
     * Splits a community into subcommunities based on spatial clustering.
     * Uses K-means clustering on the geographical coordinates.
     * 
     * @param community The community to split
     * @param numSubCommunities The target number of subcommunities
     * @return A list of subcommunities
     */
    private List<List<Node>> splitBySpatialClustering(List<Node> community, int numSubCommunities) {
        LOGGER.info("Splitting community by spatial clustering into {} subcommunities", numSubCommunities);
        
        // Use a simple K-means approach for spatial clustering
        
        // 1. Initialize K centroids randomly
        List<double[]> centroids = new ArrayList<>();
        for (int i = 0; i < numSubCommunities; i++) {
            Node randomNode = community.get((int) (Math.random() * community.size()));
            double[] centroid = new double[] {
                randomNode.getLocation().getX(),
                randomNode.getLocation().getY()
            };
            centroids.add(centroid);
        }
        
        // 2. Create K empty clusters
        List<List<Node>> clusters = new ArrayList<>();
        for (int i = 0; i < numSubCommunities; i++) {
            clusters.add(new ArrayList<>());
        }
        
        // 3. Assign nodes to clusters based on distance to centroids
        boolean changed = true;
        int iterations = 0;
        int maxIterations = 100;
        
        while (changed && iterations < maxIterations) {
            changed = false;
            iterations++;
            
            // Clear clusters
            for (List<Node> cluster : clusters) {
                cluster.clear();
            }
            
            // Assign each node to nearest centroid
            for (Node node : community) {
                int closestCentroidIndex = 0;
                double minDistance = Double.MAX_VALUE;
                
                for (int i = 0; i < centroids.size(); i++) {
                    double[] centroid = centroids.get(i);
                    double distance = Math.pow(node.getLocation().getX() - centroid[0], 2) +
                                     Math.pow(node.getLocation().getY() - centroid[1], 2);
                    
                    if (distance < minDistance) {
                        minDistance = distance;
                        closestCentroidIndex = i;
                    }
                }
                
                clusters.get(closestCentroidIndex).add(node);
            }
            
            // Calculate new centroids
            for (int i = 0; i < numSubCommunities; i++) {
                if (clusters.get(i).isEmpty()) {
                    continue;
                }
                
                double sumX = 0;
                double sumY = 0;
                
                for (Node node : clusters.get(i)) {
                    sumX += node.getLocation().getX();
                    sumY += node.getLocation().getY();
                }
                
                double[] newCentroid = new double[] {
                    sumX / clusters.get(i).size(),
                    sumY / clusters.get(i).size()
                };
                
                // Check if centroid has moved
                if (Math.abs(newCentroid[0] - centroids.get(i)[0]) > 0.0001 ||
                    Math.abs(newCentroid[1] - centroids.get(i)[1]) > 0.0001) {
                    changed = true;
                }
                
                centroids.set(i, newCentroid);
            }
        }
        
        LOGGER.info("Spatial clustering completed in {} iterations", iterations);
        
        // If we have empty clusters or clusters below min size, try to redistribute
        boolean hasBadClusters = false;
        for (List<Node> cluster : clusters) {
            if (cluster.size() < minCommunitySize) {
                hasBadClusters = true;
                break;
            }
        }
        
        if (hasBadClusters) {
            LOGGER.info("Some clusters are below minimum size, attempting to redistribute");
            clusters = redistributeNodes(clusters);
        }
        
        return clusters;
    }
    
    /**
     * Redistributes nodes from overpopulated clusters to underpopulated ones.
     * 
     * @param clusters The initial clusters
     * @return Redistributed clusters
     */
    private List<List<Node>> redistributeNodes(List<List<Node>> clusters) {
        // Create clones of the original clusters
        List<List<Node>> redistributed = new ArrayList<>();
        for (List<Node> cluster : clusters) {
            redistributed.add(new ArrayList<>(cluster));
        }
        
        // Sort clusters by size (largest first)
        redistributed.sort((c1, c2) -> Integer.compare(c2.size(), c1.size()));
        
        // Calculate average cluster size
        int totalNodes = 0;
        for (List<Node> cluster : redistributed) {
            totalNodes += cluster.size();
        }
        int avgSize = totalNodes / redistributed.size();
        
        // Try to balance the clusters while respecting minimum size
        for (int i = 0; i < redistributed.size(); i++) {
            List<Node> cluster = redistributed.get(i);
            
            // If this cluster is too small, transfer nodes from larger clusters
            if (cluster.size() < minCommunitySize) {
                int needed = minCommunitySize - cluster.size();
                
                // Try to get nodes from clusters above average size
                for (int j = 0; j < redistributed.size() && needed > 0; j++) {
                    if (i == j) continue;
                    
                    List<Node> donor = redistributed.get(j);
                    if (donor.size() > avgSize) {
                        int toTransfer = Math.min(needed, donor.size() - avgSize);
                        
                        // Only transfer if the donor would still respect minimum size
                        if (donor.size() - toTransfer >= minCommunitySize) {
                            for (int k = 0; k < toTransfer; k++) {
                                Node node = donor.remove(donor.size() - 1);
                                cluster.add(node);
                                needed--;
                            }
                        }
                    }
                }
            }
        }
        
        return redistributed;
    }
    
    /**
     * Splits a graph into subcommunities using the edge betweenness approach.
     * 
     * @param graph The graph to split
     * @param numSubCommunities The target number of subcommunities
     * @param minSize The minimum size of a subcommunity
     * @return A list of subcommunities
     */
    private List<List<Node>> splitByEdgeBetweenness(
            Graph<Node, DefaultWeightedEdge> graph, 
            int numSubCommunities, 
            int minSize) {
        
        LOGGER.info("Splitting community by edge betweenness into {} subcommunities", numSubCommunities);
        
        // Create a mutable copy of the graph that we can modify
        Graph<Node, DefaultWeightedEdge> graphCopy = cloneGraph(graph);
        
        // Keep removing edges with highest betweenness until we have enough communities
        int iteration = 0;
        int maxIterations = 100;
        
        while (iteration < maxIterations) {
            iteration++;
            
            // Calculate edge betweenness for all edges
            Map<DefaultWeightedEdge, Double> edgeBetweenness = calculateEdgeBetweenness(graphCopy);
            
            // Find edge with maximum betweenness
            DefaultWeightedEdge maxEdge = findMaxBetweennessEdge(edgeBetweenness);
            
            if (maxEdge == null) {
                LOGGER.info("No more edges to remove, stopping algorithm");
                break;
            }
            
            // Remove the edge with highest betweenness
            graphCopy.removeEdge(maxEdge);
            
            // Check if we've reached the target number of communities
            ConnectivityInspector<Node, DefaultWeightedEdge> inspector = 
                    new ConnectivityInspector<>(graphCopy);
            List<Set<Node>> connectedSets = inspector.connectedSets();
            
            // Check if we have enough valid communities (above minimum size)
            int validCommunities = 0;
            for (Set<Node> community : connectedSets) {
                if (community.size() >= minSize) {
                    validCommunities++;
                }
            }
            
            LOGGER.debug("Iteration {}: {} connected components, {} valid communities",
                      iteration, connectedSets.size(), validCommunities);
            
            if (validCommunities >= numSubCommunities) {
                LOGGER.info("Reached target number of valid communities: {}", validCommunities);
                break;
            }
        }
        
        // Get the final communities
        ConnectivityInspector<Node, DefaultWeightedEdge> inspector = 
                new ConnectivityInspector<>(graphCopy);
        List<Set<Node>> connectedSets = inspector.connectedSets();
        
        // Convert to the expected return format and filter by minimum size
        List<List<Node>> communities = new ArrayList<>();
        for (Set<Node> community : connectedSets) {
            if (community.size() >= minSize) {
                communities.add(new ArrayList<>(community));
            }
        }
        
        LOGGER.info("Edge betweenness splitting found {} communities", communities.size());
        return communities;
    }
    
    /**
     * Implements the GraphClusteringAlgorithm interface method.
     * 
     * @param graph The transportation graph to analyze
     * @return A list of communities, where each community is a list of nodes
     */
    @Override
    public List<List<Node>> findCommunities(TransportationGraph graph) {
        // Use the instance method for consistency with our internal graph
        Map<Integer, List<Node>> communityMap = detectCommunities();
        
        // Convert the map to a list of lists
        List<List<Node>> communities = new ArrayList<>(communityMap.values());
        return communities;
    }
} 