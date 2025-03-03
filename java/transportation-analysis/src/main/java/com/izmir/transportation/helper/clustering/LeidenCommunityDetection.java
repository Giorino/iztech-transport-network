package com.izmir.transportation.helper.clustering;

import java.util.ArrayList;
import java.util.Arrays;
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

import com.izmir.transportation.TransportationGraph;
import com.izmir.transportation.helper.Node;
import com.izmir.transportation.helper.clustering.leiden.Clustering;
import com.izmir.transportation.helper.clustering.leiden.LeidenAlgorithm;
import com.izmir.transportation.helper.clustering.leiden.Network;
import com.izmir.transportation.helper.clustering.leiden.util.LargeDoubleArray;
import com.izmir.transportation.helper.clustering.leiden.util.LargeIntArray;

/**
 * Implementation of Leiden community detection algorithm for transportation network analysis.
 * This class provides methods to detect communities within a transportation network
 * using the Leiden algorithm, which is known for finding well-connected communities.
 * 
 * @author yagizugurveren
 */
public class LeidenCommunityDetection {
    
    // Default values for the algorithm - adjusted for producing fewer, more meaningful communities
    private static final double DEFAULT_RESOLUTION = 0.005; // Increased from 0.001 to allow more communities
    private static final int DEFAULT_ITERATIONS = 50; // Keep high iteration count for optimization
    private static final double DEFAULT_RANDOMNESS = 0.001; // Keep low randomness for deterministic results
    
    // Configuration parameters to control community detection
    private double communityScalingFactor = 0.5; // Controls how many communities to target (higher = more communities)
    private int minCommunities = 2; // Minimum number of communities to aim for
    private int maxCommunities = 8; // Maximum number of communities to allow before merging
    private boolean adaptiveResolution = true; // Whether to use adaptive resolution values based on network size
    
    private final TransportationGraph transportationGraph;
    private Clustering clustering;
    private Map<Integer, List<Node>> communities;
    private boolean useOriginalPointsOnly = false;
    
    /**
     * Constructs a LeidenCommunityDetection instance for the given transportation graph.
     * 
     * @param transportationGraph The transportation graph to analyze
     */
    public LeidenCommunityDetection(TransportationGraph transportationGraph) {
        this.transportationGraph = transportationGraph;
        this.communities = new HashMap<>();
    }
    
    /**
     * Sets whether to use only original points for community detection or the entire graph.
     * Original points typically represent actual locations rather than road network points.
     * 
     * @param useOriginalPointsOnly True to use only original points, false to use all nodes
     */
    public void useOriginalPointsOnly(boolean useOriginalPointsOnly) {
        this.useOriginalPointsOnly = useOriginalPointsOnly;
    }
    
    /**
     * Sets the desired scaling factor for the number of communities.
     * Higher values result in more communities.
     * 
     * @param scalingFactor A value between 0.1 and 2.0 (default is 0.5)
     */
    public void setCommunityScalingFactor(double scalingFactor) {
        this.communityScalingFactor = Math.max(0.1, Math.min(2.0, scalingFactor));
    }
    
    /**
     * Sets the minimum and maximum number of communities to aim for.
     * 
     * @param min Minimum number of communities
     * @param max Maximum number of communities
     */
    public void setCommunityCountLimits(int min, int max) {
        this.minCommunities = Math.max(2, min);
        this.maxCommunities = Math.max(this.minCommunities, max);
    }
    
    /**
     * Sets whether to use adaptive resolution values based on network size.
     * 
     * @param adaptive True to use adaptive resolution, false to use fixed values
     */
    public void setAdaptiveResolution(boolean adaptive) {
        this.adaptiveResolution = adaptive;
    }
    
    /**
     * Performs community detection using the Leiden algorithm with default parameters.
     * 
     * @return A map of community IDs to lists of nodes in each community
     */
    public Map<Integer, List<Node>> detectCommunities() {
        // Get graph size to determine appropriate parameters
        int graphSize = useOriginalPointsOnly ? 
            transportationGraph.getOriginalPointsGraph().vertexSet().size() : 
            transportationGraph.getGraph().vertexSet().size();
            
        System.out.println("Performing community detection on a graph with " + graphSize + " nodes");
            
        // Set resolution values based on graph size
        double[] resolutionValues;
        
        if (adaptiveResolution) {
            // Adaptive resolution values based on graph size
            if (graphSize <= 100) {
                // For small networks, use larger resolution values (fewer communities)
                resolutionValues = new double[]{0.001, 0.005, 0.01, 0.05, 0.1, 0.2};
            } else if (graphSize <= 500) {
                // For medium networks
                resolutionValues = new double[]{0.0005, 0.001, 0.005, 0.01, 0.05, 0.1};
            } else if (graphSize <= 1000) {
                // For larger networks
                resolutionValues = new double[]{0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05};
            } else {
                // For very large networks (1000+ nodes)
                resolutionValues = new double[]{0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01};
            }
        } else {
            // Use fixed resolution values
            resolutionValues = new double[]{0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2};
        }
        
        Map<Integer, List<Node>> bestCommunities = null;
        double bestBalanceScore = Double.MAX_VALUE;
        Clustering bestClustering = null;
        double bestResolution = 0;
        
        //System.out.println("Searching for optimal community structure by testing different resolution parameters...");
        
        for (double resolution : resolutionValues) {
            //System.out.println("Trying resolution = " + resolution);
            Map<Integer, List<Node>> testCommunities = detectCommunities(resolution, DEFAULT_ITERATIONS, DEFAULT_RANDOMNESS);
            
            // Assess balance quality - lower is better
            double balanceScore = assessCommunityBalance(testCommunities, graphSize);
            
            //System.out.println("  - Found " + testCommunities.size() + " communities with balance score " + 
            //                   String.format("%.2f", balanceScore));
            
            // Keep track of the best result - only report improvement when we actually have one
            if (balanceScore < bestBalanceScore) {
                bestBalanceScore = balanceScore;
                bestCommunities = new HashMap<>(testCommunities);
                bestClustering = clustering.clone();
                bestResolution = resolution;
                System.out.println("  - New best community structure! (resolution = " + resolution + ")");
            }
        }
        
        // Use the best community structure found
        this.communities = bestCommunities;
        this.clustering = bestClustering;
        
        // Log the best resolution value
        System.out.println("Using best community structure found with resolution = " + bestResolution + 
                           " and balance score " + String.format("%.2f", bestBalanceScore));
        
        // Provide specific details about the community structure
        int singletonCount = countSingletonCommunities(communities);
        int largestCommunitySize = findLargestCommunitySize(communities);
        int totalNodes = countTotalNodes(communities);
        
        System.out.println("Structure details before rebalancing:");
        System.out.println("  - Communities: " + communities.size());
        System.out.println("  - Singleton communities: " + singletonCount + " (" + 
                           String.format("%.1f", (singletonCount * 100.0 / communities.size())) + "%)");
        System.out.println("  - Largest community size: " + largestCommunitySize + " (" + 
                           String.format("%.1f", (largestCommunitySize * 100.0 / totalNodes)) + "% of nodes)");
        
        // Check if the communities are extremely imbalanced
        if (isExtremelyImbalanced()) {
            System.out.println("Communities are extremely imbalanced. Applying mild correction...");
            mildlyRebalanceCommunities();
            
            // Update statistics after rebalancing
            singletonCount = countSingletonCommunities(communities);
            largestCommunitySize = findLargestCommunitySize(communities);
            System.out.println("Structure details after rebalancing:");
            System.out.println("  - Communities: " + communities.size());
            System.out.println("  - Singleton communities: " + singletonCount + " (" + 
                               String.format("%.1f", (singletonCount * 100.0 / communities.size())) + "%)");
            System.out.println("  - Largest community size: " + largestCommunitySize + " (" + 
                               String.format("%.1f", (largestCommunitySize * 100.0 / totalNodes)) + "% of nodes)");
        }
        
        // Apply spatial post-processing to ensure geographic coherence
        enhanceSpatialCoherence();
        
        return communities;
    }
    
    /**
     * Count the number of singleton communities (communities with only one node)
     *
     * @param communityMap The map of communities to analyze
     * @return The number of singleton communities
     */
    private int countSingletonCommunities(Map<Integer, List<Node>> communityMap) {
        int count = 0;
        for (List<Node> nodes : communityMap.values()) {
            if (nodes.size() == 1) {
                count++;
            }
        }
        return count;
    }
    
    /**
     * Find the size of the largest community
     *
     * @param communityMap The map of communities to analyze
     * @return The size of the largest community
     */
    private int findLargestCommunitySize(Map<Integer, List<Node>> communityMap) {
        int maxSize = 0;
        for (List<Node> nodes : communityMap.values()) {
            if (nodes.size() > maxSize) {
                maxSize = nodes.size();
            }
        }
        return maxSize;
    }
    
    /**
     * Count the total number of nodes across all communities
     *
     * @param communityMap The map of communities to analyze
     * @return The total number of nodes
     */
    private int countTotalNodes(Map<Integer, List<Node>> communityMap) {
        int count = 0;
        for (List<Node> nodes : communityMap.values()) {
            count += nodes.size();
        }
        return count;
    }
    
    /**
     * Performs community detection using the Leiden algorithm with custom parameters.
     * 
     * @param resolution Resolution parameter (controls size of communities, smaller value = larger communities)
     * @param iterations Number of iterations to run
     * @param randomness Randomness parameter (affects node assignment)
     * @return A map of community IDs to lists of nodes in each community
     */
    public Map<Integer, List<Node>> detectCommunities(double resolution, int iterations, double randomness) {
        // Convert the transportation graph to a format compatible with the Leiden algorithm
        Network network = convertToNetwork();
        
        // Run the Leiden algorithm
        Random random = new Random();
        LeidenAlgorithm leidenAlgorithm = new LeidenAlgorithm(resolution, iterations, randomness, random);
        
        // Create initial clustering (each node in its own cluster)
        clustering = new Clustering(network.getNNodes());
        
        // Improve the initial clustering using the Leiden algorithm
        leidenAlgorithm.improveClustering(network, clustering);
        
        // Order clusters by size (largest first)
        clustering.orderClustersByNNodes();
        
        // Map the clustering results back to the transportation graph nodes
        communities = mapClusteringToNodes(clustering);
        
        return communities;
    }
    
    /**
     * Converts the transportation graph to a Network object suitable for the Leiden algorithm.
     * This version incorporates geographic proximity as a factor in edge weights.
     *
     * @return Network object for Leiden algorithm
     */
    private Network convertToNetwork() {
        Graph<Node, DefaultWeightedEdge> graph;
        
        if (useOriginalPointsOnly) {
            graph = transportationGraph.getOriginalPointsGraph();
        } else {
            graph = transportationGraph.getGraph();
        }
        
        // Create a subgraph with only the vertices that have edges
        graph = createSubgraph(new ArrayList<>(graph.vertexSet()), graph);
        
        // Convert to LeidenNetwork format
        int nNodes = graph.vertexSet().size();
        List<Node> nodes = new ArrayList<>(graph.vertexSet());
        
        // Create a mapping from nodes to indices
        Map<Node, Integer> nodeToIndex = new HashMap<>();
        for (int i = 0; i < nodes.size(); i++) {
            nodeToIndex.put(nodes.get(i), i);
        }
        
        // Prepare arrays for network construction
        LargeIntArray[] edges = new LargeIntArray[2];
        edges[0] = new LargeIntArray(graph.edgeSet().size());
        edges[1] = new LargeIntArray(graph.edgeSet().size());
        LargeDoubleArray edgeWeights = new LargeDoubleArray(graph.edgeSet().size());
        
        // Calculate max geographic distance for normalization
        double maxGeoDist = 0.0;
        for (Node n1 : nodes) {
            for (Node n2 : nodes) {
                if (n1 != n2) {
                    double dist = calculateGeoDistance(n1, n2);
                    if (dist > maxGeoDist) maxGeoDist = dist;
                }
            }
        }
        
        // Add edges with weights that incorporate both connectivity and geographic proximity
        int edgeCount = 0;
        for (DefaultWeightedEdge e : graph.edgeSet()) {
            Node source = graph.getEdgeSource(e);
            Node target = graph.getEdgeTarget(e);
            
            double connectionWeight = graph.getEdgeWeight(e);
            // Smaller value means stronger connection in the graph
            connectionWeight = 1.0 / connectionWeight;
            
            // Get geographic distance between nodes and normalize
            double geoDistance = calculateGeoDistance(source, target) / maxGeoDist;
            
            // Combine connection weight with geographic proximity
            // Higher alpha gives more importance to network connectivity vs spatial proximity
            // We use a balanced alpha value to consider both network structure and geography
            double alpha = 0.5; // Increased from 0.3 to balance network and spatial factors
            double combinedWeight = alpha * connectionWeight + (1-alpha) * (1.0 - geoDistance);
            
            int sourceIdx = nodeToIndex.get(source);
            int targetIdx = nodeToIndex.get(target);
            
            edges[0].set(edgeCount, sourceIdx);
            edges[1].set(edgeCount, targetIdx);
            edgeWeights.set(edgeCount, combinedWeight);
            edgeCount++;
        }
        
        // Create and return the network
        try {
        return new Network(nNodes, true, edges, edgeWeights, false, true);
        } catch (Exception e) {
            System.err.println("Error creating network: " + e.getMessage());
            e.printStackTrace();
            return null;
        }
    }
    
    /**
     * Calculate the geographic distance between two nodes
     */
    private double calculateGeoDistance(Node n1, Node n2) {
        if (n1.getLocation() == null || n2.getLocation() == null) {
            return 0.0;
        }
        
        // Get coordinates
        double x1 = n1.getLocation().getX();
        double y1 = n1.getLocation().getY();
        double x2 = n2.getLocation().getX();
        double y2 = n2.getLocation().getY();
        
        // Calculate Euclidean distance
        return Math.sqrt(Math.pow(x2 - x1, 2) + Math.pow(y2 - y1, 2));
    }
    
    /**
     * Maps the clustering results back to the transportation graph nodes.
     * 
     * @param clustering The clustering results from the Leiden algorithm
     * @return A map of community IDs to lists of nodes in each community
     */
    private Map<Integer, List<Node>> mapClusteringToNodes(Clustering clustering) {
        Map<Integer, List<Node>> communities = new HashMap<>();
        Graph<Node, DefaultWeightedEdge> graph = transportationGraph.getGraph();
        
        // Create mapping from nodes to integers (same as in convertToNetwork)
        Map<Node, Integer> nodeToIndex = new HashMap<>();
        int index = 0;
        for (Node node : graph.vertexSet()) {
            nodeToIndex.put(node, index++);
        }
        
        // Reverse the mapping
        Map<Integer, Node> indexToNode = new HashMap<>();
        for (Map.Entry<Node, Integer> entry : nodeToIndex.entrySet()) {
            indexToNode.put(entry.getValue(), entry.getKey());
        }
        
        // Map clustering results to nodes
        for (int i = 0; i < clustering.getNNodes(); i++) {
            int communityId = clustering.getCluster(i);
            Node node = indexToNode.get(i);
            
            if (!communities.containsKey(communityId)) {
                communities.put(communityId, new ArrayList<>());
            }
            communities.get(communityId).add(node);
        }
        
        return communities;
    }
    
    /**
     * Assesses the balance and quality of a community structure.
     * This version includes spatial cohesion as a factor.
     *
     * @param communities Map of community IDs to lists of nodes
     * @param graphSize Size of the graph
     * @return A score representing the quality of the community structure (lower is better)
     */
    private double assessCommunityBalance(Map<Integer, List<Node>> communities, int graphSize) {
        // Count total nodes
        int totalNodes = countTotalNodes(communities);
        if (totalNodes == 0) return Double.MAX_VALUE;
        
        // Basic stats
        int communityCount = communities.size();
        int singletonCount = countSingletonCommunities(communities);
        int largestSize = findLargestCommunitySize(communities);
        
        // Extreme cases - all singletons or one giant community
        if (communityCount == totalNodes) {
            // All singletons - worst case
            return 100.0;
        } else if (communityCount == 1) {
            // Single community containing all nodes - also poor
            return 50.0;
        }
        
        // Calculate desired number of communities - based on network size and scaling factor
        // Allow for a more realistic number of communities based on network complexity
        double idealCommunityCount = Math.max(minCommunities, 
                                       Math.sqrt(totalNodes) * communityScalingFactor);
        
        // Penalties for too few or too many communities 
        double communityCountPenalty = Math.abs(communityCount - idealCommunityCount) / idealCommunityCount * 8.0;
        
        // More balanced penalty for having too many or too few communities
        if (communityCount > idealCommunityCount * 1.5) {
            communityCountPenalty *= 1.5;
        }
        
        // Penalty for singleton communities - still discourage them but less extremely
        double singletonPercentage = (double) singletonCount / communityCount;
        double singletonPenalty = singletonPercentage * 25.0;
        
        // Dominance penalty - one community has too many nodes
        double dominancePenalty = 0.0;
        if (largestSize > 0.6 * totalNodes) {
            dominancePenalty = 8.0 * ((double) largestSize / totalNodes - 0.6);
        }
        
        // Calculate Gini coefficient for community size distribution
        double giniCoefficient = calculateGiniCoefficient(communities);
        
        // Calculate spatial cohesion of communities
        double spatialCohesionPenalty = calculateSpatialCohesionPenalty(communities);
        
        // Distribution score - prefer a reasonable number of medium-sized communities
        double distributionScore = 0.0;
        int mediumSizedCount = 0;
        int smallCount = 0;
        int largeCount = 0;
        
        double avgNodesPerCommunity = (double) totalNodes / communityCount;
        for (List<Node> community : communities.values()) {
            int size = community.size();
            if (size == 1) continue; // Already counted in singleton penalty
            
            if (size < 0.5 * avgNodesPerCommunity) {
                smallCount++;
            } else if (size > 2.0 * avgNodesPerCommunity) {
                largeCount++;
            } else {
                mediumSizedCount++;
            }
        }
        
        // Reward structures with a good number of medium-sized communities
        double mediumSizedRatio = (double) mediumSizedCount / Math.max(1, communityCount - singletonCount);
        distributionScore = 10.0 * (1.0 - mediumSizedRatio);
        
        // Final score calculation - weighted sum of different metrics
        double score = 
            communityCountPenalty * 1.5 +
            singletonPenalty * 2.5 +
            dominancePenalty * 1.5 +
            giniCoefficient * 3.0 +
            distributionScore * 1.0 +
            spatialCohesionPenalty * 6.0;
        
        return score;
    }
    
    /**
     * Calculates a penalty based on spatial cohesion of communities.
     * Lower values indicate communities that are spatially coherent.
     */
    private double calculateSpatialCohesionPenalty(Map<Integer, List<Node>> communities) {
        double totalPenalty = 0.0;
        int validCommunities = 0;
        
        for (List<Node> community : communities.values()) {
            if (community.size() <= 1) continue; // Skip singletons
            
            // Calculate community centroid
            double totalX = 0.0, totalY = 0.0;
            int nodesWithLocation = 0;
            
            for (Node node : community) {
                if (node.getLocation() != null) {
                    totalX += node.getLocation().getX();
                    totalY += node.getLocation().getY();
                    nodesWithLocation++;
                }
            }
            
            if (nodesWithLocation < 2) continue; // Skip if not enough nodes with locations
            
            double centroidX = totalX / nodesWithLocation;
            double centroidY = totalY / nodesWithLocation;
            
            // Calculate average distance from centroid
            double totalDistance = 0.0;
            for (Node node : community) {
                if (node.getLocation() != null) {
                    double dx = node.getLocation().getX() - centroidX;
                    double dy = node.getLocation().getY() - centroidY;
                    totalDistance += Math.sqrt(dx*dx + dy*dy);
                }
            }
            
            // Normalize by community size and add to total penalty
            double avgDistance = totalDistance / nodesWithLocation;
            totalPenalty += avgDistance;
            validCommunities++;
        }
        
        return validCommunities > 0 ? totalPenalty / validCommunities : 0.0;
    }
    
    /**
     * Calculates the Gini coefficient for community sizes.
     * A value of 0 means all communities are the same size.
     * A value of 1 means extreme inequality (one community has all nodes).
     * 
     * @param communities The communities to analyze
     * @return The Gini coefficient (0-1)
     */
    private double calculateGiniCoefficient(Map<Integer, List<Node>> communities) {
        if (communities.size() <= 1) {
            return 0.0; // Only one community, no inequality
        }
        
        // Get sizes
        List<Integer> sizes = new ArrayList<>();
        for (List<Node> community : communities.values()) {
            sizes.add(community.size());
        }
        
        // Sort sizes
        Collections.sort(sizes);
        
        // Calculate Gini coefficient
        double sumOfDifferences = 0.0;
        double sumOfSizes = 0.0;
        
        for (int size : sizes) {
            sumOfSizes += size;
        }
        
        if (sumOfSizes == 0.0) {
            return 0.0; // No nodes in communities
        }
        
        for (int i = 0; i < sizes.size(); i++) {
            for (int j = 0; j < sizes.size(); j++) {
                sumOfDifferences += Math.abs(sizes.get(i) - sizes.get(j));
            }
        }
        
        return sumOfDifferences / (2 * sizes.size() * sizes.size() * (sumOfSizes / sizes.size()));
    }
    
    /**
     * Checks if the community distribution is extremely imbalanced (much worse than natural variations)
     * 
     * @return True if distribution is extremely imbalanced
     */
    private boolean isExtremelyImbalanced() {
        // Similar to isImbalanced() but with more extreme thresholds
        int totalNodes = useOriginalPointsOnly ? 
            transportationGraph.getOriginalPointsGraph().vertexSet().size() : 
            transportationGraph.getGraph().vertexSet().size();
            
        // Count singleton communities (communities with just 1 node)
        int singletonCount = 0;
        
        // Get largest community
        int largestSize = 0;
        for (List<Node> community : communities.values()) {
            if (community.size() > largestSize) {
                largestSize = community.size();
            }
            
            if (community.size() == 1) {
                singletonCount++;
            }
        }
        
        // Calculate percentages
        double largestPercentage = (double)largestSize / totalNodes;
        double singletonPercentage = (double)singletonCount / communities.size();
        
        // Consider extremely imbalanced if:
        // 1. One community has more than 85% of nodes, OR
        // 2. More than 80% of communities are singletons
        boolean isExtremelyDominated = largestPercentage > 0.85;
        boolean isExtremelyFragmented = singletonPercentage > 0.8 && communities.size() > 10;
        
        return isExtremelyDominated || isExtremelyFragmented;
    }
    
    /**
     * Mildly rebalances communities without forcing perfect equality.
     * This preserves the natural structure while addressing extreme imbalances.
     */
    private void mildlyRebalanceCommunities() {
        System.out.println("Applying community rebalancing to improve structure...");
        
        // Get graph to work with
        Graph<Node, DefaultWeightedEdge> graph = useOriginalPointsOnly ? 
                transportationGraph.getOriginalPointsGraph() : 
                transportationGraph.getGraph();
        
        // Find largest and smallest communities
        int largestCommunityId = -1;
        int largestSize = 0;
        List<Integer> singletonCommIds = new ArrayList<>();
        
        for (Map.Entry<Integer, List<Node>> entry : communities.entrySet()) {
            int communityId = entry.getKey();
            int size = entry.getValue().size();
            
            if (size > largestSize) {
                largestSize = size;
                largestCommunityId = communityId;
            }
            
            if (size == 1) {
                singletonCommIds.add(communityId);
            }
        }
        
        int totalNodes = useOriginalPointsOnly ? 
            transportationGraph.getOriginalPointsGraph().vertexSet().size() : 
            transportationGraph.getGraph().vertexSet().size();
        
        double largestPercentage = (double)largestSize / totalNodes;
        double singletonPercentage = (double)singletonCommIds.size() / communities.size();
        
        // Strategy depends on the type of imbalance
        boolean hasDominantCommunity = largestPercentage > 0.7 && largestCommunityId >= 0;
        boolean hasTooManySingletons = singletonPercentage > 0.7 && singletonCommIds.size() > 10;
        
        System.out.println("Rebalancing strategy:");
        if (hasDominantCommunity) {
            System.out.println("- Splitting dominant community (contains " + 
                              String.format("%.1f", largestPercentage * 100) + "% of nodes)");
        }
        if (hasTooManySingletons) {
            System.out.println("- Merging singleton communities (" + singletonCommIds.size() + 
                              " singletons, " + String.format("%.1f", singletonPercentage * 100) + "% of communities)");
        }
        
        // If most communities are singletons and we don't have a large dominant community,
        // start from scratch with a different approach
        if (singletonPercentage > 0.9 && !hasDominantCommunity) {
            System.out.println("Most communities are singletons. Using distance-based clustering instead.");
            createDistanceBasedCommunities(graph);
            return;
        }
        
        // If we have a dominant community, split it
        if (hasDominantCommunity) {
            splitDominantCommunity(largestCommunityId, graph);
        }
        
        // If we have too many tiny communities, merge some of them
        if (hasTooManySingletons) {
            mergeSingletonCommunities(singletonCommIds, graph);
        }
        
        // Update clustering object after modifications
        updateClusteringFromCommunities();
        
        System.out.println("Rebalancing complete. New community structure:");
        printCommunityStats();
    }
    
    /**
     * Prints statistics about the current community structure
     */
    private void printCommunityStats() {
        // Count by size categories
        int singletons = 0;
        int small = 0; // 2-5
        int medium = 0; // 6-20
        int large = 0; // >20
        
        for (List<Node> community : communities.values()) {
            int size = community.size();
            if (size == 1) singletons++;
            else if (size <= 5) small++;
            else if (size <= 20) medium++;
            else large++;
        }
        
        System.out.println("  Total communities: " + communities.size());
        System.out.println("  Singletons: " + singletons + " (" + 
                          String.format("%.1f", (double)singletons/communities.size()*100) + "%)");
        System.out.println("  Small (2-5): " + small + " (" + 
                          String.format("%.1f", (double)small/communities.size()*100) + "%)");
        System.out.println("  Medium (6-20): " + medium + " (" + 
                          String.format("%.1f", (double)medium/communities.size()*100) + "%)");
        System.out.println("  Large (>20): " + large + " (" + 
                          String.format("%.1f", (double)large/communities.size()*100) + "%)");
        
        // Print largest community size
        int largestSize = 0;
        for (List<Node> community : communities.values()) {
            if (community.size() > largestSize) {
                largestSize = community.size();
            }
        }
        
        int totalNodes = countTotalNodes(communities);
        System.out.println("  Largest community size: " + largestSize + " (" + 
                          String.format("%.1f", (double)largestSize/totalNodes*100) + "% of nodes)");
    }
    
    /**
     * Splits a dominant community into smaller communities
     * 
     * @param communityId The ID of the community to split
     * @param graph The graph to work with
     */
    private void splitDominantCommunity(int communityId, Graph<Node, DefaultWeightedEdge> graph) {
        List<Node> dominantCommunity = communities.get(communityId);
        System.out.println("Splitting dominant community " + communityId + 
                          " with " + dominantCommunity.size() + " nodes");
        
        // Create a subgraph of just this community
        Graph<Node, DefaultWeightedEdge> subgraph = createSubgraph(dominantCommunity, graph);
        
        // Split this community
        Map<Node, Integer> subcommunities = splitCommunity(subgraph, dominantCommunity);
        
        // Apply the split
        if (subcommunities != null && !subcommunities.isEmpty()) {
            // Remove nodes from original community
            communities.get(communityId).clear();
            
            // Create new communities and reassign nodes
            Map<Integer, Integer> subToGlobalId = new HashMap<>();
            
            for (Map.Entry<Node, Integer> entry : subcommunities.entrySet()) {
                Node node = entry.getKey();
                int subCommunityId = entry.getValue();
                
                // Map sub-community ID to global community ID
                if (!subToGlobalId.containsKey(subCommunityId)) {
                    if (subCommunityId == 0) {
                        // Keep first sub-community in the original community
                        subToGlobalId.put(subCommunityId, communityId);
                    } else {
                        // Create new community ID for others
                        int newGlobalId = findNextAvailableCommunityId();
                        subToGlobalId.put(subCommunityId, newGlobalId);
                        communities.put(newGlobalId, new ArrayList<>());
                    }
                }
                
                // Add node to appropriate community
                int globalCommunityId = subToGlobalId.get(subCommunityId);
                communities.get(globalCommunityId).add(node);
            }
            
            // Clean up any empty communities
            communities.entrySet().removeIf(entry -> entry.getValue().isEmpty());
            
            System.out.println("Split dominant community into " + subToGlobalId.size() + " sub-communities");
        }
    }
    
    /**
     * Merges singleton communities into larger groups based on connectivity
     * 
     * @param singletonIds List of community IDs for singletons
     * @param graph The graph to work with
     */
    private void mergeSingletonCommunities(List<Integer> singletonIds, Graph<Node, DefaultWeightedEdge> graph) {
        System.out.println("Merging " + singletonIds.size() + " singleton communities");
        
        // If we have too many singletons and no larger communities, create clusters from scratch
        if (singletonIds.size() > 20 && communities.size() - singletonIds.size() < 3) {
            // Collect all singleton nodes
            List<Node> allSingletonNodes = new ArrayList<>();
            for (int id : singletonIds) {
                allSingletonNodes.addAll(communities.get(id));
            }
            
            // Create a subgraph of just these nodes
            Graph<Node, DefaultWeightedEdge> subgraph = createSubgraph(allSingletonNodes, graph);
            
            // Group them based on connectivity
            Map<Node, Integer> newAssignments = clusterSingletons(subgraph, allSingletonNodes);
            
            // Apply the new assignments
            if (newAssignments != null && !newAssignments.isEmpty()) {
                // Remove all singleton communities
                for (int id : singletonIds) {
                    communities.remove(id);
                }
                
                // Create new communities from clusters
                Map<Integer, Integer> clusterToGlobalId = new HashMap<>();
                
                for (Map.Entry<Node, Integer> entry : newAssignments.entrySet()) {
                    Node node = entry.getKey();
                    int clusterId = entry.getValue();
                    
                    // Map cluster ID to global community ID
                    if (!clusterToGlobalId.containsKey(clusterId)) {
                        int newGlobalId = findNextAvailableCommunityId();
                        clusterToGlobalId.put(clusterId, newGlobalId);
                        communities.put(newGlobalId, new ArrayList<>());
                    }
                    
                    // Add node to appropriate community
                    int globalCommunityId = clusterToGlobalId.get(clusterId);
                    communities.get(globalCommunityId).add(node);
                }
                
                System.out.println("Grouped " + singletonIds.size() + " singletons into " + 
                                  clusterToGlobalId.size() + " new communities");
            return;
            }
        }
        
        // Otherwise merge pairs of well-connected singletons
        Map<Integer, List<Integer>> merges = findSingletonMerges(singletonIds, graph);
        
        // Apply the merges
        for (Map.Entry<Integer, List<Integer>> merge : merges.entrySet()) {
            int targetId = merge.getKey();
            List<Integer> sourcesToMerge = merge.getValue();
            
            for (int sourceId : sourcesToMerge) {
                if (communities.containsKey(sourceId)) {
                    communities.get(targetId).addAll(communities.get(sourceId));
                    communities.remove(sourceId);
                }
            }
        }
        
        System.out.println("Merged " + merges.values().stream().mapToInt(List::size).sum() + 
                          " singletons into " + merges.size() + " communities");
    }
    
    /**
     * Creates a distance-based community structure when Leiden algorithm produces too many singletons
     * 
     * @param graph The graph to work with
     */
    private void createDistanceBasedCommunities(Graph<Node, DefaultWeightedEdge> graph) {
        System.out.println("Creating distance-based communities from scratch");
        
        // Get all nodes
        List<Node> allNodes = new ArrayList<>(graph.vertexSet());
        int totalNodes = allNodes.size();
        
        // Determine appropriate number of communities - square root of node count is a good heuristic
        int targetCommunityCount = (int)Math.ceil(Math.sqrt(totalNodes));
        System.out.println("Target number of communities: " + targetCommunityCount);
        
        // Sort nodes by degree (descending) to select most connected nodes as seeds
        allNodes.sort((n1, n2) -> Integer.compare(graph.degreeOf(n2), graph.degreeOf(n1)));
        
        // Select seed nodes evenly spaced through the sorted list
        List<Node> seedNodes = new ArrayList<>();
        Map<Node, Integer> nodeToCommunity = new HashMap<>();
        
        // Choose evenly spaced nodes from the sorted list
        int step = Math.max(1, allNodes.size() / targetCommunityCount);
        for (int i = 0; i < allNodes.size() && seedNodes.size() < targetCommunityCount; i += step) {
            Node seed = allNodes.get(i);
            seedNodes.add(seed);
            nodeToCommunity.put(seed, seedNodes.size() - 1);
        }
        
        System.out.println("Selected " + seedNodes.size() + " seed nodes for communities");
        
        // Assign remaining nodes to closest seed
        for (Node node : allNodes) {
            if (nodeToCommunity.containsKey(node)) continue; // Skip seeds
            
            int bestCommunity = -1;
            double bestScore = Double.NEGATIVE_INFINITY;
            
            for (int i = 0; i < seedNodes.size(); i++) {
                Node seed = seedNodes.get(i);
                double score = calculateConnectionScore(node, seed, graph, nodeToCommunity, i);
                
                if (score > bestScore) {
                    bestScore = score;
                    bestCommunity = i;
                }
            }
            
            if (bestCommunity >= 0) {
                nodeToCommunity.put(node, bestCommunity);
            }
        }
        
        // Convert to our community structure
        communities.clear();
        for (Map.Entry<Node, Integer> entry : nodeToCommunity.entrySet()) {
            Node node = entry.getKey();
            int communityId = entry.getValue();
            
            if (!communities.containsKey(communityId)) {
                communities.put(communityId, new ArrayList<>());
            }
            communities.get(communityId).add(node);
        }
        
        System.out.println("Created " + communities.size() + " distance-based communities");
    }
    
    /**
     * Calculates a connection score between a node and a potential community
     * 
     * @param node The node to assign
     * @param seed The seed node of the community
     * @param graph The graph
     * @param nodeToCommunity Map of nodes to their communities
     * @param communityId The community ID to check against
     * @return A score representing connection strength (higher is better)
     */
    private double calculateConnectionScore(Node node, Node seed, Graph<Node, DefaultWeightedEdge> graph,
                                         Map<Node, Integer> nodeToCommunity, int communityId) {
        // Base score is inverse of distance to seed
        double seedDistance = 1.0;
        DefaultWeightedEdge edge = graph.getEdge(node, seed);
        if (edge != null) {
            seedDistance = graph.getEdgeWeight(edge);
        }
        double seedScore = 100.0 / (1.0 + seedDistance);
        
        // Add score for connections to other nodes in the community
        double connectionScore = 0.0;
        int connectionCount = 0;
        
        for (Map.Entry<Node, Integer> entry : nodeToCommunity.entrySet()) {
            if (entry.getValue() == communityId && !entry.getKey().equals(seed)) {
                Node otherNode = entry.getKey();
                DefaultWeightedEdge otherEdge = graph.getEdge(node, otherNode);
                if (otherEdge != null) {
                    connectionScore += 50.0 / (1.0 + graph.getEdgeWeight(otherEdge));
                    connectionCount++;
                }
            }
        }
        
        // Normalize by potential connections to avoid favoring large communities
        double normalizedConnectionScore = connectionCount > 0 ? 
                connectionScore / Math.sqrt(nodeToCommunity.entrySet().stream()
                    .filter(e -> e.getValue() == communityId).count()) : 0;
        
        // Size balancing - add penalty for joining already large communities
        int communitySize = (int)nodeToCommunity.entrySet().stream()
                .filter(e -> e.getValue() == communityId).count();
        
        int averageSize = nodeToCommunity.size() / Math.max(1, 
                (int)nodeToCommunity.values().stream().distinct().count());
        
        double sizeBalancePenalty = communitySize > averageSize * 1.5 ? 
                (communitySize - averageSize * 1.5) * 5 : 0;
        
        return seedScore + normalizedConnectionScore - sizeBalancePenalty;
    }
    
    /**
     * Creates a subgraph containing only the specified nodes
     * 
     * @param nodes The nodes to include
     * @param originalGraph The original graph
     * @return A subgraph with only the specified nodes and edges between them
     */
    private Graph<Node, DefaultWeightedEdge> createSubgraph(List<Node> nodes, Graph<Node, DefaultWeightedEdge> originalGraph) {
        Graph<Node, DefaultWeightedEdge> subgraph = new SimpleWeightedGraph<>(DefaultWeightedEdge.class);
        
        // Add vertices
        for (Node node : nodes) {
            subgraph.addVertex(node);
        }
        
        // Add edges between these nodes
        for (Node source : nodes) {
            for (Node target : nodes) {
                if (source == target) continue;
                
                DefaultWeightedEdge edge = originalGraph.getEdge(source, target);
                if (edge != null) {
                    DefaultWeightedEdge newEdge = subgraph.addEdge(source, target);
                    if (newEdge != null) {
                        subgraph.setEdgeWeight(newEdge, originalGraph.getEdgeWeight(edge));
                    }
                }
            }
        }
        
        return subgraph;
    }
    
    /**
     * Finds the next available community ID
     * 
     * @return A community ID that is not currently in use
     */
    private int findNextAvailableCommunityId() {
        return communities.keySet().stream().max(Integer::compare).orElse(-1) + 1;
    }
    
    /**
     * Finds which singleton communities should be merged
     * 
     * @param singletonIds List of singleton community IDs
     * @param graph The graph
     * @return Map of target community IDs to lists of communities to merge into them
     */
    private Map<Integer, List<Integer>> findSingletonMerges(List<Integer> singletonIds, Graph<Node, DefaultWeightedEdge> graph) {
        Map<Integer, List<Integer>> merges = new HashMap<>();
        Set<Integer> assignedIds = new HashSet<>();
        
        // Create pairs of singletons based on connectivity
        List<int[]> pairs = new ArrayList<>();
        
        for (int i = 0; i < singletonIds.size(); i++) {
            for (int j = i + 1; j < singletonIds.size(); j++) {
                int id1 = singletonIds.get(i);
                int id2 = singletonIds.get(j);
                
                if (assignedIds.contains(id1) || assignedIds.contains(id2)) {
                    continue;
                }
                
                Node node1 = communities.get(id1).get(0); // There's only one node in a singleton
                Node node2 = communities.get(id2).get(0);
                
                DefaultWeightedEdge edge = graph.getEdge(node1, node2);
                double strength = edge != null ? 1.0 / (1.0 + graph.getEdgeWeight(edge)) : 0.0;
                
                pairs.add(new int[] {id1, id2, (int)(strength * 1000)}); // Convert to sortable int
            }
        }
        
        // Sort by strength (descending)
        pairs.sort((p1, p2) -> Integer.compare(p2[2], p1[2]));
        
        // Process pairs in order of strength
        for (int[] pair : pairs) {
            int id1 = pair[0];
            int id2 = pair[1];
            
            if (assignedIds.contains(id1) || assignedIds.contains(id2)) {
                continue;
            }
            
            // Choose first one as target, second as source
            int targetId = id1;
            int sourceId = id2;
            
            // Add to merges
            if (!merges.containsKey(targetId)) {
                merges.put(targetId, new ArrayList<>());
            }
            merges.get(targetId).add(sourceId);
            
            // Mark as assigned
            assignedIds.add(targetId);
            assignedIds.add(sourceId);
        }
        
        // Handle leftovers - find best connections to existing targets
        for (int id : singletonIds) {
            if (assignedIds.contains(id)) {
                continue;
            }
            
            Node node = communities.get(id).get(0);
            int bestTargetId = -1;
            double bestStrength = 0.0;
            
            for (int targetId : merges.keySet()) {
                for (Node targetNode : communities.get(targetId)) {
                    DefaultWeightedEdge edge = graph.getEdge(node, targetNode);
            if (edge != null) {
                        double strength = 1.0 / (1.0 + graph.getEdgeWeight(edge));
                        if (strength > bestStrength) {
                            bestStrength = strength;
                            bestTargetId = targetId;
                        }
                    }
                }
                
                for (int sourceId : merges.get(targetId)) {
                    for (Node sourceNode : communities.get(sourceId)) {
                        DefaultWeightedEdge edge = graph.getEdge(node, sourceNode);
                        if (edge != null) {
                            double strength = 1.0 / (1.0 + graph.getEdgeWeight(edge));
                            if (strength > bestStrength) {
                                bestStrength = strength;
                                bestTargetId = targetId;
                            }
                        }
                    }
                }
            }
            
            // If we found a good target, add to merges
            if (bestTargetId >= 0 && bestStrength > 0.0) {
                merges.get(bestTargetId).add(id);
                assignedIds.add(id);
            }
        }
        
        // If there are still leftovers, create new pairs from them
        List<Integer> leftovers = new ArrayList<>();
        for (int id : singletonIds) {
            if (!assignedIds.contains(id)) {
                leftovers.add(id);
            }
        }
        
        // Pair up leftovers (even if not well connected)
        for (int i = 0; i < leftovers.size(); i += 2) {
            if (i + 1 < leftovers.size()) {
                int targetId = leftovers.get(i);
                int sourceId = leftovers.get(i + 1);
                
                merges.put(targetId, new ArrayList<>(List.of(sourceId)));
            } else if (i < leftovers.size() && !merges.isEmpty()) {
                // If there's an odd one left, add it to the first merge group
                int leftoverId = leftovers.get(i);
                int firstTargetId = merges.keySet().iterator().next();
                merges.get(firstTargetId).add(leftoverId);
            }
        }
        
        return merges;
    }
    
    /**
     * Clusters singleton nodes into communities
     * 
     * @param subgraph Subgraph containing only singleton nodes
     * @param nodes The singleton nodes
     * @return Mapping from nodes to new community IDs
     */
    private Map<Node, Integer> clusterSingletons(Graph<Node, DefaultWeightedEdge> subgraph, List<Node> nodes) {
        // Try a distance-based clustering approach
        int targetClusters = (int)Math.ceil(Math.sqrt(nodes.size()));
        Map<Node, Integer> assignments = new HashMap<>();
        
        // Sort nodes by degree (highest first)
        nodes.sort((n1, n2) -> Integer.compare(subgraph.degreeOf(n2), subgraph.degreeOf(n1)));
        
        // Select seed nodes (most connected, with minimum distance between them)
        List<Node> seeds = new ArrayList<>();
        for (Node node : nodes) {
            if (seeds.size() >= targetClusters) break;
            
            // Check if this node is far enough from existing seeds
            boolean isFarEnough = true;
            for (Node seed : seeds) {
                DefaultWeightedEdge edge = subgraph.getEdge(node, seed);
                if (edge != null && subgraph.getEdgeWeight(edge) < 100) { // Use a distance threshold
                    isFarEnough = false;
                    break;
                }
            }
            
            if (isFarEnough || seeds.isEmpty()) {
                seeds.add(node);
                assignments.put(node, seeds.size() - 1);
            }
        }
        
        // If we couldn't find enough seeds, just use top nodes by degree
        if (seeds.size() < Math.min(3, targetClusters)) {
            seeds.clear();
            assignments.clear();
            
            int seedCount = Math.min(nodes.size(), Math.min(3, targetClusters));
            for (int i = 0; i < seedCount; i++) {
                seeds.add(nodes.get(i));
                assignments.put(nodes.get(i), i);
            }
        }
        
        // Assign remaining nodes to closest seed
        for (Node node : nodes) {
            if (assignments.containsKey(node)) continue;
            
            Node closestSeed = null;
            double closestDistance = Double.MAX_VALUE;
            
            for (Node seed : seeds) {
                DefaultWeightedEdge edge = subgraph.getEdge(node, seed);
                double distance = edge != null ? subgraph.getEdgeWeight(edge) : Double.MAX_VALUE;
                
                if (distance < closestDistance) {
                    closestDistance = distance;
                    closestSeed = seed;
                }
            }
            
            if (closestSeed != null) {
                assignments.put(node, assignments.get(closestSeed));
            } else {
                // If no connection, assign to first community
                assignments.put(node, 0);
            }
        }
        
        return assignments;
    }
    
    /**
     * Updates the clustering object based on the current communities map.
     */
    private void updateClusteringFromCommunities() {
        // Create a mapping from node to index
        Map<Node, Integer> nodeToIndex = new HashMap<>();
        int index = 0;
        for (Node node : (useOriginalPointsOnly ? 
                transportationGraph.getOriginalPointsGraph().vertexSet() : 
                transportationGraph.getGraph().vertexSet())) {
            nodeToIndex.put(node, index++);
        }
        
        // Create clustering assignments
        int[] clusterAssignments = new int[nodeToIndex.size()];
        
        // Initialize with -1 (unassigned)
        Arrays.fill(clusterAssignments, -1);
        
        // Assign clusters based on communities
        for (Map.Entry<Integer, List<Node>> entry : communities.entrySet()) {
            int communityId = entry.getKey();
            for (Node node : entry.getValue()) {
                Integer nodeIndex = nodeToIndex.get(node);
                if (nodeIndex != null) {
                    clusterAssignments[nodeIndex] = communityId;
                }
            }
        }
        
        // Update clustering object
        this.clustering = new Clustering(clusterAssignments);
    }
    
    /**
     * Enhances the spatial coherence of communities by redistributing nodes based on geographic proximity.
     * This post-processing step improves the visual clarity of communities on the map.
     */
    private void enhanceSpatialCoherence() {
        System.out.println("Enhancing spatial coherence of communities...");
        
        // Get graph to work with
        Graph<Node, DefaultWeightedEdge> graph = useOriginalPointsOnly ? 
                transportationGraph.getOriginalPointsGraph() : 
                transportationGraph.getGraph();
                
        // First, calculate centroids for each community
        Map<Integer, double[]> communityCentroids = calculateCommunityCentroids();
        
        // Remove singleton communities by merging them into geographically closest community
        List<Integer> singletonCommunityIds = new ArrayList<>();
        for (Map.Entry<Integer, List<Node>> entry : communities.entrySet()) {
            if (entry.getValue().size() == 1) {
                singletonCommunityIds.add(entry.getKey());
            }
        }
        
        if (!singletonCommunityIds.isEmpty()) {
            System.out.println("Merging " + singletonCommunityIds.size() + " singleton communities into geographically closest communities");
            for (int communityId : singletonCommunityIds) {
                List<Node> singleton = communities.get(communityId);
                if (singleton == null || singleton.isEmpty()) continue;
                
                Node node = singleton.get(0);
                if (node.getLocation() == null) continue;
                
                // Find closest non-singleton community
                int closestCommunityId = -1;
                double closestDistance = Double.MAX_VALUE;
                
                for (Map.Entry<Integer, List<Node>> entry : communities.entrySet()) {
                    int otherCommunityId = entry.getKey();
                    List<Node> otherCommunity = entry.getValue();
                    
                    if (otherCommunityId == communityId || otherCommunity.size() <= 1) {
                        continue; // Skip self and other singletons
                    }
                    
                    // Calculate distance to this community's centroid
                    double[] centroid = communityCentroids.get(otherCommunityId);
                    if (centroid == null) continue;
                    
                    double dx = centroid[0] - node.getLocation().getX();
                    double dy = centroid[1] - node.getLocation().getY();
                    double distance = Math.sqrt(dx*dx + dy*dy);
                    
                    if (distance < closestDistance) {
                        closestDistance = distance;
                        closestCommunityId = otherCommunityId;
                    }
                }
                
                // Merge singleton into closest community
                if (closestCommunityId != -1) {
                    communities.get(closestCommunityId).add(node);
                    communities.remove(communityId);
                }
            }
            
            // Recalculate centroids after merging
            communityCentroids = calculateCommunityCentroids();
        }
        
        // Apply moderate spatial coherence - fewer passes to preserve more natural structures
        int maxPasses = 2; // Reduced from 3 to be less aggressive
        int totalReassignments = 0;
        
        for (int pass = 0; pass < maxPasses; pass++) {
            // Identify border nodes (nodes that are geographically closer to a different community centroid)
            List<Node> borderNodes = new ArrayList<>();
            Map<Node, Integer> reassignments = new HashMap<>();
            
            // Check each node to see if it's closer to a different community's centroid
            for (Map.Entry<Integer, List<Node>> entry : communities.entrySet()) {
                int currentCommunityId = entry.getKey();
                double[] currentCentroid = communityCentroids.get(currentCommunityId);
                
                // Skip communities with no valid centroid (e.g., missing location data)
                if (currentCentroid == null) continue;
                
                for (Node node : entry.getValue()) {
                    if (node.getLocation() == null) continue;
                    
                    // Find the closest community centroid
                    int closestCommunityId = findClosestCentroid(node, communityCentroids);
                    
                    // If closest is not current, it's a border node
                    if (closestCommunityId != -1 && closestCommunityId != currentCommunityId) {
                        // Only consider reassignment if it maintains network connectivity
                        boolean hasConnectionToTargetCommunity = hasConnectionToCommunity(node, closestCommunityId, graph);
                        
                        // Ensure we don't create singletons by removing the only node from a community
                        boolean wouldCreateSingleton = entry.getValue().size() == 1;
                        
                        if (hasConnectionToTargetCommunity && !wouldCreateSingleton) {
                            borderNodes.add(node);
                            reassignments.put(node, closestCommunityId);
                        }
                    }
                }
            }
            
            // Apply reassignments
            if (!borderNodes.isEmpty()) {
                int passReassignments = borderNodes.size();
                totalReassignments += passReassignments;
                System.out.println("Pass " + (pass+1) + ": Reassigning " + passReassignments + " border nodes to improve spatial coherence");
                
                for (Node node : borderNodes) {
                    int currentCommunityId = -1;
                    
                    // Find current community
                    for (Map.Entry<Integer, List<Node>> entry : communities.entrySet()) {
                        if (entry.getValue().contains(node)) {
                            currentCommunityId = entry.getKey();
                            break;
                        }
                    }
                    
                    // Move node to new community
                    if (currentCommunityId != -1) {
                        int newCommunityId = reassignments.get(node);
                        communities.get(currentCommunityId).remove(node);
                        communities.get(newCommunityId).add(node);
                    }
                }
                
                // Recalculate centroids for next pass
                communityCentroids = calculateCommunityCentroids();
                
                // Clean up any empty communities that might have been created
                communities.entrySet().removeIf(entry -> entry.getValue().isEmpty());
            } else {
                // No more border nodes to reassign, we can stop
                break;
            }
        }
        
        // Update clustering object after modifications
        updateClusteringFromCommunities();
        
        if (totalReassignments > 0) {
            // Print updated statistics
            System.out.println("Total " + totalReassignments + " nodes reassigned for spatial coherence enhancement");
            System.out.println("Communities after spatial coherence enhancement:");
            printCommunityStats();
        } else {
            System.out.println("No border nodes found for reassignment - communities already spatially coherent");
        }
        
        // Final cleanup - ensure we don't have too many communities for visualization clarity
        // Use the configurable maximum value
        if (communities.size() > maxCommunities) {
            mergeSmallerCommunities(maxCommunities);
        }
    }
    
    /**
     * Calculates the geographic centroid (center point) for each community
     *
     * @return Map of community IDs to centroids [x, y]
     */
    private Map<Integer, double[]> calculateCommunityCentroids() {
        Map<Integer, double[]> centroids = new HashMap<>();
        
        for (Map.Entry<Integer, List<Node>> entry : communities.entrySet()) {
            int communityId = entry.getKey();
            List<Node> nodes = entry.getValue();
            
            if (nodes.isEmpty()) continue;
            
            double sumX = 0.0, sumY = 0.0;
            int nodesWithLocation = 0;
            
            for (Node node : nodes) {
                if (node.getLocation() != null) {
                    sumX += node.getLocation().getX();
                    sumY += node.getLocation().getY();
                    nodesWithLocation++;
                }
            }
            
            if (nodesWithLocation > 0) {
                double[] centroid = new double[2];
                centroid[0] = sumX / nodesWithLocation; // X coordinate
                centroid[1] = sumY / nodesWithLocation; // Y coordinate
                centroids.put(communityId, centroid);
            }
        }
        
        return centroids;
    }
    
    /**
     * Finds the community with centroid closest to the given node
     *
     * @param node Node to check
     * @param centroids Map of community centroids
     * @return ID of closest community, or -1 if none found
     */
    private int findClosestCentroid(Node node, Map<Integer, double[]> centroids) {
        if (node.getLocation() == null) return -1;
        
        double nodeX = node.getLocation().getX();
        double nodeY = node.getLocation().getY();
        
        int closestCommunityId = -1;
        double closestDistance = Double.MAX_VALUE;
        
        for (Map.Entry<Integer, double[]> entry : centroids.entrySet()) {
            int communityId = entry.getKey();
            double[] centroid = entry.getValue();
            
            double dx = centroid[0] - nodeX;
            double dy = centroid[1] - nodeY;
            double distance = Math.sqrt(dx*dx + dy*dy);
            
            if (distance < closestDistance) {
                closestDistance = distance;
                closestCommunityId = communityId;
            }
        }
        
        return closestCommunityId;
    }
    
    /**
     * Checks if a node has connections to at least one node in the target community
     *
     * @param node Node to check
     * @param communityId Target community ID
     * @param graph The graph to work with
     * @return True if connected, false otherwise
     */
    private boolean hasConnectionToCommunity(Node node, int communityId, Graph<Node, DefaultWeightedEdge> graph) {
        List<Node> communityNodes = communities.get(communityId);
        if (communityNodes == null) return false;
        
        for (Node communityNode : communityNodes) {
            if (graph.containsEdge(node, communityNode)) {
                return true;
            }
        }
        
        return false;
    }
    
    /**
     * Gets the detected communities.
     * 
     * @return A map of community IDs to lists of nodes in each community
     */
    public Map<Integer, List<Node>> getCommunities() {
        return communities;
    }
    
    /**
     * Gets the raw clustering result from the Leiden algorithm.
     * 
     * @return The clustering object
     */
    public Clustering getClustering() {
        return clustering;
    }
    
    /**
     * Gets the count of communities detected.
     * 
     * @return The number of communities
     */
    public int getCommunityCount() {
        return communities.size();
    }
    
    /**
     * Gets a specific community by its ID.
     * 
     * @param communityId The ID of the community to retrieve
     * @return The list of nodes in the community, or null if not found
     */
    public List<Node> getCommunity(int communityId) {
        return communities.get(communityId);
    }
    
    /**
     * Gets detailed statistics about the community distribution
     */
    public String getCommunityStatistics() {
        StringBuilder stats = new StringBuilder();
        stats.append("Community Statistics:\n");
        stats.append("Total communities: ").append(communities.size()).append("\n");
        
        // Sort communities by size for better readability
        List<Map.Entry<Integer, List<Node>>> sortedCommunities = new ArrayList<>(communities.entrySet());
        sortedCommunities.sort((e1, e2) -> Integer.compare(e2.getValue().size(), e1.getValue().size())); // Largest first
        
        // Print individual community sizes
        for (Map.Entry<Integer, List<Node>> entry : sortedCommunities) {
            int totalNodes = useOriginalPointsOnly ? 
                transportationGraph.getOriginalPointsGraph().vertexSet().size() : 
                transportationGraph.getGraph().vertexSet().size();
            
            stats.append("Community ").append(entry.getKey())
                 .append(": ").append(entry.getValue().size())
                 .append(" nodes (").append(String.format("%.1f", (entry.getValue().size() * 100.0 / totalNodes)))
                 .append("% of total)\n");
        }
        
        return stats.toString();
    }
    
    /**
     * Splits a large community into smaller sub-communities.
     * 
     * @param subgraph The subgraph containing just the community nodes
     * @param nodes The nodes in the community
     * @return A mapping from nodes to new community IDs
     */
    private Map<Node, Integer> splitCommunity(Graph<Node, DefaultWeightedEdge> subgraph, List<Node> nodes) {
        // Use Leiden algorithm again but with higher resolution
        Map<Node, Integer> result = new HashMap<>();
        
        try {
            // Convert subgraph to Leiden Network format
            Map<Node, Integer> nodeToIndex = new HashMap<>();
            Map<Integer, Node> indexToNode = new HashMap<>();
            
            int index = 0;
            for (Node node : nodes) {
                nodeToIndex.put(node, index);
                indexToNode.put(index, node);
                index++;
            }
            
            // Create edge arrays
            int edgeCount = subgraph.edgeSet().size();
            LargeIntArray[] edges = new LargeIntArray[2];
            edges[0] = new LargeIntArray(edgeCount);
            edges[1] = new LargeIntArray(edgeCount);
            LargeDoubleArray edgeWeights = new LargeDoubleArray(edgeCount);
            
            int edgeIndex = 0;
            for (DefaultWeightedEdge edge : subgraph.edgeSet()) {
                Node source = subgraph.getEdgeSource(edge);
                Node target = subgraph.getEdgeTarget(edge);
                double weight = subgraph.getEdgeWeight(edge);
                
                edges[0].set(edgeIndex, nodeToIndex.get(source));
                edges[1].set(edgeIndex, nodeToIndex.get(target));
                edgeWeights.set(edgeIndex, weight);
                
                edgeIndex++;
            }
            
            // Create Network object
            Network network = new Network(nodes.size(), true, edges, edgeWeights, false, true);
            
            // Use higher resolution to get more communities
            double splitResolution = 1.5; // Higher than normal to force splitting
            LeidenAlgorithm leidenAlgorithm = new LeidenAlgorithm(splitResolution, 100, 0.001, new Random());
            
            // Create clustering
            Clustering subClustering = new Clustering(nodes.size());
            
            // Run algorithm
            leidenAlgorithm.improveClustering(network, subClustering);
            
            // Map results back to nodes
            for (int i = 0; i < subClustering.getNNodes(); i++) {
                int clusterId = subClustering.getCluster(i);
                Node node = indexToNode.get(i);
                result.put(node, clusterId);
            }
            
            // Check if we got multiple communities
            long distinctCommunities = result.values().stream().distinct().count();
            System.out.println("Split large community into " + distinctCommunities + " sub-communities");
            
            if (distinctCommunities <= 1) {
                // Splitting failed, try a geographic split as fallback
                System.out.println("Leiden community splitting failed, using geographic fallback");
                return splitCommunityGeographically(subgraph, nodes);
            }
            
            return result;
        } catch (Exception e) {
            System.err.println("Error splitting community: " + e.getMessage());
            e.printStackTrace();
            return splitCommunityGeographically(subgraph, nodes);
        }
    }
    
    /**
     * Splits a community geographically when network-based splitting fails.
     * 
     * @param subgraph The subgraph of the community
     * @param nodes The nodes in the community
     * @return A mapping from nodes to sub-community IDs
     */
    private Map<Node, Integer> splitCommunityGeographically(Graph<Node, DefaultWeightedEdge> subgraph, List<Node> nodes) {
        Map<Node, Integer> result = new HashMap<>();
        
        // Simple split based on X coordinate (longitude)
        // First, find min/max longitude
        double minLon = Double.MAX_VALUE;
        double maxLon = Double.MIN_VALUE;
        
        for (Node node : nodes) {
            double lon = node.getLocation().getX(); // Get X coordinate (longitude) from location Point
            minLon = Math.min(minLon, lon);
            maxLon = Math.max(maxLon, lon);
        }
        
        // Split nodes based on longitude (east-west split)
        double midLon = (minLon + maxLon) / 2;
        
        for (Node node : nodes) {
            int communityId = node.getLocation().getX() < midLon ? 0 : 1; // Get X coordinate from location Point
            result.put(node, communityId);
        }
        
        System.out.println("Split community geographically (east-west)");
        return result;
    }
    
    /**
     * Merges smaller communities to reduce the total number of communities to the target count
     * 
     * @param targetCount The desired number of communities after merging
     */
    private void mergeSmallerCommunities(int targetCount) {
        if (communities.size() <= targetCount) return;
        
        System.out.println("Reducing number of communities from " + communities.size() + " to target of " + targetCount);
        
        // Sort communities by size (smallest first)
        List<Map.Entry<Integer, List<Node>>> sortedCommunities = new ArrayList<>(communities.entrySet());
        sortedCommunities.sort((e1, e2) -> Integer.compare(e1.getValue().size(), e2.getValue().size()));
        
        // Calculate centroids for all communities
        Map<Integer, double[]> communityCentroids = calculateCommunityCentroids();
        
        // Keep merging smallest communities until we reach target count
        while (communities.size() > targetCount) {
            // Get smallest community
            if (sortedCommunities.isEmpty()) break;
            Map.Entry<Integer, List<Node>> smallest = sortedCommunities.remove(0);
            int smallestId = smallest.getKey();
            List<Node> smallestNodes = smallest.getValue();
            
            if (!communities.containsKey(smallestId) || communities.get(smallestId).isEmpty()) {
                continue; // Community was already merged
            }
            
            // Find geographically closest community to merge with
            int closestCommunityId = -1;
            double closestDistance = Double.MAX_VALUE;
            
            double[] smallestCentroid = communityCentroids.get(smallestId);
            if (smallestCentroid == null) continue;
            
            for (Map.Entry<Integer, List<Node>> entry : communities.entrySet()) {
                int otherId = entry.getKey();
                if (otherId == smallestId) continue; // Skip self
                
                double[] otherCentroid = communityCentroids.get(otherId);
                if (otherCentroid == null) continue;
                
                double dx = otherCentroid[0] - smallestCentroid[0];
                double dy = otherCentroid[1] - smallestCentroid[1];
                double distance = Math.sqrt(dx*dx + dy*dy);
                
                if (distance < closestDistance) {
                    closestDistance = distance;
                    closestCommunityId = otherId;
                }
            }
            
            // Merge communities
            if (closestCommunityId != -1) {
                System.out.println("Merging community " + smallestId + " (" + smallestNodes.size() + 
                                 " nodes) into community " + closestCommunityId + " (" + 
                                 communities.get(closestCommunityId).size() + " nodes)");
                                 
                communities.get(closestCommunityId).addAll(smallestNodes);
                communities.remove(smallestId);
                
                // Update centroids
                communityCentroids.remove(smallestId);
                communityCentroids.put(closestCommunityId, calculateCentroid(communities.get(closestCommunityId)));
                
                // Update sorted list
                sortedCommunities.clear();
                sortedCommunities.addAll(communities.entrySet());
                sortedCommunities.sort((e1, e2) -> Integer.compare(e1.getValue().size(), e2.getValue().size()));
            }
        }
        
        // Update clustering object
        updateClusteringFromCommunities();
        
        System.out.println("After merging, final community count: " + communities.size());
    }
    
    /**
     * Calculate the centroid for a single community
     */
    private double[] calculateCentroid(List<Node> nodes) {
        if (nodes == null || nodes.isEmpty()) return null;
        
        double sumX = 0.0, sumY = 0.0;
        int nodesWithLocation = 0;
        
        for (Node node : nodes) {
            if (node.getLocation() != null) {
                sumX += node.getLocation().getX();
                sumY += node.getLocation().getY();
                nodesWithLocation++;
            }
        }
        
        if (nodesWithLocation > 0) {
            double[] centroid = new double[2];
            centroid[0] = sumX / nodesWithLocation;
            centroid[1] = sumY / nodesWithLocation;
            return centroid;
        } else {
            return null;
        }
    }
} 