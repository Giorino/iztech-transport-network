package com.izmir.transportation.helper.clustering;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

import org.jgrapht.Graph;
import org.jgrapht.graph.DefaultWeightedEdge;
import org.locationtech.jts.geom.Point;

import com.izmir.transportation.TransportationGraph;
import com.izmir.transportation.helper.Node;
import com.izmir.transportation.helper.clustering.leiden.Clustering;
import com.izmir.transportation.helper.clustering.leiden.LeidenAlgorithm;
import com.izmir.transportation.helper.clustering.leiden.Network;
import com.izmir.transportation.helper.clustering.leiden.util.LargeDoubleArray;
import com.izmir.transportation.helper.clustering.leiden.util.LargeIntArray;

/**
 * Implementation of the Leiden algorithm for community detection in transportation networks.
 * This class adapts the original Leiden algorithm implementation to work with the TransportationGraph.
 * 
 * The Leiden algorithm is an improved version of the Louvain algorithm that guarantees
 * well-connected communities. It consists of three phases:
 * 1. Local moving of nodes between clusters
 * 2. Refinement of the clusters
 * 3. Aggregation of the network based on the refined clusters
 * 
 * @author yagizugurveren
 */
public class LeidenCommunityDetection implements GraphClusteringAlgorithm {
    
    private double resolution;
    private int iterations;
    private double randomness;
    private Random random;
    
    /**
     * Constructs a new LeidenCommunityDetection with default parameters.
     */
    public LeidenCommunityDetection() {
        this(0.01, 100, 0.01, new Random(42));
    }
    
    /**
     * Constructs a new LeidenCommunityDetection with specified parameters.
     * 
     * @param resolution Resolution parameter for the algorithm (higher values lead to more communities)
     * @param iterations Number of iterations to run the algorithm
     * @param randomness Randomness parameter (between 0 and 1)
     * @param random Random number generator
     */
    public LeidenCommunityDetection(double resolution, int iterations, double randomness, Random random) {
        this.resolution = resolution;
        this.iterations = iterations;
        this.randomness = randomness;
        this.random = random;
    }
    
    @Override
    public List<List<Node>> findCommunities(TransportationGraph transportGraph) {
        System.out.println("Running Leiden community detection algorithm...");
        System.out.println("Parameters: resolution=" + resolution + ", iterations=" + iterations + 
                          ", randomness=" + randomness);
        
        // Get the subgraph containing only the original points
        Graph<Node, DefaultWeightedEdge> jgraphtGraph = transportGraph.getOriginalPointsGraph();
        
        // Print original node count for debugging
        int nodeCount = jgraphtGraph.vertexSet().size();
        int edgeCount = jgraphtGraph.edgeSet().size();
        System.out.println("Original node count: " + nodeCount);
        System.out.println("Original edge count: " + edgeCount);
        
        // Check if the graph is too sparse
        double density = calculateGraphDensity(jgraphtGraph);
        System.out.println("Graph density: " + density);
        
        // Always enhance connectivity to ensure we have enough edges for community detection
        System.out.println("Enhancing graph connectivity...");
        enhanceConnectivity(jgraphtGraph);
        
        // Recalculate density
        density = calculateGraphDensity(jgraphtGraph);
        System.out.println("Enhanced graph density: " + density);
        System.out.println("Enhanced node count: " + jgraphtGraph.vertexSet().size());
        System.out.println("Enhanced edge count: " + jgraphtGraph.edgeSet().size());
        
        // Convert JGraphT graph to Leiden Network format
        Network leidenNetwork = convertToLeidenNetwork(jgraphtGraph);
        System.out.println("Leiden network created with " + leidenNetwork.getNNodes() + " nodes and " + 
                          leidenNetwork.getNEdges() + " edges");
        
        // Create and run the Leiden algorithm
        System.out.println("Creating Leiden algorithm with resolution=" + resolution);
        LeidenAlgorithm leidenAlgorithm = new LeidenAlgorithm(resolution, iterations, randomness, random);
        
        // Create initial clustering (singleton)
        Clustering initialClustering = new Clustering(leidenNetwork.getNNodes());
        System.out.println("Starting with singleton clustering (" + initialClustering.getNClusters() + " clusters)");
        
        // Improve the clustering
        leidenAlgorithm.improveClustering(leidenNetwork, initialClustering);
        
        // Calculate quality
        double quality = leidenAlgorithm.calcQuality(leidenNetwork, initialClustering);
        System.out.println("Leiden algorithm quality: " + quality);
        System.out.println("Leiden algorithm found " + initialClustering.getNClusters() + " communities");
        
        // Convert the clustering result back to our format
        return convertFromLeidenClustering(initialClustering, jgraphtGraph);
    }
    
    /**
     * Converts a JGraphT graph to the Network format required by the Leiden algorithm.
     * 
     * @param jgraphtGraph The JGraphT graph to convert
     * @return A Network object compatible with the Leiden algorithm
     */
    private Network convertToLeidenNetwork(Graph<Node, DefaultWeightedEdge> jgraphtGraph) {
        int nNodes = jgraphtGraph.vertexSet().size();
        
        // Create a mapping from Node objects to indices
        Map<Node, Integer> nodeToIndex = new HashMap<>();
        List<Node> indexToNode = new ArrayList<>(nNodes);
        
        int index = 0;
        for (Node node : jgraphtGraph.vertexSet()) {
            nodeToIndex.put(node, index);
            indexToNode.add(node);
            index++;
        }
        
        // Create edge arrays
        int nEdges = jgraphtGraph.edgeSet().size();
        LargeIntArray[] edges = new LargeIntArray[2];
        edges[0] = new LargeIntArray(nEdges);
        edges[1] = new LargeIntArray(nEdges);
        LargeDoubleArray edgeWeights = new LargeDoubleArray(nEdges);
        
        // Find min and max weights for normalization
        double minWeight = Double.MAX_VALUE;
        double maxWeight = Double.MIN_VALUE;
        for (DefaultWeightedEdge edge : jgraphtGraph.edgeSet()) {
            double weight = jgraphtGraph.getEdgeWeight(edge);
            minWeight = Math.min(minWeight, weight);
            maxWeight = Math.max(maxWeight, weight);
        }
        
        System.out.println("Edge weight range: " + minWeight + " to " + maxWeight);
        
        // Process edges
        index = 0;
        for (DefaultWeightedEdge edge : jgraphtGraph.edgeSet()) {
            Node source = jgraphtGraph.getEdgeSource(edge);
            Node target = jgraphtGraph.getEdgeTarget(edge);
            
            edges[0].set(index, nodeToIndex.get(source));
            edges[1].set(index, nodeToIndex.get(target));
            
            // Get the original weight
            double weight = jgraphtGraph.getEdgeWeight(edge);
            
            // IMPORTANT: In Leiden algorithm, LOWER weights mean STRONGER connections
            // We need to invert our weights since in our graph higher weights mean stronger connections
            
            // Use a very simple weight transformation to ensure the algorithm works properly
            // Just use a constant weight for all edges - this focuses on topology rather than weights
            double finalWeight = 0.1;
            
            edgeWeights.set(index, finalWeight);
            index++;
        }
        
        // Create the Network object with the correct constructor
        return new Network(nNodes, true, edges, edgeWeights, false, true);
    }
    
    /**
     * Converts a Leiden Clustering result back to a list of communities in our format.
     * 
     * @param clustering The Leiden Clustering result
     * @param jgraphtGraph The original JGraphT graph
     * @return A list of communities, where each community is a list of nodes
     */
    private List<List<Node>> convertFromLeidenClustering(Clustering clustering, Graph<Node, DefaultWeightedEdge> jgraphtGraph) {
        int nClusters = clustering.getNClusters();
        List<List<Node>> communities = new ArrayList<>(nClusters);
        
        // Initialize the communities list
        for (int i = 0; i < nClusters; i++) {
            communities.add(new ArrayList<>());
        }
        
        // Create a list of nodes in the same order as they were added to the Network
        List<Node> nodes = new ArrayList<>(jgraphtGraph.vertexSet());
        
        // Assign each node to its community
        for (int i = 0; i < nodes.size(); i++) {
            int cluster = clustering.getCluster(i);
            communities.get(cluster).add(nodes.get(i));
        }
        
        // Remove empty communities
        communities.removeIf(List::isEmpty);
        
        // Debug information
        System.out.println("Original node count: " + nodes.size());
        System.out.println("Original cluster count: " + nClusters);
        System.out.println("Final community count: " + communities.size());
        
        // Print community size distribution
        Map<Integer, Integer> sizeDistribution = new HashMap<>();
        for (List<Node> community : communities) {
            int size = community.size();
            sizeDistribution.put(size, sizeDistribution.getOrDefault(size, 0) + 1);
        }
        
        System.out.println("Community size distribution:");
        List<Integer> sizes = new ArrayList<>(sizeDistribution.keySet());
        Collections.sort(sizes);
        for (int size : sizes) {
            System.out.println(size + " nodes: " + sizeDistribution.get(size) + " communities");
        }
        
        return communities;
    }
    
    /**
     * Calculates the density of the graph.
     * Density = actual edges / possible edges
     * For an undirected graph: possible edges = n(n-1)/2
     * 
     * @param graph The graph to calculate density for
     * @return The density value between 0 and 1
     */
    private double calculateGraphDensity(Graph<Node, DefaultWeightedEdge> graph) {
        int n = graph.vertexSet().size();
        if (n <= 1) return 0;
        
        double possibleEdges = (n * (n - 1)) / 2.0;
        double actualEdges = graph.edgeSet().size();
        
        return actualEdges / possibleEdges;
    }
    
    /**
     * Enhances the connectivity of the graph by adding edges between nearby nodes.
     * This helps ensure that the graph has enough connectivity for community detection.
     * 
     * @param graph The graph to enhance
     */
    private void enhanceConnectivity(Graph<Node, DefaultWeightedEdge> graph) {
        List<Node> nodes = new ArrayList<>(graph.vertexSet());
        int n = nodes.size();
        
        // Calculate the minimum number of edges needed for reasonable connectivity
        // We want to ensure the graph is connected enough for community detection
        int targetEdgeCount = Math.max(3 * n, (int)(0.1 * n * (n-1) / 2));
        int currentEdgeCount = graph.edgeSet().size();
        
        if (currentEdgeCount >= targetEdgeCount) {
            System.out.println("Graph already has sufficient connectivity.");
            return;
        }
        
        System.out.println("Adding edges to enhance connectivity. Target: " + targetEdgeCount + 
                          " edges, Current: " + currentEdgeCount + " edges");
        
        // Create a list of potential edges (node pairs) sorted by distance
        List<EdgeCandidate> candidates = new ArrayList<>();
        
        for (int i = 0; i < n; i++) {
            Node source = nodes.get(i);
            Point sourcePoint = source.getLocation();
            
            for (int j = i + 1; j < n; j++) {
                Node target = nodes.get(j);
                
                // Skip if edge already exists
                if (graph.containsEdge(source, target)) {
                    continue;
                }
                
                Point targetPoint = target.getLocation();
                double distance = sourcePoint.distance(targetPoint);
                
                candidates.add(new EdgeCandidate(source, target, distance));
            }
        }
        
        // Sort candidates by distance (ascending)
        Collections.sort(candidates);
        
        // Add edges until we reach the target count or run out of candidates
        int added = 0;
        for (EdgeCandidate candidate : candidates) {
            if (currentEdgeCount + added >= targetEdgeCount) {
                break;
            }
            
            DefaultWeightedEdge edge = graph.addEdge(candidate.source, candidate.target);
            if (edge != null) {
                graph.setEdgeWeight(edge, candidate.distance);
                added++;
            }
        }
        
        System.out.println("Added " + added + " edges to enhance connectivity.");
    }
    
    /**
     * Helper class to represent a potential edge between two nodes.
     */
    private static class EdgeCandidate implements Comparable<EdgeCandidate> {
        Node source;
        Node target;
        double distance;
        
        EdgeCandidate(Node source, Node target, double distance) {
            this.source = source;
            this.target = target;
            this.distance = distance;
        }
        
        @Override
        public int compareTo(EdgeCandidate other) {
            return Double.compare(this.distance, other.distance);
        }
    }
} 