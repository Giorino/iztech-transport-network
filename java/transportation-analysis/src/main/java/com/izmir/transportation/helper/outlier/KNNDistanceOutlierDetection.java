package com.izmir.transportation.helper.outlier;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.logging.Logger;

import org.jgrapht.Graph;
import org.jgrapht.graph.DefaultWeightedEdge;
import org.locationtech.jts.geom.Coordinate;
import org.locationtech.jts.geom.Point;

import com.izmir.transportation.TransportationGraph;
import com.izmir.transportation.helper.Node;

/**
 * Implements outlier detection based on k-Nearest-Neighbor (kNN) distance - 
 * nodes that are unusually far from their k closest neighbors (isolated developments) 
 * or too close (duplicate nodes).
 * 
 * This implementation uses parallel processing to speed up calculations.
 * 
 * Time complexity: O(k · V · log V)
 */
public class KNNDistanceOutlierDetection implements OutlierDetectionAlgorithm {
    
    private static final Logger LOGGER = Logger.getLogger(KNNDistanceOutlierDetection.class.getName());
    private int k = 5; // Default k value
    private boolean useParallel = true; // Flag to enable/disable parallel processing
    private int numThreads = Runtime.getRuntime().availableProcessors(); // Default to available processors
    
    /**
     * Default constructor with k=5 and parallel processing enabled
     */
    public KNNDistanceOutlierDetection() {
        // Use default settings
    }
    
    /**
     * Constructor with custom k value and parallel processing enabled
     * 
     * @param k The number of nearest neighbors to consider
     */
    public KNNDistanceOutlierDetection(int k) {
        this.k = Math.max(1, k); // Ensure k is at least 1
    }
    
    /**
     * Constructor with options for k value and parallel processing
     * 
     * @param k The number of nearest neighbors to consider
     * @param useParallel Whether to use parallel processing
     */
    public KNNDistanceOutlierDetection(int k, boolean useParallel) {
        this.k = Math.max(1, k);
        this.useParallel = useParallel;
    }
    
    /**
     * Constructor with options for k value and number of threads
     * 
     * @param k The number of nearest neighbors to consider
     * @param numThreads Number of threads to use for parallel processing
     */
    public KNNDistanceOutlierDetection(int k, int numThreads) {
        this.k = Math.max(1, k);
        this.numThreads = Math.max(1, numThreads);
    }
    
    @Override
    public Set<Node> detectOutliers(TransportationGraph graph, double threshold) {
        LOGGER.info("Detecting outliers using " + k + "-Nearest-Neighbor distance with threshold " + threshold
                  + (useParallel ? " (parallel with " + numThreads + " threads)" : " (single-threaded)"));
        
        Set<Node> outliers = ConcurrentHashMap.newKeySet();
        Graph<Node, DefaultWeightedEdge> jgraph = graph.getGraph();
        
        List<Node> nodes = new ArrayList<>();
        // Only include nodes that are not already marked as outliers
        for (Node node : jgraph.vertexSet()) {
            if (!node.isOutlier()) {
                nodes.add(node);
            }
        }
        
        if (nodes.size() <= k + 1) {
            LOGGER.warning("Not enough nodes for kNN analysis. Need at least k+2 nodes.");
            return outliers;
        }
        
        // Calculate average distance to k nearest neighbors for each node
        Map<Node, Double> knnDistances = new ConcurrentHashMap<>();
        AtomicInteger processedCount = new AtomicInteger(0);
        int totalNodes = nodes.size();
        
        if (useParallel) {
            // Create a custom thread pool with the specified number of threads
            ForkJoinPool customThreadPool = new ForkJoinPool(numThreads);
            
            try {
                customThreadPool.submit(() -> 
                    nodes.parallelStream().forEach(node -> {
                        double avgDistance = calculateAverageKnnDistance(node, nodes);
                        knnDistances.put(node, avgDistance);
                        
                        // Log progress periodically
                        int processed = processedCount.incrementAndGet();
                        if (processed % 100 == 0 || processed == totalNodes) {
                            LOGGER.info("Processed " + processed + "/" + totalNodes + " nodes for kNN analysis");
                        }
                    })
                ).get(); // Wait for completion
            } catch (Exception e) {
                LOGGER.warning("Error in parallel kNN calculation: " + e.getMessage());
                // Fall back to sequential processing
                processedCount.set(0);
                for (Node node : nodes) {
                    double avgDistance = calculateAverageKnnDistance(node, nodes);
                    knnDistances.put(node, avgDistance);
                    
                    // Log progress periodically
                    int processed = processedCount.incrementAndGet();
                    if (processed % 100 == 0 || processed == totalNodes) {
                        LOGGER.info("Processed " + processed + "/" + totalNodes + " nodes for kNN analysis");
                    }
                }
            } finally {
                customThreadPool.shutdown();
            }
        } else {
            // Sequential processing
            for (Node node : nodes) {
                double avgDistance = calculateAverageKnnDistance(node, nodes);
                knnDistances.put(node, avgDistance);
                
                // Log progress periodically
                int processed = processedCount.incrementAndGet();
                if (processed % 100 == 0 || processed == totalNodes) {
                    LOGGER.info("Processed " + processed + "/" + totalNodes + " nodes for kNN analysis");
                }
            }
        }
        
        // Calculate mean and standard deviation of kNN distances
        List<Double> values = new ArrayList<>(knnDistances.values());
        if (values.isEmpty()) {
            LOGGER.warning("No valid kNN distance values found.");
            return outliers;
        }
        
        double mean = values.stream().mapToDouble(Double::doubleValue).average().orElse(0);
        double stdDev = calculateStandardDeviation(values, mean);
        
        LOGGER.info("kNN distance stats: mean=" + mean + ", stdDev=" + stdDev);
        
        // Mark nodes as outliers if their kNN distance is beyond the threshold
        // This detects both unusually isolated nodes (high distance) and potential duplicates (very low distance)
        for (Map.Entry<Node, Double> entry : knnDistances.entrySet()) {
            if (Math.abs(entry.getValue() - mean) > threshold * stdDev) {
                outliers.add(entry.getKey());
            }
        }
        
        LOGGER.info("Detected " + outliers.size() + " outliers using kNN distance");
        return outliers;
    }
    
    /**
     * Calculate the average distance to k nearest neighbors for a node.
     * 
     * @param node The node to calculate for
     * @param allNodes All nodes in the graph
     * @return The average distance to k nearest neighbors
     */
    private double calculateAverageKnnDistance(Node node, List<Node> allNodes) {
        List<NodeDistance> distances = new ArrayList<>();
        
        // Calculate distances to all other nodes
        for (Node other : allNodes) {
            if (!node.equals(other)) {
                double distance = calculateDistance(node, other);
                distances.add(new NodeDistance(other, distance));
            }
        }
        
        // Sort by distance (closest first)
        Collections.sort(distances);
        
        // Calculate average distance to k nearest neighbors
        double sum = 0.0;
        int count = Math.min(k, distances.size());
        for (int i = 0; i < count; i++) {
            sum += distances.get(i).distance;
        }
        
        return sum / count;
    }
    
    @Override
    public String getName() {
        return k + "-Nearest-Neighbor Distance" + (useParallel ? " (Parallel)" : "");
    }
    
    /**
     * Sets the k value for the k-nearest neighbor calculation.
     * 
     * @param k The number of nearest neighbors to consider
     * @return This instance for method chaining
     */
    public KNNDistanceOutlierDetection setK(int k) {
        this.k = Math.max(1, k);
        return this;
    }
    
    /**
     * Gets the current k value.
     * 
     * @return The k value
     */
    public int getK() {
        return k;
    }
    
    /**
     * Sets whether to use parallel processing.
     * 
     * @param useParallel true to enable parallel processing, false for single-threaded
     * @return This instance for method chaining
     */
    public KNNDistanceOutlierDetection setUseParallel(boolean useParallel) {
        this.useParallel = useParallel;
        return this;
    }
    
    /**
     * Sets the number of threads to use for parallel processing.
     * 
     * @param numThreads Number of threads (min 1)
     * @return This instance for method chaining
     */
    public KNNDistanceOutlierDetection setNumThreads(int numThreads) {
        this.numThreads = Math.max(1, numThreads);
        return this;
    }
    
    /**
     * Calculates the Euclidean distance between two nodes.
     */
    private double calculateDistance(Node node1, Node node2) {
        Point p1 = node1.getLocation();
        Point p2 = node2.getLocation();
        
        // Using Euclidean distance for simplicity
        // For geographic coordinates, a proper geodesic distance would be better
        Coordinate c1 = p1.getCoordinate();
        Coordinate c2 = p2.getCoordinate();
        
        return Math.sqrt(Math.pow(c1.x - c2.x, 2) + Math.pow(c1.y - c2.y, 2));
    }
    
    /**
     * Calculates the standard deviation of a collection of values.
     */
    private double calculateStandardDeviation(List<Double> values, double mean) {
        double sum = 0.0;
        for (double value : values) {
            sum += Math.pow(value - mean, 2);
        }
        return Math.sqrt(sum / values.size());
    }
    
    /**
     * Helper class to store a node and its distance
     */
    private static class NodeDistance implements Comparable<NodeDistance> {
        Node node;
        double distance;
        
        NodeDistance(Node node, double distance) {
            this.node = node;
            this.distance = distance;
        }
        
        @Override
        public int compareTo(NodeDistance other) {
            return Double.compare(this.distance, other.distance);
        }
    }
} 