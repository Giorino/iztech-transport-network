package com.izmir.transportation.helper.outlier;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ForkJoinPool;
import java.util.logging.Logger;

import org.jgrapht.Graph;
import org.jgrapht.alg.shortestpath.DijkstraShortestPath;
import org.jgrapht.graph.DefaultWeightedEdge;

import com.izmir.transportation.TransportationGraph;
import com.izmir.transportation.helper.Node;

/**
 * Implements outlier detection based on eccentricity - nodes whose longest 
 * shortest-path distance to the graph is extreme (very peripheral or hyper-central).
 * 
 * This implementation uses parallel processing to speed up the computation.
 * 
 * Time complexity: O(V · log V)
 */
public class EccentricityOutlierDetection implements OutlierDetectionAlgorithm {
    
    private static final Logger LOGGER = Logger.getLogger(EccentricityOutlierDetection.class.getName());
    private boolean useParallel = true; // Flag to enable/disable parallel processing
    private int numThreads = Runtime.getRuntime().availableProcessors(); // Default to available processors
    
    /**
     * Default constructor with parallel processing enabled
     */
    public EccentricityOutlierDetection() {
        // Use default settings
    }
    
    /**
     * Constructor with option to enable/disable parallel processing
     * 
     * @param useParallel Whether to use parallel processing
     */
    public EccentricityOutlierDetection(boolean useParallel) {
        this.useParallel = useParallel;
    }
    
    /**
     * Constructor with option to specify number of threads
     * 
     * @param numThreads Number of threads to use for parallel processing
     */
    public EccentricityOutlierDetection(int numThreads) {
        this.numThreads = Math.max(1, numThreads);
    }
    
    @Override
    public Set<Node> detectOutliers(TransportationGraph graph, double threshold) {
        LOGGER.info("Detecting outliers using eccentricity with threshold " + threshold 
                  + (useParallel ? " (parallel with " + numThreads + " threads)" : " (single-threaded)"));
        
        Set<Node> outliers = ConcurrentHashMap.newKeySet();
        Graph<Node, DefaultWeightedEdge> jgraph = graph.getGraph();
        
        List<Node> nodes = new ArrayList<>(jgraph.vertexSet());
        Map<Node, Double> eccentricities = new ConcurrentHashMap<>();
        
        // Calculate eccentricity for each node (maximum distance to any other node)
        if (useParallel) {
            // Use a custom thread pool with the specified number of threads
            ForkJoinPool customThreadPool = new ForkJoinPool(numThreads);
            try {
                customThreadPool.submit(() -> 
                    nodes.parallelStream()
                        .filter(node -> !node.isOutlier()) // Skip already identified outliers
                        .forEach(node -> {
                            double maxDistance = calculateEccentricity(node, nodes, jgraph);
                            eccentricities.put(node, maxDistance);
                        })
                ).get(); // Wait for completion
            } catch (Exception e) {
                LOGGER.warning("Error in parallel eccentricity calculation: " + e.getMessage());
                // Fall back to sequential processing
                for (Node node : nodes) {
                    if (node.isOutlier()) continue;
                    double maxDistance = calculateEccentricity(node, nodes, jgraph);
                    eccentricities.put(node, maxDistance);
                }
            } finally {
                customThreadPool.shutdown();
            }
        } else {
            // Sequential processing
            for (Node node : nodes) {
                if (node.isOutlier()) continue;
                double maxDistance = calculateEccentricity(node, nodes, jgraph);
                eccentricities.put(node, maxDistance);
            }
        }
        
        // Calculate mean and standard deviation
        List<Double> values = new ArrayList<>(eccentricities.values());
        // Filter out infinity values
        values.removeIf(v -> v == Double.MAX_VALUE || v == Double.POSITIVE_INFINITY);
        
        if (values.isEmpty()) {
            LOGGER.warning("No valid eccentricity values found. All nodes appear to be in disconnected components.");
            return outliers;
        }
        
        double mean = values.stream().mapToDouble(Double::doubleValue).average().orElse(0);
        double stdDev = calculateStandardDeviation(values, mean);
        
        LOGGER.info("Eccentricity stats: mean=" + mean + ", stdDev=" + stdDev);
        
        // Mark nodes as outliers if their eccentricity is beyond the threshold
        for (Map.Entry<Node, Double> entry : eccentricities.entrySet()) {
            if (entry.getValue() == Double.MAX_VALUE || entry.getValue() == Double.POSITIVE_INFINITY ||
                Math.abs(entry.getValue() - mean) > threshold * stdDev) {
                outliers.add(entry.getKey());
            }
        }
        
        LOGGER.info("Detected " + outliers.size() + " outliers using eccentricity");
        return outliers;
    }
    
    /**
     * Calculate the eccentricity of a node (maximum shortest path distance to any other node)
     */
    private double calculateEccentricity(Node node, List<Node> allNodes, Graph<Node, DefaultWeightedEdge> graph) {
        DijkstraShortestPath<Node, DefaultWeightedEdge> dijkstra = new DijkstraShortestPath<>(graph);
        double maxDistance = 0.0;
        
        for (Node target : allNodes) {
            if (node.equals(target) || target.isOutlier()) continue;
            
            try {
                double distance = dijkstra.getPathWeight(node, target);
                maxDistance = Math.max(maxDistance, distance);
            } catch (IllegalArgumentException e) {
                // No path exists - node may be in a disconnected component
                // Treat as max distance for outlier detection purposes
                maxDistance = Double.MAX_VALUE;
                break;
            }
        }
        
        return maxDistance;
    }
    
    @Override
    public String getName() {
        return "Eccentricity" + (useParallel ? " (Parallel)" : "");
    }
    
    /**
     * Sets whether to use parallel processing.
     * 
     * @param useParallel true to enable parallel processing, false for single-threaded
     * @return This instance for method chaining
     */
    public EccentricityOutlierDetection setUseParallel(boolean useParallel) {
        this.useParallel = useParallel;
        return this;
    }
    
    /**
     * Sets the number of threads to use for parallel processing.
     * 
     * @param numThreads Number of threads (min 1)
     * @return This instance for method chaining
     */
    public EccentricityOutlierDetection setNumThreads(int numThreads) {
        this.numThreads = Math.max(1, numThreads);
        return this;
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
} 