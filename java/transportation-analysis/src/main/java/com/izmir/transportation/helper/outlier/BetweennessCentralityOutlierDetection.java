package com.izmir.transportation.helper.outlier;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.logging.Logger;

import org.jgrapht.Graph;
import org.jgrapht.alg.scoring.BetweennessCentrality;
import org.jgrapht.graph.DefaultWeightedEdge;

import com.izmir.transportation.TransportationGraph;
import com.izmir.transportation.helper.Node;

/**
 * Implements outlier detection based on betweenness centrality - junctions or roads
 * that sit on far more shortest paths than others (single points of failure).
 * 
 * This implementation uses parallel processing for collecting scores.
 * 
 * Time complexity: O(V · E)
 */
public class BetweennessCentralityOutlierDetection implements OutlierDetectionAlgorithm {
    
    private static final Logger LOGGER = Logger.getLogger(BetweennessCentralityOutlierDetection.class.getName());
    private boolean useParallel = true; // Flag to enable/disable parallel processing
    private int numThreads = Runtime.getRuntime().availableProcessors(); // Default to available processors
    
    /**
     * Default constructor with parallel processing enabled
     */
    public BetweennessCentralityOutlierDetection() {
        // Use default settings
    }
    
    /**
     * Constructor with option to enable/disable parallel processing
     * 
     * @param useParallel Whether to use parallel processing
     */
    public BetweennessCentralityOutlierDetection(boolean useParallel) {
        this.useParallel = useParallel;
    }
    
    /**
     * Constructor with option to specify number of threads
     * 
     * @param numThreads Number of threads to use for parallel processing
     */
    public BetweennessCentralityOutlierDetection(int numThreads) {
        this.numThreads = Math.max(1, numThreads);
    }
    
    @Override
    public Set<Node> detectOutliers(TransportationGraph graph, double threshold) {
        LOGGER.info("Detecting outliers using betweenness centrality with threshold " + threshold 
                  + (useParallel ? " (parallel with " + numThreads + " threads)" : " (single-threaded)"));
        
        Set<Node> outliers = ConcurrentHashMap.newKeySet();
        Graph<Node, DefaultWeightedEdge> jgraph = graph.getGraph();
        
        // Use JGraphT's implementation of betweenness centrality
        LOGGER.info("Calculating betweenness centrality for all nodes...");
        BetweennessCentrality<Node, DefaultWeightedEdge> bc = new BetweennessCentrality<>(jgraph);
        Map<Node, Double> scores = new ConcurrentHashMap<>();
        
        // Get all nodes that are not already marked as outliers
        List<Node> nonOutlierNodes = new ArrayList<>();
        for (Node node : jgraph.vertexSet()) {
            if (!node.isOutlier()) {
                nonOutlierNodes.add(node);
            }
        }
        
        // Collect scores for all non-outlier nodes
        if (useParallel && nonOutlierNodes.size() > 100) { // Only use parallel for larger graphs
            LOGGER.info("Using parallel processing to collect betweenness centrality scores...");
            
            // Create thread pool
            ExecutorService executor = Executors.newFixedThreadPool(numThreads);
            
            // Submit tasks to collect scores
            for (Node node : nonOutlierNodes) {
                executor.submit(() -> {
                    double score = bc.getVertexScore(node);
                    scores.put(node, score);
                });
            }
            
            // Shutdown executor and wait for completion
            executor.shutdown();
            try {
                if (!executor.awaitTermination(5, TimeUnit.MINUTES)) {
                    LOGGER.warning("Timeout while calculating betweenness centrality scores");
                    executor.shutdownNow();
                }
            } catch (InterruptedException e) {
                LOGGER.warning("Interrupted while calculating betweenness centrality: " + e.getMessage());
                Thread.currentThread().interrupt();
            }
        } else {
            // Sequential processing
            for (Node node : nonOutlierNodes) {
                double score = bc.getVertexScore(node);
                scores.put(node, score);
            }
        }
        
        LOGGER.info("Betweenness centrality calculation complete. Analyzing " + scores.size() + " node scores...");
        
        // Calculate mean and standard deviation
        List<Double> values = new ArrayList<>(scores.values());
        if (values.isEmpty()) {
            LOGGER.warning("No valid betweenness centrality values found.");
            return outliers;
        }
        
        double mean = values.stream().mapToDouble(Double::doubleValue).average().orElse(0);
        double stdDev = calculateStandardDeviation(values, mean);
        
        LOGGER.info("Betweenness centrality stats: mean=" + mean + ", stdDev=" + stdDev);
        
        // Mark nodes as outliers if their betweenness centrality is beyond the threshold
        // (looking for unusually high betweenness values)
        for (Map.Entry<Node, Double> entry : scores.entrySet()) {
            if (entry.getValue() > mean + threshold * stdDev) {
                outliers.add(entry.getKey());
            }
        }
        
        LOGGER.info("Detected " + outliers.size() + " outliers using betweenness centrality");
        return outliers;
    }
    
    @Override
    public String getName() {
        return "Betweenness Centrality" + (useParallel ? " (Parallel)" : "");
    }
    
    /**
     * Sets whether to use parallel processing.
     * 
     * @param useParallel true to enable parallel processing, false for single-threaded
     * @return This instance for method chaining
     */
    public BetweennessCentralityOutlierDetection setUseParallel(boolean useParallel) {
        this.useParallel = useParallel;
        return this;
    }
    
    /**
     * Sets the number of threads to use for parallel processing.
     * 
     * @param numThreads Number of threads (min 1)
     * @return This instance for method chaining
     */
    public BetweennessCentralityOutlierDetection setNumThreads(int numThreads) {
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