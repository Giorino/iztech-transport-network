package com.izmir.transportation;

import java.util.Set;
import java.util.logging.Logger;

import com.izmir.transportation.helper.Node;
import com.izmir.transportation.helper.outlier.BetweennessCentralityOutlierDetection;
import com.izmir.transportation.helper.outlier.EccentricityOutlierDetection;
import com.izmir.transportation.helper.outlier.KNNDistanceOutlierDetection;
import com.izmir.transportation.helper.outlier.OutlierDetectionAlgorithm;

/**
 * Service for detecting outliers in transportation graphs.
 * Provides methods to apply different outlier detection algorithms
 * and mark nodes as outliers in the graph.
 */
public class OutlierDetectionService {
    
    private static final Logger LOGGER = Logger.getLogger(OutlierDetectionService.class.getName());
    
    /**
     * Enum of supported outlier detection algorithms.
     */
    public enum OutlierAlgorithm {
        ECCENTRICITY("eccentricity"),
        BETWEENNESS_CENTRALITY("betweenness_centrality"),
        KNN_DISTANCE("knn_distance");
        
        private final String code;
        
        OutlierAlgorithm(String code) {
            this.code = code;
        }
        
        public String getCode() {
            return code;
        }
        
        public static OutlierAlgorithm fromCode(String code) {
            for (OutlierAlgorithm algorithm : values()) {
                if (algorithm.getCode().equals(code)) {
                    return algorithm;
                }
            }
            throw new IllegalArgumentException("Unknown outlier algorithm code: " + code);
        }
    }
    
    // Default configuration
    private double threshold = 2.0; // default: 2 standard deviations
    private int kValue = 5; // default k value for KNN_DISTANCE algorithm
    private boolean useParallel = true; // Whether to use parallel processing
    private int numThreads = Runtime.getRuntime().availableProcessors(); // Default to available cores
    
    /**
     * Detects outliers in the graph using the specified algorithm and
     * marks them in the graph.
     * 
     * @param graph The transportation graph
     * @param algorithm The outlier detection algorithm to use
     * @param visualize Whether to visualize the outliers
     * @param visualizeLegend Whether to show the legend in the outlier visualization
     * @return Set of nodes identified as outliers
     */
    public Set<Node> detectOutliers(
            TransportationGraph graph, 
            OutlierAlgorithm algorithm,
            boolean visualize,
            boolean visualizeLegend) {
        
        LOGGER.info("Detecting outliers using " + algorithm.getCode() + " algorithm with threshold " + threshold +
                   (useParallel ? " (parallel with " + numThreads + " threads)" : " (single-threaded)"));
        
        OutlierDetectionAlgorithm detectionAlgorithm = createAlgorithm(algorithm);
        long startTime = System.currentTimeMillis();
        Set<Node> outliers = detectionAlgorithm.detectOutliers(graph, threshold);
        long endTime = System.currentTimeMillis();
        
        LOGGER.info("Outlier detection completed in " + (endTime - startTime) + "ms");
        
        // Mark outliers in the graph
        for (Node node : outliers) {
            node.setOutlier(true);
        }
        
        if (visualize && !outliers.isEmpty()) {
            LOGGER.info("Visualizing " + outliers.size() + " outliers");
            visualizeOutliers(graph, outliers, detectionAlgorithm.getName(), visualizeLegend);
        }
        
        return outliers;
    }
    
    /**
     * Creates the appropriate outlier detection algorithm implementation.
     */
    private OutlierDetectionAlgorithm createAlgorithm(OutlierAlgorithm algorithmType) {
        switch (algorithmType) {
            case ECCENTRICITY:
                EccentricityOutlierDetection eccentricity = new EccentricityOutlierDetection();
                eccentricity.setUseParallel(useParallel)
                           .setNumThreads(numThreads);
                return eccentricity;
            case BETWEENNESS_CENTRALITY:
                BetweennessCentralityOutlierDetection betweenness = new BetweennessCentralityOutlierDetection();
                betweenness.setUseParallel(useParallel)
                           .setNumThreads(numThreads);
                return betweenness;
            case KNN_DISTANCE:
                KNNDistanceOutlierDetection knn = new KNNDistanceOutlierDetection(kValue);
                knn.setUseParallel(useParallel)
                   .setNumThreads(numThreads);
                return knn;
            default:
                throw new IllegalArgumentException("Unknown algorithm type: " + algorithmType);
        }
    }
    
    /**
     * Visualizes the outliers in the graph.
     * This will call the appropriate visualization method in TransportationGraph.
     */
    private void visualizeOutliers(
            TransportationGraph graph, 
            Set<Node> outliers, 
            String algorithmName,
            boolean visualizeLegend) {
        
        // Call the visualization method in TransportationGraph
        graph.visualizeOutliers(outliers, algorithmName, visualizeLegend);
    }
    
    /**
     * Sets the threshold for outlier detection.
     * 
     * @param threshold The threshold in standard deviations (default: 2.0)
     * @return This service instance for method chaining
     */
    public OutlierDetectionService setThreshold(double threshold) {
        this.threshold = threshold;
        return this;
    }
    
    /**
     * Gets the current threshold value.
     * 
     * @return The threshold value
     */
    public double getThreshold() {
        return threshold;
    }
    
    /**
     * Sets the k value for KNN_DISTANCE algorithm.
     * 
     * @param kValue The k value (default: 5)
     * @return This service instance for method chaining
     */
    public OutlierDetectionService setKValue(int kValue) {
        this.kValue = Math.max(1, kValue);
        return this;
    }
    
    /**
     * Gets the current k value.
     * 
     * @return The k value
     */
    public int getKValue() {
        return kValue;
    }
    
    /**
     * Sets whether to use parallel processing.
     * 
     * @param useParallel true to enable parallel processing, false for single-threaded
     * @return This service instance for method chaining
     */
    public OutlierDetectionService setUseParallel(boolean useParallel) {
        this.useParallel = useParallel;
        return this;
    }
    
    /**
     * Gets whether parallel processing is enabled.
     * 
     * @return true if parallel processing is enabled, false otherwise
     */
    public boolean isUseParallel() {
        return useParallel;
    }
    
    /**
     * Sets the number of threads to use for parallel processing.
     * 
     * @param numThreads Number of threads (min 1)
     * @return This service instance for method chaining
     */
    public OutlierDetectionService setNumThreads(int numThreads) {
        this.numThreads = Math.max(1, numThreads);
        return this;
    }
    
    /**
     * Gets the number of threads used for parallel processing.
     * 
     * @return Number of threads
     */
    public int getNumThreads() {
        return numThreads;
    }
} 