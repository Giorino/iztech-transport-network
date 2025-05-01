package com.izmir.transportation.helper.outlier;

import java.util.Set;

import com.izmir.transportation.TransportationGraph;
import com.izmir.transportation.helper.Node;

/**
 * Interface for outlier detection algorithms in transportation networks.
 * Outlier detection algorithms identify nodes that deviate significantly
 * from other nodes in the network based on various metrics.
 */
public interface OutlierDetectionAlgorithm {
    
    /**
     * Detects outliers in the transportation graph.
     * 
     * @param graph The transportation graph
     * @param threshold The threshold value for determining outliers
     *        (typically in standard deviations from the mean)
     * @return Set of nodes identified as outliers
     */
    Set<Node> detectOutliers(TransportationGraph graph, double threshold);
    
    /**
     * Gets the name of the algorithm.
     * 
     * @return The algorithm name
     */
    String getName();
} 