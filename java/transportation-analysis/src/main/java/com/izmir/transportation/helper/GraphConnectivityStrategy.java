package com.izmir.transportation.helper;

import java.util.List;
import java.util.Map;

import org.jgrapht.Graph;
import org.jgrapht.graph.DefaultWeightedEdge;
import org.locationtech.jts.geom.Point;

import com.izmir.transportation.TransportationGraph;

/**
 * Interface defining different strategies for connecting nodes in the transportation network.
 * This allows for flexible implementation of various connectivity patterns.
 */
public interface GraphConnectivityStrategy {
    /**
     * Creates connections between nodes according to the specific strategy.
     *
     * @param points Original input points
     * @param pointToNode Mapping of input points to network nodes
     * @param network The road network graph
     * @param transportationGraph The transportation graph to add connections to
     * @return List of paths connecting the points
     */
    List<List<Point>> createConnections(
        List<Point> points,
        Map<Point, Point> pointToNode,
        Graph<Point, DefaultWeightedEdge> network,
        TransportationGraph transportationGraph
    );

    /**
     * Creates connections between nodes in parallel for better performance.
     * This is the preferred method for large networks.
     *
     * @param points Original input points
     * @param pointToNode Mapping of input points to network nodes
     * @param network The road network graph
     * @param transportationGraph The transportation graph to add connections to
     * @param numThreads Number of threads to use for parallel processing
     * @return List of paths connecting the points
     */
    default List<List<Point>> createConnectionsParallel(
        List<Point> points,
        Map<Point, Point> pointToNode,
        Graph<Point, DefaultWeightedEdge> network,
        TransportationGraph transportationGraph,
        int numThreads
    ) {
        return createConnections(points, pointToNode, network, transportationGraph);
    }

    /**
     * Calculates the total number of connections that will be processed.
     * This is used for progress tracking.
     *
     * @param points List of points to be connected
     * @return The total number of connections that will be processed
     */
    default long calculateTotalConnections(List<Point> points) {
        return 0; // Default implementation returns 0, should be overridden by concrete implementations
    }

    /**
     * Updates and displays the progress of connection creation.
     *
     * @param completedConnections Number of connections processed so far
     * @param totalConnections Total number of connections to process
     */
    default void updateProgress(long completedConnections, long totalConnections) {
        int currentPercentage = (int) ((completedConnections * 100) / totalConnections);
        System.out.printf("Progress: %d%% (%d/%d connections processed)%n", 
            currentPercentage, completedConnections, totalConnections);
    }

    /**
     * Gets the recommended number of threads for parallel processing.
     * By default, it uses the number of available processors minus 1,
     * but ensures at least 1 thread is used.
     *
     * @return The recommended number of threads
     */
    default int getRecommendedThreadCount() {
        return Math.max(1, Runtime.getRuntime().availableProcessors() - 1);
    }
} 