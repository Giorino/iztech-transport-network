package com.izmir.transportation;

import java.io.IOException;
import java.util.List;
import java.util.Map;

import org.jgrapht.Graph;
import org.jgrapht.graph.DefaultWeightedEdge;
import org.locationtech.jts.geom.Envelope;
import org.locationtech.jts.geom.Point;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.izmir.transportation.helper.strategy.CompleteGraphStrategy;
import com.izmir.transportation.helper.strategy.DelaunayTriangulationStrategy;
import com.izmir.transportation.helper.strategy.GabrielGraphStrategy;
import com.izmir.transportation.helper.strategy.GraphConnectivityStrategy;
import com.izmir.transportation.helper.strategy.KNearestNeighborsStrategy;
import com.izmir.transportation.persistence.GraphPersistenceService;

/**
 * Service for constructing transportation graphs using different strategies.
 * This service handles the loading of points, construction of the road network,
 * and application of different connectivity strategies.
 */
public class GraphConstructionService {
    private static final Logger LOGGER = LoggerFactory.getLogger(GraphConstructionService.class);
    private final GraphPersistenceService persistenceService;
    
    /**
     * Available graph construction strategies
     */
    public enum GraphStrategy {
        COMPLETE("complete"),
        K_NEAREST_NEIGHBORS("k_nearest_neighbors"),
        GABRIEL("gabriel"),
        DELAUNAY("delaunay");
        
        private final String code;
        
        GraphStrategy(String code) {
            this.code = code;
        }
        
        public String getCode() {
            return code;
        }
        
        public static GraphStrategy fromCode(String code) {
            for (GraphStrategy strategy : values()) {
                if (strategy.getCode().equals(code)) {
                    return strategy;
                }
            }
            throw new IllegalArgumentException("Unknown strategy code: " + code);
        }
    }

    public GraphConstructionService() {
        this.persistenceService = new GraphPersistenceService();
    }

    /**
     * Create a transportation graph using the specified strategy
     *
     * @param points List of points to include in the graph
     * @param strategyType Graph construction strategy to use
     * @param kValue K value for K-nearest neighbors strategy (if applicable)
     * @param useParallel Whether to use parallel processing
     * @param visualize Whether to visualize the graph after construction
     * @param savePersistent Whether to save the graph for future use
     * @return The constructed transportation graph
     * @throws Exception If an error occurs during graph construction
     */
    public TransportationGraph createGraph(
            List<Point> points, 
            GraphStrategy strategyType,
            int kValue,
            boolean useParallel,
            boolean visualize,
            boolean savePersistent) throws Exception {
        
        LOGGER.info("Creating graph with {} nodes using {} strategy", points.size(), strategyType);
        
        // Check if the graph already exists in persistent storage
        if (persistenceService.graphExists(points.size(), strategyType.getCode())) {
            LOGGER.info("Graph with {} nodes and {} strategy already exists, loading from storage",
                    points.size(), strategyType);
            return persistenceService.loadGraph(points.size(), strategyType.getCode());
        }
        
        // Get the bounding box for the points
        Envelope bbox = getBoundingBox(points);
        
        // Load the road network
        Graph<Point, DefaultWeightedEdge> network = CreateRoadNetwork.loadRoadNetwork(bbox);
        
        // Snap the points to the road network
        Map<Point, Point> pointToNode = CreateRoadNetwork.snapPointsToNetwork(points, network);
        
        // Create the transportation graph
        TransportationGraph transportationGraph = new TransportationGraph(points);
        
        // Set the graph construction method
        transportationGraph.setGraphConstructionMethod(strategyType.getCode());
        
        // Apply the selected strategy to create connections
        GraphConnectivityStrategy strategy = createStrategy(strategyType, kValue);
        
        // Create the connections
        List<List<Point>> paths;
        if (useParallel) {
            int numThreads = strategy.getRecommendedThreadCount();
            LOGGER.info("Using parallel processing with {} threads", numThreads);
            paths = strategy.createConnectionsParallel(points, pointToNode, network, transportationGraph, numThreads);
        } else {
            paths = strategy.createConnections(points, pointToNode, network, transportationGraph);
        }
        
        // Visualize the network if requested
        if (visualize) {
            transportationGraph.visualizeGraph();
        }
        
        // Save the graph if requested
        if (savePersistent) {
            try {
                String filePath = persistenceService.saveGraph(
                    transportationGraph, points.size(), strategyType.getCode());
                LOGGER.info("Graph saved to {}", filePath);
            } catch (IOException e) {
                LOGGER.error("Failed to save graph", e);
            }
        }
        
        return transportationGraph;
    }
    
    /**
     * Create an appropriate strategy based on the strategy type
     *
     * @param strategyType The type of strategy to create
     * @param kValue K value for K-nearest neighbors strategy (if applicable)
     * @return The created strategy
     */
    private GraphConnectivityStrategy createStrategy(GraphStrategy strategyType, int kValue) {
        switch (strategyType) {
            case COMPLETE:
                return new CompleteGraphStrategy();
            case K_NEAREST_NEIGHBORS:
                return new KNearestNeighborsStrategy(kValue, 100); // Use 100 as maxAttempts
            case GABRIEL:
                return new GabrielGraphStrategy();
            case DELAUNAY:
                return new DelaunayTriangulationStrategy();
            default:
                throw new IllegalArgumentException("Unknown strategy type: " + strategyType);
        }
    }
    
    /**
     * Get the bounding box for a list of points
     *
     * @param points The points to get the bounding box for
     * @return The bounding box
     */
    private Envelope getBoundingBox(List<Point> points) {
        if (points.isEmpty()) {
            throw new IllegalArgumentException("Points list cannot be empty");
        }
        
        Envelope bbox = new Envelope();
        for (Point point : points) {
            bbox.expandToInclude(point.getCoordinate());
        }
        
        // Expand the bounding box slightly to ensure all points are included
        bbox.expandBy(0.01);
        
        return bbox;
    }
} 