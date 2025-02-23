package com.izmir.transportation.helper.strategy;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicLong;

import org.jgrapht.Graph;
import org.jgrapht.GraphPath;
import org.jgrapht.alg.shortestpath.DijkstraShortestPath;
import org.jgrapht.graph.DefaultWeightedEdge;
import org.locationtech.jts.geom.Point;

import com.izmir.transportation.PointDistance;
import com.izmir.transportation.TransportationGraph;

/**
 * Implementation of GraphConnectivityStrategy that connects each node
 * to its k nearest neighbors.
 */
public class KNearestNeighborsStrategy implements GraphConnectivityStrategy {
    private final int k;
    private final int maxAttempts;

    /**
     * Creates a new KNearestNeighborsStrategy.
     *
     * @param k Number of nearest neighbors to connect to
     * @param maxAttempts Maximum number of attempts to find valid neighbors
     */
    public KNearestNeighborsStrategy(int k, int maxAttempts) {
        this.k = k;
        this.maxAttempts = maxAttempts;
    }
    
    @Override
    public long calculateTotalConnections(List<Point> points) {
        return (long) points.size() * k;
    }
    
    @Override
    public List<List<Point>> createConnectionsParallel(
            List<Point> points,
            Map<Point, Point> pointToNode,
            Graph<Point, DefaultWeightedEdge> network,
            TransportationGraph transportationGraph,
            int numThreads) {
        
        CopyOnWriteArrayList<List<Point>> paths = new CopyOnWriteArrayList<>();
        long totalConnections = calculateTotalConnections(points);
        AtomicLong completedConnections = new AtomicLong(0);
        AtomicLong lastPercentage = new AtomicLong(0);

        ExecutorService executor = Executors.newFixedThreadPool(numThreads);
        
        // multi-threaded
        for (int i = 0; i < points.size(); i++) {
            final int pointIndex = i;
            executor.submit(() -> {
                DijkstraShortestPath<Point, DefaultWeightedEdge> dijkstra = new DijkstraShortestPath<>(network);
                Point p1 = points.get(pointIndex);
                Point node1 = pointToNode.get(p1);

                // Calculate distances to all other points
                List<PointDistance> distances = new ArrayList<>();
                for (int j = 0; j < points.size(); j++) {
                    if (pointIndex != j) {
                        Point p2 = points.get(j);
                        double distance = p1.distance(p2);
                        distances.add(new PointDistance(j, distance));
                    }
                }

                distances.sort(Comparator.comparingDouble(PointDistance::getDistance));
                int connectedPaths = 0;
                int attemptIndex = 0;
                
                while (connectedPaths < k && attemptIndex < Math.min(maxAttempts * k, distances.size())) {
                    Point p2 = points.get(distances.get(attemptIndex).getIndex());
                    Point node2 = pointToNode.get(p2);

                    GraphPath<Point, DefaultWeightedEdge> path = dijkstra.getPath(node1, node2);
                    if (path != null) {
                        List<Point> pathPoints = new ArrayList<>(path.getVertexList());
                        paths.add(pathPoints);
                        
                        double pathDistance = path.getWeight();
                        transportationGraph.addConnection(p1, p2, pathDistance);
                        connectedPaths++;
                        
                        // Update progress
                        long completed = completedConnections.incrementAndGet();
                        int currentPercentage = (int) ((completed * 100) / totalConnections);
                        long lastPct = lastPercentage.get();
                        if (currentPercentage > lastPct && 
                            lastPercentage.compareAndSet(lastPct, currentPercentage)) {
                            updateProgress(completed, totalConnections);
                        }
                    }
                    attemptIndex++;
                }
            });
        }

        executor.shutdown();
        try {
            executor.awaitTermination(24, TimeUnit.HOURS);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            System.err.println("Parallel processing was interrupted: " + e.getMessage());
        }

        return new ArrayList<>(paths);
    }
    
    @Override
    public List<List<Point>> createConnections(
            List<Point> points,
            Map<Point, Point> pointToNode,
            Graph<Point, DefaultWeightedEdge> network,
            TransportationGraph transportationGraph) {
        
        return createConnectionsParallel(points, pointToNode, network, transportationGraph,
                                       getRecommendedThreadCount());
    }
} 