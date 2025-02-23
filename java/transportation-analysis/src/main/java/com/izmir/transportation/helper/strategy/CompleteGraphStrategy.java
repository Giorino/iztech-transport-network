package com.izmir.transportation.helper.strategy;

import java.util.ArrayList;
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

import com.izmir.transportation.TransportationGraph;

/**
 * Implementation of GraphConnectivityStrategy that creates a complete graph
 * where every node is connected to every other node.
 */
public class CompleteGraphStrategy implements GraphConnectivityStrategy {
    
    @Override
    public long calculateTotalConnections(List<Point> points) {
        int totalPoints = points.size();
        return ((long) totalPoints * (totalPoints - 1)) / 2;
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
            final int startIndex = i;
            executor.submit(() -> {
                DijkstraShortestPath<Point, DefaultWeightedEdge> dijkstra = new DijkstraShortestPath<>(network);
                Point p1 = points.get(startIndex);
                Point node1 = pointToNode.get(p1);

                // Connect to all points after startIndex to avoid duplicate connections
                for (int j = startIndex + 1; j < points.size(); j++) {
                    Point p2 = points.get(j);
                    Point node2 = pointToNode.get(p2);

                    GraphPath<Point, DefaultWeightedEdge> path = dijkstra.getPath(node1, node2);
                    if (path != null) {
                        List<Point> pathPoints = new ArrayList<>(path.getVertexList());
                        paths.add(pathPoints);
                        
                        double pathDistance = path.getWeight();
                        transportationGraph.addConnection(p1, p2, pathDistance);
                    }
                    
                    // Update progress
                    long completed = completedConnections.incrementAndGet();
                    int currentPercentage = (int) ((completed * 100) / totalConnections);
                    long lastPct = lastPercentage.get();
                    if (currentPercentage > lastPct && 
                        lastPercentage.compareAndSet(lastPct, currentPercentage)) {
                        updateProgress(completed, totalConnections);
                    }
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