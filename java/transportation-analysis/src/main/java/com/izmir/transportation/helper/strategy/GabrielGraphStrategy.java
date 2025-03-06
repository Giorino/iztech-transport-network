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
import org.locationtech.jts.geom.GeometryFactory;
import org.locationtech.jts.geom.Point;

import com.izmir.transportation.TransportationGraph;

/**
 * Implementation of GraphConnectivityStrategy that creates a Gabriel Graph.
 * 
 * In a Gabriel Graph, two points P and Q are connected by an edge if and only if
 * the circle with diameter PQ (the diametric circle) contains no other points from the set.
 */
public class GabrielGraphStrategy implements GraphConnectivityStrategy {
    private final GeometryFactory geometryFactory;
    
    /**
     * Creates a new GabrielGraphStrategy with default parameters.
     */
    public GabrielGraphStrategy() {
        this.geometryFactory = new GeometryFactory();
    }

    @Override
    public long calculateTotalConnections(List<Point> points) {
        // In the worst case, we could have n(n-1)/2 connections (complete graph)
        long n = points.size();
        return (n * (n - 1)) / 2;
    }

    @Override
    public List<List<Point>> createConnections(
            List<Point> points,
            Map<Point, Point> pointToNode,
            Graph<Point, DefaultWeightedEdge> network,
            TransportationGraph transportationGraph) {
        
        return createConnectionsParallel(
            points, 
            pointToNode, 
            network, 
            transportationGraph, 
            getRecommendedThreadCount()
        );
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
        
        // Process each possible pair of points in parallel
        for (int i = 0; i < points.size(); i++) {
            final int pointIndex = i;
            executor.submit(() -> {
                DijkstraShortestPath<Point, DefaultWeightedEdge> dijkstra = new DijkstraShortestPath<>(network);
                Point p1 = points.get(pointIndex);
                Point node1 = pointToNode.get(p1);
                
                // Check connection with all points with higher index to avoid duplication
                for (int j = pointIndex + 1; j < points.size(); j++) {
                    Point p2 = points.get(j);
                    Point node2 = pointToNode.get(p2);
                    
                    // Only connect if points satisfy Gabriel Graph criteria
                    if (isGabrielEdge(p1, p2, points)) {
                        GraphPath<Point, DefaultWeightedEdge> path = dijkstra.getPath(node1, node2);
                        if (path != null) {
                            List<Point> pathPoints = new ArrayList<>(path.getVertexList());
                            paths.add(pathPoints);
                            
                            double pathDistance = path.getWeight();
                            transportationGraph.addConnection(p1, p2, pathDistance);
                            
                            // Update progress
                            long completed = completedConnections.incrementAndGet();
                            int currentPercentage = (int) ((completed * 100) / totalConnections);
                            long lastPct = lastPercentage.get();
                            if (currentPercentage > lastPct && 
                                lastPercentage.compareAndSet(lastPct, currentPercentage)) {
                                updateProgress(completed, totalConnections);
                            }
                        }
                    }
                    
                    // Count as completed even if no edge is added
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
    
    /**
     * Determines if an edge between two points should exist in a Gabriel Graph.
     * Two points P and Q are connected if and only if the circle with diameter PQ
     * contains no other points from the set.
     * 
     * @param p1 First point
     * @param p2 Second point
     * @param allPoints All points in the set
     * @return true if the edge should exist in the Gabriel Graph
     */
    private boolean isGabrielEdge(Point p1, Point p2, List<Point> allPoints) {
        // Calculate center of the diametric circle (midpoint of p1 and p2)
        double centerX = (p1.getX() + p2.getX()) / 2.0;
        double centerY = (p1.getY() + p2.getY()) / 2.0;
        
        // Calculate radius (half the distance between p1 and p2)
        double radius = p1.distance(p2) / 2.0;
        double radiusSquared = radius * radius;
        
        // Check if any other point is inside the circle
        for (Point p : allPoints) {
            // Skip the two points we're checking
            if (p.equals(p1) || p.equals(p2)) {
                continue;
            }
            
            // Calculate distance squared from center to point
            double distanceSquared = Math.pow(p.getX() - centerX, 2) + Math.pow(p.getY() - centerY, 2);
            
            // If any point is inside the circle, p1 and p2 should not be connected
            if (distanceSquared < radiusSquared) {
                return false;
            }
        }
        
        // No points inside the circle, so p1 and p2 should be connected
        return true;
    }
} 