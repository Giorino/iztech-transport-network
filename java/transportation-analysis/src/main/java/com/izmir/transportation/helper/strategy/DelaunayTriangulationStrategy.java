package com.izmir.transportation.helper.strategy;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicLong;

import org.jgrapht.Graph;
import org.jgrapht.GraphPath;
import org.jgrapht.alg.shortestpath.DijkstraShortestPath;
import org.jgrapht.graph.DefaultWeightedEdge;
import org.locationtech.jts.geom.Coordinate;
import org.locationtech.jts.geom.GeometryFactory;
import org.locationtech.jts.geom.Point;
import org.locationtech.jts.triangulate.DelaunayTriangulationBuilder;
import org.locationtech.jts.triangulate.quadedge.QuadEdge;
import org.locationtech.jts.triangulate.quadedge.QuadEdgeSubdivision;

import com.izmir.transportation.TransportationGraph;

/**
 * Implementation of GraphConnectivityStrategy that creates a network based on Delaunay Triangulation.
 * 
 * In a Delaunay triangulation, an edge connects two points if and only if there exists
 * a circle passing through those points with no other points inside it.
 * This creates a triangulation that maximizes the minimum angle of all triangles.
 */
public class DelaunayTriangulationStrategy implements GraphConnectivityStrategy {
    private final GeometryFactory geometryFactory;
    
    /**
     * Creates a new DelaunayTriangulationStrategy with default parameters.
     */
    public DelaunayTriangulationStrategy() {
        this.geometryFactory = new GeometryFactory();
    }

    @Override
    public long calculateTotalConnections(List<Point> points) {
        // For Delaunay triangulation, the number of edges is approximately 3n - 6
        // where n is the number of points (for n >= 3)
        long n = points.size();
        if (n < 3) return n <= 1 ? 0 : 1;
        return 3 * n - 6;
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
        
        if (points.size() < 3) {
            // Handle special cases with less than 3 points
            return handleSpecialCases(points, pointToNode, network, transportationGraph);
        }
        
        // Perform Delaunay Triangulation using JTS
        Set<EdgePair> delaunayEdges = computeDelaunayEdges(points);
        System.out.println("Computed " + delaunayEdges.size() + " Delaunay edges");
        
        CopyOnWriteArrayList<List<Point>> paths = new CopyOnWriteArrayList<>();
        long totalConnections = delaunayEdges.size();
        AtomicLong completedConnections = new AtomicLong(0);
        AtomicLong lastPercentage = new AtomicLong(0);

        ExecutorService executor = Executors.newFixedThreadPool(numThreads);
        
        // Process each Delaunay edge in parallel
        for (EdgePair edge : delaunayEdges) {
            executor.submit(() -> {
                DijkstraShortestPath<Point, DefaultWeightedEdge> dijkstra = new DijkstraShortestPath<>(network);
                Point p1 = points.get(edge.index1);
                Point p2 = points.get(edge.index2);
                Point node1 = pointToNode.get(p1);
                Point node2 = pointToNode.get(p2);
                
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
     * Handles special cases with less than 3 points.
     * 
     * @param points The list of points
     * @param pointToNode Mapping from points to network nodes
     * @param network The road network
     * @param transportationGraph The transportation graph
     * @return A list of paths
     */
    private List<List<Point>> handleSpecialCases(
            List<Point> points,
            Map<Point, Point> pointToNode,
            Graph<Point, DefaultWeightedEdge> network,
            TransportationGraph transportationGraph) {
        
        List<List<Point>> paths = new ArrayList<>();
        
        if (points.size() <= 1) {
            return paths; // No connections possible with 0 or 1 point
        }
        
        // With exactly 2 points, just connect them
        if (points.size() == 2) {
            DijkstraShortestPath<Point, DefaultWeightedEdge> dijkstra = new DijkstraShortestPath<>(network);
            Point p1 = points.get(0);
            Point p2 = points.get(1);
            Point node1 = pointToNode.get(p1);
            Point node2 = pointToNode.get(p2);
            
            GraphPath<Point, DefaultWeightedEdge> path = dijkstra.getPath(node1, node2);
            if (path != null) {
                List<Point> pathPoints = new ArrayList<>(path.getVertexList());
                paths.add(pathPoints);
                
                double pathDistance = path.getWeight();
                transportationGraph.addConnection(p1, p2, pathDistance);
            }
        }
        
        return paths;
    }
    
    /**
     * Computes the Delaunay triangulation edges for the given points.
     * 
     * @param points The list of points to triangulate
     * @return A set of edge pairs representing the Delaunay triangulation
     */
    private Set<EdgePair> computeDelaunayEdges(List<Point> points) {
        // Prepare input for JTS Delaunay triangulation
        List<Coordinate> coordinates = new ArrayList<>();
        for (Point p : points) {
            coordinates.add(new Coordinate(p.getX(), p.getY()));
        }
        
        // Build the Delaunay triangulation
        DelaunayTriangulationBuilder builder = new DelaunayTriangulationBuilder();
        builder.setSites(coordinates);
        QuadEdgeSubdivision subdivision = builder.getSubdivision();
        
        // Extract edges from the triangulation
        Set<EdgePair> edges = new HashSet<>();
        for (Object obj : subdivision.getEdges()) {
            QuadEdge quadEdge = (QuadEdge) obj;
            if (quadEdge.isLive()) {
                Coordinate orig = quadEdge.orig().getCoordinate();
                Coordinate dest = quadEdge.dest().getCoordinate();
                
                // Find indices of points corresponding to these coordinates
                int index1 = findPointIndex(points, orig);
                int index2 = findPointIndex(points, dest);
                
                // Only add valid edges between input points
                if (index1 >= 0 && index2 >= 0 && index1 != index2) {
                    edges.add(new EdgePair(index1, index2));
                }
            }
        }
        
        return edges;
    }
    
    /**
     * Finds the index of a point in the list that matches the given coordinate.
     * 
     * @param points The list of points
     * @param coordinate The coordinate to find
     * @return The index of the matching point, or -1 if not found
     */
    private int findPointIndex(List<Point> points, Coordinate coordinate) {
        for (int i = 0; i < points.size(); i++) {
            Point p = points.get(i);
            if (Math.abs(p.getX() - coordinate.x) < 1e-10 && 
                Math.abs(p.getY() - coordinate.y) < 1e-10) {
                return i;
            }
        }
        return -1;
    }
    
    /**
     * Simple class to represent an edge between two points by their indices.
     */
    private static class EdgePair {
        private final int index1;
        private final int index2;
        
        public EdgePair(int index1, int index2) {
            // Ensure index1 <= index2 for consistent comparison
            if (index1 <= index2) {
                this.index1 = index1;
                this.index2 = index2;
            } else {
                this.index1 = index2;
                this.index2 = index1;
            }
        }
        
        @Override
        public boolean equals(Object obj) {
            if (this == obj) return true;
            if (obj == null || getClass() != obj.getClass()) return false;
            EdgePair other = (EdgePair) obj;
            return index1 == other.index1 && index2 == other.index2;
        }
        
        @Override
        public int hashCode() {
            return 31 * index1 + index2;
        }
    }
} 