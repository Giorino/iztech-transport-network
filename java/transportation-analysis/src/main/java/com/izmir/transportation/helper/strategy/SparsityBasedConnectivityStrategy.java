package com.izmir.transportation.helper.strategy;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
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
 * A connectivity strategy that controls the sparsity of the graph by limiting the number
 * of connections per node based on a sparsity percentage.
 * 
 * For example:
 * - 1% sparsity means each node connects to ~99% of other nodes (dense)
 * - 99% sparsity means each node connects to ~1% of other nodes (sparse)
 *
 * @author yagizugurveren
 */
public class SparsityBasedConnectivityStrategy implements GraphConnectivityStrategy {
    private final double sparsityPercentage;
    private final int progressUpdateInterval = 1000;
    private static final int MAX_ATTEMPTS_MULTIPLIER = 2;
    private final Map<Point, DijkstraShortestPath<Point, DefaultWeightedEdge>> dijkstraCache;

    /**
     * Creates a new SparsityBasedConnectivityStrategy.
     *
     * @param sparsityPercentage The desired sparsity level (1-99)
     *                          1 = most dense (connect to 99% of nodes)
     *                          99 = most sparse (connect to 1% of nodes)
     * @throws IllegalArgumentException if sparsityPercentage is not between 1 and 99
     */
    public SparsityBasedConnectivityStrategy(double sparsityPercentage) {
        if (sparsityPercentage < 1 || sparsityPercentage > 99) {
            throw new IllegalArgumentException("Sparsity percentage must be between 1 and 99");
        }
        this.sparsityPercentage = sparsityPercentage;
        this.dijkstraCache = new ConcurrentHashMap<>();
    }

    @Override
    public List<List<Point>> createConnections(
            List<Point> points,
            Map<Point, Point> pointToNode,
            Graph<Point, DefaultWeightedEdge> network,
            TransportationGraph transportationGraph) {
        
        int numThreads = getRecommendedThreadCount();
        
        final CopyOnWriteArrayList<List<Point>> paths = new CopyOnWriteArrayList<>();
        final AtomicLong completedConnections = new AtomicLong(0);
        final AtomicLong lastPercentage = new AtomicLong(0);

        final int totalNodes = points.size();
        final int connectionsPerNode = Math.max(1, 
            (int) Math.ceil((totalNodes - 1) * (1 - sparsityPercentage / 100.0)));

        final long totalConnections = calculateTotalConnections(points);

        ExecutorService executor = Executors.newFixedThreadPool(numThreads);

        for (int i = 0; i < points.size(); i++) {
            final int startIndex = i;
            executor.submit(() -> {
                try {
                    Point p1 = points.get(startIndex);
                    Point node1 = pointToNode.get(p1);
                    
                    if (node1 == null || !network.containsVertex(node1)) {
                        System.err.println("Error processing node " + startIndex + ": Invalid node");
                        return;
                    }

                    DijkstraShortestPath<Point, DefaultWeightedEdge> dijkstra = 
                        dijkstraCache.computeIfAbsent(node1, 
                            k -> new DijkstraShortestPath<>(network));

                    List<PointDistance> distances = new ArrayList<>();
                    for (int j = 0; j < points.size(); j++) {
                        if (j != startIndex) {
                            Point p2 = points.get(j);
                            Point node2 = pointToNode.get(p2);
                            
                            if (node2 != null && network.containsVertex(node2)) {
                                double distance = p1.distance(p2);
                                distances.add(new PointDistance(j, distance));
                            }
                        }
                    }

                    if (distances.isEmpty()) {
                        System.err.println("Warning: No valid target nodes found for node " + startIndex);
                        return;
                    }

                    distances.sort(Comparator.comparingDouble(PointDistance::getDistance));
                    
                    int connectedPaths = 0;
                    int attemptIndex = 0;
                    int maxAttempts = Math.min(connectionsPerNode * MAX_ATTEMPTS_MULTIPLIER, distances.size());
                    
                    while (connectedPaths < connectionsPerNode && attemptIndex < maxAttempts) {
                        Point p2 = points.get(distances.get(attemptIndex).getIndex());
                        Point node2 = pointToNode.get(p2);

                        if (node2 != null && network.containsVertex(node2)) {
                            try {
                                GraphPath<Point, DefaultWeightedEdge> path = dijkstra.getPath(node1, node2);
                                if (path != null && !path.getVertexList().isEmpty()) {
                                    paths.add(new ArrayList<>(path.getVertexList()));
                                    transportationGraph.addConnection(p1, p2, path.getWeight());
                                    connectedPaths++;

                                    long completed = completedConnections.incrementAndGet();
                                    int currentPercentage = (int) ((completed * 100) / totalConnections);
                                    long lastPct = lastPercentage.get();
                                    if (currentPercentage > lastPct && 
                                        lastPercentage.compareAndSet(lastPct, currentPercentage)) {
                                        updateProgress(completed, totalConnections);
                                    }
                                }
                            } catch (Exception e) {
                                System.err.println("Error finding path from node " + startIndex + 
                                    " to node " + distances.get(attemptIndex).getIndex() + 
                                    ": " + e.getMessage());
                            }
                        }
                        attemptIndex++;
                    }

                    if (connectedPaths == 0) {
                        System.err.println("Warning: Node " + startIndex + " could not establish any connections");
                    }
                } catch (Exception e) {
                    System.err.println("Error processing node " + startIndex + ": " + e.getMessage());
                    e.printStackTrace();
                }
            });
        }

        executor.shutdown();
        try {
            executor.awaitTermination(24, TimeUnit.HOURS);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            System.err.println("Path calculation was interrupted: " + e.getMessage());
        }

        return new ArrayList<>(paths);
    }

    @Override
    public long calculateTotalConnections(List<Point> points) {
        int totalNodes = points.size();
        int connectionsPerNode = (int) Math.ceil((totalNodes - 1) * (1 - sparsityPercentage / 100.0));
        connectionsPerNode = Math.max(1, connectionsPerNode);
        return (long) totalNodes * connectionsPerNode;
    }
} 