package com.izmir.transportation.helper.strategy;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import java.util.Map;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicLong;

import org.jgrapht.Graph;
import org.jgrapht.GraphPath;
import org.jgrapht.alg.shortestpath.DijkstraShortestPath;
import org.jgrapht.graph.DefaultWeightedEdge;
import org.locationtech.jts.geom.Point;

import com.izmir.transportation.TransportationGraph;

/**
 * Implementation of GraphConnectivityStrategy that creates a greedy multiplicative spanner
 * based on Althöfer's algorithm.
 * 
 * This algorithm sorts all potential edges by weight (distance) and keeps an edge if no
 * t-stretch path already exists between its endpoints in the current spanner.
 * 
 * For a stretch factor t > 1, the algorithm guarantees that the shortest path between any
 * two vertices in the spanner is at most t times the shortest path in the original graph.
 */
public class GreedySpannerStrategy implements GraphConnectivityStrategy {
    
    private final double stretchFactor;
    private boolean useCompleteGraph = false;
    
    /**
     * Creates a new GreedySpannerStrategy with the specified stretch factor.
     * 
     * @param stretchFactor The stretch factor (t) for the spanner. Must be greater than 1.
     * A common choice is 3 (corresponding to k=1 in the (2k-1)-spanner).
     */
    public GreedySpannerStrategy(double stretchFactor) {
        if (stretchFactor <= 1) {
            throw new IllegalArgumentException("Stretch factor must be greater than 1");
        }
        this.stretchFactor = stretchFactor;
    }
    
    /**
     * Creates a new GreedySpannerStrategy with the specified stretch factor and option to use complete graph.
     * 
     * @param stretchFactor The stretch factor (t) for the spanner. Must be greater than 1.
     * @param useCompleteGraph Whether to use a complete graph optimization.
     */
    public GreedySpannerStrategy(double stretchFactor, boolean useCompleteGraph) {
        this(stretchFactor);
        this.useCompleteGraph = useCompleteGraph;
    }
    
    /**
     * Creates a new GreedySpannerStrategy with default stretch factor of 3.
     */
    public GreedySpannerStrategy() {
        this(3.0); // Default stretch factor of 3 (k=1)
    }
    
    @Override
    public long calculateTotalConnections(List<Point> points) {
        int totalPoints = points.size();
        return ((long) totalPoints * (totalPoints - 1)) / 2; // Maximum possible connections
    }
    
    /**
     * Creates a spanner using a pre-computed complete graph. This is much faster than
     * calculating all pairwise shortest paths from scratch.
     * 
     * @param points The points to connect
     * @param completeGraph A pre-computed complete graph containing all points
     * @param transportationGraph The transportation graph to build
     * @param numThreads Number of threads to use for parallel processing
     * @return A list of paths in the spanner
     */
    public List<List<Point>> createSpannerFromCompleteGraph(
            List<Point> points,
            TransportationGraph completeGraph,
            TransportationGraph transportationGraph,
            int numThreads) {
        
        System.out.println("Creating greedy spanner from complete graph with " + numThreads + 
                           " threads and stretch factor " + stretchFactor);
        long startTime = System.currentTimeMillis();
        
        // Create a list of all edges from the complete graph
        List<PointPair> allEdges = new ArrayList<>();
        
        // 1. Extract all edges from the complete graph
        System.out.println("Step 1: Extracting edges from complete graph");
        long edgeExtractionStart = System.currentTimeMillis();
        
        // Get all edges from the complete graph
        for (int i = 0; i < points.size(); i++) {
            Point p1 = points.get(i);
            
            for (int j = i + 1; j < points.size(); j++) {
                Point p2 = points.get(j);
                
                // Get the weight from the complete graph - ensure it's not infinity
                double weight = completeGraph.getEdgeWeight(p1, p2);
                if (weight != Double.POSITIVE_INFINITY) {
                    // The path is just the two endpoints for a direct edge
                    List<Point> path = new ArrayList<>();
                    path.add(p1);
                    path.add(p2);
                    
                    allEdges.add(new PointPair(p1, p2, p1, p2, weight, path));
                }
            }
        }
        
        long edgeExtractionEnd = System.currentTimeMillis();
        System.out.println("Edge extraction completed in " + 
                          (edgeExtractionEnd - edgeExtractionStart) / 1000.0 + 
                          " seconds. Found " + allEdges.size() + " edges.");
        
        // If we have no edges, this means the complete graph wasn't properly built
        if (allEdges.isEmpty()) {
            System.err.println("ERROR: No edges extracted from complete graph. Falling back to standard algorithm.");
            return Collections.emptyList();
        }
        
        // 2. Sort all edges by weight (non-decreasing order)
        System.out.println("Step 2: Sorting " + allEdges.size() + " edges");
        long sortStartTime = System.currentTimeMillis();
        Collections.sort(allEdges, Comparator.comparingDouble(PointPair::getWeight));
        long sortEndTime = System.currentTimeMillis();
        System.out.println("Sorting completed in " + (sortEndTime - sortStartTime) / 1000.0 + " seconds");
        
        // 3. Initialize concurrent data structures
        CopyOnWriteArrayList<List<Point>> paths = new CopyOnWriteArrayList<>();
        AtomicLong completedConnections = new AtomicLong(0);
        AtomicLong lastPercentage = new AtomicLong(0);
        
        // Create a spanner graph to check for existing paths
        TransportationGraph spannerGraph = new TransportationGraph(points);
        
        // Use a thread-safe set to track processed edges
        Map<Integer, Boolean> processedEdges = new ConcurrentHashMap<>();
        
        // 4. Process edges in parallel using a thread pool
        System.out.println("Step 4: Processing edges to build spanner with " + numThreads + " threads");
        System.out.println("Total edges to process: " + allEdges.size());
        
        // Create a blocking queue to hold tasks for workers
        BlockingQueue<PointPair> edgeQueue = new LinkedBlockingQueue<>(allEdges);
        AtomicBoolean done = new AtomicBoolean(false);
        
        // Thread-safe counter for edges added to the spanner
        AtomicLong edgesAddedCounter = new AtomicLong(0);
        
        // Worker threads to process edges
        ExecutorService executor = Executors.newFixedThreadPool(numThreads);
        for (int i = 0; i < numThreads; i++) {
            executor.submit(() -> {
                while (!done.get()) {
                    PointPair edge = null;
                    try {
                        edge = edgeQueue.poll(100, TimeUnit.MILLISECONDS);
                    } catch (InterruptedException e) {
                        Thread.currentThread().interrupt();
                        return;
                    }
                    
                    if (edge == null) continue;
                    
                    // Get a unique identifier for this edge
                    int edgeId = edge.hashCode();
                    
                    // Skip if already processed by another thread
                    if (processedEdges.putIfAbsent(edgeId, Boolean.TRUE) != null) {
                        // Edge already processed
                        completedConnections.incrementAndGet();
                        continue;
                    }
                    
                    Point p1 = edge.getSource();
                    Point p2 = edge.getTarget();
                    double edgeWeight = edge.getWeight();
                    double maxAllowedWeight = edgeWeight * stretchFactor;
                    
                    // Check if there already exists a path in the current spanner
                    boolean pathExists;
                    synchronized (spannerGraph) {
                        pathExists = spannerGraph.hasPathWithinFactor(p1, p2, maxAllowedWeight);
                    }
                    
                    if (!pathExists) {
                        // Add to the path list (thread-safe by using CopyOnWriteArrayList)
                        paths.add(new ArrayList<>(edge.getPath()));
                        
                        // Add to both graphs - this needs to be synchronized
                        synchronized (spannerGraph) {
                            spannerGraph.addConnection(p1, p2, edgeWeight);
                        }
                        
                        synchronized (transportationGraph) {
                            // Ensure bidirectional edge with the same weight
                            transportationGraph.addConnection(p1, p2, edgeWeight);
                            // This is redundant as addConnection adds bidirectional edges internally,
                            // but making it explicit for clarity
                        }
                        
                        // Count edges added to spanner
                        edgesAddedCounter.incrementAndGet();
                    }
                    
                    // Update progress - atomic operations
                    long completed = completedConnections.incrementAndGet();
                    int currentPercentage = (int) ((completed * 100) / allEdges.size());
                    long lastPct = lastPercentage.get();
                    if (currentPercentage > lastPct && 
                        lastPercentage.compareAndSet(lastPct, currentPercentage)) {
                        long edgesAdded = edgesAddedCounter.get();
                        System.out.printf("Processing progress: %d%% (%d/%d edges processed) - %d edges in spanner (%.2f%%)%n", 
                            currentPercentage, completed, allEdges.size(), edgesAdded, 
                            (edgesAdded * 100.0 / Math.max(1, completed)));
                    }
                }
            });
        }
        
        // Wait until all edges are processed
        try {
            while (completedConnections.get() < allEdges.size()) {
                Thread.sleep(500);
            }
            
            // Signal worker threads to finish
            done.set(true);
            executor.shutdown();
            executor.awaitTermination(1, TimeUnit.MINUTES);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            System.err.println("Parallel processing was interrupted: " + e.getMessage());
        }
        
        long endTime = System.currentTimeMillis();
        System.out.println("Greedy spanner created with " + paths.size() + " edges out of " 
                          + allEdges.size() + " possible edges ("
                          + (paths.size() * 100 / Math.max(1, allEdges.size())) + "%)");
        System.out.println("Total time: " + (endTime - startTime) / 1000.0 + " seconds");
        
        return new ArrayList<>(paths);
    }
    
    @Override
    public List<List<Point>> createConnectionsParallel(
            List<Point> points,
            Map<Point, Point> pointToNode,
            Graph<Point, DefaultWeightedEdge> network,
            TransportationGraph transportationGraph,
            int numThreads) {
        
        // If useCompleteGraph is true and we already have a complete transportation graph,
        // use it to build the spanner more efficiently
        if (useCompleteGraph && transportationGraph.getEdgeCount() > 0 && 
            transportationGraph.isCompleteGraph()) {
            return createSpannerFromCompleteGraph(points, transportationGraph, 
                                              new TransportationGraph(points), numThreads);
        }
        
        System.out.println("Creating greedy spanner with " + numThreads + " threads and stretch factor " + stretchFactor);
        long startTime = System.currentTimeMillis();
        
        // Create a list of all potential edges (point pairs)
        List<PointPair> allEdges = new ArrayList<>();
        long totalConnections = calculateTotalConnections(points);
        AtomicLong completedConnections = new AtomicLong(0);
        AtomicLong lastPercentage = new AtomicLong(0);
        
        // 1. Create all possible point pairs and calculate their weights
        System.out.println("Step 1: Calculating all potential edges");
        long edgeCreationTotal = calculateTotalConnections(points);
        AtomicLong edgesCreated = new AtomicLong(0);
        AtomicLong lastCreationPercentage = new AtomicLong(0);
        
        for (int i = 0; i < points.size(); i++) {
            Point p1 = points.get(i);
            Point node1 = pointToNode.get(p1);
            
            for (int j = i + 1; j < points.size(); j++) {
                Point p2 = points.get(j);
                Point node2 = pointToNode.get(p2);
                
                // Find the shortest path between the nodes in the original network
                DijkstraShortestPath<Point, DefaultWeightedEdge> dijkstra = new DijkstraShortestPath<>(network);
                GraphPath<Point, DefaultWeightedEdge> path = dijkstra.getPath(node1, node2);
                
                if (path != null) {
                    double weight = path.getWeight();
                    allEdges.add(new PointPair(p1, p2, node1, node2, weight, path.getVertexList()));
                }
                
                // Update creation progress
                long created = edgesCreated.incrementAndGet();
                int currentPercentage = (int) ((created * 100) / edgeCreationTotal);
                long lastPct = lastCreationPercentage.get();
                if (currentPercentage > lastPct && 
                    lastCreationPercentage.compareAndSet(lastPct, currentPercentage)) {
                    System.out.printf("Edge creation progress: %d%% (%d/%d edges created)%n", 
                        currentPercentage, created, edgeCreationTotal);
                }
            }
        }
        
        // 2. Sort all edges by weight (non-decreasing order)
        System.out.println("Step 2: Sorting " + allEdges.size() + " edges");
        long sortStartTime = System.currentTimeMillis();
        Collections.sort(allEdges, Comparator.comparingDouble(PointPair::getWeight));
        long sortEndTime = System.currentTimeMillis();
        System.out.println("Sorting completed in " + (sortEndTime - sortStartTime) / 1000.0 + " seconds");
        
        // 3. Initialize concurrent data structures
        CopyOnWriteArrayList<List<Point>> paths = new CopyOnWriteArrayList<>();
        
        // Create a spanner graph to check for existing paths
        // This will contain only the original points (not the network nodes)
        TransportationGraph spannerGraph = new TransportationGraph(points);
        
        // Use a thread-safe set to track processed edges
        Map<Integer, Boolean> processedEdges = new ConcurrentHashMap<>();
        
        // 4. Process edges in parallel using a thread pool
        System.out.println("Step 4: Processing edges to build spanner with " + numThreads + " threads");
        System.out.println("Total edges to process: " + allEdges.size());
        
        // Create a blocking queue to hold tasks for workers
        BlockingQueue<PointPair> edgeQueue = new LinkedBlockingQueue<>(allEdges);
        AtomicBoolean done = new AtomicBoolean(false);
        
        // Thread-safe counter for edges added to the spanner
        AtomicLong edgesAddedCounter = new AtomicLong(0);
        
        // Worker threads to process edges
        ExecutorService executor = Executors.newFixedThreadPool(numThreads);
        for (int i = 0; i < numThreads; i++) {
            executor.submit(() -> {
                while (!done.get()) {
                    PointPair edge = null;
                    try {
                        edge = edgeQueue.poll(100, TimeUnit.MILLISECONDS);
                    } catch (InterruptedException e) {
                        Thread.currentThread().interrupt();
                        return;
                    }
                    
                    if (edge == null) continue;
                    
                    // Get a unique identifier for this edge
                    int edgeId = edge.hashCode();
                    
                    // Skip if already processed by another thread
                    if (processedEdges.putIfAbsent(edgeId, Boolean.TRUE) != null) {
                        // Edge already processed
                        completedConnections.incrementAndGet();
                        continue;
                    }
                    
                    Point p1 = edge.getSource();
                    Point p2 = edge.getTarget();
                    double edgeWeight = edge.getWeight();
                    double maxAllowedWeight = edgeWeight * stretchFactor;
                    
                    // Check if there already exists a path in the current spanner
                    boolean pathExists;
                    synchronized (spannerGraph) {
                        pathExists = spannerGraph.hasPathWithinFactor(p1, p2, maxAllowedWeight);
                    }
                    
                    if (!pathExists) {
                        // Add to the path list (thread-safe by using CopyOnWriteArrayList)
                        paths.add(new ArrayList<>(edge.getPath()));
                        
                        // Add to both graphs - this needs to be synchronized
                        synchronized (spannerGraph) {
                            spannerGraph.addConnection(p1, p2, edgeWeight);
                        }
                        
                        synchronized (transportationGraph) {
                            transportationGraph.addConnection(p1, p2, edgeWeight);
                        }
                        
                        // Count edges added to spanner
                        edgesAddedCounter.incrementAndGet();
                    }
                    
                    // Update progress - atomic operations
                    long completed = completedConnections.incrementAndGet();
                    int currentPercentage = (int) ((completed * 100) / allEdges.size());
                    long lastPct = lastPercentage.get();
                    if (currentPercentage > lastPct && 
                        lastPercentage.compareAndSet(lastPct, currentPercentage)) {
                        long edgesAdded = edgesAddedCounter.get();
                        System.out.printf("Processing progress: %d%% (%d/%d edges processed) - %d edges in spanner (%.2f%%)%n", 
                            currentPercentage, completed, allEdges.size(), edgesAdded, 
                            (edgesAdded * 100.0 / Math.max(1, completed)));
                    }
                }
            });
        }
        
        // Wait until all edges are processed
        try {
            while (completedConnections.get() < allEdges.size()) {
                Thread.sleep(500);
            }
            
            // Signal worker threads to finish
            done.set(true);
            executor.shutdown();
            executor.awaitTermination(1, TimeUnit.MINUTES);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            System.err.println("Parallel processing was interrupted: " + e.getMessage());
        }
        
        long endTime = System.currentTimeMillis();
        System.out.println("Greedy spanner created with " + paths.size() + " edges out of " 
                          + allEdges.size() + " possible edges ("
                          + (paths.size() * 100 / Math.max(1, allEdges.size())) + "%)");
        System.out.println("Total time: " + (endTime - startTime) / 1000.0 + " seconds");
        
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
    
    /**
     * Helper class to represent a potential edge in the spanner.
     */
    private static class PointPair {
        private final Point source;
        private final Point target;
        private final Point sourceNode;
        private final Point targetNode;
        private final double weight;
        private final List<Point> path;
        
        public PointPair(Point source, Point target, Point sourceNode, Point targetNode, 
                         double weight, List<Point> path) {
            this.source = source;
            this.target = target;
            this.sourceNode = sourceNode;
            this.targetNode = targetNode;
            this.weight = weight;
            this.path = path;
        }
        
        public Point getSource() {
            return source;
        }
        
        public Point getTarget() {
            return target;
        }
        
        public Point getSourceNode() {
            return sourceNode;
        }
        
        public Point getTargetNode() {
            return targetNode;
        }
        
        public double getWeight() {
            return weight;
        }
        
        public List<Point> getPath() {
            return path;
        }
        
        @Override
        public int hashCode() {
            // Create a unique hash for this edge pair (order independent)
            return source.hashCode() ^ target.hashCode();
        }
        
        @Override
        public boolean equals(Object obj) {
            if (!(obj instanceof PointPair)) return false;
            PointPair other = (PointPair) obj;
            // Edge is the same regardless of direction
            return (source.equals(other.source) && target.equals(other.target)) ||
                   (source.equals(other.target) && target.equals(other.source));
        }
    }
} 