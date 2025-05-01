package com.izmir.transportation.cost;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.text.DecimalFormat;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Date;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Properties;
import java.util.Random;
import java.util.Set;

import org.jgrapht.Graph;
import org.jgrapht.alg.tour.ChristofidesThreeHalvesApproxMetricTSP;
import org.jgrapht.alg.tour.HeldKarpTSP;
import org.jgrapht.graph.DefaultWeightedEdge;
import org.jgrapht.graph.SimpleWeightedGraph;

import com.izmir.transportation.TransportationGraph;
import com.izmir.transportation.helper.Node;

/**
 * An optimized transportation cost analyzer that uses more sophisticated algorithms
 * for bus route optimization and node assignment to reduce overall costs.
 * This class extends the functionality of TransportationCostAnalyzer with more
 * advanced optimization techniques.
 * 
 * @author yagizugurveren
 */
public class OptimizedTransportationCostAnalyzer extends TransportationCostAnalyzer {
    
    // Configuration values loaded from properties
    public static final int BUS_CAPACITY = loadIntProperty("transportation.bus.capacity", 50);
    public static final double FUEL_EFFICIENCY = loadDoubleProperty("transportation.fuel.efficiency", 0.35);
    public static final double FUEL_COST_PER_LITER = loadDoubleProperty("transportation.fuel.cost.per.liter", 1.8);
    public static final double ADDITIONAL_COST_PER_BUS = loadDoubleProperty("transportation.bus.fixed.cost", 100.0);
    public static final double MAX_BUS_DISTANCE_KM = loadDoubleProperty("transportation.bus.max.distance.km", 150.0);
    
    // Minibus configuration values
    public static final int MINIBUS_CAPACITY = loadIntProperty("transportation.minibus.capacity", 25);
    public static final double MINIBUS_FUEL_EFFICIENCY = loadDoubleProperty("transportation.minibus.fuel.efficiency", 0.25);
    public static final double ADDITIONAL_COST_PER_MINIBUS = loadDoubleProperty("transportation.minibus.fixed.cost", 60.0);
    
    // Vehicle type threshold
    public static final int VEHICLE_TYPE_THRESHOLD = loadIntProperty("transportation.vehicle.type.threshold", 25);
    
    private static final boolean USE_CLUSTERING_FOR_BUS_ASSIGNMENT = true; // Whether to use clustering for bus assignment
    
    // Flag to determine whether to use minibuses for small communities
    private boolean useMinibus = true;
    
    /**
     * Loads a property as an integer with a default value.
     * 
     * @param propertyName Name of the property to load
     * @param defaultValue Default value if property is not found
     * @return Loaded property value or default value
     */
    private static int loadIntProperty(String propertyName, int defaultValue) {
        return Integer.parseInt(loadProperty(propertyName, String.valueOf(defaultValue)));
    }
    
    /**
     * Loads a property as a double with a default value.
     * 
     * @param propertyName Name of the property to load
     * @param defaultValue Default value if property is not found
     * @return Loaded property value or default value
     */
    private static double loadDoubleProperty(String propertyName, double defaultValue) {
        return Double.parseDouble(loadProperty(propertyName, String.valueOf(defaultValue)));
    }
    
    /**
     * Loads a property as a string with a default value.
     * 
     * @param propertyName Name of the property to load
     * @param defaultValue Default value if property is not found
     * @return Loaded property value or default value
     */
    private static String loadProperty(String propertyName, String defaultValue) {
        Properties properties = new Properties();
        try (InputStream input = OptimizedTransportationCostAnalyzer.class.getClassLoader()
                .getResourceAsStream("application.properties")) {
            
            if (input == null) {
                System.out.println("Unable to find application.properties, using default values");
                return defaultValue;
            }
            
            properties.load(input);
            return properties.getProperty(propertyName, defaultValue);
        } catch (IOException ex) {
            System.out.println("Error loading application.properties for property " + 
                              propertyName + ", using default value: " + defaultValue);
            return defaultValue;
        }
    }
    
    // Static initializer to log the loaded configuration values
    static {
        System.out.println("\nLoaded transportation cost parameters:");
        System.out.println("Bus Capacity: " + BUS_CAPACITY);
        System.out.println("Fuel Efficiency: " + FUEL_EFFICIENCY + " liters/km");
        System.out.println("Fuel Cost: " + FUEL_COST_PER_LITER + " per liter");
        System.out.println("Fixed Cost per Bus: " + ADDITIONAL_COST_PER_BUS);
        System.out.println("Max Bus Distance: " + MAX_BUS_DISTANCE_KM + " km");
        System.out.println("Minibus Capacity: " + MINIBUS_CAPACITY);
        System.out.println("Minibus Fuel Efficiency: " + MINIBUS_FUEL_EFFICIENCY + " liters/km");
        System.out.println("Fixed Cost per Minibus: " + ADDITIONAL_COST_PER_MINIBUS);
        System.out.println("Vehicle Type Threshold: " + VEHICLE_TYPE_THRESHOLD + " nodes\n");
    }
    
    /**
     * Creates a new OptimizedTransportationCostAnalyzer for the given transportation graph.
     * 
     * @param transportationGraph The transportation graph to analyze
     */
    public OptimizedTransportationCostAnalyzer(TransportationGraph transportationGraph) {
        super(transportationGraph);
    }
    
    /**
     * Creates a new OptimizedTransportationCostAnalyzer for the given transportation graph
     * with the option to use minibuses for small communities.
     * 
     * @param transportationGraph The transportation graph to analyze
     * @param useMinibus Whether to use minibuses for small communities (true) or only buses (false)
     */
    public OptimizedTransportationCostAnalyzer(TransportationGraph transportationGraph, boolean useMinibus) {
        super(transportationGraph);
        this.useMinibus = useMinibus;
    }
    
    /**
     * Analyzes transportation costs for all communities in the graph with advanced optimization.
     * This method overrides the base implementation to use more sophisticated algorithms.
     * 
     * @param communities Map of community IDs to lists of nodes in each community
     */
    @Override
    public void analyzeTransportationCosts(Map<Integer, List<Node>> communities) {
        System.out.println("Analyzing transportation costs with advanced optimization for " + 
                          communities.size() + " communities...");
        
        // Clear any previous analysis
        getCommunityCosts().clear();
        
        // For each community, calculate the optimized transportation cost
        for (Map.Entry<Integer, List<Node>> entry : communities.entrySet()) {
            int communityId = entry.getKey();
            List<Node> communityNodes = entry.getValue();
            
            // Calculate costs for this community using optimized methods
            CommunityTransportationCost communityCost = calculateOptimizedCommunityTransportationCost(communityId, communityNodes);
            getCommunityCosts().put(communityId, communityCost);
        }
        
        System.out.println("Optimized transportation cost analysis completed.");
    }
    
    /**
     * Calculates transportation costs for a single community using advanced optimization techniques.
     * 
     * @param communityId The ID of the community
     * @param nodes The nodes in the community
     * @return The transportation cost analysis for the community
     */
    private CommunityTransportationCost calculateOptimizedCommunityTransportationCost(int communityId, List<Node> nodes) {
        // Step 1: Determine the vehicle type based on community size
        VehicleType vehicleType = determineVehicleType(nodes.size());
        
        // Step 2: Determine number of vehicles needed based on node count and vehicle capacity
        int nodeCount = nodes.size();
        int vehiclesNeeded = (int) Math.ceil((double) nodeCount / vehicleType.getCapacity());
        
        // Create the cost object with vehicle type information
        OptimizedCommunityTransportationCost cost = new OptimizedCommunityTransportationCost(communityId, nodeCount, vehiclesNeeded);
        cost.setVehicleType(vehicleType);
        
        // Step 3: Use advanced techniques to optimize routes and assignments
        if (vehiclesNeeded > 0) {
            if (USE_CLUSTERING_FOR_BUS_ASSIGNMENT && vehiclesNeeded > 1) {
                assignNodesToVehiclesWithClustering(nodes, vehiclesNeeded, cost, vehicleType);
            } else {
                assignNodesToVehiclesWithTSP(nodes, vehiclesNeeded, cost, vehicleType);
            }
        }
        
        return cost;
    }
    
    /**
     * Determines the appropriate vehicle type based on community size.
     * If useMinibus is false, always returns BUS regardless of size.
     * 
     * @param communitySize The number of nodes in the community
     * @return The appropriate vehicle type
     */
    private VehicleType determineVehicleType(int communitySize) {
        if (useMinibus && communitySize <= VEHICLE_TYPE_THRESHOLD) {
            return VehicleType.MINIBUS;
        } else {
            return VehicleType.BUS;
        }
    }
    
    /**
     * Assigns nodes to vehicles using a spatial clustering approach to group nearby nodes.
     * This method tries to create balanced clusters that minimize travel distances.
     * 
     * @param nodes The nodes to assign to vehicles
     * @param vehicleCount The number of vehicles available
     * @param cost The cost object to update with vehicle data
     * @param vehicleType The type of vehicle to use
     */
    private void assignNodesToVehiclesWithClustering(List<Node> nodes, int vehicleCount, 
                                              OptimizedCommunityTransportationCost cost, VehicleType vehicleType) {
        System.out.println("Using spatial clustering to assign " + nodes.size() + " nodes to " + 
                           vehicleCount + " " + vehicleType.name().toLowerCase() + "es");
        
        // Step 1: Find a central point (centroid) for the community
        double[] centroid = calculateCentroid(nodes);
        
        // Step 2: Sort nodes by their distance from the centroid
        List<Node> sortedNodes = new ArrayList<>(nodes);
        sortedNodes.sort((a, b) -> {
            double distA = calculateGeoDistance(a, centroid[0], centroid[1]);
            double distB = calculateGeoDistance(b, centroid[0], centroid[1]);
            return Double.compare(distA, distB);
        });
        
        // Step 3: Divide the area into spatial clusters (pie slices)
        List<List<Node>> clusters = new ArrayList<>();
        
        // Create empty clusters
        for (int i = 0; i < vehicleCount; i++) {
            clusters.add(new ArrayList<>());
        }
        
        // Assign nodes to clusters based on their angle from centroid
        for (Node node : sortedNodes) {
            double angle = calculateAngle(node, centroid[0], centroid[1]);
            int clusterIndex = (int) ((angle / (2 * Math.PI)) * vehicleCount) % vehicleCount;
            clusters.get(clusterIndex).add(node);
        }
        
        // Step 4: Balance clusters to ensure no vehicle is overloaded
        balanceClusters(clusters, nodes.size() / vehicleCount);
        
        // Step 5: For each cluster, calculate an optimal route
        for (int i = 0; i < clusters.size(); i++) {
            List<Node> clusterNodes = clusters.get(i);
            
            if (clusterNodes.isEmpty()) {
                continue;
            }
            
            // Calculate an optimal route for this cluster using TSP
            BusRoute route = calculateOptimalTSPRoute(clusterNodes);
            
            // Add the bus/minibus to the cost object
            OptimizedBusInfo vehicleInfo = new OptimizedBusInfo(i + 1, clusterNodes.size(), route, vehicleType);
            cost.addBus(vehicleInfo);
            
            System.out.println(vehicleType.name() + " " + (i + 1) + ": " + clusterNodes.size() + 
                               " nodes, route distance: " + String.format("%.2f", route.getDistanceKm()) + " km");
        }
    }
    
    /**
     * Assigns nodes to vehicles using TSP (Traveling Salesman Problem) optimization
     * to find the shortest route for each vehicle.
     * 
     * @param nodes The nodes to assign to vehicles
     * @param vehicleCount The number of vehicles available
     * @param cost The cost object to update with vehicle data
     * @param vehicleType The type of vehicle to use
     */
    private void assignNodesToVehiclesWithTSP(List<Node> nodes, int vehicleCount, 
                                      OptimizedCommunityTransportationCost cost, VehicleType vehicleType) {
        
        // Simple assignment strategy: divide nodes evenly among vehicles
        int nodesPerVehicle = (int) Math.ceil((double) nodes.size() / vehicleCount);
        
        for (int vehicleIndex = 0; vehicleIndex < vehicleCount; vehicleIndex++) {
            // Get the subset of nodes for this vehicle
            int startIndex = vehicleIndex * nodesPerVehicle;
            int endIndex = Math.min(startIndex + nodesPerVehicle, nodes.size());
            
            if (startIndex >= nodes.size()) {
                // No more nodes to assign
                break;
            }
            
            List<Node> vehicleNodes = new ArrayList<>(nodes.subList(startIndex, endIndex));
            
            // Calculate optimal route for this vehicle using TSP
            BusRoute route = calculateOptimalTSPRoute(vehicleNodes);
            
            // Add the vehicle to the cost object
            OptimizedBusInfo vehicleInfo = new OptimizedBusInfo(vehicleIndex + 1, vehicleNodes.size(), route, vehicleType);
            cost.addBus(vehicleInfo);
        }
    }
    
    /**
     * Calculates an optimal route for a vehicle to visit all assigned nodes using TSP algorithms.
     * This method attempts to use the best available TSP solver based on the number of nodes.
     * 
     * @param nodes The nodes to visit
     * @return The calculated bus route
     */
    private BusRoute calculateOptimalTSPRoute(List<Node> nodes) {
        // If no nodes or only one node, return empty route
        if (nodes.isEmpty()) {
            return new BusRoute(new ArrayList<>(), 0.0);
        }
        
        if (nodes.size() == 1) {
            return new BusRoute(new ArrayList<>(nodes), 0.0);
        }
        
        // For very small sets of nodes, we can use an exact algorithm
        if (nodes.size() <= 10) {
            return calculateExactTSPRoute(nodes);
        } else {
            // For larger sets, use an approximation algorithm
            return calculateApproximateTSPRoute(nodes);
        }
    }
    
    /**
     * Calculates an exact solution to the TSP problem using Held-Karp algorithm.
     * This is only feasible for small numbers of nodes (up to 10-12) due to exponential complexity.
     * 
     * @param nodes The nodes to visit
     * @return The calculated bus route
     */
    private BusRoute calculateExactTSPRoute(List<Node> nodes) {
        // Create a complete graph with distances between all nodes
        SimpleWeightedGraph<Node, DefaultWeightedEdge> completeGraph = createCompleteGraph(nodes);
        
        // Use Held-Karp algorithm for exact TSP solution
        HeldKarpTSP<Node, DefaultWeightedEdge> tspSolver = new HeldKarpTSP<>();
        
        try {
            // Calculate the tour
            List<Node> tour = tspSolver.getTour(completeGraph).getVertexList();
            
            // Remove the duplicate last node (tour end = tour start)
            if (!tour.isEmpty() && tour.get(0).equals(tour.get(tour.size() - 1))) {
                tour.remove(tour.size() - 1);
            }
            
            // Calculate the total distance
            double totalDistance = calculateRouteDistance(tour, completeGraph);
            
            return new BusRoute(tour, totalDistance / 1000.0); // Convert to km
        } catch (Exception e) {
            System.err.println("Error calculating exact TSP route: " + e.getMessage());
            
            // Fall back to approximate method
            return calculateApproximateTSPRoute(nodes);
        }
    }
    
    /**
     * Calculates an approximate solution to the TSP problem using Christofides algorithm.
     * This is suitable for larger sets of nodes.
     * 
     * @param nodes The nodes to visit
     * @return The calculated bus route
     */
    private BusRoute calculateApproximateTSPRoute(List<Node> nodes) {
        // Create a complete graph with distances between all nodes
        SimpleWeightedGraph<Node, DefaultWeightedEdge> completeGraph = createCompleteGraph(nodes);
        
        // Use Christofides algorithm for approximate TSP solution
        ChristofidesThreeHalvesApproxMetricTSP<Node, DefaultWeightedEdge> tspSolver = 
            new ChristofidesThreeHalvesApproxMetricTSP<>();
        
        try {
            // Calculate the tour
            List<Node> tour = tspSolver.getTour(completeGraph).getVertexList();
            
            // Remove the duplicate last node (tour end = tour start)
            if (!tour.isEmpty() && tour.get(0).equals(tour.get(tour.size() - 1))) {
                tour.remove(tour.size() - 1);
            }
            
            // Calculate the total distance
            double totalDistance = calculateRouteDistance(tour, completeGraph);
            
            return new BusRoute(tour, totalDistance / 1000.0); // Convert to km
        } catch (Exception e) {
            System.err.println("Error calculating approximate TSP route: " + e.getMessage());
            
            // Fall back to nearest neighbor approach
            return calculateNearestNeighborRoute(nodes);
        }
    }
    
    /**
     * Creates a complete graph where every node is connected to every other node
     * with edge weights representing the geographic distance between nodes.
     * 
     * @param nodes The nodes to include in the graph
     * @return A complete weighted graph
     */
    private SimpleWeightedGraph<Node, DefaultWeightedEdge> createCompleteGraph(List<Node> nodes) {
        SimpleWeightedGraph<Node, DefaultWeightedEdge> graph = 
            new SimpleWeightedGraph<>(DefaultWeightedEdge.class);
        
        // Add all nodes to the graph
        for (Node node : nodes) {
            graph.addVertex(node);
        }
        
        // Connect every node to every other node
        for (int i = 0; i < nodes.size(); i++) {
            for (int j = i + 1; j < nodes.size(); j++) {
                Node source = nodes.get(i);
                Node target = nodes.get(j);
                
                // Calculate the geographic distance between nodes
                double distance = calculateGeoDistance(source, target);
                
                // Add the edge with the distance as weight
                DefaultWeightedEdge edge = graph.addEdge(source, target);
                if (edge != null) {
                    graph.setEdgeWeight(edge, distance);
                }
            }
        }
        
        return graph;
    }
    
    /**
     * Calculates the geographic distance between two nodes in meters.
     * 
     * @param node1 The first node
     * @param node2 The second node
     * @return The distance in meters
     */
    private double calculateGeoDistance(Node node1, Node node2) {
        double lon1 = node1.getLocation().getX();
        double lat1 = node1.getLocation().getY();
        double lon2 = node2.getLocation().getX();
        double lat2 = node2.getLocation().getY();
        
        return calculateGeoDistance(lon1, lat1, lon2, lat2);
    }
    
    /**
     * Calculates the geographic distance between a node and a point in meters.
     * 
     * @param node The node
     * @param lon The longitude of the point
     * @param lat The latitude of the point
     * @return The distance in meters
     */
    private double calculateGeoDistance(Node node, double lon, double lat) {
        double nodeLon = node.getLocation().getX();
        double nodeLat = node.getLocation().getY();
        
        return calculateGeoDistance(nodeLon, nodeLat, lon, lat);
    }
    
    /**
     * Calculates the geographic distance between two points using the Haversine formula.
     * 
     * @param lon1 Longitude of the first point
     * @param lat1 Latitude of the first point
     * @param lon2 Longitude of the second point
     * @param lat2 Latitude of the second point
     * @return The distance in meters
     */
    private double calculateGeoDistance(double lon1, double lat1, double lon2, double lat2) {
        final int R = 6371000; // Earth radius in meters
        
        double latDistance = Math.toRadians(lat2 - lat1);
        double lonDistance = Math.toRadians(lon2 - lon1);
        
        double a = Math.sin(latDistance / 2) * Math.sin(latDistance / 2)
                 + Math.cos(Math.toRadians(lat1)) * Math.cos(Math.toRadians(lat2))
                 * Math.sin(lonDistance / 2) * Math.sin(lonDistance / 2);
        
        double c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
        
        return R * c; // Distance in meters
    }
    
    /**
     * Calculates the angle (in radians) of a node relative to a center point.
     * 
     * @param node The node
     * @param centerLon The longitude of the center point
     * @param centerLat The latitude of the center point
     * @return The angle in radians (0 to 2π)
     */
    private double calculateAngle(Node node, double centerLon, double centerLat) {
        double nodeLon = node.getLocation().getX();
        double nodeLat = node.getLocation().getY();
        
        double angle = Math.atan2(nodeLat - centerLat, nodeLon - centerLon);
        
        // Convert to range 0 to 2π
        if (angle < 0) {
            angle += 2 * Math.PI;
        }
        
        return angle;
    }
    
    /**
     * Calculates the centroid (weighted center) of a set of nodes.
     * 
     * @param nodes The nodes to calculate the centroid for
     * @return An array with [longitude, latitude] of the centroid
     */
    private double[] calculateCentroid(List<Node> nodes) {
        double sumLon = 0;
        double sumLat = 0;
        
        for (Node node : nodes) {
            sumLon += node.getLocation().getX();
            sumLat += node.getLocation().getY();
        }
        
        int count = nodes.size();
        return new double[]{sumLon / count, sumLat / count};
    }
    
    /**
     * Calculates the total distance of a route through the given graph.
     * 
     * @param route The ordered list of nodes in the route
     * @param graph The graph containing the edges
     * @return The total distance of the route
     */
    private double calculateRouteDistance(List<Node> route, Graph<Node, DefaultWeightedEdge> graph) {
        double totalDistance = 0;
        
        for (int i = 0; i < route.size() - 1; i++) {
            Node current = route.get(i);
            Node next = route.get(i + 1);
            
            DefaultWeightedEdge edge = graph.getEdge(current, next);
            if (edge != null) {
                totalDistance += graph.getEdgeWeight(edge);
            } else {
                // If no direct edge, calculate geographic distance
                totalDistance += calculateGeoDistance(current, next);
            }
        }
        
        // Add distance from last to first node to complete the circuit
        if (route.size() > 1) {
            Node last = route.get(route.size() - 1);
            Node first = route.get(0);
            
            DefaultWeightedEdge edge = graph.getEdge(last, first);
            if (edge != null) {
                totalDistance += graph.getEdgeWeight(edge);
            } else {
                totalDistance += calculateGeoDistance(last, first);
            }
        }
        
        return totalDistance;
    }
    
    /**
     * Balances the clusters to ensure no cluster has significantly more nodes than others.
     * 
     * @param clusters The list of clusters to balance
     * @param targetSize The target size for each cluster
     */
    private void balanceClusters(List<List<Node>> clusters, int targetSize) {
        boolean changed;
        
        do {
            changed = false;
            
            // Find the largest and smallest clusters
            int maxIndex = -1;
            int minIndex = -1;
            int maxSize = Integer.MIN_VALUE;
            int minSize = Integer.MAX_VALUE;
            
            for (int i = 0; i < clusters.size(); i++) {
                int size = clusters.get(i).size();
                
                if (size > maxSize) {
                    maxSize = size;
                    maxIndex = i;
                }
                
                if (size < minSize) {
                    minSize = size;
                    minIndex = i;
                }
            }
            
            // If the difference is significant, move a node from largest to smallest
            if (maxIndex != minIndex && maxSize > targetSize && maxSize - minSize > 1) {
                List<Node> maxCluster = clusters.get(maxIndex);
                List<Node> minCluster = clusters.get(minIndex);
                
                // Find the node in maxCluster that is closest to minCluster's centroid
                double[] minCentroid = calculateCentroid(minCluster);
                
                Node bestNode = null;
                double bestDistance = Double.MAX_VALUE;
                
                for (Node node : maxCluster) {
                    double distance = calculateGeoDistance(node, minCentroid[0], minCentroid[1]);
                    if (distance < bestDistance) {
                        bestDistance = distance;
                        bestNode = node;
                    }
                }
                
                if (bestNode != null) {
                    maxCluster.remove(bestNode);
                    minCluster.add(bestNode);
                    changed = true;
                }
            }
        } while (changed);
    }
    
    /**
     * Calculates a route using a nearest-neighbor approach (backup method).
     * 
     * @param nodes The nodes to visit
     * @return The calculated bus route
     */
    private BusRoute calculateNearestNeighborRoute(List<Node> nodes) {
        // If no nodes or only one node, return empty route
        if (nodes.isEmpty()) {
            return new BusRoute(new ArrayList<>(), 0.0);
        }
        
        if (nodes.size() == 1) {
            return new BusRoute(new ArrayList<>(nodes), 0.0);
        }
        
        // Start with a random node
        Random random = new Random(42); // Fixed seed for reproducibility
        
        List<Node> tour = new ArrayList<>();
        Set<Node> unvisited = new HashSet<>(nodes);
        
        // Start with a random node
        Node current = nodes.get(random.nextInt(nodes.size()));
        tour.add(current);
        unvisited.remove(current);
        
        double totalDistance = 0;
        
        // Build the tour using nearest-neighbor heuristic
        while (!unvisited.isEmpty()) {
            Node nearest = null;
            double minDistance = Double.MAX_VALUE;
            
            for (Node node : unvisited) {
                double distance = calculateGeoDistance(current, node);
                if (distance < minDistance) {
                    minDistance = distance;
                    nearest = node;
                }
            }
            
            if (nearest != null) {
                tour.add(nearest);
                unvisited.remove(nearest);
                totalDistance += minDistance;
                current = nearest;
            } else {
                // This shouldn't happen, but just in case
                break;
            }
        }
        
        // Add the return to starting point if needed
        if (tour.size() > 1) {
            totalDistance += calculateGeoDistance(tour.get(tour.size() - 1), tour.get(0));
        }
        
        return new BusRoute(tour, totalDistance / 1000.0); // Convert to km
    }
    
    /**
     * Prints a detailed summary of the transportation cost analysis to the console.
     * This version includes additional optimization details and vehicle types.
     */
    @Override
    public void printCostSummary() {
        if (getCommunityCosts().isEmpty()) {
            System.out.println("No transportation cost analysis available.");
            return;
        }
        
        DecimalFormat df = new DecimalFormat("#,##0.00");
        
        System.out.println("\n=== OPTIMIZED TRANSPORTATION COST ANALYSIS SUMMARY ===");
        System.out.println(String.format("%-15s %-15s %-15s %-15s %-15s %-15s %-15s",
                "Community ID", "Node Count", "Vehicle Type", "Vehicles", "Distance (km)", "Fuel (L)", "Cost"));
        System.out.println(String.format("%-15s %-15s %-15s %-15s %-15s %-15s %-15s",
                "------------", "----------", "------------", "--------", "-------------", "--------", "----"));
        
        double totalDistance = 0;
        double totalFuel = 0;
        double totalCost = 0;
        int totalBuses = 0;
        int totalMinibuses = 0;
        int totalNodes = 0;
        
        for (CommunityTransportationCost cost : getCommunityCosts().values()) {
            // Get vehicle type
            VehicleType vehicleType = VehicleType.BUS; // Default
            
            if (cost instanceof OptimizedCommunityTransportationCost) {
                vehicleType = ((OptimizedCommunityTransportationCost) cost).getVehicleType();
            }
            
            String vehicleTypeStr = vehicleType.name();
            
            // Calculate fixed cost based on vehicle type
            double fixedCostPerVehicle = vehicleType.getFixedCost();
            double busCost = cost.getTotalFuelCost() + (cost.getBusCount() * fixedCostPerVehicle);
            
            System.out.println(String.format("%-15d %-15d %-15s %-15d %-15s %-15s %-15s",
                    cost.getCommunityId(),
                    cost.getNodeCount(),
                    vehicleTypeStr,
                    cost.getBusCount(),
                    df.format(cost.getTotalDistanceKm()),
                    df.format(cost.getTotalFuelLiters()),
                    df.format(busCost)));
            
            totalDistance += cost.getTotalDistanceKm();
            totalFuel += cost.getTotalFuelLiters();
            totalCost += busCost;
            
            if (vehicleType == VehicleType.MINIBUS) {
                totalMinibuses += cost.getBusCount();
            } else {
                totalBuses += cost.getBusCount();
            }
            
            totalNodes += cost.getNodeCount();
        }
        
        System.out.println(String.format("%-15s %-15s %-15s %-15s %-15s %-15s %-15s",
                "------------", "----------", "------------", "--------", "-------------", "--------", "----"));
        System.out.println(String.format("%-15s %-15d %-15s %-15s %-15s %-15s %-15s",
                "TOTAL",
                totalNodes,
                "",
                "B:" + totalBuses + " M:" + totalMinibuses,
                df.format(totalDistance),
                df.format(totalFuel),
                df.format(totalCost)));
        
        System.out.println("\nFixed cost per bus: " + ADDITIONAL_COST_PER_BUS);
        System.out.println("Fixed cost per minibus: " + ADDITIONAL_COST_PER_MINIBUS);
        System.out.println("Total fixed cost: " + df.format((totalBuses * ADDITIONAL_COST_PER_BUS) + 
                                                          (totalMinibuses * ADDITIONAL_COST_PER_MINIBUS)));
        System.out.println("Total fuel cost: " + df.format(totalFuel * FUEL_COST_PER_LITER));
        System.out.println("Total cost: " + df.format(totalCost));
    }
    
    /**
     * Saves the transportation cost analysis with metadata (clustering algorithm, graph strategy, etc.)
     * to a CSV file in a structured folder.
     * 
     * @param clusteringAlgorithm The clustering algorithm used in the analysis
     * @param graphStrategy The graph construction strategy used
     * @param kValue The k value used for graph construction (if applicable)
     * @throws IOException If there is an error writing to the file
     */
    public void saveAnalysisWithMetadata(String clusteringAlgorithm, String graphStrategy, int kValue) throws IOException {
        // Create a timestamp for the filename
        SimpleDateFormat dateFormat = new SimpleDateFormat("yyyy-MM-dd_HHmmss");
        String timestamp = dateFormat.format(new Date());
        
        // Create directory structure: results/cost_analysis/YYYY-MM-DD/
        String dateOnly = timestamp.split("_")[0];
        String baseDir = "results/cost_analysis/" + dateOnly;
        
        // Create directories if they don't exist
        Files.createDirectories(Paths.get(baseDir));
        
        // Create filename with metadata
        String filename = baseDir + "/transportation_cost_" + clusteringAlgorithm + "_" + 
                         graphStrategy + "_K" + kValue + "_" + timestamp + ".csv";
        
        try (BufferedWriter writer = new BufferedWriter(new FileWriter(filename))) {
            // Write metadata header
            writer.write("# Analysis Metadata\n");
            writer.write("# Timestamp: " + timestamp + "\n");
            writer.write("# Clustering Algorithm: " + clusteringAlgorithm + "\n");
            writer.write("# Graph Construction Strategy: " + graphStrategy + "\n");
            writer.write("# K Value: " + kValue + "\n");
            writer.write("# Bus Capacity: " + BUS_CAPACITY + "\n");
            writer.write("# Minibus Capacity: " + MINIBUS_CAPACITY + "\n");
            writer.write("# Fuel Efficiency (Bus): " + FUEL_EFFICIENCY + " liters/km\n");
            writer.write("# Fuel Efficiency (Minibus): " + MINIBUS_FUEL_EFFICIENCY + " liters/km\n");
            writer.write("# Fuel Cost: " + FUEL_COST_PER_LITER + " per liter\n");
            writer.write("# Fixed Cost (Bus): " + ADDITIONAL_COST_PER_BUS + "\n");
            writer.write("# Fixed Cost (Minibus): " + ADDITIONAL_COST_PER_MINIBUS + "\n\n");
            
            // Write regular CSV content
            // Write header
            writer.write("Community_ID,Node_Count,Vehicle_Type,Vehicles_Required,Total_Distance_Km,Total_Fuel_Liters," +
                        "Fuel_Cost,Fixed_Cost,Total_Cost\n");
            
            // Write each community's data
            for (CommunityTransportationCost cost : getCommunityCosts().values()) {
                VehicleType vehicleType = VehicleType.BUS; // Default
                
                if (cost instanceof OptimizedCommunityTransportationCost) {
                    vehicleType = ((OptimizedCommunityTransportationCost) cost).getVehicleType();
                }
                
                String vehicleTypeStr = vehicleType.name();
                
                double fixedCostPerVehicle = vehicleType.getFixedCost();
                double fixedCost = cost.getBusCount() * fixedCostPerVehicle;
                double fuelCost = cost.getTotalFuelLiters() * FUEL_COST_PER_LITER;
                double totalCost = fuelCost + fixedCost;
                
                writer.write(String.format("%d,%d,%s,%d,%.2f,%.2f,%.2f,%.2f,%.2f\n",
                        cost.getCommunityId(),
                        cost.getNodeCount(),
                        vehicleTypeStr,
                        cost.getBusCount(),
                        cost.getTotalDistanceKm(),
                        cost.getTotalFuelLiters(),
                        fuelCost,
                        fixedCost,
                        totalCost));
            }
            
            // Add a blank line before vehicle details
            writer.write("\n");
            writer.write("Community_ID,Vehicle_Type,Vehicle_Number,Nodes_Carried,Distance_Km,Fuel_Liters," +
                        "Fuel_Cost,Fixed_Cost,Total_Cost\n");
            
            // Write detailed vehicle data for each community
            for (CommunityTransportationCost cost : getCommunityCosts().values()) {
                int communityId = cost.getCommunityId();
                
                for (BusInfo vehicle : cost.getBuses()) {
                    VehicleType vehicleType = VehicleType.BUS; // Default
                    
                    if (vehicle instanceof OptimizedBusInfo) {
                        vehicleType = ((OptimizedBusInfo) vehicle).getVehicleType();
                    }
                    
                    String vehicleTypeStr = vehicleType.name();
                    
                    double fixedCostPerVehicle = vehicleType.getFixedCost();
                    double fuelCost = vehicle.getFuelLiters() * FUEL_COST_PER_LITER;
                    double totalCost = fuelCost + fixedCostPerVehicle;
                    
                    writer.write(String.format("%d,%s,%d,%d,%.2f,%.2f,%.2f,%.2f,%.2f\n",
                            communityId,
                            vehicleTypeStr,
                            vehicle.getBusNumber(),
                            vehicle.getNodeCount(),
                            vehicle.getRouteDistanceKm(),
                            vehicle.getFuelLiters(),
                            fuelCost,
                            fixedCostPerVehicle,
                            totalCost));
                }
            }
        }
        
        System.out.println("Optimized transportation cost analysis with metadata saved to " + filename);
    }
    
    /**
     * Enum representing the types of vehicles available for transportation.
     */
    public enum VehicleType {
        BUS(BUS_CAPACITY, FUEL_EFFICIENCY, ADDITIONAL_COST_PER_BUS),
        MINIBUS(MINIBUS_CAPACITY, MINIBUS_FUEL_EFFICIENCY, ADDITIONAL_COST_PER_MINIBUS);
        
        private final int capacity;
        private final double fuelEfficiency;
        private final double fixedCost;
        
        VehicleType(int capacity, double fuelEfficiency, double fixedCost) {
            this.capacity = capacity;
            this.fuelEfficiency = fuelEfficiency;
            this.fixedCost = fixedCost;
        }
        
        public int getCapacity() {
            return capacity;
        }
        
        public double getFuelEfficiency() {
            return fuelEfficiency;
        }
        
        public double getFixedCost() {
            return fixedCost;
        }
    }
    
    /**
     * Extended version of CommunityTransportationCost that includes vehicle type information.
     */
    public class OptimizedCommunityTransportationCost extends CommunityTransportationCost {
        private VehicleType vehicleType = VehicleType.BUS; // Default to standard bus
        
        public OptimizedCommunityTransportationCost(int communityId, int nodeCount, int busCount) {
            super(communityId, nodeCount, busCount);
        }
        
        public VehicleType getVehicleType() {
            return vehicleType;
        }
        
        public void setVehicleType(VehicleType vehicleType) {
            this.vehicleType = vehicleType;
        }
        
        @Override
        public void addBus(BusInfo bus) {
            super.addBus(bus);
        }
        
        @Override
        public double getTotalFuelLiters() {
            double total = 0;
            for (BusInfo bus : getBuses()) {
                if (bus instanceof OptimizedBusInfo) {
                    total += ((OptimizedBusInfo) bus).getFuelLiters();
                } else {
                    total += bus.getFuelLiters();
                }
            }
            return total;
        }
    }
    
    /**
     * Extended version of BusInfo that includes vehicle type information.
     */
    public class OptimizedBusInfo extends BusInfo {
        private VehicleType vehicleType = VehicleType.BUS; // Default to standard bus
        
        public OptimizedBusInfo(int busNumber, int nodeCount, BusRoute route, VehicleType vehicleType) {
            super(busNumber, nodeCount, route);
            this.vehicleType = vehicleType;
        }
        
        public VehicleType getVehicleType() {
            return vehicleType;
        }
        
        @Override
        public double getFuelLiters() {
            return getRouteDistanceKm() * vehicleType.getFuelEfficiency();
        }
        
        @Override
        public double getFuelCost() {
            return getFuelLiters() * FUEL_COST_PER_LITER;
        }
    }
} 