package com.izmir.transportation.cost;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.jgrapht.Graph;
import org.jgrapht.alg.interfaces.ShortestPathAlgorithm;
import org.jgrapht.alg.shortestpath.DijkstraShortestPath;
import org.jgrapht.graph.DefaultWeightedEdge;

import com.izmir.transportation.TransportationGraph;
import com.izmir.transportation.helper.Node;

/**
 * A class that analyzes transportation costs for communities in the transportation network.
 * This class calculates the number of buses needed, their routes, and fuel costs for
 * each community detected by the clustering algorithm.
 * 
 * @author yagizugurveren
 */
public class TransportationCostAnalyzer {
    
    // Constants for cost calculation
    protected static final int BUS_CAPACITY = 50; // Each bus can carry 50 nodes/people
    protected static final double FUEL_EFFICIENCY = 0.35; // Liters per kilometer
    protected static final double FUEL_COST_PER_LITER = 1.8; // Cost in currency units per liter
    
    // The transportation graph containing the network data
    private final TransportationGraph transportationGraph;
    
    // Map of community ID to the analysis results
    private final Map<Integer, CommunityTransportationCost> communityCosts;
    
    /**
     * Creates a new TransportationCostAnalyzer for the given transportation graph.
     * 
     * @param transportationGraph The transportation graph to analyze
     */
    public TransportationCostAnalyzer(TransportationGraph transportationGraph) {
        this.transportationGraph = transportationGraph;
        this.communityCosts = new HashMap<>();
    }
    
    /**
     * Analyzes transportation costs for all communities in the graph.
     * 
     * @param communities Map of community IDs to lists of nodes in each community
     */
    public void analyzeTransportationCosts(Map<Integer, List<Node>> communities) {
        System.out.println("Analyzing transportation costs for " + communities.size() + " communities...");
        
        // Clear any previous analysis
        communityCosts.clear();
        
        // For each community, calculate the transportation cost
        for (Map.Entry<Integer, List<Node>> entry : communities.entrySet()) {
            int communityId = entry.getKey();
            List<Node> communityNodes = entry.getValue();
            
            // Calculate costs for this community
            CommunityTransportationCost cost = calculateCommunityTransportationCost(communityId, communityNodes);
            communityCosts.put(communityId, cost);
        }
        
        System.out.println("Transportation cost analysis completed.");
    }
    
    /**
     * Calculates transportation costs for a single community.
     * 
     * @param communityId The ID of the community
     * @param nodes The nodes in the community
     * @return The transportation cost analysis for the community
     */
    private CommunityTransportationCost calculateCommunityTransportationCost(int communityId, List<Node> nodes) {
        // Step 1: Determine number of buses needed based on node count
        int nodeCount = nodes.size();
        int busesNeeded = (int) Math.ceil((double) nodeCount / BUS_CAPACITY);
        
        // Create the cost object
        CommunityTransportationCost cost = new CommunityTransportationCost(communityId, nodeCount, busesNeeded);
        
        // Step 2: Determine bus routes and calculate distances
        if (busesNeeded > 0) {
            assignNodesToBuses(nodes, busesNeeded, cost);
        }
        
        return cost;
    }
    
    /**
     * Assigns nodes to buses and calculates routes for each bus.
     * 
     * @param nodes The nodes to assign to buses
     * @param busCount The number of buses available
     * @param cost The cost object to update with bus data
     */
    private void assignNodesToBuses(List<Node> nodes, int busCount, CommunityTransportationCost cost) {
        // Simple assignment strategy: divide nodes evenly among buses
        int nodesPerBus = (int) Math.ceil((double) nodes.size() / busCount);
        
        for (int busIndex = 0; busIndex < busCount; busIndex++) {
            // Get the subset of nodes for this bus
            int startIndex = busIndex * nodesPerBus;
            int endIndex = Math.min(startIndex + nodesPerBus, nodes.size());
            
            if (startIndex >= nodes.size()) {
                // No more nodes to assign
                break;
            }
            
            List<Node> busNodes = new ArrayList<>(nodes.subList(startIndex, endIndex));
            
            // Calculate optimal route for this bus
            BusRoute route = calculateOptimalRoute(busNodes);
            
            // Add the bus to the cost object
            cost.addBus(new BusInfo(busIndex + 1, busNodes.size(), route));
        }
    }
    
    /**
     * Calculates an optimal route for a bus to visit all assigned nodes.
     * Uses a greedy nearest-neighbor approach to approximate a solution.
     * 
     * @param nodes The nodes to visit
     * @return The calculated bus route
     */
    private BusRoute calculateOptimalRoute(List<Node> nodes) {
        // If no nodes or only one node, return empty route
        if (nodes.isEmpty()) {
            return new BusRoute(new ArrayList<>(), 0.0);
        }
        
        if (nodes.size() == 1) {
            return new BusRoute(new ArrayList<>(nodes), 0.0);
        }
        
        // Get the graph with real distances (not normalized weights)
        Graph<Node, DefaultWeightedEdge> graph = transportationGraph.getGraph();
        
        // Use Dijkstra's algorithm for shortest paths
        DijkstraShortestPath<Node, DefaultWeightedEdge> dijkstra = new DijkstraShortestPath<>(graph);
        
        // Start with the first node as current node
        Node currentNode = nodes.get(0);
        List<Node> remainingNodes = new ArrayList<>(nodes.subList(1, nodes.size()));
        
        // Build the route using greedy nearest-neighbor approach
        List<Node> route = new ArrayList<>();
        route.add(currentNode);
        
        double totalDistance = 0.0;
        
        while (!remainingNodes.isEmpty()) {
            // Find the nearest node
            Node nearestNode = null;
            double shortestDistance = Double.MAX_VALUE;
            ShortestPathAlgorithm.SingleSourcePaths<Node, DefaultWeightedEdge> paths = null;
            
            // Try to get paths from current node
            try {
                paths = dijkstra.getPaths(currentNode);
            } catch (IllegalArgumentException e) {
                // Node might not be in the graph, try an alternative approach
                System.out.println("Warning: Node not found in graph: " + currentNode);
                // Skip this node and continue with the next one if possible
                if (remainingNodes.isEmpty()) {
                    break;
                }
                currentNode = remainingNodes.remove(0);
                route.add(currentNode);
                continue;
            }
            
            for (Node node : remainingNodes) {
                try {
                    double distance = paths.getWeight(node);
                    if (distance < shortestDistance) {
                        shortestDistance = distance;
                        nearestNode = node;
                    }
                } catch (IllegalArgumentException e) {
                    // Target node might not be in the graph or not reachable
                    System.out.println("Warning: Path not found to node: " + node);
                }
            }
            
            // If we found a nearest node, add it to the route
            if (nearestNode != null) {
                remainingNodes.remove(nearestNode);
                route.add(nearestNode);
                totalDistance += shortestDistance;
                currentNode = nearestNode;
            } else {
                // If we can't find any reachable node, break the loop
                System.out.println("Warning: No reachable nodes found for bus route");
                break;
            }
        }
        
        // Convert distance from graph weight to kilometers
        // Assuming the original distances in the graph are in meters
        double distanceInKm = totalDistance / 1000.0;
        
        return new BusRoute(route, distanceInKm);
    }
    
    /**
     * Saves the transportation cost analysis to a CSV file.
     * 
     * @param filename The name of the file to save to
     * @throws IOException If there is an error writing to the file
     */
    public void saveAnalysisToCSV(String filename) throws IOException {
        try (BufferedWriter writer = new BufferedWriter(new FileWriter(filename))) {
            // Write header
            writer.write("Community_ID,Node_Count,Buses_Required,Total_Distance_Km,Total_Fuel_Liters,Total_Fuel_Cost\n");
            
            // Write each community's data
            for (CommunityTransportationCost cost : communityCosts.values()) {
                writer.write(String.format("%d,%d,%d,%.2f,%.2f,%.2f\n",
                        cost.getCommunityId(),
                        cost.getNodeCount(),
                        cost.getBusCount(),
                        cost.getTotalDistanceKm(),
                        cost.getTotalFuelLiters(),
                        cost.getTotalFuelCost()));
            }
            
            // Add a blank line before bus details
            writer.write("\n");
            writer.write("Community_ID,Bus_Number,Nodes_Carried,Distance_Km,Fuel_Liters,Fuel_Cost\n");
            
            // Write detailed bus data for each community
            for (CommunityTransportationCost cost : communityCosts.values()) {
                int communityId = cost.getCommunityId();
                
                for (BusInfo bus : cost.getBuses()) {
                    writer.write(String.format("%d,%d,%d,%.2f,%.2f,%.2f\n",
                            communityId,
                            bus.getBusNumber(),
                            bus.getNodeCount(),
                            bus.getRouteDistanceKm(),
                            bus.getFuelLiters(),
                            bus.getFuelCost()));
                }
            }
        }
        
        System.out.println("Transportation cost analysis saved to " + filename);
    }
    
    /**
     * Gets the transportation cost analysis for all communities.
     * 
     * @return Map of community IDs to transportation costs
     */
    public Map<Integer, CommunityTransportationCost> getCommunityCosts() {
        return communityCosts;
    }
    
    /**
     * Gets the transportation cost analysis for a specific community.
     * 
     * @param communityId The ID of the community
     * @return The transportation cost analysis, or null if not found
     */
    public CommunityTransportationCost getCommunityCost(int communityId) {
        return communityCosts.get(communityId);
    }
    
    /**
     * Prints a summary of the transportation cost analysis to the console.
     */
    public void printCostSummary() {
        if (communityCosts.isEmpty()) {
            System.out.println("No transportation cost analysis available.");
            return;
        }
        
        DecimalFormat df = new DecimalFormat("#,##0.00");
        
        System.out.println("\n=== TRANSPORTATION COST ANALYSIS SUMMARY ===");
        System.out.println(String.format("%-15s %-15s %-15s %-15s %-15s %-15s",
                "Community ID", "Node Count", "Buses", "Distance (km)", "Fuel (L)", "Cost"));
        System.out.println(String.format("%-15s %-15s %-15s %-15s %-15s %-15s",
                "------------", "----------", "-----", "-------------", "--------", "----"));
        
        double totalDistance = 0;
        double totalFuel = 0;
        double totalCost = 0;
        int totalBuses = 0;
        int totalNodes = 0;
        
        for (CommunityTransportationCost cost : communityCosts.values()) {
            System.out.println(String.format("%-15d %-15d %-15d %-15s %-15s %-15s",
                    cost.getCommunityId(),
                    cost.getNodeCount(),
                    cost.getBusCount(),
                    df.format(cost.getTotalDistanceKm()),
                    df.format(cost.getTotalFuelLiters()),
                    df.format(cost.getTotalFuelCost())));
            
            totalDistance += cost.getTotalDistanceKm();
            totalFuel += cost.getTotalFuelLiters();
            totalCost += cost.getTotalFuelCost();
            totalBuses += cost.getBusCount();
            totalNodes += cost.getNodeCount();
        }
        
        System.out.println(String.format("%-15s %-15s %-15s %-15s %-15s %-15s",
                "------------", "----------", "-----", "-------------", "--------", "----"));
        System.out.println(String.format("%-15s %-15d %-15d %-15s %-15s %-15s",
                "TOTAL",
                totalNodes,
                totalBuses,
                df.format(totalDistance),
                df.format(totalFuel),
                df.format(totalCost)));
        
        System.out.println("\n=== BUS DETAILS ===");
        for (CommunityTransportationCost cost : communityCosts.values()) {
            System.out.println("\nCommunity " + cost.getCommunityId() + " (" + cost.getNodeCount() + " nodes):");
            System.out.println(String.format("%-10s %-15s %-15s %-15s %-15s",
                    "Bus #", "Nodes Carried", "Distance (km)", "Fuel (L)", "Cost"));
            System.out.println(String.format("%-10s %-15s %-15s %-15s %-15s",
                    "-----", "-------------", "-------------", "--------", "----"));
            
            for (BusInfo bus : cost.getBuses()) {
                System.out.println(String.format("%-10d %-15d %-15s %-15s %-15s",
                        bus.getBusNumber(),
                        bus.getNodeCount(),
                        df.format(bus.getRouteDistanceKm()),
                        df.format(bus.getFuelLiters()),
                        df.format(bus.getFuelCost())));
            }
        }
    }
    
    /**
     * Inner class representing the transportation cost for a community.
     */
    public class CommunityTransportationCost {
        private final int communityId;
        private final int nodeCount;
        private final int busCount;
        private final List<BusInfo> buses;
        
        /**
         * Creates a new CommunityTransportationCost instance.
         * 
         * @param communityId The ID of the community
         * @param nodeCount The number of nodes in the community
         * @param busCount The number of buses needed
         */
        public CommunityTransportationCost(int communityId, int nodeCount, int busCount) {
            this.communityId = communityId;
            this.nodeCount = nodeCount;
            this.busCount = busCount;
            this.buses = new ArrayList<>();
        }
        
        /**
         * Adds a bus to this community.
         * 
         * @param bus The bus information to add
         */
        public void addBus(BusInfo bus) {
            buses.add(bus);
        }
        
        /**
         * Gets the total distance traveled by all buses in kilometers.
         * 
         * @return The total distance in kilometers
         */
        public double getTotalDistanceKm() {
            double total = 0;
            for (BusInfo bus : buses) {
                total += bus.getRouteDistanceKm();
            }
            return total;
        }
        
        /**
         * Gets the total fuel consumption in liters.
         * 
         * @return The total fuel consumption in liters
         */
        public double getTotalFuelLiters() {
            double total = 0;
            for (BusInfo bus : buses) {
                total += bus.getFuelLiters();
            }
            return total;
        }
        
        /**
         * Gets the total fuel cost.
         * 
         * @return The total fuel cost
         */
        public double getTotalFuelCost() {
            double total = 0;
            for (BusInfo bus : buses) {
                total += bus.getFuelCost();
            }
            return total;
        }
        
        /**
         * Gets the ID of the community.
         * 
         * @return The community ID
         */
        public int getCommunityId() {
            return communityId;
        }
        
        /**
         * Gets the number of nodes in the community.
         * 
         * @return The node count
         */
        public int getNodeCount() {
            return nodeCount;
        }
        
        /**
         * Gets the number of buses assigned to the community.
         * 
         * @return The bus count
         */
        public int getBusCount() {
            return busCount;
        }
        
        /**
         * Gets the list of buses assigned to the community.
         * 
         * @return The list of buses
         */
        public List<BusInfo> getBuses() {
            return buses;
        }
    }
    
    /**
     * Inner class representing information about a bus and its route.
     */
    public class BusInfo {
        private final int busNumber;
        private final int nodeCount;
        private final BusRoute route;
        
        /**
         * Creates a new BusInfo instance.
         * 
         * @param busNumber The bus number/ID
         * @param nodeCount The number of nodes assigned to this bus
         * @param route The route for this bus
         */
        public BusInfo(int busNumber, int nodeCount, BusRoute route) {
            this.busNumber = busNumber;
            this.nodeCount = nodeCount;
            this.route = route;
        }
        
        /**
         * Gets the bus number/ID.
         * 
         * @return The bus number
         */
        public int getBusNumber() {
            return busNumber;
        }
        
        /**
         * Gets the number of nodes assigned to this bus.
         * 
         * @return The node count
         */
        public int getNodeCount() {
            return nodeCount;
        }
        
        /**
         * Gets the distance traveled by this bus in kilometers.
         * 
         * @return The route distance in kilometers
         */
        public double getRouteDistanceKm() {
            return route.getDistanceKm();
        }
        
        /**
         * Gets the fuel consumption in liters.
         * 
         * @return The fuel consumption in liters
         */
        public double getFuelLiters() {
            return route.getDistanceKm() * FUEL_EFFICIENCY;
        }
        
        /**
         * Gets the fuel cost.
         * 
         * @return The fuel cost
         */
        public double getFuelCost() {
            return getFuelLiters() * FUEL_COST_PER_LITER;
        }
        
        /**
         * Gets the route for this bus.
         * 
         * @return The bus route
         */
        public BusRoute getRoute() {
            return route;
        }
    }
    
    /**
     * Inner class representing a bus route.
     */
    public class BusRoute {
        private final List<Node> nodes;
        private final double distanceKm;
        
        /**
         * Creates a new BusRoute instance.
         * 
         * @param nodes The ordered list of nodes to visit
         * @param distanceKm The total distance of the route in kilometers
         */
        public BusRoute(List<Node> nodes, double distanceKm) {
            this.nodes = nodes;
            this.distanceKm = distanceKm;
        }
        
        /**
         * Gets the ordered list of nodes to visit.
         * 
         * @return The list of nodes
         */
        public List<Node> getNodes() {
            return nodes;
        }
        
        /**
         * Gets the total distance of the route in kilometers.
         * 
         * @return The distance in kilometers
         */
        public double getDistanceKm() {
            return distanceKm;
        }
    }
} 