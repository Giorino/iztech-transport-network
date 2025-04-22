package com.izmir.transportation.cost;

import java.io.IOException;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.logging.Logger;

import com.izmir.transportation.TransportationGraph;
import com.izmir.transportation.cost.OptimizedTransportationCostAnalyzer.OptimizedCommunityTransportationCost;
import com.izmir.transportation.cost.OptimizedTransportationCostAnalyzer.VehicleType;
import com.izmir.transportation.helper.Node;
import com.izmir.transportation.helper.clustering.LeidenCommunityDetection;

/**
 * A class that demonstrates how to use the OptimizedTransportationCostAnalyzer
 * to analyze the transportation costs for communities in the transportation network.
 * 
 * @author yagizugurveren
 */
public class TransportationCostAnalysis {
    private static final Logger LOGGER = Logger.getLogger(TransportationCostAnalysis.class.getName());

    /**
     * Performs transportation cost analysis on the given transportation graph
     * using the optimized transportation cost analyzer.
     * 
     * @param transportationGraph The transportation graph to analyze
     */
    public static void analyzeCosts(TransportationGraph transportationGraph) {
        System.out.println("Starting transportation cost analysis...");
        
        // Step 1: Detect communities
        System.out.println("Detecting communities using Leiden algorithm...");
        LeidenCommunityDetection leidenCommunityDetection = new LeidenCommunityDetection(transportationGraph);
        
        // Configure the algorithm to use original points for clearer communities
        leidenCommunityDetection.useOriginalPointsOnly(true);
        
        // Detect communities
        Map<Integer, List<Node>> communities = leidenCommunityDetection.detectCommunities();
        
        // Print statistics
        String communityStats = leidenCommunityDetection.getCommunityStatistics();
        System.out.println("Community Detection Results:\n" + communityStats);
        
        analyzeCosts(transportationGraph, communities);
    }
    
    /**
     * Performs transportation cost analysis on the given transportation graph
     * and communities using the optimized transportation cost analyzer.
     * 
     * @param transportationGraph The transportation graph to analyze
     * @param communities The communities to analyze
     * @return A map of community ID to ClusterMetrics containing cost and distance information.
     */
    public static Map<Integer, ClusterMetrics> analyzeCosts(
            TransportationGraph transportationGraph, 
            Map<Integer, List<Node>> communities) {
        System.out.println("Starting transportation cost analysis with pre-detected communities...");
        
        // Step 1: Print community information
        System.out.println("Found " + communities.size() + " communities for analysis");
        
        // Step 2: Analyze transportation costs using optimized analyzer
        System.out.println("\nAnalyzing transportation costs with optimized analyzer...");
        OptimizedTransportationCostAnalyzer analyzer = new OptimizedTransportationCostAnalyzer(transportationGraph);
        analyzer.analyzeTransportationCosts(communities);
        
        // Step 2.5: Extract metrics after analysis
        Map<Integer, ClusterMetrics> clusterMetricsMap = new HashMap<>();
        Map<Integer, TransportationCostAnalyzer.CommunityTransportationCost> communityCosts = analyzer.getCommunityCosts();
        
        for (Map.Entry<Integer, TransportationCostAnalyzer.CommunityTransportationCost> entry : communityCosts.entrySet()) {
            int communityId = entry.getKey();
            TransportationCostAnalyzer.CommunityTransportationCost costData = entry.getValue();
            
            // Extract total distance and total fuel cost
            double totalDistance = costData.getTotalDistanceKm();
            double totalFuelCost = costData.getTotalFuelLiters() * OptimizedTransportationCostAnalyzer.FUEL_COST_PER_LITER;
            
            // Get vehicle type if available
            String vehicleType = "BUS"; // Default
            double fixedCost = costData.getBusCount() * OptimizedTransportationCostAnalyzer.ADDITIONAL_COST_PER_BUS;
            
            if (costData instanceof OptimizedCommunityTransportationCost) {
                OptimizedCommunityTransportationCost optimizedCostData = (OptimizedCommunityTransportationCost) costData;
                VehicleType vType = optimizedCostData.getVehicleType();
                vehicleType = vType.name();
                
                // Calculate fixed cost based on vehicle type
                fixedCost = costData.getBusCount() * vType.getFixedCost();
            }
            
            double totalCost = totalFuelCost + fixedCost;
            
            // Create enhanced ClusterMetrics with vehicle type information
            ClusterMetrics metrics = new ClusterMetrics(
                communityId, 
                totalDistance, 
                totalFuelCost,
                fixedCost,
                vehicleType,
                costData.getBusCount()
            );
            
            clusterMetricsMap.put(communityId, metrics);
        }
        
        // Step 3: Print summary of optimized analysis results
        System.out.println("\nPrinting cost analysis summary...");
        analyzer.printCostSummary();
        
        // Step 4: Save optimized results to CSV
        try {
            String filename = "transportation_cost_analysis.csv";
            analyzer.saveAnalysisToCSV(filename);
            System.out.println("\n===================================================");
            System.out.println("Detailed transportation cost analysis saved to: " + System.getProperty("user.dir") + "/" + filename);
            System.out.println("===================================================");
        } catch (IOException e) {
            System.err.println("Error saving analysis to CSV: " + e.getMessage());
        }
        
        System.out.println("Transportation cost analysis completed.");
        return clusterMetricsMap;
    }
    
    /**
     * Saves the transportation cost analysis with metadata to a structured folder.
     * 
     * @param graph The transportation graph
     * @param communities Map of community IDs to lists of nodes in each community
     * @param clusteringAlgorithm The clustering algorithm used in the analysis
     * @param graphStrategy The graph construction strategy used
     * @param kValue The k value used for graph construction (if applicable)
     */
    public static void saveAnalysisWithMetadata(
            TransportationGraph graph, Map<Integer, List<Node>> communities,
            String clusteringAlgorithm, String graphStrategy, int kValue) {
        
        LOGGER.info("Saving transportation cost analysis with metadata");
        
        // Use the optimized analyzer 
        OptimizedTransportationCostAnalyzer analyzer = new OptimizedTransportationCostAnalyzer(graph);
        
        // Make sure we have the latest analysis
        analyzer.analyzeTransportationCosts(communities);
        
        try {
            // Save with metadata
            analyzer.saveAnalysisWithMetadata(clusteringAlgorithm, graphStrategy, kValue);
        } catch (IOException e) {
            LOGGER.warning("Failed to save cost analysis with metadata: " + e.getMessage());
        }
    }
    
    /**
     * Main method for standalone testing.
     * 
     * @param args Command line arguments (not used)
     */
    public static void main(String[] args) {
        System.out.println("This class should be used through the main application.");
        System.out.println("Please run the CreateRoadNetwork class instead.");
    }
} 