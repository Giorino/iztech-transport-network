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
        analyzeCosts(transportationGraph, true);
    }
    
    /**
     * Performs transportation cost analysis on the given transportation graph
     * using the optimized transportation cost analyzer with the option to use minibuses.
     * 
     * @param transportationGraph The transportation graph to analyze
     * @param useMinibus Whether to use minibuses for small communities (true) or only buses (false)
     */
    public static void analyzeCosts(TransportationGraph transportationGraph, boolean useMinibus) {
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
        
        analyzeCosts(transportationGraph, communities, useMinibus);
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
        return analyzeCosts(transportationGraph, communities, true);
    }
    
    /**
     * Performs transportation cost analysis on the given transportation graph
     * and communities using the optimized transportation cost analyzer with the option to use minibuses.
     * 
     * @param transportationGraph The transportation graph to analyze
     * @param communities The communities to analyze
     * @param useMinibus Whether to use minibuses for small communities (true) or only buses (false)
     * @return A map of community ID to ClusterMetrics containing cost and distance information.
     */
    public static Map<Integer, ClusterMetrics> analyzeCosts(
            TransportationGraph transportationGraph, 
            Map<Integer, List<Node>> communities,
            boolean useMinibus) {
        System.out.println("Starting transportation cost analysis with pre-detected communities...");
        if (!useMinibus) {
            System.out.println("Minibus usage is disabled. Using only buses for all communities.");
        }
        
        // Step 1: Print community information
        System.out.println("Found " + communities.size() + " communities for analysis");
        
        // Step 2: Analyze transportation costs using optimized analyzer
        System.out.println("\nAnalyzing transportation costs with optimized analyzer...");
        OptimizedTransportationCostAnalyzer analyzer = new OptimizedTransportationCostAnalyzer(transportationGraph, useMinibus);
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
        saveAnalysisWithMetadata(graph, communities, clusteringAlgorithm, graphStrategy, kValue, true);
    }
    
    /**
     * Saves the transportation cost analysis with metadata to a structured folder.
     * 
     * @param graph The transportation graph
     * @param communities Map of community IDs to lists of nodes in each community
     * @param clusteringAlgorithm The clustering algorithm used in the analysis
     * @param graphStrategy The graph construction strategy used
     * @param kValue The k value used for graph construction (if applicable)
     * @param useMinibus Whether to use minibuses for small communities (true) or only buses (false)
     */
    public static void saveAnalysisWithMetadata(
            TransportationGraph graph, Map<Integer, List<Node>> communities,
            String clusteringAlgorithm, String graphStrategy, int kValue, boolean useMinibus) {
        
        LOGGER.info("Saving transportation cost analysis with metadata");
        
        // Use the optimized analyzer 
        OptimizedTransportationCostAnalyzer analyzer = new OptimizedTransportationCostAnalyzer(graph, useMinibus);
        
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
     * Analyzes transportation costs using both vehicle options (with/without minibuses)
     * and exports the results.
     * 
     * @param graph The transportation graph
     * @param communities Map of communities to their nodes
     * @param clusteringAlgorithm The clustering algorithm used
     * @param graphStrategy The graph construction strategy
     * @param outlierAlgorithm The outlier detection algorithm used
     * @param applyOutlierDetection Whether outlier detection is applied
     * @throws IOException If there's an error exporting the results
     */
    public static void analyzeAndCompareVehicleOptions(
            TransportationGraph graph, 
            Map<Integer, List<Node>> communities,
            String clusteringAlgorithm,
            String graphStrategy,
            String outlierAlgorithm,
            boolean applyOutlierDetection) throws IOException {
        
        LOGGER.info("Analyzing transportation costs with both vehicle options (with/without minibuses)");
        
        // Analysis with minibuses
        OptimizedTransportationCostAnalyzer optimizedAnalyzer = new OptimizedTransportationCostAnalyzer(graph, true);
        optimizedAnalyzer.analyzeTransportationCosts(communities);
        
        // Analysis with buses only
        OptimizedTransportationCostAnalyzer busesOnlyAnalyzer = new OptimizedTransportationCostAnalyzer(graph, false);
        busesOnlyAnalyzer.analyzeTransportationCosts(communities);
        
        // Export both analyses
        TransportationAnalysisExporter.exportAnalyses(
                clusteringAlgorithm, 
                graphStrategy,
                outlierAlgorithm,
                applyOutlierDetection,
                optimizedAnalyzer, 
                busesOnlyAnalyzer);
        
        LOGGER.info("Both analyses have been exported successfully");
        
        // Print a summary of both analyses
        System.out.println("\n===== ANALYSIS SUMMARY =====");
        System.out.println("ANALYSIS WITH MINIBUSES:");
        optimizedAnalyzer.printCostSummary();
        System.out.println("\nANALYSIS WITH BUSES ONLY:");
        busesOnlyAnalyzer.printCostSummary();
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