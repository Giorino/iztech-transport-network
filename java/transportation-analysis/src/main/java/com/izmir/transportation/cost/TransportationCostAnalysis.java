package com.izmir.transportation.cost;

import java.io.IOException;
import java.util.List;
import java.util.Map;

import com.izmir.transportation.TransportationGraph;
import com.izmir.transportation.helper.Node;
import com.izmir.transportation.helper.clustering.LeidenCommunityDetection;

/**
 * A class that demonstrates how to use the OptimizedTransportationCostAnalyzer
 * to analyze the transportation costs for communities in the transportation network.
 * 
 * @author yagizugurveren
 */
public class TransportationCostAnalysis {

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
     */
    public static void analyzeCosts(TransportationGraph transportationGraph, Map<Integer, List<Node>> communities) {
        System.out.println("Starting transportation cost analysis with pre-detected communities...");
        
        // Step 1: Print community information
        System.out.println("Found " + communities.size() + " communities for analysis");
        
        // Step 2: Analyze transportation costs using optimized analyzer
        System.out.println("\nAnalyzing transportation costs with optimized analyzer...");
        OptimizedTransportationCostAnalyzer analyzer = new OptimizedTransportationCostAnalyzer(transportationGraph);
        analyzer.analyzeTransportationCosts(communities);
        
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