package com.izmir.analysis;

/**
 * Main runner class for transportation analysis.
 * Provides a command-line interface to run the transportation analysis.
 */
public class AnalysisRunner {
    
    public static void main(String[] args) {
        System.out.println("Starting Transportation Network Analysis...");
        
        // Run the transportation analyzer
        TransportationAnalyzer analyzer = new TransportationAnalyzer();
        analyzer.analyze();
        
        System.out.println("Analysis complete!");
    }
} 