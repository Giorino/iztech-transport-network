package com.izmir;

import com.izmir.transportation.CreateRoadNetwork;
import com.izmir.transportation.IzmirBayGraph;

/**
 * Main application class for the Iztech Transportation Analysis project.
 * This class coordinates the execution of three main processes:
 * 1. Generation of random vertices based on population centers (IzmirBayGraph)
 * 2. Creation of a road network connecting these vertices (CreateRoadNetwork)
 * 3. Analysis and visualization of the transportation network graph
 * 
 * @author yagizugurveren
 */
public class App 
{
    /**
     * The main entry point of the application.
     * Executes the transportation analysis in three steps:
     * 1. Generates random vertices using population-weighted distribution
     * 2. Creates a road network by connecting these vertices based on OpenStreetMap data
     * 3. Visualizes the resulting transportation network graph with weighted edges
     *
     * @param args Command line arguments (not used in current implementation)
     */
    public static void main( String[] args )
    {
        try {
            System.out.println("Starting Iztech Transportation Analysis...");
            
            // Step 1: Generate random points using IzmirBayGraph
            System.out.println("\nStep 1: Generating random vertices...");
            IzmirBayGraph.main(args);
            
            // Step 2: Create road network using the generated points
            System.out.println("\nStep 2: Creating road (edge) network...");
            CreateRoadNetwork.main(args);
            
            // Step 3: The transportation graph visualization is now handled automatically
            // by CreateRoadNetwork when it creates the paths
            
            System.out.println("\nAnalysis completed successfully!");
        } catch (Exception e) {
            System.err.println("Error during analysis: " + e.getMessage());
            e.printStackTrace();
        }
    }
}
