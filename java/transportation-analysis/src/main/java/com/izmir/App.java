package com.izmir;

import com.izmir.transportation.CreateRoadNetwork;
import com.izmir.transportation.IzmirBayGraph;

import java.util.logging.Logger;

/**
 * Main application class for the Iztech Transportation Analysis project.
 * 1. Generation of random vertices based on population centers (IzmirBayGraph)
 * 2. Creation of a road network connecting these vertices (CreateRoadNetwork)
 * 3. Analysis and visualization of the transportation network graph
 * 
 * @author yagizugurveren
 */
public class App 
{
    static Logger LOGGER = Logger.getLogger(App.class.getName());

    public static void main( String[] args )
    {
        try {
            LOGGER.info("Starting Iztech Transportation Analysis...");
            // Step 1: Generate random points using IzmirBayGraph
            LOGGER.info("Step 1: Generating random vertices...");
            IzmirBayGraph.main(args);
            
            // Step 2: Create road network using the generated points
            LOGGER.info("Step 2: Creating road (edge) network...");
            CreateRoadNetwork.main(args);
            
            // Step 3: The transportation graph visualization is now handled automatically
            // by CreateRoadNetwork when it creates the paths
            
            LOGGER.info("Iztech Transportation Analysis completed successfully.");
        } catch (Exception e) {
            LOGGER.severe("Error during analysis: " + e.getMessage());
            e.printStackTrace();
        }
    }
}
