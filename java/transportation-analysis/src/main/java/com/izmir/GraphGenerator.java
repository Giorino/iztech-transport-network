package com.izmir;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.List;
import java.util.logging.Level;
import java.util.logging.Logger;

import org.locationtech.jts.geom.Point;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ArrayNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
import com.izmir.transportation.GraphConstructionService;
import com.izmir.transportation.IzmirBayGraph;
import com.izmir.transportation.TransportationGraph;

/**
 * Standalone class for generating transportation graphs using different strategies.
 * This class:
 * 1. Generates 50 nodes based on population centers (using IzmirBayGraph)
 * 2. Saves these nodes to a JSON file
 * 3. Creates four different graphs using the following strategies:
 *    - Complete Graph
 *    - Gabriel Graph
 *    - Delaunay Triangulation
 *    - K-Nearest Neighbors
 * 4. Visualizes and saves each graph
 * 
 * @author davondeveloper
 */
public class GraphGenerator {
    private static final Logger LOGGER = Logger.getLogger(GraphGenerator.class.getName());
    
    // Configuration properties
    private static final int NODE_COUNT = 25;
    private static final boolean USE_PARALLEL = true;
    private static final boolean VISUALIZE_GRAPH = true;
    private static final boolean SAVE_GRAPH = true;
    private static final int K_VALUE = 5; // K value for K-Nearest Neighbors
    
    // File paths
    private static final String NODES_JSON_FILE = "nodes.json";
    private static final String OUTPUT_DIR = "output";
    
    public static void main(String[] args) {
        try {
            LOGGER.info("Starting Graph Generator...");
            
            // Step 1: Generate random points using IzmirBayGraph
            LOGGER.info("Step 1: Generating " + NODE_COUNT + " random vertices...");
            List<Point> points = IzmirBayGraph.generatePoints(NODE_COUNT);
            LOGGER.info("Generated " + points.size() + " points.");
            
            // Step 2: Save points to JSON file
            LOGGER.info("Step 2: Saving points to JSON file...");
            savePointsToJson(points, NODES_JSON_FILE);
            LOGGER.info("Points saved to " + NODES_JSON_FILE);
            
            // Create output directory if it doesn't exist
            createOutputDirectory();
            
            // Step 3: Create and save graphs using different strategies
            LOGGER.info("Step 3: Creating graphs using different strategies...");
            
            // Create graphs - each one will display in its own window
            // Add a small delay between each to let the visualizations render properly
            
            LOGGER.info("Creating Complete Graph...");
            createGraph(points, GraphConstructionService.GraphStrategy.COMPLETE, "Complete Graph");
            Thread.sleep(3000); // Wait 3 seconds
            
            LOGGER.info("Creating Gabriel Graph...");
            createGraph(points, GraphConstructionService.GraphStrategy.GABRIEL, "Gabriel Graph");  
            Thread.sleep(3000); // Wait 3 seconds
            
            LOGGER.info("Creating Delaunay Triangulation...");
            createGraph(points, GraphConstructionService.GraphStrategy.DELAUNAY, "Delaunay Triangulation");
            Thread.sleep(3000); // Wait 3 seconds
            
            LOGGER.info("Creating K-Nearest Neighbors Graph...");
            createGraph(points, GraphConstructionService.GraphStrategy.K_NEAREST_NEIGHBORS, "K-Nearest Neighbors");
            
            LOGGER.info("Graph generation completed successfully.");
            
        } catch (Exception e) {
            LOGGER.log(Level.SEVERE, "Error during graph generation: " + e.getMessage(), e);
        }
    }
    
    /**
     * Saves the list of points to a JSON file.
     * 
     * @param points List of points to save
     * @param fileName Target JSON file name
     * @throws IOException If an error occurs during file writing
     */
    private static void savePointsToJson(List<Point> points, String fileName) throws IOException {
        ObjectMapper mapper = new ObjectMapper();
        ArrayNode pointsArray = mapper.createArrayNode();
        
        for (int i = 0; i < points.size(); i++) {
            Point point = points.get(i);
            ObjectNode pointNode = mapper.createObjectNode();
            pointNode.put("id", i);
            pointNode.put("x", point.getX());
            pointNode.put("y", point.getY());
            pointsArray.add(pointNode);
        }
        
        ObjectNode rootNode = mapper.createObjectNode();
        rootNode.put("count", points.size());
        rootNode.set("points", pointsArray);
        
        try (FileWriter writer = new FileWriter(fileName)) {
            writer.write(mapper.writerWithDefaultPrettyPrinter().writeValueAsString(rootNode));
        }
    }
    
    /**
     * Creates a graph using the specified strategy and visualizes it.
     * 
     * @param points List of points to use for graph construction
     * @param strategy Graph construction strategy to use
     * @param strategyName Name of the strategy (for logging and file names)
     * @throws Exception If an error occurs during graph construction
     */
    private static void createGraph(
            List<Point> points, 
            GraphConstructionService.GraphStrategy strategy,
            String strategyName) throws Exception {
        
        LOGGER.info("Creating " + strategyName + "...");
        
        GraphConstructionService graphService = new GraphConstructionService();
        
        // Set the graph construction method name in the TransportationGraph
        // This will appear in the visualization window title
        TransportationGraph graph = graphService.createGraph(
            points, strategy, K_VALUE, USE_PARALLEL, VISUALIZE_GRAPH, SAVE_GRAPH);
        
        // Set the graph construction method to our custom name
        graph.setGraphConstructionMethod(strategyName);
        
        LOGGER.info(strategyName + " created with " + graph.getEdgeCount() + " edges.");
    }
    
    /**
     * Creates the output directory if it doesn't exist.
     */
    private static void createOutputDirectory() {
        File outputDir = new File(OUTPUT_DIR);
        if (!outputDir.exists()) {
            if (outputDir.mkdirs()) {
                LOGGER.info("Created output directory: " + OUTPUT_DIR);
            } else {
                LOGGER.warning("Failed to create output directory: " + OUTPUT_DIR);
            }
        }
    }
} 