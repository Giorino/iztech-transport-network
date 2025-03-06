package com.izmir.transportation.persistence;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.SerializationFeature;
import com.fasterxml.jackson.databind.module.SimpleModule;
import com.izmir.transportation.TransportationGraph;
import com.izmir.transportation.helper.Edge;
import com.izmir.transportation.helper.Node;
import org.jgrapht.Graph;
import org.jgrapht.graph.DefaultWeightedEdge;
import org.locationtech.jts.geom.Coordinate;
import org.locationtech.jts.geom.GeometryFactory;
import org.locationtech.jts.geom.LineString;
import org.locationtech.jts.geom.Point;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.*;

/**
 * Service for persisting and loading transportation graph data.
 * This allows for separation of graph construction and clustering operations.
 */
public class GraphPersistenceService {
    private static final Logger LOGGER = LoggerFactory.getLogger(GraphPersistenceService.class);
    private static final String DATA_DIRECTORY = "graph_data";
    private static final GeometryFactory GEOMETRY_FACTORY = new GeometryFactory();
    private final ObjectMapper objectMapper;

    public GraphPersistenceService() {
        objectMapper = new ObjectMapper();
        objectMapper.enable(SerializationFeature.INDENT_OUTPUT);
        
        // Register custom serializers/deserializers for geometry objects
        SimpleModule module = new SimpleModule();
        module.addSerializer(Point.class, new PointSerializer());
        module.addDeserializer(Point.class, new PointDeserializer());
        module.addSerializer(LineString.class, new LineStringSerializer());
        module.addDeserializer(LineString.class, new LineStringDeserializer());
        objectMapper.registerModule(module);
        
        // Create data directory if it doesn't exist
        try {
            Files.createDirectories(Paths.get(DATA_DIRECTORY));
        } catch (IOException e) {
            LOGGER.error("Failed to create data directory", e);
        }
    }

    /**
     * Generate a filename based on the graph parameters
     * 
     * @param nodeCount Number of nodes in the graph
     * @param strategyName Name of the graph construction strategy used
     * @return Filename
     */
    public String generateFilename(int nodeCount, String strategyName) {
        return String.format("graph_%d_%s.json", nodeCount, 
                strategyName.toLowerCase().replace(" ", "_"));
    }

    /**
     * Save a transportation graph to a JSON file
     * 
     * @param graph The graph to save
     * @param nodeCount Number of nodes in the graph
     * @param strategyName Name of the graph construction strategy used
     * @return The path where the graph was saved
     * @throws IOException If there's an error writing the file
     */
    public String saveGraph(TransportationGraph graph, int nodeCount, String strategyName) throws IOException {
        String filename = generateFilename(nodeCount, strategyName);
        Path filePath = Paths.get(DATA_DIRECTORY, filename);
        
        // Convert the graph to a persistable format
        GraphData graphData = new GraphData();
        graphData.setNodeCount(nodeCount);
        graphData.setStrategyName(strategyName);
        
        // Extract all nodes
        Map<String, NodeData> nodeDataMap = new HashMap<>();
        for (Node node : graph.getPointToNode().values()) {
            NodeData nodeData = new NodeData();
            nodeData.setId(node.getId());
            nodeData.setX(node.getLocation().getX());
            nodeData.setY(node.getLocation().getY());
            nodeData.setPopulationCenter(node.isPopulationCenter());
            nodeData.setPopulationWeight(node.getPopulationWeight());
            nodeDataMap.put(node.getId(), nodeData);
        }
        graphData.setNodes(new ArrayList<>(nodeDataMap.values()));
        
        // Extract all edges
        List<EdgeData> edgeDataList = new ArrayList<>();
        for (Edge edge : graph.getEdgeMap().values()) {
            EdgeData edgeData = new EdgeData();
            edgeData.setId(edge.getId());
            edgeData.setSourceId(edge.getSource().getId());
            edgeData.setTargetId(edge.getTarget().getId());
            edgeData.setDistance(edge.getOriginalDistance());
            edgeData.setNormalizedWeight(edge.getNormalizedWeight());
            
            // Convert geometry to coordinate array
            Coordinate[] coordinates = edge.getGeometry().getCoordinates();
            double[][] coords = new double[coordinates.length][2];
            for (int i = 0; i < coordinates.length; i++) {
                coords[i][0] = coordinates[i].x;
                coords[i][1] = coordinates[i].y;
            }
            edgeData.setCoordinates(coords);
            
            edgeDataList.add(edgeData);
        }
        graphData.setEdges(edgeDataList);
        
        // Write to file
        objectMapper.writeValue(filePath.toFile(), graphData);
        LOGGER.info("Graph saved to {}", filePath);
        return filePath.toString();
    }

    /**
     * Check if a graph with the given parameters already exists
     * 
     * @param nodeCount Number of nodes in the graph
     * @param strategyName Name of the graph construction strategy
     * @return true if the graph exists, false otherwise
     */
    public boolean graphExists(int nodeCount, String strategyName) {
        String filename = generateFilename(nodeCount, strategyName);
        Path filePath = Paths.get(DATA_DIRECTORY, filename);
        return Files.exists(filePath);
    }

    /**
     * Load a graph from a JSON file
     * 
     * @param nodeCount Number of nodes in the graph
     * @param strategyName Name of the graph construction strategy
     * @return The loaded transportation graph, or null if the file doesn't exist
     * @throws IOException If there's an error reading the file
     */
    public TransportationGraph loadGraph(int nodeCount, String strategyName) throws IOException {
        String filename = generateFilename(nodeCount, strategyName);
        Path filePath = Paths.get(DATA_DIRECTORY, filename);
        
        if (!Files.exists(filePath)) {
            LOGGER.warn("Graph file not found: {}", filePath);
            return null;
        }
        
        // Read the graph data from file
        GraphData graphData = objectMapper.readValue(filePath.toFile(), GraphData.class);
        
        // Create a new transportation graph
        List<Point> originalPoints = new ArrayList<>();
        
        // Recreate all nodes
        Map<String, Node> idToNodeMap = new HashMap<>();
        for (NodeData nodeData : graphData.getNodes()) {
            Point point = GEOMETRY_FACTORY.createPoint(new Coordinate(nodeData.getX(), nodeData.getY()));
            originalPoints.add(point);
            
            Node node = new Node(
                nodeData.getId(), 
                point, 
                nodeData.isPopulationCenter(), 
                nodeData.getPopulationWeight()
            );
            idToNodeMap.put(node.getId(), node);
        }
        
        // Create the transportation graph
        TransportationGraph graph = new TransportationGraph(originalPoints);
        
        // Recreate all edges
        for (EdgeData edgeData : graphData.getEdges()) {
            Node source = idToNodeMap.get(edgeData.getSourceId());
            Node target = idToNodeMap.get(edgeData.getTargetId());
            
            // Recreate geometry
            Coordinate[] coordinates = new Coordinate[edgeData.getCoordinates().length];
            for (int i = 0; i < edgeData.getCoordinates().length; i++) {
                coordinates[i] = new Coordinate(
                    edgeData.getCoordinates()[i][0],
                    edgeData.getCoordinates()[i][1]
                );
            }
            LineString geometry = GEOMETRY_FACTORY.createLineString(coordinates);
            
            // Add the edge to the graph
            DefaultWeightedEdge edge = graph.addConnection(
                source.getLocation(), 
                target.getLocation(), 
                edgeData.getDistance()
            );
            
            // Update the edge with normalized weight (which will be recalculated anyway)
            graph.getEdgeMap().get(edge).normalizeWeight(edgeData.getDistance());
        }
        
        LOGGER.info("Graph loaded from {}", filePath);
        return graph;
    }

    /**
     * Get a list of all saved graphs
     * 
     * @return List of GraphInfo objects containing metadata about saved graphs
     */
    public List<GraphInfo> listSavedGraphs() {
        List<GraphInfo> result = new ArrayList<>();
        File dataDir = new File(DATA_DIRECTORY);
        
        if (!dataDir.exists() || !dataDir.isDirectory()) {
            return result;
        }
        
        File[] files = dataDir.listFiles((dir, name) -> name.startsWith("graph_") && name.endsWith(".json"));
        if (files == null) {
            return result;
        }
        
        for (File file : files) {
            try {
                GraphData graphData = objectMapper.readValue(file, GraphData.class);
                GraphInfo info = new GraphInfo();
                info.setFilename(file.getName());
                info.setNodeCount(graphData.getNodeCount());
                info.setStrategyName(graphData.getStrategyName());
                info.setEdgeCount(graphData.getEdges().size());
                info.setFileSize(file.length());
                info.setLastModified(new Date(file.lastModified()));
                result.add(info);
            } catch (IOException e) {
                LOGGER.error("Error reading graph file: {}", file.getName(), e);
            }
        }
        
        return result;
    }
} 