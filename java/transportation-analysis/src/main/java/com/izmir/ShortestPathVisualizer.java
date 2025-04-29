package com.izmir;

import java.awt.BasicStroke;
import java.awt.Color;
import java.awt.Dimension;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.RenderingHints;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.logging.Level;
import java.util.logging.Logger;

import javax.swing.JButton;
import javax.swing.JFrame;
import javax.swing.JOptionPane;
import javax.swing.JPanel;
import javax.swing.JSlider;
import javax.swing.SwingUtilities;
import javax.swing.event.ChangeEvent;
import javax.swing.event.ChangeListener;

import org.jgrapht.Graph;
import org.jgrapht.GraphPath;
import org.jgrapht.alg.interfaces.ShortestPathAlgorithm;
import org.jgrapht.alg.shortestpath.DijkstraShortestPath;
import org.jgrapht.graph.DefaultWeightedEdge;
import org.jgrapht.graph.SimpleWeightedGraph;
import org.locationtech.jts.geom.Coordinate;
import org.locationtech.jts.geom.GeometryFactory;
import org.locationtech.jts.geom.Point;

import com.izmir.transportation.IzmirBayGraph;

/**
 * A standalone application for visualizing shortest paths between random nodes.
 * 
 * This class:
 * 1. Generates random nodes
 * 2. Creates a graph with these nodes 
 * 3. Calculates and visualizes shortest paths between selected nodes
 * 
 * @author davondeveloper
 */
public class ShortestPathVisualizer {
    private static final Logger LOGGER = Logger.getLogger(ShortestPathVisualizer.class.getName());
    
    // Configuration properties
    private static final int NODE_COUNT = 100;
    private static final int DISPLAY_WIDTH = 800;
    private static final int DISPLAY_HEIGHT = 600;
    private static final int NODE_SIZE = 10;
    private static final Color NODE_COLOR = Color.BLUE;
    private static final Color EDGE_COLOR = Color.LIGHT_GRAY;
    private static final Color PATH_COLOR = Color.RED;
    private static final Color SELECTED_NODE_COLOR = Color.GREEN;
    private static final int PATH_STROKE_WIDTH = 3;
    
    // Graph types
    public enum GraphType {
        COMPLETE("Complete Graph"),
        DISTANCE_LIMITED("Distance Limited");
        
        private final String displayName;
        
        GraphType(String displayName) {
            this.displayName = displayName;
        }
        
        @Override
        public String toString() {
            return displayName;
        }
    }
    
    // Graph and node storage
    private Graph<Point, DefaultWeightedEdge> graph;
    private List<Point> nodes;
    private final GeometryFactory geometryFactory;
    private Point selectedSourceNode;
    private Point selectedTargetNode;
    private GraphPath<Point, DefaultWeightedEdge> currentPath;
    private final Map<Point, java.awt.Point> pointToScreenCoordinates;
    private GraphType currentGraphType = GraphType.COMPLETE;
    private double maxDistanceThreshold = 0.1; // Default distance threshold
    
    /**
     * Main entry point for the application.
     */
    public static void main(String[] args) {
        SwingUtilities.invokeLater(() -> {
            try {
                ShortestPathVisualizer visualizer = new ShortestPathVisualizer();
                visualizer.createAndShowGUI();
            } catch (Exception e) {
                LOGGER.log(Level.SEVERE, "Error starting application", e);
                JOptionPane.showMessageDialog(null, 
                    "Error starting application: " + e.getMessage(), 
                    "Error", JOptionPane.ERROR_MESSAGE);
            }
        });
    }
    
    /**
     * Constructor initializes the visualizer.
     */
    public ShortestPathVisualizer() {
        geometryFactory = new GeometryFactory();
        pointToScreenCoordinates = new HashMap<>();
        generateRandomNodes();
        buildGraph();
    }
    
    /**
     * Generates random nodes using IzmirBayGraph or a pure random approach.
     */
    private void generateRandomNodes() {
        try {
            LOGGER.info("Generating random nodes...");
            
            // Try to use IzmirBayGraph for more realistic points based on population centers
            try {
                nodes = IzmirBayGraph.generatePoints(NODE_COUNT);
                LOGGER.info("Generated " + nodes.size() + " nodes using IzmirBayGraph.");
            } catch (Exception e) {
                LOGGER.warning("Could not generate nodes using IzmirBayGraph, using pure random approach: " + e.getMessage());
                // Fallback to pure random approach
                generatePureRandomNodes();
            }
            
            // If IzmirBayGraph returned fewer nodes than requested, fill with random nodes
            if (nodes.size() < NODE_COUNT) {
                int remaining = NODE_COUNT - nodes.size();
                LOGGER.info("IzmirBayGraph returned fewer nodes than requested, adding " + remaining + " random nodes.");
                nodes.addAll(generatePureRandomNodes(remaining));
            }
            
            // Scale the nodes to fit the display and calculate screen coordinates
            scaleNodesToDisplay();
            
        } catch (Exception e) {
            LOGGER.log(Level.SEVERE, "Error generating nodes", e);
            throw new RuntimeException("Failed to generate nodes", e);
        }
    }
    
    /**
     * Generates pure random nodes without using IzmirBayGraph.
     */
    private void generatePureRandomNodes() {
        nodes = generatePureRandomNodes(NODE_COUNT);
    }
    
    /**
     * Generates a specified number of pure random nodes.
     * 
     * @param count Number of nodes to generate
     * @return List of randomly generated points
     */
    private List<Point> generatePureRandomNodes(int count) {
        List<Point> randomNodes = new ArrayList<>();
        Random random = new Random();
        
        for (int i = 0; i < count; i++) {
            // Generate random coordinates (longitude and latitude)
            double longitude = 27.0 + random.nextDouble() * 0.5; // Roughly Izmir region
            double latitude = 38.3 + random.nextDouble() * 0.5;
            
            Point point = geometryFactory.createPoint(new Coordinate(longitude, latitude));
            randomNodes.add(point);
        }
        
        return randomNodes;
    }
    
    /**
     * Scales the nodes to fit the display area and calculates screen coordinates.
     */
    private void scaleNodesToDisplay() {
        // Find min and max coordinates
        double minX = Double.MAX_VALUE, minY = Double.MAX_VALUE;
        double maxX = Double.MIN_VALUE, maxY = Double.MIN_VALUE;
        
        for (Point point : nodes) {
            minX = Math.min(minX, point.getX());
            minY = Math.min(minY, point.getY());
            maxX = Math.max(maxX, point.getX());
            maxY = Math.max(maxY, point.getY());
        }
        
        // Calculate scaling factors with margins
        int margin = 50;
        double xScale = (DISPLAY_WIDTH - 2 * margin) / (maxX - minX);
        double yScale = (DISPLAY_HEIGHT - 2 * margin) / (maxY - minY);
        
        // Calculate screen coordinates for each node
        pointToScreenCoordinates.clear();
        for (Point point : nodes) {
            int screenX = (int) ((point.getX() - minX) * xScale) + margin;
            int screenY = DISPLAY_HEIGHT - ((int) ((point.getY() - minY) * yScale) + margin); // Y-axis is inverted in screen coordinates
            
            pointToScreenCoordinates.put(point, new java.awt.Point(screenX, screenY));
        }
    }
    
    /**
     * Builds the graph using the current graph type.
     */
    private void buildGraph() {
        LOGGER.info("Building graph using " + currentGraphType + " approach...");
        
        // Create a new graph
        graph = new SimpleWeightedGraph<>(DefaultWeightedEdge.class);
        
        // Add all nodes to the graph
        for (Point point : nodes) {
            graph.addVertex(point);
        }
        
        // Find maximum possible distance to normalize the distance threshold
        double maxDistance = 0;
        for (int i = 0; i < nodes.size(); i++) {
            for (int j = i + 1; j < nodes.size(); j++) {
                maxDistance = Math.max(maxDistance, nodes.get(i).distance(nodes.get(j)));
            }
        }
        
        // Add edges based on the graph type
        for (int i = 0; i < nodes.size(); i++) {
            Point p1 = nodes.get(i);
            
            for (int j = i + 1; j < nodes.size(); j++) {
                Point p2 = nodes.get(j);
                double distance = p1.distance(p2);
                
                boolean shouldConnect = false;
                if (currentGraphType == GraphType.COMPLETE) {
                    shouldConnect = true;
                } else if (currentGraphType == GraphType.DISTANCE_LIMITED) {
                    // Connect only if distance is below threshold
                    shouldConnect = distance <= (maxDistanceThreshold * maxDistance);
                }
                
                if (shouldConnect) {
                    DefaultWeightedEdge edge = graph.addEdge(p1, p2);
                    if (edge != null) {
                        graph.setEdgeWeight(edge, distance);
                    }
                }
            }
        }
        
        LOGGER.info("Graph built with " + graph.vertexSet().size() + " nodes and " + 
                    graph.edgeSet().size() + " edges.");
    }
    
    /**
     * Calculates the shortest path between two nodes using Dijkstra's algorithm.
     * 
     * @param source The source node
     * @param target The target node
     * @return The shortest path as a GraphPath object
     */
    private GraphPath<Point, DefaultWeightedEdge> calculateShortestPath(Point source, Point target) {
        if (source == null || target == null) {
            return null;
        }
        
        try {
            // Use Dijkstra's algorithm to find the shortest path
            ShortestPathAlgorithm<Point, DefaultWeightedEdge> dijkstra = 
                    new DijkstraShortestPath<>(graph);
            return dijkstra.getPath(source, target);
        } catch (Exception e) {
            LOGGER.log(Level.WARNING, "Error calculating shortest path", e);
            return null;
        }
    }
    
    /**
     * Creates and shows the GUI for the application.
     */
    private void createAndShowGUI() {
        JFrame frame = new JFrame("Shortest Path Visualizer");
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        
        // Create visualization panel
        VisualizationPanel visualizationPanel = new VisualizationPanel();
        visualizationPanel.setPreferredSize(new Dimension(DISPLAY_WIDTH, DISPLAY_HEIGHT));
        
        // Create control panel with buttons
        JPanel controlPanel = new JPanel();
        
        // Create graph type selection button
        JButton graphTypeButton = new JButton("Graph Type: " + currentGraphType);
        graphTypeButton.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                // Toggle between graph types
                if (currentGraphType == GraphType.COMPLETE) {
                    currentGraphType = GraphType.DISTANCE_LIMITED;
                } else {
                    currentGraphType = GraphType.COMPLETE;
                }
                graphTypeButton.setText("Graph Type: " + currentGraphType);
                
                // Rebuild the graph with the new type
                buildGraph();
                currentPath = null; // Clear current path
                visualizationPanel.repaint();
            }
        });
        
        // Create a slider for distance threshold (for DISTANCE_LIMITED)
        JSlider distanceSlider = new JSlider(1, 100, (int)(maxDistanceThreshold * 100));
        distanceSlider.setPreferredSize(new Dimension(150, distanceSlider.getPreferredSize().height));
        distanceSlider.addChangeListener(new ChangeListener() {
            @Override
            public void stateChanged(ChangeEvent e) {
                if (!distanceSlider.getValueIsAdjusting()) {
                    maxDistanceThreshold = distanceSlider.getValue() / 100.0;
                    
                    // Rebuild the graph if using distance limited type
                    if (currentGraphType == GraphType.DISTANCE_LIMITED) {
                        buildGraph();
                        currentPath = null; // Clear current path
                        visualizationPanel.repaint();
                    }
                }
            }
        });
        
        // Create reset selection button
        JButton resetButton = new JButton("Reset Selection");
        resetButton.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                selectedSourceNode = null;
                selectedTargetNode = null;
                currentPath = null;
                visualizationPanel.repaint();
            }
        });
        
        // Add controls to control panel
        controlPanel.add(graphTypeButton);
        controlPanel.add(distanceSlider);
        controlPanel.add(resetButton);
        
        // Add panels to frame
        frame.add(visualizationPanel, java.awt.BorderLayout.CENTER);
        frame.add(controlPanel, java.awt.BorderLayout.SOUTH);
        
        frame.pack();
        frame.setLocationRelativeTo(null);
        frame.setVisible(true);
    }
    
    /**
     * Panel for visualizing the graph and shortest paths.
     */
    private class VisualizationPanel extends JPanel {
        private static final long serialVersionUID = 1L;
        
        public VisualizationPanel() {
            setBackground(Color.WHITE);
            
            // Add mouse listener to select nodes
            addMouseListener(new java.awt.event.MouseAdapter() {
                @Override
                public void mouseClicked(java.awt.event.MouseEvent e) {
                    handleMouseClick(e.getX(), e.getY());
                }
            });
        }
        
        /**
         * Handles mouse clicks to select source and target nodes.
         * 
         * @param x The x-coordinate of the click
         * @param y The y-coordinate of the click
         */
        private void handleMouseClick(int x, int y) {
            // Find the closest node to the click
            Point closestNode = findClosestNode(x, y);
            
            if (closestNode != null) {
                // If no source node is selected, this is the source
                if (selectedSourceNode == null) {
                    selectedSourceNode = closestNode;
                    currentPath = null; // Clear current path
                } 
                // If source is already selected but not target, this is the target
                else if (selectedTargetNode == null) {
                    // Don't allow selecting the same node as source and target
                    if (!closestNode.equals(selectedSourceNode)) {
                        selectedTargetNode = closestNode;
                        
                        // Calculate shortest path
                        currentPath = calculateShortestPath(selectedSourceNode, selectedTargetNode);
                        
                        if (currentPath == null) {
                            LOGGER.warning("No path found between selected nodes.");
                            JOptionPane.showMessageDialog(getParent(), 
                                    "No path found between selected nodes.", 
                                    "No Path", JOptionPane.INFORMATION_MESSAGE);
                        } else {
                            double pathWeight = currentPath.getWeight();
                            LOGGER.info("Path found with weight: " + pathWeight);
                        }
                    }
                } 
                // If both are already selected, start over with this as the source
                else {
                    selectedSourceNode = closestNode;
                    selectedTargetNode = null;
                    currentPath = null;
                }
                
                // Redraw the panel
                repaint();
            }
        }
        
        /**
         * Finds the closest node to a given screen coordinate.
         * 
         * @param x The x-coordinate
         * @param y The y-coordinate
         * @return The closest node
         */
        private Point findClosestNode(int x, int y) {
            double minDistance = Double.MAX_VALUE;
            Point closestNode = null;
            
            for (Point node : nodes) {
                java.awt.Point screenCoord = pointToScreenCoordinates.get(node);
                if (screenCoord != null) {
                    double distance = Math.hypot(x - screenCoord.x, y - screenCoord.y);
                    if (distance < minDistance && distance < NODE_SIZE * 2) {
                        minDistance = distance;
                        closestNode = node;
                    }
                }
            }
            
            return closestNode;
        }
        
        @Override
        protected void paintComponent(Graphics g) {
            super.paintComponent(g);
            
            Graphics2D g2d = (Graphics2D) g.create();
            g2d.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
            
            // Draw edges
            g2d.setColor(EDGE_COLOR);
            g2d.setStroke(new BasicStroke(1));
            
            for (DefaultWeightedEdge edge : graph.edgeSet()) {
                Point source = graph.getEdgeSource(edge);
                Point target = graph.getEdgeTarget(edge);
                
                java.awt.Point sourceCoord = pointToScreenCoordinates.get(source);
                java.awt.Point targetCoord = pointToScreenCoordinates.get(target);
                
                if (sourceCoord != null && targetCoord != null) {
                    g2d.drawLine(sourceCoord.x, sourceCoord.y, targetCoord.x, targetCoord.y);
                }
            }
            
            // Draw shortest path if available
            if (currentPath != null) {
                g2d.setColor(PATH_COLOR);
                g2d.setStroke(new BasicStroke(PATH_STROKE_WIDTH));
                
                List<DefaultWeightedEdge> pathEdges = currentPath.getEdgeList();
                for (DefaultWeightedEdge edge : pathEdges) {
                    Point source = graph.getEdgeSource(edge);
                    Point target = graph.getEdgeTarget(edge);
                    
                    java.awt.Point sourceCoord = pointToScreenCoordinates.get(source);
                    java.awt.Point targetCoord = pointToScreenCoordinates.get(target);
                    
                    if (sourceCoord != null && targetCoord != null) {
                        g2d.drawLine(sourceCoord.x, sourceCoord.y, targetCoord.x, targetCoord.y);
                    }
                }
                
                // Draw path weight
                g2d.setColor(Color.BLACK);
                String pathInfo = String.format("Path length: %.2f", currentPath.getWeight());
                g2d.drawString(pathInfo, 10, 20);
            }
            
            // Draw nodes
            for (Point node : nodes) {
                java.awt.Point coord = pointToScreenCoordinates.get(node);
                if (coord != null) {
                    // Select color based on whether the node is selected
                    if (node.equals(selectedSourceNode)) {
                        g2d.setColor(SELECTED_NODE_COLOR);
                    } else if (node.equals(selectedTargetNode)) {
                        g2d.setColor(PATH_COLOR);
                    } else {
                        g2d.setColor(NODE_COLOR);
                    }
                    
                    // Draw the node
                    g2d.fillOval(coord.x - NODE_SIZE/2, coord.y - NODE_SIZE/2, NODE_SIZE, NODE_SIZE);
                    g2d.setColor(Color.BLACK);
                    g2d.drawOval(coord.x - NODE_SIZE/2, coord.y - NODE_SIZE/2, NODE_SIZE, NODE_SIZE);
                }
            }
            
            // Draw instructions
            g2d.setColor(Color.BLACK);
            if (selectedSourceNode == null) {
                g2d.drawString("Click to select source node", 10, DISPLAY_HEIGHT - 10);
            } else if (selectedTargetNode == null) {
                g2d.drawString("Click to select target node", 10, DISPLAY_HEIGHT - 10);
            } else {
                g2d.drawString("Click to select new source node", 10, DISPLAY_HEIGHT - 10);
            }
            
            g2d.dispose();
        }
    }
} 