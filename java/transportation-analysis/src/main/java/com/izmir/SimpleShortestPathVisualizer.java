package com.izmir;

import java.awt.BasicStroke;
import java.awt.Color;
import java.awt.Dimension;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.Point;
import java.awt.RenderingHints;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.PriorityQueue;
import java.util.Random;
import java.util.Set;
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

/**
 * A standalone application for visualizing shortest paths between random nodes.
 * This implementation is self-contained and doesn't rely on external libraries.
 * 
 * This class:
 * 1. Generates random nodes
 * 2. Creates a graph with these nodes
 * 3. Calculates and visualizes shortest paths between selected nodes
 * 
 * @author davondeveloper
 */
public class SimpleShortestPathVisualizer {
    private static final Logger LOGGER = Logger.getLogger(SimpleShortestPathVisualizer.class.getName());
    
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
    
    // Simple graph types
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
    
    // Node class to represent a point
    public static class Node {
        private final int id;
        private final double x;
        private final double y;
        
        public Node(int id, double x, double y) {
            this.id = id;
            this.x = x;
            this.y = y;
        }
        
        public int getId() {
            return id;
        }
        
        public double getX() {
            return x;
        }
        
        public double getY() {
            return y;
        }
        
        public double distance(Node other) {
            return Math.sqrt(Math.pow(this.x - other.x, 2) + Math.pow(this.y - other.y, 2));
        }
        
        @Override
        public boolean equals(Object obj) {
            if (this == obj) return true;
            if (obj == null || getClass() != obj.getClass()) return false;
            Node node = (Node) obj;
            return id == node.id;
        }
        
        @Override
        public int hashCode() {
            return Integer.hashCode(id);
        }
        
        @Override
        public String toString() {
            return "Node{" + "id=" + id + ", x=" + x + ", y=" + y + '}';
        }
    }
    
    // Edge class to represent a connection between nodes
    public static class Edge {
        private final Node source;
        private final Node target;
        private final double weight;
        
        public Edge(Node source, Node target, double weight) {
            this.source = source;
            this.target = target;
            this.weight = weight;
        }
        
        public Node getSource() {
            return source;
        }
        
        public Node getTarget() {
            return target;
        }
        
        public double getWeight() {
            return weight;
        }
        
        @Override
        public boolean equals(Object obj) {
            if (this == obj) return true;
            if (obj == null || getClass() != obj.getClass()) return false;
            Edge edge = (Edge) obj;
            return (source.equals(edge.source) && target.equals(edge.target)) ||
                   (source.equals(edge.target) && target.equals(edge.source));
        }
        
        @Override
        public int hashCode() {
            return source.hashCode() + target.hashCode();
        }
        
        @Override
        public String toString() {
            return "Edge{" + "source=" + source.id + ", target=" + target.id + ", weight=" + weight + '}';
        }
    }
    
    // Simple graph implementation
    public static class SimpleGraph {
        private final Map<Node, List<Edge>> adjacencyList = new HashMap<>();
        private final Set<Edge> edges = new HashSet<>();
        private final Set<Node> nodes = new HashSet<>();
        
        public void addNode(Node node) {
            nodes.add(node);
            if (!adjacencyList.containsKey(node)) {
                adjacencyList.put(node, new ArrayList<>());
            }
        }
        
        public boolean addEdge(Node source, Node target, double weight) {
            // Check if nodes exist
            if (!nodes.contains(source) || !nodes.contains(target)) {
                return false;
            }
            
            // Create the edge
            Edge edge = new Edge(source, target, weight);
            
            // Check if edge already exists
            if (edges.contains(edge)) {
                return false;
            }
            
            // Add edge to set
            edges.add(edge);
            
            // Add to adjacency list
            adjacencyList.get(source).add(edge);
            adjacencyList.get(target).add(edge);
            
            return true;
        }
        
        public Set<Node> getNodes() {
            return Collections.unmodifiableSet(nodes);
        }
        
        public Set<Edge> getEdges() {
            return Collections.unmodifiableSet(edges);
        }
        
        public List<Edge> getEdges(Node node) {
            return adjacencyList.getOrDefault(node, Collections.emptyList());
        }
        
        public Node getOppositeNode(Node node, Edge edge) {
            if (edge.getSource().equals(node)) {
                return edge.getTarget();
            } else if (edge.getTarget().equals(node)) {
                return edge.getSource();
            } else {
                return null;
            }
        }
        
        public int getNodeCount() {
            return nodes.size();
        }
        
        public int getEdgeCount() {
            return edges.size();
        }
        
        public void clear() {
            nodes.clear();
            edges.clear();
            adjacencyList.clear();
        }
    }
    
    // Path class to represent a path in the graph
    public static class Path {
        private final List<Edge> edges = new ArrayList<>();
        private double weight = 0;
        
        public void addEdge(Edge edge) {
            edges.add(edge);
            weight += edge.getWeight();
        }
        
        public List<Edge> getEdges() {
            return Collections.unmodifiableList(edges);
        }
        
        public double getWeight() {
            return weight;
        }
    }
    
    // Dijkstra's algorithm for shortest path
    public static class DijkstraShortestPath {
        private final SimpleGraph graph;
        
        public DijkstraShortestPath(SimpleGraph graph) {
            this.graph = graph;
        }
        
        public Path getPath(Node source, Node target) {
            if (source.equals(target)) {
                return new Path();
            }
            
            // Initialize
            Map<Node, Double> distances = new HashMap<>();
            Map<Node, Edge> previous = new HashMap<>();
            PriorityQueue<Node> queue = new PriorityQueue<>(Comparator.comparingDouble(distances::get));
            Set<Node> visited = new HashSet<>();
            
            // Set initial distances
            for (Node node : graph.getNodes()) {
                distances.put(node, Double.POSITIVE_INFINITY);
            }
            distances.put(source, 0.0);
            queue.add(source);
            
            // Dijkstra's algorithm
            while (!queue.isEmpty()) {
                Node current = queue.poll();
                
                if (current.equals(target)) {
                    break;
                }
                
                if (visited.contains(current)) {
                    continue;
                }
                
                visited.add(current);
                
                double currentDistance = distances.get(current);
                
                for (Edge edge : graph.getEdges(current)) {
                    Node neighbor = graph.getOppositeNode(current, edge);
                    
                    if (visited.contains(neighbor)) {
                        continue;
                    }
                    
                    double newDistance = currentDistance + edge.getWeight();
                    
                    if (newDistance < distances.get(neighbor)) {
                        distances.put(neighbor, newDistance);
                        previous.put(neighbor, edge);
                        
                        // Update queue
                        queue.remove(neighbor);
                        queue.add(neighbor);
                    }
                }
            }
            
            // Build path
            if (!previous.containsKey(target)) {
                return null; // No path found
            }
            
            Path path = new Path();
            Node current = target;
            LinkedList<Edge> edgeStack = new LinkedList<>();
            
            while (previous.containsKey(current)) {
                Edge edge = previous.get(current);
                edgeStack.addFirst(edge);
                current = (edge.getSource().equals(current)) ? edge.getTarget() : edge.getSource();
            }
            
            for (Edge edge : edgeStack) {
                path.addEdge(edge);
            }
            
            return path;
        }
    }
    
    // Graph and node storage
    private SimpleGraph graph = new SimpleGraph();
    private List<Node> nodes = new ArrayList<>();
    private Node selectedSourceNode;
    private Node selectedTargetNode;
    private Path currentPath;
    private final Map<Node, Point> nodeToScreenCoordinates = new HashMap<>();
    private GraphType currentGraphType = GraphType.DISTANCE_LIMITED;
    private double maxDistanceThreshold = 0.2; // Default distance threshold
    
    /**
     * Main entry point for the application.
     */
    public static void main(String[] args) {
        SwingUtilities.invokeLater(() -> {
            try {
                SimpleShortestPathVisualizer visualizer = new SimpleShortestPathVisualizer();
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
    public SimpleShortestPathVisualizer() {
        generateRandomNodes();
        buildGraph();
    }
    
    /**
     * Generates random nodes.
     */
    private void generateRandomNodes() {
        try {
            LOGGER.info("Generating random nodes...");
            nodes.clear();
            
            // Generate random nodes
            Random random = new Random();
            for (int i = 0; i < NODE_COUNT; i++) {
                // Generate random coordinates (in range 0-1 for simplicity)
                double x = random.nextDouble();
                double y = random.nextDouble();
                
                Node node = new Node(i, x, y);
                nodes.add(node);
            }
            
            LOGGER.info("Generated " + nodes.size() + " nodes.");
            
            // Scale the nodes to fit the display and calculate screen coordinates
            scaleNodesToDisplay();
            
        } catch (Exception e) {
            LOGGER.log(Level.SEVERE, "Error generating nodes", e);
            throw new RuntimeException("Failed to generate nodes", e);
        }
    }
    
    /**
     * Scales the nodes to fit the display area and calculates screen coordinates.
     */
    private void scaleNodesToDisplay() {
        // Find min and max coordinates
        double minX = Double.MAX_VALUE, minY = Double.MAX_VALUE;
        double maxX = Double.MIN_VALUE, maxY = Double.MIN_VALUE;
        
        for (Node node : nodes) {
            minX = Math.min(minX, node.getX());
            minY = Math.min(minY, node.getY());
            maxX = Math.max(maxX, node.getX());
            maxY = Math.max(maxY, node.getY());
        }
        
        // Calculate scaling factors with margins
        int margin = 50;
        double xScale = (DISPLAY_WIDTH - 2 * margin) / (maxX - minX);
        double yScale = (DISPLAY_HEIGHT - 2 * margin) / (maxY - minY);
        
        // Calculate screen coordinates for each node
        nodeToScreenCoordinates.clear();
        for (Node node : nodes) {
            int screenX = (int) ((node.getX() - minX) * xScale) + margin;
            int screenY = DISPLAY_HEIGHT - ((int) ((node.getY() - minY) * yScale) + margin); // Y-axis is inverted in screen coordinates
            
            nodeToScreenCoordinates.put(node, new Point(screenX, screenY));
        }
    }
    
    /**
     * Builds the graph using the current graph type.
     */
    private void buildGraph() {
        LOGGER.info("Building graph using " + currentGraphType + " approach...");
        
        // Clear previous graph
        graph.clear();
        
        // Add all nodes to the graph
        for (Node node : nodes) {
            graph.addNode(node);
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
            Node node1 = nodes.get(i);
            
            for (int j = i + 1; j < nodes.size(); j++) {
                Node node2 = nodes.get(j);
                double distance = node1.distance(node2);
                
                boolean shouldConnect = false;
                if (currentGraphType == GraphType.COMPLETE) {
                    shouldConnect = true;
                } else if (currentGraphType == GraphType.DISTANCE_LIMITED) {
                    // Connect only if distance is below threshold
                    shouldConnect = distance <= (maxDistanceThreshold * maxDistance);
                }
                
                if (shouldConnect) {
                    graph.addEdge(node1, node2, distance);
                }
            }
        }
        
        LOGGER.info("Graph built with " + graph.getNodeCount() + " nodes and " + 
                    graph.getEdgeCount() + " edges.");
    }
    
    /**
     * Calculates the shortest path between two nodes using Dijkstra's algorithm.
     * 
     * @param source The source node
     * @param target The target node
     * @return The shortest path
     */
    private Path calculateShortestPath(Node source, Node target) {
        if (source == null || target == null) {
            return null;
        }
        
        try {
            // Use Dijkstra's algorithm to find the shortest path
            DijkstraShortestPath dijkstra = new DijkstraShortestPath(graph);
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
        JFrame frame = new JFrame("Simple Shortest Path Visualizer");
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
        
        // Create regenerate nodes button
        JButton regenerateButton = new JButton("Regenerate Nodes");
        regenerateButton.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                generateRandomNodes();
                buildGraph();
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
        controlPanel.add(regenerateButton);
        
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
            Node closestNode = findClosestNode(x, y);
            
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
        private Node findClosestNode(int x, int y) {
            double minDistance = Double.MAX_VALUE;
            Node closestNode = null;
            
            for (Node node : nodes) {
                Point screenCoord = nodeToScreenCoordinates.get(node);
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
            
            for (Edge edge : graph.getEdges()) {
                Point sourceCoord = nodeToScreenCoordinates.get(edge.getSource());
                Point targetCoord = nodeToScreenCoordinates.get(edge.getTarget());
                
                if (sourceCoord != null && targetCoord != null) {
                    g2d.drawLine(sourceCoord.x, sourceCoord.y, targetCoord.x, targetCoord.y);
                }
            }
            
            // Draw shortest path if available
            if (currentPath != null) {
                g2d.setColor(PATH_COLOR);
                g2d.setStroke(new BasicStroke(PATH_STROKE_WIDTH));
                
                List<Edge> pathEdges = currentPath.getEdges();
                for (Edge edge : pathEdges) {
                    Point sourceCoord = nodeToScreenCoordinates.get(edge.getSource());
                    Point targetCoord = nodeToScreenCoordinates.get(edge.getTarget());
                    
                    if (sourceCoord != null && targetCoord != null) {
                        g2d.drawLine(sourceCoord.x, sourceCoord.y, targetCoord.x, targetCoord.y);
                    }
                }
                
                // Draw path weight
                g2d.setColor(Color.BLACK);
                String pathInfo = String.format("Path length: %.4f", currentPath.getWeight());
                g2d.drawString(pathInfo, 10, 20);
            }
            
            // Draw nodes
            for (Node node : nodes) {
                Point coord = nodeToScreenCoordinates.get(node);
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
                    
                    // Draw node id
                    g2d.drawString(String.valueOf(node.getId()), coord.x - 5, coord.y - 5);
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