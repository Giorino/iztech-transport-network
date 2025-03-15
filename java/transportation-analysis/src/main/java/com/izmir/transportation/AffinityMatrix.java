package com.izmir.transportation;

import java.awt.Color;
import java.awt.Font;
import java.awt.GridLayout;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.logging.Logger;

import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JPanel;
import javax.swing.JScrollPane;
import javax.swing.SwingConstants;
import javax.swing.SwingUtilities;
import javax.swing.border.LineBorder;

import org.jgrapht.Graph;
import org.jgrapht.graph.DefaultWeightedEdge;

import com.izmir.transportation.helper.Edge;
import com.izmir.transportation.helper.Node;

/**
 * A class for creating and visualizing the affinity matrix of the transportation network.
 * The affinity matrix represents the edge weights between nodes, with zeros on the diagonal
 * since nodes cannot connect to themselves.
 *
 * @author yagizugurveren
 */
public class AffinityMatrix {
    private static final Logger LOGGER = Logger.getLogger(AffinityMatrix.class.getName());
    private final double[][] matrix;
    private final List<Node> nodes;
    private final Map<Node, Integer> nodeIndices;
    private static final int NUM_THREADS = Runtime.getRuntime().availableProcessors(); // Use available CPU cores
    private int nonZeroEntries = 0;

    /**
     * Creates a new AffinityMatrix from a transportation graph using parallel processing.
     *
     * @param graph The transportation graph
     * @param edgeMap Mapping of graph edges to custom Edge objects
     */
    public AffinityMatrix(Graph<Node, DefaultWeightedEdge> graph, Map<DefaultWeightedEdge, Edge> edgeMap) {
        LOGGER.info("Creating affinity matrix using " + NUM_THREADS + " threads...");
        System.out.println("Creating affinity matrix using " + NUM_THREADS + " threads...");
        long startTime = System.currentTimeMillis();

        // First collect all nodes from the graph (not just connected ones)
        nodes = new ArrayList<>(graph.vertexSet());
        nodeIndices = new HashMap<>();
        
        LOGGER.info("Total nodes in graph: " + nodes.size());
        System.out.println("Total nodes in graph: " + nodes.size());
        
        for (int i = 0; i < nodes.size(); i++) {
            nodeIndices.put(nodes.get(i), i);
        }

        matrix = new double[nodes.size()][nodes.size()];
        
        // First ensure all edges are normalized
        double maxDistance = 0.0;
        for (DefaultWeightedEdge graphEdge : graph.edgeSet()) {
            Edge edge = edgeMap.get(graphEdge);
            if (edge != null) {
                maxDistance = Math.max(maxDistance, edge.getOriginalDistance());
            }
        }
        
        LOGGER.info("Maximum edge distance: " + maxDistance);
        System.out.println("Maximum edge distance: " + maxDistance);
        
        // Now normalize all edges
        if (maxDistance > 0) {
            for (Map.Entry<DefaultWeightedEdge, Edge> entry : edgeMap.entrySet()) {
                Edge edge = entry.getValue();
                edge.normalizeWeight(maxDistance);
            }
        }
        
        // Create a thread pool
        ExecutorService executor = Executors.newFixedThreadPool(NUM_THREADS);
        
        // Store the final value of maxDistance for lambda use
        final double finalMaxDistance = maxDistance;
        
        // Fill matrix directly from graph edge weights (not from edgeMap)
        for (int i = 0; i < nodes.size(); i++) {
            final int rowIndex = i;
            
            // Store effectively final copies of objects needed in lambda
            final Map<DefaultWeightedEdge, Edge> finalEdgeMap = edgeMap;
            final Graph<Node, DefaultWeightedEdge> finalGraph = graph;
            
            executor.submit(() -> {
                Node sourceNode = nodes.get(rowIndex);
                for (int j = 0; j < nodes.size(); j++) {
                    Node targetNode = nodes.get(j);
                    
                    // Skip self-connections
                    if (sourceNode.equals(targetNode)) {
                        continue;
                    }
                    
                    // Get the edge from the graph
                    DefaultWeightedEdge graphEdge = finalGraph.getEdge(sourceNode, targetNode);
                    if (graphEdge != null) {
                        double weight = finalGraph.getEdgeWeight(graphEdge);
                        
                        // Alternative: Get weight from edgeMap if graph weight is not available
                        if (weight <= 0 && finalEdgeMap.containsKey(graphEdge)) {
                            Edge edge = finalEdgeMap.get(graphEdge);
                            if (edge != null) {
                                weight = edge.getNormalizedWeight();
                                if (weight <= 0) {
                                    // Fall back to raw distance if normalized weight is not available
                                    weight = (finalMaxDistance > 0) ? edge.getOriginalDistance() / finalMaxDistance : 0;
                                }
                            }
                        }
                        
                        // Set the weight in the matrix
                        synchronized (matrix) {
                            matrix[rowIndex][j] = weight;
                            if (weight > 0) {
                                nonZeroEntries++;
                            }
                        }
                    }
                }
            });
        }
        
        // Shutdown the executor and wait for all tasks to complete
        executor.shutdown();
        try {
            executor.awaitTermination(1, TimeUnit.MINUTES);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            System.err.println("Matrix creation was interrupted: " + e.getMessage());
        }

        long endTime = System.currentTimeMillis();
        LOGGER.info("Affinity matrix created in " + (endTime - startTime) + " ms with " + 
                   nonZeroEntries + " non-zero entries out of " + (nodes.size() * nodes.size()) + " total entries.");
        System.out.println("Affinity matrix created in " + (endTime - startTime) + " ms with " + 
                          nonZeroEntries + " non-zero entries out of " + (nodes.size() * nodes.size()) + " total entries.");
        
        // Validate matrix has some values
        boolean hasValues = false;
        for (int i = 0; i < matrix.length && !hasValues; i++) {
            for (int j = 0; j < matrix[i].length; j++) {
                if (matrix[i][j] > 0) {
                    hasValues = true;
                    break;
                }
            }
        }
        
        if (!hasValues) {
            LOGGER.warning("WARNING: Affinity matrix has no positive values! This will result in a CSV full of zeros.");
            System.out.println("WARNING: Affinity matrix has no positive values! This will result in a CSV full of zeros.");
        }
    }

    /**
     * Saves the affinity matrix to a CSV file.
     *
     * @param filename The name of the file to save to
     * @throws IOException If there is an error writing the file
     */
    public void saveToCSV(String filename) throws IOException {
        try (FileWriter writer = new FileWriter(filename)) {
            // Write header with node IDs
            for (int i = 0; i < nodes.size(); i++) {
                writer.write(nodes.get(i).getId());
                if (i < nodes.size() - 1) {
                    writer.write(",");
                }
            }
            writer.write("\n");

            // Write matrix data
            for (int i = 0; i < matrix.length; i++) {
                for (int j = 0; j < matrix[i].length; j++) {
                    writer.write(String.format("%.4f", matrix[i][j]));
                    if (j < matrix[i].length - 1) {
                        writer.write(",");
                    }
                }
                writer.write("\n");
            }
        }
        
        LOGGER.info("Saved affinity matrix to " + filename);
        System.out.println("Saved affinity matrix to " + filename);
    }

    /**
     * Displays the CSV file in a grid format.
     *
     * @param csvFile The CSV file to display
     * @throws IOException If there is an error reading the file
     */
    public static void displayCSV(String csvFile) throws IOException {
        // Wait for the file to be fully written
        File file = new File(csvFile);
        int maxAttempts = 10;
        int attempts = 0;
        while (!file.exists() && attempts < maxAttempts) {
            try {
                Thread.sleep(100); // Wait 100ms
                attempts++;
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                throw new IOException("Interrupted while waiting for file");
            }
        }

        if (!file.exists()) {
            throw new IOException("CSV file not found: " + csvFile);
        }

        if (file.length() == 0) {
            throw new IOException("CSV file is empty: " + csvFile);
        }

        List<String> headers = new ArrayList<>();
        List<List<String>> data = new ArrayList<>();
        
        // Read the CSV file
        try (BufferedReader reader = new BufferedReader(new FileReader(csvFile))) {
            String headerLine = reader.readLine();
            if (headerLine == null) {
                throw new IOException("CSV file has no header line");
            }
            
            String[] headerArray = headerLine.split(",");
            headers.addAll(List.of(headerArray));
            
            if (headers.isEmpty()) {
                throw new IOException("No headers found in CSV file");
            }
            
            String line;
            while ((line = reader.readLine()) != null) {
                String[] values = line.split(",");
                if (values.length != headers.size()) {
                    throw new IOException("Inconsistent number of columns in CSV file");
                }
                data.add(List.of(values));
            }
            
            if (data.isEmpty()) {
                throw new IOException("No data found in CSV file");
            }
        }

        // Create and show the frame in the Event Dispatch Thread
        SwingUtilities.invokeLater(() -> {
            try {
                JFrame frame = new JFrame("Affinity Matrix");
                frame.setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);
                
                JPanel mainPanel = new JPanel(new GridLayout(data.size() + 1, headers.size() + 1));
                mainPanel.setBackground(Color.WHITE);
                
                // Add empty cell in top-left corner
                mainPanel.add(createCell(""));
                
                // Add headers
                for (String header : headers) {
                    mainPanel.add(createCell(header));
                }
                
                // Add row headers and data
                for (int i = 0; i < data.size(); i++) {
                    mainPanel.add(createCell(headers.get(i))); // Row header
                    List<String> row = data.get(i);
                    for (String value : row) {
                        mainPanel.add(createCell(value));
                    }
                }
                
                JScrollPane scrollPane = new JScrollPane(mainPanel);
                frame.add(scrollPane);
                
                frame.setSize(800, 600);
                frame.setLocationRelativeTo(null);
                frame.setVisible(true);
                
                System.out.println("Affinity matrix visualization displayed");
            } catch (Exception e) {
                System.err.println("Error displaying affinity matrix: " + e.getMessage());
                e.printStackTrace();
            }
        });
    }
    
    private static JLabel createCell(String text) {
        JLabel label = new JLabel(text, SwingConstants.CENTER);
        label.setBorder(new LineBorder(Color.BLACK));
        label.setOpaque(true);
        label.setBackground(Color.WHITE);
        label.setFont(new Font("Monospaced", Font.PLAIN, 12));
        return label;
    }

    /**
     * Gets the affinity matrix.
     *
     * @return The affinity matrix as a 2D array of doubles
     */
    public double[][] getMatrix() {
        return matrix;
    }
} 