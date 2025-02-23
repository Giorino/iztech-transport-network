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
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

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
    private final double[][] matrix;
    private final List<Node> nodes;
    private final Map<Node, Integer> nodeIndices;
    private static final int NUM_THREADS = Runtime.getRuntime().availableProcessors(); // Use available CPU cores

    /**
     * Creates a new AffinityMatrix from a transportation graph using parallel processing.
     *
     * @param graph The transportation graph
     * @param edgeMap Mapping of graph edges to custom Edge objects
     */
    public AffinityMatrix(Graph<Node, DefaultWeightedEdge> graph, Map<DefaultWeightedEdge, Edge> edgeMap) {
        System.out.println("Creating affinity matrix using " + NUM_THREADS + " threads...");
        long startTime = System.currentTimeMillis();

        // First collect connected nodes
        Set<Node> connectedNodes = new HashSet<>();
        for (DefaultWeightedEdge graphEdge : graph.edgeSet()) {
            Edge edge = edgeMap.get(graphEdge);
            if (edge != null) {
                connectedNodes.add(edge.getSource());
                connectedNodes.add(edge.getTarget());
            }
        }
        
        nodes = new ArrayList<>(connectedNodes);
        nodeIndices = new HashMap<>();
        
        for (int i = 0; i < nodes.size(); i++) {
            nodeIndices.put(nodes.get(i), i);
        }

        matrix = new double[nodes.size()][nodes.size()];
        
        // Create a thread pool
        ExecutorService executor = Executors.newFixedThreadPool(NUM_THREADS);
        
        // Calculate chunk size for each thread
        int chunkSize = Math.max(1, graph.edgeSet().size() / NUM_THREADS);
        
        // Convert edge set to list for parallel processing
        List<DefaultWeightedEdge> edges = new ArrayList<>(graph.edgeSet());
        
        // Process edges in parallel
        for (int i = 0; i < edges.size(); i += chunkSize) {
            final int start = i;
            final int end = Math.min(start + chunkSize, edges.size());
            
            executor.submit(() -> {
                for (int j = start; j < end; j++) {
                    DefaultWeightedEdge graphEdge = edges.get(j);
                    Edge edge = edgeMap.get(graphEdge);
                    if (edge != null) {
                        int sourceIndex = nodeIndices.get(edge.getSource());
                        int targetIndex = nodeIndices.get(edge.getTarget());
                        
                        synchronized (matrix) {
                            matrix[sourceIndex][targetIndex] = edge.getNormalizedWeight();
                            matrix[targetIndex][sourceIndex] = edge.getNormalizedWeight();
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
        System.out.println("Affinity matrix created in " + (endTime - startTime) + " ms");
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
                    writer.write(String.format("%.2f", matrix[i][j]));
                    if (j < matrix[i].length - 1) {
                        writer.write(",");
                    }
                }
                writer.write("\n");
            }
        }
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