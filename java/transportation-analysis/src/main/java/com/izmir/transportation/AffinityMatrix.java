package com.izmir.transportation;

import java.awt.Color;
import java.awt.Font;
import java.awt.GridLayout;
import java.io.BufferedReader;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JPanel;
import javax.swing.JScrollPane;
import javax.swing.SwingConstants;
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

    /**
     * Creates a new AffinityMatrix from a transportation graph.
     *
     * @param graph The transportation graph
     * @param edgeMap Mapping of graph edges to custom Edge objects
     */
    public AffinityMatrix(Graph<Node, DefaultWeightedEdge> graph, Map<DefaultWeightedEdge, Edge> edgeMap) {

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
        
        for (DefaultWeightedEdge graphEdge : graph.edgeSet()) {
            Edge edge = edgeMap.get(graphEdge);
            if (edge != null) {
                int sourceIndex = nodeIndices.get(edge.getSource());
                int targetIndex = nodeIndices.get(edge.getTarget());
                
                matrix[sourceIndex][targetIndex] = edge.getNormalizedWeight();
                matrix[targetIndex][sourceIndex] = edge.getNormalizedWeight();
            }
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

            for (int i = 0; i < nodes.size(); i++) {
                writer.write(nodes.get(i).getId());
                if (i < nodes.size() - 1) {
                    writer.write(",");
                }
            }
            writer.write("\n");

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
        List<String> headers = new ArrayList<>();
        List<List<String>> data = new ArrayList<>();
        
        try (BufferedReader reader = new BufferedReader(new FileReader(csvFile))) {
            String headerLine = reader.readLine();
            if (headerLine != null) {
                String[] headerArray = headerLine.split(",");
                headers.addAll(List.of(headerArray));
            }
            
            String line;
            while ((line = reader.readLine()) != null) {
                String[] values = line.split(",");
                data.add(List.of(values));
            }
        }
        
        JFrame frame = new JFrame("Affinity Matrix");
        frame.setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);
        
        JPanel mainPanel = new JPanel(new GridLayout(data.size() + 1, headers.size() + 1));
        mainPanel.setBackground(Color.WHITE);
        
        mainPanel.add(createCell(""));
        
        for (String header : headers) {
            mainPanel.add(createCell(header));
        }
        
        for (int i = 0; i < data.size(); i++) {
            mainPanel.add(createCell(headers.get(i)));
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