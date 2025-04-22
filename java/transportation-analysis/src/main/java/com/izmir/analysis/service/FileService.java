package com.izmir.analysis.service;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVParser;
import org.apache.commons.csv.CSVPrinter;
import org.apache.commons.csv.CSVRecord;

import com.izmir.analysis.model.TransportationData;

/**
 * Service class for file operations related to transportation analysis.
 * Handles reading from and writing to CSV files.
 */
public class FileService {
    
    private static final Pattern FILENAME_PATTERN = 
            Pattern.compile("transportation_cost_([A-Z]+)_([A-Z_]+)_K(\\d+)_.*\\.csv");
    
    /**
     * Parses a transportation cost CSV file.
     * 
     * @param file The CSV file to parse
     * @return A list of TransportationData objects parsed from the file
     * @throws IOException If an I/O error occurs
     */
    public List<TransportationData> parseTransportationCostFile(File file) throws IOException {
        List<TransportationData> dataList = new ArrayList<>();
        
        // First, read all lines to process data section properly
        List<String> allLines = Files.readAllLines(file.toPath(), java.nio.charset.StandardCharsets.UTF_8);
        
        // Find where the actual data starts (after comments and headers)
        int dataStartIndex = 0;
        while (dataStartIndex < allLines.size() && 
               (allLines.get(dataStartIndex).startsWith("#") || 
                allLines.get(dataStartIndex).contains("Community_ID"))) {
            dataStartIndex++;
        }
        
        // If we found actual data rows
        if (dataStartIndex < allLines.size()) {
            // Find the header line (the line with "Community_ID")
            int headerIndex = 0;
            for (int i = 0; i < allLines.size(); i++) {
                if (allLines.get(i).contains("Community_ID") && 
                    allLines.get(i).contains("Node_Count") && 
                    allLines.get(i).contains("Vehicle_Type")) {
                    headerIndex = i;
                    break;
                }
            }
            
            if (headerIndex > 0) {
                // Create a temporary file with just the header and data
                Path tempFile = Files.createTempFile("transportation_data_", ".csv");
                try {
                    // Write header and data to temp file
                    List<String> csvData = new ArrayList<>();
                    csvData.add(allLines.get(headerIndex));
                    csvData.addAll(allLines.subList(dataStartIndex, allLines.size()));
                    
                    // Filter out the second data section if it exists
                    int secondHeaderIndex = -1;
                    for (int i = 0; i < csvData.size(); i++) {
                        if (i > 0 && csvData.get(i).contains("Community_ID") && 
                            csvData.get(i).contains("Vehicle_Type")) {
                            secondHeaderIndex = i;
                            break;
                        }
                    }
                    
                    if (secondHeaderIndex > 0) {
                        csvData = csvData.subList(0, secondHeaderIndex);
                    }
                    
                    Files.write(tempFile, csvData, java.nio.charset.StandardCharsets.UTF_8);
                    
                    // Parse the CSV with proper headers
                    try (java.io.Reader reader = Files.newBufferedReader(tempFile);
                         CSVParser csvParser = new CSVParser(reader, CSVFormat.DEFAULT.withFirstRecordAsHeader())) {
                        
                        for (CSVRecord record : csvParser) {
                            try {
                                TransportationData data = new TransportationData();
                                data.setCommunityId(Integer.parseInt(record.get("Community_ID")));
                                data.setNodeCount(Integer.parseInt(record.get("Node_Count")));
                                data.setVehicleType(record.get("Vehicle_Type"));
                                data.setVehiclesRequired(Integer.parseInt(record.get("Vehicles_Required")));
                                data.setTotalDistanceKm(Double.parseDouble(record.get("Total_Distance_Km")));
                                data.setTotalFuelLiters(Double.parseDouble(record.get("Total_Fuel_Liters")));
                                data.setFuelCost(Double.parseDouble(record.get("Fuel_Cost")));
                                data.setFixedCost(Double.parseDouble(record.get("Fixed_Cost")));
                                data.setTotalCost(Double.parseDouble(record.get("Total_Cost")));
                                
                                dataList.add(data);
                            } catch (IllegalArgumentException e) {
                                System.err.println("Error parsing record: " + record);
                            }
                        }
                    } finally {
                        // Delete temp file
                        Files.deleteIfExists(tempFile);
                    }
                } catch (Exception e) {
                    Files.deleteIfExists(tempFile);
                    throw e;
                }
            }
        }
        
        return dataList;
    }
    
    /**
     * Extracts method information from a transportation cost CSV filename.
     * 
     * @param filename The filename to parse
     * @return A map containing the clustering algorithm and graph method
     */
    public Map<String, String> extractMethodFromFilename(String filename) {
        Map<String, String> result = new HashMap<>();
        
        Matcher matcher = FILENAME_PATTERN.matcher(filename);
        if (matcher.matches()) {
            String clusteringAlgo = matcher.group(1);
            String graphMethod = matcher.group(2);
            String kValue = matcher.group(3);
            
            result.put("clusteringAlgo", clusteringAlgo);
            result.put("graphMethod", graphMethod);
            result.put("kValue", kValue);
            result.put("methodCombo", clusteringAlgo + " + " + graphMethod);
        }
        
        return result;
    }
    
    /**
     * Creates a directory if it doesn't exist.
     * 
     * @param dirPath The path of the directory to create
     * @throws IOException If an I/O error occurs
     */
    public void createDirectoryIfNotExists(String dirPath) throws IOException {
        Path path = Paths.get(dirPath);
        if (!Files.exists(path)) {
            Files.createDirectories(path);
        }
    }
    
    /**
     * Writes data to a CSV file.
     * 
     * @param filePath The path of the CSV file to write
     * @param headers The headers for the CSV file
     * @param records The records to write to the CSV file
     * @throws IOException If an I/O error occurs
     */
    public void writeCSV(String filePath, String[] headers, List<String[]> records) throws IOException {
        try (CSVPrinter printer = new CSVPrinter(new FileWriter(filePath), 
                CSVFormat.DEFAULT.withHeader(headers))) {
            
            for (String[] record : records) {
                printer.printRecord((Object[]) record);
            }
        }
    }
    
    /**
     * Writes method comparison data to a CSV file.
     * 
     * @param filePath The path of the CSV file to write
     * @param data A list of maps with method comparison data
     * @throws IOException If an I/O error occurs
     */
    public void writeMethodComparisonData(String filePath, List<Map<String, Object>> data) throws IOException {
        try (CSVPrinter printer = new CSVPrinter(new FileWriter(filePath), 
                CSVFormat.DEFAULT.withHeader("Method", "Fuel", "Routes", "Valid %", "Avg Size", "Minibus %"))) {
            
            for (Map<String, Object> row : data) {
                printer.printRecord(
                    row.get("Method"),
                    row.get("Fuel"),
                    row.get("Routes"),
                    row.get("Valid %"),
                    row.get("Avg Size"), 
                    row.get("Minibus %")
                );
                
                // Print LaTeX table row to console
                System.out.printf("%s & %s & %s & %s & %s & %s \\\\\n", 
                    row.get("Method"), 
                    row.get("Fuel"), 
                    row.get("Routes"), 
                    row.get("Valid %"), 
                    row.get("Avg Size"), 
                    row.get("Minibus %")
                );
            }
        }
    }
} 