package com.izmir.analysis;

import java.awt.Color;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVPrinter;
import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartUtils;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.axis.CategoryAxis;
import org.jfree.chart.axis.CategoryLabelPositions;
import org.jfree.chart.plot.CategoryPlot;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.chart.renderer.category.BarRenderer;
import org.jfree.data.category.DefaultCategoryDataset;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.izmir.analysis.model.TransportationData;
import com.izmir.analysis.service.FileService;

/**
 * Analyzer for transportation data across different clustering algorithms and graph construction methods.
 * Parses CSV files with transportation cost data and generates analysis results and visualizations.
 */
public class TransportationAnalyzer {
    
    private static final String CSV_DIR = "Bachelor_Thesis/csv_files";
    private static final String OUTPUT_DIR = "analysis_output";
    private static final String IMG_DIR = "img";
    
    private final Map<String, Double> fuelConsumption = new HashMap<>();
    private final Map<String, List<Integer>> clusterSizes = new HashMap<>();
    private final Map<String, Map<String, Integer>> clusterDistribution = new HashMap<>();
    private final Map<String, Map<String, Integer>> vehicleAllocation = new HashMap<>();
    private final Map<String, Integer> routeCount = new HashMap<>();
    private final Map<String, Double> validClusters = new HashMap<>();
    private final Map<String, Double> avgClusterSize = new HashMap<>();
    
    private final FileService fileService;
    private boolean useJFreeChart = true;
    
    public TransportationAnalyzer() {
        this.fileService = new FileService();
    }
    
    public void analyze() {
        try {
            parseCSVFiles();
            createOutputDirectories();
            generateTextBasedCharts();
            generateSummaryTable();
            
            // Try to generate JFreeChart visualizations
            try {
                if (useJFreeChart) {
                    generateFuelComparisonChart();
                    generateClusterDistributionChart();
                    generateVehicleAllocationChart();
                }
            } catch (Exception e) {
                System.out.println("WARNING: JFreeChart visualization failed. Text-based output is still available.");
                e.printStackTrace();
            }
            
            System.out.println("Analysis complete!");
        } catch (Exception e) {
            System.err.println("Error during analysis: " + e.getMessage());
            e.printStackTrace();
        }
    }
    
    private void parseCSVFiles() throws IOException {
        File csvDirectory = new File(CSV_DIR);
        File[] csvFiles = csvDirectory.listFiles((dir, name) -> 
                name.endsWith(".csv") && name.startsWith("transportation_cost_"));
        
        if (csvFiles == null || csvFiles.length == 0) {
            throw new IOException("No transportation cost CSV files found in " + CSV_DIR);
        }
        
        for (File file : csvFiles) {
            Map<String, String> methodInfo = fileService.extractMethodFromFilename(file.getName());
            
            if (!methodInfo.isEmpty()) {
                String methodCombo = methodInfo.get("methodCombo");
                List<TransportationData> dataList = fileService.parseTransportationCostFile(file);
                
                if (!dataList.isEmpty()) {
                    // Calculate and store metrics
                    List<Integer> sizes = dataList.stream()
                        .map(TransportationData::getNodeCount)
                        .collect(Collectors.toList());
                    
                    double totalFuel = dataList.stream()
                        .mapToDouble(TransportationData::getTotalFuelLiters)
                        .sum();
                    
                    // Store results
                    fuelConsumption.put(methodCombo, totalFuel);
                    clusterSizes.put(methodCombo, sizes);
                    routeCount.put(methodCombo, sizes.size());
                    
                    // Calculate cluster distribution
                    int belowMin = 0, minibus = 0, bus = 0, aboveMax = 0;
                    for (int size : sizes) {
                        if (size < 10) belowMin++;
                        else if (size <= 25) minibus++;
                        else if (size <= 50) bus++;
                        else aboveMax++;
                    }
                    
                    Map<String, Integer> distribution = new HashMap<>();
                    distribution.put("<10", belowMin);
                    distribution.put("10-25", minibus);
                    distribution.put("26-50", bus);
                    distribution.put(">50", aboveMax);
                    clusterDistribution.put(methodCombo, distribution);
                    
                    Map<String, Integer> allocation = new HashMap<>();
                    allocation.put("minibus", minibus);
                    allocation.put("standard_bus", bus);
                    vehicleAllocation.put(methodCombo, allocation);
                    
                    validClusters.put(methodCombo, 100.0 * (minibus + bus) / sizes.size());
                    avgClusterSize.put(methodCombo, sizes.stream()
                        .mapToDouble(Integer::doubleValue)
                        .average()
                        .orElse(0.0));
                }
            }
        }
    }
    
    private void createOutputDirectories() throws IOException {
        fileService.createDirectoryIfNotExists(OUTPUT_DIR);
        fileService.createDirectoryIfNotExists(IMG_DIR);
    }
    
    private void generateSummaryTable() throws IOException {
        String outputFilePath = Paths.get(OUTPUT_DIR, "method_comparison.csv").toString();
        
        try (CSVPrinter printer = new CSVPrinter(new FileWriter(outputFilePath), 
                CSVFormat.DEFAULT.withHeader("Method", "Fuel", "Routes", "Valid %", "Avg Size", "Minibus %"))) {
            
            List<Map.Entry<String, Double>> sortedEntries = new ArrayList<>(fuelConsumption.entrySet());
            sortedEntries.sort(Map.Entry.comparingByValue());
            
            for (Map.Entry<String, Double> entry : sortedEntries) {
                String method = entry.getKey();
                double fuel = entry.getValue();
                int routes = routeCount.get(method);
                double valid = validClusters.get(method);
                double avgSize = avgClusterSize.get(method);
                double minibusPercent = 100.0 * vehicleAllocation.get(method).get("minibus") / routes;
                
                printer.printRecord(
                    method, 
                    String.format("%.1f", fuel), 
                    routes, 
                    String.format("%.1f%%", valid), 
                    String.format("%.1f", avgSize), 
                    String.format("%.1f%%", minibusPercent)
                );
                
                // Print LaTeX table row to console
                System.out.printf("%s & %.1f & %d & %.1f\\%% & %.1f & %.1f\\%% \\\\\n", 
                        method, fuel, routes, valid, avgSize, minibusPercent);
            }
        }
        
        saveFuelComparisonData();
        saveClusterDistributionData();
        saveVehicleAllocationData();
        saveEfficiencyVsRoutesData();
    }
    
    private void saveFuelComparisonData() throws IOException {
        String outputFilePath = Paths.get(OUTPUT_DIR, "fuel_comparison.csv").toString();
        List<String[]> records = new ArrayList<>();
        
        for (Map.Entry<String, Double> entry : fuelConsumption.entrySet()) {
            String[] parts = entry.getKey().split(" \\+ ");
            records.add(new String[] {parts[0], parts[1], String.valueOf(entry.getValue())});
        }
        
        fileService.writeCSV(outputFilePath, new String[] {"Clustering", "Graph", "Fuel"}, records);
    }
    
    private void saveClusterDistributionData() throws IOException {
        String outputFilePath = Paths.get(OUTPUT_DIR, "cluster_distribution.csv").toString();
        List<String[]> records = new ArrayList<>();
        
        for (Map.Entry<String, Map<String, Integer>> entry : clusterDistribution.entrySet()) {
            String method = entry.getKey();
            for (Map.Entry<String, Integer> sizeRange : entry.getValue().entrySet()) {
                records.add(new String[] {method, sizeRange.getKey(), String.valueOf(sizeRange.getValue())});
            }
        }
        
        fileService.writeCSV(outputFilePath, new String[] {"Method", "Size_Range", "Count"}, records);
    }
    
    private void saveVehicleAllocationData() throws IOException {
        String outputFilePath = Paths.get(OUTPUT_DIR, "vehicle_allocation.csv").toString();
        List<String[]> records = new ArrayList<>();
        
        for (Map.Entry<String, Map<String, Integer>> entry : vehicleAllocation.entrySet()) {
            String method = entry.getKey();
            for (Map.Entry<String, Integer> vehicleType : entry.getValue().entrySet()) {
                records.add(new String[] {method, vehicleType.getKey(), String.valueOf(vehicleType.getValue())});
            }
        }
        
        fileService.writeCSV(outputFilePath, new String[] {"Method", "Vehicle_Type", "Count"}, records);
    }
    
    private void saveEfficiencyVsRoutesData() throws IOException {
        String outputFilePath = Paths.get(OUTPUT_DIR, "efficiency_vs_routes.csv").toString();
        List<String[]> records = new ArrayList<>();
        
        for (String method : fuelConsumption.keySet()) {
            records.add(new String[] {
                method, 
                String.valueOf(routeCount.get(method)), 
                String.valueOf(fuelConsumption.get(method))
            });
        }
        
        fileService.writeCSV(outputFilePath, new String[] {"Method", "Routes", "Fuel"}, records);
    }
    
    private void generateFuelComparisonChart() throws IOException {
        // Skip JFreeChart generation if using fallback
        if (!useJFreeChart) {
            return;
        }
        
        // Group by clustering algorithm
        Map<String, Map<String, Double>> clusteringGraphFuel = new HashMap<>();
        
        for (Map.Entry<String, Double> entry : fuelConsumption.entrySet()) {
            String[] parts = entry.getKey().split(" \\+ ");
            String clustering = parts[0];
            String graph = parts[1];
            
            clusteringGraphFuel.computeIfAbsent(clustering, k -> new HashMap<>())
                    .put(graph, entry.getValue());
        }
        
        // Get unique graph methods
        List<String> graphMethods = fuelConsumption.keySet().stream()
                .map(key -> key.split(" \\+ ")[1])
                .distinct()
                .collect(Collectors.toList());
        
        // Create dataset
        DefaultCategoryDataset dataset = new DefaultCategoryDataset();
        
        for (Map.Entry<String, Map<String, Double>> entry : clusteringGraphFuel.entrySet()) {
            String clustering = entry.getKey();
            Map<String, Double> graphValues = entry.getValue();
            
            for (String graph : graphMethods) {
                Double value = graphValues.get(graph);
                if (value != null) {
                    dataset.addValue(value, clustering, graph);
                }
            }
        }
        
        // Create chart
        JFreeChart chart = ChartFactory.createBarChart(
                "Fuel Consumption by Method Combination",  // title
                "Graph Construction Method",               // x-axis label
                "Total Fuel Consumption",                  // y-axis label
                dataset,
                PlotOrientation.VERTICAL,
                true,                                      // include legend
                true,                                      // tooltips
                false                                      // URLs
        );
        
        // Customize chart
        CategoryPlot plot = chart.getCategoryPlot();
        BarRenderer renderer = (BarRenderer) plot.getRenderer();
        
        // Set custom colors
        renderer.setSeriesPaint(0, new Color(31, 119, 180));    // SPECTRAL
        renderer.setSeriesPaint(1, new Color(255, 127, 14));    // LEIDEN
        renderer.setSeriesPaint(2, new Color(44, 160, 44));     // MVAGC
        
        // Adjust x-axis labels for better readability
        CategoryAxis domainAxis = plot.getDomainAxis();
        domainAxis.setCategoryLabelPositions(CategoryLabelPositions.createUpRotationLabelPositions(Math.PI / 6.0));
        
        // Save chart
        File outputFile = new File(IMG_DIR, "fuel_comparison.png");
        ChartUtils.saveChartAsPNG(outputFile, chart, 900, 600);
    }
    
    private void generateClusterDistributionChart() throws IOException {
        // Skip JFreeChart generation if using fallback
        if (!useJFreeChart) {
            return;
        }
        
        // Get list of clustering algorithms
        List<String> clusteringAlgos = fuelConsumption.keySet().stream()
                .map(key -> key.split(" \\+ ")[0])
                .distinct()
                .collect(Collectors.toList());
        
        // Create one chart per clustering algorithm
        for (String algo : clusteringAlgos) {
            DefaultCategoryDataset dataset = new DefaultCategoryDataset();
            
            // Get data for this algorithm
            Map<String, Map<String, Integer>> algoData = new HashMap<>();
            for (Map.Entry<String, Map<String, Integer>> entry : clusterDistribution.entrySet()) {
                if (entry.getKey().startsWith(algo)) {
                    String graphMethod = entry.getKey().split(" \\+ ")[1];
                    algoData.put(graphMethod, entry.getValue());
                }
            }
            
            // Add data to dataset
            for (Map.Entry<String, Map<String, Integer>> entry : algoData.entrySet()) {
                String graphMethod = entry.getKey();
                Map<String, Integer> distribution = entry.getValue();
                
                for (String sizeRange : Arrays.asList("<10", "10-25", "26-50", ">50")) {
                    dataset.addValue(distribution.get(sizeRange), sizeRange, graphMethod);
                }
            }
            
            // Create chart
            JFreeChart chart = ChartFactory.createStackedBarChart(
                    algo + " Algorithm - Cluster Size Distribution",
                    "Graph Construction Method",
                    "Number of Clusters",
                    dataset,
                    PlotOrientation.VERTICAL,
                    true,
                    true,
                    false
            );
            
            // Customize chart
            CategoryPlot plot = chart.getCategoryPlot();
            BarRenderer renderer = (BarRenderer) plot.getRenderer();
            
            // Set custom colors
            renderer.setSeriesPaint(0, new Color(214, 39, 40));   // <10
            renderer.setSeriesPaint(1, new Color(31, 119, 180));  // 10-25
            renderer.setSeriesPaint(2, new Color(44, 160, 44));   // 26-50
            renderer.setSeriesPaint(3, new Color(255, 127, 14));  // >50
            
            // Save chart
            File outputFile = new File(IMG_DIR, "cluster_distribution_" + algo + ".png");
            ChartUtils.saveChartAsPNG(outputFile, chart, 800, 600);
        }
        
        // Create combined chart image using metadata to reference individual charts
        Map<String, String> algoChartPaths = new HashMap<>();
        for (String algo : clusteringAlgos) {
            algoChartPaths.put(algo, "cluster_distribution_" + algo + ".png");
        }
        
        // Save chart paths as JSON for potential use in a viewer
        ObjectMapper mapper = new ObjectMapper();
        mapper.writeValue(new File(OUTPUT_DIR, "cluster_charts.json"), algoChartPaths);
    }
    
    private void generateVehicleAllocationChart() throws IOException {
        // Skip JFreeChart generation if using fallback
        if (!useJFreeChart) {
            return;
        }
        
        DefaultCategoryDataset dataset = new DefaultCategoryDataset();
        
        List<String> methods = new ArrayList<>(fuelConsumption.keySet());
        methods.sort((m1, m2) -> Double.compare(fuelConsumption.get(m1), fuelConsumption.get(m2)));
        
        for (String method : methods) {
            Map<String, Integer> allocation = vehicleAllocation.get(method);
            dataset.addValue(allocation.get("minibus"), "Minibus", method);
            dataset.addValue(allocation.get("standard_bus"), "Standard Bus", method);
        }
        
        // Create chart
        JFreeChart chart = ChartFactory.createStackedBarChart(
                "Vehicle Type Allocation by Method",
                "Method Combination",
                "Number of Routes",
                dataset,
                PlotOrientation.VERTICAL,
                true,
                true,
                false
        );
        
        // Customize chart
        CategoryPlot plot = chart.getCategoryPlot();
        BarRenderer renderer = (BarRenderer) plot.getRenderer();
        
        // Set custom colors
        renderer.setSeriesPaint(0, new Color(31, 119, 180));  // Minibus
        renderer.setSeriesPaint(1, new Color(255, 127, 14));  // Standard Bus
        
        // Adjust x-axis labels for better readability
        CategoryAxis domainAxis = plot.getDomainAxis();
        domainAxis.setCategoryLabelPositions(CategoryLabelPositions.createUpRotationLabelPositions(Math.PI / 6.0));
        
        // Save chart
        File outputFile = new File(IMG_DIR, "vehicle_allocation.png");
        ChartUtils.saveChartAsPNG(outputFile, chart, 1200, 800);
    }
    
    private void generateEfficiencyVsRoutesChart() throws IOException {
        // This would typically be a scatter plot, but JFreeChart setup for scatter plots is more complex
        // For simplicity, this implementation saves the data to CSV
        // The actual visualization could be created with a different library or tool
        System.out.println("Efficiency vs Routes data saved to CSV for external visualization.");
    }
    
    /**
     * Generates simple text-based chart data when JFreeChart fails
     */
    private void generateTextBasedCharts() {
        try {
            // Generate method comparison data
            List<Map<String, Object>> comparisonData = new ArrayList<>();
            
            for (Map.Entry<String, Double> entry : fuelConsumption.entrySet()) {
                String[] parts = entry.getKey().split(" \\+ ");
                String clustering = parts[0];
                String graph = parts[1];
                
                double fuelConsumption = entry.getValue();
                int routes = routeCount.get(entry.getKey());
                double validPercentage = 100.0 * routes / clusterSizes.get(entry.getKey()).size();
                double avgClusterSize = this.avgClusterSize.get(entry.getKey());
                double minibusPercentage = 100.0 * vehicleAllocation.get(entry.getKey()).get("minibus") / routes;
                
                Map<String, Object> row = new HashMap<>();
                row.put("Method", clustering + " + " + graph);
                row.put("Fuel", String.format("%.2f", fuelConsumption));
                row.put("Routes", routes);
                row.put("Valid %", String.format("%.2f", validPercentage));
                row.put("Avg Size", String.format("%.2f", avgClusterSize));
                row.put("Minibus %", String.format("%.2f", minibusPercentage));
                
                comparisonData.add(row);
            }
            
            // Write method comparison data
            fileService.writeMethodComparisonData(Paths.get(OUTPUT_DIR, "method_comparison.csv").toString(), comparisonData);
            
            // Generate fuel comparison data
            try (CSVPrinter printer = new CSVPrinter(new FileWriter(Paths.get(OUTPUT_DIR, "fuel_comparison.csv").toString()), 
                    CSVFormat.DEFAULT.withHeader("Method", "Fuel Consumption"))) {
                
                for (Map.Entry<String, Double> entry : fuelConsumption.entrySet()) {
                    printer.printRecord(entry.getKey(), String.format("%.2f", entry.getValue()));
                }
            }
            
            // Generate cluster distribution data
            try (CSVPrinter printer = new CSVPrinter(new FileWriter(Paths.get(OUTPUT_DIR, "cluster_distribution.csv").toString()), 
                    CSVFormat.DEFAULT.withHeader("Method", "Cluster Size", "Count"))) {
                
                for (Map.Entry<String, Map<String, Integer>> entry : clusterDistribution.entrySet()) {
                    String method = entry.getKey();
                    for (Map.Entry<String, Integer> sizeRange : entry.getValue().entrySet()) {
                        printer.printRecord(method, sizeRange.getKey(), sizeRange.getValue());
                    }
                }
            }
            
            // Generate vehicle allocation data
            try (CSVPrinter printer = new CSVPrinter(new FileWriter(Paths.get(OUTPUT_DIR, "vehicle_allocation.csv").toString()), 
                    CSVFormat.DEFAULT.withHeader("Method", "Vehicle Type", "Count"))) {
                
                for (Map.Entry<String, Map<String, Integer>> entry : vehicleAllocation.entrySet()) {
                    String method = entry.getKey();
                    for (Map.Entry<String, Integer> vehicleType : entry.getValue().entrySet()) {
                        printer.printRecord(method, vehicleType.getKey(), vehicleType.getValue());
                    }
                }
            }
            
            // Generate efficiency vs routes data
            try (CSVPrinter printer = new CSVPrinter(new FileWriter(Paths.get(OUTPUT_DIR, "efficiency_vs_routes.csv").toString()), 
                    CSVFormat.DEFAULT.withHeader("Method", "Efficiency", "Valid Routes"))) {
                
                for (Map.Entry<String, Integer> entry : routeCount.entrySet()) {
                    String method = entry.getKey();
                    int validRoutes = entry.getValue();
                    double methodFuel = this.fuelConsumption.getOrDefault(method, 0.0);
                    double efficiency = validRoutes > 0 ? methodFuel / validRoutes : 0;
                    
                    printer.printRecord(method, String.format("%.2f", efficiency), validRoutes);
                }
            }
            
        } catch (IOException e) {
            System.err.println("Error generating text-based charts: " + e.getMessage());
            e.printStackTrace();
        }
    }
} 