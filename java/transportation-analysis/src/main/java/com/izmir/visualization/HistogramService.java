package com.izmir.visualization;

import java.io.File;
import java.io.IOException;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Date;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;
import java.util.logging.Level;
import java.util.logging.Logger;

import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartUtils;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.axis.CategoryAxis;
import org.jfree.chart.axis.CategoryLabelPositions;
import org.jfree.chart.plot.CategoryPlot;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.chart.renderer.category.BarRenderer;
import org.jfree.data.category.DefaultCategoryDataset;

import com.izmir.transportation.cost.ClusterMetrics;

/**
 * Service class for generating charts from cluster metrics data.
 *
 * @author yagizugurveren
 */
public class HistogramService {

    private static final Logger LOGGER = Logger.getLogger(HistogramService.class.getName());
    private static final int CHART_WIDTH = 800;
    private static final int CHART_HEIGHT = 600;
    private static final int MAX_VISIBLE_LABELS = 20; // Maximum number of visible labels on x-axis

    /**
     * Generates and saves bar charts for total distance and total cost from cluster metrics.
     *
     * @param clusterMetricsMap A map of community ID to ClusterMetrics.
     * @param algorithmName     The name of the clustering algorithm used (for filenames).
     */
    public void generateHistograms(Map<Integer, ClusterMetrics> clusterMetricsMap, String algorithmName) {
        if (clusterMetricsMap == null || clusterMetricsMap.isEmpty()) {
            LOGGER.warning("Cluster metrics map is empty or null. Cannot generate charts.");
            return;
        }

        LOGGER.info("Generating bar charts for " + algorithmName + "...");

        try {
            generateAndSaveBarChart(clusterMetricsMap, "Total Distance per Cluster", 
                    "Cluster Number", "Distance (km)", 
                    algorithmName + "_Distance_Chart", 
                    metrics -> metrics.getTotalDistance());
            
            generateAndSaveBarChart(clusterMetricsMap, "Total Cost per Cluster", 
                    "Cluster Number", "Cost", 
                    algorithmName + "_Cost_Chart", 
                    metrics -> metrics.getTotalCost());
            
            // Add vehicle type distribution chart
            generateVehicleTypeDistributionChart(clusterMetricsMap, algorithmName);
            
            // Add cost comparison chart by vehicle type
            generateCostComparisonByVehicleType(clusterMetricsMap, algorithmName);
            
            LOGGER.info("Charts generated successfully.");
        } catch (IOException e) {
            LOGGER.log(Level.SEVERE, "Failed to save charts for " + algorithmName, e);
        }
    }

    /**
     * Generates a bar chart showing the distribution of vehicle types across clusters.
     * 
     * @param clusterMetricsMap The map of cluster IDs to their metrics
     * @param algorithmName The name of the clustering algorithm used
     * @throws IOException If saving the chart fails
     */
    private void generateVehicleTypeDistributionChart(Map<Integer, ClusterMetrics> clusterMetricsMap, 
                                                     String algorithmName) throws IOException {
        DefaultCategoryDataset dataset = new DefaultCategoryDataset();
        int busCount = 0;
        int minibusCount = 0;
        
        for (ClusterMetrics metrics : clusterMetricsMap.values()) {
            if ("BUS".equals(metrics.getVehicleType())) {
                busCount++;
            } else if ("MINIBUS".equals(metrics.getVehicleType())) {
                minibusCount++;
            }
        }
        
        // Add data to dataset
        dataset.addValue(busCount, "Count", "Standard Bus");
        dataset.addValue(minibusCount, "Count", "Minibus");
        
        JFreeChart barChart = ChartFactory.createBarChart(
                "Vehicle Type Distribution", 
                "Vehicle Type", 
                "Number of Clusters", 
                dataset, 
                PlotOrientation.VERTICAL, 
                false, // No legend
                true,  // Tooltips
                false  // URLs
        );
        
        // Create a timestamped filename
        String timestamp = new SimpleDateFormat("yyyyMMdd_HHmmss").format(new Date());
        String filename = algorithmName + "_VehicleType_Distribution_" + timestamp + ".png";
        
        // Ensure output directory exists
        File outputDir = new File("output/charts");
        if (!outputDir.exists()) {
            if (!outputDir.mkdirs()) {
                LOGGER.severe("Failed to create output directory: " + outputDir.getAbsolutePath());
                throw new IOException("Failed to create output directory: " + outputDir.getAbsolutePath());
            }
        }

        File chartFile = new File(outputDir, filename);
        ChartUtils.saveChartAsPNG(chartFile, barChart, CHART_WIDTH, CHART_HEIGHT);
        LOGGER.info("Saved vehicle type distribution chart: " + chartFile.getAbsolutePath());
    }
    
    /**
     * Generates a bar chart comparing costs by vehicle type.
     * 
     * @param clusterMetricsMap The map of cluster IDs to their metrics
     * @param algorithmName The name of the clustering algorithm used
     * @throws IOException If saving the chart fails
     */
    private void generateCostComparisonByVehicleType(Map<Integer, ClusterMetrics> clusterMetricsMap, 
                                                   String algorithmName) throws IOException {
        DefaultCategoryDataset dataset = new DefaultCategoryDataset();
        double busTotalCost = 0;
        double minibusTotalCost = 0;
        double busFuelCost = 0;
        double minibusFuelCost = 0;
        double busFixedCost = 0;
        double minibusFixedCost = 0;
        
        for (ClusterMetrics metrics : clusterMetricsMap.values()) {
            if ("BUS".equals(metrics.getVehicleType())) {
                busTotalCost += metrics.getTotalCost();
                busFuelCost += metrics.getFuelCost();
                busFixedCost += metrics.getFixedCost();
            } else if ("MINIBUS".equals(metrics.getVehicleType())) {
                minibusTotalCost += metrics.getTotalCost();
                minibusFuelCost += metrics.getFuelCost();
                minibusFixedCost += metrics.getFixedCost();
            }
        }
        
        // Add data to dataset
        dataset.addValue(busTotalCost, "Total Cost", "Standard Bus");
        dataset.addValue(minibusTotalCost, "Total Cost", "Minibus");
        dataset.addValue(busFuelCost, "Fuel Cost", "Standard Bus");
        dataset.addValue(minibusFuelCost, "Fuel Cost", "Minibus");
        dataset.addValue(busFixedCost, "Fixed Cost", "Standard Bus");
        dataset.addValue(minibusFixedCost, "Fixed Cost", "Minibus");
        
        JFreeChart barChart = ChartFactory.createBarChart(
                "Cost Comparison by Vehicle Type", 
                "Vehicle Type", 
                "Cost", 
                dataset, 
                PlotOrientation.VERTICAL, 
                true,  // Show legend
                true,  // Tooltips
                false  // URLs
        );
        
        // Create a timestamped filename
        String timestamp = new SimpleDateFormat("yyyyMMdd_HHmmss").format(new Date());
        String filename = algorithmName + "_VehicleType_CostComparison_" + timestamp + ".png";
        
        // Ensure output directory exists
        File outputDir = new File("output/charts");
        if (!outputDir.exists()) {
            if (!outputDir.mkdirs()) {
                LOGGER.severe("Failed to create output directory: " + outputDir.getAbsolutePath());
                throw new IOException("Failed to create output directory: " + outputDir.getAbsolutePath());
            }
        }

        File chartFile = new File(outputDir, filename);
        ChartUtils.saveChartAsPNG(chartFile, barChart, CHART_WIDTH, CHART_HEIGHT);
        LOGGER.info("Saved cost comparison chart: " + chartFile.getAbsolutePath());
    }

    /**
     * Functional interface for extracting metric values
     */
    @FunctionalInterface
    private interface MetricExtractor {
        double extract(ClusterMetrics metrics);
    }
    
    /**
     * Helper method to generate and save a single bar chart.
     *
     * @param clusterMetricsMap The map of cluster IDs to their metrics.
     * @param title             The title of the chart.
     * @param xAxisLabel        The label for the X-axis.
     * @param yAxisLabel        The label for the Y-axis.
     * @param baseFileName      The base filename for saving the chart image.
     * @param metricExtractor   Function to extract the required metric value
     * @throws IOException      If saving the chart fails.
     */
    private void generateAndSaveBarChart(
            Map<Integer, ClusterMetrics> clusterMetricsMap, 
            String title, 
            String xAxisLabel, 
            String yAxisLabel, 
            String baseFileName,
            MetricExtractor metricExtractor) throws IOException {
        
        if (clusterMetricsMap == null || clusterMetricsMap.isEmpty()) {
            LOGGER.warning("No data provided for chart: " + title);
            return;
        }

        // Use TreeMap to sort the clusters by ID
        Map<Integer, ClusterMetrics> sortedMap = new TreeMap<>(clusterMetricsMap);
        int totalClusters = sortedMap.size();
        
        DefaultCategoryDataset dataset = new DefaultCategoryDataset();
        List<String> labels = new ArrayList<>();
        
        // Determine label visibility interval
        int labelInterval = totalClusters <= MAX_VISIBLE_LABELS ? 1 : 
                            Math.max(1, totalClusters / MAX_VISIBLE_LABELS);
        
        // Add each cluster's data to the dataset with sequential numbering
        int clusterNumber = 1;
        for (Map.Entry<Integer, ClusterMetrics> entry : sortedMap.entrySet()) {
            ClusterMetrics metrics = entry.getValue();
            double value = metricExtractor.extract(metrics);
            
            // Use sequential cluster numbers instead of actual cluster IDs
            String label = String.valueOf(clusterNumber);
            // Add vehicle type information to the label
            String vehicleType = metrics.getVehicleType();
            String displayLabel = label + " (" + (vehicleType != null ? vehicleType : "BUS") + ")";
            
            // Only add visible labels to our list
            if ((clusterNumber - 1) % labelInterval == 0 || clusterNumber == 1 || clusterNumber == totalClusters) {
                labels.add(displayLabel);
                dataset.addValue(value, "Clusters", displayLabel);
            } else {
                // Empty label for most clusters to avoid crowding
                dataset.addValue(value, "Clusters", "");
            }
            clusterNumber++;
        }

        JFreeChart barChart = ChartFactory.createBarChart(
                title, 
                xAxisLabel, 
                yAxisLabel, 
                dataset, 
                PlotOrientation.VERTICAL, 
                false, // No legend needed for single series
                true,  // Tooltips
                false  // URLs
        );
        
        // Customize the x-axis
        CategoryPlot plot = (CategoryPlot) barChart.getPlot();
        CategoryAxis domainAxis = plot.getDomainAxis();
        
        // Rotate labels for better readability and set spacing
        domainAxis.setCategoryLabelPositions(CategoryLabelPositions.UP_45);
        domainAxis.setLowerMargin(0.01);  // Space between edge and first bar
        domainAxis.setUpperMargin(0.01);  // Space between edge and last bar
        domainAxis.setCategoryMargin(0.1); // Space between categories (as proportion)
        
        // Adjust bar width
        BarRenderer renderer = (BarRenderer) plot.getRenderer();
        renderer.setMaximumBarWidth(0.05); // Make bars thinner when there are many
        
        // Create a timestamped filename
        String timestamp = new SimpleDateFormat("yyyyMMdd_HHmmss").format(new Date());
        String filename = baseFileName + "_" + timestamp + ".png";
        
        // Ensure output directory exists
        File outputDir = new File("output/charts");
        if (!outputDir.exists()) {
            if (!outputDir.mkdirs()) {
                LOGGER.severe("Failed to create output directory: " + outputDir.getAbsolutePath());
                throw new IOException("Failed to create output directory: " + outputDir.getAbsolutePath());
            }
        }

        File chartFile = new File(outputDir, filename);
        ChartUtils.saveChartAsPNG(chartFile, barChart, CHART_WIDTH, CHART_HEIGHT);
        LOGGER.info("Saved chart: " + chartFile.getAbsolutePath());
    }
} 