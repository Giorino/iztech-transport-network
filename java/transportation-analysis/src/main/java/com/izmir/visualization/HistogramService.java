package com.izmir.visualization;

import java.io.File;
import java.io.IOException;
import java.text.SimpleDateFormat;
import java.util.Collection;
import java.util.Date;
import java.util.Map;
import java.util.logging.Level;
import java.util.logging.Logger;

import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartUtils;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.data.statistics.HistogramDataset;
import org.jfree.data.statistics.HistogramType;

import com.izmir.transportation.cost.ClusterMetrics;

/**
 * Service class for generating histograms from cluster metrics data.
 *
 * @author yagizugurveren
 */
public class HistogramService {

    private static final Logger LOGGER = Logger.getLogger(HistogramService.class.getName());
    private static final int HISTOGRAM_WIDTH = 800;
    private static final int HISTOGRAM_HEIGHT = 600;
    private static final int BINS = 15; // Number of bins for the histogram

    /**
     * Generates and saves histograms for total distance and total cost from cluster metrics.
     *
     * @param clusterMetricsMap A map of community ID to ClusterMetrics.
     * @param algorithmName     The name of the clustering algorithm used (for filenames).
     */
    public void generateHistograms(Map<Integer, ClusterMetrics> clusterMetricsMap, String algorithmName) {
        if (clusterMetricsMap == null || clusterMetricsMap.isEmpty()) {
            LOGGER.warning("Cluster metrics map is empty or null. Cannot generate histograms.");
            return;
        }

        LOGGER.info("Generating histograms for " + algorithmName + "...");

        Collection<ClusterMetrics> metricsCollection = clusterMetricsMap.values();
        double[] distances = metricsCollection.stream().mapToDouble(ClusterMetrics::getTotalDistanceKm).toArray();
        double[] costs = metricsCollection.stream().mapToDouble(ClusterMetrics::getTotalCost).toArray();

        try {
            generateAndSaveHistogram(distances, "Total Distance per Cluster (km)", "Distance (km)", algorithmName + "_Distance_Histogram");
            generateAndSaveHistogram(costs, "Total Cost per Cluster", "Cost", algorithmName + "_Cost_Histogram");
            LOGGER.info("Histograms generated successfully.");
        } catch (IOException e) {
            LOGGER.log(Level.SEVERE, "Failed to save histograms for " + algorithmName, e);
        }
    }

    /**
     * Helper method to generate and save a single histogram.
     *
     * @param data        The data array for the histogram.
     * @param title       The title of the chart.
     * @param xAxisLabel  The label for the X-axis.
     * @param baseFileName The base filename for saving the chart image.
     * @throws IOException If saving the chart fails.
     */
    private void generateAndSaveHistogram(double[] data, String title, String xAxisLabel, String baseFileName) throws IOException {
        if (data == null || data.length == 0) {
            LOGGER.warning("No data provided for histogram: " + title);
            return;
        }

        HistogramDataset dataset = new HistogramDataset();
        dataset.setType(HistogramType.FREQUENCY);
        dataset.addSeries("Frequency", data, BINS);

        JFreeChart histogram = ChartFactory.createHistogram(
                title, 
                xAxisLabel, 
                "Frequency (Number of Clusters)", 
                dataset, 
                PlotOrientation.VERTICAL, 
                false, // No legend needed for single series
                true,  // Tooltips
                false  // URLs
        );

        // Create a timestamped filename
        String timestamp = new SimpleDateFormat("yyyyMMdd_HHmmss").format(new Date());
        String filename = baseFileName + "_" + timestamp + ".png";
        
        // Ensure output directory exists
        File outputDir = new File("output/histograms");
        if (!outputDir.exists()) {
            if (!outputDir.mkdirs()) {
                LOGGER.severe("Failed to create output directory: " + outputDir.getAbsolutePath());
                throw new IOException("Failed to create output directory: " + outputDir.getAbsolutePath());
            }
        }

        File chartFile = new File(outputDir, filename);
        ChartUtils.saveChartAsPNG(chartFile, histogram, HISTOGRAM_WIDTH, HISTOGRAM_HEIGHT);
        LOGGER.info("Saved histogram: " + chartFile.getAbsolutePath());
    }
} 