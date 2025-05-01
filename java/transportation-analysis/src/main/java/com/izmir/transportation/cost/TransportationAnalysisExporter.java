package com.izmir.transportation.cost;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.text.DecimalFormat;
import java.text.DecimalFormatSymbols;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.Locale;
import java.util.Map;

import com.izmir.transportation.cost.OptimizedTransportationCostAnalyzer.OptimizedCommunityTransportationCost;
import com.izmir.transportation.cost.OptimizedTransportationCostAnalyzer.VehicleType;
import com.izmir.transportation.cost.TransportationCostAnalyzer.CommunityTransportationCost;

/**
 * Utility class to export transportation cost analyses to CSV files
 * in a format that's safe for importing into Google Sheets.
 * 
 * @author yagizugurveren
 */
public class TransportationAnalysisExporter {
    
    private static final String BASE_EXPORT_DIR = "results/transportation_analyses";
    private static final SimpleDateFormat DATE_FORMAT = new SimpleDateFormat("yyyy-MM-dd");
    private static final SimpleDateFormat TIME_FORMAT = new SimpleDateFormat("HHmmss");
    
    // Create a DecimalFormat with European number format (dot as thousands separator, comma as decimal separator)
    private static final DecimalFormat DECIMAL_FORMAT;
    
    static {
        DecimalFormatSymbols symbols = new DecimalFormatSymbols(Locale.GERMAN);
        symbols.setDecimalSeparator(',');
        symbols.setGroupingSeparator('.');
        DECIMAL_FORMAT = new DecimalFormat("#,##0.00", symbols);
    }
    
    /**
     * Exports transportation cost analyses to CSV files in a format safe for Google Sheets.
     * 
     * @param clusteringAlgorithm The clustering algorithm used
     * @param graphStrategy The graph construction strategy used
     * @param analyzers Array of analyzers containing the analysis results (first with minibus, second buses only)
     * @throws IOException If there's an error writing to the files
     */
    public static void exportAnalyses(
            String clusteringAlgorithm, 
            String graphStrategy,
            OptimizedTransportationCostAnalyzer... analyzers) throws IOException {
        
        // Create directory structure
        Date now = new Date();
        String todayDate = DATE_FORMAT.format(now);
        String currentTime = TIME_FORMAT.format(now);
        
        // Create directory: results/transportation_analyses/YYYY-MM-DD/
        String dateDir = BASE_EXPORT_DIR + "/" + todayDate;
        File dir = new File(dateDir);
        if (!dir.exists()) {
            dir.mkdirs();
        }
        
        // Base part of the filename including the date
        String baseFilename = todayDate + "_" + clusteringAlgorithm + "_" + graphStrategy + "_" + currentTime;
        
        // Export files with correct vehicle mode in the filename
        if (analyzers.length >= 1) {
            // First analysis (with minibus)
            String filename1 = dateDir + "/" + baseFilename + "_WithMinibus.csv";
            exportSimplifiedAnalysis(filename1, analyzers[0]);
        }
        
        if (analyzers.length >= 2) {
            // Second analysis (buses only)
            String filename2 = dateDir + "/" + baseFilename + "_OnlyBuses.csv";
            exportSimplifiedAnalysis(filename2, analyzers[1]);
        }
    }
    
    /**
     * Exports a simplified analysis file with just community-level data.
     * 
     * @param filename Output filename
     * @param analyzer The analyzer containing the analysis results
     * @throws IOException If there's an error writing to the file
     */
    private static void exportSimplifiedAnalysis(
            String filename,
            OptimizedTransportationCostAnalyzer analyzer) throws IOException {
        
        try (BufferedWriter writer = new BufferedWriter(new FileWriter(filename))) {
            // Write only the header row
            writer.write("Community_ID;Node_Count;Vehicle_Type;Vehicles_Required;");
            writer.write("Total_Distance_Km;Total_Fuel_Liters;Fuel_Cost;Fixed_Cost;Total_Cost\n");
            
            // Write each community's data
            Map<Integer, CommunityTransportationCost> costs = analyzer.getCommunityCosts();
            
            for (CommunityTransportationCost cost : costs.values()) {
                VehicleType vehicleType = VehicleType.BUS; // Default
                
                if (cost instanceof OptimizedCommunityTransportationCost) {
                    vehicleType = ((OptimizedCommunityTransportationCost) cost).getVehicleType();
                }
                
                String vehicleTypeStr = vehicleType.name();
                
                double fixedCostPerVehicle = vehicleType.getFixedCost();
                double fixedCost = cost.getBusCount() * fixedCostPerVehicle;
                double fuelCost = cost.getTotalFuelLiters() * OptimizedTransportationCostAnalyzer.FUEL_COST_PER_LITER;
                double totalCost = fuelCost + fixedCost;
                
                writer.write(cost.getCommunityId() + ";");
                writer.write(cost.getNodeCount() + ";");
                writer.write(vehicleTypeStr + ";");
                writer.write(cost.getBusCount() + ";");
                writer.write(formatDecimal(cost.getTotalDistanceKm()) + ";");
                writer.write(formatDecimal(cost.getTotalFuelLiters()) + ";");
                writer.write(formatDecimal(fuelCost) + ";");
                writer.write(formatDecimal(fixedCost) + ";");
                writer.write(formatDecimal(totalCost) + "\n");
            }
        }
        
        System.out.println("Simplified analysis exported to: " + filename);
    }
    
    /**
     * Formats a decimal value to use European number format (e.g., 2.187,80)
     * 
     * @param value The decimal value to format
     * @return Formatted string with European number format
     */
    private static String formatDecimal(double value) {
        return DECIMAL_FORMAT.format(value);
    }
} 