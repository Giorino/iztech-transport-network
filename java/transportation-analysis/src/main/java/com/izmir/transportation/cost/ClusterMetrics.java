package com.izmir.transportation.cost;

/**
 * Class to hold metrics for a community/cluster including total distance, cost, and vehicle type.
 * 
 * @author yagizugurveren
 */
public class ClusterMetrics {
    private final int communityId;
    private final double totalDistance;
    private final double totalCost;
    private String vehicleType; // "BUS" or "MINIBUS"
    private double fuelCost;
    private double fixedCost;
    private int vehicleCount;
    
    /**
     * Creates a new ClusterMetrics instance with basic metrics.
     * 
     * @param communityId The ID of the community
     * @param totalDistance The total distance traveled in kilometers
     * @param totalCost The total cost
     */
    public ClusterMetrics(int communityId, double totalDistance, double totalCost) {
        this.communityId = communityId;
        this.totalDistance = totalDistance;
        this.totalCost = totalCost;
        this.vehicleType = "BUS"; // Default vehicle type
        this.vehicleCount = 1;
    }
    
    /**
     * Creates a new ClusterMetrics instance with detailed metrics.
     * 
     * @param communityId The ID of the community
     * @param totalDistance The total distance traveled in kilometers
     * @param fuelCost The fuel cost
     * @param fixedCost The fixed cost
     * @param vehicleType The type of vehicle used
     * @param vehicleCount The number of vehicles needed
     */
    public ClusterMetrics(int communityId, double totalDistance, double fuelCost, 
                          double fixedCost, String vehicleType, int vehicleCount) {
        this.communityId = communityId;
        this.totalDistance = totalDistance;
        this.fuelCost = fuelCost;
        this.fixedCost = fixedCost;
        this.totalCost = fuelCost + fixedCost;
        this.vehicleType = vehicleType;
        this.vehicleCount = vehicleCount;
    }
    
    /**
     * Gets the community ID.
     * 
     * @return The community ID
     */
    public int getCommunityId() {
        return communityId;
    }
    
    /**
     * Gets the total distance traveled in kilometers.
     * 
     * @return The total distance
     */
    public double getTotalDistance() {
        return totalDistance;
    }
    
    /**
     * Gets the total cost.
     * 
     * @return The total cost
     */
    public double getTotalCost() {
        return totalCost;
    }
    
    /**
     * Gets the fuel cost.
     * 
     * @return The fuel cost
     */
    public double getFuelCost() {
        return fuelCost;
    }
    
    /**
     * Gets the fixed cost.
     * 
     * @return The fixed cost
     */
    public double getFixedCost() {
        return fixedCost;
    }
    
    /**
     * Gets the vehicle type.
     * 
     * @return The vehicle type
     */
    public String getVehicleType() {
        return vehicleType;
    }
    
    /**
     * Sets the vehicle type.
     * 
     * @param vehicleType The vehicle type to set
     */
    public void setVehicleType(String vehicleType) {
        this.vehicleType = vehicleType;
    }
    
    /**
     * Gets the number of vehicles needed.
     * 
     * @return The vehicle count
     */
    public int getVehicleCount() {
        return vehicleCount;
    }
    
    /**
     * Sets the number of vehicles needed.
     * 
     * @param vehicleCount The vehicle count to set
     */
    public void setVehicleCount(int vehicleCount) {
        this.vehicleCount = vehicleCount;
    }
    
    @Override
    public String toString() {
        return "ClusterMetrics{" +
                "communityId=" + communityId +
                ", totalDistance=" + totalDistance +
                ", totalCost=" + totalCost +
                ", vehicleType='" + vehicleType + '\'' +
                ", vehicleCount=" + vehicleCount +
                '}';
    }
} 