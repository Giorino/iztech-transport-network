package com.izmir.analysis.model;

/**
 * Model class representing transportation data from CSV files.
 * Contains information about a transportation route, including community ID,
 * node count, vehicle type, fuel consumption, etc.
 */
public class TransportationData {
    private int communityId;
    private int nodeCount;
    private String vehicleType;
    private int vehiclesRequired;
    private double totalDistanceKm;
    private double totalFuelLiters;
    private double fuelCost;
    private double fixedCost;
    private double totalCost;
    
    // Default constructor
    public TransportationData() {
    }
    
    // Constructor with all fields
    public TransportationData(int communityId, int nodeCount, String vehicleType, int vehiclesRequired,
                             double totalDistanceKm, double totalFuelLiters, double fuelCost,
                             double fixedCost, double totalCost) {
        this.communityId = communityId;
        this.nodeCount = nodeCount;
        this.vehicleType = vehicleType;
        this.vehiclesRequired = vehiclesRequired;
        this.totalDistanceKm = totalDistanceKm;
        this.totalFuelLiters = totalFuelLiters;
        this.fuelCost = fuelCost;
        this.fixedCost = fixedCost;
        this.totalCost = totalCost;
    }
    
    // Getters and setters
    public int getCommunityId() {
        return communityId;
    }
    
    public void setCommunityId(int communityId) {
        this.communityId = communityId;
    }
    
    public int getNodeCount() {
        return nodeCount;
    }
    
    public void setNodeCount(int nodeCount) {
        this.nodeCount = nodeCount;
    }
    
    public String getVehicleType() {
        return vehicleType;
    }
    
    public void setVehicleType(String vehicleType) {
        this.vehicleType = vehicleType;
    }
    
    public int getVehiclesRequired() {
        return vehiclesRequired;
    }
    
    public void setVehiclesRequired(int vehiclesRequired) {
        this.vehiclesRequired = vehiclesRequired;
    }
    
    public double getTotalDistanceKm() {
        return totalDistanceKm;
    }
    
    public void setTotalDistanceKm(double totalDistanceKm) {
        this.totalDistanceKm = totalDistanceKm;
    }
    
    public double getTotalFuelLiters() {
        return totalFuelLiters;
    }
    
    public void setTotalFuelLiters(double totalFuelLiters) {
        this.totalFuelLiters = totalFuelLiters;
    }
    
    public double getFuelCost() {
        return fuelCost;
    }
    
    public void setFuelCost(double fuelCost) {
        this.fuelCost = fuelCost;
    }
    
    public double getFixedCost() {
        return fixedCost;
    }
    
    public void setFixedCost(double fixedCost) {
        this.fixedCost = fixedCost;
    }
    
    public double getTotalCost() {
        return totalCost;
    }
    
    public void setTotalCost(double totalCost) {
        this.totalCost = totalCost;
    }
    
    @Override
    public String toString() {
        return "TransportationData{" +
                "communityId=" + communityId +
                ", nodeCount=" + nodeCount +
                ", vehicleType='" + vehicleType + '\'' +
                ", vehiclesRequired=" + vehiclesRequired +
                ", totalDistanceKm=" + totalDistanceKm +
                ", totalFuelLiters=" + totalFuelLiters +
                ", fuelCost=" + fuelCost +
                ", fixedCost=" + fixedCost +
                ", totalCost=" + totalCost +
                '}';
    }
} 