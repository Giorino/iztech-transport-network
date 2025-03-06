package com.izmir.transportation.persistence;

/**
 * Data transfer object representing node data for persistence.
 */
public class NodeData {
    private String id;
    private double x;
    private double y;
    private boolean populationCenter;
    private double populationWeight;

    public String getId() {
        return id;
    }

    public void setId(String id) {
        this.id = id;
    }

    public double getX() {
        return x;
    }

    public void setX(double x) {
        this.x = x;
    }

    public double getY() {
        return y;
    }

    public void setY(double y) {
        this.y = y;
    }

    public boolean isPopulationCenter() {
        return populationCenter;
    }

    public void setPopulationCenter(boolean populationCenter) {
        this.populationCenter = populationCenter;
    }

    public double getPopulationWeight() {
        return populationWeight;
    }

    public void setPopulationWeight(double populationWeight) {
        this.populationWeight = populationWeight;
    }
} 