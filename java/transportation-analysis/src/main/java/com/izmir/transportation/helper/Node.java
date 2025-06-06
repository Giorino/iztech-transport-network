package com.izmir.transportation.helper;

import org.locationtech.jts.geom.Point;

/**
 * Represents a node (vertex) in the transportation network.
 * Each node has a unique identifier and geographical coordinates.
 *
 * @author yagizugurveren
 */
public class Node {
    private final String id;
    private final Point location;
    private final boolean isPopulationCenter;
    private final double populationWeight;
    private boolean isOutlier;

    /**
     * Creates a new Node with the specified parameters.
     *
     * @param id Unique identifier for the node
     * @param location Geographical location of the node
     * @param isPopulationCenter Whether this node represents a population center
     * @param populationWeight Population weight (if this is a population center)
     */
    public Node(String id, Point location, boolean isPopulationCenter, double populationWeight) {
        this.id = id;
        this.location = location;
        this.isPopulationCenter = isPopulationCenter;
        this.populationWeight = populationWeight;
        this.isOutlier = false;
    }

    /**
     * Creates a new Node that is not a population center.
     *
     * @param id Unique identifier for the node
     * @param location Geographical location of the node
     */
    public Node(String id, Point location) {
        this(id, location, false, 0.0);
    }

    public String getId() {
        return id;
    }

    public Point getLocation() {
        return location;
    }

    public boolean isPopulationCenter() {
        return isPopulationCenter;
    }

    public double getPopulationWeight() {
        return populationWeight;
    }

    /**
     * Checks if this node is marked as an outlier.
     * 
     * @return true if the node is an outlier, false otherwise
     */
    public boolean isOutlier() {
        return isOutlier;
    }

    /**
     * Marks or unmarks this node as an outlier.
     * 
     * @param isOutlier true to mark as outlier, false otherwise
     */
    public void setOutlier(boolean isOutlier) {
        this.isOutlier = isOutlier;
    }

    /**
     * Resets the outlier status of this node to false.
     */
    public void resetOutlier() {
        this.isOutlier = false;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        Node node = (Node) o;
        return id.equals(node.id);
    }

    @Override
    public int hashCode() {
        return id.hashCode();
    }

    @Override
    public String toString() {
        return "Node{" +
                "id='" + id + '\'' +
                ", location=" + location +
                (isPopulationCenter ? ", populationWeight=" + populationWeight : "") +
                (isOutlier ? ", outlier=true" : "") +
                '}';
    }
} 