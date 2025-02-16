package com.izmir.transportation;

/**
 * Helper class to store point indices and their distances.
 * Used in the road network creation process to track distances between points
 * and their original indices in the point list.
 *
 * @author yagizugurveren
 */
public class PointDistance {
    /** Index of the point in the original list */
    final int index;

    /** Distance to the reference point */
    final double distance;

    /**
     * Constructs a new PointDistance object.
     *
     * @param index Index of the point in the original list
     * @param distance Euclidean distance to the reference point
     */
    public PointDistance(int index, double distance) {
        this.index = index;
        this.distance = distance;
    }

    /**
     * Gets the index of the point.
     *
     * @return The index of the point in the original list
     */
    public int getIndex() {
        return index;
    }

    /**
     * Gets the distance to the reference point.
     *
     * @return The Euclidean distance to the reference point
     */
    public double getDistance() {
        return distance;
    }
} 