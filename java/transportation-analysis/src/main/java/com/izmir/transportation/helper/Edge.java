package com.izmir.transportation.helper;

import org.locationtech.jts.geom.LineString;

/**
 * Represents an edge in the transportation network.
 * Each edge connects two nodes and has a normalized weight representing the inverse of its distance.
 *
 * @author yagizugurveren
 */
public class Edge {
    private final String id;
    private final Node source;
    private final Node target;
    private final LineString geometry;
    private final double originalDistance; // in meters
    private double normalizedWeight; // between 0.001 and 0.999

    private static final double MIN_NORMALIZED_WEIGHT = 0.001;
    private static final double MAX_NORMALIZED_WEIGHT = 0.999;

    /**
     * Creates a new Edge with the specified parameters.
     *
     * @param id Unique identifier for the edge
     * @param source Source node
     * @param target Target node
     * @param geometry LineString representing the physical path of the edge
     * @param distance Distance in meters
     */
    public Edge(String id, Node source, Node target, LineString geometry, double distance) {
        this.id = id;
        this.source = source;
        this.target = target;
        this.geometry = geometry;
        this.originalDistance = distance;
        this.normalizedWeight = 0.0;
    }

    /**
     * Normalizes the weight of this edge based on the maximum distance in the network.
     * The normalized weight is inversely proportional to the distance.
     *
     * @param maxDistance The maximum distance found in the network
     */
    public void normalizeWeight(double maxDistance) {
        if (maxDistance <= 0) {
            throw new IllegalArgumentException("Maximum distance must be positive");
        }

        double normalizedDistance = MIN_NORMALIZED_WEIGHT +
            (originalDistance / maxDistance) * (MAX_NORMALIZED_WEIGHT - MIN_NORMALIZED_WEIGHT);

        // Normalize
        this.normalizedWeight = 1.0 - normalizedDistance;
    }

    public String getId() {
        return id;
    }

    public Node getSource() {
        return source;
    }

    public Node getTarget() {
        return target;
    }

    public LineString getGeometry() {
        return geometry;
    }

    public double getOriginalDistance() {
        return originalDistance;
    }

    public double getNormalizedWeight() {
        return normalizedWeight;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        Edge edge = (Edge) o;
        return id.equals(edge.id);
    }

    @Override
    public int hashCode() {
        return id.hashCode();
    }

    @Override
    public String toString() {
        return "Edge{" +
                "id='" + id + '\'' +
                ", source=" + source.getId() +
                ", target=" + target.getId() +
                ", distance=" + originalDistance +
                ", normalizedWeight=" + normalizedWeight +
                '}';
    }
} 