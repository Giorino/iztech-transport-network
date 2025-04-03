package com.izmir.transportation.cost;

/**
 * Simple data class to hold metrics for a single transportation cluster/community.
 *
 * @author yagizugurveren
 */
public class ClusterMetrics {
    private final int communityId;
    private final double totalDistanceKm;
    private final double totalCost;

    public ClusterMetrics(int communityId, double totalDistanceKm, double totalCost) {
        this.communityId = communityId;
        this.totalDistanceKm = totalDistanceKm;
        this.totalCost = totalCost;
    }

    public int getCommunityId() {
        return communityId;
    }

    public double getTotalDistanceKm() {
        return totalDistanceKm;
    }

    public double getTotalCost() {
        return totalCost;
    }

    @Override
    public String toString() {
        return "ClusterMetrics{" +
                "communityId=" + communityId +
                ", totalDistanceKm=" + totalDistanceKm +
                ", totalCost=" + totalCost +
                '}';
    }
} 