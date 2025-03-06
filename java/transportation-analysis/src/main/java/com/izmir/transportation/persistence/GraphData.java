package com.izmir.transportation.persistence;

import java.util.List;

/**
 * Data transfer object representing graph data for persistence.
 */
public class GraphData {
    private int nodeCount;
    private String strategyName;
    private List<NodeData> nodes;
    private List<EdgeData> edges;

    public int getNodeCount() {
        return nodeCount;
    }

    public void setNodeCount(int nodeCount) {
        this.nodeCount = nodeCount;
    }

    public String getStrategyName() {
        return strategyName;
    }

    public void setStrategyName(String strategyName) {
        this.strategyName = strategyName;
    }

    public List<NodeData> getNodes() {
        return nodes;
    }

    public void setNodes(List<NodeData> nodes) {
        this.nodes = nodes;
    }

    public List<EdgeData> getEdges() {
        return edges;
    }

    public void setEdges(List<EdgeData> edges) {
        this.edges = edges;
    }
} 