package com.izmir.transportation.persistence;

import java.util.Date;

/**
 * Data transfer object representing metadata about saved graphs.
 */
public class GraphInfo {
    private String filename;
    private int nodeCount;
    private String strategyName;
    private int edgeCount;
    private long fileSize;
    private Date lastModified;

    public String getFilename() {
        return filename;
    }

    public void setFilename(String filename) {
        this.filename = filename;
    }

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

    public int getEdgeCount() {
        return edgeCount;
    }

    public void setEdgeCount(int edgeCount) {
        this.edgeCount = edgeCount;
    }

    public long getFileSize() {
        return fileSize;
    }

    public void setFileSize(long fileSize) {
        this.fileSize = fileSize;
    }

    public Date getLastModified() {
        return lastModified;
    }

    public void setLastModified(Date lastModified) {
        this.lastModified = lastModified;
    }

    @Override
    public String toString() {
        return String.format(
            "Graph [%s]: %d nodes, %s strategy, %d edges, %.2f MB, last modified: %s",
            filename, nodeCount, strategyName, edgeCount, 
            fileSize / (1024.0 * 1024.0), lastModified
        );
    }
} 