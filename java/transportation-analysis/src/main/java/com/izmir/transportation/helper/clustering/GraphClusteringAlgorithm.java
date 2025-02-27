package com.izmir.transportation.helper.clustering;

import java.util.List;

import com.izmir.transportation.TransportationGraph;

public interface GraphClusteringAlgorithm {
    List<List<com.izmir.transportation.helper.Node>> findCommunities(TransportationGraph graph);
} 