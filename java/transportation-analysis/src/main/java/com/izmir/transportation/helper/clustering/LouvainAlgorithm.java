package com.izmir.transportation.helper.clustering;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import org.jgrapht.Graph;
import org.jgrapht.graph.DefaultWeightedEdge;

import com.izmir.transportation.TransportationGraph;
import com.izmir.transportation.helper.Node;

public class LouvainAlgorithm implements GraphClusteringAlgorithm {
    private final double resolution;
    private final int maxIterations;

    public LouvainAlgorithm(double resolution, int maxIterations) {
        this.resolution = resolution;
        this.maxIterations = maxIterations;
    }

    @Override
    public List<List<Node>> findCommunities(TransportationGraph graph) {
        Graph<Node, DefaultWeightedEdge> currentGraph = graph.getGraph();
        Map<Node, Integer> partition = initializeSingletonPartition(currentGraph);
        boolean changed = true;
        int iteration = 0;

        while (changed && iteration < maxIterations) {
            changed = false;
            // First phase: modularity optimization
            for (Node node : currentGraph.vertexSet()) {
                int bestCommunity = findBestCommunity(currentGraph, node, partition);
                if (bestCommunity != partition.get(node)) {
                    partition.put(node, bestCommunity);
                    changed = true;
                }
            }
            iteration++;
        }
        
        return convertPartitionToCommunities(partition);
    }

    private Map<Node, Integer> initializeSingletonPartition(Graph<Node, DefaultWeightedEdge> graph) {
        Map<Node, Integer> partition = new HashMap<>();
        int communityId = 0;
        for (Node node : graph.vertexSet()) {
            partition.put(node, communityId++);
        }
        return partition;
    }

    private int findBestCommunity(Graph<Node, DefaultWeightedEdge> graph, Node node, Map<Node, Integer> partition) {
        int currentCommunity = partition.get(node);
        double bestModularity = calculateModularity(graph, node, currentCommunity, partition);
        int bestCommunity = currentCommunity;

        Set<Integer> neighborCommunities = getNeighborCommunities(graph, node, partition);
        for (int community : neighborCommunities) {
            if (community != currentCommunity) {
                double modularity = calculateModularity(graph, node, community, partition);
                if (modularity > bestModularity) {
                    bestModularity = modularity;
                    bestCommunity = community;
                }
            }
        }

        return bestCommunity;
    }

    private Set<Integer> getNeighborCommunities(Graph<Node, DefaultWeightedEdge> graph, Node node, Map<Node, Integer> partition) {
        Set<Integer> communities = new HashSet<>();
        for (DefaultWeightedEdge edge : graph.edgesOf(node)) {
            Node neighbor = graph.getEdgeSource(edge);
            if (neighbor.equals(node)) {
                neighbor = graph.getEdgeTarget(edge);
            }
            communities.add(partition.get(neighbor));
        }
        return communities;
    }

    private double calculateModularity(Graph<Node, DefaultWeightedEdge> graph, Node node, int community, Map<Node, Integer> partition) {
        double internalWeight = 0.0;
        double totalWeight = 0.0;
        double communityWeight = 0.0;

        // Calculate weights
        for (DefaultWeightedEdge edge : graph.edgesOf(node)) {
            Node neighbor = graph.getEdgeSource(edge);
            if (neighbor.equals(node)) {
                neighbor = graph.getEdgeTarget(edge);
            }
            double weight = graph.getEdgeWeight(edge);
            totalWeight += weight;
            
            if (partition.get(neighbor) == community) {
                internalWeight += weight;
            }
        }

        // Calculate community weight
        for (Node n : graph.vertexSet()) {
            if (partition.get(n) == community) {
                for (DefaultWeightedEdge edge : graph.edgesOf(n)) {
                    communityWeight += graph.getEdgeWeight(edge);
                }
            }
        }

        double m2 = graph.edgeSet().stream()
                        .mapToDouble(graph::getEdgeWeight)
                        .sum() * 2;

        return internalWeight - resolution * (totalWeight * communityWeight) / m2;
    }

    private List<List<Node>> convertPartitionToCommunities(Map<Node, Integer> partition) {
        Map<Integer, List<Node>> communityMap = new HashMap<>();
        
        for (Map.Entry<Node, Integer> entry : partition.entrySet()) {
            communityMap.computeIfAbsent(entry.getValue(), k -> new ArrayList<>())
                       .add(entry.getKey());
        }
        
        return new ArrayList<>(communityMap.values());
    }
} 