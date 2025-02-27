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

/**
 * A custom implementation of the Leiden algorithm for community detection.
 * 
 * This implementation is now deprecated in favor of the LeidenClusteringAdapter
 * which uses the more efficient and accurate implementation from the leiden package.
 * 
 * @deprecated Use {@link LeidenClusteringAdapter} instead.
 */
@Deprecated
public class CustomLeidenAlgorithm implements GraphClusteringAlgorithm {
    private final double resolution;
    private final int iterations;

    public CustomLeidenAlgorithm(double resolution, int iterations) {
        this.resolution = resolution;
        this.iterations = iterations;
    }

    @Override
    public List<List<Node>> findCommunities(TransportationGraph graph) {
        Graph<Node, DefaultWeightedEdge> jgraphtGraph = graph.getGraph();
        
        Map<Node, Integer> partition = initializeSpatialPartition(jgraphtGraph);
        double totalGraphWeight = calculateTotalGraphWeight(jgraphtGraph);
        
        int maxIterationsWithoutImprovement = 5;
        int iterationsWithoutImprovement = 0;
        
        for (int i = 0; i < iterations && iterationsWithoutImprovement < maxIterationsWithoutImprovement; i++) {
            Map<Node, Integer> oldPartition = new HashMap<>(partition);
            
            partition = localMovingPhase(jgraphtGraph, partition, totalGraphWeight);
            partition = enforceMinimumCommunitySize(jgraphtGraph, partition, 15);
            partition = mergeSmallCommunities(jgraphtGraph, partition);
            partition = mergeSingletonCommunities(jgraphtGraph, partition);
            partition = forceMergeSmallCommunities(jgraphtGraph, partition, 20);
            
            if (partitionsSimilar(oldPartition, partition)) {
                iterationsWithoutImprovement++;
            } else {
                iterationsWithoutImprovement = 0;
            }
        }
        
        partition = forceMergeSmallCommunities(jgraphtGraph, partition, 20);
        return convertPartitionToCommunities(partition);
    }

    private double calculateTotalGraphWeight(Graph<Node, DefaultWeightedEdge> graph) {
        double totalWeight = 0.0;
        for (DefaultWeightedEdge e : graph.edgeSet()) {
            totalWeight += graph.getEdgeWeight(e);
        }
        return totalWeight;
    }

    private boolean partitionsSimilar(Map<Node, Integer> partition1, Map<Node, Integer> partition2) {
        int differences = 0;
        for (Map.Entry<Node, Integer> entry : partition1.entrySet()) {
            if (!entry.getValue().equals(partition2.get(entry.getKey()))) {
                differences++;
            }
        }
        return differences < partition1.size() * 0.05;
    }

    private Map<Node, Integer> initializeSpatialPartition(Graph<Node, DefaultWeightedEdge> graph) {
        Map<Node, Integer> partition = new HashMap<>();
        List<Node> nodes = new ArrayList<>(graph.vertexSet());
        Set<Node> unassigned = new HashSet<>(nodes);
        int communityId = 0;
        
        nodes.sort((n1, n2) -> Integer.compare(graph.degreeOf(n2), graph.degreeOf(n1)));
        
        while (!unassigned.isEmpty()) {
            // Pick the most connected unassigned node as the center
            Node center = nodes.stream()
                .filter(unassigned::contains)
                .findFirst()
                .orElse(unassigned.iterator().next());
                
            Set<Node> community = new HashSet<>();
            community.add(center);
            
            double avgDistance = calculateAverageDistance(graph, center);
            double distanceThreshold = Math.min(0.02, Math.max(0.005, avgDistance * 2));
            
            List<Node> nearbyNodes = unassigned.stream()
                .filter(n -> n != center && center.getLocation().distance(n.getLocation()) < distanceThreshold)
                .sorted((n1, n2) -> Double.compare(
                    center.getLocation().distance(n1.getLocation()),
                    center.getLocation().distance(n2.getLocation())))
                .toList();
            
            for (Node node : nearbyNodes) {
                if (community.size() >= 15) break;  
                community.add(node);
            }
            
            for (Node node : community) {
                partition.put(node, communityId);
                unassigned.remove(node);
            }
            communityId++;
        }
        
        return partition;
    }

    private double calculateAverageDistance(Graph<Node, DefaultWeightedEdge> graph, Node node) {
        List<Double> distances = new ArrayList<>();
        for (DefaultWeightedEdge edge : graph.edgesOf(node)) {
            Node neighbor = graph.getEdgeSource(edge);
            if (neighbor.equals(node)) {
                neighbor = graph.getEdgeTarget(edge);
            }
            distances.add(node.getLocation().distance(neighbor.getLocation()));
        }
        return distances.stream().mapToDouble(d -> d).average().orElse(0.01);
    }

    private Map<Node, Integer> forceMergeSmallCommunities(Graph<Node, DefaultWeightedEdge> graph, 
                                                         Map<Node, Integer> partition,
                                                         int minSize) {
        Map<Integer, List<Node>> communities = new HashMap<>();
        for (Node node : graph.vertexSet()) {
            communities.computeIfAbsent(partition.get(node), k -> new ArrayList<>()).add(node);
        }
        
        Map<Node, Integer> newPartition = new HashMap<>(partition);
        
        List<Integer> smallComms = communities.entrySet().stream()
            .filter(e -> e.getValue().size() < minSize)
            .map(Map.Entry::getKey)
            .toList();
        
        for (Integer smallComm : smallComms) {
            List<Node> smallCommNodes = communities.get(smallComm);
            
            Map<Integer, Double> commDistances = new HashMap<>();
            
            Map<Integer, double[]> commCenters = calculateCommunityCenters(communities);
            double[] smallCenter = commCenters.get(smallComm);
            
            for (Map.Entry<Integer, List<Node>> entry : communities.entrySet()) {
                int otherComm = entry.getKey();
                if (otherComm != smallComm && entry.getValue().size() >= minSize) {
                    double[] otherCenter = commCenters.get(otherComm);
                    double distance = Math.sqrt(
                        Math.pow(smallCenter[0] - otherCenter[0], 2) +
                        Math.pow(smallCenter[1] - otherCenter[1], 2)
                    );
                    commDistances.put(otherComm, distance);
                }
            }
            
            if (!commDistances.isEmpty()) {
                int targetComm = commDistances.entrySet().stream()
                    .min(Map.Entry.comparingByValue())
                    .map(Map.Entry::getKey)
                    .orElse(smallComm);
                
                for (Node node : smallCommNodes) {
                    newPartition.put(node, targetComm);
                }
            }
        }
        
        return newPartition;
    }

    private Map<Integer, double[]> calculateCommunityCenters(Map<Integer, List<Node>> communities) {
        Map<Integer, double[]> centers = new HashMap<>();
        
        for (Map.Entry<Integer, List<Node>> entry : communities.entrySet()) {
            double sumX = 0, sumY = 0;
            List<Node> nodes = entry.getValue();
            
            for (Node node : nodes) {
                sumX += node.getLocation().getX();
                sumY += node.getLocation().getY();
            }
            
            centers.put(entry.getKey(), new double[]{sumX / nodes.size(), sumY / nodes.size()});
        }
        
        return centers;
    }

    private Map<Node, Integer> mergeSmallCommunities(Graph<Node, DefaultWeightedEdge> graph, Map<Node, Integer> partition) {
        Map<Integer, List<Node>> communities = new HashMap<>();
        for (Node node : graph.vertexSet()) {
            communities.computeIfAbsent(partition.get(node), k -> new ArrayList<>()).add(node);
        }
        
        Map<Node, Integer> newPartition = new HashMap<>(partition);
        
        List<Map.Entry<Integer, List<Node>>> sortedCommunities = 
            communities.entrySet().stream()
                .sorted((e1, e2) -> Integer.compare(e1.getValue().size(), e2.getValue().size()))
                .toList();
        
        for (Map.Entry<Integer, List<Node>> entry : sortedCommunities) {
            if (entry.getValue().size() < 25) {  
                int sourceComm = entry.getKey();
                int bestTargetComm = findBestCommunityToMerge(graph, sourceComm, communities, partition);
                
                if (bestTargetComm != sourceComm) {
                    for (Node node : entry.getValue()) {
                        newPartition.put(node, bestTargetComm);
                    }
                }
            }
        }
        
        return newPartition;
    }

    private int findBestCommunityToMerge(Graph<Node, DefaultWeightedEdge> graph, 
                                       int sourceComm,
                                       Map<Integer, List<Node>> communities,
                                       Map<Node, Integer> partition) {
        Map<Integer, Double> commScores = new HashMap<>();
        List<Node> sourceCommunity = communities.get(sourceComm);
        
        double centerX = 0, centerY = 0;
        for (Node node : sourceCommunity) {
            centerX += node.getLocation().getX();
            centerY += node.getLocation().getY();
        }
        centerX /= sourceCommunity.size();
        centerY /= sourceCommunity.size();
        
        for (Node node : sourceCommunity) {
            for (DefaultWeightedEdge edge : graph.edgesOf(node)) {
                Node neighbor = graph.getEdgeSource(edge);
                if (neighbor.equals(node)) {
                    neighbor = graph.getEdgeTarget(edge);
                }
                
                int neighborComm = partition.get(neighbor);
                if (neighborComm != sourceComm) {
                    double weight = graph.getEdgeWeight(edge);
                    double distance = node.getLocation().distance(neighbor.getLocation());
                    double score = weight / (1 + distance);
                    
                    commScores.merge(neighborComm, score, Double::sum);
                }
            }
        }
        
        if (commScores.isEmpty()) {
            return sourceComm;
        }
        
        return commScores.entrySet().stream()
            .max(Map.Entry.comparingByValue())
            .map(Map.Entry::getKey)
            .orElse(sourceComm);
    }

    private double calculateQuality(Graph<Node, DefaultWeightedEdge> graph, 
                                  Node node, 
                                  int community, 
                                  Map<Node, Integer> partition) {
        double totalGraphWeight = calculateTotalGraphWeight(graph);
        double internalEdges = 0.0;
        double totalEdges = 0.0;
        double communityEdges = 0.0;
        int communitySize = 0;
        
        double centerX = 0, centerY = 0;
        int count = 0;
        for (Node n : graph.vertexSet()) {
            if (partition.get(n) == community) {
                centerX += n.getLocation().getX();
                centerY += n.getLocation().getY();
                count++;
                communitySize++;
            }
        }
        if (count > 0) {
            centerX /= count;
            centerY /= count;
        }
        
        for (DefaultWeightedEdge edge : graph.edgesOf(node)) {
            Node neighbor = graph.getEdgeSource(edge);
            if (neighbor.equals(node)) {
                neighbor = graph.getEdgeTarget(edge);
            }
            double weight = graph.getEdgeWeight(edge);
            double distance = node.getLocation().distance(neighbor.getLocation());
            
            double adjustedWeight = weight / (1 + distance);
            
            totalEdges += adjustedWeight;
            if (partition.get(neighbor) == community) {
                internalEdges += adjustedWeight;
                communityEdges += adjustedWeight;
            }
        }
        
        double distanceToCenter = Math.sqrt(
            Math.pow(node.getLocation().getX() - centerX, 2) +
            Math.pow(node.getLocation().getY() - centerY, 2)
        );
        
        double densityTerm = internalEdges / Math.max(1.0, totalEdges);
        double modularityTerm = (internalEdges / totalGraphWeight) - 
                               resolution * (totalEdges * communityEdges) / (totalGraphWeight * totalGraphWeight);
        
        double sizeBonus = communitySize < 20 ? 0.5 : 0.0;
        double sizePenalty = Math.abs(communitySize - 30) / 60.0;  
        
        return densityTerm + modularityTerm + sizeBonus - sizePenalty * resolution;
    }

    private Map<Node, Integer> initializeConnectedPartition(Graph<Node, DefaultWeightedEdge> graph) {
        Map<Node, Integer> partition = new HashMap<>();
        int communityId = 0;
        Set<Node> unassigned = new HashSet<>(graph.vertexSet());
        
        while (!unassigned.isEmpty()) {
            Node startNode = unassigned.iterator().next();
            Set<Node> community = new HashSet<>();
            community.add(startNode);
            
            for (DefaultWeightedEdge edge : graph.edgesOf(startNode)) {
                Node neighbor = graph.getEdgeSource(edge);
                if (neighbor.equals(startNode)) {
                    neighbor = graph.getEdgeTarget(edge);
                }
                if (unassigned.contains(neighbor) && graph.getEdgeWeight(edge) > 0.5) {
                    community.add(neighbor);
                }
            }
            
            for (Node node : community) {
                partition.put(node, communityId);
                unassigned.remove(node);
            }
            communityId++;
        }
        
        return partition;
    }

    private Map<Node, Integer> enforceMinimumCommunitySize(Graph<Node, DefaultWeightedEdge> graph, 
                                                         Map<Node, Integer> partition, 
                                                         int minSize) {
        Map<Integer, List<Node>> communities = new HashMap<>();
        for (Node node : graph.vertexSet()) {
            int community = partition.get(node);
            communities.computeIfAbsent(community, k -> new ArrayList<>()).add(node);
        }
        
        Map<Node, Integer> newPartition = new HashMap<>(partition);
        List<Integer> smallCommunities = communities.entrySet().stream()
            .filter(e -> e.getValue().size() < minSize)
            .map(Map.Entry::getKey)
            .toList();
        
        for (Integer smallCommunityId : smallCommunities) {
            List<Node> communityNodes = communities.get(smallCommunityId);
            
            Map<Integer, Double> connectionStrengths = new HashMap<>();
            
            for (Node node : communityNodes) {
                for (DefaultWeightedEdge edge : graph.edgesOf(node)) {
                    Node neighbor = graph.getEdgeSource(edge);
                    if (neighbor.equals(node)) {
                        neighbor = graph.getEdgeTarget(edge);
                    }
                    
                    int neighborCommunity = partition.get(neighbor);
                    if (neighborCommunity != smallCommunityId && 
                        communities.get(neighborCommunity).size() >= minSize) {
                        connectionStrengths.merge(neighborCommunity, 
                                                graph.getEdgeWeight(edge), 
                                                Double::sum);
                    }
                }
            }
            
            if (!connectionStrengths.isEmpty()) {
                int targetCommunity = connectionStrengths.entrySet().stream()
                    .max(Map.Entry.comparingByValue())
                    .map(Map.Entry::getKey)
                    .orElse(smallCommunityId);
                
                for (Node node : communityNodes) {
                    newPartition.put(node, targetCommunity);
                }
            }
        }
        
        return newPartition;
    }

    private Map<Node, Integer> localMovingPhase(Graph<Node, DefaultWeightedEdge> graph, 
                                              Map<Node, Integer> partition,
                                              double totalGraphWeight) {
        Map<Node, Integer> newPartition = new HashMap<>(partition);
        Map<Integer, Set<Node>> communities = new HashMap<>();
        boolean improved = true;
        int maxAttempts = 3; 
        int attempts = 0;
        
        for (Node node : graph.vertexSet()) {
            int community = partition.get(node);
            communities.computeIfAbsent(community, k -> new HashSet<>()).add(node);
        }
        
        while (improved && attempts < maxAttempts) {
            improved = false;
            attempts++;
            
            for (Node node : graph.vertexSet()) {
                int oldCommunity = newPartition.get(node);
                Set<Integer> neighborCommunities = getNeighborCommunities(graph, node, newPartition);
                
                double bestQuality = calculateQuality(graph, node, oldCommunity, newPartition);
                int bestCommunity = oldCommunity;
                
                for (int community : neighborCommunities) {
                    double quality = calculateQuality(graph, node, community, newPartition);
                    if (quality > bestQuality) {
                        bestQuality = quality;
                        bestCommunity = community;
                    }
                }
                
                if (bestCommunity != oldCommunity) {
                    newPartition.put(node, bestCommunity);
                    communities.get(oldCommunity).remove(node);
                    communities.computeIfAbsent(bestCommunity, k -> new HashSet<>()).add(node);
                    improved = true;
                }
            }
        }
        
        return newPartition;
    }

    private Map<Node, Integer> refinementPhase(Graph<Node, DefaultWeightedEdge> graph, Map<Node, Integer> partition) {
        Map<Node, Integer> refinedPartition = new HashMap<>(partition);
        for (Node node : graph.vertexSet()) {
            Set<Integer> neighborCommunities = getNeighborCommunities(graph, node, refinedPartition);
            double maxQuality = calculateQuality(graph, node, refinedPartition.get(node), refinedPartition);
            int bestCommunity = refinedPartition.get(node);
            
            for (int community : neighborCommunities) {
                double quality = calculateQuality(graph, node, community, refinedPartition);
                if (quality > maxQuality) {
                    maxQuality = quality;
                    bestCommunity = community;
                }
            }
            refinedPartition.put(node, bestCommunity);
        }
        return refinedPartition;
    }

    private Map<Node, Integer> aggregateGraph(Graph<Node, DefaultWeightedEdge> graph, Map<Node, Integer> partition) {
        Map<Node, Integer> aggregatedPartition = new HashMap<>();
        int newCommunityId = 0;
        Map<Integer, Integer> oldToNewCommunity = new HashMap<>();
        
        for (Node node : graph.vertexSet()) {
            int oldCommunity = partition.get(node);
            if (!oldToNewCommunity.containsKey(oldCommunity)) {
                oldToNewCommunity.put(oldCommunity, newCommunityId++);
            }
            aggregatedPartition.put(node, oldToNewCommunity.get(oldCommunity));
        }
        return aggregatedPartition;
    }

    private List<List<Node>> convertPartitionToCommunities(Map<Node, Integer> partition) {
        Map<Integer, List<Node>> communityMap = new HashMap<>();
        
        for (Map.Entry<Node, Integer> entry : partition.entrySet()) {
            communityMap.computeIfAbsent(entry.getValue(), k -> new ArrayList<>())
                       .add(entry.getKey());
        }
        
        return new ArrayList<>(communityMap.values());
    }

    private int findBestCommunity(Graph<Node, DefaultWeightedEdge> graph, Node node, Map<Node, Integer> partition) {
        Set<Integer> neighborCommunities = getNeighborCommunities(graph, node, partition);
        double maxQuality = calculateQuality(graph, node, partition.get(node), partition);
        int bestCommunity = partition.get(node);
        
        for (int community : neighborCommunities) {
            double quality = calculateQuality(graph, node, community, partition);
            if (quality > maxQuality) {
                maxQuality = quality;
                bestCommunity = community;
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

    private Map<Node, Integer> mergeSingletonCommunities(Graph<Node, DefaultWeightedEdge> graph, Map<Node, Integer> partition) {
        Map<Integer, List<Node>> communities = new HashMap<>();
        for (Node node : graph.vertexSet()) {
            int community = partition.get(node);
            communities.computeIfAbsent(community, k -> new ArrayList<>()).add(node);
        }
        
        List<Integer> singletonCommunities = communities.entrySet().stream()
            .filter(e -> e.getValue().size() == 1)
            .map(Map.Entry::getKey)
            .toList();
        
        Map<Node, Integer> newPartition = new HashMap<>(partition);
        
        for (Integer communityId : singletonCommunities) {
            Node singletonNode = communities.get(communityId).get(0);
            
            double bestQuality = Double.NEGATIVE_INFINITY;
            int bestCommunity = communityId;
            
            for (DefaultWeightedEdge edge : graph.edgesOf(singletonNode)) {
                Node neighbor = graph.getEdgeSource(edge);
                if (neighbor.equals(singletonNode)) {
                    neighbor = graph.getEdgeTarget(edge);
                }
                
                int neighborCommunity = partition.get(neighbor);
                if (neighborCommunity != communityId && communities.get(neighborCommunity).size() > 1) {
                    double quality = calculateMergeQuality(graph, singletonNode, neighborCommunity, partition);
                    if (quality > bestQuality) {
                        bestQuality = quality;
                        bestCommunity = neighborCommunity;
                    }
                }
            }
            
            if (bestCommunity != communityId) {
                newPartition.put(singletonNode, bestCommunity);
            }
        }
        
        return newPartition;
    }

    private double calculateMergeQuality(Graph<Node, DefaultWeightedEdge> graph, Node node, int targetCommunity, Map<Node, Integer> partition) {
        double connectionsToTarget = 0.0;
        double totalConnections = 0.0;
        
        for (DefaultWeightedEdge edge : graph.edgesOf(node)) {
            Node neighbor = graph.getEdgeSource(edge);
            if (neighbor.equals(node)) {
                neighbor = graph.getEdgeTarget(edge);
            }
            
            double weight = graph.getEdgeWeight(edge);
            totalConnections += weight;
            
            if (partition.get(neighbor) == targetCommunity) {
                connectionsToTarget += weight;
            }
        }
        
        return (connectionsToTarget / totalConnections) * (1.0 + Math.log(1.0 + connectionsToTarget));
    }
} 