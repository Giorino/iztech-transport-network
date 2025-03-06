package com.izmir.transportation;

import com.izmir.transportation.cost.TransportationCostAnalysis;
import com.izmir.transportation.helper.Node;
import com.izmir.transportation.helper.clustering.GraphClusteringAlgorithm;
import com.izmir.transportation.helper.clustering.LeidenCommunityDetection;
import com.izmir.transportation.helper.clustering.SpectralClustering;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Service for performing clustering operations on the transportation graph.
 * This allows for separation of graph construction and clustering operations.
 */
public class ClusteringService {
    private static final Logger LOGGER = LoggerFactory.getLogger(ClusteringService.class);
    
    /**
     * Available clustering algorithms
     */
    public enum ClusteringAlgorithm {
        LEIDEN("leiden"),
        SPECTRAL("spectral");
        
        private final String code;
        
        ClusteringAlgorithm(String code) {
            this.code = code;
        }
        
        public String getCode() {
            return code;
        }
        
        public static ClusteringAlgorithm fromCode(String code) {
            for (ClusteringAlgorithm algorithm : values()) {
                if (algorithm.getCode().equals(code)) {
                    return algorithm;
                }
            }
            throw new IllegalArgumentException("Unknown algorithm code: " + code);
        }
    }
    
    /**
     * Apply a clustering algorithm to the transportation graph
     *
     * @param graph The transportation graph to cluster
     * @param algorithm The clustering algorithm to use
     * @param visualize Whether to visualize the clusters
     * @return Map of community IDs to lists of nodes
     */
    public Map<Integer, List<Node>> performClustering(
            TransportationGraph graph, 
            ClusteringAlgorithm algorithm,
            boolean visualize) {
        
        LOGGER.info("Performing clustering using {} algorithm", algorithm);
        
        // Create the affinity matrix if it doesn't exist
        if (graph.getAffinityMatrix() == null) {
            graph.createAffinityMatrix();
        }
        
        // Apply the appropriate clustering algorithm
        Map<Integer, List<Node>> communities;
        
        if (algorithm == ClusteringAlgorithm.LEIDEN) {
            // Use Leiden algorithm
            LeidenCommunityDetection leidenAlgorithm = new LeidenCommunityDetection(graph);
            communities = leidenAlgorithm.detectCommunities();
        } else if (algorithm == ClusteringAlgorithm.SPECTRAL) {
            // Use Spectral algorithm
            SpectralClustering spectralAlgorithm = new SpectralClustering(graph);
            communities = spectralAlgorithm.detectCommunities();
        } else {
            throw new IllegalArgumentException("Unsupported algorithm: " + algorithm);
        }
        
        LOGGER.info("Found {} communities", communities.size());
        
        // Visualize the communities if requested
        if (visualize && !communities.isEmpty()) {
            List<List<Node>> communityList = new ArrayList<>(communities.values());
            graph.visualizeCommunities(communityList, algorithm.toString());
            
            // Save the community data
            graph.saveCommunityData(communities, algorithm.toString());
        }
        
        // Perform transportation cost analysis
        LOGGER.info("Performing transportation cost analysis...");
        performCostAnalysis(graph, communities);
        
        return communities;
    }
    
    /**
     * Performs transportation cost analysis on the detected communities
     *
     * @param graph The transportation graph
     * @param communities The detected communities
     */
    private void performCostAnalysis(TransportationGraph graph, Map<Integer, List<Node>> communities) {
        try {
            LOGGER.info("Starting transportation cost analysis...");
            TransportationCostAnalysis.analyzeCosts(graph, communities);
            LOGGER.info("Transportation cost analysis completed.");
        } catch (Exception e) {
            LOGGER.error("Error performing transportation cost analysis", e);
        }
    }
} 