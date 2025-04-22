package com.izmir.transportation.helper.clustering;

import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Queue;
import java.util.Set;
import java.util.stream.Collectors;

import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.EigenDecomposition;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;
import org.apache.commons.math3.ml.clustering.CentroidCluster;
import org.apache.commons.math3.ml.clustering.Clusterable;
import org.apache.commons.math3.ml.clustering.KMeansPlusPlusClusterer;
import org.apache.commons.math3.ml.distance.EuclideanDistance;
import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;
import org.jgrapht.Graph;
import org.jgrapht.graph.DefaultWeightedEdge;
import org.locationtech.jts.geom.Coordinate;
import org.locationtech.jts.geom.Point;

import com.izmir.transportation.AffinityMatrix;
import com.izmir.transportation.TransportationGraph;
import com.izmir.transportation.helper.Edge;
import com.izmir.transportation.helper.Node;

/**
 * Implementation of Spectral Clustering for community detection in transportation networks.
 * 
 * Spectral clustering works by:
 * 1. Constructing a similarity matrix from the graph structure
 * 2. Computing the graph Laplacian matrix
 * 3. Finding eigenvalues and eigenvectors of the Laplacian
 * 4. Using the k smallest non-zero eigenvalues and corresponding eigenvectors
 * 5. Projecting nodes into this eigenspace and clustering with k-means
 * 
 * This implementation is specifically designed for transportation networks,
 * considering both network connectivity and geographic proximity.
 * 
 * @author yagizugurveren
 */
public class SpectralClustering implements GraphClusteringAlgorithm {
    
    // Default parameters for spectral clustering
    private static final double DEFAULT_SIGMA = 0.5;  // Controls similarity scaling
    private static final int DEFAULT_NUM_CLUSTERS = 8;
    private static final int DEFAULT_MAX_ITERATIONS = 100;
    private static final double DEFAULT_GEO_WEIGHT = 0.3;  // Weight for geographic distance
    private static final double EPSILON = 1e-10;  // Small value to avoid numerical issues
    
    // Configuration parameters
    private TransportationGraph transportationGraph;
    private int numberOfClusters = DEFAULT_NUM_CLUSTERS;
    private double sigma = DEFAULT_SIGMA;
    private boolean useNormalizedCut = true;
    private boolean useOriginalPointsOnly = false;
    private int maxIterations = DEFAULT_MAX_ITERATIONS;
    private double geoWeight = DEFAULT_GEO_WEIGHT;
    private int minCommunitySize = 1; // Minimum size for a community
    private boolean preventSingletons = false; // Whether to prevent singleton communities
    
    // Configuration object
    private SpectralClusteringConfig config;
    
    // Results storage
    private Map<Integer, List<Node>> communities;
    private double[] silhouetteScores;
    private double modularity;
    
    /**
     * Constructs a SpectralClustering instance for the given transportation graph.
     * 
     * @param transportationGraph The transportation graph to analyze
     */
    public SpectralClustering(TransportationGraph transportationGraph) {
        this.transportationGraph = transportationGraph;
        this.communities = new HashMap<>();
        this.config = new SpectralClusteringConfig();
    }
    
    /**
     * Constructs a SpectralClustering instance with a config object.
     * 
     * @param transportationGraph The transportation graph to analyze
     * @param config Configuration parameters for spectral clustering
     */
    public SpectralClustering(TransportationGraph transportationGraph, SpectralClusteringConfig config) {
        this.transportationGraph = transportationGraph;
        this.communities = new HashMap<>();
        this.config = config;
        
        // Apply configuration
        this.numberOfClusters = config.getNumberOfClusters();
        this.sigma = config.getSigma();
        this.useNormalizedCut = config.isUseNormalizedCut();
        this.maxIterations = config.getMaxIterations();
        this.geoWeight = config.getGeoWeight();
        this.minCommunitySize = config.getMinCommunitySize();
        this.preventSingletons = config.isPreventSingletons();
    }
    
    /**
     * Sets the configuration object for spectral clustering.
     * 
     * @param config The configuration object
     * @return This SpectralClustering instance for method chaining
     */
    public SpectralClustering setConfig(SpectralClusteringConfig config) {
        this.config = config;
        
        // Apply configuration
        this.numberOfClusters = config.getNumberOfClusters();
        this.sigma = config.getSigma();
        this.useNormalizedCut = config.isUseNormalizedCut();
        this.maxIterations = config.getMaxIterations();
        this.geoWeight = config.getGeoWeight();
        this.minCommunitySize = config.getMinCommunitySize();
        this.preventSingletons = config.isPreventSingletons();
        
        return this;
    }
    
    /**
     * Sets the number of clusters (communities) to find.
     * 
     * @param k Number of clusters to find (must be at least 2)
     * @return This SpectralClustering instance for method chaining
     */
    public SpectralClustering setNumberOfClusters(int k) {
        this.numberOfClusters = Math.max(2, k);
        return this;
    }
    
    /**
     * Sets the sigma parameter that controls the scaling of the similarity measure.
     * Higher values lead to more connections between distant nodes.
     * 
     * @param sigma Scaling parameter (typical range: 0.1 to 1.0)
     * @return This SpectralClustering instance for method chaining
     */
    public SpectralClustering setSigma(double sigma) {
        this.sigma = Math.max(0.01, sigma);
        return this;
    }
    
    /**
     * Sets whether to use normalized spectral clustering (normalized cut) or unnormalized.
     * Normalized cut typically produces more balanced communities.
     * 
     * @param useNormalizedCut True to use normalized cut, false for unnormalized
     * @return This SpectralClustering instance for method chaining
     */
    public SpectralClustering setUseNormalizedCut(boolean useNormalizedCut) {
        this.useNormalizedCut = useNormalizedCut;
        return this;
    }
    
    /**
     * Sets whether to use only original points for community detection or the entire graph.
     * Original points typically represent actual locations rather than road network points.
     * 
     * @param useOriginalPointsOnly True to use only original points, false to use all nodes
     * @return This SpectralClustering instance for method chaining
     */
    public SpectralClustering useOriginalPointsOnly(boolean useOriginalPointsOnly) {
        this.useOriginalPointsOnly = useOriginalPointsOnly;
        return this;
    }
    
    /**
     * Sets the maximum number of iterations for k-means clustering.
     * 
     * @param maxIterations Maximum number of iterations
     * @return This SpectralClustering instance for method chaining
     */
    public SpectralClustering setMaxIterations(int maxIterations) {
        this.maxIterations = Math.max(10, maxIterations);
        return this;
    }
    
    /**
     * Sets the weight given to geographic distance in the similarity calculation.
     * Higher values give more importance to geographic proximity.
     * 
     * @param geoWeight Weight for geographic distance (0.0 to 1.0)
     * @return This SpectralClustering instance for method chaining
     */
    public SpectralClustering setGeographicWeight(double geoWeight) {
        this.geoWeight = Math.max(0.0, Math.min(1.0, geoWeight));
        return this;
    }

    /**
     * Detects communities in the transportation network using spectral clustering.
     * 
     * @return A map of community IDs to lists of nodes in each community
     */
    public Map<Integer, List<Node>> detectCommunities() {
        try {
            // Get the appropriate graph based on configuration
            Graph<Node, DefaultWeightedEdge> graph;
            if (useOriginalPointsOnly) {
                graph = transportationGraph.getOriginalPointsGraph();
            } else {
                graph = transportationGraph.getGraph();
            }
            
            // Get all nodes from the original transportation graph
            Set<Node> allNodes = transportationGraph.getGraph().vertexSet();
            int totalNodeCount = allNodes.size();
            
            // Get nodes in a consistent order from the active graph
            List<Node> nodes = new ArrayList<>(graph.vertexSet());
            int activeNodeCount = nodes.size();
            
            System.out.println("SpectralClustering: Processing " + activeNodeCount + " connected nodes out of " + totalNodeCount + " total nodes");
            
            // If we're missing nodes in the active graph, log the discrepancy
            if (activeNodeCount < totalNodeCount) {
                System.out.println("WARNING: " + (totalNodeCount - activeNodeCount) + " nodes are isolated or not included in the active graph");
                System.out.println("These nodes will be assigned to communities in a post-processing step");
            }
            
            // First, identify disconnected components in the graph
            // These will be processed separately to ensure isolated clusters are preserved
            List<Set<Node>> connectedComponents = findAllConnectedComponents(graph);
            System.out.println("Found " + connectedComponents.size() + " disconnected components in the graph");
            
            // Get config values
            int maxClusters = config != null ? config.getNumberOfClusters() : numberOfClusters;
            int maxSize = config != null ? config.getMaxClusterSize() : 0;
            boolean forceNumClusters = config != null && config.isForceNumClusters();
            double maxDiameter = config != null ? config.getMaxCommunityDiameter() : 0.0;
            int minSize = config != null ? config.getMinCommunitySize() : minCommunitySize;
            
            // Initialize the communities map
            communities = new HashMap<>();
            int nextCommunityId = 0;
            
            // Process large components with spectral clustering
            // and small components with direct geographic clustering
            List<Set<Node>> largeComponents = new ArrayList<>();
            List<Set<Node>> smallComponents = new ArrayList<>();
            
            // Threshold for "small" components - use the minimum size as threshold
            // Small components below minSize will be preserved only if they're isolated
            int smallComponentThreshold = Math.max(minSize, 5);
            
            for (Set<Node> component : connectedComponents) {
                if (component.size() >= minSize || component.size() > smallComponentThreshold) {
                    largeComponents.add(component);
                } else {
                    smallComponents.add(component);
                }
            }
            
            System.out.println("Processing " + largeComponents.size() + " large components with spectral clustering");
            System.out.println("Processing " + smallComponents.size() + " small components with direct geographic clustering");
            
            // Process small components first
            // If a small component meets the minimum size requirement, keep it as-is
            // Otherwise, mark for later merging
            List<List<Node>> smallComponentsToMerge = new ArrayList<>();
            
            for (Set<Node> component : smallComponents) {
                if (component.size() >= minSize) {
                    // If component is large enough, keep it as a separate community
                    communities.put(nextCommunityId++, new ArrayList<>(component));
                    System.out.println("Preserved isolated component with " + component.size() + 
                                     " nodes as community " + (nextCommunityId-1) + " (meets minimum size)");
                } else {
                    // If too small, collect for later merging
                    smallComponentsToMerge.add(new ArrayList<>(component));
                    System.out.println("Small component with " + component.size() + 
                                     " nodes is below minimum size (" + minSize + 
                                     "), will be processed later");
                }
            }
            
            // For the remaining nodes in large components, apply spectral clustering
            if (!largeComponents.isEmpty()) {
                // Combine large components for spectral clustering
                Set<Node> remainingNodes = new HashSet<>();
                for (Set<Node> component : largeComponents) {
                    remainingNodes.addAll(component);
                }
                
                // Create a subgraph of just these nodes
                List<Node> largeComponentNodes = new ArrayList<>(remainingNodes);
                
                // Calculate how many clusters to allocate to the large components
                int remainingClusters = maxClusters - communities.size();
                if (remainingClusters <= 0) remainingClusters = maxClusters;
                
                try {
                    // Create similarity matrix for these nodes
                    RealMatrix similarityMatrix = buildSimilarityMatrix(graph, largeComponentNodes);
                    
                    // Compute Laplacian
                    RealMatrix laplacian;
                    if (useNormalizedCut) {
                        laplacian = computeNormalizedLaplacian(similarityMatrix);
                    } else {
                        laplacian = computeUnnormalizedLaplacian(similarityMatrix);
                    }
                    
                    // Add small perturbation for numerical stability
                    addNumericalStability(laplacian);
                    
                    // Compute eigendecomposition
                    System.out.println("Computing eigendecomposition...");
                    EigenDecomposition eigendecomposition = new EigenDecomposition(laplacian);
                    
                    // Get eigenvectors for embedding
                    RealMatrix embedding = getEigenvectorEmbedding(eigendecomposition);
                    
                    // Normalize rows if using normalized cut
                    if (useNormalizedCut) {
                        normalizeRows(embedding);
                    }
                    
                    // Cluster the embedding
                    System.out.println("Clustering embedding...");
                    Map<Integer, List<Node>> spectralCommunities = clusterEmbedding(embedding, largeComponentNodes);
                    
                    // Renumber the communities
                    for (List<Node> community : spectralCommunities.values()) {
                        communities.put(nextCommunityId++, community);
                    }
                    
                    // Handle maximum cluster size if needed
                    if (maxSize > 0) {
                        System.out.println("Splitting oversized communities to maximum size " + maxSize);
                        communities = enforceMaximumClusterSize(communities, maxSize);
                    }
                    
                    // Handle maximum community diameter if needed
                    if (maxDiameter > 0) {
                        System.out.println("Checking community diameters...");
                        communities = enforceMaximumDiameter(communities, maxDiameter);
                    }
                } catch (Exception e) {
                    System.err.println("Error during spectral clustering of large components: " + e.getMessage());
                    e.printStackTrace();
                    
                    // Fallback to geographic clustering for remaining nodes
                    System.out.println("Falling back to geographic clustering for large components due to error");
                    Map<Integer, List<Node>> fallbackCommunities = geographicClusteringWithMaxSize(
                            largeComponentNodes, maxSize, nextCommunityId);
                    communities.putAll(fallbackCommunities);
                }
            }
            
            // Force exact number of clusters if requested
            if (forceNumClusters && communities.size() != maxClusters) {
                System.out.println("Forcing exact number of clusters: " + maxClusters);
                communities = forceNumberOfClusters(communities, nodes, maxClusters);
            }
            
            // Handle small components that were too small for minimum requirements
            if (!smallComponentsToMerge.isEmpty()) {
                System.out.println("Processing " + smallComponentsToMerge.size() + 
                                 " small components that didn't meet minimum size requirement");
                
                // Process each small component and merge with closest community
                for (List<Node> smallComponent : smallComponentsToMerge) {
                    if (smallComponent.isEmpty()) continue;
                    
                    // Calculate centroid of this small component
                    double[] smallCentroid = calculateCentroid(smallComponent);
                    int closestCommunityId = -1;
                    double closestDistance = Double.MAX_VALUE;
                    
                    // Find closest community by centroid distance
                    for (Map.Entry<Integer, List<Node>> entry : communities.entrySet()) {
                        int communityId = entry.getKey();
                        List<Node> communityNodes = entry.getValue();
                        
                        // Skip if community is already too large
                        if (maxSize > 0 && communityNodes.size() + smallComponent.size() > maxSize) {
                            continue;
                        }
                        
                        double[] communityCentroid = calculateCentroid(communityNodes);
                        double distance = calculateDistance(smallCentroid, communityCentroid);
                        
                        if (distance < closestDistance) {
                            closestDistance = distance;
                            closestCommunityId = communityId;
                        }
                    }
                    
                    // If we found a community to merge with
                    if (closestCommunityId >= 0) {
                        communities.get(closestCommunityId).addAll(smallComponent);
                        System.out.println("Merged small component with " + smallComponent.size() + 
                                         " nodes into community " + closestCommunityId);
                    } else {
                        // If no community found (e.g., all are too large), create new community
                        communities.put(nextCommunityId, smallComponent);
                        System.out.println("Created new community " + nextCommunityId + 
                                         " for small component with " + smallComponent.size() + 
                                         " nodes (couldn't find suitable merge target)");
                        nextCommunityId++;
                    }
                }
            }
            
            // Balance community sizes if needed - always do this to enforce minimum size
            System.out.println("Performing final balancing to enforce minimum community size of " + minSize);
            communities = balanceCommunitySize(communities, minSize, maxSize);
            
            // IMPORTANT: Add missing/isolated nodes to communities - ALWAYS do this step
            System.out.println("Ensuring all nodes are assigned to communities...");
            communities = includeIsolatedNodes(communities, allNodes);
            
            // Calculate clustering quality metrics
            calculateClusteringMetrics(communities, graph);
            
            // Final verification of node count
            int assignedNodeCount = 0;
            for (List<Node> community : communities.values()) {
                assignedNodeCount += community.size();
            }
            
            if (assignedNodeCount != totalNodeCount) {
                System.err.println("ERROR: Node count mismatch after clustering! Expected " + totalNodeCount + 
                                  " nodes, but assigned " + assignedNodeCount + " nodes. Running emergency node assignment...");
                
                // Emergency fix - do one more includeIsolatedNodes pass to make sure ALL nodes are included
                communities = includeIsolatedNodes(communities, allNodes);
                
                // Re-count nodes
                assignedNodeCount = 0;
                for (List<Node> community : communities.values()) {
                    assignedNodeCount += community.size();
                }
                
                if (assignedNodeCount != totalNodeCount) {
                    System.err.println("CRITICAL ERROR: Still missing nodes after emergency fix. Expected " + 
                                      totalNodeCount + " nodes, but have " + assignedNodeCount);
                } else {
                    System.out.println("Emergency node assignment successful. All " + totalNodeCount + " nodes assigned.");
                }
            } else {
                System.out.println("Verification complete: All " + totalNodeCount + " nodes successfully assigned to communities.");
            }
            
            System.out.println("Spectral clustering complete. Found " + communities.size() + " communities with " + assignedNodeCount + " total nodes.");
            return communities;
            
        } catch (Exception e) {
            System.err.println("Error during spectral clustering: " + e.getMessage());
            e.printStackTrace();
            Map<Integer, List<Node>> fallback = geographicFallbackClustering(transportationGraph.getGraph().vertexSet(), 40);
            
            // Verify fallback assignment
            Set<Node> allNodes = transportationGraph.getGraph().vertexSet();
            int totalNodeCount = allNodes.size();
            int assignedNodeCount = 0;
            for (List<Node> community : fallback.values()) {
                assignedNodeCount += community.size();
            }
            
            if (assignedNodeCount != totalNodeCount) {
                System.err.println("WARNING: Fallback clustering assigned " + assignedNodeCount + 
                                  " nodes out of " + totalNodeCount + ". Running emergency node assignment...");
                fallback = includeIsolatedNodes(fallback, allNodes);
            }
            
            return fallback;
        }
    }
    
    @Override
    public List<List<Node>> findCommunities(TransportationGraph graph) {
        // Set our graph and detect communities
        // This method is required by GraphClusteringAlgorithm interface
        this.transportationGraph = graph;
        Map<Integer, List<Node>> communitiesMap = detectCommunities();
        return new ArrayList<>(communitiesMap.values());
    }
    
    /**
     * Builds a similarity matrix from the graph structure, incorporating both
     * network connectivity and geographic proximity.
     * 
     * @param graph The graph to analyze
     * @param nodes Ordered list of nodes
     * @return Similarity matrix
     */
    private RealMatrix buildSimilarityMatrix(Graph<Node, DefaultWeightedEdge> graph, List<Node> nodes) {
        int n = nodes.size();
        RealMatrix similarityMatrix = new Array2DRowRealMatrix(n, n);
        
        // Try to get existing affinity matrix (if available)
        AffinityMatrix existingMatrix = transportationGraph.getAffinityMatrix();
        Map<DefaultWeightedEdge, Edge> edgeMap = transportationGraph.getEdgeMap();
        
        // Get max geographic distance for normalization
        double maxGeoDist = 0.0;
        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) {
                double geoDist = calculateGeoDistance(nodes.get(i), nodes.get(j));
                maxGeoDist = Math.max(maxGeoDist, geoDist);
            }
        }
        
        // Use a reasonable sigma value based on the data
        double effectiveSigma = maxGeoDist / 20.0;
        System.out.println("Using effective sigma of " + effectiveSigma + " based on max distance of " + maxGeoDist);
        
        // Fill similarity matrix
        for (int i = 0; i < n; i++) {
            for (int j = i; j < n; j++) {
                double similarity;
                
                if (i == j) {
                    // Self-similarity is 1.0
                    similarity = 1.0;
                } else {
                    Node node1 = nodes.get(i);
                    Node node2 = nodes.get(j);
                    
                    // Calculate geographic similarity with more stable decay for distant nodes
                    double geoDist = calculateGeoDistance(node1, node2);
                    // Use standard Gaussian similarity with better numerical properties
                    double geoSimilarity = Math.exp(-geoDist / effectiveSigma);
                    
                    // Check if there's a direct edge in the graph
                    DefaultWeightedEdge edge = graph.getEdge(node1, node2);
                    
                    if (edge != null) {
                        // Get edge weight (normalized to 0-1 range)
                        double networkSimilarity;
                        
                        if (edgeMap.containsKey(edge)) {
                            // Use normalized weight from our edge map
                            networkSimilarity = 1.0 - edgeMap.get(edge).getNormalizedWeight();
                        } else {
                            // Fallback to graph edge weight
                            double weight = graph.getEdgeWeight(edge);
                            networkSimilarity = Math.exp(-weight / effectiveSigma);
                        }
                        
                        // Combine network and geographic similarity with balanced geo emphasis
                        similarity = (1.0 - geoWeight) * networkSimilarity + geoWeight * geoSimilarity;
                    } else {
                        // No direct edge, use reduced distance-based similarity
                        similarity = geoSimilarity * 0.5; // Apply moderate penalty for non-connected nodes
                    }
                    
                    // Ensure similarity is numerically stable
                    if (similarity < 1e-10) {
                        similarity = 0.0; // Treat very small values as zero
                    }
                }
                
                // Set symmetric values in the matrix
                similarityMatrix.setEntry(i, j, similarity);
                similarityMatrix.setEntry(j, i, similarity);
            }
        }
        
        // Count non-zero entries for diagnostic purposes
        int nonZeroEntries = 0;
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                if (similarityMatrix.getEntry(i, j) > 0.001) {
                    nonZeroEntries++;
                }
            }
        }
        System.out.println("Created similarity matrix with " + nonZeroEntries + " non-zero entries out of " + (n * n) + " total");
        
        return similarityMatrix;
    }
    
    /**
     * Computes the unnormalized Laplacian matrix: L = D - W
     * Where D is the degree matrix and W is the similarity matrix.
     * 
     * @param similarityMatrix The similarity matrix
     * @return Unnormalized Laplacian matrix
     */
    private RealMatrix computeUnnormalizedLaplacian(RealMatrix similarityMatrix) {
        int n = similarityMatrix.getRowDimension();
        RealMatrix laplacian = new Array2DRowRealMatrix(n, n);
        
        // Compute degree matrix and subtract similarity matrix
        for (int i = 0; i < n; i++) {
            double degree = 0.0;
            for (int j = 0; j < n; j++) {
                degree += similarityMatrix.getEntry(i, j);
            }
            
            for (int j = 0; j < n; j++) {
                if (i == j) {
                    laplacian.setEntry(i, j, degree - similarityMatrix.getEntry(i, j));
                } else {
                    laplacian.setEntry(i, j, -similarityMatrix.getEntry(i, j));
                }
            }
        }
        
        return laplacian;
    }
    
    /**
     * Computes the normalized Laplacian matrix: L = I - D^(-1/2) W D^(-1/2)
     * Where D is the degree matrix, W is the similarity matrix, and I is the identity matrix.
     * 
     * @param similarityMatrix The similarity matrix
     * @return Normalized Laplacian matrix
     */
    private RealMatrix computeNormalizedLaplacian(RealMatrix similarityMatrix) {
        int n = similarityMatrix.getRowDimension();
        
        // Compute degrees for each node
        double[] degrees = new double[n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                degrees[i] += similarityMatrix.getEntry(i, j);
            }
            // Ensure no division by zero
            degrees[i] = Math.max(degrees[i], EPSILON);
        }
        
        // Compute D^(-1/2)
        double[] sqrtInvDegrees = new double[n];
        for (int i = 0; i < n; i++) {
            sqrtInvDegrees[i] = 1.0 / Math.sqrt(degrees[i]);
        }
        
        // Compute normalized Laplacian: I - D^(-1/2) W D^(-1/2)
        RealMatrix identity = MatrixUtils.createRealIdentityMatrix(n);
        RealMatrix laplacian = new Array2DRowRealMatrix(n, n);
        
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                double value = sqrtInvDegrees[i] * similarityMatrix.getEntry(i, j) * sqrtInvDegrees[j];
                laplacian.setEntry(i, j, i == j ? 1.0 - value : -value);
            }
        }
        
        return laplacian;
    }
    
    /**
     * Adds a small value to the diagonal of the matrix for numerical stability.
     * 
     * @param matrix The matrix to stabilize
     */
    private void addNumericalStability(RealMatrix matrix) {
        for (int i = 0; i < matrix.getRowDimension(); i++) {
            matrix.setEntry(i, i, matrix.getEntry(i, i) + EPSILON);
        }
    }
    
    /**
     * Gets the eigenvectors corresponding to the k smallest non-zero eigenvalues
     * and creates an embedding matrix where each row represents a node.
     * 
     * @param eigendecomposition Eigendecomposition of the Laplacian
     * @return Matrix of eigenvector embedding
     */
    private RealMatrix getEigenvectorEmbedding(EigenDecomposition eigendecomposition) {
        int n = eigendecomposition.getRealEigenvalues().length;
        int k = Math.min(numberOfClusters, n - 1);
        
        // Get eigenvalues and sort indices by eigenvalue magnitude
        double[] eigenvalues = eigendecomposition.getRealEigenvalues();
        
        // Create array of indices and sort it
        Integer[] indices = new Integer[n];
        for (int i = 0; i < n; i++) {
            indices[i] = i;
        }
        
        Arrays.sort(indices, new Comparator<Integer>() {
            @Override
            public int compare(Integer i, Integer j) {
                return Double.compare(Math.abs(eigenvalues[i]), Math.abs(eigenvalues[j]));
            }
        });
        
        // Skip the smallest eigenvalue (which should be near zero)
        // and get the next k eigenvalues
        int[] selectedIndices = new int[k];
        for (int i = 0; i < k; i++) {
            selectedIndices[i] = indices[i + 1];  // Skip first (smallest)
        }
        
        // Print eigenvalues for debugging
        System.out.println("Selected eigenvalues:");
        for (int i = 0; i < k; i++) {
            System.out.printf("λ%d = %.6f%n", i+1, eigenvalues[selectedIndices[i]]);
        }
        
        // Extract corresponding eigenvectors
        double[][] embedding = new double[n][k];
        for (int i = 0; i < k; i++) {
            RealVector eigenvector = eigendecomposition.getEigenvector(selectedIndices[i]);
            for (int j = 0; j < n; j++) {
                embedding[j][i] = eigenvector.getEntry(j);
            }
        }
        
        return new Array2DRowRealMatrix(embedding);
    }
    
    /**
     * Normalizes each row in the matrix to have unit length.
     * This is required for normalized spectral clustering.
     * 
     * @param matrix The matrix to normalize
     */
    private void normalizeRows(RealMatrix matrix) {
        for (int i = 0; i < matrix.getRowDimension(); i++) {
            double[] row = matrix.getRow(i);
            double norm = 0.0;
            
            for (double val : row) {
                norm += val * val;
            }
            
            norm = Math.sqrt(norm);
            
            if (norm > EPSILON) {
                for (int j = 0; j < row.length; j++) {
                    matrix.setEntry(i, j, row[j] / norm);
                }
            }
        }
    }
    
    /**
     * Clusters the embedding matrix using k-means and maps clusters back to nodes.
     * 
     * @param embedding Matrix of eigenvector embedding
     * @param nodes List of nodes in the same order as the embedding rows
     * @return Map of community IDs to lists of nodes
     */
    private Map<Integer, List<Node>> clusterEmbedding(RealMatrix embedding, List<Node> nodes) {
        Map<Integer, List<Node>> result = new HashMap<>();
        
        // Convert embedding to points for k-means
        List<EmbeddingPoint> points = new ArrayList<>();
        for (int i = 0; i < embedding.getRowDimension(); i++) {
            points.add(new EmbeddingPoint(embedding.getRow(i), i));
        }
        
        // Determine appropriate number of clusters based on minCommunitySize
        int effectiveNumClusters = numberOfClusters;
        if (minCommunitySize > 1) {
            // Calculate maximum possible number of clusters given minimum size
            int maxPossibleClusters = nodes.size() / minCommunitySize;
            effectiveNumClusters = Math.min(numberOfClusters, maxPossibleClusters);
            System.out.println("Adjusted number of clusters from " + numberOfClusters + 
                           " to " + effectiveNumClusters + " based on minimum community size " + 
                           minCommunitySize);
        }
        
        // Ensure we have at least 2 clusters
        effectiveNumClusters = Math.max(2, effectiveNumClusters);
        
        // Perform k-means clustering
        KMeansPlusPlusClusterer<EmbeddingPoint> clusterer = 
            new KMeansPlusPlusClusterer<>(effectiveNumClusters, maxIterations, new EuclideanDistance());
        
        List<CentroidCluster<EmbeddingPoint>> clusters = clusterer.cluster(points);
        
        // Map clusters back to nodes
        for (int i = 0; i < clusters.size(); i++) {
            CentroidCluster<EmbeddingPoint> cluster = clusters.get(i);
            List<Node> community = new ArrayList<>();
            
            for (EmbeddingPoint point : cluster.getPoints()) {
                int nodeIndex = point.getIndex();
                community.add(nodes.get(nodeIndex));
            }
            
            result.put(i, community);
        }
        
        // Check for small communities and handle them if needed
        if (preventSingletons || minCommunitySize > 1) {
            result = handleSmallCommunities(result, nodes);
        }
        
        return result;
    }
    
    /**
     * Handles small communities based on configuration settings.
     * Merges communities smaller than minCommunitySize into their closest neighbors.
     * 
     * @param communities The original communities
     * @param nodes All nodes in the graph
     * @return Updated communities with small ones merged
     */
    private Map<Integer, List<Node>> handleSmallCommunities(Map<Integer, List<Node>> communities, List<Node> nodes) {
        // Make a working copy
        Map<Integer, List<Node>> result = new HashMap<>(communities);
        
        // Identify small communities
        List<Integer> smallCommunityIds = new ArrayList<>();
        for (Map.Entry<Integer, List<Node>> entry : result.entrySet()) {
            if (entry.getValue().size() < minCommunitySize || 
                (preventSingletons && entry.getValue().size() == 1)) {
                smallCommunityIds.add(entry.getKey());
            }
        }
        
        if (smallCommunityIds.isEmpty()) {
            return result; // No small communities to handle
        }
        
        System.out.println("Found " + smallCommunityIds.size() + 
                       " communities smaller than minimum size " + minCommunitySize);
        
        // Calculate centroids for all communities (for distance calculations)
        Map<Integer, double[]> centroids = new HashMap<>();
        for (Map.Entry<Integer, List<Node>> entry : result.entrySet()) {
            centroids.put(entry.getKey(), calculateCentroid(entry.getValue()));
        }
        
        // Process each small community
        for (Integer smallId : smallCommunityIds) {
            List<Node> smallCommunity = result.get(smallId);
            if (smallCommunity == null || smallCommunity.isEmpty()) {
                continue; // Skip if already processed
            }
            
            // Find closest community
            int closestId = -1;
            double minDistance = Double.MAX_VALUE;
            
            double[] smallCentroid = centroids.get(smallId);
            
            for (Map.Entry<Integer, double[]> entry : centroids.entrySet()) {
                int otherId = entry.getKey();
                if (otherId == smallId || smallCommunityIds.contains(otherId)) {
                    continue; // Skip self or other small communities
                }
                
                double dist = calculateDistance(smallCentroid, entry.getValue());
                if (dist < minDistance) {
                    minDistance = dist;
                    closestId = otherId;
                }
            }
            
            // If we found a community to merge with
            if (closestId != -1) {
                // Merge small community into closest
                List<Node> targetCommunity = result.get(closestId);
                targetCommunity.addAll(smallCommunity);
                
                // Update centroid of target community
                centroids.put(closestId, calculateCentroid(targetCommunity));
                
                // Remove small community
                result.remove(smallId);
                centroids.remove(smallId);
                
                System.out.println("Merged community " + smallId + " (size " + smallCommunity.size() + 
                               ") into community " + closestId + " (new size " + targetCommunity.size() + ")");
            }
        }
        
        return result;
    }
    
    /**
     * Calculate the centroid (average position) of a set of nodes
     * 
     * @param nodes The nodes
     * @return The centroid as a double array
     */
    private double[] calculateCentroid(List<Node> nodes) {
        if (nodes == null || nodes.isEmpty()) {
            return new double[2];
        }
        
        double[] sum = new double[2];
        for (Node node : nodes) {
            Point point = node.getLocation();
            sum[0] += point.getX();
            sum[1] += point.getY();
        }
        
        return new double[] { sum[0] / nodes.size(), sum[1] / nodes.size() };
    }
    
    /**
     * Calculate Euclidean distance between two points
     * 
     * @param p1 First point
     * @param p2 Second point
     * @return Euclidean distance
     */
    private double calculateDistance(double[] p1, double[] p2) {
        double dx = p1[0] - p2[0];
        double dy = p1[1] - p2[1];
        return Math.sqrt(dx * dx + dy * dy);
    }
    
    /**
     * Calculates metrics to evaluate the quality of the clustering.
     * 
     * @param communities Map of community IDs to lists of nodes
     * @param graph The graph being analyzed
     */
    private void calculateClusteringMetrics(Map<Integer, List<Node>> communities, Graph<Node, DefaultWeightedEdge> graph) {
        // Calculate silhouette scores
        silhouetteScores = calculateSilhouetteScores(communities, graph);
        
        // Calculate modularity
        modularity = calculateModularity(communities, graph);
    }
    
    /**
     * Calculates silhouette scores for each community to measure cluster quality.
     * The silhouette score measures how similar an object is to its own cluster
     * compared to other clusters.
     * 
     * @param communities Map of community IDs to lists of nodes
     * @param graph The graph being analyzed
     * @return Array of silhouette scores for each community
     */
    private double[] calculateSilhouetteScores(Map<Integer, List<Node>> communities, Graph<Node, DefaultWeightedEdge> graph) {
        double[] scores = new double[communities.size()];
        int communityIndex = 0;
        
        for (Map.Entry<Integer, List<Node>> entry : communities.entrySet()) {
            List<Node> community = entry.getValue();
            double communityScore = 0.0;
            
            for (Node node : community) {
                // Calculate average distance to nodes in same community (a)
                double a = calculateAverageDistanceWithin(node, community, graph);
                
                // Calculate minimum average distance to nodes in other communities (b)
                double b = Double.MAX_VALUE;
                
                for (Map.Entry<Integer, List<Node>> otherEntry : communities.entrySet()) {
                    if (!otherEntry.getKey().equals(entry.getKey())) {
                        double avgDist = calculateAverageDistanceWithin(node, otherEntry.getValue(), graph);
                        b = Math.min(b, avgDist);
                    }
                }
                
                // Calculate silhouette score for this node
                double silhouette;
                if (community.size() <= 1) {
                    silhouette = 0.0;  // Single node community
                } else if (a < b) {
                    silhouette = 1.0 - (a / b);
                } else if (a > b) {
                    silhouette = (b / a) - 1.0;
                } else {
                    silhouette = 0.0;
                }
                
                communityScore += silhouette;
            }
            
            // Average silhouette score for this community
            scores[communityIndex++] = community.isEmpty() ? 0.0 : communityScore / community.size();
        }
        
        return scores;
    }
    
    /**
     * Calculates the average distance between a node and all nodes in a community.
     * 
     * @param node The node to calculate distances from
     * @param community The community to calculate distances to
     * @param graph The graph containing the nodes
     * @return Average distance
     */
    private double calculateAverageDistanceWithin(Node node, List<Node> community, Graph<Node, DefaultWeightedEdge> graph) {
        if (community.size() <= 1) {
            return 0.0;
        }
        
        double totalDistance = 0.0;
        int count = 0;
        
        for (Node other : community) {
            if (!node.equals(other)) {
                // Use both network distance and geographic distance
                double geoDist = calculateGeoDistance(node, other);
                
                // Check if there's a direct edge
                DefaultWeightedEdge edge = graph.getEdge(node, other);
                double networkDist = edge != null ? graph.getEdgeWeight(edge) : Double.MAX_VALUE;
                
                // Combine distances with the same weighting as in similarity calculation
                double combinedDist = (1.0 - geoWeight) * networkDist + geoWeight * geoDist;
                
                totalDistance += combinedDist;
                count++;
            }
        }
        
        return count > 0 ? totalDistance / count : 0.0;
    }
    
    /**
     * Calculates the modularity of the clustering, which measures the strength
     * of division into communities.
     * 
     * @param communities Map of community IDs to lists of nodes
     * @param graph The graph being analyzed
     * @return Modularity value (higher is better)
     */
    private double calculateModularity(Map<Integer, List<Node>> communities, Graph<Node, DefaultWeightedEdge> graph) {
        double m = graph.edgeSet().size();
        if (m == 0) return 0.0;
        
        // Create mapping from node to community ID
        Map<Node, Integer> nodeCommunities = new HashMap<>();
        for (Map.Entry<Integer, List<Node>> entry : communities.entrySet()) {
            for (Node node : entry.getValue()) {
                nodeCommunities.put(node, entry.getKey());
            }
        }
        
        double modularity = 0.0;
        
        for (DefaultWeightedEdge e : graph.edgeSet()) {
            Node source = graph.getEdgeSource(e);
            Node target = graph.getEdgeTarget(e);
            
            Integer sourceComm = nodeCommunities.get(source);
            Integer targetComm = nodeCommunities.get(target);
            
            if (sourceComm != null && targetComm != null && sourceComm.equals(targetComm)) {
                // Edge is within a community
                double edgeWeight = graph.getEdgeWeight(e);
                modularity += edgeWeight - (graph.degreeOf(source) * graph.degreeOf(target)) / (2 * m);
            }
        }
        
        return modularity / (2 * m);
    }
    
    /**
     * Calculates the geographic distance between two nodes.
     * 
     * @param n1 First node
     * @param n2 Second node
     * @return Euclidean distance in the geographic coordinate system
     */
    private double calculateGeoDistance(Node n1, Node n2) {
        // Robustness check: handle null nodes or null locations
        if (n1 == null || n2 == null) {
            return Double.MAX_VALUE;
        }
        
        Point p1 = n1.getLocation();
        Point p2 = n2.getLocation();
        
        // Robustness check: handle null coordinates
        if (p1 == null || p2 == null) {
            return Double.MAX_VALUE;
        }
        
        // Get coordinates
        Coordinate c1 = p1.getCoordinate();
        Coordinate c2 = p2.getCoordinate();
        
        // Robustness check: handle null or invalid coordinates
        if (c1 == null || c2 == null || 
            Double.isNaN(c1.x) || Double.isNaN(c1.y) || 
            Double.isNaN(c2.x) || Double.isNaN(c2.y) ||
            Double.isInfinite(c1.x) || Double.isInfinite(c1.y) ||
            Double.isInfinite(c2.x) || Double.isInfinite(c2.y)) {
            return Double.MAX_VALUE;
        }
        
        // Using Euclidean distance for simplicity
        // In a real-world scenario, using Haversine formula would be more accurate
        double dx = c1.x - c2.x;
        double dy = c1.y - c2.y;
        double distance = Math.sqrt(dx * dx + dy * dy);
        
        // Final check to ensure distance is valid
        if (Double.isNaN(distance) || Double.isInfinite(distance)) {
            return Double.MAX_VALUE;
        }
        
        return distance;
    }
    
    /**
     * Provides a fallback clustering method when spectral clustering fails.
     * This uses a simpler approach based on node degree and geographic proximity.
     * 
     * @return Map of community IDs to lists of nodes
     */
    private Map<Integer, List<Node>> fallbackClustering() {
        System.out.println("Using fallback clustering method...");
        Map<Integer, List<Node>> fallbackCommunities = new HashMap<>();
        
        try {
            Graph<Node, DefaultWeightedEdge> graph;
            if (useOriginalPointsOnly) {
                graph = transportationGraph.getOriginalPointsGraph();
            } else {
                graph = transportationGraph.getGraph();
            }
            
            // Sort nodes by degree (higher degree first)
            List<Node> nodes = new ArrayList<>(graph.vertexSet());
            nodes.sort((a, b) -> Integer.compare(graph.degreeOf(b), graph.degreeOf(a)));
            
            // Select seed nodes (high-degree nodes spread geographically)
            List<Node> seeds = new ArrayList<>();
            double minDistBetweenSeeds = 0.05;  // Minimum distance between seeds
            
            for (Node node : nodes) {
                boolean tooClose = false;
                
                for (Node seed : seeds) {
                    if (calculateGeoDistance(node, seed) < minDistBetweenSeeds) {
                        tooClose = true;
                        break;
                    }
                }
                
                if (!tooClose) {
                    seeds.add(node);
                }
                
                if (seeds.size() >= numberOfClusters) {
                    break;
                }
            }
            
            // Ensure we have enough seeds (fallback to top degree nodes if needed)
            while (seeds.size() < numberOfClusters && seeds.size() < nodes.size()) {
                Node candidate = nodes.get(seeds.size());
                if (!seeds.contains(candidate)) {
                    seeds.add(candidate);
                }
            }
            
            // Assign each node to closest seed
            for (int i = 0; i < seeds.size(); i++) {
                fallbackCommunities.put(i, new ArrayList<>());
                fallbackCommunities.get(i).add(seeds.get(i));  // Add seed to its own community
            }
            
            for (Node node : nodes) {
                if (seeds.contains(node)) continue;  // Skip seeds
                
                int closestSeed = -1;
                double minDist = Double.MAX_VALUE;
                
                for (int i = 0; i < seeds.size(); i++) {
                    double dist = calculateGeoDistance(node, seeds.get(i));
                    if (dist < minDist) {
                        minDist = dist;
                        closestSeed = i;
                    }
                }
                
                if (closestSeed >= 0) {
                    fallbackCommunities.get(closestSeed).add(node);
                }
            }
            
            // Store results
            this.communities = fallbackCommunities;
            
        } catch (Exception e) {
            System.err.println("Error in fallback clustering: " + e.getMessage());
            e.printStackTrace();
            
            // Final fallback: create a single community with all nodes
            List<Node> allNodes = new ArrayList<>(transportationGraph.getGraph().vertexSet());
            fallbackCommunities.put(0, allNodes);
            this.communities = fallbackCommunities;
        }
        
        return fallbackCommunities;
    }
    
    /**
     * Gets detailed statistics about the detected communities.
     * 
     * @return String containing community statistics
     */
    public String getCommunityStatistics() {
        if (communities == null || communities.isEmpty()) {
            return "No communities detected yet. Call detectCommunities() first.";
        }
        
        StringBuilder sb = new StringBuilder();
        sb.append("\n=== SPECTRAL CLUSTERING COMMUNITY STATISTICS ===\n\n");
        
        // Overall statistics
        sb.append(String.format("Total communities: %d\n", communities.size()));
        
        int totalNodes = 0;
        for (List<Node> community : communities.values()) {
            totalNodes += community.size();
        }
        sb.append(String.format("Total nodes: %d\n", totalNodes));
        
        // Parameters used
        sb.append(String.format("\nParameters:\n"));
        sb.append(String.format("- Number of clusters (k): %d\n", numberOfClusters));
        sb.append(String.format("- Sigma (similarity scaling): %.2f\n", sigma));
        sb.append(String.format("- Using normalized cut: %s\n", useNormalizedCut));
        sb.append(String.format("- Using original points only: %s\n", useOriginalPointsOnly));
        sb.append(String.format("- Geographic weight: %.2f\n", geoWeight));
        
        // Quality metrics
        sb.append(String.format("\nClustering quality:\n"));
        sb.append(String.format("- Modularity: %.4f\n", modularity));
        
        if (silhouetteScores != null) {
            double avgSilhouette = 0.0;
            for (double score : silhouetteScores) {
                avgSilhouette += score;
            }
            avgSilhouette /= silhouetteScores.length;
            
            sb.append(String.format("- Average silhouette score: %.4f\n", avgSilhouette));
        }
        
        // Community details
        sb.append("\nCommunity details:\n");
        DecimalFormat df = new DecimalFormat("#.##");
        
        DescriptiveStatistics communitySizeStats = new DescriptiveStatistics();
        
        for (Map.Entry<Integer, List<Node>> entry : communities.entrySet()) {
            int communityId = entry.getKey();
            List<Node> community = entry.getValue();
            
            communitySizeStats.addValue(community.size());
            
            sb.append(String.format("Community %d: %d nodes", communityId, community.size()));
            
            if (silhouetteScores != null && communityId < silhouetteScores.length) {
                sb.append(String.format(" (silhouette: %s)", df.format(silhouetteScores[communityId])));
            }
            
            sb.append("\n");
        }
        
        // Distribution statistics
        sb.append("\nCommunity size distribution:\n");
        sb.append(String.format("- Min size: %.0f nodes\n", communitySizeStats.getMin()));
        sb.append(String.format("- Max size: %.0f nodes\n", communitySizeStats.getMax()));
        sb.append(String.format("- Mean size: %.2f nodes\n", communitySizeStats.getMean()));
        sb.append(String.format("- Standard deviation: %.2f nodes\n", communitySizeStats.getStandardDeviation()));
        
        // Calculate Gini coefficient to measure inequality in community sizes
        double gini = calculateGiniCoefficient(communities);
        sb.append(String.format("- Gini coefficient (size inequality): %.4f\n", gini));
        
        sb.append("\n=== END OF SPECTRAL CLUSTERING STATISTICS ===\n");
        
        return sb.toString();
    }
    
    /**
     * Calculates the Gini coefficient to measure inequality in community sizes.
     * 
     * @param communities Map of community IDs to lists of nodes
     * @return Gini coefficient (0 = perfect equality, 1 = perfect inequality)
     */
    private double calculateGiniCoefficient(Map<Integer, List<Node>> communities) {
        int n = communities.size();
        if (n <= 1) return 0.0;
        
        // Extract community sizes
        int[] sizes = new int[n];
        int i = 0;
        for (List<Node> community : communities.values()) {
            sizes[i++] = community.size();
        }
        
        // Sort sizes in ascending order
        Arrays.sort(sizes);
        
        // Calculate Gini coefficient
        double sumOfDifferences = 0.0;
        double sumOfSizes = 0.0;
        
        for (int size : sizes) {
            sumOfSizes += size;
        }
        
        if (sumOfSizes == 0.0) return 0.0;
        
        for (i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                sumOfDifferences += Math.abs(sizes[i] - sizes[j]);
            }
        }
        
        return sumOfDifferences / (2.0 * n * sumOfSizes);
    }
    
    /**
     * Gets the detected communities.
     * 
     * @return Map of community IDs to lists of nodes
     */
    public Map<Integer, List<Node>> getCommunities() {
        return communities;
    }
    
    /**
     * Helper class to store points with their original indices for k-means clustering.
     */
    private class EmbeddingPoint implements Clusterable {
        private final double[] point;
        private final int index;
        
        public EmbeddingPoint(double[] point, int index) {
            this.point = point;
            this.index = index;
        }
        
        @Override
        public double[] getPoint() {
            return point;
        }
        
        public int getIndex() {
            return index;
        }
    }

    /**
     * Enforce maximum cluster size by splitting oversized clusters
     *
     * @param communities Current communities
     * @param maxSize Maximum allowed cluster size
     * @return Updated communities with no community larger than maxSize
     */
    private Map<Integer, List<Node>> enforceMaximumClusterSize(Map<Integer, List<Node>> communities, int maxSize) {
        // Make a deep copy of the input
        Map<Integer, List<Node>> result = new HashMap<>();
        int nextId = communities.keySet().stream().mapToInt(Integer::intValue).max().orElse(-1) + 1;
        
        System.out.println("Strictly enforcing maximum community size of " + maxSize + " nodes");
        
        // Track which nodes have been assigned
        Set<Node> assignedNodes = new HashSet<>();
        
        // Process communities in order of size (largest first)
        List<Map.Entry<Integer, List<Node>>> communitiesBySizeDesc = new ArrayList<>(communities.entrySet());
        communitiesBySizeDesc.sort((e1, e2) -> Integer.compare(e2.getValue().size(), e1.getValue().size()));
        
        for (Map.Entry<Integer, List<Node>> entry : communitiesBySizeDesc) {
            List<Node> community = entry.getValue();
            
            // Skip already processed nodes
            community = community.stream()
                .filter(node -> !assignedNodes.contains(node))
                .collect(Collectors.toList());
            
            if (community.isEmpty()) {
                continue;
            }
            
            // If community is small enough, keep it as is
            if (community.size() <= maxSize) {
                result.put(entry.getKey(), community);
                assignedNodes.addAll(community);
                continue;
            }
            
            // For larger communities, we need to split them
            System.out.println("Splitting community " + entry.getKey() + " with " + community.size() + " nodes");
            
            // Use k-means with k = ceiling(size/maxSize) to split the community
            int k = (int) Math.ceil((double) community.size() / maxSize);
            
            // Convert nodes to points for clustering
            List<EmbeddingPoint> points = new ArrayList<>();
            for (int i = 0; i < community.size(); i++) {
                Node node = community.get(i);
                // Create a point based on node coordinates
                double[] coords = new double[] {
                    node.getLocation().getX(),
                    node.getLocation().getY()
                };
                points.add(new EmbeddingPoint(coords, i));
            }
            
            // Ensure we're using a proper distance metric for geographical coordinates
            KMeansPlusPlusClusterer<EmbeddingPoint> clusterer = 
                new KMeansPlusPlusClusterer<>(k, 100, new EuclideanDistance());
            
            // Perform clustering
            List<CentroidCluster<EmbeddingPoint>> clusters = clusterer.cluster(points);
            
            // Distribute nodes to subcommunities
            Map<Integer, List<Node>> subcommunities = new HashMap<>();
            for (int i = 0; i < clusters.size(); i++) {
                subcommunities.put(i, new ArrayList<>());
            }
            
            // Assign each node to its subcommunity
            for (int i = 0; i < clusters.size(); i++) {
                CentroidCluster<EmbeddingPoint> cluster = clusters.get(i);
                for (EmbeddingPoint point : cluster.getPoints()) {
                    subcommunities.get(i).add(community.get(point.getIndex()));
                }
            }
            
            // Further split any subcommunities that are still too large
            List<List<Node>> finalSubcommunities = new ArrayList<>();
            for (List<Node> subcommunity : subcommunities.values()) {
                if (subcommunity.size() > maxSize) {
                    // Split again if needed using geographic partitioning
                    int subK = (int) Math.ceil((double) subcommunity.size() / maxSize);
                    finalSubcommunities.addAll(geographicPartitioning(subcommunity, subK));
                } else {
                    finalSubcommunities.add(subcommunity);
                }
            }
            
            // Add all subcommunities to result
            boolean originalIdUsed = false;
            for (List<Node> subCommunity : finalSubcommunities) {
                if (!originalIdUsed) {
                    result.put(entry.getKey(), subCommunity);
                    originalIdUsed = true;
                } else {
                    result.put(nextId++, subCommunity);
                }
                assignedNodes.addAll(subCommunity);
            }
        }
        
        // Handle any remaining nodes that weren't assigned
        List<Node> unassignedNodes = new ArrayList<>();
        for (List<Node> community : communities.values()) {
            for (Node node : community) {
                if (!assignedNodes.contains(node)) {
                    unassignedNodes.add(node);
                }
            }
        }
        
        if (!unassignedNodes.isEmpty()) {
            System.out.println("Assigning " + unassignedNodes.size() + " remaining nodes");
            Map<Integer, List<Node>> additionalCommunities = 
                geographicClusteringWithMaxSize(unassignedNodes, maxSize, nextId);
            result.putAll(additionalCommunities);
        }
        
        // Verify maximum size constraint is met
        boolean allValid = true;
        for (Map.Entry<Integer, List<Node>> entry : result.entrySet()) {
            if (entry.getValue().size() > maxSize) {
                System.out.println("Warning: Community " + entry.getKey() + 
                               " still has " + entry.getValue().size() + 
                               " nodes (exceeds max " + maxSize + ")");
                allValid = false;
            }
        }
        
        if (!allValid) {
            System.out.println("Some communities still exceed maximum size. Performing final adjustment.");
            return performFinalSizeAdjustment(result, maxSize);
        }
        
        System.out.println("All communities are within size limits. Total communities: " + result.size());
        return result;
    }
    
    /**
     * Performs a final adjustment to strictly enforce maximum community size
     * by splitting any remaining oversized communities.
     *
     * @param communities The communities map
     * @param maxSize The maximum allowed size
     * @return Updated communities map with strict size enforcement
     */
    private Map<Integer, List<Node>> performFinalSizeAdjustment(Map<Integer, List<Node>> communities, int maxSize) {
        Map<Integer, List<Node>> result = new HashMap<>();
        int nextId = communities.keySet().stream().mapToInt(Integer::intValue).max().orElse(-1) + 1;
        
        for (Map.Entry<Integer, List<Node>> entry : communities.entrySet()) {
            if (entry.getValue().size() <= maxSize) {
                // Keep communities under the limit as they are
                result.put(entry.getKey(), entry.getValue());
            } else {
                // For oversized communities, split them into chunks of maxSize
                List<Node> community = entry.getValue();
                List<List<Node>> chunks = new ArrayList<>();
                
                // Sort nodes by location for better spatial coherence
                community.sort(Comparator.comparingDouble(n -> n.getLocation().getX()));
                
                for (int i = 0; i < community.size(); i += maxSize) {
                    int end = Math.min(i + maxSize, community.size());
                    chunks.add(new ArrayList<>(community.subList(i, end)));
                }
                
                // Add chunks as separate communities
                boolean originalIdUsed = false;
                for (List<Node> chunk : chunks) {
                    if (!originalIdUsed) {
                        result.put(entry.getKey(), chunk);
                        originalIdUsed = true;
                    } else {
                        result.put(nextId++, chunk);
                    }
                }
            }
        }
        
        return result;
    }
    
    /**
     * Geographic partitioning for splitting communities based on spatial location
     *
     * @param nodes The nodes to partition
     * @param k Number of partitions to create
     * @return List of node lists representing the partitions
     */
    private List<List<Node>> geographicPartitioning(List<Node> nodes, int k) {
        // Convert nodes to EmbeddingPoints for k-means
        List<EmbeddingPoint> points = new ArrayList<>();
        for (int i = 0; i < nodes.size(); i++) {
            Node node = nodes.get(i);
            double[] coords = new double[] {
                node.getLocation().getX(),
                node.getLocation().getY()
            };
            points.add(new EmbeddingPoint(coords, i));
        }
        
        // Use k-means to cluster based on geographic location
        KMeansPlusPlusClusterer<EmbeddingPoint> clusterer = 
            new KMeansPlusPlusClusterer<>(k, 50, new EuclideanDistance());
        
        List<CentroidCluster<EmbeddingPoint>> clusters = clusterer.cluster(points);
        
        // Convert clusters back to node lists
        List<List<Node>> result = new ArrayList<>();
        for (CentroidCluster<EmbeddingPoint> cluster : clusters) {
            List<Node> partition = new ArrayList<>();
            for (EmbeddingPoint point : cluster.getPoints()) {
                partition.add(nodes.get(point.getIndex()));
            }
            result.add(partition);
        }
        
        return result;
    }
    
    /**
     * Performs geographic clustering with strict maximum size enforcement
     *
     * @param nodes List of nodes to cluster
     * @param maxSize Maximum size per cluster
     * @param startId Starting ID for new communities
     * @return Map of community IDs to node lists
     */
    private Map<Integer, List<Node>> geographicClusteringWithMaxSize(List<Node> nodes, int maxSize, int startId) {
        Map<Integer, List<Node>> result = new HashMap<>();
        
        if (nodes.isEmpty()) {
            return result;
        }
        
        int k = (int) Math.ceil((double) nodes.size() / maxSize);
        List<List<Node>> partitions = geographicPartitioning(nodes, k);
        
        int id = startId;
        for (List<Node> partition : partitions) {
            result.put(id++, partition);
        }
        
        return result;
    }

    /**
     * Force a specific number of clusters by merging or splitting as needed
     *
     * @param communities Current communities
     * @param nodes All nodes in the graph
     * @param targetCount Target number of clusters
     * @return Updated communities with exactly targetCount clusters
     */
    private Map<Integer, List<Node>> forceNumberOfClusters(Map<Integer, List<Node>> communities, List<Node> nodes, int targetCount) {
        int currentCount = communities.size();
        
        if (currentCount == targetCount) {
            return communities;
        }
        
        Map<Integer, List<Node>> result = new HashMap<>(communities);
        
        if (currentCount < targetCount) {
            // Need to split communities to get more clusters
            // Start with the largest communities
            while (result.size() < targetCount) {
                // Find largest community
                Map.Entry<Integer, List<Node>> largest = null;
                for (Map.Entry<Integer, List<Node>> entry : result.entrySet()) {
                    if (largest == null || entry.getValue().size() > largest.getValue().size()) {
                        largest = entry;
                    }
                }
                
                if (largest == null || largest.getValue().size() <= minCommunitySize * 1.4) {
                    // Can't split any further without violating minimum size constraint
                    System.out.println("Warning: Cannot create " + targetCount + " clusters without violating minimum size constraint");
                    break;
                }
                
                // Split the largest community in two
                List<Node> community = largest.getValue();
                
                // Sort nodes by distance from centroid
                double[] centroid = calculateCentroid(community);
                community.sort((a, b) -> {
                    double distA = pointDistance(a.getLocation().getX(), a.getLocation().getY(), centroid[0], centroid[1]);
                    double distB = pointDistance(b.getLocation().getX(), b.getLocation().getY(), centroid[0], centroid[1]);
                    return Double.compare(distA, distB);
                });
                
                // Create two new communities
                int splitPoint = community.size() / 2;
                List<Node> community1 = new ArrayList<>(community.subList(0, splitPoint));
                List<Node> community2 = new ArrayList<>(community.subList(splitPoint, community.size()));
                
                // Replace the original community with the first split
                result.put(largest.getKey(), community1);
                
                // Add the second split as a new community
                int newId = result.keySet().stream().max(Integer::compare).orElse(0) + 1;
                result.put(newId, community2);
            }
        } else {
            // Need to merge communities to reduce cluster count
            // Merge smallest communities first
            while (result.size() > targetCount) {
                // Find two smallest communities
                int smallest1 = -1, smallest2 = -1;
                
                for (Integer id : result.keySet()) {
                    if (smallest1 == -1 || result.get(id).size() < result.get(smallest1).size()) {
                        smallest2 = smallest1;
                        smallest1 = id;
                    } else if (smallest2 == -1 || result.get(id).size() < result.get(smallest2).size()) {
                        smallest2 = id;
                    }
                }
                
                if (smallest1 != -1 && smallest2 != -1) {
                    // Merge these two communities
                    List<Node> merged = new ArrayList<>(result.get(smallest1));
                    merged.addAll(result.get(smallest2));
                    result.put(smallest1, merged);
                    result.remove(smallest2);
                } else {
                    break; // Cannot merge further
                }
            }
        }
        
        return result;
    }

    /**
     * Calculate Euclidean distance between two points
     */
    private double pointDistance(double x1, double y1, double x2, double y2) {
        return Math.sqrt(Math.pow(x1 - x2, 2) + Math.pow(y1 - y2, 2));
    }

    /**
     * Apply geographic post-processing to improve geographic cohesion
     *
     * @param communities Current communities
     * @return Updated communities with improved geographic cohesion
     */
    private Map<Integer, List<Node>> improveGeographicCohesion(Map<Integer, List<Node>> communities) {
        System.out.println("Improving geographic cohesion of communities...");
        Map<Integer, List<Node>> improvedCommunities = new HashMap<>();
        int nextCommunityId = communities.size() + 1;
        
        // For each community
        for (Map.Entry<Integer, List<Node>> entry : communities.entrySet()) {
            int communityId = entry.getKey();
            List<Node> nodes = entry.getValue();
            
            // Skip small communities (they're already cohesive)
            if (nodes.size() < 20) {
                improvedCommunities.put(communityId, new ArrayList<>(nodes));
                continue;
            }
            
            // Calculate community centroid
            double[] centroid = calculateCentroid(nodes);
            
            // Calculate distance from each node to centroid
            List<NodeDistance> distances = new ArrayList<>();
            for (Node node : nodes) {
                double distance = calculateDistance(
                    new double[] {node.getLocation().getX(), node.getLocation().getY()}, 
                    centroid
                );
                distances.add(new NodeDistance(node, distance));
            }
            
            // Sort by distance to centroid
            distances.sort(Comparator.comparingDouble(NodeDistance::getDistance));
            
            // Check if we have outliers (nodes much further from centroid than others)
            double medianDistance = distances.get(distances.size() / 2).getDistance();
            double outlierThreshold = medianDistance * 3.0; // 3x median is considered an outlier
            
            List<Node> outliers = new ArrayList<>();
            List<Node> coreNodes = new ArrayList<>();
            
            for (NodeDistance nd : distances) {
                if (nd.getDistance() > outlierThreshold) {
                    outliers.add(nd.getNode());
                } else {
                    coreNodes.add(nd.getNode());
                }
            }
            
            // If no outliers, keep the community as is
            if (outliers.isEmpty()) {
                improvedCommunities.put(communityId, nodes);
                continue;
            }
            
            System.out.println("Found " + outliers.size() + " geographic outliers in community " + communityId);
            
            // Put core nodes in the original community
            improvedCommunities.put(communityId, coreNodes);
            
            // Group outliers into new communities based on proximity
            Map<Integer, List<Node>> outlierGroups = clusterOutliers(outliers, nextCommunityId);
            improvedCommunities.putAll(outlierGroups);
            
            nextCommunityId += outlierGroups.size();
        }
        
        return improvedCommunities;
    }
    
    /**
     * Cluster outlier nodes into new communities based on geographic proximity
     * 
     * @param outliers List of outlier nodes
     * @param startId Starting community ID for new communities
     * @return Map of new community IDs to lists of nodes
     */
    private Map<Integer, List<Node>> clusterOutliers(List<Node> outliers, int startId) {
        Map<Integer, List<Node>> result = new HashMap<>();
        
        // If only a few outliers, put them all in one new community
        if (outliers.size() < 5) {
            result.put(startId, outliers);
            return result;
        }
        
        // Perform simple agglomerative clustering based on geographic distance
        // This is a greedy algorithm that works well for small numbers of nodes
        
        // Start with each node in its own cluster
        Map<Integer, List<Node>> clusters = new HashMap<>();
        for (int i = 0; i < outliers.size(); i++) {
            List<Node> cluster = new ArrayList<>();
            cluster.add(outliers.get(i));
            clusters.put(i, cluster);
        }
        
        // Repeatedly merge the closest clusters until we have a reasonable number
        // or until no clusters are close enough to merge
        double mergeThreshold = 2000.0; // 2km - maximum distance for merging
        
        while (clusters.size() > 1) {
            // Find the closest pair of clusters
            int bestI = -1;
            int bestJ = -1;
            double bestDistance = Double.MAX_VALUE;
            
            for (int i : clusters.keySet()) {
                for (int j : clusters.keySet()) {
                    if (i >= j) continue;
                    
                    // Calculate minimum distance between clusters
                    double minDistance = Double.MAX_VALUE;
                    for (Node ni : clusters.get(i)) {
                        for (Node nj : clusters.get(j)) {
                            double distance = calculateGeoDistance(ni, nj);
                            minDistance = Math.min(minDistance, distance);
                        }
                    }
                    
                    if (minDistance < bestDistance) {
                        bestDistance = minDistance;
                        bestI = i;
                        bestJ = j;
                    }
                }
            }
            
            // If the closest clusters are too far apart, stop merging
            if (bestDistance > mergeThreshold) {
                break;
            }
            
            // Merge the closest clusters
            List<Node> merged = new ArrayList<>(clusters.get(bestI));
            merged.addAll(clusters.get(bestJ));
            clusters.remove(bestJ);
            clusters.put(bestI, merged);
        }
        
        // Convert to final community IDs
        int id = startId;
        for (List<Node> cluster : clusters.values()) {
            result.put(id++, cluster);
        }
        
        return result;
    }
    
    /**
     * Helper class to store a node and its distance to a reference point
     */
    private static class NodeDistance {
        private final Node node;
        private final double distance;
        
        public NodeDistance(Node node, double distance) {
            this.node = node;
            this.distance = distance;
        }
        
        public Node getNode() {
            return node;
        }
        
        public double getDistance() {
            return distance;
        }
    }

    /**
     * Enforce maximum diameter constraint
     *
     * @param communities Current communities
     * @param maxDiameter Maximum allowed community diameter in meters
     * @return Updated communities with no community larger than maxDiameter
     */
    private Map<Integer, List<Node>> enforceMaximumDiameter(Map<Integer, List<Node>> communities, double maxDiameter) {
        System.out.println("Enforcing maximum community diameter of " + maxDiameter + " meters");
        Map<Integer, List<Node>> result = new HashMap<>();
        int nextCommunityId = communities.size() + 1;
        
        for (Map.Entry<Integer, List<Node>> entry : communities.entrySet()) {
            int communityId = entry.getKey();
            List<Node> nodes = entry.getValue();
            
            // Skip small communities (they're likely already within diameter constraints)
            if (nodes.size() < 10) {
                result.put(communityId, new ArrayList<>(nodes));
                continue;
            }
            
            // Calculate community diameter
            double diameter = calculateCommunityDiameter(nodes);
            
            // If within limit, keep as is
            if (diameter <= maxDiameter) {
                result.put(communityId, new ArrayList<>(nodes));
                continue;
            }
            
            System.out.println("Community " + communityId + " has diameter " + String.format("%.2f", diameter) +
                             " meters, exceeding limit of " + maxDiameter + " meters. Splitting.");
            
            // Split the community using spatial partitioning
            Map<Integer, List<Node>> splitCommunities = spatialPartitioning(nodes, maxDiameter, nextCommunityId);
            result.putAll(splitCommunities);
            
            // Update next ID
            nextCommunityId += splitCommunities.size();
        }
        
        return result;
    }
    
    /**
     * Calculate the geographic diameter of a community (maximum distance between any two nodes)
     * 
     * @param nodes List of nodes in the community
     * @return Diameter in meters
     */
    private double calculateCommunityDiameter(List<Node> nodes) {
        double maxDistance = 0.0;
        
        for (int i = 0; i < nodes.size(); i++) {
            for (int j = i + 1; j < nodes.size(); j++) {
                double distance = calculateGeoDistance(nodes.get(i), nodes.get(j));
                maxDistance = Math.max(maxDistance, distance);
            }
        }
        
        return maxDistance;
    }
    
    /**
     * Split a community into spatially coherent parts that respect the diameter constraint
     * 
     * @param nodes Nodes to split
     * @param maxDiameter Maximum allowed diameter
     * @param startId Starting ID for new communities
     * @return Map of community IDs to node lists
     */
    private Map<Integer, List<Node>> spatialPartitioning(List<Node> nodes, double maxDiameter, int startId) {
        Map<Integer, List<Node>> result = new HashMap<>();
        
        // Find the two most distant nodes
        Node node1 = null;
        Node node2 = null;
        double maxDistance = 0.0;
        
        for (int i = 0; i < nodes.size(); i++) {
            for (int j = i + 1; j < nodes.size(); j++) {
                double distance = calculateGeoDistance(nodes.get(i), nodes.get(j));
                if (distance > maxDistance) {
                    maxDistance = distance;
                    node1 = nodes.get(i);
                    node2 = nodes.get(j);
                }
            }
        }
        
        // Create two clusters with these nodes as seeds
        List<Node> cluster1 = new ArrayList<>();
        List<Node> cluster2 = new ArrayList<>();
        
        cluster1.add(node1);
        cluster2.add(node2);
        
        // Assign each remaining node to the closest centroid
        for (Node node : nodes) {
            if (node != node1 && node != node2) {
                double dist1 = calculateGeoDistance(node, node1);
                double dist2 = calculateGeoDistance(node, node2);
                
                if (dist1 < dist2) {
                    cluster1.add(node);
                } else {
                    cluster2.add(node);
                }
            }
        }
        
        // Check if both clusters respect the diameter constraint
        double diameter1 = calculateCommunityDiameter(cluster1);
        double diameter2 = calculateCommunityDiameter(cluster2);
        
        // Recursively split clusters that are still too large
        if (diameter1 > maxDiameter && cluster1.size() > 10) {
            Map<Integer, List<Node>> split1 = spatialPartitioning(cluster1, maxDiameter, startId);
            result.putAll(split1);
            startId += split1.size();
        } else {
            result.put(startId++, cluster1);
        }
        
        if (diameter2 > maxDiameter && cluster2.size() > 10) {
            Map<Integer, List<Node>> split2 = spatialPartitioning(cluster2, maxDiameter, startId);
            result.putAll(split2);
        } else {
            result.put(startId, cluster2);
        }
        
        return result;
    }

    /**
     * Balances community sizes by merging small communities with their geographic neighbors
     * and splitting large communities if necessary
     * 
     * @param communities Map of community IDs to lists of nodes
     * @param minSize Minimum community size
     * @param maxSize Maximum community size (0 = no maximum)
     * @return Balanced communities
     */
    private Map<Integer, List<Node>> balanceCommunitySize(Map<Integer, List<Node>> communities, int minSize, int maxSize) {
        Map<Integer, List<Node>> balancedCommunities = new HashMap<>();
        List<Integer> communitiesToMerge = new ArrayList<>();
        
        // First pass: identify communities that need to be merged (small ones)
        for (Map.Entry<Integer, List<Node>> entry : communities.entrySet()) {
            int communityId = entry.getKey();
            List<Node> nodes = entry.getValue();
            
            // Communities smaller than minSize will be merged
            if (nodes.size() < minSize) {
                communitiesToMerge.add(communityId);
                System.out.println("Community " + communityId + " with size " + nodes.size() + 
                                 " is below minimum size (" + minSize + ") and will be merged");
            } else {
                balancedCommunities.put(communityId, new ArrayList<>(nodes));
            }
        }
        
        // Special case: if ALL communities are too small, try to merge them together first
        if (balancedCommunities.isEmpty() && !communitiesToMerge.isEmpty()) {
            System.out.println("All communities are below minimum size. Attempting to combine small communities.");
            
            // Collect all small communities by geographical proximity
            List<List<Node>> smallCommunityNodes = new ArrayList<>();
            for (Integer id : communitiesToMerge) {
                smallCommunityNodes.add(communities.get(id));
            }
            
            // Sort by size (largest first) to prioritize largest small communities
            smallCommunityNodes.sort((a, b) -> b.size() - a.size());
            
            // Process each small community
            List<Node> currentCombined = new ArrayList<>(smallCommunityNodes.get(0));
            int lastAssignedId = 0;
            
            for (int i = 1; i < smallCommunityNodes.size(); i++) {
                List<Node> nextCommunity = smallCommunityNodes.get(i);
                
                // If combining won't exceed max size (if specified)
                if (maxSize <= 0 || currentCombined.size() + nextCommunity.size() <= maxSize) {
                    // Combine communities
                    currentCombined.addAll(nextCommunity);
                } else {
                    // Current combined group is as large as it can be
                    if (currentCombined.size() >= minSize) {
                        // If it meets minimum size, save it
                        balancedCommunities.put(lastAssignedId++, currentCombined);
                        System.out.println("Created combined community with " + currentCombined.size() + " nodes");
                        // Start a new combined group
                        currentCombined = new ArrayList<>(nextCommunity);
                    } else {
                        // If still too small, force combination and accept exceeding max size
                        System.out.println("Warning: Combining communities exceeds max size but is necessary to meet minimum size");
                        currentCombined.addAll(nextCommunity);
                    }
                }
                
                // If current combined is now large enough, save it and start a new one
                if (currentCombined.size() >= minSize && i < smallCommunityNodes.size() - 1) {
                    balancedCommunities.put(lastAssignedId++, currentCombined);
                    System.out.println("Created combined community with " + currentCombined.size() + " nodes");
                    currentCombined = new ArrayList<>();
                }
            }
            
            // Handle any remaining combined nodes
            if (!currentCombined.isEmpty()) {
                if (currentCombined.size() >= minSize) {
                    balancedCommunities.put(lastAssignedId, currentCombined);
                    System.out.println("Created final combined community with " + currentCombined.size() + " nodes");
                } else if (!balancedCommunities.isEmpty()) {
                    // If too small, find closest community to merge with
                    double[] centroid = calculateCentroid(currentCombined);
                    int closestId = -1;
                    double minDist = Double.MAX_VALUE;
                    
                    for (Map.Entry<Integer, List<Node>> entry : balancedCommunities.entrySet()) {
                        double[] otherCentroid = calculateCentroid(entry.getValue());
                        double dist = calculateDistance(centroid, otherCentroid);
                        if (dist < minDist) {
                            minDist = dist;
                            closestId = entry.getKey();
                        }
                    }
                    
                    if (closestId >= 0) {
                        balancedCommunities.get(closestId).addAll(currentCombined);
                        System.out.println("Merged remaining small group with " + currentCombined.size() + 
                                         " nodes into community " + closestId);
                    } else {
                        // Last resort, keep as a separate community
                        balancedCommunities.put(lastAssignedId, currentCombined);
                        System.out.println("Keeping small community with " + currentCombined.size() + 
                                         " nodes as separate community (could not merge)");
                    }
                } else {
                    // No other communities exist, keep this one
                    balancedCommunities.put(lastAssignedId, currentCombined);
                    System.out.println("Keeping small community with " + currentCombined.size() + 
                                    " nodes as the only community (below minimum size)");
                }
            }
            
            // If we've handled all communities by combining, return the result
            if (communitiesToMerge.size() == communities.size()) {
                System.out.println("Successfully balanced all communities by combining small ones");
                return balancedCommunities;
            }
            
            // Otherwise, continue with the regular merging process for any remaining communities
            List<Integer> remainingToMerge = new ArrayList<>();
            for (Integer id : communitiesToMerge) {
                if (!balancedCommunities.containsKey(id) && communities.containsKey(id)) {
                    remainingToMerge.add(id);
                }
            }
            communitiesToMerge = remainingToMerge;
        }
        
        // Second pass: merge small communities with geographically closest communities
        for (Integer smallCommunityId : communitiesToMerge) {
            List<Node> smallCommunity = communities.get(smallCommunityId);
            
            // Find closest community based on distance between centroids
            double[] smallCentroid = calculateCentroid(smallCommunity);
            Integer closestCommunityId = null;
            double closestDistance = Double.MAX_VALUE;
            
            for (Map.Entry<Integer, List<Node>> entry : balancedCommunities.entrySet()) {
                int communityId = entry.getKey();
                List<Node> communityNodes = entry.getValue();
                
                // Skip if merging would exceed the maximum size
                if (communityNodes.size() + smallCommunity.size() > maxSize) {
                    continue;
                }
                
                double[] centroid = calculateCentroid(communityNodes);
                double distance = calculateDistance(smallCentroid, centroid);
                
                if (distance < closestDistance) {
                    closestDistance = distance;
                    closestCommunityId = communityId;
                }
            }
            
            // If we found a suitable community to merge with
            if (closestCommunityId != null) {
                List<Node> targetCommunity = balancedCommunities.get(closestCommunityId);
                targetCommunity.addAll(smallCommunity);
                System.out.println("Merged community " + smallCommunityId + " into community " + 
                                 closestCommunityId + " (new size: " + targetCommunity.size() + ")");
            } else {
                // If no suitable community found, keep it as is
                System.out.println("Could not find suitable merge target for community " + 
                                 smallCommunityId + ", keeping it separate");
                balancedCommunities.put(smallCommunityId, smallCommunity);
            }
        }
        
        return balancedCommunities;
    }
    
    /**
     * Calculate average distance from a node to a list of nodes
     * 
     * @param node The node
     * @param nodes List of target nodes
     * @return Average distance in geographic units
     */
    private double calculateAverageDistanceToNodes(Node node, List<Node> nodes) {
        if (nodes == null || nodes.isEmpty()) {
            return Double.MAX_VALUE; // Return maximum distance for empty lists
        }
        
        double totalDistance = 0.0;
        int validDistances = 0;
        
        for (Node targetNode : nodes) {
            if (targetNode != null) {
                double distance = calculateGeoDistance(node, targetNode);
                if (!Double.isNaN(distance) && !Double.isInfinite(distance)) {
                    totalDistance += distance;
                    validDistances++;
                }
            }
        }
        
        // If no valid distances were calculated, return MAX_VALUE
        if (validDistances == 0) {
            return Double.MAX_VALUE;
        }
        
        return totalDistance / validDistances;
    }

    /**
     * Geographic fallback clustering method using k-means on coordinates
     */
    private Map<Integer, List<Node>> geographicFallbackClustering(Set<Node> nodeSet, int k) {
        System.out.println("Using geographic fallback clustering with k=" + k);
        List<Node> nodes = new ArrayList<>(nodeSet);
        return directGeographicClustering(nodes, k);
    }
    
    /**
     * Direct k-means clustering on geographic coordinates
     */
    private Map<Integer, List<Node>> directGeographicClustering(List<Node> nodes, int k) {
        int n = nodes.size();
        
        // Create data points for k-means
        List<EmbeddingPoint> points = new ArrayList<>(n);
        for (int i = 0; i < n; i++) {
            Node node = nodes.get(i);
            Point location = node.getLocation();
            double[] coords = new double[] {location.getX(), location.getY()};
            points.add(new EmbeddingPoint(coords, i));
        }
        
        // Perform k-means clustering
        KMeansPlusPlusClusterer<EmbeddingPoint> clusterer = new KMeansPlusPlusClusterer<>(
            k, 500, new EuclideanDistance());
        
        List<CentroidCluster<EmbeddingPoint>> clusters = clusterer.cluster(points);
        
        // Convert clusters to community map
        Map<Integer, List<Node>> communities = new HashMap<>();
        int communityId = 0;
        
        for (CentroidCluster<EmbeddingPoint> cluster : clusters) {
            List<Node> community = new ArrayList<>();
            
            for (EmbeddingPoint point : cluster.getPoints()) {
                community.add(nodes.get(point.getIndex()));
            }
            
            if (!community.isEmpty()) {
                communities.put(communityId++, community);
            }
        }
        
        return communities;
    }

    /**
     * Identify and split disconnected subgroups within communities
     * 
     * @param communities Current communities
     * @param graph The original graph
     * @param proximityThreshold Maximum distance between nodes to be considered connected (meters)
     * @return Updated communities with disconnected subgroups split
     */
    private Map<Integer, List<Node>> splitDisconnectedSubgroups(Map<Integer, List<Node>> communities, 
                                                              Graph<Node, DefaultWeightedEdge> graph,
                                                              double proximityThreshold) {
        Map<Integer, List<Node>> result = new HashMap<>();
        int nextCommunityId = communities.size();
        
        for (Map.Entry<Integer, List<Node>> entry : communities.entrySet()) {
            int communityId = entry.getKey();
            List<Node> nodes = entry.getValue();
            
            // Skip small communities
            if (nodes.size() < 15) {
                result.put(communityId, new ArrayList<>(nodes));
                continue;
            }
            
            // Create a proximity graph for this community
            // Two nodes are connected if they are within proximityThreshold meters of each other
            Map<Node, Set<Node>> proximityGraph = buildProximityGraph(nodes, proximityThreshold);
            
            // Find connected components in the proximity graph
            List<List<Node>> connectedComponents = findConnectedComponents(proximityGraph);
            
            // If only one connected component, keep the community as is
            if (connectedComponents.size() == 1) {
                result.put(communityId, nodes);
                continue;
            }
            
            // Log information about the split
            System.out.println("Found " + connectedComponents.size() + " disconnected subgroups in community " + 
                            communityId + " (size " + nodes.size() + ")");
            
            // Sort connected components by size (largest first)
            connectedComponents.sort((a, b) -> b.size() - a.size());
            
            // Keep the largest component with the original community ID
            result.put(communityId, connectedComponents.get(0));
            System.out.println("  - Keeping largest subgroup with " + connectedComponents.get(0).size() + 
                            " nodes as community " + communityId);
            
            // Create new communities for the other components
            for (int i = 1; i < connectedComponents.size(); i++) {
                List<Node> component = connectedComponents.get(i);
                
                // Skip very small components (they'll be handled by the balancing step later)
                if (component.size() < 5) {
                    continue;
                }
                
                result.put(nextCommunityId, component);
                System.out.println("  - Creating new community " + nextCommunityId + 
                                " with " + component.size() + " nodes");
                nextCommunityId++;
            }
        }
        
        return result;
    }
    
    /**
     * Build a proximity graph where nodes are connected if they're within the specified distance
     * 
     * @param nodes List of nodes
     * @param proximityThreshold Maximum distance for connection (meters)
     * @return Map representing the proximity graph
     */
    private Map<Node, Set<Node>> buildProximityGraph(List<Node> nodes, double proximityThreshold) {
        Map<Node, Set<Node>> graph = new HashMap<>();
        
        // Initialize graph
        for (Node node : nodes) {
            graph.put(node, new HashSet<>());
        }
        
        // Connect nodes that are within the proximity threshold
        for (int i = 0; i < nodes.size(); i++) {
            Node node1 = nodes.get(i);
            
            for (int j = i + 1; j < nodes.size(); j++) {
                Node node2 = nodes.get(j);
                
                double distance = calculateGeoDistance(node1, node2);
                
                if (distance <= proximityThreshold) {
                    // Add bidirectional connection
                    graph.get(node1).add(node2);
                    graph.get(node2).add(node1);
                }
            }
        }
        
        return graph;
    }
    
    /**
     * Find all connected components in a graph using BFS
     * 
     * @param graph The graph represented as a map of nodes to their adjacent nodes
     * @return List of connected components
     */
    private List<List<Node>> findConnectedComponents(Map<Node, Set<Node>> graph) {
        List<List<Node>> components = new ArrayList<>();
        Set<Node> visited = new HashSet<>();
        
        for (Node node : graph.keySet()) {
            if (!visited.contains(node)) {
                // Found a new unvisited node, start a new component
                List<Node> component = new ArrayList<>();
                Queue<Node> queue = new LinkedList<>();
                
                queue.add(node);
                visited.add(node);
                
                // BFS to find all nodes in this component
                while (!queue.isEmpty()) {
                    Node current = queue.poll();
                    component.add(current);
                    
                    for (Node neighbor : graph.get(current)) {
                        if (!visited.contains(neighbor)) {
                            visited.add(neighbor);
                            queue.add(neighbor);
                        }
                    }
                }
                
                components.add(component);
            }
        }
        
        return components;
    }

    /**
     * Ensures all nodes from the original graph are assigned to communities,
     * including isolated nodes that may not have been included in the clustering.
     * This is critical for making sure no nodes are lost in the final clustering result.
     *
     * @param communities The current community assignments
     * @param allNodes All nodes in the original graph
     * @return Updated communities with all nodes assigned
     */
    private Map<Integer, List<Node>> includeIsolatedNodes(Map<Integer, List<Node>> communities, Set<Node> allNodes) {
        // Create a copy of the communities
        Map<Integer, List<Node>> updatedCommunities = new HashMap<>();
        for (Map.Entry<Integer, List<Node>> entry : communities.entrySet()) {
            updatedCommunities.put(entry.getKey(), new ArrayList<>(entry.getValue()));
        }
        
        // Find nodes that aren't assigned to any community
        Set<Node> assignedNodes = new HashSet<>();
        for (List<Node> community : updatedCommunities.values()) {
            assignedNodes.addAll(community);
        }
        
        Set<Node> unassignedNodes = new HashSet<>(allNodes);
        unassignedNodes.removeAll(assignedNodes);
        
        if (unassignedNodes.isEmpty()) {
            System.out.println("All nodes are assigned to communities. No action needed.");
            return updatedCommunities;
        }
        
        System.out.println("Found " + unassignedNodes.size() + " unassigned nodes.");
        
        // Extract isolated unassigned nodes, those that form their own connected components
        Map<Node, Set<Node>> proximityGraph = buildProximityGraph(new ArrayList<>(unassignedNodes), 500); // 500m proximity threshold
        List<List<Node>> isolatedComponents = findConnectedComponents(proximityGraph);
        
        System.out.println("Unassigned nodes form " + isolatedComponents.size() + " isolated components");
        
        // For components with multiple nodes, create a new community for each
        int nextCommunityId = updatedCommunities.keySet().stream().max(Integer::compare).orElse(-1) + 1;
        List<Node> processedNodes = new ArrayList<>();
        
        for (List<Node> component : isolatedComponents) {
            if (component.size() > 1) {
                // Create a new community for this connected component
                updatedCommunities.put(nextCommunityId, new ArrayList<>(component));
                System.out.println("Created new community " + nextCommunityId + " for connected component with " + 
                                 component.size() + " nodes");
                nextCommunityId++;
                processedNodes.addAll(component);
            }
        }
        
        // Remove processed nodes from unassigned
        unassignedNodes.removeAll(processedNodes);
        
        // For remaining isolated nodes (singletons), find the closest community
        if (!unassignedNodes.isEmpty()) {
            System.out.println("Assigning " + unassignedNodes.size() + " remaining singleton nodes to nearest communities");
            
            // For each unassigned node
            for (Node node : unassignedNodes) {
                int bestCommunityId = -1;
                double bestDistance = Double.MAX_VALUE;
                
                // Find the closest community by geographic distance
                for (Map.Entry<Integer, List<Node>> entry : updatedCommunities.entrySet()) {
                    List<Node> communityNodes = entry.getValue();
                    if (communityNodes.isEmpty()) continue;
                    
                    double avgDistance = calculateAverageDistanceToNodes(node, communityNodes);
                    if (avgDistance < bestDistance) {
                        bestDistance = bestDistance;
                        bestCommunityId = entry.getKey();
                    }
                }
                
                // If found a community, add the node
                if (bestCommunityId >= 0) {
                    updatedCommunities.get(bestCommunityId).add(node);
                } else {
                    // If no community found (shouldn't happen), create a new one
                    List<Node> newCommunity = new ArrayList<>();
                    newCommunity.add(node);
                    updatedCommunities.put(nextCommunityId++, newCommunity);
                }
            }
        }
        
        // Final verification - ensure all nodes are assigned
        Set<Node> finalAssignedNodes = new HashSet<>();
        for (List<Node> community : updatedCommunities.values()) {
            finalAssignedNodes.addAll(community);
        }
        
        if (!finalAssignedNodes.equals(allNodes)) {
            Set<Node> stillMissing = new HashSet<>(allNodes);
            stillMissing.removeAll(finalAssignedNodes);
            System.out.println("WARNING: " + stillMissing.size() + " nodes are still unassigned after inclusion process!");
            
            // Last resort - create individual communities
            for (Node node : stillMissing) {
                List<Node> singletonCommunity = new ArrayList<>();
                singletonCommunity.add(node);
                updatedCommunities.put(nextCommunityId++, singletonCommunity);
            }
        }
        
        return updatedCommunities;
    }
    
    /**
     * Finds all connected components in the graph.
     * A connected component is a subgraph where every pair of nodes has a path between them.
     * 
     * @param graph The graph to analyze
     * @return A list of sets, where each set represents a connected component
     */
    private List<Set<Node>> findAllConnectedComponents(Graph<Node, DefaultWeightedEdge> graph) {
        org.jgrapht.alg.connectivity.ConnectivityInspector<Node, DefaultWeightedEdge> inspector = 
            new org.jgrapht.alg.connectivity.ConnectivityInspector<>(graph);
        
        return inspector.connectedSets();
    }
} 