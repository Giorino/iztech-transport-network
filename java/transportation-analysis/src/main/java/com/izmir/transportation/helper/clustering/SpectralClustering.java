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
            
            // Get nodes in a consistent order
            List<Node> nodes = new ArrayList<>(graph.vertexSet());
            
            // Create similarity matrix
            RealMatrix similarityMatrix = buildSimilarityMatrix(graph, nodes);
            
            // Get config values
            int maxClusters = config != null ? config.getNumberOfClusters() : numberOfClusters;
            int maxSize = config != null ? config.getMaxClusterSize() : 0;
            boolean forceNumClusters = config != null && config.isForceNumClusters();
            double maxDiameter = config != null ? config.getMaxCommunityDiameter() : 0.0;
            int minSize = config != null ? config.getMinCommunitySize() : minCommunitySize;
            
            try {
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
                System.out.println("Clustering in eigenspace...");
                communities = clusterEmbedding(embedding, nodes);
            } catch (Exception e) {
                System.err.println("Error in spectral decomposition: " + e.getMessage());
                System.out.println("Using direct k-means clustering on geographic coordinates...");
                
                // Fall back to direct k-means clustering on geographic coordinates
                communities = directGeographicClustering(nodes, maxClusters);
            }
            
            // Apply maximum diameter constraint first (if specified)
            if (maxDiameter > 0) {
                System.out.println("Enforcing maximum community diameter of " + maxDiameter + " meters");
                communities = enforceMaximumDiameter(communities, maxDiameter);
            }
            
            // If we need to enforce a specific number of clusters
            if (forceNumClusters && communities.size() != maxClusters) {
                System.out.println("Forcing " + maxClusters + " clusters (currently have " + communities.size() + ")");
                communities = forceNumberOfClusters(communities, nodes, maxClusters);
            }
            
            // Always apply maximum cluster size constraint
            if (maxSize > 0) {
                System.out.println("Enforcing maximum cluster size of " + maxSize);
                communities = enforceMaximumClusterSize(communities, maxSize);
                
                // Check if we still have any oversize communities after the first pass
                boolean hasOversizeCommunities = false;
                for (List<Node> community : communities.values()) {
                    if (community.size() > maxSize) {
                        hasOversizeCommunities = true;
                        System.out.println("Warning: Found community with size " + community.size() + " which exceeds maximum of " + maxSize);
                    }
                }
                
                // Run again if needed to handle any remaining oversized communities
                if (hasOversizeCommunities) {
                    System.out.println("Running additional pass to split remaining oversized communities");
                    communities = enforceMaximumClusterSize(communities, maxSize);
                }
            }
            
            // Apply geographic post-processing
            communities = improveGeographicCohesion(communities);
            
            // Add a new step to identify and split disconnected subgroups
            System.out.println("Identifying and splitting disconnected subgroups within communities...");
            communities = splitDisconnectedSubgroups(communities, graph, maxDiameter/2);
            
            // Final step: Balance community sizes by merging very small communities with nearby ones
            if (minSize > 10) {
                System.out.println("Performing final balancing by merging very small communities");
                communities = balanceCommunitySize(communities, minSize, maxSize);
            }
            
            // Calculate clustering quality metrics
            calculateClusteringMetrics(communities, graph);
            
            System.out.println("Spectral clustering complete. Found " + communities.size() + " communities.");
            return communities;
            
        } catch (Exception e) {
            System.err.println("Error during spectral clustering: " + e.getMessage());
            e.printStackTrace();
            return geographicFallbackClustering(transportationGraph.getGraph().vertexSet(), 40);
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
        if (n1 == null || n2 == null || n1.getLocation() == null || n2.getLocation() == null) {
            return Double.MAX_VALUE;
        }
        
        Point p1 = n1.getLocation();
        Point p2 = n2.getLocation();
        
        // Using Euclidean distance for simplicity
        // In a real-world scenario, using Haversine formula would be more accurate
        Coordinate c1 = p1.getCoordinate();
        Coordinate c2 = p2.getCoordinate();
        
        return Math.sqrt(Math.pow(c1.x - c2.x, 2) + Math.pow(c1.y - c2.y, 2));
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
        Map<Integer, List<Node>> result = new HashMap<>();
        int nextId = communities.size();
        
        for (Map.Entry<Integer, List<Node>> entry : communities.entrySet()) {
            List<Node> community = entry.getValue();
            
            if (community.size() <= maxSize) {
                // Community is already small enough
                result.put(entry.getKey(), community);
            } else {
                // Community needs to be split
                int numSplits = (int) Math.ceil((double) community.size() / maxSize);
                
                // Use k-means to split the community
                double[][] points = new double[community.size()][2];
                for (int i = 0; i < community.size(); i++) {
                    Node node = community.get(i);
                    points[i][0] = node.getLocation().getX();
                    points[i][1] = node.getLocation().getY();
                }
                
                // Create embedding points for clustering
                List<EmbeddingPoint> embeddingPoints = new ArrayList<>();
                for (int i = 0; i < community.size(); i++) {
                    embeddingPoints.add(new EmbeddingPoint(points[i], i));
                }
                
                // Use Apache Commons k-means clustering
                KMeansPlusPlusClusterer<EmbeddingPoint> clusterer = 
                    new KMeansPlusPlusClusterer<>(numSplits, maxIterations, new EuclideanDistance());
                
                List<CentroidCluster<EmbeddingPoint>> clusters = clusterer.cluster(embeddingPoints);
                
                // Create subcommunities
                Map<Integer, List<Node>> subcommunities = new HashMap<>();
                
                for (int i = 0; i < clusters.size(); i++) {
                    subcommunities.put(i, new ArrayList<>());
                }
                
                // Assign nodes to subcommunities
                for (int i = 0; i < clusters.size(); i++) {
                    CentroidCluster<EmbeddingPoint> cluster = clusters.get(i);
                    for (EmbeddingPoint point : cluster.getPoints()) {
                        subcommunities.get(i).add(community.get(point.getIndex()));
                    }
                }
                
                // Add first subcommunity with original ID
                boolean originalIdUsed = false;
                
                for (List<Node> subcommunity : subcommunities.values()) {
                    if (!originalIdUsed) {
                        result.put(entry.getKey(), subcommunity);
                        originalIdUsed = true;
                    } else {
                        result.put(nextId++, subcommunity);
                    }
                }
            }
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
     * Performs a final balancing step to merge very small communities with nearby ones
     * 
     * @param communities Current communities
     * @param minSize Minimum desired community size
     * @param maxSize Maximum allowed community size
     * @return Balanced communities
     */
    private Map<Integer, List<Node>> balanceCommunitySize(Map<Integer, List<Node>> communities, int minSize, int maxSize) {
        Map<Integer, List<Node>> balancedCommunities = new HashMap<>();
        List<Integer> communitiesToMerge = new ArrayList<>();
        
        // First pass: identify communities that are too small
        for (Map.Entry<Integer, List<Node>> entry : communities.entrySet()) {
            int communityId = entry.getKey();
            List<Node> nodes = entry.getValue();
            
            if (nodes.size() < minSize * 0.6) { // Communities below 60% of min size
                communitiesToMerge.add(communityId);
                System.out.println("Community " + communityId + " with size " + nodes.size() + 
                                 " is too small and will be merged");
            } else {
                balancedCommunities.put(communityId, new ArrayList<>(nodes));
            }
        }
        
        // Second pass: merge small communities with geographically closest communities
        for (Integer smallCommunityId : communitiesToMerge) {
            List<Node> smallCommunity = communities.get(smallCommunityId);
            
            // Find closest community based on distance between centroids
            double[] smallCentroid = calculateCentroid(smallCommunity);
            Integer closestCommunityId = null;
            double closestDistance = Double.MAX_VALUE;
            
            // Consider only communities that won't exceed maxSize after merging
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
        
        // Third pass: check if any communities are still too small and try node-by-node reassignment
        Map<Integer, List<Node>> finalCommunities = new HashMap<>(balancedCommunities);
        
        for (Map.Entry<Integer, List<Node>> entry : balancedCommunities.entrySet()) {
            int communityId = entry.getKey();
            List<Node> nodes = entry.getValue();
            
            if (nodes.size() < minSize * 0.8 && nodes.size() > 1) { // Still small but not singleton
                List<Node> nodesToReassign = new ArrayList<>(nodes);
                finalCommunities.remove(communityId);
                
                // Reassign nodes one by one to the closest community
                for (Node node : nodesToReassign) {
                    Integer bestCommunityId = null;
                    double bestDistance = Double.MAX_VALUE;
                    
                    for (Map.Entry<Integer, List<Node>> targetEntry : finalCommunities.entrySet()) {
                        int targetId = targetEntry.getKey();
                        List<Node> targetNodes = targetEntry.getValue();
                        
                        // Skip if target is already at max size
                        if (targetNodes.size() >= maxSize) {
                            continue;
                        }
                        
                        // Calculate average distance to this community
                        double avgDistance = calculateAverageDistanceToNodes(node, targetNodes);
                        
                        if (avgDistance < bestDistance) {
                            bestDistance = avgDistance;
                            bestCommunityId = targetId;
                        }
                    }
                    
                    // Assign to best community or create a new one if none found
                    if (bestCommunityId != null) {
                        finalCommunities.get(bestCommunityId).add(node);
                    } else {
                        // As last resort, put in a new community
                        List<Node> newCommunity = new ArrayList<>();
                        newCommunity.add(node);
                        finalCommunities.put(communityId, newCommunity);
                    }
                }
                
                System.out.println("Reassigned nodes from small community " + communityId);
            }
        }
        
        return finalCommunities;
    }
    
    /**
     * Calculate average distance from a node to a list of nodes
     * 
     * @param node The node
     * @param nodes List of target nodes
     * @return Average distance in geographic units
     */
    private double calculateAverageDistanceToNodes(Node node, List<Node> nodes) {
        double totalDistance = 0.0;
        
        for (Node targetNode : nodes) {
            totalDistance += calculateGeoDistance(node, targetNode);
        }
        
        return totalDistance / nodes.size();
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
} 