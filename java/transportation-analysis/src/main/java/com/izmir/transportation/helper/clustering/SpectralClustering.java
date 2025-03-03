package com.izmir.transportation.helper.clustering;

import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

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
            
            // Calculate clustering quality metrics
            calculateClusteringMetrics(communities, graph);
            
            System.out.println("Spectral clustering complete. Found " + communities.size() + " communities.");
            return communities;
            
        } catch (Exception e) {
            System.err.println("Error during spectral clustering: " + e.getMessage());
            e.printStackTrace();
            return fallbackClustering();
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
                            networkSimilarity = Math.exp(-weight / sigma);
                        }
                        
                        // Calculate geographic similarity (1.0 means close, 0.0 means far)
                        double geoDist = calculateGeoDistance(node1, node2);
                        double geoSimilarity = Math.exp(-geoDist / (sigma * maxGeoDist));
                        
                        // Combine network and geographic similarity
                        similarity = (1.0 - geoWeight) * networkSimilarity + geoWeight * geoSimilarity;
                    } else {
                        // No direct edge, use distance-based similarity with steeper decay
                        double geoDist = calculateGeoDistance(node1, node2);
                        similarity = Math.exp(-geoDist / (sigma * maxGeoDist));
                        
                        // Apply additional penalty for non-connected nodes
                        similarity *= 0.5;
                    }
                }
                
                // Set symmetric values in the matrix
                similarityMatrix.setEntry(i, j, similarity);
                similarityMatrix.setEntry(j, i, similarity);
            }
        }
        
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
        
        // Perform k-means clustering
        KMeansPlusPlusClusterer<EmbeddingPoint> clusterer = 
            new KMeansPlusPlusClusterer<>(numberOfClusters, maxIterations, new EuclideanDistance());
        
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
        
        return result;
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
} 