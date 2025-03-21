package com.izmir.transportation.helper.clustering;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;
import java.util.stream.IntStream;

import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.DiagonalMatrix;
import org.apache.commons.math3.linear.LUDecomposition;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;
import org.apache.commons.math3.linear.SingularValueDecomposition;
import org.jgrapht.Graph;
import org.jgrapht.graph.DefaultWeightedEdge;
import org.locationtech.jts.geom.Point;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.izmir.transportation.TransportationGraph;
import com.izmir.transportation.helper.Node;

/**
 * Implementation of the MvAGC (Multi-view Attributed Graph Clustering) algorithm.
 * 
 * This algorithm is designed for clustering multi-view attributed graphs, using:
 * 1. Graph filtering to smooth node features
 * 2. Anchor-based approximation to reduce computational complexity
 * 3. Multi-view integration to combine information from different graph views
 * 
 * The algorithm works in the following steps:
 * 1. Select anchor nodes using importance sampling
 * 2. Apply graph filter to smooth node features
 * 3. Learn a projection matrix using anchor nodes
 * 4. Project nodes to a low-dimensional space
 * 5. Apply K-means clustering on the projected features
 * 
 * @author yagizugurveren
 */
public class MvAGCClustering implements GraphClusteringAlgorithm {
    
    private static final Logger LOGGER = LoggerFactory.getLogger(MvAGCClustering.class);
    
    private final TransportationGraph transportationGraph;
    private int numClusters = 8; // Default number of clusters
    private int numAnchors = 100; // Default number of anchor nodes
    private int filterOrder = 2; // Default filter order (k)
    private double alpha = 5.0; // Default weight for the second term
    private double importanceSamplingPower = 1.0; // Default power for importance sampling
    private double gamma = -1.0; // Parameter for view fusion
    private double[] viewWeights = {1.0, 1.0}; // Default view weights
    private int minClusterSize = 5; // Default minimum cluster size
    private int maxClusterSize = 0; // Default maximum cluster size (0 means no limit)
    private boolean forceMinClusters = false; // Whether to force minimum number of clusters
    
    /**
     * Constructor with the transportation graph
     * 
     * @param transportationGraph The graph to cluster
     */
    public MvAGCClustering(TransportationGraph transportationGraph) {
        this.transportationGraph = transportationGraph;
    }
    
    /**
     * Set the number of clusters to detect
     * 
     * @param numClusters Number of clusters (at least 2)
     * @return This clustering instance for method chaining
     */
    public MvAGCClustering setNumClusters(int numClusters) {
        this.numClusters = Math.max(2, numClusters);
        LOGGER.info("Number of clusters set to {}", this.numClusters);
        return this;
    }
    
    /**
     * Set the number of anchor nodes to use
     * 
     * @param numAnchors Number of anchor nodes (should be less than total nodes)
     * @return This clustering instance for method chaining
     */
    public MvAGCClustering setNumAnchors(int numAnchors) {
        this.numAnchors = numAnchors;
        LOGGER.info("Number of anchor nodes set to {}", this.numAnchors);
        return this;
    }
    
    /**
     * Set the filter order (k)
     * 
     * @param filterOrder Order of the graph filter (higher means more smoothing)
     * @return This clustering instance for method chaining
     */
    public MvAGCClustering setFilterOrder(int filterOrder) {
        this.filterOrder = Math.max(1, filterOrder);
        LOGGER.info("Filter order set to {}", this.filterOrder);
        return this;
    }
    
    /**
     * Set the alpha parameter (weight for the second term)
     * 
     * @param alpha Alpha value (larger means more importance on graph structure)
     * @return This clustering instance for method chaining
     */
    public MvAGCClustering setAlpha(double alpha) {
        this.alpha = Math.max(0.1, alpha);
        LOGGER.info("Alpha parameter set to {}", this.alpha);
        return this;
    }
    
    /**
     * Set the importance sampling power parameter
     * 
     * @param power Power value for importance sampling (higher means more bias toward high-degree nodes)
     * @return This clustering instance for method chaining
     */
    public MvAGCClustering setImportanceSamplingPower(double power) {
        this.importanceSamplingPower = Math.max(0.1, power);
        LOGGER.info("Importance sampling power set to {}", this.importanceSamplingPower);
        return this;
    }
    
    /**
     * Set the minimum size for a cluster
     * 
     * @param minSize Minimum number of nodes per cluster (at least 1)
     * @return This clustering instance for method chaining
     */
    public MvAGCClustering setMinClusterSize(int minSize) {
        this.minClusterSize = Math.max(1, minSize);
        LOGGER.info("Minimum cluster size set to {}", this.minClusterSize);
        return this;
    }
    
    /**
     * Set the maximum size for a cluster
     * 
     * @param maxSize Maximum number of nodes per cluster (0 means no limit)
     * @return This clustering instance for method chaining
     */
    public MvAGCClustering setMaxClusterSize(int maxSize) {
        this.maxClusterSize = Math.max(0, maxSize);
        LOGGER.info("Maximum cluster size set to {}", this.maxClusterSize);
        return this;
    }
    
    /**
     * Set whether to force the minimum number of clusters
     * 
     * @param force Whether to force at least numClusters clusters
     * @return This clustering instance for method chaining
     */
    public MvAGCClustering setForceMinClusters(boolean force) {
        this.forceMinClusters = force;
        LOGGER.info("Force minimum clusters set to {}", this.forceMinClusters);
        return this;
    }
    
    /**
     * Determine the community/cluster assignment for each node in the graph
     * 
     * @return Map of community IDs to lists of nodes
     */
    public Map<Integer, List<Node>> detectCommunities() {
        LOGGER.info("Starting MvAGC clustering with {} clusters", numClusters);
        LOGGER.info("Min cluster size: {}, Max cluster size: {}, Force min clusters: {}", 
                   minClusterSize, maxClusterSize > 0 ? maxClusterSize : "unlimited", forceMinClusters);
        
        // Get the main graph and node features
        Graph<Node, DefaultWeightedEdge> graph = transportationGraph.getGraph();
        List<Node> nodes = new ArrayList<>(graph.vertexSet());
        int numNodes = nodes.size();
        
        // Ensure we don't try to use more anchor nodes than we have nodes
        numAnchors = Math.min(numAnchors, numNodes);
        
        // Create node feature matrix (X)
        RealMatrix features = createFeatureMatrix(nodes);
        
        // Create multiple views of the graph (adjacency matrices)
        List<RealMatrix> adjacencyViews = createGraphViews(graph, nodes);
        
        // Sample anchor nodes
        int[] anchorIndices = sampleAnchorNodes(adjacencyViews, numAnchors, importanceSamplingPower);
        
        // Apply graph filtering to get smoothed features
        List<RealMatrix> filteredFeatures = new ArrayList<>();
        List<RealMatrix> filteredAnchorFeatures = new ArrayList<>();
        List<RealMatrix> adjacencyAnchors = new ArrayList<>();
        
        for (int i = 0; i < adjacencyViews.size(); i++) {
            RealMatrix adj = adjacencyViews.get(i);
            
            // Normalize adjacency matrix
            RealMatrix normalizedAdj = normalizeAdjacency(adj);
            
            // Calculate filter matrix: G = I - 0.5 * (I - A)
            RealMatrix identity = MatrixUtils.createRealIdentityMatrix(numNodes);
            RealMatrix laplacian = identity.subtract(normalizedAdj);
            RealMatrix filter = identity.subtract(laplacian.scalarMultiply(0.5));
            
            // Apply filtering k times
            RealMatrix filtered = features.copy();
            for (int k = 0; k < filterOrder; k++) {
                filtered = filter.multiply(filtered);
            }
            
            filteredFeatures.add(filtered);
            
            // Extract anchor node features
            RealMatrix anchorFeatures = MatrixUtils.createRealMatrix(anchorIndices.length, filtered.getColumnDimension());
            for (int j = 0; j < anchorIndices.length; j++) {
                anchorFeatures.setRow(j, filtered.getRow(anchorIndices[j]));
            }
            filteredAnchorFeatures.add(anchorFeatures);
            
            // Extract anchor rows from adjacency matrix
            RealMatrix anchorAdj = MatrixUtils.createRealMatrix(anchorIndices.length, numNodes);
            for (int j = 0; j < anchorIndices.length; j++) {
                anchorAdj.setRow(j, normalizedAdj.getRow(anchorIndices[j]));
            }
            adjacencyAnchors.add(anchorAdj);
        }
        
        // Learn a unified projection matrix S
        RealMatrix S = learnProjectionMatrix(filteredFeatures, filteredAnchorFeatures, adjacencyAnchors);
        
        // Normalize S for spectral clustering
        RealVector rowSums = computeRowSums(S);
        double[] diagD = new double[rowSums.getDimension()];
        for (int i = 0; i < rowSums.getDimension(); i++) {
            diagD[i] = rowSums.getEntry(i) == 0 ? 0 : 1.0 / Math.sqrt(rowSums.getEntry(i));
        }
        
        DiagonalMatrix D = new DiagonalMatrix(diagD);
        RealMatrix normalizedS = D.multiply(S);
        
        // Compute S * S^T for SVD
        RealMatrix SST = normalizedS.multiply(normalizedS.transpose());
        
        // Apply SVD to get eigenvectors for clustering
        SingularValueDecomposition svd = new SingularValueDecomposition(SST);
        RealMatrix eigenvectors = svd.getU();
        double[] singularValues = svd.getSingularValues();
        
        // Log singular values to understand their distribution
        StringBuilder svBuilder = new StringBuilder("Singular values: ");
        for (int i = 0; i < Math.min(singularValues.length, 20); i++) {
            svBuilder.append(String.format("%.4e", singularValues[i]));
            if (i < Math.min(singularValues.length, 20) - 1) {
                svBuilder.append(", ");
            }
        }
        LOGGER.info(svBuilder.toString());
        
        // Check if we need to force multiple communities
        if (singularValues.length > 1 && singularValues[1] / singularValues[0] < 0.001) {
            LOGGER.warn("Singular value gap is very large, which may lead to only one community being detected");
            // Apply artificial perturbation to create more structure
            for (int i = 1; i < Math.min(numClusters, singularValues.length); i++) {
                singularValues[i] = Math.max(singularValues[i], singularValues[0] * 0.1);
            }
        }
        
        // Make sure we don't request more eigenvectors than available
        int actualNumClusters = Math.min(numClusters, eigenvectors.getColumnDimension());
        
        // Ensure we have at least 2 clusters
        actualNumClusters = Math.max(2, actualNumClusters);
        LOGGER.info("Using {} clusters for spectral embedding", actualNumClusters);
        
        // Take top actualNumClusters eigenvectors
        RealMatrix embeddingMatrix;
        try {
            embeddingMatrix = eigenvectors.getSubMatrix(0, eigenvectors.getRowDimension()-1, 0, actualNumClusters-1);
        } catch (Exception e) {
            LOGGER.error("Error extracting eigenvectors: {}", e.getMessage());
            // Fallback to fewer eigenvectors if needed
            actualNumClusters = Math.min(actualNumClusters, eigenvectors.getColumnDimension());
            embeddingMatrix = eigenvectors.getSubMatrix(0, eigenvectors.getRowDimension()-1, 0, actualNumClusters-1);
        }
        
        // Apply sigma^-0.5 scaling
        double[] sigmaPowNegHalf = new double[actualNumClusters];
        for (int i = 0; i < actualNumClusters; i++) {
            sigmaPowNegHalf[i] = singularValues[i] == 0 ? 0 : 1.0 / Math.sqrt(singularValues[i]);
        }
        
        DiagonalMatrix sigma = new DiagonalMatrix(sigmaPowNegHalf);
        RealMatrix C_hat = sigma.multiply(embeddingMatrix.transpose()).multiply(normalizedS);
        
        // Normalize rows to unit length to ensure points lie on unit hypersphere
        RealMatrix normalizedC_hat = normalizeRows(C_hat);
        
        // Transpose C_hat for k-means
        RealMatrix clusteringFeatures = normalizedC_hat.transpose();
        
        // Apply k-means clustering to find actualNumClusters clusters
        int[] clusterAssignments = kMeansClustering(clusteringFeatures, actualNumClusters);
        
        // Convert cluster assignments to the required format
        Map<Integer, List<Node>> communities = new HashMap<>();
        for (int i = 0; i < clusterAssignments.length; i++) {
            int cluster = clusterAssignments[i];
            if (!communities.containsKey(cluster)) {
                communities.put(cluster, new ArrayList<>());
            }
            communities.get(cluster).add(nodes.get(i));
        }
        
        // Apply minimum cluster size constraint
        if (minClusterSize > 1) {
            LOGGER.info("Enforcing minimum cluster size of {}", minClusterSize);
            communities = removeSmallCommunities(communities, minClusterSize);
        }
        
        // Force minimum number of clusters if needed
        if (forceMinClusters && communities.size() < numClusters) {
            LOGGER.info("Forcing at least {} clusters (currently have {})", numClusters, communities.size());
            communities = forceMinimumClusters(communities, nodes, numClusters);
        }
        
        // Apply maximum cluster size constraint
        if (maxClusterSize > 0) {
            LOGGER.info("Enforcing maximum cluster size of {}", maxClusterSize);
            communities = enforceMaximumClusterSize(communities, maxClusterSize);
            
            // Check if we still have any oversized communities
            boolean hasOversizedCommunities = false;
            for (List<Node> community : communities.values()) {
                if (community.size() > maxClusterSize) {
                    hasOversizedCommunities = true;
                    LOGGER.warn("Community still exceeds maximum size: {} > {}", 
                               community.size(), maxClusterSize);
                }
            }
            
            // Apply a second pass if needed
            if (hasOversizedCommunities) {
                LOGGER.info("Applying second pass to split remaining oversized communities");
                communities = enforceMaximumClusterSize(communities, maxClusterSize);
            }
        }
        
        LOGGER.info("MvAGC clustering completed, found {} communities", communities.size());
        
        // Force at least 2 communities if we ended up with only 1
        if (communities.size() == 1 && nodes.size() > 10) {
            LOGGER.warn("Only one community found, forcing split based on geography");
            communities = forceSplitByGeography(nodes, Math.min(13, numClusters));
        }
        
        return communities;
    }
    
    /**
     * Create feature matrix from node attributes
     * 
     * @param nodes List of nodes
     * @return Feature matrix where each row is a node and each column is a feature
     */
    private RealMatrix createFeatureMatrix(List<Node> nodes) {
        int numNodes = nodes.size();
        
        // For simplicity, we'll use node coordinates as features
        // In a real application, you might want to use more meaningful features
        int numFeatures = 2; // X and Y coordinates
        RealMatrix features = MatrixUtils.createRealMatrix(numNodes, numFeatures);
        
        for (int i = 0; i < numNodes; i++) {
            Node node = nodes.get(i);
            Point location = node.getLocation();
            features.setEntry(i, 0, location.getX());
            features.setEntry(i, 1, location.getY());
        }
        
        return features;
    }
    
    /**
     * Create multiple views of the graph as adjacency matrices
     * 
     * @param graph The graph
     * @param nodes Ordered list of nodes
     * @return List of adjacency matrices
     */
    private List<RealMatrix> createGraphViews(Graph<Node, DefaultWeightedEdge> graph, List<Node> nodes) {
        int numNodes = nodes.size();
        List<RealMatrix> views = new ArrayList<>();
        boolean isComplete = isCompleteGraph(graph);
        
        LOGGER.info("Graph appears to be {}", isComplete ? "complete" : "not complete");
        
        if (isComplete) {
            // For complete graphs, create views based on different distance metrics
            
            // First view: Geographical distance (inverse)
            RealMatrix adjacency1 = MatrixUtils.createRealMatrix(numNodes, numNodes);
            
            for (int i = 0; i < numNodes; i++) {
                for (int j = i + 1; j < numNodes; j++) {
                    Node node1 = nodes.get(i);
                    Node node2 = nodes.get(j);
                    
                    double distance = calculateGeographicalDistance(node1, node2);
                    double weight = 1.0 / (1.0 + distance / 1000.0); // Scale distance
                    
                    adjacency1.setEntry(i, j, weight);
                    adjacency1.setEntry(j, i, weight); // Ensure symmetry
                }
            }
            
            views.add(adjacency1);
            
            // Second view: Modified distance with different scaling
            RealMatrix adjacency2 = MatrixUtils.createRealMatrix(numNodes, numNodes);
            
            for (int i = 0; i < numNodes; i++) {
                for (int j = i + 1; j < numNodes; j++) {
                    Node node1 = nodes.get(i);
                    Node node2 = nodes.get(j);
                    
                    double distance = calculateGeographicalDistance(node1, node2);
                    // Use exponential decay for second view to create more contrast
                    double weight = Math.exp(-distance / 5000.0);
                    
                    adjacency2.setEntry(i, j, weight);
                    adjacency2.setEntry(j, i, weight); // Ensure symmetry
                }
            }
            
            views.add(adjacency2);
        } else {
            // Regular graph case - use connectivity for first view
            RealMatrix adjacency1 = MatrixUtils.createRealMatrix(numNodes, numNodes);
            
            for (int i = 0; i < numNodes; i++) {
                Node source = nodes.get(i);
                Set<DefaultWeightedEdge> edges = graph.edgesOf(source);
                
                for (DefaultWeightedEdge edge : edges) {
                    Node target = graph.getEdgeSource(edge);
                    if (target.equals(source)) {
                        target = graph.getEdgeTarget(edge);
                    }
                    
                    int j = nodes.indexOf(target);
                    if (j >= 0) {
                        double weight = graph.getEdgeWeight(edge);
                        adjacency1.setEntry(i, j, weight);
                        adjacency1.setEntry(j, i, weight); // Ensure symmetry
                    }
                }
            }
            
            views.add(adjacency1);
            
            // Second view based on geographical proximity
            RealMatrix adjacency2 = MatrixUtils.createRealMatrix(numNodes, numNodes);
            
            for (int i = 0; i < numNodes; i++) {
                for (int j = i + 1; j < numNodes; j++) {
                    Node node1 = nodes.get(i);
                    Node node2 = nodes.get(j);
                    
                    double distance = calculateGeographicalDistance(node1, node2);
                    double weight = 1.0 / (1.0 + distance / 1000.0); // Scale distance
                    
                    adjacency2.setEntry(i, j, weight);
                    adjacency2.setEntry(j, i, weight); // Ensure symmetry
                }
            }
            
            views.add(adjacency2);
        }
        
        return views;
    }
    
    /**
     * Check if the graph is likely a complete graph
     * 
     * @param graph The graph to check
     * @return True if the graph appears to be complete
     */
    private boolean isCompleteGraph(Graph<Node, DefaultWeightedEdge> graph) {
        int numNodes = graph.vertexSet().size();
        long maxEdges = (long) numNodes * (numNodes - 1) / 2; // Maximum possible edges in a complete graph
        int actualEdges = graph.edgeSet().size();
        
        // If the graph has at least 95% of the maximum edges, consider it practically complete
        return actualEdges >= 0.95 * maxEdges;
    }
    
    /**
     * Normalize adjacency matrix using symmetric normalization
     * 
     * @param adjacency Adjacency matrix
     * @return Normalized adjacency matrix
     */
    private RealMatrix normalizeAdjacency(RealMatrix adjacency) {
        int n = adjacency.getRowDimension();
        
        // Add self-loops (add identity)
        RealMatrix adjacencyWithSelfLoops = adjacency.copy();
        for (int i = 0; i < n; i++) {
            adjacencyWithSelfLoops.addToEntry(i, i, 1.0);
        }
        
        // Calculate degree vector
        RealVector degrees = computeRowSums(adjacencyWithSelfLoops);
        
        // Create D^(-1/2)
        double[] dPowNegHalf = new double[n];
        for (int i = 0; i < n; i++) {
            dPowNegHalf[i] = 1.0 / Math.sqrt(degrees.getEntry(i));
        }
        
        DiagonalMatrix dPowNegHalfMatrix = new DiagonalMatrix(dPowNegHalf);
        
        // D^(-1/2) * A * D^(-1/2)
        return dPowNegHalfMatrix.multiply(adjacencyWithSelfLoops).multiply(dPowNegHalfMatrix);
    }
    
    /**
     * Compute row sums of a matrix
     * 
     * @param matrix Input matrix
     * @return Vector of row sums
     */
    private RealVector computeRowSums(RealMatrix matrix) {
        int rows = matrix.getRowDimension();
        double[] sums = new double[rows];
        
        for (int i = 0; i < rows; i++) {
            double sum = 0.0;
            for (int j = 0; j < matrix.getColumnDimension(); j++) {
                sum += matrix.getEntry(i, j);
            }
            sums[i] = sum;
        }
        
        return new ArrayRealVector(sums);
    }
    
    /**
     * Sample anchor nodes using importance sampling
     * 
     * @param adjacencyViews List of adjacency matrices
     * @param numAnchors Number of anchor nodes to sample
     * @param alpha Power parameter for importance sampling
     * @return Array of sampled node indices
     */
    private int[] sampleAnchorNodes(List<RealMatrix> adjacencyViews, int numAnchors, double alpha) {
        int numNodes = adjacencyViews.get(0).getRowDimension();
        
        if (numNodes <= numAnchors) {
            // If we want as many anchors as nodes, just return all indices
            return IntStream.range(0, numNodes).toArray();
        }
        
        // For large graphs, ensure a reasonable number of anchors
        numAnchors = Math.min(numAnchors, numNodes / 2);
        
        // Calculate combined degree from all views
        double[] degrees = new double[numNodes];
        for (RealMatrix adj : adjacencyViews) {
            for (int i = 0; i < numNodes; i++) {
                for (int j = 0; j < numNodes; j++) {
                    degrees[i] += adj.getEntry(i, j);
                }
            }
        }
        
        // For complete graphs, introduce geographical diversity
        // Add noise to the degrees to create more variation
        Random random = new Random(42);
        for (int i = 0; i < numNodes; i++) {
            degrees[i] *= (1.0 + 0.2 * random.nextGaussian());
        }
        
        // Apply power for importance sampling
        double[] importanceWeights = new double[numNodes];
        double weightSum = 0.0;
        for (int i = 0; i < numNodes; i++) {
            importanceWeights[i] = Math.pow(degrees[i], alpha);
            weightSum += importanceWeights[i];
        }
        
        // Normalize to probabilities
        double[] probabilities = new double[numNodes];
        for (int i = 0; i < numNodes; i++) {
            probabilities[i] = importanceWeights[i] / weightSum;
        }
        
        // Convert to cumulative probabilities
        double[] cumulativeProbabilities = new double[numNodes];
        cumulativeProbabilities[0] = probabilities[0];
        for (int i = 1; i < numNodes; i++) {
            cumulativeProbabilities[i] = cumulativeProbabilities[i-1] + probabilities[i];
        }
        
        // Sample anchor nodes without replacement
        boolean[] selected = new boolean[numNodes];
        int[] anchorIndices = new int[numAnchors];
        
        int anchorCount = 0;
        while (anchorCount < numAnchors) {
            double r = random.nextDouble();
            int selectedIndex = binarySearch(cumulativeProbabilities, r);
            
            if (selectedIndex < 0 || selectedIndex >= numNodes) {
                continue; // Skip invalid indices
            }
            
            if (!selected[selectedIndex]) {
                selected[selectedIndex] = true;
                anchorIndices[anchorCount++] = selectedIndex;
            }
        }
        
        LOGGER.info("Selected {} anchor nodes out of {} total nodes", numAnchors, numNodes);
        return anchorIndices;
    }
    
    /**
     * Binary search to find the index where the value would be inserted
     * 
     * @param array Sorted array
     * @param value Value to find
     * @return Index where value would be inserted
     */
    private int binarySearch(double[] array, double value) {
        int low = 0;
        int high = array.length - 1;
        
        while (low < high) {
            int mid = (low + high) / 2;
            if (array[mid] > value) {
                high = mid;
            } else {
                low = mid + 1;
            }
        }
        
        return low;
    }
    
    /**
     * Learn a unified projection matrix by optimizing across multiple views
     * 
     * @param filteredFeatures List of filtered feature matrices for each view
     * @param filteredAnchorFeatures List of filtered anchor feature matrices for each view
     * @param adjacencyAnchors List of anchor adjacency matrices for each view
     * @return Projection matrix S
     */
    private RealMatrix learnProjectionMatrix(
            List<RealMatrix> filteredFeatures, 
            List<RealMatrix> filteredAnchorFeatures, 
            List<RealMatrix> adjacencyAnchors) {
        
        int numViews = filteredFeatures.size();
        int numAnchors = adjacencyAnchors.get(0).getRowDimension();
        int numNodes = filteredFeatures.get(0).getRowDimension();
        
        // Add a small regularization factor for numerical stability
        double regFactor = 1e-6;
        
        // Create identity matrix for the regularization term
        RealMatrix identityMatrix = MatrixUtils.createRealIdentityMatrix(numAnchors);
        
        // Initialize the projection matrix S using least squares on the first view
        RealMatrix S = initializeProjectionMatrix(
                filteredAnchorFeatures.get(0), 
                filteredFeatures.get(0), 
                adjacencyAnchors.get(0), 
                alpha);
        
        // Iteratively optimize S
        for (int iteration = 0; iteration < 20; iteration++) {
            try {
                // Update view weights
                double[] weights = new double[numViews];
                double weightSum = 0.0;
                
                for (int i = 0; i < numViews; i++) {
                    // Calculate reconstruction error
                    RealMatrix diff1 = filteredFeatures.get(i).transpose().subtract(
                            filteredAnchorFeatures.get(i).transpose().multiply(S));
                    
                    // Calculate graph structure preservation error
                    RealMatrix diff2 = S.subtract(adjacencyAnchors.get(i));
                    
                    // Compute combined error
                    double error = Math.pow(getNorm(diff1), 2) + alpha * Math.pow(getNorm(diff2), 2);
                    error = Math.max(error, 1e-10); // Avoid division by zero
                    
                    // Update weight using multiplicative update rule
                    weights[i] = Math.pow(-error / gamma, 1.0 / (gamma - 1.0));
                    weightSum += weights[i];
                }
                
                // If weightSum is too small, reset to uniform weights
                if (weightSum < 1e-10) {
                    for (int i = 0; i < numViews; i++) {
                        weights[i] = 1.0 / numViews;
                    }
                    weightSum = 1.0;
                }
                
                // Normalize weights
                for (int i = 0; i < numViews; i++) {
                    viewWeights[i] = weights[i] / weightSum;
                }
                
                // Update S using the weighted combination of views
                RealMatrix leftTerm = MatrixUtils.createRealMatrix(numAnchors, numAnchors);
                RealMatrix rightTerm = MatrixUtils.createRealMatrix(numAnchors, numNodes);
                
                for (int i = 0; i < numViews; i++) {
                    RealMatrix anchorFeatures = filteredAnchorFeatures.get(i);
                    RealMatrix features = filteredFeatures.get(i);
                    RealMatrix anchorAdj = adjacencyAnchors.get(i);
                    
                    // Left term: X_a * X_a^T + alpha * I
                    RealMatrix weightedAnchorProduct = anchorFeatures.multiply(anchorFeatures.transpose()).add(
                            identityMatrix.scalarMultiply(alpha + regFactor)); // Add regularization
                    leftTerm = leftTerm.add(weightedAnchorProduct.scalarMultiply(viewWeights[i]));
                    
                    // Right term: X_a * X^T + alpha * A_a
                    RealMatrix weightedFeatureProduct = anchorFeatures.multiply(features.transpose()).add(
                            anchorAdj.scalarMultiply(alpha));
                    rightTerm = rightTerm.add(weightedFeatureProduct.scalarMultiply(viewWeights[i]));
                }
                
                // Add additional regularization to ensure the matrix is well-conditioned
                for (int i = 0; i < numAnchors; i++) {
                    leftTerm.addToEntry(i, i, regFactor);
                }
                
                // Solve for S: S = (X_a * X_a^T + alpha * I)^(-1) * (X_a * X^T + alpha * A_a)
                LUDecomposition decomposition = new LUDecomposition(leftTerm);
                if (decomposition.getSolver().isNonSingular()) {
                    S = decomposition.getSolver().solve(rightTerm);
                } else {
                    LOGGER.warn("Matrix is singular at iteration {}, adding more regularization", iteration);
                    // Add more regularization and try again
                    for (int i = 0; i < numAnchors; i++) {
                        leftTerm.addToEntry(i, i, regFactor * 10);
                    }
                    decomposition = new LUDecomposition(leftTerm);
                    if (decomposition.getSolver().isNonSingular()) {
                        S = decomposition.getSolver().solve(rightTerm);
                    } else {
                        LOGGER.warn("Matrix is still singular, using previous S");
                    }
                }
            } catch (Exception e) {
                LOGGER.error("Error during matrix optimization at iteration {}: {}", iteration, e.getMessage());
                break;
            }
        }
        
        return S;
    }
    
    /**
     * Initialize the projection matrix using least squares
     * 
     * @param anchorFeatures Anchor features
     * @param features All features
     * @param anchorAdj Anchor adjacency matrix
     * @param alpha Regularization parameter
     * @return Initial projection matrix S
     */
    private RealMatrix initializeProjectionMatrix(
            RealMatrix anchorFeatures, 
            RealMatrix features, 
            RealMatrix anchorAdj, 
            double alpha) {
        
        int numAnchors = anchorFeatures.getRowDimension();
        
        // Create identity matrix
        RealMatrix identityMatrix = MatrixUtils.createRealIdentityMatrix(numAnchors);
        
        // Left term: X_a * X_a^T + alpha * I
        RealMatrix leftTerm = anchorFeatures.multiply(anchorFeatures.transpose()).add(
                identityMatrix.scalarMultiply(alpha));
        
        // Right term: X_a * X^T + alpha * A_a
        RealMatrix rightTerm = anchorFeatures.multiply(features.transpose()).add(
                anchorAdj.scalarMultiply(alpha));
        
        // Solve S = (X_a * X_a^T + alpha * I)^(-1) * (X_a * X^T + alpha * A_a)
        LUDecomposition decomposition = new LUDecomposition(leftTerm);
        if (decomposition.getSolver().isNonSingular()) {
            return decomposition.getSolver().solve(rightTerm);
        } else {
            LOGGER.warn("Matrix is singular during initialization, using identity matrix");
            return MatrixUtils.createRealIdentityMatrix(numAnchors);
        }
    }
    
    /**
     * Calculate Frobenius norm of a matrix
     * 
     * @param matrix The matrix
     * @return Frobenius norm
     */
    private double getNorm(RealMatrix matrix) {
        double sumSquared = 0.0;
        
        for (int i = 0; i < matrix.getRowDimension(); i++) {
            for (int j = 0; j < matrix.getColumnDimension(); j++) {
                sumSquared += Math.pow(matrix.getEntry(i, j), 2);
            }
        }
        
        return Math.sqrt(sumSquared);
    }
    
    /**
     * Perform k-means clustering
     * 
     * @param data Data matrix (each row is a data point)
     * @param k Number of clusters
     * @return Cluster assignments
     */
    private int[] kMeansClustering(RealMatrix data, int k) {
        try {
            int numPoints = data.getRowDimension();
            int numDimensions = data.getColumnDimension();
            
            // Log data dimensionality
            LOGGER.info("K-means input data: {} points with {} dimensions", numPoints, numDimensions);
            
            // Ensure k is not larger than the number of points
            if (k > numPoints) {
                LOGGER.warn("Requested {} clusters but only have {} points, reducing to {}", 
                          k, numPoints, numPoints);
                k = numPoints;
            }
            
            // If k is 1, just assign everything to one cluster
            if (k == 1) {
                LOGGER.warn("Only one cluster requested, assigning all points to it");
                return new int[numPoints]; // All zeros
            }
            
            // Initialize centroids using k-means++ method for better initialization
            RealMatrix centroids = initializeKMeansPlusPlusCentroids(data, k);
            
            // K-means iterations
            int[] assignments = new int[numPoints];
            boolean changed = true;
            int maxIterations = 300; // Increase max iterations for better convergence
            int iteration = 0;
            
            while (changed && iteration < maxIterations) {
                changed = false;
                
                // Assign points to nearest centroid
                for (int i = 0; i < numPoints; i++) {
                    double[] point = data.getRow(i);
                    int bestCluster = 0;
                    double bestDistance = Double.MAX_VALUE;
                    
                    for (int j = 0; j < k; j++) {
                        double[] centroid = centroids.getRow(j);
                        double distance = calculateEuclideanDistance(point, centroid);
                        
                        if (distance < bestDistance) {
                            bestDistance = distance;
                            bestCluster = j;
                        }
                    }
                    
                    if (assignments[i] != bestCluster) {
                        assignments[i] = bestCluster;
                        changed = true;
                    }
                }
                
                // Check if we have empty clusters
                boolean hasEmptyCluster = false;
                int[] counts = new int[k];
                for (int i = 0; i < numPoints; i++) {
                    counts[assignments[i]]++;
                }
                
                for (int i = 0; i < k; i++) {
                    if (counts[i] == 0) {
                        hasEmptyCluster = true;
                        LOGGER.warn("Empty cluster detected at iteration {}: cluster {}", iteration, i);
                        
                        // Find the cluster with the most points
                        int largestCluster = 0;
                        int maxCount = counts[0];
                        for (int j = 1; j < k; j++) {
                            if (counts[j] > maxCount) {
                                maxCount = counts[j];
                                largestCluster = j;
                            }
                        }
                        
                        // Find the point in the largest cluster that's farthest from its centroid
                        int farthestPoint = -1;
                        double maxDistance = -1;
                        for (int j = 0; j < numPoints; j++) {
                            if (assignments[j] == largestCluster) {
                                double distance = calculateEuclideanDistance(data.getRow(j), centroids.getRow(largestCluster));
                                if (distance > maxDistance) {
                                    maxDistance = distance;
                                    farthestPoint = j;
                                }
                            }
                        }
                        
                        if (farthestPoint != -1) {
                            // Reassign the farthest point to the empty cluster
                            assignments[farthestPoint] = i;
                            counts[largestCluster]--;
                            counts[i]++;
                            changed = true;
                        }
                    }
                }
                
                // Update centroids
                RealMatrix newCentroids = MatrixUtils.createRealMatrix(k, numDimensions);
                
                for (int i = 0; i < numPoints; i++) {
                    int cluster = assignments[i];
                    double[] point = data.getRow(i);
                    double[] centroid = newCentroids.getRow(cluster);
                    
                    for (int j = 0; j < numDimensions; j++) {
                        centroid[j] += point[j];
                    }
                    
                    newCentroids.setRow(cluster, centroid);
                }
                
                // Normalize centroids by count
                for (int i = 0; i < k; i++) {
                    if (counts[i] > 0) {
                        double[] centroid = newCentroids.getRow(i);
                        for (int j = 0; j < numDimensions; j++) {
                            centroid[j] /= counts[i];
                        }
                        newCentroids.setRow(i, centroid);
                    } else {
                        // If a cluster is still empty (should not happen after the fix above)
                        newCentroids.setRow(i, centroids.getRow(i));
                    }
                }
                
                centroids = newCentroids;
                iteration++;
                
                // Log progress every 50 iterations
                if (iteration % 50 == 0) {
                    LOGGER.info("K-means iteration {}/{}", iteration, maxIterations);
                }
            }
            
            // Check final cluster sizes
            int[] clusterSizes = new int[k];
            for (int i = 0; i < numPoints; i++) {
                clusterSizes[assignments[i]]++;
            }
            
            StringBuilder sizesBuilder = new StringBuilder("Final cluster sizes: ");
            for (int i = 0; i < k; i++) {
                sizesBuilder.append("Cluster ").append(i).append(": ").append(clusterSizes[i]).append(" points");
                if (i < k - 1) {
                    sizesBuilder.append(", ");
                }
            }
            LOGGER.info(sizesBuilder.toString());
            
            LOGGER.info("K-means converged after {} iterations", iteration);
            return assignments;
        } catch (Exception e) {
            LOGGER.error("Error during k-means clustering: {}", e.getMessage(), e);
            // Return a simple partition as fallback
            return fallbackClustering(data.getRowDimension(), k);
        }
    }
    
    /**
     * Initialize centroids using the k-means++ algorithm
     * 
     * @param data Data matrix
     * @param k Number of clusters
     * @return Matrix of initial centroids
     */
    private RealMatrix initializeKMeansPlusPlusCentroids(RealMatrix data, int k) {
        int numPoints = data.getRowDimension();
        int numDimensions = data.getColumnDimension();
        
        RealMatrix centroids = MatrixUtils.createRealMatrix(k, numDimensions);
        Random random = new Random(42); // Fixed seed for reproducibility
        
        // Choose the first centroid randomly
        int firstCentroidIndex = random.nextInt(numPoints);
        centroids.setRow(0, data.getRow(firstCentroidIndex));
        
        // Choose remaining centroids with probability proportional to squared distance
        for (int i = 1; i < k; i++) {
            double[] distanceSquared = new double[numPoints];
            double sumDistanceSquared = 0.0;
            
            // Calculate squared distance from each point to nearest centroid
            for (int j = 0; j < numPoints; j++) {
                double minDistance = Double.MAX_VALUE;
                double[] point = data.getRow(j);
                
                for (int c = 0; c < i; c++) {
                    double[] centroid = centroids.getRow(c);
                    double distance = calculateEuclideanDistance(point, centroid);
                    minDistance = Math.min(minDistance, distance);
                }
                
                distanceSquared[j] = minDistance * minDistance;
                sumDistanceSquared += distanceSquared[j];
            }
            
            // If sum is too small, use random initialization instead
            if (sumDistanceSquared < 1e-10) {
                LOGGER.warn("Distance sum too small in k-means++, using random initialization");
                int randomIndex = random.nextInt(numPoints);
                centroids.setRow(i, data.getRow(randomIndex));
                continue;
            }
            
            // Choose next centroid with probability proportional to squared distance
            double r = random.nextDouble() * sumDistanceSquared;
            double accumulator = 0.0;
            int selectedIndex = -1;
            
            for (int j = 0; j < numPoints; j++) {
                accumulator += distanceSquared[j];
                if (accumulator >= r) {
                    selectedIndex = j;
                    break;
                }
            }
            
            // If selection fails, choose randomly
            if (selectedIndex == -1) {
                selectedIndex = random.nextInt(numPoints);
            }
            
            centroids.setRow(i, data.getRow(selectedIndex));
        }
        
        return centroids;
    }
    
    /**
     * Fallback clustering method when k-means fails
     * 
     * @param numPoints Number of data points
     * @param k Number of clusters
     * @return Simple cluster assignments
     */
    private int[] fallbackClustering(int numPoints, int k) {
        LOGGER.info("Using fallback clustering method to partition {} points into {} clusters", numPoints, k);
        int[] assignments = new int[numPoints];
        int pointsPerCluster = numPoints / k;
        
        for (int i = 0; i < numPoints; i++) {
            int cluster = Math.min(i / pointsPerCluster, k - 1);
            assignments[i] = cluster;
        }
        
        return assignments;
    }
    
    /**
     * Calculate Euclidean distance between two vectors
     * 
     * @param v1 First vector
     * @param v2 Second vector
     * @return Euclidean distance
     */
    private double calculateEuclideanDistance(double[] v1, double[] v2) {
        double sum = 0.0;
        for (int i = 0; i < v1.length; i++) {
            double diff = v1[i] - v2[i];
            sum += diff * diff;
        }
        return Math.sqrt(sum);
    }
    
    /**
     * Calculate geographical distance between two nodes
     * 
     * @param node1 First node
     * @param node2 Second node
     * @return Distance in meters
     */
    private double calculateGeographicalDistance(Node node1, Node node2) {
        Point p1 = node1.getLocation();
        Point p2 = node2.getLocation();
        
        double earthRadius = 6371000; // meters
        double lat1 = Math.toRadians(p1.getY());
        double lon1 = Math.toRadians(p1.getX());
        double lat2 = Math.toRadians(p2.getY());
        double lon2 = Math.toRadians(p2.getX());
        
        double dLat = lat2 - lat1;
        double dLon = lon2 - lon1;
        
        double a = Math.sin(dLat/2) * Math.sin(dLat/2) +
                   Math.cos(lat1) * Math.cos(lat2) *
                   Math.sin(dLon/2) * Math.sin(dLon/2);
        double c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1-a));
        
        return earthRadius * c;
    }
    
    /**
     * Normalize rows of a matrix to unit length
     * 
     * @param matrix Input matrix
     * @return Matrix with normalized rows
     */
    private RealMatrix normalizeRows(RealMatrix matrix) {
        int rows = matrix.getRowDimension();
        int cols = matrix.getColumnDimension();
        RealMatrix normalized = MatrixUtils.createRealMatrix(rows, cols);
        
        for (int i = 0; i < rows; i++) {
            double[] row = matrix.getRow(i);
            double norm = 0.0;
            for (double val : row) {
                norm += val * val;
            }
            norm = Math.sqrt(norm);
            
            if (norm > 1e-10) {
                for (int j = 0; j < cols; j++) {
                    normalized.setEntry(i, j, matrix.getEntry(i, j) / norm);
                }
            } else {
                // If norm is too small, don't normalize
                normalized.setRow(i, row);
            }
        }
        
        return normalized;
    }
    
    /**
     * Remove communities that are too small
     * 
     * @param communities Original communities
     * @param minSize Minimum community size
     * @return Processed communities
     */
    private Map<Integer, List<Node>> removeSmallCommunities(Map<Integer, List<Node>> communities, int minSize) {
        Map<Integer, List<Node>> result = new HashMap<>();
        List<Node> orphanNodes = new ArrayList<>();
        
        // Identify small communities and collect their nodes
        for (Map.Entry<Integer, List<Node>> entry : communities.entrySet()) {
            if (entry.getValue().size() >= minSize) {
                result.put(entry.getKey(), entry.getValue());
            } else {
                orphanNodes.addAll(entry.getValue());
                LOGGER.info("Removing small community of size {}", entry.getValue().size());
            }
        }
        
        // If all communities are small, keep the largest one
        if (result.isEmpty() && !communities.isEmpty()) {
            Map.Entry<Integer, List<Node>> largest = null;
            for (Map.Entry<Integer, List<Node>> entry : communities.entrySet()) {
                if (largest == null || entry.getValue().size() > largest.getValue().size()) {
                    largest = entry;
                }
            }
            if (largest != null) {
                result.put(largest.getKey(), largest.getValue());
                orphanNodes.removeAll(largest.getValue());
            }
        }
        
        // Assign orphan nodes to the closest community
        if (!orphanNodes.isEmpty() && !result.isEmpty()) {
            for (Node node : orphanNodes) {
                int bestCommunity = -1;
                double bestDistance = Double.MAX_VALUE;
                
                for (Map.Entry<Integer, List<Node>> entry : result.entrySet()) {
                    double avgDistance = calculateAverageDistance(node, entry.getValue());
                    if (avgDistance < bestDistance) {
                        bestDistance = avgDistance;
                        bestCommunity = entry.getKey();
                    }
                }
                
                if (bestCommunity != -1) {
                    result.get(bestCommunity).add(node);
                } else {
                    // This shouldn't happen, but just in case
                    int firstKey = result.keySet().iterator().next();
                    result.get(firstKey).add(node);
                }
            }
        }
        
        return result;
    }
    
    /**
     * Calculate average geographical distance between a node and a list of nodes
     * 
     * @param node The node
     * @param nodes List of nodes
     * @return Average distance
     */
    private double calculateAverageDistance(Node node, List<Node> nodes) {
        double totalDistance = 0.0;
        for (Node other : nodes) {
            totalDistance += calculateGeographicalDistance(node, other);
        }
        return totalDistance / nodes.size();
    }
    
    /**
     * Force split communities based on geography when clustering fails
     * 
     * @param nodes List of all nodes
     * @param numClusters Number of clusters to create
     * @return Map of community IDs to node lists
     */
    private Map<Integer, List<Node>> forceSplitByGeography(List<Node> nodes, int numClusters) {
        LOGGER.info("Using geographical splitting to create {} communities", numClusters);
        Map<Integer, List<Node>> communities = new HashMap<>();
        
        // Create feature matrix using just geographical coordinates
        RealMatrix features = createFeatureMatrix(nodes);
        
        // Apply k-means directly on geographical coordinates
        int[] assignments = kMeansClustering(features, numClusters);
        
        // Build communities from assignments
        for (int i = 0; i < assignments.length; i++) {
            int cluster = assignments[i];
            if (!communities.containsKey(cluster)) {
                communities.put(cluster, new ArrayList<>());
            }
            communities.get(cluster).add(nodes.get(i));
        }
        
        return communities;
    }
    
    /**
     * Force a minimum number of clusters by splitting larger communities
     * 
     * @param communities Current communities
     * @param nodes All nodes in the graph
     * @param minClusters Minimum number of clusters required
     * @return Updated communities with at least minClusters clusters
     */
    private Map<Integer, List<Node>> forceMinimumClusters(Map<Integer, List<Node>> communities, 
                                                        List<Node> nodes, 
                                                        int minClusters) {
        Map<Integer, List<Node>> result = new HashMap<>(communities);
        
        // If we already have enough clusters, return as is
        if (result.size() >= minClusters) {
            return result;
        }
        
        int nextClusterId = 0;
        for (int id : result.keySet()) {
            nextClusterId = Math.max(nextClusterId, id + 1);
        }
        
        // Keep splitting largest communities until we have enough
        while (result.size() < minClusters) {
            // Find the largest community
            Map.Entry<Integer, List<Node>> largest = null;
            for (Map.Entry<Integer, List<Node>> entry : result.entrySet()) {
                if (largest == null || entry.getValue().size() > largest.getValue().size()) {
                    largest = entry;
                }
            }
            
            // If largest is too small to split, stop
            if (largest == null || largest.getValue().size() < minClusterSize * 2) {
                LOGGER.warn("Cannot create {} clusters without violating minimum size constraint", minClusters);
                break;
            }
            
            // Split the largest community using geographic distance
            List<Node> communityNodes = largest.getValue();
            
            // Find the two nodes farthest apart
            double maxDistance = 0;
            Node farthest1 = null;
            Node farthest2 = null;
            
            for (int i = 0; i < communityNodes.size(); i++) {
                for (int j = i + 1; j < communityNodes.size(); j++) {
                    double distance = calculateGeographicalDistance(
                        communityNodes.get(i), communityNodes.get(j));
                    if (distance > maxDistance) {
                        maxDistance = distance;
                        farthest1 = communityNodes.get(i);
                        farthest2 = communityNodes.get(j);
                    }
                }
            }
            
            // Split into two clusters based on distance to these two nodes
            List<Node> cluster1 = new ArrayList<>();
            List<Node> cluster2 = new ArrayList<>();
            
            for (Node node : communityNodes) {
                double dist1 = calculateGeographicalDistance(node, farthest1);
                double dist2 = calculateGeographicalDistance(node, farthest2);
                
                if (dist1 <= dist2) {
                    cluster1.add(node);
                } else {
                    cluster2.add(node);
                }
            }
            
            // Replace original cluster with cluster1 and add cluster2
            result.put(largest.getKey(), cluster1);
            result.put(nextClusterId++, cluster2);
            
            LOGGER.info("Split community {} ({} nodes) into two: {} and {} nodes", 
                      largest.getKey(), communityNodes.size(), cluster1.size(), cluster2.size());
        }
        
        return result;
    }
    
    /**
     * Enforce maximum cluster size by splitting oversized communities
     * 
     * @param communities Current communities
     * @param maxSize Maximum allowed community size
     * @return Updated communities with no community larger than maxSize
     */
    private Map<Integer, List<Node>> enforceMaximumClusterSize(Map<Integer, List<Node>> communities, 
                                                             int maxSize) {
        Map<Integer, List<Node>> result = new HashMap<>();
        int nextClusterId = 0;
        
        // Find next available cluster ID
        for (int id : communities.keySet()) {
            nextClusterId = Math.max(nextClusterId, id + 1);
        }
        
        for (Map.Entry<Integer, List<Node>> entry : communities.entrySet()) {
            int clusterId = entry.getKey();
            List<Node> nodes = entry.getValue();
            
            // If size is within limit, keep as is
            if (nodes.size() <= maxSize) {
                result.put(clusterId, new ArrayList<>(nodes));
                continue;
            }
            
            // Otherwise, split into multiple communities
            LOGGER.info("Splitting community {} with size {} to enforce max size {}", 
                      clusterId, nodes.size(), maxSize);
            
            // Create geographically-based clusters
            int numSubClusters = (int) Math.ceil((double) nodes.size() / maxSize);
            
            // K-means clustering based on geographic coordinates
            double[][] coordinates = new double[nodes.size()][2];
            for (int i = 0; i < nodes.size(); i++) {
                Point location = nodes.get(i).getLocation();
                coordinates[i][0] = location.getX();
                coordinates[i][1] = location.getY();
            }
            
            RealMatrix coordMatrix = MatrixUtils.createRealMatrix(coordinates);
            int[] assignments = kMeansClustering(coordMatrix, numSubClusters);
            
            // Create new communities
            Map<Integer, List<Node>> subCommunities = new HashMap<>();
            for (int i = 0; i < assignments.length; i++) {
                int subClusterId = assignments[i];
                if (!subCommunities.containsKey(subClusterId)) {
                    subCommunities.put(subClusterId, new ArrayList<>());
                }
                subCommunities.get(subClusterId).add(nodes.get(i));
            }
            
            // Add first subcommunity with original ID
            if (!subCommunities.isEmpty()) {
                Integer firstKey = subCommunities.keySet().iterator().next();
                result.put(clusterId, subCommunities.get(firstKey));
                subCommunities.remove(firstKey);
            }
            
            // Add other subcommunities with new IDs
            for (List<Node> subCommunity : subCommunities.values()) {
                result.put(nextClusterId++, subCommunity);
            }
            
            LOGGER.info("Split community {} into {} sub-communities", clusterId, numSubClusters);
        }
        
        return result;
    }
    
    /**
     * This is the method required by GraphClusteringAlgorithm interface
     */
    @Override
    public List<List<Node>> findCommunities(TransportationGraph graph) {
        Map<Integer, List<Node>> communities = detectCommunities();
        return new ArrayList<>(communities.values());
    }
} 