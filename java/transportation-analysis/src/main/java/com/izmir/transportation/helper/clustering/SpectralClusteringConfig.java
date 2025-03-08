package com.izmir.transportation.helper.clustering;

/**
 * Configuration class for Spectral Clustering algorithm.
 * Provides parameters to control the behavior of spectral clustering,
 * particularly for preventing singleton communities.
 * 
 * @author yagizugurveren
 */
public class SpectralClusteringConfig {
    
    // Default parameter values
    private static final double DEFAULT_SIGMA = 0.5;  // Controls similarity scaling
    private static final int DEFAULT_NUM_CLUSTERS = 8;
    private static final int DEFAULT_MAX_ITERATIONS = 100;
    private static final double DEFAULT_GEO_WEIGHT = 0.3;  // Weight for geographic distance
    private static final boolean DEFAULT_USE_NORMALIZED_CUT = true;
    private static final int DEFAULT_MIN_COMMUNITY_SIZE = 1; // Default does not enforce a minimum
    private static final boolean DEFAULT_PREVENT_SINGLETONS = false;
    
    // Configuration parameters
    private int numberOfClusters = DEFAULT_NUM_CLUSTERS;
    private double sigma = DEFAULT_SIGMA;
    private boolean useNormalizedCut = DEFAULT_USE_NORMALIZED_CUT;
    private int maxIterations = DEFAULT_MAX_ITERATIONS;
    private double geoWeight = DEFAULT_GEO_WEIGHT;
    private int minCommunitySize = DEFAULT_MIN_COMMUNITY_SIZE;
    private boolean preventSingletons = DEFAULT_PREVENT_SINGLETONS;
    
    /**
     * Default constructor with default parameters
     */
    public SpectralClusteringConfig() {
        // Uses default values
    }
    
    /**
     * Constructor with custom number of clusters
     * 
     * @param numberOfClusters Number of clusters to find
     */
    public SpectralClusteringConfig(int numberOfClusters) {
        this.numberOfClusters = Math.max(2, numberOfClusters);
    }
    
    /**
     * Sets the number of clusters (communities) to find.
     * 
     * @param k Number of clusters to find (must be at least 2)
     * @return This config instance for method chaining
     */
    public SpectralClusteringConfig setNumberOfClusters(int k) {
        this.numberOfClusters = Math.max(2, k);
        return this;
    }
    
    /**
     * Sets the sigma parameter that controls the scaling of the similarity measure.
     * Higher values lead to more connections between distant nodes.
     * 
     * @param sigma Scaling parameter (typical range: 0.1 to 1.0)
     * @return This config instance for method chaining
     */
    public SpectralClusteringConfig setSigma(double sigma) {
        this.sigma = Math.max(0.01, sigma);
        return this;
    }
    
    /**
     * Sets whether to use normalized spectral clustering (normalized cut) or unnormalized.
     * Normalized cut typically produces more balanced communities.
     * 
     * @param useNormalizedCut True to use normalized cut, false for unnormalized
     * @return This config instance for method chaining
     */
    public SpectralClusteringConfig setUseNormalizedCut(boolean useNormalizedCut) {
        this.useNormalizedCut = useNormalizedCut;
        return this;
    }
    
    /**
     * Sets the maximum number of iterations for k-means clustering.
     * 
     * @param maxIterations Maximum number of iterations
     * @return This config instance for method chaining
     */
    public SpectralClusteringConfig setMaxIterations(int maxIterations) {
        this.maxIterations = Math.max(10, maxIterations);
        return this;
    }
    
    /**
     * Sets the weight given to geographic distance in the similarity calculation.
     * Higher values give more importance to geographic proximity.
     * 
     * @param geoWeight Weight for geographic distance (0.0 to 1.0)
     * @return This config instance for method chaining
     */
    public SpectralClusteringConfig setGeographicWeight(double geoWeight) {
        this.geoWeight = Math.max(0.0, Math.min(1.0, geoWeight));
        return this;
    }
    
    /**
     * Sets the minimum community size. The algorithm will attempt to prevent
     * communities smaller than this size by adjusting the eigenvector embedding.
     * 
     * @param minSize Minimum community size
     * @return This config instance for method chaining
     */
    public SpectralClusteringConfig setMinCommunitySize(int minSize) {
        this.minCommunitySize = Math.max(1, minSize);
        return this;
    }
    
    /**
     * Sets whether to specifically prevent singleton communities.
     * If true, the algorithm will use heuristics to avoid creating communities with single nodes.
     * 
     * @param prevent True to prevent singleton communities
     * @return This config instance for method chaining
     */
    public SpectralClusteringConfig setPreventSingletons(boolean prevent) {
        this.preventSingletons = prevent;
        return this;
    }
    
    // Getters
    
    public int getNumberOfClusters() {
        return numberOfClusters;
    }
    
    public double getSigma() {
        return sigma;
    }
    
    public boolean isUseNormalizedCut() {
        return useNormalizedCut;
    }
    
    public int getMaxIterations() {
        return maxIterations;
    }
    
    public double getGeoWeight() {
        return geoWeight;
    }
    
    public int getMinCommunitySize() {
        return minCommunitySize;
    }
    
    public boolean isPreventSingletons() {
        return preventSingletons;
    }
} 