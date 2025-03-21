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
    private static final int DEFAULT_MAX_CLUSTER_SIZE = 0; // 0 means no maximum size limit
    private static final boolean DEFAULT_FORCE_NUM_CLUSTERS = false; // Don't force exact number by default
    private static final double DEFAULT_MAX_COMMUNITY_DIAMETER = 0.0; // 0 means no maximum diameter constraint
    
    // Configuration parameters
    private int numberOfClusters = DEFAULT_NUM_CLUSTERS;
    private double sigma = DEFAULT_SIGMA;
    private boolean useNormalizedCut = DEFAULT_USE_NORMALIZED_CUT;
    private int maxIterations = DEFAULT_MAX_ITERATIONS;
    private double geoWeight = DEFAULT_GEO_WEIGHT;
    private int minCommunitySize = DEFAULT_MIN_COMMUNITY_SIZE;
    private boolean preventSingletons = DEFAULT_PREVENT_SINGLETONS;
    private int maxClusterSize = DEFAULT_MAX_CLUSTER_SIZE;
    private boolean forceNumClusters = DEFAULT_FORCE_NUM_CLUSTERS;
    private double maxCommunityDiameter = DEFAULT_MAX_COMMUNITY_DIAMETER;
    
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
    
    /**
     * Sets the maximum community size. The algorithm will attempt to prevent
     * communities larger than this size by splitting them.
     * 
     * @param maxSize Maximum community size (0 means no maximum)
     * @return This config instance for method chaining
     */
    public SpectralClusteringConfig setMaxClusterSize(int maxSize) {
        this.maxClusterSize = Math.max(0, maxSize);
        return this;
    }
    
    /**
     * Sets whether to force the algorithm to create exactly the specified number of clusters.
     * If true, the algorithm will ensure that exactly numberOfClusters communities are created.
     * 
     * @param force True to force the exact number of clusters
     * @return This config instance for method chaining
     */
    public SpectralClusteringConfig setForceNumClusters(boolean force) {
        this.forceNumClusters = force;
        return this;
    }
    
    /**
     * Sets the maximum geographic diameter allowed for a community in meters.
     * Communities exceeding this diameter will be split into smaller ones.
     * A value of 0 means no maximum diameter constraint.
     * 
     * @param maxDiameter Maximum diameter in meters (0 for no constraint)
     * @return This config instance for method chaining
     */
    public SpectralClusteringConfig setMaxCommunityDiameter(double maxDiameter) {
        this.maxCommunityDiameter = Math.max(0, maxDiameter);
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
    
    /**
     * Gets the maximum community size limit
     * 
     * @return Maximum community size (0 means no maximum)
     */
    public int getMaxClusterSize() {
        return maxClusterSize;
    }
    
    /**
     * Checks if the exact number of clusters should be forced
     * 
     * @return True if the exact number of clusters should be forced
     */
    public boolean isForceNumClusters() {
        return forceNumClusters;
    }
    
    /**
     * Gets the maximum community diameter constraint
     * 
     * @return Maximum community diameter in meters (0 means no constraint)
     */
    public double getMaxCommunityDiameter() {
        return maxCommunityDiameter;
    }
} 