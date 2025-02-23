package com.izmir.transportation.helper.clustering;

import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.EigenDecomposition;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;
import org.apache.commons.math3.ml.clustering.CentroidCluster;
import org.apache.commons.math3.ml.clustering.DoublePoint;
import org.apache.commons.math3.ml.clustering.KMeansPlusPlusClusterer;
import org.apache.commons.math3.ml.distance.EuclideanDistance;

import com.izmir.transportation.AffinityMatrix;
import com.izmir.transportation.TransportationGraph;
import com.izmir.transportation.helper.Node;

public class SpectralClustering implements GraphClusteringAlgorithm {
    private final int clusters;
    private final int maxIterations;
    private static final double EPSILON = 1e-10;

    public SpectralClustering(int clusters, int maxIterations) {
        this.clusters = clusters;
        this.maxIterations = maxIterations;
    }

    @Override
    public List<List<Node>> findCommunities(TransportationGraph graph) {
        try {
            AffinityMatrix matrix = graph.getAffinityMatrix();
            RealMatrix laplacian = computeNormalizedLaplacian(matrix);
            
            // Add small perturbation for numerical stability
            addNumericalStability(laplacian);
            
            // Compute eigenvectors
            EigenDecomposition eigen = new EigenDecomposition(laplacian);
            RealMatrix eigenvectors = getFirstKEigenvectors(eigen, clusters);
            
            // Normalize rows before clustering
            normalizeRows(eigenvectors);
            
            return clusterEigenVectors(eigenvectors, new ArrayList<>(graph.getPointToNode().values()));
        } catch (Exception e) {
            System.err.println("Warning: Spectral clustering failed with error: " + e.getMessage());
            System.err.println("Falling back to simpler clustering method...");
            return fallbackClustering(graph);
        }
    }

    private void addNumericalStability(RealMatrix matrix) {
        for (int i = 0; i < matrix.getRowDimension(); i++) {
            matrix.setEntry(i, i, matrix.getEntry(i, i) + EPSILON);
        }
    }

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

    private RealMatrix computeNormalizedLaplacian(AffinityMatrix matrix) {
        double[][] affinity = matrix.getMatrix();
        int n = affinity.length;
        RealMatrix W = new Array2DRowRealMatrix(affinity);
        
        // Compute degree matrix
        double[] degrees = new double[n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                degrees[i] += affinity[i][j];
            }
            // Add small value to avoid division by zero
            degrees[i] = Math.max(degrees[i], EPSILON);
        }
        
        // Compute D^(-1/2)
        double[] sqrtInvDegrees = new double[n];
        for (int i = 0; i < n; i++) {
            sqrtInvDegrees[i] = 1.0 / Math.sqrt(degrees[i]);
        }
        
        // Compute normalized Laplacian: I - D^(-1/2) W D^(-1/2)
        RealMatrix I = MatrixUtils.createRealIdentityMatrix(n);
        RealMatrix result = I.copy();
        
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                double val = sqrtInvDegrees[i] * W.getEntry(i, j) * sqrtInvDegrees[j];
                result.setEntry(i, j, i == j ? 1.0 - val : -val);
            }
        }
        
        return result;
    }

    private RealMatrix getFirstKEigenvectors(EigenDecomposition eigen, int k) {
        int n = eigen.getRealEigenvalues().length;
        double[] eigenvalues = eigen.getRealEigenvalues();
        
        // Get indices of k smallest eigenvalues (excluding the first one)
        int[] indices = IntStream.range(0, n)
                               .boxed()
                               .sorted((i, j) -> Double.compare(Math.abs(eigenvalues[i]), Math.abs(eigenvalues[j])))
                               .skip(1) // Skip the first eigenvalue
                               .limit(k)
                               .mapToInt(Integer::intValue)
                               .toArray();
        
        // Extract corresponding eigenvectors
        double[][] selectedEigenvectors = new double[n][k];
        for (int i = 0; i < k; i++) {
            RealVector eigenvector = eigen.getEigenvector(indices[i]);
            for (int j = 0; j < n; j++) {
                selectedEigenvectors[j][i] = eigenvector.getEntry(j);
            }
        }
        
        return new Array2DRowRealMatrix(selectedEigenvectors);
    }

    private List<List<Node>> clusterEigenVectors(RealMatrix eigenvectors, List<Node> nodes) {
        List<DoublePoint> points = new ArrayList<>();
        for (int i = 0; i < eigenvectors.getRowDimension(); i++) {
            points.add(new DoublePoint(eigenvectors.getRow(i)));
        }

        // Perform k-means clustering with multiple attempts
        KMeansPlusPlusClusterer<DoublePoint> clusterer = 
            new KMeansPlusPlusClusterer<>(clusters, maxIterations, new EuclideanDistance());
        List<CentroidCluster<DoublePoint>> clusters = clusterer.cluster(points);

        // Convert clusters back to nodes
        List<List<Node>> communities = new ArrayList<>();
        for (CentroidCluster<DoublePoint> cluster : clusters) {
            List<Node> community = cluster.getPoints().stream()
                .map(point -> nodes.get(points.indexOf(point)))
                .collect(Collectors.toList());
            communities.add(community);
        }

        return communities;
    }

    private List<List<Node>> fallbackClustering(TransportationGraph graph) {
        // Simple fallback method: use degree-based clustering
        List<Node> nodes = new ArrayList<>(graph.getPointToNode().values());
        nodes.sort((a, b) -> Integer.compare(
            graph.getGraph().degreeOf(b),
            graph.getGraph().degreeOf(a)
        ));
        
        List<List<Node>> communities = new ArrayList<>();
        int nodesPerCluster = Math.max(1, nodes.size() / clusters);
        
        for (int i = 0; i < clusters; i++) {
            int start = i * nodesPerCluster;
            int end = Math.min(start + nodesPerCluster, nodes.size());
            if (i == clusters - 1) {
                end = nodes.size(); // Last cluster gets remaining nodes
            }
            communities.add(new ArrayList<>(nodes.subList(start, end)));
        }
        
        return communities;
    }
} 