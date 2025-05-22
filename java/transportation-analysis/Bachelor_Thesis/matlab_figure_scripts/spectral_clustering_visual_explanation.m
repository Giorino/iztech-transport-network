%% SPECTRAL CLUSTERING VISUAL EXPLANATION
% This script provides a visual explanation of spectral clustering
% with clear, presentation-ready figures for each step of the process.

clear all; close all; clc;
set(0,'defaultTextInterpreter','latex');
set(0,'defaultAxesFontSize',14);
set(0,'defaultLegendInterpreter','latex');
set(0,'defaultAxesTickLabelInterpreter','latex');

%% 1. CREATE A GRAPH WITH CLEAR CLUSTERS
% We'll create data with 3 distinct clusters in 2D space

% Create clusters with different centers and some noise
rng(42); % For reproducibility
n_per_cluster = 20;
n_clusters = 3;
n = n_per_cluster * n_clusters;

% Cluster centers
centers = [1 1; 5 5; 1 5];

% Generate points
X = zeros(n, 2);
true_labels = zeros(n, 1);

for i = 1:n_clusters
    idx = (i-1)*n_per_cluster+1 : i*n_per_cluster;
    X(idx, :) = centers(i,:) + 0.5*randn(n_per_cluster, 2);
    true_labels(idx) = i;
end

% Plot the original data points - using built-in plotting instead of gscatter
figure('Position', [100, 500, 500, 400], 'Name', 'Original Data');
cluster_markers = {'bo', 'ro', 'go'};  % Blue, red, green circles
hold on;
for i = 1:n_clusters
    cluster_idx = (true_labels == i);
    plot(X(cluster_idx,1), X(cluster_idx,2), cluster_markers{i}, 'MarkerSize', 8, 'MarkerFaceColor', cluster_markers{i}(1));
end
hold off;
title('\textbf{Original Data Points}', 'FontSize', 16);
xlabel('$x_1$'); ylabel('$x_2$');
grid on;
legend('Cluster 1', 'Cluster 2', 'Cluster 3', 'Location', 'best');
axis equal;

% Define RGB values for later use
rgb_colors = [0 0 1; 1 0 0; 0 0.8 0];  % Blue, Red, Green

%% 2. BUILD THE SIMILARITY GRAPH
% We construct a similarity graph using a Gaussian kernel

% Compute pairwise distances
D = squareform(pdist(X));

% Convert distances to similarities using Gaussian kernel
sigma = 1; % Bandwidth parameter
S = exp(-D.^2 / (2*sigma^2));

% Threshold small similarities (optional)
S = S .* (S > 0.01);

% Plot the similarity matrix
figure('Position', [600, 500, 500, 400], 'Name', 'Similarity Matrix');
subplot(1,2,1);
imagesc(S);
title('\textbf{Similarity Matrix}', 'FontSize', 16);
axis square;
colorbar;
colormap(cool); % Changed to a blueish colormap

% Visualize the graph
subplot(1,2,2);
G = graph(S, 'omitselfloops');
% Use edge weight to determine line width
LWidths = 5*G.Edges.Weight/max(G.Edges.Weight);
% Plot with fixed alpha (transparency) and variable line width, blue nodes, and no labels
p = plot(G, 'XData', X(:,1), 'YData', X(:,2), 'NodeColor', 'b', 'MarkerSize', 6, 'LineWidth', LWidths, 'EdgeAlpha', 0.5, 'NodeLabel', {});
title('\textbf{Similarity Graph}', 'FontSize', 16);
axis equal;
% Removed grid from this subplot

%% 3. COMPUTE THE DEGREE MATRIX
% The degree matrix is diagonal with node degrees

% Compute node degrees
degrees = sum(S, 2);

% Create degree matrix
D = diag(degrees);

% Visualize the degree matrix
figure('Position', [100, 100, 800, 400], 'Name', 'Degree Matrix');
subplot(1,2,1);
imagesc(D);
title('\textbf{Degree Matrix $\mathbf{D}$}', 'FontSize', 16);
axis square;
colorbar;
colormap(cool); % Changed to a blueish colormap

% Visualize the graph with node sizes proportional to degrees
subplot(1,2,2);
% Plot with fixed alpha and variable line width
p = plot(G, 'XData', X(:,1), 'YData', X(:,2), 'LineWidth', LWidths, 'EdgeAlpha', 0.5);
p.NodeCData = degrees;
p.MarkerSize = 4 + 10*degrees/max(degrees);
colorbar;
colormap(gca, 'cool'); % Already blueish
title('\textbf{Graph with Node Degrees}', 'FontSize', 16);
axis equal;
grid on;

%% 4. COMPUTE THE LAPLACIAN MATRIX
% The Laplacian matrix is L = D - S

% Compute unnormalized Laplacian
L = D - S;

% Compute normalized Laplacian (often used in spectral clustering)
D_sqrt_inv = diag(1./sqrt(degrees));
L_norm = eye(n) - D_sqrt_inv * S * D_sqrt_inv;

% Visualize the Laplacian matrices
figure('Position', [600, 100, 900, 400], 'Name', 'Laplacian Matrices');
subplot(1,2,1);
imagesc(L);
title('\textbf{Unnormalized Laplacian $\mathbf{L = D - S}$}', 'FontSize', 16);
axis square;
colorbar;
colormap(cool); % Changed to a blueish colormap

subplot(1,2,2);
imagesc(L_norm);
title('\textbf{Normalized Laplacian $\mathbf{L_{norm} = I - D^{-1/2}SD^{-1/2}}$}', 'FontSize', 16);
axis square;
colorbar;
colormap(cool); % Changed to a blueish colormap

%% 5. EIGENDECOMPOSITION OF THE LAPLACIAN
% The eigenvectors of the Laplacian contain clustering information

% Compute eigendecomposition (use normalized Laplacian for better results)
[V, E] = eig(L_norm);
eigenvalues = diag(E);

% Sort eigenvalues and eigenvectors
[eigenvalues, idx] = sort(eigenvalues);
V = V(:, idx);

% Plot the eigenvalues
figure('Position', [100, 500, 900, 400], 'Name', 'Eigenvalues');
subplot(1,2,1);
bar(eigenvalues);
title('\textbf{Sorted Eigenvalues of Laplacian}', 'FontSize', 16);
xlabel('Index');
ylabel('Eigenvalue');
grid on;

subplot(1,2,2);
plot(eigenvalues, 'o-', 'LineWidth', 2, 'MarkerSize', 8, 'MarkerFaceColor', 'b');
title('\textbf{Eigenvalue Gap Shows Number of Clusters}', 'FontSize', 16);
xlabel('Index');
ylabel('Eigenvalue');
grid on;
hold on;
plot([n_clusters, n_clusters], [0, eigenvalues(n_clusters+1)], 'r--', 'LineWidth', 2);
text(n_clusters+0.5, eigenvalues(n_clusters+1)/2, 'Spectral Gap', 'Color', 'r', 'FontSize', 14);

%% 6. VISUALIZE THE EIGENVECTORS
% Eigenvectors corresponding to the smallest eigenvalues provide the embedding

% Select k eigenvectors for k clusters
k = n_clusters;
embedding = V(:, 1:k);  % Use first k eigenvectors

% Visualize the first few eigenvectors
figure('Position', [600, 500, 900, 400], 'Name', 'Eigenvectors');
colors_by_true_label = zeros(n, 3);
for i = 1:n
    colors_by_true_label(i, :) = rgb_colors(true_labels(i), :);
end

for i = 1:min(4, k+1)
    subplot(1, min(4, k+1), i);
    scatter(X(:,1), X(:,2), 50, V(:,i), 'filled');
    colormap(gca, 'cool'); % Changed to a blueish colormap
    colorbar;
    title(sprintf('\\textbf{Eigenvector %d}', i));
    axis equal;
    grid on;
end

%% 7. SPECTRAL EMBEDDING
% The eigenvectors form a low-dimensional embedding

figure('Position', [100, 100, 900, 450], 'Name', 'Spectral Embedding');

% For 3 clusters, we can visualize the 3D embedding
if k >= 3
    subplot(1,2,1);
    scatter3(V(:,1), V(:,2), V(:,3), 70, colors_by_true_label, 'filled', 'MarkerEdgeColor', 'k');
    title('\textbf{Spectral Embedding (First 3 Eigenvectors)}', 'FontSize', 16);
    xlabel('Eigenvector 1');
    ylabel('Eigenvector 2');
    zlabel('Eigenvector 3');
    grid on;
    view([-45, 30]);
end

% 2D embedding using the first two eigenvectors
subplot(1,2,2);
scatter(V(:,1), V(:,2), 70, colors_by_true_label, 'filled', 'MarkerEdgeColor', 'k');
title('\textbf{Spectral Embedding (First 2 Eigenvectors)}', 'FontSize', 16);
xlabel('Eigenvector 1');
ylabel('Eigenvector 2');
grid on;
axis equal;

%% 8. CLUSTERING USING K-MEANS ON THE SPECTRAL EMBEDDING
% Apply k-means on the eigenvector embedding

% Apply k-means to the embedding
spectral_labels = kmeans(embedding, k, 'Replicates', 10);

% Visualize the clustering results
figure('Position', [600, 100, 900, 400], 'Name', 'Clustering Results');

% Original points with spectral clustering labels
subplot(1,2,1);
hold on;
for i = 1:n_clusters
    cluster_idx = (spectral_labels == i);
    plot(X(cluster_idx,1), X(cluster_idx,2), cluster_markers{i}, 'MarkerSize', 8, 'MarkerFaceColor', cluster_markers{i}(1));
end
hold off;
title('\textbf{Spectral Clustering Result}', 'FontSize', 16);
xlabel('$x_1$'); ylabel('$x_2$');
grid on;
axis equal;

% Compare with ground truth
subplot(1,2,2);
hold on;
for i = 1:n_clusters
    cluster_idx = (true_labels == i);
    plot(X(cluster_idx,1), X(cluster_idx,2), cluster_markers{i}, 'MarkerSize', 8, 'MarkerFaceColor', cluster_markers{i}(1));
end
hold off;
title('\textbf{Ground Truth Clusters}', 'FontSize', 16);
xlabel('$x_1$'); ylabel('$x_2$');
grid on;
axis equal;

%% 9. STEP-BY-STEP VISUAL SUMMARY
% Create a summary figure showing the entire process

figure('Position', [100, 100, 1200, 500], 'Name', 'Spectral Clustering Process');

% 1. Start with data
subplot(1,4,1);
hold on;
for i = 1:n_clusters
    cluster_idx = (true_labels == i);
    plot(X(cluster_idx,1), X(cluster_idx,2), cluster_markers{i}, 'MarkerSize', 8, 'MarkerFaceColor', cluster_markers{i}(1));
end
hold off;
title('\textbf{1. Input Data}', 'FontSize', 14);
axis equal; grid on;

% 2. Construct similarity graph and Laplacian
subplot(1,4,2);
imagesc(L_norm);
title('\textbf{2. Laplacian Matrix}', 'FontSize', 14);
axis square; colorbar;
colormap(cool); % Changed to a blueish colormap

% 3. Extract eigenvectors
subplot(1,4,3);
scatter(V(:,1), V(:,2), 70, colors_by_true_label, 'filled');
title('\textbf{3. Spectral Embedding}', 'FontSize', 14);
grid on; axis equal;

% 4. Apply clustering
subplot(1,4,4);
hold on;
for i = 1:n_clusters
    cluster_idx = (spectral_labels == i);
    plot(X(cluster_idx,1), X(cluster_idx,2), cluster_markers{i}, 'MarkerSize', 8, 'MarkerFaceColor', cluster_markers{i}(1));
end
hold off;
title('\textbf{4. Final Clustering}', 'FontSize', 14);
axis equal; grid on;

%% SAVE FIGURES FOR PRESENTATION
% Uncomment this section to save all figures
% figHandles = findall(0, 'Type', 'figure');
% for i = 1:length(figHandles)
%     fig = figHandles(i);
%     figName = fig.Name;
%     saveas(fig, ['spectral_clustering_' strrep(figName, ' ', '_') '.png']);
% end

fprintf('Spectral clustering visualization complete!\n');
fprintf('You can use these figures in your presentation to explain the process.\n'); 