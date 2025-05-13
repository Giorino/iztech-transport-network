% MATLAB Script: plot_problem_large_scale_graph.m
% Generates a GRAPH figure illustrating the problem of a large number of students.

clear; clc; close all;

% --- Node Coordinate Generation (Many Nodes) ---
rng(123); % Set fixed seed for reproducibility

num_total_nodes_visual = 2000; % Reduced from 1800, but still large for graph viz

centers = [[20,30]; [70,20]; [30,70]; [70,80]; [50,50]; [40,50]; [60,40]];
stds =    [6; 7; 5; 6; 8; 7; 6];
nums_per_cluster = round(num_total_nodes_visual / size(centers,1));

all_nodes_xy = [];
for i = 1:size(centers,1)
    cluster_nodes = centers(i,:) + stds(i) * randn(nums_per_cluster, 2);
    all_nodes_xy = [all_nodes_xy; cluster_nodes];
end

% Ensure we are close to the target number
if size(all_nodes_xy,1) > num_total_nodes_visual
    all_nodes_xy = all_nodes_xy(1:num_total_nodes_visual,:);
elseif size(all_nodes_xy,1) < num_total_nodes_visual
    needed = num_total_nodes_visual - size(all_nodes_xy,1);
    extra_nodes = centers(1,:) + stds(1) * randn(needed,2);
    all_nodes_xy = [all_nodes_xy; extra_nodes];
end

all_nodes_xy(all_nodes_xy < 0) = 0;
all_nodes_xy(all_nodes_xy > 100) = 100;
num_total_nodes = size(all_nodes_xy,1);

% --- Base Graph Generation (Dense to show complexity) ---
connection_radius = 10; % Smaller radius for denser look with many nodes
adj_matrix = zeros(num_total_nodes, num_total_nodes);
for i = 1:num_total_nodes
    for j = i+1:num_total_nodes % Avoid self-loops and duplicate edges
        dist = norm(all_nodes_xy(i,:) - all_nodes_xy(j,:));
        if dist <= connection_radius
            adj_matrix(i,j) = 1;
            adj_matrix(j,i) = 1; % Undirected graph
        end
    end
end
G_dense = graph(adj_matrix);

% --- Figure Specific Plotting ---
figure('Position', [100, 100, 700, 650]);

% Plot the dense graph
% Smaller markers and thinner lines for density
plot(G_dense, 'XData', all_nodes_xy(:,1), 'YData', all_nodes_xy(:,2), ...
    'NodeColor', [0.2 0.4 0.8], 'EdgeColor', [0.6 0.7 0.9], ...
    'MarkerSize', 3, 'LineWidth', 0.1, 'NodeLabel', {}); 
hold on;

%annotation_str = sprintf('Approx. %d Nodes: High Computational Complexity', num_total_nodes);
%text(50, 3, annotation_str, 'HorizontalAlignment', 'center', 'FontSize', 11, ...
%     'FontWeight', 'bold', 'BackgroundColor', [1 1 0.9], 'EdgeColor', 'k');

hold off;
axis([0 100 0 100]);
axis off; % Turn off all axis lines, ticks, and labels
set(gcf, 'Color', 'w');
% title('Problem: Large Number of Nodes (Graph View)', 'FontSize', 14); % Removed caption

% --- Save Figure ---
img_dir = '../img';
if ~exist(img_dir, 'dir')
   mkdir(img_dir);
end
saveas(gcf, fullfile(img_dir, 'problem_large_scale.png'));
disp(sprintf('Graph figure saved as %s', fullfile(img_dir, 'problem_large_scale.png'))); 