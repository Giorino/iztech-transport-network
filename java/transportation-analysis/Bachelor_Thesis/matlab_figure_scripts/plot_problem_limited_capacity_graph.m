% MATLAB Script: plot_problem_limited_capacity_graph.m
% Generates a GRAPH figure illustrating the problem of limited service capacity.

clear; clc; close all;

% --- Common Node Coordinate Generation (Consistent with previous graph script, Reduced Node Count) ---
rng(123); % Set fixed seed for reproducibility

% Main dense cluster
num_nodes_main = 30; % Reduced
main_center = [30, 40];
main_std = 7;
nodes_main_xy = main_center + main_std * randn(num_nodes_main, 2);

% Secondary sparser cluster
num_nodes_secondary = 20; % Reduced
secondary_center = [70, 65];
secondary_std = 6;
nodes_secondary_xy = secondary_center + secondary_std * randn(num_nodes_secondary, 2);

% General spread (background nodes)
num_nodes_spread = 15; % Reduced
spread_center = [50,50];
spread_std = 25;
nodes_spread_xy_candidates = spread_center + spread_std * randn(num_nodes_spread*2,2);
nodes_spread_xy = nodes_spread_xy_candidates(nodes_spread_xy_candidates(:,1)>0 & nodes_spread_xy_candidates(:,1)<100 & nodes_spread_xy_candidates(:,2)>0 & nodes_spread_xy_candidates(:,2)<100,:);
if size(nodes_spread_xy,1) > num_nodes_spread
    nodes_spread_xy = nodes_spread_xy(1:num_nodes_spread,:);
end

% For this plot, we will form one large graph, then select subgraphs visually.
% We won't use the specific 'isolated' points from the previous script here,
% but the base distribution helps define varied density.
all_nodes_xy = [nodes_main_xy; nodes_secondary_xy; nodes_spread_xy];
num_total_nodes = size(all_nodes_xy, 1);

% --- Base Graph Generation ---
connection_radius = 19; % Adjusted slightly for visual clustering
adj_matrix = zeros(num_total_nodes, num_total_nodes);
for i = 1:num_total_nodes
    for j = i+1:num_total_nodes
        dist = norm(all_nodes_xy(i,:) - all_nodes_xy(j,:));
        if dist <= connection_radius
            adj_matrix(i,j) = 1;
            adj_matrix(j,i) = 1;
        end
    end
end
G_base = graph(adj_matrix);

% --- Figure Specific Plotting ---
figure('Position', [100, 100, 950, 600]); % Adjusted height slightly
h_plot = plot(G_base, 'XData', all_nodes_xy(:,1), 'YData', all_nodes_xy(:,2), ...
    'NodeColor', [0.8 0.8 0.8], 'EdgeColor', [0.9 0.9 0.9], 'LineWidth', 0.7, 'MarkerSize', 7, 'NodeLabel', {}); % Base nodes larger, no labels
    
hold on;

% Define three hypothetical visual clusters by selecting node indices
% Cluster 1: Too Few (e.g., 5 nodes)
% Pick some nodes that are relatively close to each other but few in number
% For simplicity, pick nodes from a sparser region of nodes_spread_xy or an edge of a cluster
idx_too_few = find(all_nodes_xy(:,1) < 20 & all_nodes_xy(:,2) < 35); 
if length(idx_too_few) > 4, idx_too_few = idx_too_few(randperm(length(idx_too_few), 4)); 
elseif isempty(idx_too_few) && num_total_nodes >=4, idx_too_few = datasample(1:num_total_nodes, 4, 'Replace', false); end 
num_too_few = length(idx_too_few);

% Cluster 2: Too Many (e.g., 60 nodes)
% Pick a large portion of the main dense cluster
idx_too_many = find(all_nodes_xy(:,1) > 20 & all_nodes_xy(:,1) < 50 & all_nodes_xy(:,2) > 25 & all_nodes_xy(:,2) < 55);
if length(idx_too_many) > 25, idx_too_many = idx_too_many(randperm(length(idx_too_many),25));
elseif isempty(idx_too_many) && num_total_nodes >= 25, idx_too_many = datasample(1:num_total_nodes, 25, 'Replace', false);end
num_too_many = length(idx_too_many);

% Cluster 3: Optimal Size (e.g., 30 nodes)
% Pick from the secondary cluster or a well-formed part of main
idx_optimal = find(all_nodes_xy(:,1) > 60 & all_nodes_xy(:,1) < 85 & all_nodes_xy(:,2) > 55 & all_nodes_xy(:,2) < 80);
if length(idx_optimal) > 15, idx_optimal = idx_optimal(randperm(length(idx_optimal),15)); 
elseif isempty(idx_optimal) && num_total_nodes >=15, idx_optimal = datasample(1:num_total_nodes, 15, 'Replace', false); end 
num_optimal = length(idx_optimal);

idx_too_many = setdiff(idx_too_many, idx_too_few, 'stable');
% Ensure idx_optimal, idx_too_few, and idx_too_many are column vectors for robust concatenation
idx_optimal = setdiff(idx_optimal(:), [idx_too_few(:); idx_too_many(:)], 'stable');
num_too_many = length(idx_too_many);
num_optimal = length(idx_optimal);

node_marker_size_highlight = 9; % Marker size for highlighted cluster nodes

if ~isempty(idx_too_few)
    highlight(h_plot, idx_too_few, 'NodeColor', 'r', 'MarkerSize', node_marker_size_highlight);
    edges_to_highlight_few = find(ismember(G_base.Edges.EndNodes(:,1),idx_too_few) & ismember(G_base.Edges.EndNodes(:,2),idx_too_few));
    if ~isempty(edges_to_highlight_few), highlight(h_plot, 'Edges', edges_to_highlight_few, 'EdgeColor', 'r', 'LineWidth', 2); end
    % Text annotation removed from near cluster, will use legend
end

if ~isempty(idx_too_many)
    highlight(h_plot, idx_too_many, 'NodeColor', 'b', 'MarkerSize', node_marker_size_highlight);
    edges_to_highlight_many = find(ismember(G_base.Edges.EndNodes(:,1),idx_too_many) & ismember(G_base.Edges.EndNodes(:,2),idx_too_many));
    if ~isempty(edges_to_highlight_many), highlight(h_plot, 'Edges', edges_to_highlight_many, 'EdgeColor', 'b', 'LineWidth', 2); end
    % Text annotation removed
end

if ~isempty(idx_optimal)
    highlight(h_plot, idx_optimal, 'NodeColor', [0 .6 0], 'MarkerSize', node_marker_size_highlight); 
    edges_to_highlight_opt = find(ismember(G_base.Edges.EndNodes(:,1),idx_optimal) & ismember(G_base.Edges.EndNodes(:,2),idx_optimal));
    if ~isempty(edges_to_highlight_opt), highlight(h_plot, 'Edges', edges_to_highlight_opt, 'EdgeColor', [0 .6 0], 'LineWidth', 2); end
    % Text annotation removed
end

% Add a general text box for context if needed, or rely on caption
% text(5, 95, 'Bus Capacity: 10-50 Nodes', 'FontSize', 10, 'FontWeight', 'bold', 'BackgroundColor', [1 1 0.8]);

hold off;
axis([0 100 0 100]);
axis off; % Turn off all axis lines, ticks, and labels
set(gcf, 'Color', 'w'); 
% title('Problem: Limited Service Capacity (Graph View)', 'FontSize', 14); % Removed caption

% Add a legend to explain cluster colors
% Need to plot dummy points for legend handles as graph highlight doesn't directly support it well
hold on;
    h_leg_too_few = plot(NaN,NaN,'ro','MarkerFaceColor','r', 'MarkerSize',node_marker_size_highlight, 'LineWidth',2, 'DisplayName',sprintf('Too Few (N=%d)', num_too_few));
    h_leg_too_many = plot(NaN,NaN,'bo','MarkerFaceColor','b', 'MarkerSize',node_marker_size_highlight, 'LineWidth',2, 'DisplayName',sprintf('Too Many (N=%d)', num_too_many));
    h_leg_optimal = plot(NaN,NaN,'o','MarkerEdgeColor',[0 .6 0],'MarkerFaceColor',[0 .6 0], 'MarkerSize',node_marker_size_highlight, 'LineWidth',2, 'DisplayName',sprintf('Optimal (N=%d)', num_optimal));
    h_leg_base = plot(NaN,NaN,'o','MarkerEdgeColor',[0.8 0.8 0.8],'MarkerFaceColor',[0.8 0.8 0.8], 'MarkerSize',7, 'DisplayName','Other Nodes');
hold off;
legend([h_leg_too_few, h_leg_too_many, h_leg_optimal, h_leg_base], 'Location', 'northeastoutside', 'FontSize', 10);

% --- Save Figure ---
img_dir = '../img';
if ~exist(img_dir, 'dir')
   mkdir(img_dir);
end
saveas(gcf, fullfile(img_dir, 'problem_limited_capacity.png'));
disp(sprintf('Graph figure saved as %s', fullfile(img_dir, 'problem_limited_capacity.png'))); 