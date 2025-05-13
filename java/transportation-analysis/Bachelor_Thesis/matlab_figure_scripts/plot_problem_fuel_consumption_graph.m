% MATLAB Script: plot_problem_fuel_consumption_graph.m
% Generates a GRAPH figure illustrating fuel consumption issues.

clear; clc; close all;

% --- Common Node Coordinate Generation (Reduced Node Count) ---
rng(123); % Set fixed seed for reproducibility

% Cluster A for Panel 1
num_nodes_c1 = 10; 
center_c1 = [25, 30];
std_c1 = 6;
nodes_c1_xy = center_c1 + std_c1 * randn(num_nodes_c1, 2);

% Cluster B for Panel 1
num_nodes_c2 = 8;
center_c2 = [70, 60];
std_c2 = 5;
nodes_c2_xy = center_c2 + std_c2 * randn(num_nodes_c2, 2);

% All nodes for this specific visualization
all_nodes_xy = [nodes_c1_xy; nodes_c2_xy];
num_total_nodes = size(all_nodes_xy, 1);

% Campus Node (conceptually, not part of the base graph G_base for this viz)
campus_xy = [50, 85];

% --- Base Graph Generation (Minimal, just to show nodes initially) ---
% For this plot, the routes are more important than dense underlying graph structure.
% We can create a graph with no edges initially or a very sparse one.
G_base = graph(); % Create an empty graph
G_base = addnode(G_base, num_total_nodes); % Add nodes

% --- Figure Specific Plotting ---
figure('Position', [100, 100, 1100, 550]);
node_marker_size = 8;

% Panel A: Inefficient Routes
subplot(1,2,1);
h_plot_A = plot(G_base, 'XData', all_nodes_xy(:,1), 'YData', all_nodes_xy(:,2), ...
    'NodeColor', [0.5 0.7 1], 'EdgeColor', 'none', 'MarkerSize', node_marker_size, 'NodeLabel', {}); % No base edges shown explicitly
hold on;
plot(campus_xy(1), campus_xy(2), 'ks', 'MarkerFaceColor', 'g', 'MarkerSize', 10, 'LineWidth', 1.5); % Campus marker

% Inefficient Route 1 (Conceptual path)
% Connects first 5 nodes of C1 then to campus, inefficiently
inefficient_path1_indices = [1:5, 1]; % Path nodes from C1, closing loop for visual path
path1_coords_x = [all_nodes_xy(inefficient_path1_indices(1:5),1); campus_xy(1)];
path1_coords_y = [all_nodes_xy(inefficient_path1_indices(1:5),2); campus_xy(2)];
% Create a somewhat winding path by adding intermediate points
plot_inefficient_route(path1_coords_x, path1_coords_y, 'r--');

% Inefficient Route 2 (Conceptual path)
% Connects first 4 nodes of C2 then to campus, inefficiently
inefficient_path2_indices = [num_nodes_c1+1 : num_nodes_c1+4, num_nodes_c1+1];
path2_coords_x = [all_nodes_xy(inefficient_path2_indices(1:4),1); campus_xy(1)];
path2_coords_y = [all_nodes_xy(inefficient_path2_indices(1:4),2); campus_xy(2)];
plot_inefficient_route(path2_coords_x, path2_coords_y, 'm--');

axis([0 100 0 100]);
axis off; % Turn off all axis lines, ticks, and labels for subplot 1
set(gcf, 'Color', 'w'); % Ensure figure background is white (can be here or later)
% title('(a) Inefficient Routes / High Cost', 'FontSize', 12); % Removed caption for subplot 1

% Panel B: Optimized Routes
subplot(1,2,2);
h_plot_B = plot(G_base, 'XData', all_nodes_xy(:,1), 'YData', all_nodes_xy(:,2), ...
    'NodeColor', [0.5 0.7 1], 'EdgeColor', 'none', 'MarkerSize', node_marker_size, 'NodeLabel', {});
hold on;
plot(campus_xy(1), campus_xy(2), 'ks', 'MarkerFaceColor', 'g', 'MarkerSize', 10, 'LineWidth', 1.5);

% Optimized Route 1 (More direct for C1)
% Simple sorted order for visual directness
[~, c1_order] = sort(all_nodes_xy(1:num_nodes_c1,1)); % Sort by x-coord for simple path
opt_path1_coords_x = [all_nodes_xy(c1_order,1); campus_xy(1)];
opt_path1_coords_y = [all_nodes_xy(c1_order,2); campus_xy(2)];
plot(opt_path1_coords_x, opt_path1_coords_y, 'g-', 'LineWidth', 1.8);

% Optimized Route 2 (More direct for C2)
[~, c2_order] = sort(all_nodes_xy(num_nodes_c1+1:end,1));
c2_indices_in_all = num_nodes_c1 + c2_order;
opt_path2_coords_x = [all_nodes_xy(c2_indices_in_all,1); campus_xy(1)];
opt_path2_coords_y = [all_nodes_xy(c2_indices_in_all,2); campus_xy(2)];
plot(opt_path2_coords_x, opt_path2_coords_y, 'c-', 'LineWidth', 1.8);

axis([0 100 0 100]);
axis off; % Turn off all axis lines, ticks, and labels for subplot 2
% title('(b) Optimized Routes / Lower Cost', 'FontSize', 12); % Removed caption for subplot 2

% Add a single legend for the figure
lgd_ax = axes('Position',[0.4 0.01 0.2 0.05],'Visible','off'); % Hidden axes for legend
p1 = plot(lgd_ax, NaN,NaN,'r--','LineWidth',1.5, 'DisplayName','Inefficient Route'); hold on;
p2 = plot(lgd_ax, NaN,NaN,'g-','LineWidth',1.8, 'DisplayName','Optimized Route');
p3 = plot(lgd_ax, NaN,NaN,'o','MarkerFaceColor',[0.5 0.7 1],'MarkerEdgeColor',[0.5 0.7 1], 'MarkerSize',node_marker_size, 'DisplayName','Student Node');
p4 = plot(lgd_ax, NaN,NaN,'ks','MarkerFaceColor','g','MarkerSize',10, 'DisplayName','Campus');
legend([p1,p2,p3,p4], 'Location', 'North', 'Orientation', 'horizontal', 'FontSize', 10);

% --- Save Figure ---
img_dir = '../img';
if ~exist(img_dir, 'dir')
   mkdir(img_dir);
end
saveas(gcf, fullfile(img_dir, 'problem_fuel_consumption.png'));
disp(sprintf('Graph figure saved as %s', fullfile(img_dir, 'problem_fuel_consumption.png')));

% Helper function to plot a slightly winding inefficient route
function plot_inefficient_route(x_coords, y_coords, style)
    if length(x_coords) < 2
        plot(x_coords, y_coords, style, 'LineWidth', 1.5);
        return;
    end
    plot_x = [];
    plot_y = [];
    for k = 1:length(x_coords)-1
        plot_x = [plot_x, x_coords(k)];
        plot_y = [plot_y, y_coords(k)];
        % Add a slight deviation for winding effect
        if k < length(x_coords)-1 % Don't deviate last segment to campus too much
            mid_x = (x_coords(k) + x_coords(k+1))/2 + randi([-7,7]);
            mid_y = (y_coords(k) + y_coords(k+1))/2 + randi([-7,7]);
            plot_x = [plot_x, mid_x];
            plot_y = [plot_y, mid_y];
        end
    end
    plot_x = [plot_x, x_coords(end)];
    plot_y = [plot_y, y_coords(end)];
    plot(plot_x, plot_y, style, 'LineWidth', 1.5);
end 