% MATLAB Script: plot_problem_isolated_students_graph.m
% Generates a GRAPH figure illustrating the problem of isolated student locations.

clear; clc; close all;

% --- Common Node Coordinate Generation (Reduced Node Count) ---
rng(123); % Set fixed seed for reproducibility

% Main dense cluster
num_nodes_main = 30; 
main_center = [30, 40];
main_std = 7;
nodes_main_xy = main_center + main_std * randn(num_nodes_main, 2);

% Secondary sparser cluster
num_nodes_secondary = 20; 
secondary_center = [70, 65];
secondary_std = 6;
nodes_secondary_xy = secondary_center + secondary_std * randn(num_nodes_secondary, 2);

% General spread (background nodes)
num_nodes_spread = 15; 
spread_center = [50,50];
spread_std = 25;
nodes_spread_xy_candidates = spread_center + spread_std * randn(num_nodes_spread*2,2);
nodes_spread_xy = nodes_spread_xy_candidates(nodes_spread_xy_candidates(:,1)>0 & nodes_spread_xy_candidates(:,1)<100 & nodes_spread_xy_candidates(:,2)>0 & nodes_spread_xy_candidates(:,2)<100,:);
if size(nodes_spread_xy,1) > num_nodes_spread
    nodes_spread_xy = nodes_spread_xy(1:num_nodes_spread,:);
end

% Specific isolated nodes for demonstration (keeping these 3)
isolated_node1_xy = [5, 5];
isolated_node2_xy = [95, 90];
isolated_node3_xy = [15, 85]; 

all_nodes_xy = [nodes_main_xy; nodes_secondary_xy; nodes_spread_xy; isolated_node1_xy; isolated_node2_xy; isolated_node3_xy];
num_total_nodes = size(all_nodes_xy, 1);

% Campus Location (for route reference if needed, not a graph node itself here)
campus_xy = [50, 50]; 

% --- Base Graph Generation ---
connection_radius = 18; 
adj_matrix = zeros(num_total_nodes, num_total_nodes);

for i = 1:num_total_nodes
    for j = i+1:num_total_nodes
        dist = norm(all_nodes_xy(i,:) - all_nodes_xy(j,:));
        if dist <= connection_radius
            is_i_isolated = any(all(bsxfun(@eq, all_nodes_xy(i,:), [isolated_node1_xy; isolated_node2_xy; isolated_node3_xy]),2));
            is_j_isolated = any(all(bsxfun(@eq, all_nodes_xy(j,:), [isolated_node1_xy; isolated_node2_xy; isolated_node3_xy]),2));
            
            if is_i_isolated && is_j_isolated 
                 if dist < connection_radius * 0.7 % Only connect isolated if very close to each other
                    adj_matrix(i,j) = 1; adj_matrix(j,i) = 1;
                 end
            elseif ~(is_i_isolated || is_j_isolated) 
                 adj_matrix(i,j) = 1; adj_matrix(j,i) = 1;
            % Do not connect isolated nodes to main clusters by default radius rule
            % unless they are extremely close (which they are defined not to be)
            end
        end
    end
end

G_base = graph(adj_matrix);

idx_isolated1 = find(all(bsxfun(@eq, all_nodes_xy, isolated_node1_xy),2));
idx_isolated2 = find(all(bsxfun(@eq, all_nodes_xy, isolated_node2_xy),2));
idx_isolated3 = find(all(bsxfun(@eq, all_nodes_xy, isolated_node3_xy),2));
isolated_node_indices = [idx_isolated1; idx_isolated2; idx_isolated3];

% --- Figure Specific Plotting ---
figure('Position', [100, 100, 800, 650]);
h_plot = plot(G_base, 'XData', all_nodes_xy(:,1), 'YData', all_nodes_xy(:,2), ...
    'NodeColor', [0.5 0.7 1], 'EdgeColor', [0.6 0.6 0.6], 'LineWidth', 0.7, 'MarkerSize', 8, 'NodeLabel', {}); % Increased MarkerSize, Ensure no node labels
    
hold on;

% Highlight isolated nodes - make them distinctly larger
highlight(h_plot, isolated_node_indices, 'NodeColor', 'r', 'MarkerSize', 10);

% Removed the conceptual magenta path and its associated text labels.

hold off;
axis([0 100 0 100]);
axis off; % Turn off all axis lines, ticks, and labels
set(gcf, 'Color', 'w'); % Set figure background to white
% title('Problem: Isolated Student Locations (Graph View)', 'FontSize', 14); % Removed caption
% xlabel('X Coordinate'); % Removed
% ylabel('Y Coordinate'); % Removed

% --- Save Figure ---
img_dir = '../img'; 
if ~exist(img_dir, 'dir')
   mkdir(img_dir);
end
saveas(gcf, fullfile(img_dir, 'problem_isolated_students.png'));
disp(sprintf('Graph figure saved as %s', fullfile(img_dir, 'problem_isolated_students.png'))); 