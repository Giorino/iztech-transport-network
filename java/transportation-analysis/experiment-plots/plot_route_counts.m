% Plot number of routes by graph type and clustering algorithm
clear;
close all;

% Data extracted from the bar chart
graphTypes = {'Delaunay', 'Gabriel', 'KNN'};
algorithms = {'Spectral', 'Leiden', 'MVAGC'};

% Number of routes for each graph type and algorithm
% [Spectral, Leiden, MVAGC] for each graph type
routes = [
    73, 66, 76;  % Delaunay
    69, 60, 74;  % Gabriel
    68, 65, 69   % KNN
];

% Create figure
figure('Position', [100, 100, 1000, 800]);

% Create the grouped bar chart
bar_handle = bar(routes);

% Set colors to match the image
bar_handle(1).FaceColor = [0, 0.4470, 0.7410]; % Blue for Spectral
bar_handle(2).FaceColor = [0.8500, 0.3250, 0.0980]; % Orange for Leiden
bar_handle(3).FaceColor = [0.4660, 0.6740, 0.1880]; % Green for MVAGC

% Add labels and title
title('Number of Routes by Graph Type and Clustering Algorithm', 'FontSize', 24, 'FontWeight', 'bold');
xlabel('Graph Construction Method', 'FontSize', 22);
ylabel('Number of Routes (Routes)', 'FontSize', 22);
xticklabels(graphTypes);
set(gca, 'FontSize', 18); % Increase font size for axis tick labels

% Create legend with larger font
legend(algorithms, 'Location', 'northeast', 'FontSize', 18);

% Get positions for the bar labels
[ngroups, nbars] = size(routes);
x = nan(nbars, ngroups);
for i = 1:nbars
    x(i,:) = bar_handle(i).XEndPoints;
end
y = routes';

% Add data labels directly above each bar with reduced offset
textOffset = 0.3; % Reduced offset to bring labels closer to bars
for i = 1:size(y, 1)
    for j = 1:size(y, 2)
        text(x(i,j), y(i,j) + textOffset, num2str(y(i,j)), ...
            'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', 'FontSize', 16, 'FontWeight', 'bold');
    end
end

% Set y-axis limits to match the image
ylim([55, 80]);
yticks(55:5:80);

% Add grid
grid on;

% Adjust overall appearance
set(gcf, 'Color', 'white'); % White background
box on; % Add box around plot

% Save the figure
saveas(gcf, 'route_counts_comparison.png');
saveas(gcf, 'route_counts_comparison.fig');

fprintf('Plot created and saved as route_counts_comparison.png and route_counts_comparison.fig\n'); 