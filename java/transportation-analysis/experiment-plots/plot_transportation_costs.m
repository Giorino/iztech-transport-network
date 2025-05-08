% Plot transportation costs by graph type and clustering algorithm
clear;
close all;

% Data extracted from the bar chart
graphTypes = {'Delaunay', 'Gabriel', 'KNN'};
algorithms = {'Spectral', 'Leiden', 'MVMGC'};

% Transportation costs for each graph type and algorithm
% [Spectral, Leiden, MVMGC] for each graph type
costs = [
    133.145, 121.154, 144.275;  % Delaunay
    128.645, 110.964, 135.655;  % Gabriel
    122.845, 118.523, 124.112   % KNN
] * 1000; % Scale to match the y-axis values in the image

% Create figure
figure('Position', [100, 100, 1000, 800]);

% Create the grouped bar chart
bar_handle = bar(costs);

% Set colors to match the image
bar_handle(1).FaceColor = [0, 0.4470, 0.7410]; % Blue for Spectral
bar_handle(2).FaceColor = [0.8500, 0.3250, 0.0980]; % Orange for Leiden
bar_handle(3).FaceColor = [0.4660, 0.6740, 0.1880]; % Green for MVMGC

% Add labels and title
title('Total Transportation Cost by Graph Type and Clustering Algorithm', 'FontSize', 24, 'FontWeight', 'bold');
xlabel('Graph Construction Method', 'FontSize', 22);
ylabel('Total Transportation Cost (TL)', 'FontSize', 22);
xticklabels(graphTypes);
set(gca, 'FontSize', 18); % Increase font size for axis tick labels

% Create legend with larger font
legend(algorithms, 'Location', 'northeast', 'FontSize', 18);

% Get positions for the bar labels
[ngroups, nbars] = size(costs);
x = nan(nbars, ngroups);
for i = 1:nbars
    x(i,:) = bar_handle(i).XEndPoints;
end
y = costs';

% Add data labels directly above each bar with fixed offset
textOffset = 800; % Fixed offset for all labels
for i = 1:size(y, 1)
    for j = 1:size(y, 2)
        text(x(i,j), y(i,j) + textOffset, num2str(y(i,j)/1000, '%.3f'), ...
            'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', 'FontSize', 16, 'FontWeight', 'bold');
    end
end

% Set y-axis limits to match the image
ylim([100000, 150000]);
yticks(100000:10000:150000);
yticklabels({'100,000', '110,000', '120,000', '130,000', '140,000', '150,000'});

% Add grid
grid on;

% Adjust overall appearance
set(gcf, 'Color', 'white'); % White background
box on; % Add box around plot

% Save the figure
saveas(gcf, 'transportation_costs_comparison.png');
saveas(gcf, 'transportation_costs_comparison.fig');

fprintf('Plot created and saved as transportation_costs_comparison.png and transportation_costs_comparison.fig\n'); 