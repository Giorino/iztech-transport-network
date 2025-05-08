% Plot the impact of outlier detection on transportation costs
% This script compares costs with and without outlier detection side by side
% for each algorithm and graph type combination

clear;
close all;

% Define the clustering algorithms and graph types
clusteringAlgorithms = {'Leiden', 'MVAGC', 'Spectral'};
graphTypes = {'Delaunay', 'Gabriel', 'K-Nearest Neighbors'};

% Total costs with outliers (from 2025-05-01 files)
% [Delaunay, Gabriel, KNN] for each algorithm
costsWithOutliers = [
    121145.10, 110364.09, 118522.79;  % Leiden
    143428.94, 135654.57, 124112.47;  % MVAGC 
    133245.21, 127944.52, 121944.72   % Spectral
];

% Total costs without outliers (from 2025-05-04 files)
% [Delaunay, Gabriel, KNN] for each algorithm
costsWithoutOutliers = [
    124274.31, 115095.07, 123075.14;  % Leiden
    151810.10, 130155.86, 124970.64;  % MVAGC
    128448.02, 123648.12, 122696.43   % Spectral
];

% Calculate the percentage difference
percentageDiff = (costsWithoutOutliers - costsWithOutliers) ./ costsWithOutliers * 100;

% Create a figure
figure('Position', [100, 100, 1400, 800]);

% Prepare data for side-by-side comparison
% We'll create 9 groups (3 algorithms × 3 graph types)
groupLabels = cell(1, 9);
groupIdx = 1;

% Create group labels for the x-axis
for i = 1:length(clusteringAlgorithms)
    for j = 1:length(graphTypes)
        groupLabels{groupIdx} = [clusteringAlgorithms{i} '-' graphTypes{j}];
        groupIdx = groupIdx + 1;
    end
end

% Reshape data for grouped bar chart
data = zeros(2, 9); % 2 rows: with/without outliers, 9 columns: algorithm-graph combinations
dataIdx = 1;

for i = 1:length(clusteringAlgorithms)
    for j = 1:length(graphTypes)
        data(1, dataIdx) = costsWithOutliers(i, j);      % With outliers
        data(2, dataIdx) = costsWithoutOutliers(i, j);   % Without outliers
        dataIdx = dataIdx + 1;
    end
end

% Create the grouped bar chart
b = bar(data');

% Set colors explicitly for each bar series
set(b(1), 'FaceColor', [0.2, 0.6, 0.8], 'DisplayName', 'With Outliers');  % Blue for with outliers
set(b(2), 'FaceColor', [0.8, 0.4, 0.2], 'DisplayName', 'Without Outliers');  % Orange for without outliers

% Add title and labels
title('Impact of Outlier Detection on Transportation Costs', 'FontSize', 24, 'FontWeight', 'bold');
ylabel('Total Transportation Cost (TL)', 'FontSize', 22);

% Increase font size for x-tick labels (algorithm-graph combinations)
set(gca, 'XTick', 1:9, 'XTickLabel', groupLabels, 'FontSize', 20, 'FontWeight', 'bold');
xtickangle(45);  % Angle the labels for better readability

% Add legend with only the bar series
legend(b, 'FontSize', 18, 'Location', 'best');

% Add data labels and percentage differences
[ngroups, nbars] = size(data');
x = zeros(nbars, ngroups);

for i = 1:nbars
    x(i,:) = b(i).XEndPoints;
end

y = data';

% Add labels for the bar heights
for i = 1:size(y, 1)
    for j = 1:size(y, 2)
        text(x(j,i), y(i,j) + 2000, num2str(y(i,j), '%.0f'), ...
            'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', 'FontSize', 12, 'FontWeight', 'bold');
    end
end

% Add percentage difference labels between bars with larger font and background boxes
for i = 1:size(y, 1)
    diffPercent = (y(i,2) - y(i,1)) / y(i,1) * 100;
    
    % Position the text between the bars
    xPos = mean([x(1,i), x(2,i)]);
    yPos = min(y(i,1), y(i,2)) - 5000;
    
    % Format based on whether it's an increase or decrease
    % REVERSED LOGIC: cost increase (positive diffPercent) shown as negative impact 
    if diffPercent > 0
        diffText = sprintf('-%.1f%%', diffPercent);
        textColor = [0, 0.5, 0]; % Red for cost increase (negative impact)
        bgColor = [0.9, 1, 0.9]; % Light red background
    else
        diffText = sprintf('+%.1f%%', abs(diffPercent));
        textColor = [0.7, 0, 0]; % Green for cost decrease (positive impact)
        bgColor = [1, 0.9, 0.9]; % Light green background
    end
    
    % Increased font size from 11 to 16 and added background box
    text(xPos, yPos, diffText, ...
        'HorizontalAlignment', 'center', 'VerticalAlignment', 'top', ...
        'FontSize', 24, 'FontWeight', 'bold', 'Color', textColor, ...
        'BackgroundColor', bgColor, 'EdgeColor', 'black', 'Margin', 3, ...
        'LineWidth', 1);
end

% Adjust y-axis to accommodate all labels
ylim([80000, 160000]);

% Add grid for readability
grid on;

% Add vertical lines to separate the algorithm groups
hold on;
for i = 1:2
    line([i*3+0.5, i*3+0.5], ylim, 'Color', [0.5 0.5 0.5], 'LineStyle', '--', 'HandleVisibility', 'off');
end
hold off;

% Add algorithm group labels with increased font size and background boxes
algorithmCenters = [2, 5, 8];  % Centers of each algorithm group
for i = 1:length(clusteringAlgorithms)
    text(algorithmCenters(i), 155000, clusteringAlgorithms{i}, ...
        'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', ...
        'FontSize', 24, 'FontWeight', 'bold', ...
        'BackgroundColor', [0.95, 0.95, 0.95], 'EdgeColor', 'none', 'Margin', 3); % Added background
end

% Adjust overall appearance
set(gcf, 'Color', 'white'); % White background

% Save the figure
saveas(gcf, 'outlier_impact_sidebyside.png');
saveas(gcf, 'outlier_impact_sidebyside.fig');

fprintf('Plot created and saved as outlier_impact_sidebyside.png and outlier_impact_sidebyside.fig\n'); 