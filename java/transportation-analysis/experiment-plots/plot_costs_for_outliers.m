% Plot costs against standard deviation thresholds for outlier detection
% This script uses hardcoded values extracted from CSV files in the results directory

clear;
close all;

% Hardcoded data extracted from the CSV files
% Standard deviation thresholds
stdDevs = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5];

% Total costs (extracted from the last line of each CSV file)
% Values have been converted from comma decimal separator to dot decimal separator
costs = [59785.11, 102415.61, 106858.33, 111260.47, 110519.07, 104349.44, 113427.48, 113669.45, 115296.48, 110372.06, 117361.37, 113909.39, 108965.68, 115576.48, 116538.86];

% Number of outliers for each threshold (extracted from the filenames)
numOutliers = [828, 202, 127, 80, 50, 33, 19, 16, 12, 10, 8, 7, 6, 5, 3];

% Create the figure
figure('Position', [100, 100, 1000, 600]);

% Plot cost vs standard deviation
yyaxis left
h1 = plot(stdDevs, costs, 'b-o', 'LineWidth', 1.0, 'MarkerSize', 4, 'MarkerFaceColor', 'b');
ylabel('Total Cost', 'FontSize', 20);
ylim([50000, 120000]);

% Also plot number of outliers on secondary y-axis (using log scale for better visibility)
yyaxis right
h2 = plot(stdDevs, numOutliers, 'r--*', 'LineWidth', 1.0, 'MarkerSize', 4);
ylabel('Number of Outliers', 'FontSize', 20);
set(gca, 'YScale', 'default'); % Use log scale for outliers count

% Add title and labels
title('Cost and Number of Outliers vs. Standard Deviation Threshold', 'FontSize', 14);
xlabel('Standard Deviation Threshold', 'FontSize', 20);
grid on;

% Add legend using the stored handles
legend([h1, h2], {'Total Cost', 'Number of Outliers'}, 'Location', 'best');

% Add data labels for costs
yyaxis left
for i = 1:length(stdDevs)
    text(stdDevs(i), costs(i), ['  ' num2str(costs(i), '%.1f')], 'FontSize', 11, 'VerticalAlignment', 'bottom');
end

% Add data labels for number of outliers
yyaxis right
for i = 1:length(stdDevs)
    text(stdDevs(i), numOutliers(i), ['  ' num2str(numOutliers(i))], 'FontSize', 11, 'VerticalAlignment', 'top', 'Color', 'r');
end

% Save the figure
saveas(gcf, 'outlier_cost_analysis.png');
saveas(gcf, 'outlier_cost_analysis.fig');

fprintf('Analysis completed. Figure saved as outlier_cost_analysis.png and outlier_cost_analysis.fig\n'); 