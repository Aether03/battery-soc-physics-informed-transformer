%% Merge_Lifecycle_Dataset.m
clear; clc; close all;

% 1. Load the Master Metadata Logbook
disp('Reading metadata_B0005.csv...');
metaData = readtable('metadata_B0005.csv');

% 2. Filter out 'impedance' cycles (We only want charge and discharge)
validIdx = ~strcmp(metaData.type, 'impedance');
cycleData = metaData(validIdx, :);

% ─────────────────────────────────────────────────────────────────────────
% ⚠️ THE GENERALIZATION DIAL ⚠️
% Start small! Merging the first 10 files gives you ~3 days of continuous data.
% If you change this to height(cycleData), it will merge ALL 300+ files,
% but remember your BOL physics engine might struggle with End-of-Life capacity!
numFilesToMerge = 20; 
% ─────────────────────────────────────────────────────────────────────────

disp(['Merging the first ', num2str(numFilesToMerge), ' charge/discharge files...']);

% 3. Initialize the Master Table and Time Tracker
masterTable = table();
currentTimeOffset = 0.0;

% 4. Loop through the metadata and stitch the files together
for i = 1:numFilesToMerge
    
    currentFile = cycleData.filename{i};
    cycleType   = cycleData.type{i};
    
    fprintf('Processing %03d/%03d: %s (%s)...\n', i, numFilesToMerge, currentFile, cycleType);
    
    % Read the individual CSV
    tempData = readtable(currentFile);
    
    % Standardize Columns: Extract ONLY what HA-PIT needs
    columnsToKeep = {'Voltage_measured', 'Current_measured', 'Temperature_measured', 'Time'};
    tempData = tempData(:, columnsToKeep);
    
    % Stitch the Timeline: Shift the start time to exactly 1 second after the last cycle ended
    tempData.Time = tempData.Time + currentTimeOffset;
    
    % Append to the Master Table
    masterTable = [masterTable; tempData];
    
    % Update the offset for the next file in the loop
    currentTimeOffset = tempData.Time(end) + 1.0; 
end

% 5. Save the final massive dataset
outputName = sprintf('Merged_B0005_%d_Cycles.csv', numFilesToMerge);
writetable(masterTable, outputName);
fprintf('\nSuccess! Created %s with %d total rows.\n', outputName, height(masterTable));

% 6. Plot the continuous current profile to verify the stitching
figure('Name', 'Continuous Lifecycle Verification', 'Position', [100 100 1000 400]);
plot(masterTable.Time / 3600, masterTable.Current_measured, 'b', 'LineWidth', 1);
yline(0, 'k--', 'LineWidth', 1);
xlabel('Time (Hours)'); ylabel('Current (Amps)');
title(sprintf('Continuous Current Profile (%d Files Merged)', numFilesToMerge));
grid on;