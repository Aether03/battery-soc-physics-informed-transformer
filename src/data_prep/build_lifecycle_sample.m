%% Generate_Lifecycle_Dataset.m
clear; clc; close all;

disp('Reading metadata_B0005.csv...');
metaData = readtable('metadata_B0005.csv');

% 1. Filter out 'impedance' cycles (Keep only charge and discharge)
validIdx = ~strcmp(metaData.type, 'impedance');
cycleData = metaData(validIdx, :);
numValidCycles = height(cycleData);

% 2. Select the Target Cycles (BOL, MOL, EOL)
% We extract 4 rows (2 charge + 2 discharge cycles) from each stage of life.
bol_idx = 1 : 4;                                      % Beginning of Life (Cycles 1-2)
mol_idx = round(numValidCycles/2)-1 : round(numValidCycles/2)+2; % Middle of Life
eol_idx = numValidCycles-3 : numValidCycles;          % End of Life (Final cycles)

targetIndices = [bol_idx, mol_idx, eol_idx];
targetData = cycleData(targetIndices, :);

disp('Selected the following cycles for the Lifecycle Dataset:');
disp(targetData(:, {'type', 'filename', 'Capacity'}));

% 3. Initialize the Master Table and Time Tracker
masterTable = table();
currentTimeOffset = 0.0;

% 4. Loop through the selected files and stitch them together
for i = 1:height(targetData)
    
    currentFile = targetData.filename{i};
    cycleType   = targetData.type{i};
    cap         = targetData.Capacity(i);
    
    if isnan(cap)
        capStr = '(N/A - Charge)';
    else
        capStr = sprintf('%.3f A·h', cap);
    end
    
    fprintf('Stitching %02d/%02d: %s (%-9s) | SOH: %s\n', i, height(targetData), currentFile, cycleType, capStr);
    
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

% 5. Save the final dataset
outputName = 'Merged_B0005_Lifecycle_Sample.csv';
writetable(masterTable, outputName);
fprintf('\nSuccess! Created %s with %d total rows.\n', outputName, height(masterTable));

% 6. Plot the continuous current profile to verify the stitching
figure('Name', 'Lifecycle Verification', 'Position', [100 100 1000 400]);
plot(masterTable.Time / 3600, masterTable.Current_measured, 'b', 'LineWidth', 1);
yline(0, 'k--', 'LineWidth', 1);
xlabel('Time (Hours)'); ylabel('Current (Amps)');
title('Continuous Current Profile: BOL → MOL → EOL');
grid on;