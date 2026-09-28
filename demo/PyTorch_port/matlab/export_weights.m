% Set HAPIT_REPO to the battery-soc-physics-informed-transformer repo root first.
repoPath = getenv('HAPIT_REPO');
assert(~isempty(repoPath), 'Set the HAPIT_REPO environment variable to the source repo root.');
portDir = fileparts(fileparts(mfilename('fullpath')));
addpath(fullfile(repoPath, 'src', 'matlab'));
checkpointPath = fullfile(repoPath, 'models', 'hapit_v8p6_best_checkpoint.mat');
outPath = fullfile(portDir, 'weights_raw.mat');

data = load(checkpointPath);
net = data.best_net;
L = net.Learnables;

exportStruct = struct();
n = height(L);
fprintf('Exporting %d learnable parameters...\n', n);

for i = 1:n
    layerName = L.Layer(i);
    paramName = L.Parameter(i);
    fieldName = strcat(layerName, '__', paramName);
    fieldName = strrep(fieldName, '-', '_');
    value = extractdata(L.Value{i});
    exportStruct.(fieldName) = value;
    fprintf('  %-30s shape=%s\n', fieldName, mat2str(size(value)));
end

save(outPath, '-struct', 'exportStruct', '-v7');
fprintf('\nSaved %d arrays to %s\n', n, outPath);
