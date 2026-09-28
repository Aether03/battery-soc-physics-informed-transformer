% Set HAPIT_REPO to the battery-soc-physics-informed-transformer repo root first.
repoPath = getenv('HAPIT_REPO');
assert(~isempty(repoPath), 'Set the HAPIT_REPO environment variable to the source repo root.');
addpath(fullfile(repoPath, 'src', 'matlab'));
checkpointPath = fullfile(repoPath, 'models', 'hapit_v8p6_best_checkpoint.mat');
data = load(checkpointPath);
fns = fieldnames(data);
fprintf('Top-level variables in checkpoint:\n');
for i = 1:numel(fns)
    v = data.(fns{i});
    fprintf('  %s : %s\n', fns{i}, class(v));
end

for i = 1:numel(fns)
    v = data.(fns{i});
    if isa(v, 'dlnetwork')
        fprintf('\nFound dlnetwork in variable: %s\n', fns{i});
        fprintf('Number of Learnables rows: %d\n', height(v.Learnables));
        disp(v.Learnables(:, {'Layer', 'Parameter'}));
    end
end
