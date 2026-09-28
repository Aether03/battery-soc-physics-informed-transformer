% Set HAPIT_REPO to the battery-soc-physics-informed-transformer repo root first.
repoPath = getenv('HAPIT_REPO');
assert(~isempty(repoPath), 'Set the HAPIT_REPO environment variable to the source repo root.');
portDir = fileparts(fileparts(mfilename('fullpath')));
addpath(fullfile(repoPath, 'src', 'matlab'));
checkpointPath = fullfile(repoPath, 'models', 'hapit_v8p6_best_checkpoint.mat');
outPath = fullfile(portDir, 'reference_io.mat');

data = load(checkpointPath);
net = data.best_net;

rng(42);
seqLen = 480;
numFeatures = 5;
X = rand(numFeatures, 1, seqLen) * 2 - 1;  % fixed random input, C x B x T, range [-1,1]

dlX = dlarray(X, 'CBT');
dlY = predict(net, dlX);
Y = extractdata(dlY);

fprintf('Input shape: %s\n', mat2str(size(X)));
fprintf('Output shape: %s\n', mat2str(size(Y)));
fprintf('Output sample [1,1,1:5]: %s\n', mat2str(squeeze(Y(1,1,1:5))'));
fprintf('Output sample [2,1,1:5]: %s\n', mat2str(squeeze(Y(2,1,1:5))'));

save(outPath, 'X', 'Y', '-v7');
fprintf('Saved reference input/output to %s\n', outPath);
