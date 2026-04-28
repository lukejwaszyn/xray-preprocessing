function preprocess_xray(varargin)
% PREPROCESS_XRAY  X-ray preprocessing for dry-ice sublimation imaging.
%
% No args: launches file pickers.
%     preprocess_xray()
%
% With args:
%     preprocess_xray('object', 'file.tiff', 'dark', 'dark.tiff', ...
%                     'out', 'outputs')
%
% Optional name-value args: object, dark, flat, out, prefix.
%
% Author: Luke Waszyn
% Lab:    Penn State HEATER Lab, Experimental Frost Setup

% Tweak these if needed.
PCT_LOW    = 1;
PCT_HIGH   = 99;
CLAHE_CLIP = 0.01;
GAMMA      = 0.6;

p = inputParser;
addParameter(p, 'object', '', @ischar);
addParameter(p, 'dark',   '', @ischar);
addParameter(p, 'flat',   '', @ischar);
addParameter(p, 'out',    'outputs', @ischar);
addParameter(p, 'prefix', '', @ischar);
parse(p, varargin{:});
opt = p.Results;

% File pickers if no object path given.
if isempty(opt.object)
    fprintf('HEATER Lab X-ray Preprocessing\n\n');
    [f,d] = uigetfile({'*.tif;*.tiff','TIFF';'*.*','All'}, 'OBJECT TIFF (required)');
    if isequal(f,0); error('No object file selected.'); end
    opt.object = fullfile(d, f);

    [f,d] = uigetfile({'*.tif;*.tiff','TIFF';'*.*','All'}, 'DARK TIFF (optional)');
    if ~isequal(f,0); opt.dark = fullfile(d, f); end

    [f,d] = uigetfile({'*.tif;*.tiff','TIFF';'*.*','All'}, 'FLAT TIFF (optional)');
    if ~isequal(f,0); opt.flat = fullfile(d, f); end

    pickedOut = uigetdir(pwd, 'OUTPUT folder');
    if ~isequal(pickedOut,0); opt.out = pickedOut; end
end

fprintf('  object: %s\n', opt.object);
fprintf('  dark:   %s\n', ifelse(isempty(opt.dark), '(none)', opt.dark));
fprintf('  flat:   %s\n', ifelse(isempty(opt.flat), '(none)', opt.flat));
fprintf('  out:    %s\n\n', opt.out);

if ~exist(opt.out, 'dir'); mkdir(opt.out); end
prefix_ = '';
if ~isempty(opt.prefix); prefix_ = [opt.prefix '_']; end

% Load stacks.
obj = loadTiff(opt.object);
fprintf('[load] object: %dx%dx%d\n', size(obj,1), size(obj,2), size(obj,3));

dark = [];
if ~isempty(opt.dark)
    dark = loadTiff(opt.dark);
    fprintf('[load] dark: %dx%dx%d', size(dark,1), size(dark,2), size(dark,3));
    if all(dark(:) == 0)
        fprintf('   WARNING: dark all zeros, subtraction is a no-op\n');
    else
        fprintf('\n');
    end
end

flat = [];
if ~isempty(opt.flat)
    flat = loadTiff(opt.flat);
    fprintf('[load] flat: %dx%dx%d\n', size(flat,1), size(flat,2), size(flat,3));
end

% Drop empty/partial pages.
keep = [];
for k = 1:size(obj, 3)
    if 100 * nnz(obj(:,:,k)) / numel(obj(:,:,k)) > 50
        keep(end+1) = k; %#ok<AGROW>
    end
end
if isempty(keep); error('No usable frames.'); end
frames = obj(:,:,keep);
fprintf('[frames] using %d of %d: %s\n', numel(keep), size(obj,3), mat2str(keep));

% Average dark/flat across pages if they're stacks.
if ~isempty(dark) && size(dark,3) > 1; dark = mean(double(dark), 3); end
if ~isempty(flat) && size(flat,3) > 1; flat = mean(double(flat), 3); end

% Mean-stack reduces shot noise by sqrt(N).
meanFrame = mean(double(frames), 3);

nFrames = numel(keep);
allRaw  = cell(1, nFrames + 1);
allName = cell(1, nFrames + 1);
for i = 1:nFrames
    allRaw{i}  = double(frames(:,:,i));
    allName{i} = sprintf('%sframe%02d', prefix_, i-1);
end
allRaw{end}  = meanFrame;
allName{end} = sprintf('%smean_stack', prefix_);

for i = 1:numel(allRaw)
    raw  = allRaw{i};
    name = allName{i};

    corrected  = flatField(raw, dark, flat);
    normalized = percentileNormalize(corrected, PCT_LOW, PCT_HIGH);
    clahe      = adapthisteq(normalized, 'ClipLimit', CLAHE_CLIP, 'NumTiles', [8 8]);
    gammaImg   = imadjust(normalized, [], [], GAMMA);

    frameDir = fullfile(opt.out, name);
    if ~exist(frameDir, 'dir'); mkdir(frameDir); end
    saveImage(normalized, frameDir, 'normalized');
    saveImage(clahe,      frameDir, 'clahe');
    saveImage(gammaImg,   frameDir, 'gamma');

    figDir = fullfile(opt.out, 'figures');
    if ~exist(figDir, 'dir'); mkdir(figDir); end
    saveComparisonFigure(raw, normalized, clahe, gammaImg, ...
        fullfile(figDir, [name '_comparison.png']), ...
        sprintf('%s | pct=[%g,%g] CLAHE=%g gamma=%g', ...
                name, PCT_LOW, PCT_HIGH, CLAHE_CLIP, GAMMA));

    fprintf('[save] %s\n', name);
end

fprintf('\nDone. Outputs in: %s\n', opt.out);
end


% Helpers
function stack = loadTiff(path)
% Multi-page TIFF reader using the Tiff class (most reliable).
    info = imfinfo(path);
    n = numel(info);
    t = Tiff(path, 'r');
    cleanupObj = onCleanup(@() close(t));
    first = read(t);
    stack = zeros(size(first,1), size(first,2), n, 'like', first);
    stack(:,:,1) = first;
    for k = 2:n
        nextDirectory(t);
        stack(:,:,k) = read(t);
    end
end


function out = flatField(raw, dark, flat)
% (raw - dark) / (flat - dark), with graceful fallback if either is missing.
    out = double(raw);
    if ~isempty(dark)
        out = out - double(dark);
    end
    if ~isempty(flat)
        flatVal = double(flat);
        if ~isempty(dark); flatVal = flatVal - double(dark); end
        flatVal(flatVal <= 0) = 1;
        out = out ./ flatVal;
    end
end


function out = percentileNormalize(img, pLow, pHigh)
% Stretch [pLow, pHigh] percentiles to [0, 1]. Clips hot/dead pixels.
    lo = prctile(img(:), pLow);
    hi = prctile(img(:), pHigh);
    if hi <= lo
        out = zeros(size(img));
        return;
    end
    out = (double(img) - lo) / (hi - lo);
    out = max(0, min(1, out));
end


function saveImage(img, outdir, name)
% 16-bit TIFF for ML, 8-bit PNG for viewing.
    img = max(0, min(1, img));
    img16 = uint16(round(img * 65535));
    img8  = uint8( round(img * 255));
    imwrite(img16, fullfile(outdir, [name '.tiff']));
    imwrite(img8,  fullfile(outdir, [name '.png']));
end


function saveComparisonFigure(raw, normalized, clahe, gammaImg, outpath, titleStr)
    fig = figure('Visible','off','Position',[100 100 1200 1000]);
    subplot(2,2,1); imshow(raw, []);          title('Raw');         colorbar;
    subplot(2,2,2); imshow(normalized,[0 1]); title('Normalized');  colorbar;
    subplot(2,2,3); imshow(clahe,[0 1]);      title('CLAHE');       colorbar;
    subplot(2,2,4); imshow(gammaImg,[0 1]);   title('Gamma');       colorbar;
    sgtitle(titleStr, 'FontWeight', 'bold');
    exportgraphics(fig, outpath, 'Resolution', 120);
    close(fig);
end


function s = ifelse(cond, a, b)
    if cond; s = a; else; s = b; end
end
