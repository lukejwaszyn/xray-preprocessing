function preprocess_xray(varargin)
% PREPROCESS_XRAY  X-ray preprocessing for dry-ice sublimation imaging.
%
%   PREPROCESS_XRAY() launches interactive file pickers for object / dark /
%   flat TIFFs and an output folder, then runs the pipeline.
%
%   PREPROCESS_XRAY('object', objPath, 'out', outDir, ...) runs
%   non-interactively. Name-value options:
%       'object'   (required path to object TIFF stack)
%       'dark'     (optional path to dark TIFF; default [])
%       'flat'     (optional path to flat TIFF; default [])
%       'out'      (output folder; default 'outputs')
%       'frames'   ('auto' (default), 'all', or [startIdx stopIdx] 1-based inclusive)
%       'prefix'   (filename prefix; default '')
%       'pctLow'   (percentile low; default 1)
%       'pctHigh'  (percentile high; default 99)
%       'clipLim'  (CLAHE clip limit; default 0.01)
%       'gamma'    (gamma value; default 0.6)
%
%   Pipeline (per frame):
%     1. Optional flat-field correction: (raw - dark) / (flat - dark)
%     2. Percentile normalization to [0, 1]
%     3. CLAHE for local contrast + gamma for global
%     4. Save 16-bit TIFF + 8-bit PNG of each stage, plus comparison figure
%
%   Author:  Luke Waszyn
%   Lab:     Penn State HEATER Lab -- Experimental Frost Setup

% ------------------------------------------------------------------------
% Parse inputs
% ------------------------------------------------------------------------
p = inputParser;
addParameter(p, 'object',  '', @ischar);
addParameter(p, 'dark',    '', @ischar);
addParameter(p, 'flat',    '', @ischar);
addParameter(p, 'out',     'outputs', @ischar);
addParameter(p, 'frames',  'auto');
addParameter(p, 'prefix',  '', @ischar);
addParameter(p, 'pctLow',  1,    @isnumeric);
addParameter(p, 'pctHigh', 99,   @isnumeric);
addParameter(p, 'clipLim', 0.01, @isnumeric);
addParameter(p, 'gamma',   0.6,  @isnumeric);
parse(p, varargin{:});
opt = p.Results;

% ------------------------------------------------------------------------
% Interactive mode if no object path given
% ------------------------------------------------------------------------
if isempty(opt.object)
    fprintf('============================================================\n');
    fprintf('  HEATER Lab -- X-ray Preprocessing Pipeline\n');
    fprintf('  Experimental Frost Setup\n');
    fprintf('============================================================\n\n');

    [f, d] = uigetfile({'*.tif;*.tiff','TIFF files';'*.*','All'}, ...
                       'Select OBJECT TIFF stack (required)');
    if isequal(f, 0); error('No object TIFF selected; aborting.'); end
    opt.object = fullfile(d, f);

    [f, d] = uigetfile({'*.tif;*.tiff','TIFF files';'*.*','All'}, ...
                       'Select DARK TIFF (optional, Cancel to skip)');
    if ~isequal(f, 0); opt.dark = fullfile(d, f); end

    [f, d] = uigetfile({'*.tif;*.tiff','TIFF files';'*.*','All'}, ...
                       'Select FLAT TIFF (optional, Cancel to skip)');
    if ~isequal(f, 0); opt.flat = fullfile(d, f); end

    outPick = uigetdir(pwd, 'Select OUTPUT folder');
    if ~isequal(outPick, 0); opt.out = outPick; end
end

fprintf('  object: %s\n', opt.object);
fprintf('  dark:   %s\n', ternary(isempty(opt.dark), '(none)', opt.dark));
fprintf('  flat:   %s\n', ternary(isempty(opt.flat), '(none)', opt.flat));
fprintf('  out:    %s\n\n', opt.out);

if ~exist(opt.out, 'dir'); mkdir(opt.out); end
prefixStr = '';
if ~isempty(opt.prefix); prefixStr = [opt.prefix '_']; end

% ------------------------------------------------------------------------
% Load stacks
% ------------------------------------------------------------------------
objStack = loadTiffStack(opt.object);
fprintf('[load] object stack: %dx%dx%d\n', size(objStack,1), size(objStack,2), size(objStack,3));

dark = [];
if ~isempty(opt.dark)
    dark = loadTiffStack(opt.dark);
    fprintf('[load] dark: %dx%dx%d', size(dark,1), size(dark,2), size(dark,3));
    if all(dark(:) == 0)
        fprintf('   WARNING: dark is all zeros -- subtraction is a no-op\n');
    else
        fprintf('\n');
    end
end

flat = [];
if ~isempty(opt.flat)
    flat = loadTiffStack(opt.flat);
    fprintf('[load] flat: %dx%dx%d\n', size(flat,1), size(flat,2), size(flat,3));
end

% ------------------------------------------------------------------------
% Frame selection
% ------------------------------------------------------------------------
nTotal = size(objStack, 3);
if ischar(opt.frames) || isstring(opt.frames)
    spec = lower(char(opt.frames));
    if strcmp(spec, 'auto')
        kept = autoDetectUsable(objStack, 50);
        fprintf('[frames] auto-detected %d usable frames: %s\n', numel(kept), mat2str(kept));
    elseif strcmp(spec, 'all')
        kept = 1:nTotal;
        fprintf('[frames] using all %d frames\n', nTotal);
    else
        error('Unknown frame spec: %s', spec);
    end
else
    rng = opt.frames;
    kept = rng(1):rng(2);
    fprintf('[frames] using explicit range %d:%d (%d frames)\n', rng(1), rng(2), numel(kept));
end
frames = objStack(:,:,kept);

% ------------------------------------------------------------------------
% Flatten dark/flat to 2-D if they are stacks
% ------------------------------------------------------------------------
if ~isempty(dark) && size(dark,3) > 1; dark = mean(double(dark), 3); end
if ~isempty(flat) && size(flat,3) > 1; flat = mean(double(flat), 3); end

% ------------------------------------------------------------------------
% Process each frame + mean stack
% ------------------------------------------------------------------------
meanFrame = mean(double(frames), 3);
fprintf('[stack] mean frame: min=%.1f max=%.1f mean=%.1f\n', ...
        min(meanFrame(:)), max(meanFrame(:)), mean(meanFrame(:)));

allInputs = cell(1, numel(kept) + 1);
for i = 1:numel(kept)
    allInputs{i} = struct('stem', sprintf('%sframe%02d', prefixStr, i-1), ...
                          'raw', double(frames(:,:,i)));
end
allInputs{end} = struct('stem', sprintf('%smean_stack', prefixStr), 'raw', meanFrame);

for i = 1:numel(allInputs)
    stem = allInputs{i}.stem;
    raw  = allInputs{i}.raw;

    corrected  = flatFieldCorrect(raw, flat, dark);
    normalized = percentileNormalize(corrected, opt.pctLow, opt.pctHigh);
    clahe      = adapthisteq(normalized, 'ClipLimit', opt.clipLim, ...
                              'NumTiles', [8 8], 'Distribution', 'uniform');
    gammaImg   = imadjust(normalized, [], [], opt.gamma);

    % Save stages
    frameDir = fullfile(opt.out, stem);
    if ~exist(frameDir, 'dir'); mkdir(frameDir); end
    saveStage(normalized, frameDir, 'normalized');
    saveStage(clahe,      frameDir, 'clahe');
    saveStage(gammaImg,   frameDir, 'gamma');

    % Comparison figure
    figDir = fullfile(opt.out, 'figures');
    if ~exist(figDir, 'dir'); mkdir(figDir); end
    makeComparisonFigure(raw, normalized, clahe, gammaImg, ...
        fullfile(figDir, sprintf('%s_comparison.png', stem)), ...
        sprintf('%s | pct=[%.1f,%.1f] CLAHE clip=%.3f gamma=%.2f', ...
                stem, opt.pctLow, opt.pctHigh, opt.clipLim, opt.gamma));

    makeNormalizedFigure(normalized, ...
        fullfile(figDir, sprintf('%s_normalized.png', stem)));

    fprintf('[save] %s: normalized/clahe/gamma + comparison figure\n', stem);
end

fprintf('\nDone. Outputs in: %s\n', opt.out);
end

% ========================================================================
% Helpers
% ========================================================================
function stack = loadTiffStack(path)
% Load a multi-page TIFF as an H x W x N array.
    info = imfinfo(path);
    n = numel(info);
    first = imread(path, 1);
    stack = zeros(size(first,1), size(first,2), n, 'like', first);
    stack(:,:,1) = first;
    for k = 2:n
        stack(:,:,k) = imread(path, k);
    end
end

function kept = autoDetectUsable(stack, minNonZeroPct)
% Return frame indices where >minNonZeroPct% of pixels are nonzero.
    n = size(stack, 3);
    kept = [];
    for k = 1:n
        p = stack(:,:,k);
        nzPct = 100 * nnz(p) / numel(p);
        if nzPct > minNonZeroPct
            kept(end+1) = k; %#ok<AGROW>
        end
    end
    if isempty(kept)
        error('No usable frames (all pages <=%g%% nonzero).', minNonZeroPct);
    end
end

function out = flatFieldCorrect(raw, flat, dark)
% Standard flat-field: (raw - dark) / (flat - dark). Falls back gracefully
% if dark or flat are missing.
    rawD = double(raw);
    if isempty(dark); darkD = zeros(size(rawD)); else; darkD = double(dark); end
    if isempty(flat)
        out = rawD - darkD;
    else
        flatD = double(flat);
        denom = flatD - darkD;
        denom(denom <= 0) = 1;          % guard dead pixels
        out = (rawD - darkD) ./ denom;
    end
end

function out = percentileNormalize(img, pLow, pHigh)
% Clip to [pLow, pHigh] percentiles, scale to [0, 1].
    lo = prctile(img(:), pLow);
    hi = prctile(img(:), pHigh);
    if hi <= lo
        out = zeros(size(img));
        return;
    end
    out = (double(img) - lo) / (hi - lo);
    out = max(0, min(1, out));
end

function saveStage(img01, outdir, stem)
% Save a [0,1] double image as both 16-bit TIFF and 8-bit PNG.
    img16 = uint16(round(max(0,min(1,img01)) * 65535));
    img8  = uint8( round(max(0,min(1,img01)) * 255));
    imwrite(img16, fullfile(outdir, [stem '.tiff']));
    imwrite(img8,  fullfile(outdir, [stem '.png']));
end

function makeComparisonFigure(raw, norm, clahe, gammaImg, outpath, titleStr)
    fig = figure('Visible','off','Position',[100 100 1200 1000]);
    subplot(2,2,1); imshow(raw, []);      title('Raw (auto-scaled)'); colorbar;
    subplot(2,2,2); imshow(norm,[0 1]);   title('Percentile-normalized'); colorbar;
    subplot(2,2,3); imshow(clahe,[0 1]);  title('CLAHE'); colorbar;
    subplot(2,2,4); imshow(gammaImg,[0 1]); title('Gamma'); colorbar;
    sgtitle(titleStr, 'FontWeight', 'bold');
    exportgraphics(fig, outpath, 'Resolution', 120);
    close(fig);
end

function makeNormalizedFigure(img01, outpath)
    fig = figure('Visible','off','Position',[100 100 900 700]);
    imshow(img01, [0 1]); colorbar; colormap(gray);
    title('Normalized image', 'FontWeight', 'bold');
    grid on;
    exportgraphics(fig, outpath, 'Resolution', 120);
    close(fig);
end

function s = ternary(cond, a, b)
    if cond; s = a; else; s = b; end
end
