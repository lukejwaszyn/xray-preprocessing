# X-ray Preprocessing for Frost Formation Imaging

Preprocessing pipeline for X-ray imagery of sublimating dry ice. Output feeds into a planned ML proof-of-concept that renders frost formation from sparse imaging.

**Author:** Luke Waszyn
**Lab:** Penn State HEATER Lab, Experimental Frost Setup

Available in Python and MATLAB. Pick whichever fits your workflow; both produce the same output.

This is *preprocessing* (data prep before ML), not the ML model itself. The model is planned but not implemented yet. See [Roadmap](#roadmap).

## What it does

Takes a multi-page X-ray TIFF (the sample) plus optional dark/flat frames. Outputs normalized and contrast-enhanced versions of each frame, plus a mean-stacked composite with better signal-to-noise.

### Pipeline

```
INPUT: object.tiff, [dark.tiff], [flat.tiff]

# 1. Drop empty pages from acquisition glitches
frames = pages where >50% of pixels are nonzero

# 2. Flat-field correction (per frame)
corrected = (frame - dark) / (flat - dark)
# Falls back to (frame - dark) if no flat.
# Falls back to frame if no dark.

# 3. Percentile normalization to [0, 1]
lo = 1st percentile of corrected
hi = 99th percentile of corrected
normalized = clip((corrected - lo) / (hi - lo), 0, 1)

# 4. Contrast enhancement
clahe = CLAHE(normalized, clip_limit=0.01)
gamma = normalized ^ 0.6

# 5. Mean stack
mean_stack = mean(frames)
# repeat steps 2-4 on mean_stack

# 6. Save
write 16-bit TIFF (full grayscale, for ML)
write 8-bit PNG  (for viewing)
write 4-panel comparison figure
```

### Why each step

| Step | Why |
|---|---|
| **Frame selection** | Acquisition saves blank pages on failed captures. Drop them automatically. |
| **Flat-field** | Standard X-ray correction. Removes detector baseline (dark) and beam non-uniformity (flat). |
| **Percentile clipping** | Hot/dead pixels at the extremes will collapse the dynamic range. Clipping at 1/99 keeps >98% of pixels intact. |
| **CLAHE** | Sample and background have very different contrasts. Local equalization makes frost edges visible without blowing out bright areas. |
| **Gamma** | Less aggressive alternative to CLAHE. Doesn't amplify noise the same way. |
| **Mean stacking** | Averaging N frames cuts shot noise by sqrt(N). With 7 frames, ~2.6x cleaner. |

## Grayscale preservation

Pixel intensities are continuous through the whole pipeline. Nothing gets thresholded to binary. A pixel reading 1850 stays a number close to 1850.

| Stage | Type | Levels |
|---|---|---|
| Raw input | uint16 | ~65,000 |
| Math stages | float32 | continuous |
| **Saved .tiff** | **uint16** | **~65,000** |
| Saved .png | uint8 | 256 |

**Use the .tiff files for ML.** PNGs are quantized to 8-bit for previewing only.

Verify on your own outputs:

```matlab
img = imread('outputs/mean_stack/normalized.tiff');
fprintf('Distinct gray levels: %d\n', numel(unique(img(:))));
```

```python
import tifffile, numpy as np
img = tifffile.imread('outputs/mean_stack/normalized.tiff')
print(f"Distinct gray levels: {len(np.unique(img))}")
```

## Python

### Setup

```bash
git clone https://github.com/lukejwaszyn/xray-preprocessing.git
cd xray-preprocessing
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Run

File picker:
```bash
python src/preprocess.py
```

Command line:
```bash
python src/preprocess.py --object file.tiff --dark dark.tiff --out outputs
```

Flags: `--object` (required), `--dark`, `--flat`, `--out`, `--prefix`.

## MATLAB

Requires R2020a+ and the Image Processing Toolbox.

```matlab
cd matlab
preprocess_xray()
```

Or:
```matlab
preprocess_xray('object', 'file.tiff', 'dark', 'dark.tiff', 'out', 'outputs')
```

Args: `object`, `dark`, `flat`, `out`, `prefix`.

## Output structure

```
outputs/
├── frame00/
│   ├── normalized.tiff      16-bit, for ML
│   ├── normalized.png       8-bit, for viewing
│   ├── clahe.tiff / .png
│   └── gamma.tiff / .png
├── frame01/  ...
├── mean_stack/              best SNR
└── figures/
    ├── frame00_comparison.png
    └── ...
```

## Roadmap

Currently: normalization + contrast. Planned next:

- Resolution upscaling (bicubic first, then learned SR if needed).
- Frost-formation renderer. ML model interpolating frost growth from sparse frames. Input: 16-bit normalized TIFFs. Ground truth: mass-loss time series from the experimental log.

## Known data issues

- **Dark frame:** the test dark.tiff was effectively all zeros. Needs reacquiring.
- **Empty pages:** test object1_hole.tiff had 21 pages but only 7 with real data.
- **No flat field yet.** When available, pass with `--flat` / `'flat'`.

## Layout

```
.
├── src/preprocess.py
├── matlab/preprocess_xray.m
├── notebooks/01_explore.ipynb
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md
```

## License

MIT

## Example outputs

Run on `specimen_with_frost.tiff` with `noobject.tiff` as the flat field. See `examples/`.

| File | Stage |
|---|---|
| `mean_stack_comparison.png` | All four stages side by side |
| `normalized.png` | After flat-field + percentile normalization |
| `clahe.png` | After CLAHE local contrast enhancement |
| `gamma.png` | After gamma correction |
