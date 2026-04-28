# X-ray Preprocessing for Frost Formation Imaging

Preprocessing pipeline for X-ray imagery of sublimating dry ice. Output feeds into an ML proof-of-concept that renders frost formation from sparse imaging data.

**Author:** Luke Waszyn
**Lab:** Penn State HEATER Lab — Experimental Frost Setup

Available in both Python and MATLAB. Pick whichever fits your workflow — the two implementations produce the same output.

> Note on terminology: this is *preprocessing* (preparing data before it goes to an ML model), not the model itself. The ML/rendering "processing" stage is planned but not implemented yet. See [Roadmap](#roadmap) below.

---

## What it does

Given a multi-page X-ray TIFF (the sample) and optional dark/flat-field TIFFs, the pipeline produces normalized and contrast-enhanced versions of each frame, plus a mean-stacked composite with improved signal-to-noise.

### Pipeline pseudocode

```
INPUT: object.tiff, [dark.tiff], [flat.tiff]

# Stage 1 — Frame selection
stack <- load(object.tiff)
frames <- pages where >50% of pixels are nonzero  # drops empty pages

# Stage 2 — Flat-field correction (per frame)
for each frame in frames:
    corrected <- (frame - dark) / (flat - dark)
    # falls back to (frame - dark) if no flat
    # falls back to frame if no dark

# Stage 3 — Percentile normalization (per frame)
    lo <- 1st percentile of corrected
    hi <- 99th percentile of corrected
    normalized <- clip((corrected - lo) / (hi - lo), 0, 1)

# Stage 4 — Contrast enhancement (per frame)
    clahe <- CLAHE(normalized, clip_limit=0.01)
    gamma <- normalized ^ 0.6

# Stage 5 — Signal averaging
mean_stack <- mean(frames)
# repeat stages 2-4 on mean_stack

# Stage 6 — Save
for each result:
    write 16-bit TIFF (full grayscale precision, for ML)
    write 8-bit PNG  (for viewing)
    write 4-panel comparison figure
```

### Why each step

| Step | Why |
|---|---|
| **Frame selection** | Acquisition software saves empty/partial pages when captures fail. We drop them automatically so you don't have to pre-clean the TIFF. |
| **Flat-field correction** | Standard X-ray radiometry. Removes detector baseline (dark) and corrects for beam-intensity non-uniformity (flat). |
| **Percentile clipping** | A handful of hot/dead pixels at the extremes will otherwise collapse the useful dynamic range during min-max stretching. Clipping at 1/99 preserves >98% of pixels. |
| **CLAHE** | Sample and background have very different contrast levels; global histogram equalization over- or under-corrects regions. CLAHE adapts locally, making frost/edge features visible without blowing out the bright background. |
| **Gamma** | Alternative enhancement. Doesn't amplify noise the way CLAHE can. Saved alongside CLAHE for comparison. |
| **Mean stacking** | An N-frame average reduces shot noise by √N. With ~7 frames that's ~2.6× cleaner output. |

---

## Grayscale preservation

This pipeline works with **continuous pixel intensities** at every stage. There is no thresholding to binary (no "is something there or not" decision). A pixel reading 1850 stays a number close to 1850 throughout. The only place pixels get clamped is at the percentile extremes (top/bottom 1%), and that's intentional — it removes detector defects without affecting real signal.

Precision at each stage:

| Stage | Type | Levels |
|---|---|---|
| Raw input | `uint16` | ~65,000 |
| Flat-field math | `float32` | continuous |
| Percentile-normalized | `float32` in [0, 1] | continuous |
| CLAHE / Gamma | `float32` in [0, 1] | continuous |
| **Saved `.tiff`** | **`uint16`** | **~65,000** |
| Saved `.png` | `uint8` | 256 |

**For the ML stage, always read from the `.tiff` files, not the `.png`s.** The PNGs are quantized to 8-bit for display purposes only and lose ~99% of the dynamic range.

You can verify grayscale preservation on your own outputs:

```matlab
% MATLAB
img = imread('outputs/mean_stack/normalized.tiff');
fprintf('Distinct gray levels: %d\n', numel(unique(img(:))));
% Should print thousands (typically 30,000 - 60,000)
```

```python
# Python
import tifffile, numpy as np
img = tifffile.imread('outputs/mean_stack/normalized.tiff')
print(f"Distinct gray levels: {len(np.unique(img))}")
```

---

## Python — Usage

### Setup (one-time)

```bash
git clone https://github.com/lukejwaszyn/xray-preprocessing.git
cd xray-preprocessing
python3 -m venv .venv
source .venv/bin/activate           # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Run it

**Interactive (easiest)** — no arguments, file pickers open:

```bash
python src/preprocess.py
```

You'll be prompted for:
1. Object TIFF (required)
2. Dark TIFF (optional — Cancel to skip)
3. Flat TIFF (optional — Cancel to skip)
4. Output folder

**Command line:**

```bash
python src/preprocess.py --object file.tiff --dark dark.tiff --out outputs
```

| Flag | Default | Purpose |
|---|---|---|
| `--object` | *(required)* | Input TIFF |
| `--dark` | none | Dark-frame TIFF |
| `--flat` | none | Flat-field TIFF |
| `--out` | `outputs` | Output folder |
| `--prefix` | none | Filename prefix to keep multiple runs separate |

---

## MATLAB — Usage

Requires MATLAB R2020a+ and the Image Processing Toolbox.

**Interactive:**

```matlab
cd matlab
preprocess_xray()
```

**With arguments:**

```matlab
preprocess_xray('object', 'file.tiff', 'dark', 'dark.tiff', 'out', 'outputs')
```

Optional name-value arguments: `object`, `dark`, `flat`, `out`, `prefix`.

---

## Output structure

```
outputs/
├── frame00/
│   ├── normalized.tiff   # 16-bit, for ML
│   ├── normalized.png    # 8-bit, for viewing
│   ├── clahe.tiff / .png
│   └── gamma.tiff / .png
├── frame01/  ...etc
├── mean_stack/           # averaged composite, best SNR
└── figures/
    ├── frame00_comparison.png       # 4-panel: raw / norm / CLAHE / gamma
    ├── mean_stack_comparison.png
    └── ...
```

---

## Roadmap

Current version covers **normalization + contrast enhancement only**. Planned next stages:

- **Resolution upscaling** — start with bicubic/Lanczos, move to learned super-resolution (SwinIR, Real-ESRGAN) if needed.
- **Frost-formation renderer** — neural model trained to interpolate frost growth between sparse frames. Inputs: the 16-bit normalized TIFFs. Ground truth: the mass-loss time series from the experimental log.

## Known data-acquisition issues

- **Dark integrity:** the test `dark.tiff` was effectively all zeros (66 nonzero pixels out of ~90 million). Dark frames need reacquiring with the detector in standard readout mode.
- **Empty frames:** the test `object1_hole.tiff` had 21 pages but only ~7 with real data. Confirm acquisition trigger settings.
- **No flat field:** when an open-beam exposure becomes available, pass it with `--flat` / `'flat'` to enable full flat-field correction.

## Repository layout

```
.
├── src/preprocess.py         # Python pipeline
├── matlab/preprocess_xray.m  # MATLAB port
├── notebooks/01_explore.ipynb
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md
```

## License

MIT
