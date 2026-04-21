# X-ray Preprocessing for Frost Formation Imaging

Preprocessing pipeline for X-ray imagery of sublimating dry ice, supporting an ML proof-of-concept that renders frost formation from sparse imaging data.

**Author:** Luke Waszyn
**Lab:** Penn State HEATER Lab — Experimental Frost Setup

Available in both Python and MATLAB. The two implementations are feature-equivalent; pick whichever fits your workflow.

---

## What it does

Given a multi-page X-ray TIFF (the sample) and optional dark/flat-field TIFFs, the pipeline produces normalized and contrast-enhanced versions of each frame, plus a mean-stacked composite with improved signal-to-noise.

### Pipeline pseudocode

```
INPUT: object.tiff, [dark.tiff], [flat.tiff]

# Stage 1: Frame selection
stack <- load(object.tiff)
if auto-detect:
    frames <- pages where >50% of pixels are nonzero
else:
    frames <- stack[start:stop]

# Stage 2: Flat-field correction (per frame)
for each frame in frames:
    corrected <- (frame - dark) / (flat - dark)
    # falls back to (frame - dark) if no flat
    # falls back to frame if no dark

# Stage 3: Percentile normalization (per frame)
    lo <- 1st percentile of corrected
    hi <- 99th percentile of corrected
    normalized <- clip((corrected - lo) / (hi - lo), 0, 1)

# Stage 4: Contrast enhancement (per frame)
    clahe    <- CLAHE(normalized, clip_limit=0.01)
    gamma    <- normalized ^ 0.6

# Stage 5: Signal averaging
mean_stack <- mean(frames)
# repeat stages 2-4 on mean_stack

# Stage 6: Save
for each result:
    write 16-bit TIFF (for downstream ML)
    write 8-bit PNG (for visual inspection)
    write 4-panel comparison figure
```

### Why each step

| Step | Why |
|---|---|
| **Auto frame selection** | Acquisition software saves empty/partial pages; we drop them automatically so the user doesn't have to pre-clean the TIFF. |
| **Flat-field correction** | Standard X-ray radiometry. Removes detector response non-uniformity and beam-intensity variation. |
| **Percentile clipping** | A handful of hot/dead pixels at the extremes will otherwise collapse the useful dynamic range during min-max stretching. Clipping at 1/99 preserves >98% of pixels. |
| **CLAHE** | The sample (dry ice) and background have very different contrast levels; global histogram equalization over- or under-corrects regions. CLAHE adapts locally, making frost/edge features visible without blowing out the bright background. |
| **Gamma** | Alternative enhancement that doesn't amplify noise the way CLAHE can. Included for comparison. |
| **Mean stacking** | N-frame average reduces shot noise by √N. With ~7 frames that's ~2.6× cleaner output. |

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

You'll be prompted to select:
1. The object TIFF (required)
2. A dark TIFF (optional — click Cancel to skip)
3. A flat TIFF (optional — click Cancel to skip)
4. An output folder

**Command-line (for scripting):**

```bash
python src/preprocess.py --object path/to/file.tiff --out outputs
```

| Flag | Default | Purpose |
|---|---|---|
| `--object` | *(required in CLI mode)* | Input TIFF (single or multi-page) |
| `--dark` | none | Dark-frame TIFF |
| `--flat` | none | Flat-field TIFF |
| `--out` | `outputs` | Output directory |
| `--frames` | `auto` | `auto`, `all`, or `start:stop` (e.g. `0:7`) |
| `--prefix` | none | Filename prefix to keep multiple runs separate |

**Example — batch process multiple TIFFs into one folder:**

```bash
python src/preprocess.py --object run1.tiff --out outputs --prefix run1
python src/preprocess.py --object run2.tiff --out outputs --prefix run2
```

### Notebook

An exploration notebook is included for parameter tuning and figure iteration:

```bash
jupyter lab notebooks/01_explore.ipynb
```

---

## MATLAB — Usage

### Requirements
- MATLAB R2020a or newer
- Image Processing Toolbox (for `adapthisteq`, `imadjust`, `imread`, `imwrite`)

### Run it

**Interactive** — no arguments, file dialogs open:

```matlab
cd matlab
preprocess_xray()
```

**With arguments:**

```matlab
preprocess_xray('object', 'path/to/file.tiff', ...
                'dark',   'path/to/dark.tiff', ...
                'out',    'outputs')
```

**Named options:** `object`, `dark`, `flat`, `out`, `frames`, `prefix`, `pctLow`, `pctHigh`, `clipLim`, `gamma`. The `frames` option accepts `'auto'`, `'all'`, or a 2-element vector `[start stop]` (1-based inclusive).

---

## Output structure

```
outputs/
├── frame00/
│   ├── normalized.tiff     # 16-bit, for ML/downstream processing
│   ├── normalized.png      # 8-bit, for viewing
│   ├── clahe.tiff / .png
│   └── gamma.tiff / .png
├── frame01/                # ...one per input frame
├── mean_stack/             # averaged composite (best SNR)
└── figures/
    ├── frame00_comparison.png         # 4-panel: raw / normalized / CLAHE / gamma
    ├── frame00_normalized.png
    ├── mean_stack_comparison.png
    └── ...
```

---

## Roadmap

The current version covers **normalization + contrast enhancement**. Planned next stages:

- **Resolution upscaling** — start with bicubic/Lanczos, move to learned SR (SwinIR, Real-ESRGAN) if the contrast-enhanced output supports it.
- **Frost-formation renderer** — neural model trained to interpolate/extrapolate frost growth between sparse frames. Inputs: the 16-bit normalized TIFFs produced here. Ground truth: the mass-loss time series from the experimental log (weight, volume, density).

## Known data-acquisition issues to flag to the lab

- **Dark file integrity:** the test `dark.tiff` supplied to this project was effectively all zeros (66 nonzero pixels out of ~90 million across 83 pages). Dark frames should be reacquired with the detector in standard readout mode.
- **Frame count mismatch:** the test `object1_hole.tiff` had 21 pages but only ~7 contained real data. Confirm acquisition settings / trigger behavior.
- **No flat-field available:** when an open-beam (no sample) exposure is available, pass it with `--flat` / `'flat'` to enable full flat-field correction.

## Repository layout

```
.
├── src/
│   └── preprocess.py        # Python pipeline (CLI + interactive)
├── matlab/
│   └── preprocess_xray.m    # MATLAB port (same pipeline)
├── notebooks/
│   └── 01_explore.ipynb     # interactive tuning / figure-making
├── requirements.txt
├── .gitignore
└── README.md
```

## License

MIT
