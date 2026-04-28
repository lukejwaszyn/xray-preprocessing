"""
X-ray preprocessing pipeline for dry-ice sublimation imaging.

Author:  Luke Waszyn
Lab:     Penn State HEATER Lab -- Experimental Frost Setup

This script normalizes and contrast-enhances multi-page X-ray TIFF stacks.
All operations preserve grayscale intensity (no thresholding into binary).
The output 16-bit TIFFs retain ~65,000 levels of detail for downstream ML.

Run with no arguments for an interactive file picker:
    python preprocess.py

Or from the command line:
    python preprocess.py --object file.tiff --dark dark.tiff --out outputs
"""

import sys
import argparse
from pathlib import Path

import numpy as np
import tifffile
from skimage import exposure, img_as_ubyte, img_as_uint
import matplotlib.pyplot as plt


# ----------------------------------------------------------------------
# Pipeline parameters -- tweak here if needed
# ----------------------------------------------------------------------
PCT_LOW    = 1.0    # clip the bottom 1% of pixels (kills dead pixels)
PCT_HIGH   = 99.0   # clip the top 1% of pixels (kills hot pixels / saturation)
CLAHE_CLIP = 0.01   # CLAHE strength: higher = more contrast, more noise
GAMMA      = 0.6    # gamma < 1 brightens midtones, > 1 darkens them


# ----------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------
def load_tiff(path):
    """Load a TIFF as a (N_pages, H, W) array. Single-page files become (1, H, W)."""
    arr = tifffile.imread(str(path))
    if arr.ndim == 2:
        arr = arr[None, ...]
    return arr


def find_usable_frames(stack, min_nonzero_pct=50):
    """
    Filter out empty / partial pages from a TIFF stack.

    Some acquisition software saves blank pages when frames fail. We keep
    only pages where more than 50% of pixels are nonzero -- a real frame
    is typically 95%+ nonzero, so this comfortably separates good from bad.
    """
    keep = []
    for i in range(len(stack)):
        nonzero_pct = 100 * np.count_nonzero(stack[i]) / stack[i].size
        if nonzero_pct > min_nonzero_pct:
            keep.append(i)
    if not keep:
        raise ValueError("No usable frames found (all pages mostly empty).")
    return stack[keep], keep


# ----------------------------------------------------------------------
# The actual processing -- each function takes intensities, returns intensities
# ----------------------------------------------------------------------
def flat_field(raw, dark=None, flat=None):
    """
    Standard X-ray flat-field correction:  (raw - dark) / (flat - dark)

    This removes the detector's baseline offset (dark) and corrects for
    beam-intensity non-uniformity (flat).

    If only dark is provided: returns raw - dark.
    If neither: returns raw unchanged (as float for downstream math).
    """
    raw = raw.astype(np.float32)

    if dark is not None:
        # If dark is a stack, average it down to a single 2-D frame
        if dark.ndim == 3:
            dark = dark.mean(axis=0)
        raw = raw - dark.astype(np.float32)

    if flat is not None:
        if flat.ndim == 3:
            flat = flat.mean(axis=0)
        flat = flat.astype(np.float32)
        if dark is not None:
            flat = flat - dark.astype(np.float32)
        # Avoid division by zero in dead detector regions
        flat[flat <= 0] = 1.0
        raw = raw / flat

    return raw


def percentile_normalize(img, pct_low=1.0, pct_high=99.0):
    """
    Stretch the image so that the 1st-99th percentile range maps to [0, 1].

    This is what makes the output look clean. A naive min-max stretch gets
    destroyed by a few hot pixels near 65535 -- the "real" pixels end up
    crushed into a tiny range. Percentile clipping ignores the extremes
    and stretches the actual signal range across the full output.

    Grayscale is fully preserved: a pixel that was darker than another in
    the input is still darker (just rescaled to a 0-1 float).
    """
    lo = np.percentile(img, pct_low)
    hi = np.percentile(img, pct_high)
    if hi <= lo:
        return np.zeros_like(img, dtype=np.float32)
    out = (img.astype(np.float32) - lo) / (hi - lo)
    return np.clip(out, 0.0, 1.0)


def apply_clahe(img, clip_limit=0.01):
    """
    Contrast Limited Adaptive Histogram Equalization.

    CLAHE divides the image into tiles and equalizes each one's histogram
    separately, then blends them. This boosts local contrast (good for
    seeing frost edges) without blowing out brighter regions the way a
    single global stretch would.

    Input must be in [0, 1]. Output is also [0, 1].
    """
    h, w = img.shape
    # Use a tile size that's ~1/8 of the image -- a reasonable default
    kernel = (max(8, h // 8), max(8, w // 8))
    return exposure.equalize_adapthist(img, kernel_size=kernel, clip_limit=clip_limit)


def apply_gamma(img, gamma=0.6):
    """
    Gamma correction: output = input ^ gamma.

    Gamma < 1 brightens midtones (useful when sample is dark on a bright
    background); gamma > 1 darkens them. Less aggressive than CLAHE and
    doesn't amplify noise the same way, so we save it as an alternative.
    """
    return exposure.adjust_gamma(img, gamma=gamma)


# ----------------------------------------------------------------------
# Saving
# ----------------------------------------------------------------------
def save_image(img, outdir, name):
    """
    Save a [0, 1] float image two ways:
      - 16-bit TIFF: full ~65,000 grayscale levels, use this for ML
      - 8-bit PNG:   256 levels, just for human viewing
    """
    outdir.mkdir(parents=True, exist_ok=True)
    img = np.clip(img, 0, 1)
    # 16-bit TIFF preserves precision -- this is the one for downstream work
    tifffile.imwrite(outdir / f"{name}.tiff", img_as_uint(img))
    # 8-bit PNG is fine for previewing in Finder / image viewers
    plt.imsave(outdir / f"{name}.png", img_as_ubyte(img), cmap="gray", vmin=0, vmax=255)


def save_comparison_figure(raw, normalized, clahe, gamma, outpath, title):
    """4-panel figure showing each pipeline stage side-by-side."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    panels = [
        ("Raw", raw),
        ("Percentile-normalized", normalized),
        ("CLAHE", clahe),
        ("Gamma", gamma),
    ]
    for ax, (label, img) in zip(axes.flat, panels):
        im = ax.imshow(img, cmap="gray")
        ax.set_title(label)
        plt.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle(title, fontweight="bold")
    fig.tight_layout()
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, dpi=120, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------
# Main pipeline
# ----------------------------------------------------------------------
def process_one(raw, dark=None, flat=None):
    """Run a single frame through all pipeline stages. Returns a dict of results."""
    corrected  = flat_field(raw, dark=dark, flat=flat)
    normalized = percentile_normalize(corrected, PCT_LOW, PCT_HIGH)
    clahe      = apply_clahe(normalized, CLAHE_CLIP)
    gamma      = apply_gamma(normalized, GAMMA)
    return {
        "raw": raw,
        "normalized": normalized,
        "clahe": clahe,
        "gamma": gamma,
    }


def run(object_path, out_dir, dark_path=None, flat_path=None, prefix=""):
    """End-to-end pipeline: load, filter frames, process, save."""
    out_dir = Path(out_dir)
    prefix_ = f"{prefix}_" if prefix else ""

    # Load the object stack and pick out the real frames
    obj = load_tiff(object_path)
    print(f"[load] object: {obj.shape}")
    frames, kept = find_usable_frames(obj)
    print(f"[frames] using {len(kept)} of {len(obj)} frames: {kept}")

    # Load dark / flat if provided
    dark = load_tiff(dark_path) if dark_path else None
    flat = load_tiff(flat_path) if flat_path else None
    if dark is not None:
        print(f"[load] dark: {dark.shape}")
        if (dark == 0).all():
            print("       WARNING: dark is all zeros -- subtraction is a no-op")
    if flat is not None:
        print(f"[load] flat: {flat.shape}")

    # Mean-stack: averaging N frames reduces shot noise by sqrt(N).
    # With ~7 frames, the mean stack is ~2.6x cleaner than any single one.
    mean_frame = frames.mean(axis=0).astype(np.float32)

    # Process every individual frame plus the mean stack
    inputs = [(f"{prefix_}frame{i:02d}", frames[i]) for i in range(len(frames))]
    inputs.append((f"{prefix_}mean_stack", mean_frame))

    for name, raw in inputs:
        result = process_one(raw, dark=dark, flat=flat)
        # Save each stage. The .tiff is the 16-bit "real" output;
        # the .png is just for previewing.
        for stage in ("normalized", "clahe", "gamma"):
            save_image(result[stage], out_dir / name, stage)
        # Comparison figure for the lab notebook / Vineet
        save_comparison_figure(
            result["raw"], result["normalized"], result["clahe"], result["gamma"],
            out_dir / "figures" / f"{name}_comparison.png",
            title=f"{name}  |  pct=[{PCT_LOW},{PCT_HIGH}]  CLAHE={CLAHE_CLIP}  gamma={GAMMA}",
        )
        print(f"[save] {name}")

    print(f"\nDone. Outputs in: {out_dir.resolve()}")


# ----------------------------------------------------------------------
# Interactive mode (file pickers) -- triggered when run with no arguments
# ----------------------------------------------------------------------
def pick_files():
    """Open native file dialogs to choose inputs and output folder."""
    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)

    print("=" * 60)
    print("  HEATER Lab -- X-ray Preprocessing")
    print("=" * 60)
    print("\nA file picker will open for each prompt.")
    print("Click Cancel on optional prompts to skip.\n")

    obj = filedialog.askopenfilename(
        title="Select OBJECT TIFF (required)",
        filetypes=[("TIFF", "*.tif *.tiff"), ("All", "*.*")])
    if not obj:
        print("No object file selected. Exiting.")
        sys.exit(1)

    dark = filedialog.askopenfilename(
        title="Select DARK TIFF (optional, Cancel to skip)",
        filetypes=[("TIFF", "*.tif *.tiff"), ("All", "*.*")]) or None

    flat = filedialog.askopenfilename(
        title="Select FLAT TIFF (optional, Cancel to skip)",
        filetypes=[("TIFF", "*.tif *.tiff"), ("All", "*.*")]) or None

    out = filedialog.askdirectory(title="Select OUTPUT folder") or "outputs"
    root.destroy()

    print(f"  object: {obj}")
    print(f"  dark:   {dark or '(none)'}")
    print(f"  flat:   {flat or '(none)'}")
    print(f"  out:    {out}\n")
    return obj, dark, flat, out


if __name__ == "__main__":
    # If no CLI args, pop up file pickers
    if len(sys.argv) == 1:
        obj, dark, flat, out = pick_files()
        run(obj, out, dark_path=dark, flat_path=flat)
    else:
        ap = argparse.ArgumentParser(
            description="X-ray preprocessing pipeline (HEATER Lab).")
        ap.add_argument("--object", required=True, help="Object TIFF stack")
        ap.add_argument("--dark",   default=None,  help="Dark TIFF (optional)")
        ap.add_argument("--flat",   default=None,  help="Flat TIFF (optional)")
        ap.add_argument("--out",    default="outputs", help="Output folder")
        ap.add_argument("--prefix", default="",    help="Output filename prefix")
        a = ap.parse_args()
        run(a.object, a.out, dark_path=a.dark, flat_path=a.flat, prefix=a.prefix)
