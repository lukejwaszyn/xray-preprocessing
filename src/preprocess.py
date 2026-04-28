"""
X-ray preprocessing for dry-ice sublimation imaging.

Author: Luke Waszyn
Lab:    Penn State HEATER Lab, Experimental Frost Setup

Run with no args for a file picker:
    python preprocess.py
Or pass paths directly:
    python preprocess.py --object file.tiff --dark dark.tiff --out outputs
"""

import sys
import argparse
from pathlib import Path

import numpy as np
import tifffile
from skimage import exposure, img_as_ubyte, img_as_uint
import matplotlib.pyplot as plt


# Tweak these if needed.
PCT_LOW    = 1.0
PCT_HIGH   = 99.0
CLAHE_CLIP = 0.01
GAMMA      = 0.6


def load_tiff(path):
    arr = tifffile.imread(str(path))
    if arr.ndim == 2:
        arr = arr[None, ...]
    return arr


def find_usable_frames(stack, min_nonzero_pct=50):
    # Drop pages that are mostly empty (failed captures save blank frames).
    keep = [i for i in range(len(stack))
            if 100 * np.count_nonzero(stack[i]) / stack[i].size > min_nonzero_pct]
    if not keep:
        raise ValueError("No usable frames.")
    return stack[keep], keep


def flat_field(raw, dark=None, flat=None):
    # (raw - dark) / (flat - dark), with graceful fallback if either is missing.
    raw = raw.astype(np.float32)
    if dark is not None:
        if dark.ndim == 3:
            dark = dark.mean(axis=0)
        raw = raw - dark.astype(np.float32)
    if flat is not None:
        if flat.ndim == 3:
            flat = flat.mean(axis=0)
        flat = flat.astype(np.float32)
        if dark is not None:
            flat = flat - dark.astype(np.float32)
        flat[flat <= 0] = 1.0
        raw = raw / flat
    return raw


def percentile_normalize(img, pct_low=1.0, pct_high=99.0):
    # Stretch [pct_low, pct_high] percentiles to [0, 1]. Clips hot/dead pixels.
    lo = np.percentile(img, pct_low)
    hi = np.percentile(img, pct_high)
    if hi <= lo:
        return np.zeros_like(img, dtype=np.float32)
    return np.clip((img.astype(np.float32) - lo) / (hi - lo), 0.0, 1.0)


def apply_clahe(img, clip_limit=0.01):
    h, w = img.shape
    kernel = (max(8, h // 8), max(8, w // 8))
    return exposure.equalize_adapthist(img, kernel_size=kernel, clip_limit=clip_limit)


def apply_gamma(img, gamma=0.6):
    return exposure.adjust_gamma(img, gamma=gamma)


def save_image(img, outdir, name):
    # 16-bit TIFF for ML, 8-bit PNG for viewing.
    outdir.mkdir(parents=True, exist_ok=True)
    img = np.clip(img, 0, 1)
    tifffile.imwrite(outdir / f"{name}.tiff", img_as_uint(img))
    plt.imsave(outdir / f"{name}.png", img_as_ubyte(img), cmap="gray", vmin=0, vmax=255)


def save_comparison_figure(raw, normalized, clahe, gamma, outpath, title):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    panels = [("Raw", raw), ("Normalized", normalized),
              ("CLAHE", clahe), ("Gamma", gamma)]
    for ax, (label, img) in zip(axes.flat, panels):
        im = ax.imshow(img, cmap="gray")
        ax.set_title(label)
        plt.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle(title, fontweight="bold")
    fig.tight_layout()
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, dpi=120, bbox_inches="tight")
    plt.close(fig)


def process_one(raw, dark=None, flat=None):
    corrected  = flat_field(raw, dark=dark, flat=flat)
    normalized = percentile_normalize(corrected, PCT_LOW, PCT_HIGH)
    clahe      = apply_clahe(normalized, CLAHE_CLIP)
    gamma      = apply_gamma(normalized, GAMMA)
    return {"raw": raw, "normalized": normalized, "clahe": clahe, "gamma": gamma}


def run(object_path, out_dir, dark_path=None, flat_path=None, prefix=""):
    out_dir = Path(out_dir)
    prefix_ = f"{prefix}_" if prefix else ""

    obj = load_tiff(object_path)
    print(f"[load] object: {obj.shape}")
    frames, kept = find_usable_frames(obj)
    print(f"[frames] using {len(kept)} of {len(obj)}: {kept}")

    dark = load_tiff(dark_path) if dark_path else None
    flat = load_tiff(flat_path) if flat_path else None
    if dark is not None:
        print(f"[load] dark: {dark.shape}")
        if (dark == 0).all():
            print("       WARNING: dark all zeros, subtraction is a no-op")
    if flat is not None:
        print(f"[load] flat: {flat.shape}")

    # Mean-stack reduces shot noise by sqrt(N).
    mean_frame = frames.mean(axis=0).astype(np.float32)

    inputs = [(f"{prefix_}frame{i:02d}", frames[i]) for i in range(len(frames))]
    inputs.append((f"{prefix_}mean_stack", mean_frame))

    for name, raw in inputs:
        result = process_one(raw, dark=dark, flat=flat)
        for stage in ("normalized", "clahe", "gamma"):
            save_image(result[stage], out_dir / name, stage)
        save_comparison_figure(
            result["raw"], result["normalized"], result["clahe"], result["gamma"],
            out_dir / "figures" / f"{name}_comparison.png",
            title=f"{name} | pct=[{PCT_LOW},{PCT_HIGH}] CLAHE={CLAHE_CLIP} gamma={GAMMA}")
        print(f"[save] {name}")

    print(f"\nDone. Outputs in: {out_dir.resolve()}")


def pick_files():
    import tkinter as tk
    from tkinter import filedialog
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)

    print("HEATER Lab X-ray Preprocessing\n")
    obj = filedialog.askopenfilename(title="OBJECT TIFF (required)",
        filetypes=[("TIFF", "*.tif *.tiff"), ("All", "*.*")])
    if not obj:
        print("No object file selected.")
        sys.exit(1)
    dark = filedialog.askopenfilename(title="DARK TIFF (optional)",
        filetypes=[("TIFF", "*.tif *.tiff"), ("All", "*.*")]) or None
    flat = filedialog.askopenfilename(title="FLAT TIFF (optional)",
        filetypes=[("TIFF", "*.tif *.tiff"), ("All", "*.*")]) or None
    out = filedialog.askdirectory(title="OUTPUT folder") or "outputs"
    root.destroy()

    print(f"  object: {obj}")
    print(f"  dark:   {dark or '(none)'}")
    print(f"  flat:   {flat or '(none)'}")
    print(f"  out:    {out}\n")
    return obj, dark, flat, out


if __name__ == "__main__":
    if len(sys.argv) == 1:
        obj, dark, flat, out = pick_files()
        run(obj, out, dark_path=dark, flat_path=flat)
    else:
        ap = argparse.ArgumentParser(description="X-ray preprocessing (HEATER Lab).")
        ap.add_argument("--object", required=True)
        ap.add_argument("--dark",   default=None)
        ap.add_argument("--flat",   default=None)
        ap.add_argument("--out",    default="outputs")
        ap.add_argument("--prefix", default="")
        a = ap.parse_args()
        run(a.object, a.out, dark_path=a.dark, flat_path=a.flat, prefix=a.prefix)
