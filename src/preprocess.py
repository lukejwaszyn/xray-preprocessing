"""
X-ray preprocessing pipeline for dry-ice sublimation imaging.

Current stage: percentile normalization + contrast/dynamic-range enhancement.
Flat-field and super-resolution stages are stubbed for later addition.

Author:  Luke Waszyn
Lab:     Penn State HEATER Lab -- Experimental Frost Setup
Context: Proof-of-concept for AI/ML rendering of frost formation from sparse
         X-ray imaging of sublimating dry ice.
"""

from __future__ import annotations

import sys
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import tifffile
from skimage import exposure, img_as_ubyte, img_as_uint
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class PreprocConfig:
    """Knobs for the preprocessing pipeline. Tweak here, not in the code."""
    # Percentile clipping for normalization -- kills hot/dead pixels before
    # stretching. 1-99 is a conservative default for noisy X-ray data.
    pct_low: float = 1.0
    pct_high: float = 99.0

    # CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe_clip_limit: float = 0.01   # 0.01-0.03 typical; higher = more contrast, more noise
    clahe_kernel_divisor: int = 8    # kernel size = image_dim // divisor

    # Gamma correction (applied to percentile-normalized image as alternative view)
    gamma: float = 0.6               # <1 brightens dark regions, >1 darkens

    # Frame selection: pages 0-6 have real data in object1_hole.tiff; page 7
    # is partial; 8+ are empty. Adjust based on your actual file.
    usable_frames: tuple[int, int] = (0, 7)   # [start, stop)


# ---------------------------------------------------------------------------
# IO
# ---------------------------------------------------------------------------

def load_tiff_stack(path: str | Path) -> np.ndarray:
    """Load a multi-page TIFF as a (N, H, W) array."""
    arr = tifffile.imread(str(path))
    if arr.ndim == 2:
        arr = arr[None, ...]
    return arr


def select_usable_frames(stack: np.ndarray, start: int, stop: int) -> np.ndarray:
    """Slice a known-good range out of the stack."""
    return stack[start:stop]


def auto_detect_usable_frames(
    stack: np.ndarray,
    min_nonzero_pct: float = 50.0,
) -> tuple[np.ndarray, list[int]]:
    """
    Automatically find frames with real data by filtering out pages that are
    mostly zero. Returns (filtered_stack, list_of_original_indices_kept).

    A frame is kept if more than `min_nonzero_pct` percent of its pixels are
    nonzero. Default 50% comfortably separates real frames (~97% nonzero in
    the test data) from partial/empty ones (<90%).
    """
    if stack.ndim == 2:
        stack = stack[None, ...]

    kept_indices = []
    for i in range(len(stack)):
        nz_pct = 100.0 * np.count_nonzero(stack[i]) / stack[i].size
        if nz_pct > min_nonzero_pct:
            kept_indices.append(i)

    if not kept_indices:
        raise ValueError(
            f"No usable frames found (all pages have <={min_nonzero_pct}% "
            f"nonzero pixels). Check your TIFF file."
        )

    return stack[kept_indices], kept_indices


def parse_frame_range(spec: str, max_frames: int) -> tuple[int, int]:
    """
    Parse a 'start:stop' string into (start, stop) ints. Supports:
      "0:7"   -> (0, 7)
      ":5"    -> (0, 5)
      "3:"    -> (3, max_frames)
      "all"   -> (0, max_frames)
    """
    spec = spec.strip().lower()
    if spec == "all":
        return (0, max_frames)
    if ":" not in spec:
        raise ValueError(f"Frame range must be 'start:stop' or 'all', got: {spec!r}")
    start_s, stop_s = spec.split(":", 1)
    start = int(start_s) if start_s else 0
    stop = int(stop_s) if stop_s else max_frames
    if start < 0 or stop > max_frames or start >= stop:
        raise ValueError(
            f"Invalid frame range {start}:{stop} for stack with {max_frames} frames."
        )
    return (start, stop)


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

def flat_field_correct(
    raw: np.ndarray,
    flat: np.ndarray | None = None,
    dark: np.ndarray | None = None,
) -> np.ndarray:
    """
    Standard flat-field normalization: (raw - dark) / (flat - dark).

    Stubbed for now -- flat is None, and the provided dark.tiff is all zeros,
    so this currently passes raw through as float. When a real flat image
    becomes available, pass it here and the math "just works".
    """
    raw_f = raw.astype(np.float32)

    if dark is None or (dark == 0).all():
        dark_f = np.zeros_like(raw_f)
    else:
        dark_f = dark.astype(np.float32)
        # If dark is a stack, average it
        if dark_f.ndim == 3:
            dark_f = dark_f.mean(axis=0)

    if flat is None:
        # No flat available -> skip the division, return (raw - dark)
        return raw_f - dark_f

    flat_f = flat.astype(np.float32)
    if flat_f.ndim == 3:
        flat_f = flat_f.mean(axis=0)

    denom = flat_f - dark_f
    # Guard against divide-by-zero in detector dead regions
    denom = np.where(denom <= 0, 1.0, denom)
    return (raw_f - dark_f) / denom


def percentile_normalize(
    img: np.ndarray,
    pct_low: float = 1.0,
    pct_high: float = 99.0,
) -> np.ndarray:
    """
    Clip to [pct_low, pct_high] percentiles, then scale to [0, 1].

    This is what makes images like Vineet's reference look clean -- a raw
    min-max stretch gets destroyed by a handful of hot pixels near 65535.
    """
    lo = np.percentile(img, pct_low)
    hi = np.percentile(img, pct_high)
    if hi <= lo:
        # Degenerate image (all one value); return zeros to avoid NaN
        return np.zeros_like(img, dtype=np.float32)
    out = (img.astype(np.float32) - lo) / (hi - lo)
    return np.clip(out, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Contrast enhancement
# ---------------------------------------------------------------------------

def apply_clahe(img01: np.ndarray, clip_limit: float, kernel_divisor: int) -> np.ndarray:
    """CLAHE on a [0,1] float image. Returns [0,1] float."""
    h, w = img01.shape
    kernel_size = (max(8, h // kernel_divisor), max(8, w // kernel_divisor))
    return exposure.equalize_adapthist(img01, kernel_size=kernel_size, clip_limit=clip_limit)


def apply_gamma(img01: np.ndarray, gamma: float) -> np.ndarray:
    """Gamma correction on [0,1] float."""
    return exposure.adjust_gamma(img01, gamma=gamma)


# ---------------------------------------------------------------------------
# Full per-frame pipeline
# ---------------------------------------------------------------------------

def process_frame(
    raw: np.ndarray,
    cfg: PreprocConfig,
    flat: np.ndarray | None = None,
    dark: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """
    Run one frame through the pipeline.

    Returns a dict of intermediate stages so you can inspect/save any of them.
    """
    corrected = flat_field_correct(raw, flat=flat, dark=dark)
    normalized = percentile_normalize(corrected, cfg.pct_low, cfg.pct_high)
    clahe = apply_clahe(normalized, cfg.clahe_clip_limit, cfg.clahe_kernel_divisor)
    gamma = apply_gamma(normalized, cfg.gamma)
    return {
        "raw": raw,
        "corrected": corrected,
        "normalized": normalized,
        "clahe": clahe,
        "gamma": gamma,
    }


# ---------------------------------------------------------------------------
# Saving / figures
# ---------------------------------------------------------------------------

def save_stage(img01: np.ndarray, outdir: Path, stem: str) -> None:
    """Save a [0,1] float image as both 16-bit TIFF (precision) and 8-bit PNG (viewing)."""
    outdir.mkdir(parents=True, exist_ok=True)
    img16 = img_as_uint(np.clip(img01, 0, 1))
    img8 = img_as_ubyte(np.clip(img01, 0, 1))
    tifffile.imwrite(outdir / f"{stem}.tiff", img16)
    # Use imwrite via tifffile for PNG too? No -- use matplotlib to avoid deps.
    plt.imsave(outdir / f"{stem}.png", img8, cmap="gray", vmin=0, vmax=255)


def make_comparison_figure(stages: dict[str, np.ndarray], outpath: Path, title: str) -> None:
    """Multi-panel figure matching Vineet's style (colorbar, axis ticks)."""
    to_show = [
        ("Raw (auto-scaled)", stages["raw"]),
        ("Percentile-normalized", stages["normalized"]),
        ("CLAHE", stages["clahe"]),
        ("Gamma", stages["gamma"]),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    for ax, (name, img) in zip(axes.ravel(), to_show):
        im = ax.imshow(img, cmap="gray")
        ax.set_title(name)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(title, fontsize=14, fontweight="bold")
    fig.tight_layout()
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, dpi=120, bbox_inches="tight")
    plt.close(fig)


def make_vineet_style_figure(img01: np.ndarray, outpath: Path, title: str = "Normalized image") -> None:
    """Single-panel figure matching the reference image from Vineet exactly."""
    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(img01, cmap="gray", vmin=0, vmax=1)
    ax.set_title(title, fontweight="bold")
    ax.grid(True, color="k", linewidth=0.5, alpha=0.5)
    plt.colorbar(im, ax=ax)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, dpi=120, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def run(
    object_tiff: str | Path,
    outdir: str | Path,
    dark_tiff: str | Path | None = None,
    flat_tiff: str | Path | None = None,
    cfg: PreprocConfig | None = None,
    frame_mode: str = "auto",          # "auto", "config", or "start:stop" string
    prefix: str | None = None,         # optional prefix on output filenames
) -> None:
    """
    End-to-end: load, slice to usable frames, process each + mean-stack, save all.

    Frame selection:
      frame_mode="auto"    -> auto-detect frames with >50% nonzero pixels (recommended)
      frame_mode="config"  -> use cfg.usable_frames (legacy behavior)
      frame_mode="0:7"     -> explicit start:stop string
    """
    cfg = cfg or PreprocConfig()
    outdir = Path(outdir)
    stem_prefix = f"{prefix}_" if prefix else ""

    # Load object stack
    obj_stack = load_tiff_stack(object_tiff)
    print(f"[load] object: {obj_stack.shape}")

    # Select frames
    if frame_mode == "auto":
        frames, kept = auto_detect_usable_frames(obj_stack)
        print(f"[frames] auto-detected {len(kept)} usable frames: {kept}")
    elif frame_mode == "config":
        frames = select_usable_frames(obj_stack, *cfg.usable_frames)
        print(f"[frames] using config range {cfg.usable_frames}: {frames.shape[0]} frames")
    else:
        start, stop = parse_frame_range(frame_mode, len(obj_stack))
        frames = obj_stack[start:stop]
        print(f"[frames] using explicit range {start}:{stop}: {frames.shape[0]} frames")

    if len(frames) == 0:
        raise ValueError("No frames selected; nothing to process.")

    # Dark / flat
    dark = load_tiff_stack(dark_tiff) if dark_tiff else None
    flat = load_tiff_stack(flat_tiff) if flat_tiff else None
    if dark is not None:
        all_zero = bool((dark == 0).all())
        print(f"[load] dark: {dark.shape}  (all-zero: {all_zero})")
        if all_zero:
            print("       WARNING: dark frame is all zeros -- dark subtraction is a no-op.")
    if flat is not None:
        print(f"[load] flat: {flat.shape}")

    # Mean-stacked frame for better SNR
    mean_frame = frames.mean(axis=0).astype(np.float32)
    print(f"[stack] mean frame: min={mean_frame.min():.1f}, "
          f"max={mean_frame.max():.1f}, mean={mean_frame.mean():.1f}")

    # Process each frame + the mean stack
    all_inputs = [(f"{stem_prefix}frame{i:02d}", frames[i]) for i in range(len(frames))]
    all_inputs.append((f"{stem_prefix}mean_stack", mean_frame))

    for stem, raw in all_inputs:
        stages = process_frame(raw, cfg, flat=flat, dark=dark)

        # Save each stage's image
        frame_dir = outdir / stem
        for stage_name in ("normalized", "clahe", "gamma"):
            save_stage(stages[stage_name], frame_dir, stage_name)

        # Comparison figure
        make_comparison_figure(
            stages,
            outdir / "figures" / f"{stem}_comparison.png",
            title=f"{stem}  |  pct=[{cfg.pct_low},{cfg.pct_high}]  "
                  f"CLAHE clip={cfg.clahe_clip_limit}  gamma={cfg.gamma}",
        )

        # Vineet-style single panel (normalized only) for direct side-by-side
        make_vineet_style_figure(
            stages["normalized"],
            outdir / "figures" / f"{stem}_normalized_vineet_style.png",
        )

        print(f"[save] {stem}: normalized/clahe/gamma + comparison figure")

    print(f"\nDone. Outputs in: {outdir.resolve()}")


def _pick_file_gui(title: str, required: bool = True) -> str | None:
    """
    Native file picker via tkinter. Returns a path string, or None if the user
    cancels (only allowed when required=False).
    """
    import tkinter as tk
    from tkinter import filedialog, messagebox

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    path = filedialog.askopenfilename(
        title=title,
        filetypes=[("TIFF files", "*.tif *.tiff"), ("All files", "*.*")],
    )
    if not path and required:
        messagebox.showerror("No file selected", f"{title} is required. Exiting.")
        root.destroy()
        sys.exit(1)
    root.destroy()
    return path or None


def _pick_dir_gui(title: str, initial: str = ".") -> str:
    """Native directory picker for output folder."""
    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    path = filedialog.askdirectory(title=title, initialdir=initial, mustexist=False)
    root.destroy()
    return path or "outputs"


def interactive_main() -> None:
    """Run the pipeline with file-picker prompts. Ideal for non-CLI users."""
    print("=" * 60)
    print("  HEATER Lab -- X-ray Preprocessing Pipeline")
    print("  Experimental Frost Setup")
    print("=" * 60)
    print()
    print("A file picker will open for each prompt.")
    print("Close the picker (or click Cancel) on optional prompts to skip.\n")

    obj = _pick_file_gui("Select OBJECT TIFF stack (required)", required=True)
    print(f"  object: {obj}")

    dark = _pick_file_gui("Select DARK TIFF (optional -- cancel to skip)", required=False)
    print(f"  dark:   {dark or '(none)'}")

    flat = _pick_file_gui("Select FLAT TIFF (optional -- cancel to skip)", required=False)
    print(f"  flat:   {flat or '(none)'}")

    out = _pick_dir_gui("Select OUTPUT folder")
    print(f"  out:    {out}\n")

    run(obj, out, dark_tiff=dark, flat_tiff=flat, frame_mode="auto")


if __name__ == "__main__":
    import argparse

    # If invoked with no CLI args, launch the interactive GUI flow.
    if len(sys.argv) == 1:
        interactive_main()
        sys.exit(0)

    p = argparse.ArgumentParser(
        description="X-ray preprocessing pipeline (HEATER Lab). "
                    "Run with no arguments for an interactive file picker.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--object", required=True, help="Path to object TIFF stack.")
    p.add_argument("--dark", default=None, help="Path to dark TIFF stack (optional).")
    p.add_argument("--flat", default=None, help="Path to flat TIFF stack (optional).")
    p.add_argument("--out", default="outputs", help="Output directory.")
    p.add_argument(
        "--frames", default="auto",
        help="Frame selection: 'auto' (detect usable), 'all', or 'start:stop' (e.g. '0:7').",
    )
    p.add_argument(
        "--prefix", default=None,
        help="Prefix for output filenames (e.g. 'run1') to prevent clobbering when "
             "processing multiple TIFFs into the same --out folder.",
    )
    args = p.parse_args()

    run(
        args.object,
        args.out,
        dark_tiff=args.dark,
        flat_tiff=args.flat,
        frame_mode=args.frames,
        prefix=args.prefix,
    )
