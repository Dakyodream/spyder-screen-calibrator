#!/usr/bin/env python3
"""
run_calibration.py — Command-line entry point for SpyderCheckr Screen Calibrator.

Usage
-----
  python run_calibration.py [--passes N] [--output DIR]

Options
-------
  --passes N      Number of capture+correction passes (default: 3)
  --output DIR    Directory for captures and ICC profile (default: ./output)
  --no-gui        Run headless (capture → compute → save ICC, no Tkinter window)
                  Useful for scripting. Requires --image <path> in headless mode.
  --image PATH    (headless only) Path to a pre-captured photo instead of
                  triggering the camera.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Ensure the project root is in sys.path so `spyder_calibrate` is importable
# regardless of how/where the script is invoked.
sys.path.insert(0, str(Path(__file__).resolve().parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Camera-based screen calibration using a SpyderCheckr 24",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--passes", type=int, default=3, metavar="N",
                        help="Number of correction passes")
    parser.add_argument("--output", type=Path, default=Path("output"),
                        help="Output directory for captures and ICC profile")
    parser.add_argument("--no-gui", action="store_true",
                        help="Run headless (no Tkinter window)")
    parser.add_argument("--image", type=Path, default=None,
                        help="(Headless) Path to pre-captured photo (skips camera)")
    parser.add_argument("--debug", action="store_true",
                        help="Enable DEBUG logging")
    return parser.parse_args()


def run_headless(args: argparse.Namespace) -> None:
    """Single-shot headless calibration from a pre-existing photo."""
    import cv2
    import numpy as np

    from spyder_calibrate.camera import Camera
    from spyder_calibrate.correction import build_icc_profile, compute_correction_matrix
    from spyder_calibrate.detection import detect_both_halves
    from spyder_calibrate.references import ALL_PATCHES

    reference_lab = np.array(
        [(L, a, b) for (_, L, a, b) in ALL_PATCHES],
        dtype=np.float64,
    )

    if args.image is not None:
        logger.info("Loading image: %s", args.image)
        img = cv2.imread(str(args.image), cv2.IMREAD_UNCHANGED)
        if img is None:
            logger.error("Could not load image: %s", args.image)
            sys.exit(1)
        if img.dtype == np.uint16:
            img_bgr = img.astype(np.float32) / 65535.0
        else:
            img_bgr = img.astype(np.float32) / 255.0
    else:
        logger.info("Capturing from camera…")
        cam = Camera()
        cam.connect()
        cam.configure_for_calibration()
        raw_path = cam.capture(output_dir=args.output / "captures")
        img_bgr = Camera.raw_to_bgr(raw_path)
        cam.disconnect()

    logger.info("Detecting patches…")
    physical, screen = detect_both_halves(img_bgr, debug=True)

    # Save debug image if available
    if screen.debug_img is not None:
        debug_path = args.output / "debug_headless.jpg"
        args.output.mkdir(parents=True, exist_ok=True)
        import cv2 as _cv2
        _cv2.imwrite(str(debug_path), (screen.debug_img * 255).astype("uint8")
                     if screen.debug_img.dtype != np.uint8 else screen.debug_img)
        logger.info("Debug image: %s", debug_path)

    logger.info("Computing correction…")
    result = compute_correction_matrix(screen.all_bgr, reference_lab)
    logger.info("ΔE before: %.2f  |  ΔE after: %.2f",
                result.delta_e_before, result.delta_e_after)

    icc_path = args.output / "calibration.icc"
    build_icc_profile(result.matrix, icc_path)
    logger.info("ICC profile saved: %s", icc_path)
    print(f"\nDone. ICC profile: {icc_path.resolve()}")
    print(f"To apply:  colormgr import-profile {icc_path.resolve()}")
    print(f"Or:        xcalib -d :0 {icc_path.resolve()}")


def run_gui(args: argparse.Namespace) -> None:
    """Launch the full Tkinter GUI."""
    from spyder_calibrate.main import CalibrationApp
    app = CalibrationApp()
    app.run()


def main() -> None:
    args = parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    if args.no_gui:
        run_headless(args)
    else:
        run_gui(args)


if __name__ == "__main__":
    main()