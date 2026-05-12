"""
calibrator.py — Screen WB calibration orchestrator.

Approach
--------
A camera is a reliable colorimeter for NEUTRAL GRAYS but not for saturated
colours (camera-screen metamerism, backlight bleed on dark patches).

For each pass the session:
  1. Displays each of the 4 brightest neutral patches fullscreen.
  2. Captures a RAW frame and samples the centre region.
  3. Computes a per-channel diagonal WB correction from those 4 measurements,
     normalised to the white point to decouple exposure from colour error.

Result: an ICC display profile that corrects the screen's white-point drift
(colour temperature error) without trying to correct gamut or tone — which
requires a hardware colorimeter.
"""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import Callable

import numpy as np

from .camera import Camera
from .correction import CorrectionResult, apply_matrix_to_srgb, build_icc_profile, compute_wb_correction
from .detection import sample_fullscreen_center
from .references import NEUTRAL_PATCHES_SRGB

logger = logging.getLogger(__name__)


class CalibrationSession:
    """
    Multi-pass screen WB calibration session.

    Only the 4 brightest neutral patches are captured (White, Neutral 8,
    Neutral 6.5, Neutral 5).  This keeps the total capture count low
    (4 × n_passes) and avoids the backlight-bleed artefacts that corrupt
    dark-patch measurements.

    Parameters
    ----------
    output_dir:
        Directory where captures and the ICC profile are saved.
    n_passes:
        Number of capture+correction iterations (default: 3).
    on_patch_progress:
        Callback ``(pass_num, patch_idx, n_patches, overall_pct, patch_name)``.
    on_pass_complete:
        Callback ``(pass_num, total_passes, de_before, de_after)``.
    on_complete:
        Callback ``(icc_path)`` when the profile is ready.
    on_error:
        Callback ``(exception)`` on any fatal error.
    """

    def __init__(
        self,
        output_dir: Path | None = None,
        n_passes: int = 3,
        on_patch_progress: Callable[[int, int, int, int, str], None] | None = None,
        on_pass_complete: Callable[[int, int, float, float], None] | None = None,
        on_complete: Callable[[Path], None] | None = None,
        on_error: Callable[[Exception], None] | None = None,
    ) -> None:
        self.output_dir = Path(output_dir or "output")
        self.n_passes = n_passes
        self.on_patch_progress = on_patch_progress
        self.on_pass_complete = on_pass_complete
        self.on_complete = on_complete
        self.on_error = on_error

        self._camera = Camera()
        self._matrix: np.ndarray | None = None
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

        self.display_ref = None  # CalibrationDisplay instance, injected by main.py

    @property
    def current_matrix(self) -> np.ndarray | None:
        return self._matrix

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True, name="CalibrationLoop")
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()

    def wait(self, timeout: float | None = None) -> None:
        if self._thread is not None:
            self._thread.join(timeout=timeout)

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def _run(self) -> None:
        try:
            logger.info("=== WB Calibration Session Start ===")
            logger.info("Using %d neutral patches × %d passes", len(NEUTRAL_PATCHES_SRGB), self.n_passes)
            self.output_dir.mkdir(parents=True, exist_ok=True)
            pass_dir = self.output_dir / "captures"
            pass_dir.mkdir(exist_ok=True)

            logger.info("Connecting to camera…")
            self._camera.connect()
            self._camera.configure_for_calibration()
            time.sleep(1.0)

            n_patches = len(NEUTRAL_PATCHES_SRGB)
            total_captures = self.n_passes * n_patches

            for pass_num in range(1, self.n_passes + 1):
                if self._stop_event.is_set():
                    logger.info("Calibration stopped by user.")
                    break

                logger.info("--- Pass %d / %d ---", pass_num, self.n_passes)
                measured_bgr: list[tuple[float, float, float]] = []

                for patch_idx, (name, r_ref, g_ref, b_ref) in enumerate(NEUTRAL_PATCHES_SRGB):
                    if self._stop_event.is_set():
                        break

                    # Apply accumulated correction to the displayed colour
                    if self._matrix is not None:
                        r_disp, g_disp, b_disp = apply_matrix_to_srgb(
                            (r_ref, g_ref, b_ref), self._matrix
                        )
                    else:
                        r_disp, g_disp, b_disp = r_ref, g_ref, b_ref

                    logger.debug("Pass %d  patch %d/%d — %s  display(%d,%d,%d)",
                                 pass_num, patch_idx + 1, n_patches, name,
                                 r_disp, g_disp, b_disp)

                    # Show colour fullscreen
                    if self.display_ref is not None:
                        self.display_ref.set_solid_color(r_disp, g_disp, b_disp)
                    time.sleep(0.5)  # let screen and AWB settle

                    # Capture and sample centre
                    raw_path = self._camera.capture(output_dir=pass_dir)
                    img_bgr = Camera.raw_to_bgr(raw_path)
                    measured = sample_fullscreen_center(img_bgr)
                    measured_bgr.append(measured)

                    logger.debug("  measured BGR: (%.1f, %.1f, %.1f)", *measured)

                    done = (pass_num - 1) * n_patches + patch_idx + 1
                    pct = int(done / total_captures * 100)
                    if self.on_patch_progress:
                        self.on_patch_progress(pass_num, patch_idx + 1, n_patches, pct, name)

                if self._stop_event.is_set() or len(measured_bgr) < n_patches:
                    break

                # Compute diagonal WB correction
                result: CorrectionResult = compute_wb_correction(
                    measured_bgr=measured_bgr,
                    reference_srgb=[(r, g, b) for (_, r, g, b) in NEUTRAL_PATCHES_SRGB],
                )

                # Compose with previous pass
                if self._matrix is None:
                    self._matrix = result.matrix
                else:
                    self._matrix = result.matrix @ self._matrix

                logger.info(
                    "Pass %d — ΔE before: %.2f  after: %.2f",
                    pass_num, result.delta_e_before, result.delta_e_after,
                )

                if self.on_pass_complete:
                    self.on_pass_complete(pass_num, self.n_passes,
                                          result.delta_e_before, result.delta_e_after)

                if pass_num < self.n_passes:
                    time.sleep(1.0)

            # Switch display to comparison grid
            if self.display_ref is not None:
                self.display_ref.show_comparison(self._matrix)

            # Build ICC profile
            if self._matrix is not None:
                icc_path = self.output_dir / "calibration.icc"
                build_icc_profile(self._matrix, icc_path)
                logger.info("ICC profile ready: %s", icc_path)
                if self.on_complete:
                    self.on_complete(icc_path)
            else:
                logger.warning("No correction matrix computed.")

        except Exception as exc:
            logger.exception("Calibration error: %s", exc)
            if self.on_error:
                self.on_error(exc)
        finally:
            self._camera.disconnect()
