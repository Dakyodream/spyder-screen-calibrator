"""
calibrator.py — Real-time calibration orchestrator.

Coordinates the full multi-pass calibration loop:
  1. Show the on-screen target (CalibrationDisplay).
  2. Trigger a camera capture (Camera).
  3. Detect patch colours from both halves of the photo (detect_both_halves).
  4. Compute correction matrix (compute_correction_matrix).
  5. Update the on-screen display with the corrected colours.
  6. Repeat for N passes (default 3).
  7. Export an ICC profile.

Designed to run in a background thread so the Tkinter main loop stays
responsive.
"""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import Callable

import numpy as np

from .camera import Camera
from .correction import CorrectionResult, build_icc_profile, compute_correction_matrix
from .detection import detect_both_halves
from .references import ALL_PATCHES, COLOR_PATCHES, GRAY_PATCHES

logger = logging.getLogger(__name__)


# Full Lab reference array (32 rows: 24 color + 8 gray)
_REFERENCE_LAB = np.array(
    [(L, a, b) for (_, L, a, b) in ALL_PATCHES],
    dtype=np.float64,
)


# ---------------------------------------------------------------------------
# CalibrationSession
# ---------------------------------------------------------------------------

class CalibrationSession:
    """
    Manages a complete multi-pass calibration session.

    Parameters
    ----------
    output_dir:
        Directory where RAW captures and the final ICC profile are saved.
    n_passes:
        Number of capture + correction iterations (default: 3).
    on_progress:
        Optional callback ``(pass_number, total_passes, delta_e)`` called
        after each pass with the current ΔE improvement.
    on_complete:
        Optional callback ``(icc_path)`` called when the profile is ready.
    on_error:
        Optional callback ``(exception)`` called on any fatal error.
    """

    def __init__(
        self,
        output_dir: Path | None = None,
        n_passes: int = 3,
        on_progress: Callable[[int, int, float, float], None] | None = None,
        on_complete: Callable[[Path], None] | None = None,
        on_error: Callable[[Exception], None] | None = None,
    ) -> None:
        self.output_dir = Path(output_dir or "output")
        self.n_passes = n_passes
        self.on_progress = on_progress
        self.on_complete = on_complete
        self.on_error = on_error

        self._camera = Camera()
        self._matrix: np.ndarray | None = None
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

        # Injected by main.py after the display window is created
        self.display_update_fn: Callable[[np.ndarray | None], None] | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the calibration loop in a background thread."""
        self._thread = threading.Thread(target=self._run, daemon=True, name="CalibrationLoop")
        self._thread.start()

    def stop(self) -> None:
        """Signal the calibration loop to stop gracefully."""
        self._stop_event.set()

    def wait(self, timeout: float | None = None) -> None:
        """Wait for the calibration thread to finish."""
        if self._thread is not None:
            self._thread.join(timeout=timeout)

    # ------------------------------------------------------------------
    # Main calibration loop
    # ------------------------------------------------------------------

    def _run(self) -> None:
        try:
            logger.info("=== SpyderCheckr Calibration Session Start ===")
            self.output_dir.mkdir(parents=True, exist_ok=True)
            pass_dir = self.output_dir / "captures"
            pass_dir.mkdir(exist_ok=True)

            # Connect camera
            logger.info("Connecting to camera…")
            self._camera.connect()
            self._camera.configure_for_calibration()

            # Wait a moment for the display to stabilise
            time.sleep(1.5)

            for pass_num in range(1, self.n_passes + 1):
                if self._stop_event.is_set():
                    logger.info("Calibration stopped by user.")
                    break

                logger.info("--- Pass %d / %d ---", pass_num, self.n_passes)

                # 1. Capture
                logger.info("Triggering capture…")
                raw_path = self._camera.capture(output_dir=pass_dir)

                # 2. Convert RAW → BGR
                logger.info("Converting RAW to BGR…")
                img_bgr = Camera.raw_to_bgr(raw_path)

                # 3. Detect patches
                logger.info("Detecting patches…")
                physical, screen = detect_both_halves(img_bgr, debug=True)

                # Save debug image
                if screen.debug_img is not None:
                    debug_path = pass_dir / f"debug_pass{pass_num}.jpg"
                    import cv2
                    cv2.imwrite(str(debug_path), (screen.debug_img * 255).astype("uint8"))
                    logger.info("Debug image saved: %s", debug_path)

                # 4. Compute correction from screen measurements vs reference
                logger.info("Computing correction matrix…")
                result: CorrectionResult = compute_correction_matrix(
                    measured_bgr=screen.all_bgr,
                    reference_lab=_REFERENCE_LAB,
                )

                # Accumulate matrix (compose with previous if any)
                if self._matrix is None:
                    self._matrix = result.matrix
                else:
                    # Chain: new correction applied after previous
                    self._matrix = result.matrix @ self._matrix

                logger.info(
                    "Pass %d complete — ΔE before: %.2f, after: %.2f",
                    pass_num, result.delta_e_before, result.delta_e_after,
                )

                # 5. Update display
                if self.display_update_fn is not None:
                    self.display_update_fn(self._matrix)

                # 6. Report progress
                if self.on_progress:
                    self.on_progress(pass_num, self.n_passes,
                                     result.delta_e_before, result.delta_e_after)

                # Short pause before next pass
                if pass_num < self.n_passes:
                    time.sleep(2.0)

            # 7. Build ICC profile
            if self._matrix is not None:
                icc_path = self.output_dir / "calibration.icc"
                build_icc_profile(self._matrix, icc_path)
                logger.info("ICC profile ready: %s", icc_path)
                if self.on_complete:
                    self.on_complete(icc_path)
            else:
                logger.warning("No correction matrix computed — ICC profile not generated.")

        except Exception as exc:
            logger.exception("Calibration session error: %s", exc)
            if self.on_error:
                self.on_error(exc)
        finally:
            self._camera.disconnect()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def current_matrix(self) -> np.ndarray | None:
        return self._matrix
