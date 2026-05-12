"""
camera.py — Canon M50 MK2 camera control via the gphoto2 Python bindings.

Uses libgphoto2 directly (persistent USB session) instead of spawning
gphoto2 CLI subprocesses, which avoids gvfs interference and timeout issues.

Requires:
    pip install gphoto2
    sudo apt install libgphoto2-dev dcraw   (or rawtherapee)
"""

from __future__ import annotations

import logging
import subprocess
import tempfile
import time
from pathlib import Path

import gphoto2 as gp
import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Camera class
# ---------------------------------------------------------------------------

class Camera:
    """
    High-level wrapper around the gphoto2 Python bindings.

    Maintains a persistent USB session for the entire calibration run,
    avoiding the reconnect/gvfs-race issues that occur with CLI subprocesses.

    Usage::

        cam = Camera()
        cam.connect()
        cam.configure_for_calibration()
        raw_path = cam.capture(output_dir=Path("output"))
        img_bgr  = Camera.raw_to_bgr(raw_path)
        cam.disconnect()
    """

    def __init__(self) -> None:
        self._context: gp.Context | None = None
        self._camera:  gp.Camera  | None = None
        self.connected: bool = False
        self._tmpdir: tempfile.TemporaryDirectory | None = None

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------

    def connect(self) -> None:
        """Kill gvfs grabbers, then open a persistent libgphoto2 session."""
        self._kill_camera_grabbers()
        time.sleep(0.8)

        self._context = gp.Context()
        self._camera  = gp.Camera()
        self._camera.init(self._context)
        self.connected = True

        abilities = self._camera.get_abilities()
        logger.info("Connected to: %s", abilities.model)

    def disconnect(self) -> None:
        """Close the libgphoto2 session."""
        if self._camera is not None:
            try:
                self._camera.exit(self._context)
            except Exception:
                pass
            self._camera  = None
            self._context = None
        self.connected = False
        if self._tmpdir is not None:
            self._tmpdir.cleanup()
            self._tmpdir = None

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    def configure_for_calibration(self) -> None:
        """
        Set camera to a deterministic state for colorimetric capture:
          - Manual exposure (M)
          - ISO 800  (indoor natural light)
          - Shutter 1/10 s
          - Aperture f/5.6
          - White balance: Daylight
          - Picture style: Neutral
          - Capture target: Internal RAM (sdram) for reliable gphoto2 download
          - Image format: RAW (CR3)
        """
        self._assert_connected()
        settings = {
            "/main/capturesettings/autoexposuremode": "Manual",
            "/main/imgsettings/iso":                  "1600",
            "/main/capturesettings/shutterspeed":     "1/8",
            "/main/capturesettings/aperture":         "5.6",
            "/main/imgsettings/whitebalance":         "Auto",
            "/main/capturesettings/picturestyle":     "Neutral",
            "/main/settings/capturetarget":           "Internal RAM",
            "/main/imgsettings/imageformat":          "RAW",
        }
        config = self._camera.get_config(self._context)
        for key, value in settings.items():
            try:
                widget = _find_widget(config, key)
                widget.set_value(value)
                logger.debug("Set %s = %s", key, value)
            except Exception as exc:
                logger.warning("Could not set %s to %s: %s", key, value, exc)

        self._camera.set_config(config, self._context)
        logger.info("Camera configured for calibration.")

    def set_exposure(self, iso: str, shutter: str, aperture: str) -> None:
        """Manually override exposure triangle (values as strings, e.g. '400', '1/60', '4')."""
        self._assert_connected()
        config = self._camera.get_config(self._context)
        for key, value in [
            ("/main/imgsettings/iso", iso),
            ("/main/capturesettings/shutterspeed", shutter),
            ("/main/capturesettings/aperture", aperture),
        ]:
            try:
                _find_widget(config, key).set_value(value)
            except Exception as exc:
                logger.warning("Could not set %s: %s", key, exc)
        self._camera.set_config(config, self._context)

    # ------------------------------------------------------------------
    # Capture
    # ------------------------------------------------------------------

    def capture(self, output_dir: Path | None = None) -> Path:
        """
        Trigger a capture and download the RAW file.

        Uses the persistent libgphoto2 session — no subprocess, no gvfs race.

        Returns
        -------
        Path to the downloaded .cr3 file.
        """
        self._assert_connected()

        if output_dir is None:
            if self._tmpdir is None:
                self._tmpdir = tempfile.TemporaryDirectory(prefix="ssc_")
            output_dir = Path(self._tmpdir.name)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Triggering capture…")
        file_path = self._camera.capture(gp.GP_CAPTURE_IMAGE, self._context)
        logger.info("Captured: %s%s", file_path.folder, file_path.name)

        # Download from camera RAM
        camera_file = gp.CameraFile()
        self._camera.file_get(
            file_path.folder,
            file_path.name,
            gp.GP_FILE_TYPE_NORMAL,
            camera_file,
            self._context,
        )

        local_path = output_dir / file_path.name
        camera_file.save(str(local_path))
        logger.info("Saved to: %s", local_path)

        # Clean up from camera RAM (optional but keeps things tidy)
        try:
            self._camera.file_delete(file_path.folder, file_path.name, self._context)
        except Exception:
            pass

        return local_path

    # ------------------------------------------------------------------
    # RAW conversion
    # ------------------------------------------------------------------

    @staticmethod
    def raw_to_bgr(raw_path: Path, max_width: int = 2048) -> np.ndarray:
        """
        Convert a RAW file (including CR3) to a BGR float32 numpy array [0, 1].

        Uses rawpy (libraw) as primary converter — supports CR3 from Canon M50 MK2.
        Falls back to rawtherapee-cli if rawpy fails.
        """
        import cv2
        raw_path = Path(raw_path)

        # --- Primary: rawpy (libraw) ---
        try:
            import rawpy
            with rawpy.imread(str(raw_path)) as raw:
                # postprocess returns uint16 RGB in sRGB
                rgb = raw.postprocess(
                    use_camera_wb=True,        # use in-camera white balance
                    output_color=rawpy.ColorSpace.sRGB,
                    output_bps=16,             # 16-bit output
                    no_auto_bright=True,       # keep linear response
                    gamma=(1, 1),              # linear — no gamma encoding yet
                )
            logger.debug("rawpy: shape=%s dtype=%s min=%d max=%d",
                         rgb.shape, rgb.dtype, rgb.min(), rgb.max())

            # Convert RGB → BGR, normalise to float32 [0, 1]
            # rawpy with gamma=(1,1) gives linear light — re-apply sRGB gamma
            # so that colours match what the screen displays.
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            bgr = bgr.astype(np.float32) / 65535.0
            # sRGB gamma encoding (linear → display-referred)
            bgr = np.where(bgr <= 0.0031308,
                           bgr * 12.92,
                           1.055 * np.power(np.clip(bgr, 1e-9, 1.0), 1.0/2.4) - 0.055)
            bgr = np.clip(bgr, 0.0, 1.0)

        except Exception as exc:
            logger.warning("rawpy failed (%s) — falling back to rawtherapee-cli.", exc)
            tiff_path = raw_path.with_suffix(".tiff")
            _run_rawtherapee(raw_path, tiff_path)
            img = cv2.imread(str(tiff_path), cv2.IMREAD_UNCHANGED)
            if img is None:
                raise RuntimeError(f"Could not load TIFF from rawtherapee: {tiff_path}")
            bgr = img.astype(np.float32) / (65535.0 if img.dtype == np.uint16 else 255.0)
            tiff_path.unlink(missing_ok=True)

        # Sanity check
        if bgr.max() < 0.001:
            raise RuntimeError(
                f"Converted image is entirely black (max={bgr.max():.6f}). "
                "Check that the CR3 file is valid and not corrupted."
            )

        # Downscale if needed
        h, w = bgr.shape[:2]
        if w > max_width:
            scale = max_width / w
            bgr = cv2.resize(bgr, (max_width, int(h * scale)), interpolation=cv2.INTER_AREA)

        logger.debug("raw_to_bgr: final shape=%s min=%.4f max=%.4f", bgr.shape, bgr.min(), bgr.max())
        return bgr

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _assert_connected(self) -> None:
        if not self.connected or self._camera is None:
            raise RuntimeError("Camera not connected. Call connect() first.")

    @staticmethod
    def _kill_camera_grabbers() -> None:
        """Kill gvfs processes that grab the USB device."""
        for proc in ("gvfs-gphoto2-volume-monitor", "gvfsd-gphoto2"):
            subprocess.run(["pkill", "-f", proc], capture_output=True)
        time.sleep(0.3)


# ---------------------------------------------------------------------------
# Widget helper
# ---------------------------------------------------------------------------

def _find_widget(config: gp.CameraWidget, key: str) -> gp.CameraWidget:
    """
    Find a config widget by full path or by name.
    Raises gp.GPhoto2Error if not found.
    """
    # Try full path first (e.g. /main/imgsettings/iso)
    name = key.split("/")[-1]
    try:
        return config.get_child_by_name(name)
    except gp.GPhoto2Error:
        pass
    # Walk the tree
    for i in range(config.count_children()):
        child = config.get_child(i)
        try:
            return child.get_child_by_name(name)
        except gp.GPhoto2Error:
            pass
    raise gp.GPhoto2Error(gp.GP_ERROR, f"Widget '{key}' not found in camera config")


# ---------------------------------------------------------------------------
# RAW converter helpers
# ---------------------------------------------------------------------------

def _run_dcraw(raw_path: Path, tiff_path: Path) -> bool:
    check = subprocess.run(["which", "dcraw"], capture_output=True)
    if check.returncode != 0:
        logger.warning("dcraw not found, trying rawtherapee-cli.")
        return False

    result = subprocess.run(
        ["dcraw", "-v", "-w", "-o", "1", "-q", "3", "-T", "-6", str(raw_path)],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        logger.error("dcraw failed: %s", result.stderr)
        return False

    dcraw_out = raw_path.with_suffix(".tiff")
    if dcraw_out != tiff_path and dcraw_out.exists():
        dcraw_out.rename(tiff_path)
    return tiff_path.exists()


def _run_rawtherapee(raw_path: Path, tiff_path: Path) -> bool:
    check = subprocess.run(["which", "rawtherapee-cli"], capture_output=True)
    if check.returncode != 0:
        raise RuntimeError(
            "Neither dcraw nor rawtherapee-cli found.\n"
            "  sudo apt install dcraw\n"
            "  sudo apt install rawtherapee"
        )
    out_dir = tiff_path.parent
    result = subprocess.run(
        ["rawtherapee-cli", "-o", str(out_dir), "-t", "-b16", "-c", str(raw_path)],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"rawtherapee-cli failed: {result.stderr}")
    for ext in (".tif", ".tiff"):
        candidate = out_dir / (raw_path.stem + ext)
        if candidate.exists():
            if candidate != tiff_path:
                candidate.rename(tiff_path)
            return True
    return False
