"""
camera.py — Canon M50 MK2 (and compatible) camera control via gPhoto2.

Handles:
  - Camera detection and connection
  - RAW capture and download
  - Exposure / white-balance settings for calibration shoots
  - Live-view frame grab (optional, for focus assistance)
"""

from __future__ import annotations

import logging
import subprocess
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# gPhoto2 wrapper helpers
# ---------------------------------------------------------------------------

def _gphoto2(*args: str, check: bool = True) -> subprocess.CompletedProcess:
    """Run a gphoto2 command and return the CompletedProcess result."""
    cmd = ["gphoto2", *args]
    logger.debug("gphoto2 cmd: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    if check and result.returncode != 0:
        raise RuntimeError(
            f"gphoto2 error (rc={result.returncode}):\n"
            f"stdout: {result.stdout}\nstderr: {result.stderr}"
        )
    return result


def list_cameras() -> list[str]:
    """Return a list of detected camera model strings."""
    result = _gphoto2("--auto-detect", check=False)
    lines = result.stdout.strip().splitlines()
    # Skip header lines (first two)
    cameras = []
    for line in lines[2:]:
        line = line.strip()
        if line:
            cameras.append(line)
    return cameras


# ---------------------------------------------------------------------------
# Camera class
# ---------------------------------------------------------------------------

class Camera:
    """
    High-level wrapper around gphoto2 for calibration captures.

    Usage::

        cam = Camera()
        cam.connect()
        cam.configure_for_calibration()
        raw_path = cam.capture(output_dir=Path("output"))
        img_bgr = cam.raw_to_bgr(raw_path)
        cam.disconnect()
    """

    def __init__(self) -> None:
        self.connected: bool = False
        self._tmpdir: tempfile.TemporaryDirectory | None = None

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------

    def connect(self) -> None:
        """Detect and connect to the first available camera."""
        cameras = list_cameras()
        if not cameras:
            raise RuntimeError(
                "No camera detected. Make sure the Canon M50 MK2 is connected "
                "via USB and switched on, and that gphoto2 is installed."
            )
        logger.info("Detected cameras: %s", cameras)
        # Kill any process that may have grabbed the USB device (e.g. gvfs)
        self._kill_camera_grabbers()
        self.connected = True
        logger.info("Camera connected.")

    def disconnect(self) -> None:
        """Release resources."""
        self.connected = False
        if self._tmpdir is not None:
            self._tmpdir.cleanup()
            self._tmpdir = None

    # ------------------------------------------------------------------
    # Camera configuration
    # ------------------------------------------------------------------

    def configure_for_calibration(self) -> None:
        """
        Set camera to a deterministic state suitable for colorimetric capture:
          - Manual exposure (M)
          - Fixed ISO (100 or 200)
          - Fixed shutter speed (1/60 or 1/30)
          - Fixed aperture (f/5.6 or similar)
          - White balance: Daylight (5500 K) — keeps response predictable
          - Picture style: Neutral (flat, no sharpening / contrast)
          - Image format: RAW (CR3 for M50 MK2)
        """
        self._assert_connected()
        settings = {
            "/main/capturesettings/autoexposuremode": "Manual",
            "/main/imgsettings/iso": "100",
            "/main/capturesettings/shutterspeed": "0.0333",   # ~1/30
            "/main/capturesettings/aperture": "5.6",
            "/main/imgsettings/whitebalance": "Daylight",
            "/main/capturesettings/picturestyle": "Neutral",
            "/main/imgsettings/imageformat": "RAW",
        }
        for key, value in settings.items():
            try:
                _gphoto2("--set-config-value", f"{key}={value}")
                logger.debug("Set %s = %s", key, value)
            except RuntimeError as exc:
                # Non-fatal — some bodies use different config keys
                logger.warning("Could not set %s: %s", key, exc)

        logger.info("Camera configured for calibration.")

    def set_exposure(self, iso: int, shutter: str, aperture: str) -> None:
        """Manually override exposure triangle."""
        self._assert_connected()
        _gphoto2("--set-config-value", f"/main/imgsettings/iso={iso}")
        _gphoto2("--set-config-value", f"/main/capturesettings/shutterspeed={shutter}")
        _gphoto2("--set-config-value", f"/main/capturesettings/aperture={aperture}")

    # ------------------------------------------------------------------
    # Capture
    # ------------------------------------------------------------------

    def capture(self, output_dir: Path | None = None) -> Path:
        """
        Trigger a capture, download the RAW file and return its local path.

        Parameters
        ----------
        output_dir:
            Directory where the downloaded RAW file is saved.
            Defaults to a temporary directory managed by this instance.

        Returns
        -------
        Path
            Absolute path to the downloaded RAW file (.cr3 or .cr2).
        """
        self._assert_connected()

        if output_dir is None:
            if self._tmpdir is None:
                self._tmpdir = tempfile.TemporaryDirectory(prefix="ssc_")
            output_dir = Path(self._tmpdir.name)

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Triggering capture…")
        _gphoto2(
            "--capture-image-and-download",
            "--keep",                     # keep on camera
            "--filename", str(output_dir / "%f.%C"),
        )

        # Find the most recently downloaded RAW file
        raw_files = sorted(
            list(output_dir.glob("*.cr3")) + list(output_dir.glob("*.cr2")),
            key=lambda p: p.stat().st_mtime,
        )
        if not raw_files:
            raise RuntimeError(
                f"No RAW file found in {output_dir} after capture. "
                "Check that the camera is set to RAW output."
            )
        raw_path = raw_files[-1]
        logger.info("Downloaded RAW: %s", raw_path)
        return raw_path

    # ------------------------------------------------------------------
    # RAW conversion
    # ------------------------------------------------------------------

    @staticmethod
    def raw_to_bgr(raw_path: Path, max_width: int = 2048) -> np.ndarray:
        """
        Convert a RAW file to a linear-ish BGR numpy array using dcraw/rawtherapee-cli.

        Uses ``dcraw`` (preferred) or ``rawtherapee-cli`` as fallback.
        The output is a 16-bit TIFF loaded as uint16 then normalised to float32 [0,1].

        Parameters
        ----------
        raw_path:
            Path to the .cr3 / .cr2 file.
        max_width:
            Resize to this width if larger (keeps aspect ratio).

        Returns
        -------
        np.ndarray
            BGR float32 image, values in [0, 1].
        """
        raw_path = Path(raw_path)
        tiff_path = raw_path.with_suffix(".tiff")

        # Try dcraw first
        dcraw_ok = _run_dcraw(raw_path, tiff_path)
        if not dcraw_ok:
            _run_rawtherapee(raw_path, tiff_path)

        if not tiff_path.exists():
            raise RuntimeError(f"RAW conversion failed — no TIFF produced for {raw_path}")

        img = cv2.imread(str(tiff_path), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise RuntimeError(f"Could not load converted TIFF: {tiff_path}")

        # Normalise to float32 [0, 1]
        if img.dtype == np.uint16:
            img = img.astype(np.float32) / 65535.0
        else:
            img = img.astype(np.float32) / 255.0

        # Optional downscale
        h, w = img.shape[:2]
        if w > max_width:
            scale = max_width / w
            img = cv2.resize(img, (max_width, int(h * scale)), interpolation=cv2.INTER_AREA)

        tiff_path.unlink(missing_ok=True)
        return img

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _assert_connected(self) -> None:
        if not self.connected:
            raise RuntimeError("Camera not connected. Call connect() first.")

    @staticmethod
    def _kill_camera_grabbers() -> None:
        """Kill gvfs-gphoto2-volume-monitor which blocks USB access."""
        for proc in ("gvfs-gphoto2-volume-monitor", "gvfsd-gphoto2"):
            subprocess.run(["pkill", "-f", proc], capture_output=True)
            time.sleep(0.2)


# ---------------------------------------------------------------------------
# RAW converter helpers (module-level so they can be tested independently)
# ---------------------------------------------------------------------------

def _run_dcraw(raw_path: Path, tiff_path: Path) -> bool:
    """
    Convert raw_path → tiff_path using dcraw.
    Returns True on success, False if dcraw is not available.
    """
    check = subprocess.run(["which", "dcraw"], capture_output=True)
    if check.returncode != 0:
        logger.warning("dcraw not found, will try rawtherapee-cli.")
        return False

    result = subprocess.run(
        [
            "dcraw",
            "-v",          # verbose
            "-w",          # use camera white balance
            "-o", "1",     # output colorspace: sRGB
            "-q", "3",     # AHD demosaicing
            "-T",          # output TIFF
            "-6",          # 16-bit output
            str(raw_path),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        logger.error("dcraw failed: %s", result.stderr)
        return False

    # dcraw names output as <rawname>.tiff in the same directory
    dcraw_out = raw_path.with_suffix(".tiff")
    if dcraw_out != tiff_path and dcraw_out.exists():
        dcraw_out.rename(tiff_path)
    return tiff_path.exists()


def _run_rawtherapee(raw_path: Path, tiff_path: Path) -> bool:
    """
    Convert raw_path → tiff_path using rawtherapee-cli as fallback.
    Returns True on success.
    """
    check = subprocess.run(["which", "rawtherapee-cli"], capture_output=True)
    if check.returncode != 0:
        raise RuntimeError(
            "Neither dcraw nor rawtherapee-cli is available. "
            "Install one of them:\n"
            "  sudo apt install dcraw\n"
            "  sudo apt install rawtherapee"
        )

    out_dir = tiff_path.parent
    result = subprocess.run(
        [
            "rawtherapee-cli",
            "-o", str(out_dir),
            "-t",           # output TIFF
            "-b16",         # 16-bit
            "-c", str(raw_path),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"rawtherapee-cli failed: {result.stderr}")

    # rawtherapee outputs <name>.tif or <name>.tiff
    for ext in (".tif", ".tiff"):
        candidate = out_dir / (raw_path.stem + ext)
        if candidate.exists():
            if candidate != tiff_path:
                candidate.rename(tiff_path)
            return True
    return False
