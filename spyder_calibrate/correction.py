"""
correction.py — Colorimetric correction engine.

Steps
-----
1. Convert all measured BGR values to CIE Lab (D50).
2. Compute a least-squares 3×3 linear correction matrix that maps
   *measured screen Lab* → *reference Lab* for all 32 patches.
3. Optionally iterate (re-display corrected patches, re-shoot, refine).
4. Build a minimal ICC display profile embedding the correction as a
   matrix + TRC (tone reproduction curve) using the ``colour`` library.
   Falls back to a LUT-based approach via ``littlecms2`` (python-lcms2)
   if ``colour`` is unavailable.

The generated profile targets sRGB primaries with a custom white point and
matrix, making it suitable for loading with ``colord`` or ``xcalib``.
"""

from __future__ import annotations

import logging
import struct
from datetime import datetime
from pathlib import Path
from typing import NamedTuple

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Colour space conversion helpers
# ---------------------------------------------------------------------------

# sRGB → XYZ (D65) standard matrix
_SRGB_TO_XYZ_D65 = np.array([
    [0.4124564, 0.3575761, 0.1804375],
    [0.2126729, 0.7151522, 0.0721750],
    [0.0193339, 0.1191920, 0.9503041],
], dtype=np.float64)

# Bradford chromatic adaptation D65 → D50
_D65_TO_D50 = np.array([
    [ 1.0478112,  0.0228866, -0.0501270],
    [ 0.0295424,  0.9904844, -0.0170491],
    [-0.0092345,  0.0150436,  0.9866797],
], dtype=np.float64)

_SRGB_TO_XYZ_D50 = _D65_TO_D50 @ _SRGB_TO_XYZ_D65

# D50 white point XYZ
_D50_XYZ = np.array([0.96422, 1.00000, 0.82521])


def _srgb_to_linear(v: np.ndarray) -> np.ndarray:
    """Apply sRGB inverse gamma to get linear light values."""
    v = np.clip(v, 0.0, 1.0)
    return np.where(v <= 0.04045, v / 12.92, ((v + 0.055) / 1.055) ** 2.4)


def _linear_to_srgb(v: np.ndarray) -> np.ndarray:
    """Apply sRGB gamma encoding."""
    v = np.clip(v, 0.0, 1.0)
    return np.where(v <= 0.0031308, v * 12.92, 1.055 * v ** (1.0 / 2.4) - 0.055)


def bgr_uint8_to_lab(bgr_list: list[tuple[float, float, float]]) -> np.ndarray:
    """
    Convert a list of uint8 BGR tuples (0-255) to CIE Lab (D50).

    Returns
    -------
    np.ndarray of shape (N, 3), dtype float64
    """
    result = []
    for b, g, r in bgr_list:
        rgb = np.array([r, g, b], dtype=np.float64) / 255.0
        lin = _srgb_to_linear(rgb)
        xyz = _SRGB_TO_XYZ_D50 @ lin
        lab = _xyz_to_lab(xyz, _D50_XYZ)
        result.append(lab)
    return np.array(result, dtype=np.float64)


def _xyz_to_lab(xyz: np.ndarray, white: np.ndarray) -> np.ndarray:
    f = xyz / white
    epsilon, kappa = 0.008856, 903.3
    f = np.where(f > epsilon, f ** (1.0 / 3.0), (kappa * f + 16.0) / 116.0)
    L = 116.0 * f[1] - 16.0
    a = 500.0 * (f[0] - f[1])
    b = 200.0 * (f[1] - f[2])
    return np.array([L, a, b])


# ---------------------------------------------------------------------------
# Correction matrix computation
# ---------------------------------------------------------------------------

class CorrectionResult(NamedTuple):
    matrix: np.ndarray          # 3×3 float64, maps linear sRGB → linear sRGB
    delta_e_before: float       # mean ΔE2000 before correction
    delta_e_after: float        # mean ΔE2000 after correction
    patch_errors: np.ndarray    # per-patch ΔE after correction


def compute_correction_matrix(
    measured_bgr: list[tuple[float, float, float]],
    reference_lab: np.ndarray,
) -> CorrectionResult:
    """
    Compute the least-squares 3×3 linear matrix M such that::

        M @ measured_linear_rgb ≈ reference_linear_rgb

    Parameters
    ----------
    measured_bgr:
        List of (B, G, R) float values in [0, 255] sampled from the photo
        of the *screen* patches.
    reference_lab:
        N×3 array of CIE Lab D50 reference values (from references.py).

    Returns
    -------
    CorrectionResult
    """
    n = min(len(measured_bgr), len(reference_lab))

    # Convert measured to linear RGB (R, G, B order)
    meas_linear = np.array([
        _srgb_to_linear(np.array([r, g, b], dtype=np.float64) / 255.0)
        for b, g, r in measured_bgr[:n]
    ])  # shape (N, 3)

    # Convert reference Lab → XYZ → linear sRGB
    ref_linear = np.array([
        _lab_to_linear_srgb(lab) for lab in reference_lab[:n]
    ])  # shape (N, 3)

    # ΔE before
    meas_lab_before = np.array([
        _xyz_to_lab(_SRGB_TO_XYZ_D50 @ lin, _D50_XYZ) for lin in meas_linear
    ])
    de_before = float(np.mean(_delta_e_simple(meas_lab_before, reference_lab[:n])))

    # Least-squares: solve M @ meas.T ≈ ref.T  → M = ref.T @ pinv(meas.T)
    # shape: M (3,3)
    matrix, _, _, _ = np.linalg.lstsq(meas_linear, ref_linear, rcond=None)
    matrix = matrix.T   # shape (3, 3): maps input column → output column

    # ΔE after
    corrected = (matrix @ meas_linear.T).T
    corrected_clipped = np.clip(corrected, 0.0, 1.0)
    corrected_lab = np.array([
        _xyz_to_lab(_SRGB_TO_XYZ_D50 @ lin, _D50_XYZ)
        for lin in corrected_clipped
    ])
    patch_de = _delta_e_simple(corrected_lab, reference_lab[:n])
    de_after = float(np.mean(patch_de))

    logger.info("ΔE before: %.2f  |  ΔE after: %.2f", de_before, de_after)
    return CorrectionResult(
        matrix=matrix,
        delta_e_before=de_before,
        delta_e_after=de_after,
        patch_errors=patch_de,
    )


def _lab_to_linear_srgb(lab: np.ndarray) -> np.ndarray:
    """CIE Lab (D50) → linear sRGB [0, 1]."""
    L, a, b = lab
    fy = (L + 16.0) / 116.0
    fx = a / 500.0 + fy
    fz = fy - b / 200.0

    epsilon, kappa = 0.008856, 903.3

    def f_inv(t: float) -> float:
        return t ** 3 if t ** 3 > epsilon else (116.0 * t - 16.0) / kappa

    xyz = np.array([f_inv(fx), f_inv(fy), f_inv(fz)]) * _D50_XYZ
    xyz_to_srgb_d50 = np.linalg.inv(_SRGB_TO_XYZ_D50)
    linear = xyz_to_srgb_d50 @ xyz
    return np.clip(linear, 0.0, 1.0)


def _delta_e_simple(lab_a: np.ndarray, lab_b: np.ndarray) -> np.ndarray:
    """Simplified ΔE76 (Euclidean distance in Lab space)."""
    return np.sqrt(np.sum((lab_a - lab_b) ** 2, axis=1))


# ---------------------------------------------------------------------------
# WB-only diagonal correction (reliable with a camera)
# ---------------------------------------------------------------------------

def compute_wb_correction(
    measured_bgr: list[tuple[float, float, float]],
    reference_srgb: list[tuple[int, int, int]],
) -> CorrectionResult:
    """
    Compute a per-channel diagonal correction matrix from neutral-patch measurements.

    Why diagonal / why neutral patches only
    ----------------------------------------
    A camera photographs a screen through a scene-calibrated pipeline that
    introduces cross-channel contamination for saturated colours (metamerism).
    Neutral grays are immune to this effect — their measured R:G:B ratio is a
    reliable indicator of the screen's white-point error.

    Algorithm
    ---------
    1. Convert measured and reference values to linear light.
    2. Normalise both to the white (first) patch — decouples absolute exposure
       from the colour error we actually care about.
    3. Fit a per-channel scalar gain via least-squares through the origin.
       Gain < 1  → that channel is over-represented on screen (reduce it).
    4. Normalise gains so max(gains) = 1 — ensures display values stay ≤ 255
       (an ICC display profile can only reduce channels, not boost them).
    5. Return a 3×3 diagonal matrix M such that the OS will send
       M @ colour_linear  to the GPU instead of  colour_linear.

    Parameters
    ----------
    measured_bgr:
        Camera-sampled (B, G, R) float values in [0, 255] for each neutral patch,
        in the same order as ``reference_srgb``.
    reference_srgb:
        (R, G, B) uint8 values that were displayed for each patch.
    """
    assert len(measured_bgr) == len(reference_srgb), "Patch count mismatch"

    # Convert to linear RGB [0, 1]  (BGR → RGB for measured)
    meas_lin = np.array([
        _srgb_to_linear(np.array([r, g, b], dtype=np.float64) / 255.0)
        for b, g, r in measured_bgr
    ])
    ref_lin = np.array([
        _srgb_to_linear(np.array([r, g, b], dtype=np.float64) / 255.0)
        for r, g, b in reference_srgb
    ])

    # ΔE before correction
    meas_lab = np.array([_xyz_to_lab(_SRGB_TO_XYZ_D50 @ lin, _D50_XYZ) for lin in meas_lin])
    ref_lab  = np.array([_xyz_to_lab(_SRGB_TO_XYZ_D50 @ lin, _D50_XYZ) for lin in ref_lin])
    de_before = float(np.mean(_delta_e_simple(meas_lab, ref_lab)))

    # Normalise to white point (first patch) — isolates colour error from exposure
    white_meas = meas_lin[0] + 1e-9
    white_ref  = ref_lin[0]  + 1e-9
    meas_norm  = meas_lin / white_meas   # white → (1, 1, 1)
    ref_norm   = ref_lin  / white_ref    # white → (1, 1, 1)

    # Per-channel least-squares scalar through origin:
    # gain_c = Σ(meas_c · ref_c) / Σ(meas_c²)
    gains = np.array([
        np.dot(meas_norm[:, c], ref_norm[:, c]) / (np.dot(meas_norm[:, c], meas_norm[:, c]) + 1e-9)
        for c in range(3)
    ])

    # Normalise so max gain = 1 → no channel can clip when ICC is applied
    gains = gains / (gains.max() + 1e-9)

    matrix = np.diag(gains)
    logger.info("WB gains  R=%.3f  G=%.3f  B=%.3f", *gains)

    # ΔE after: apply gains in normalised linear space, rescale to ref white
    corrected_norm = (matrix @ meas_norm.T).T
    corrected_abs  = np.clip(corrected_norm * white_ref, 0.0, 1.0)
    corr_lab  = np.array([_xyz_to_lab(_SRGB_TO_XYZ_D50 @ lin, _D50_XYZ) for lin in corrected_abs])
    patch_de  = _delta_e_simple(corr_lab, ref_lab)
    de_after  = float(np.mean(patch_de))

    logger.info("WB ΔE before: %.2f  |  ΔE after: %.2f", de_before, de_after)
    return CorrectionResult(
        matrix=matrix,
        delta_e_before=de_before,
        delta_e_after=de_after,
        patch_errors=patch_de,
    )


# ---------------------------------------------------------------------------
# Utility: apply matrix to a display patch colour
# ---------------------------------------------------------------------------

def apply_matrix_to_srgb(
    rgb_uint8: tuple[int, int, int],
    matrix: np.ndarray,
) -> tuple[int, int, int]:
    """
    Apply a linear correction matrix to an sRGB uint8 triplet.

    1. Decode gamma  (sRGB → linear)
    2. Apply 3×3 matrix
    3. Re-encode gamma (linear → sRGB)
    4. Clamp and return as uint8
    """
    r, g, b = rgb_uint8
    lin = _srgb_to_linear(np.array([r, g, b], dtype=np.float64) / 255.0)
    corrected = np.clip(matrix @ lin, 0.0, 1.0)
    enc = _linear_to_srgb(corrected)
    return (
        int(np.clip(round(enc[0] * 255), 0, 255)),
        int(np.clip(round(enc[1] * 255), 0, 255)),
        int(np.clip(round(enc[2] * 255), 0, 255)),
    )


# ---------------------------------------------------------------------------
# ICC profile generation
# ---------------------------------------------------------------------------

def build_icc_profile(matrix: np.ndarray, output_path: Path) -> Path:
    """
    Build a minimal ICC v2 matrix/shaper display profile and write it to disk.

    The profile embeds:
    - Profile class: display device
    - Color space: RGB
    - PCS: XYZ (D50)
    - Red/Green/Blue colorant tags (rXYZ, gXYZ, bXYZ)
    - Tone reproduction curves (rTRC, gTRC, bTRC) — sRGB gamma 2.2 approx
    - Media white point

    Parameters
    ----------
    matrix:
        3×3 correction matrix (linear sRGB → linear sRGB).
        We compose this with the standard sRGB→XYZ matrix to derive
        the ICC colorant primaries.
    output_path:
        Where to write the .icc file.

    Returns
    -------
    Path to the written profile.
    """
    # Composed matrix: sRGB (corrected) → XYZ D50
    composed = _SRGB_TO_XYZ_D50 @ np.linalg.inv(matrix)

    # Extract RGB primaries (columns of the composed matrix)
    r_xyz = composed[:, 0]
    g_xyz = composed[:, 1]
    b_xyz = composed[:, 2]

    tags: list[tuple[bytes, bytes]] = []

    # --- colorant tags ---
    tags.append((b"rXYZ", _xyz_tag(r_xyz)))
    tags.append((b"gXYZ", _xyz_tag(g_xyz)))
    tags.append((b"bXYZ", _xyz_tag(b_xyz)))

    # --- TRC — sRGB-like curve (256-entry LUT) ---
    trc = _srgb_trc_tag()
    tags.append((b"rTRC", trc))
    tags.append((b"gTRC", trc))
    tags.append((b"bTRC", trc))

    # --- media white point ---
    tags.append((b"wtpt", _xyz_tag(_D50_XYZ)))

    # --- description ---
    desc = "SpyderCheckr Calibration Profile"
    tags.append((b"desc", _desc_tag(desc)))

    # --- copyright ---
    tags.append((b"cprt", _text_tag("Generated by spyder-screen-calibrator (MIT)")))

    # Assemble profile bytes
    profile_bytes = _assemble_icc(tags)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(profile_bytes)
    logger.info("ICC profile written to %s (%d bytes)", output_path, len(profile_bytes))
    return output_path


# ---------------------------------------------------------------------------
# ICC binary helpers  (ICC specification v2, big-endian)
# ---------------------------------------------------------------------------

def _s15f16(v: float) -> bytes:
    """Encode a float as ICC s15Fixed16Number (big-endian signed 32-bit)."""
    raw = int(round(v * 65536))
    return struct.pack(">i", raw)


def _xyz_tag(xyz: np.ndarray) -> bytes:
    """Build an XYZType tag (signature 'XYZ ')."""
    return b"XYZ " + b"\x00" * 4 + _s15f16(xyz[0]) + _s15f16(xyz[1]) + _s15f16(xyz[2])


def _srgb_trc_tag(n: int = 256) -> bytes:
    """Build a curveType TRC tag using the sRGB transfer function."""
    entries = []
    for i in range(n):
        t = i / (n - 1)
        linear = t ** 2.2   # approximate sRGB
        val = int(np.clip(round(linear * 65535), 0, 65535))
        entries.append(struct.pack(">H", val))
    header = b"curv" + b"\x00" * 4 + struct.pack(">I", n)
    return header + b"".join(entries)


def _desc_tag(text: str) -> bytes:
    """Build a textDescriptionType tag (ICC v2)."""
    ascii_bytes = text.encode("ascii", errors="replace") + b"\x00"
    n = len(ascii_bytes)
    tag = b"desc" + b"\x00" * 4
    tag += struct.pack(">I", n) + ascii_bytes
    # Pad to align to 4 bytes
    tag += b"\x00" * ((4 - len(tag) % 4) % 4)
    # Unicode and ScriptCode counts (both zero)
    tag += struct.pack(">I", 0) + struct.pack(">H", 0) + b"\x00"
    return tag


def _text_tag(text: str) -> bytes:
    """Build a textType tag."""
    return b"text" + b"\x00" * 4 + text.encode("ascii", errors="replace") + b"\x00"


def _assemble_icc(tags: list[tuple[bytes, bytes]]) -> bytes:
    """
    Assemble an ICC v2 profile from a list of (signature, data) tuples.
    Computes the tag table and header automatically.
    """
    # Compute tag offsets (after 128-byte header + 4-byte tag count + 12*n tag table)
    n_tags = len(tags)
    tag_table_size = 4 + 12 * n_tags
    data_start = 128 + tag_table_size

    offsets: list[int] = []
    current = data_start
    for _, data in tags:
        offsets.append(current)
        # Align to 4 bytes
        current += len(data) + (4 - len(data) % 4) % 4

    profile_size = current

    # --- Header (128 bytes) ---
    now = datetime.utcnow()
    # Manual header build
    header = bytearray(128)
    struct.pack_into(">I", header, 0, profile_size)
    header[4:8] = b"    "      # preferred CMM
    struct.pack_into(">I", header, 8, 0x02100000)  # version 2.1
    header[12:16] = b"mntr"
    header[16:20] = b"RGB "
    header[20:24] = b"XYZ "
    struct.pack_into(">H", header, 24, now.year)
    struct.pack_into(">H", header, 26, now.month)
    struct.pack_into(">H", header, 28, now.day)
    struct.pack_into(">H", header, 30, now.hour)
    struct.pack_into(">H", header, 32, now.minute)
    struct.pack_into(">H", header, 34, now.second)
    header[36:40] = b"acsp"
    # rendering intent = 0 (perceptual), rest zeros

    # --- Tag table ---
    table = struct.pack(">I", n_tags)
    for (sig, data), offset in zip(tags, offsets):
        table += sig + struct.pack(">I", offset) + struct.pack(">I", len(data))

    # --- Tag data ---
    tag_data = b""
    for _, data in tags:
        tag_data += data
        padding = (4 - len(data) % 4) % 4
        tag_data += b"\x00" * padding

    profile = bytes(header) + table + tag_data

    # Patch in actual profile size
    profile = struct.pack(">I", len(profile)) + profile[4:]
    return profile
