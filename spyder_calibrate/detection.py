"""
detection.py — Automatic detection and colour extraction from a photo
containing both the physical SpyderCheckr 24 and the on-screen target.

Strategy
--------
1. Split the image vertically into LEFT (physical card) and RIGHT (screen).
2. On each half, detect the rectangular patch grid using contour analysis:
   - Blur + adaptive threshold → find rectangles
   - Cluster by size and alignment to identify the 6×4 grid
   - Sort patches in reading order (row-major, left→right, top→bottom)
3. Sample the centre of each detected patch (avoid edges / borders).
4. Return two lists of mean BGR colours:
   - ``physical_bgr``  — sampled from the physical card
   - ``screen_bgr``    — sampled from the on-screen render

The grayscale strip (8 patches) is extracted similarly from the lower region
of each half.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class DetectionResult:
    """Colour samples extracted from one detection pass."""
    color_bgr: list[tuple[float, float, float]]   # 24 values
    gray_bgr:  list[tuple[float, float, float]]   # 8 values
    debug_img: np.ndarray | None = None            # annotated image (optional)

    @property
    def all_bgr(self) -> list[tuple[float, float, float]]:
        return self.color_bgr + self.gray_bgr


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def detect_both_halves(
    image_bgr: np.ndarray,
    debug: bool = False,
) -> tuple[DetectionResult, DetectionResult]:
    """
    Detect patches in both halves of the calibration photo.

    Parameters
    ----------
    image_bgr:
        Full BGR float32 [0,1] or uint8 image captured by the camera.
    debug:
        If True, annotate the image with detected patch outlines.

    Returns
    -------
    (physical, screen):
        DetectionResult for the physical card (left) and on-screen target (right).
    """
    # Ensure uint8 for OpenCV operations
    if image_bgr.dtype != np.uint8:
        img8 = (np.clip(image_bgr, 0, 1) * 255).astype(np.uint8)
    else:
        img8 = image_bgr.copy()

    h, w = img8.shape[:2]
    mid = w // 2

    left_half  = img8[:, :mid]
    right_half = img8[:, mid:]

    debug_img = img8.copy() if debug else None

    physical = _detect_half(left_half, debug_img=debug_img, x_offset=0)
    screen   = _detect_half(right_half, debug_img=debug_img, x_offset=mid)

    if debug and debug_img is not None:
        # Draw midline
        cv2.line(debug_img, (mid, 0), (mid, h), (0, 255, 255), 2)
        physical.debug_img = debug_img
        screen.debug_img   = debug_img

    return physical, screen


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_GRID_COLS = 4   # portrait orientation
_GRID_ROWS = 6
_N_COLOR   = _GRID_COLS * _GRID_ROWS   # 24
_N_GRAY    = 8


def _detect_half(
    half: np.ndarray,
    debug_img: np.ndarray | None = None,
    x_offset: int = 0,
) -> DetectionResult:
    """
    Detect the SpyderCheckr patch layout within one image half.

    Falls back to a uniform-grid crop if automatic contour detection fails.
    """
    rects = _find_patch_rectangles(half)

    if len(rects) >= _N_COLOR:
        color_rects, gray_rects = _classify_rects(rects, half.shape)
    else:
        logger.warning(
            "Auto-detection found only %d rectangles (need %d). "
            "Falling back to uniform grid.",
            len(rects), _N_COLOR,
        )
        color_rects, gray_rects = _fallback_grid(half.shape)

    color_bgr = [_mean_colour(half, r) for r in color_rects[:_N_COLOR]]
    gray_bgr  = [_mean_colour(half, r) for r in gray_rects[:_N_GRAY]]

    # Pad if fewer detected (shouldn't happen after fallback)
    _pad = (0.0, 0.0, 0.0)
    while len(color_bgr) < _N_COLOR:
        color_bgr.append(_pad)
    while len(gray_bgr) < _N_GRAY:
        gray_bgr.append(_pad)

    if debug_img is not None:
        for rx, ry, rw, rh in color_rects + gray_rects:
            cv2.rectangle(
                debug_img,
                (x_offset + rx, ry),
                (x_offset + rx + rw, ry + rh),
                (0, 255, 0), 2,
            )

    return DetectionResult(color_bgr=color_bgr, gray_bgr=gray_bgr)


def _find_patch_rectangles(gray_or_bgr: np.ndarray) -> list[tuple[int, int, int, int]]:
    """
    Detect coloured patch rectangles using two complementary strategies:
    1. Morphological approach: threshold bright/saturated regions, find contours
    2. Edge-based: Canny on grayscale
    Both results are merged and deduplicated.
    """
    if len(gray_or_bgr.shape) == 3:
        bgr = gray_or_bgr
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    else:
        gray = gray_or_bgr
        bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    h_img, w_img = gray.shape[:2]
    area_img = h_img * w_img

    rects: list[tuple[int, int, int, int]] = []

    # --- Strategy 1: morphological on grayscale ---
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # Try multiple threshold levels to catch both dark and light patches
    for thresh_val in [30, 60, 100, 150]:
        _, binary = cv2.threshold(blurred, thresh_val, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        rects.extend(_contours_to_rects(contours, area_img))

    # --- Strategy 2: Canny edges ---
    edges = cv2.Canny(blurred, 20, 80)
    edges = cv2.dilate(edges, None, iterations=3)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rects.extend(_contours_to_rects(contours, area_img))

    # --- Strategy 3: adaptive threshold ---
    adaptive = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 51, 5
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    adaptive = cv2.morphologyEx(adaptive, cv2.MORPH_CLOSE, kernel, iterations=3)
    contours, _ = cv2.findContours(adaptive, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rects.extend(_contours_to_rects(contours, area_img))

    rects = _deduplicate_rects(rects)
    return rects


def _contours_to_rects(
    contours,
    area_img: int,
    min_frac: float = 0.003,
    max_frac: float = 0.18,
    min_aspect: float = 0.4,
    max_aspect: float = 3.0,
) -> list[tuple[int, int, int, int]]:
    """Filter contours by area and aspect ratio and return bounding rects."""
    result = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        aspect = w / max(h, 1)
        if not (min_frac * area_img < area < max_frac * area_img):
            continue
        if not (min_aspect < aspect < max_aspect):
            continue
        result.append((x, y, w, h))
    return result


def _deduplicate_rects(
    rects: list[tuple[int, int, int, int]],
    iou_threshold: float = 0.3,
) -> list[tuple[int, int, int, int]]:
    """Simple greedy NMS to remove overlapping rectangle detections."""
    if not rects:
        return []
    rects_sorted = sorted(rects, key=lambda r: r[2] * r[3], reverse=True)
    kept: list[tuple[int, int, int, int]] = []
    for rect in rects_sorted:
        if all(_iou(rect, k) < iou_threshold for k in kept):
            kept.append(rect)
    return kept


def _iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix = max(ax, bx)
    iy = max(ay, by)
    ix2 = min(ax + aw, bx + bw)
    iy2 = min(ay + ah, by + bh)
    iw, ih = max(0, ix2 - ix), max(0, iy2 - iy)
    inter = iw * ih
    union = aw * ah + bw * bh - inter
    return inter / max(union, 1)


def _classify_rects(
    rects: list[tuple[int, int, int, int]],
    shape: tuple[int, int, int],
) -> tuple[list[tuple[int, int, int, int]], list[tuple[int, int, int, int]]]:
    """
    Split detected rectangles into color-grid rects (top region) and
    grayscale-strip rects (bottom region), then sort each group in
    reading order.
    """
    h_img = shape[0]
    threshold_y = int(h_img * 0.97)   # portrait 6-row grid: row 6 centre ~91.7%

    color_rects = [r for r in rects if r[1] + r[3] // 2 < threshold_y]
    gray_rects  = [r for r in rects if r[1] + r[3] // 2 >= threshold_y]

    color_rects = _sort_reading_order(color_rects, cols=_GRID_COLS)
    gray_rects  = sorted(gray_rects, key=lambda r: r[0])   # left → right

    return color_rects, gray_rects


def _sort_reading_order(
    rects: list[tuple[int, int, int, int]],
    cols: int,
) -> list[tuple[int, int, int, int]]:
    """Sort rectangles into row-major order."""
    if not rects:
        return []
    # Estimate row height
    avg_h = np.median([r[3] for r in rects])
    row_tol = avg_h * 0.6

    rows: list[list[tuple[int, int, int, int]]] = []
    remaining = sorted(rects, key=lambda r: r[1])
    while remaining:
        pivot_y = remaining[0][1]
        row = [r for r in remaining if abs(r[1] - pivot_y) < row_tol]
        row.sort(key=lambda r: r[0])
        rows.append(row)
        remaining = [r for r in remaining if r not in row]

    return [rect for row in rows for rect in row]


def _fallback_grid(
    shape: tuple[int, int, int],
) -> tuple[list[tuple[int, int, int, int]], list[tuple[int, int, int, int]]]:
    """
    Generate a uniform grid as a fallback when contour detection fails.
    Divides the half-image into a 6×4 color grid (top 75%) and 8×1 gray
    strip (bottom 20%).
    """
    h, w = shape[:2]
    pad_x = int(w * 0.05)
    pad_y = int(h * 0.05)
    usable_w = w - 2 * pad_x
    usable_h = h - 2 * pad_y

    grid_h = int(usable_h * 0.75)
    cell_w = usable_w // _GRID_COLS
    cell_h = grid_h // _GRID_ROWS

    color_rects = []
    for row in range(_GRID_ROWS):
        for col in range(_GRID_COLS):
            x = pad_x + col * cell_w
            y = pad_y + row * cell_h
            color_rects.append((x, y, cell_w, cell_h))

    gray_y = pad_y + grid_h + int(usable_h * 0.03)
    gray_h = int(usable_h * 0.17)
    gray_cell_w = usable_w // _N_GRAY
    gray_rects = [
        (pad_x + i * gray_cell_w, gray_y, gray_cell_w, gray_h)
        for i in range(_N_GRAY)
    ]

    return color_rects, gray_rects


def _mean_colour(
    image: np.ndarray,
    rect: tuple[int, int, int, int],
    sample_fraction: float = 0.5,
) -> tuple[float, float, float]:
    """
    Return the mean BGR colour of the central region of a rectangle.

    ``sample_fraction`` controls what fraction of the patch interior is sampled
    (avoids patch borders).
    """
    x, y, w, h = rect
    margin_x = int(w * (1 - sample_fraction) / 2)
    margin_y = int(h * (1 - sample_fraction) / 2)

    x1 = max(x + margin_x, 0)
    y1 = max(y + margin_y, 0)
    x2 = min(x + w - margin_x, image.shape[1])
    y2 = min(y + h - margin_y, image.shape[0])

    roi = image[y1:y2, x1:x2]
    if roi.size == 0:
        return (0.0, 0.0, 0.0)

    mean = cv2.mean(roi)[:3]
    return (float(mean[0]), float(mean[1]), float(mean[2]))
