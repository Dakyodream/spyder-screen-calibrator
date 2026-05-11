"""
display.py — Full-screen calibration target renderer.

Opens a borderless Tkinter window that:
  - Occupies the RIGHT half of the screen (left half is reserved for the
    physical SpyderCheckr 24 card placed in front of the camera).
  - Renders the 24 color patches + 8 grayscale patches in the same
    spatial arrangement as the physical card.
  - Can optionally apply a correction matrix (3×3 + offset) in real time
    to visualise how the calibrated screen would look.
"""

from __future__ import annotations
from .screen_utils import get_workarea

import tkinter as tk
from typing import Callable

import numpy as np

from .references import (
    COLOR_PATCHES_SRGB,
    GRAY_PATCHES_SRGB,
)

# ---------------------------------------------------------------------------
# Layout constants
# ---------------------------------------------------------------------------

_COLS = 4          # colour patch columns (portrait: 4 cols × 6 rows)
_ROWS = 6          # colour patch rows
_GRAY_COUNT = 8    # grayscale patches in a horizontal strip


def _clamp(v: float) -> int:
    return max(0, min(255, int(round(v))))


def _apply_matrix(rgb: tuple[int, int, int], matrix: np.ndarray | None) -> tuple[int, int, int]:
    """Apply a 3×3 correction matrix to an sRGB triplet (0-255)."""
    if matrix is None:
        return rgb
    v = np.array(rgb, dtype=np.float32) / 255.0
    v_corrected = matrix @ v
    return (
        _clamp(v_corrected[0] * 255),
        _clamp(v_corrected[1] * 255),
        _clamp(v_corrected[2] * 255),
    )


def _rgb_to_hex(r: int, g: int, b: int) -> str:
    return f"#{r:02x}{g:02x}{b:02x}"


# ---------------------------------------------------------------------------
# CalibrationDisplay
# ---------------------------------------------------------------------------

class CalibrationDisplay:
    """
    Tkinter window showing the on-screen calibration target.

    Parameters
    ----------
    on_ready:
        Optional callback fired once the window is fully mapped.
    correction_matrix:
        Optional 3×3 numpy array applied to every patch colour before display.
        Pass ``None`` for uncorrected (first pass) display.
    """

    def __init__(
        self,
        on_ready: Callable[[], None] | None = None,
        correction_matrix: np.ndarray | None = None,
    ) -> None:
        self._on_ready = on_ready
        self._matrix = correction_matrix
        self._root: tk.Tk | None = None
        self._canvas: tk.Canvas | None = None
        self._capture_mode: bool = False  # when True, left half is black

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def show(self) -> None:
        """Build and display the window. Blocks until the window is closed."""
        self._root = tk.Tk()
        self._root.title("SpyderCheckr Calibration Target")
        self._root.configure(bg="black")
        self._root.attributes("-fullscreen", False)
        self._root.resizable(False, False)

        # Detect screen dimensions
        wa_x, wa_y, wa_w, wa_h = get_workarea()

        # Occupy the RIGHT half of the work area
        win_w = wa_w // 2
        win_h = wa_h
        self._root.geometry(f"{win_w}x{win_h}+{wa_x + win_w}+{wa_y}")
        self._root.overrideredirect(True)   # borderless

        self._canvas = tk.Canvas(
            self._root,
            width=win_w,
            height=win_h,
            bg="black",
            highlightthickness=0,
        )
        self._canvas.pack()

        self._draw(win_w, win_h)

        if self._on_ready:
            self._root.after(200, self._on_ready)

        # Allow ESC to close
        self._root.bind("<Escape>", lambda _e: self.close())

        self._root.mainloop()

    def close(self) -> None:
        """Close the display window."""
        if self._root is not None:
            self._root.destroy()
            self._root = None

    def update_matrix(self, matrix: np.ndarray | None) -> None:
        """
        Update the correction matrix and redraw all patches.
        Safe to call from any thread via ``root.after``.
        """
        self._matrix = matrix
        if self._root is not None and self._canvas is not None:
            _, _, wa_w, wa_h = get_workarea()
            self._canvas.delete("patch")
            self._draw(wa_w // 2, wa_h, tag="patch")

    # ------------------------------------------------------------------
    # Drawing helpers
    # ------------------------------------------------------------------

    def set_capture_mode(self, active: bool) -> None:
        """
        Switch capture mode on/off.
        When active, a black overlay covers the LEFT half of the screen
        so the camera only sees the colour target on the right.
        """
        self._capture_mode = active
        if self._root is not None and self._canvas is not None:
            self._root.after(0, self._refresh_overlay)

    def _refresh_overlay(self) -> None:
        """
        In capture mode: expand window to full screen, shift the colour target
        to the RIGHT half, and fill the LEFT half with black.
        In normal mode: shrink back to right half only.
        """
        if self._root is None or self._canvas is None:
            return
        _, _, wa_w, wa_h = get_workarea()
        win_w = wa_w // 2

        self._canvas.delete("overlay")

        if self._capture_mode:
            # Expand window to full screen starting at x=0
            self._root.geometry(f"{wa_w}x{wa_h}+0+0")
            self._canvas.configure(width=wa_w)
            # Black rectangle covers the LEFT half
            self._canvas.create_rectangle(
                0, 0, win_w, wa_h,
                fill="black", outline="",
                tags="overlay",
            )
            # Shift all patch drawing to the RIGHT half by moving canvas items
            # (patches were drawn at x coords 0..win_w, shift them by win_w)
            self._canvas.move("patch", win_w, 0)
        else:
            # Move patches back to left of canvas before shrinking
            self._canvas.move("patch", -win_w, 0)
            self._root.geometry(f"{win_w}x{wa_h}+{win_w}+0")
            self._canvas.configure(width=win_w)

    def _draw(self, w: int, h: int, tag: str = "patch") -> None:
        """Render all patches onto the canvas."""
        assert self._canvas is not None

        padding = int(w * 0.04)
        usable_w = w - 2 * padding
        usable_h = h - 2 * padding

        # --- Color patch grid (top 75% of usable height) ----------------
        grid_h = int(usable_h * 0.75)
        cell_w = usable_w // _COLS
        cell_h = grid_h // _ROWS
        margin = 4  # px gap between patches

        patches_srgb = [(r, g, b) for (_, r, g, b) in COLOR_PATCHES_SRGB]

        for idx, (r, g, b) in enumerate(patches_srgb):
            row = idx // _COLS
            col = idx % _COLS
            r2, g2, b2 = _apply_matrix((r, g, b), self._matrix)
            color = _rgb_to_hex(r2, g2, b2)

            x0 = padding + col * cell_w + margin
            y0 = padding + row * cell_h + margin
            x1 = x0 + cell_w - 2 * margin
            y1 = y0 + cell_h - 2 * margin

            self._canvas.create_rectangle(x0, y0, x1, y1, fill=color, outline="", tags=tag)

            # Patch name label (small, white)
            name = COLOR_PATCHES_SRGB[idx][0]
            self._canvas.create_text(
                (x0 + x1) // 2, y1 - 8,
                text=name, fill="white",
                font=("Helvetica", 7),
                tags=tag,
            )

        # --- Grayscale strip (bottom 20% of usable height) --------------
        gray_y_start = padding + grid_h + int(usable_h * 0.03)
        gray_h = int(usable_h * 0.18)
        gray_cell_w = usable_w // _GRAY_COUNT

        for idx, (_, r, g, b) in enumerate(GRAY_PATCHES_SRGB):
            r2, g2, b2 = _apply_matrix((r, g, b), self._matrix)
            color = _rgb_to_hex(r2, g2, b2)

            x0 = padding + idx * gray_cell_w + margin
            y0 = gray_y_start
            x1 = x0 + gray_cell_w - 2 * margin
            y1 = y0 + gray_h - 2 * margin

            self._canvas.create_rectangle(x0, y0, x1, y1, fill=color, outline="", tags=tag)

        # --- Instruction label ------------------------------------------
        self._canvas.create_text(
            w // 2, h - padding // 2,
            text="Place the physical SpyderCheckr 24 on the LEFT half — align patches visually",
            fill="#888888",
            font=("Helvetica", 10),
            tags=tag,
        )
