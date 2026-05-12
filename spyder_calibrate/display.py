"""
display.py — Calibration display window.

Two operating modes
-------------------
CALIBRATION  — fullscreen solid-colour panel (one patch at a time).
               The camera captures each colour; the panel covers the whole screen.
COMPARISON   — right-half 4×6 patch grid with optional correction matrix applied,
               for the before/after preview shown after calibration completes.
"""

from __future__ import annotations
from .screen_utils import get_workarea

import tkinter as tk
from typing import Callable

import numpy as np

from .references import COLOR_PATCHES_SRGB, GRAY_PATCHES_SRGB

_COLS = 4
_ROWS = 6
_GRAY_COUNT = 8


def _clamp(v: float) -> int:
    return max(0, min(255, int(round(v))))


def _apply_matrix(rgb: tuple[int, int, int], matrix: np.ndarray | None) -> tuple[int, int, int]:
    """Apply a 3×3 linear-sRGB correction matrix to an sRGB (0-255) triplet."""
    if matrix is None:
        return rgb
    v = np.array(rgb, dtype=np.float64) / 255.0
    # Decode gamma: sRGB → linear
    lin = np.where(v <= 0.04045, v / 12.92, ((v + 0.055) / 1.055) ** 2.4)
    # Apply correction in linear space
    corrected = np.clip(matrix @ lin, 0.0, 1.0)
    # Re-encode gamma: linear → sRGB
    enc = np.where(corrected <= 0.0031308,
                   corrected * 12.92,
                   1.055 * corrected ** (1.0 / 2.4) - 0.055)
    return (
        _clamp(enc[0] * 255),
        _clamp(enc[1] * 255),
        _clamp(enc[2] * 255),
    )


def _rgb_to_hex(r: int, g: int, b: int) -> str:
    return f"#{r:02x}{g:02x}{b:02x}"


class CalibrationDisplay:
    """
    Tkinter display window for calibration and comparison.

    During calibration: fullscreen solid colour (one patch at a time).
    After calibration: right-half 4×6 grid for before/after comparison.

    Usage
    -----
    1. Call ``show()`` from a background thread — it blocks until closed.
    2. During calibration, call ``set_solid_color(r, g, b)`` from any thread.
    3. After calibration, call ``show_comparison(matrix)`` to switch to grid view.
    4. Toggle ``update_matrix(matrix)`` for before/after preview.
    """

    def __init__(self, on_ready: Callable[[], None] | None = None) -> None:
        self._on_ready = on_ready
        self._matrix: np.ndarray | None = None
        self._root: tk.Tk | None = None
        self._canvas: tk.Canvas | None = None
        self._in_comparison: bool = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def show(self) -> None:
        """Build fullscreen window and enter the Tkinter main loop (blocks)."""
        wa_x, wa_y, wa_w, wa_h = get_workarea()

        self._root = tk.Tk()
        self._root.title("Calibration Display")
        self._root.configure(bg="black")
        self._root.overrideredirect(True)
        self._root.geometry(f"{wa_w}x{wa_h}+{wa_x}+{wa_y}")

        self._canvas = tk.Canvas(
            self._root, width=wa_w, height=wa_h,
            bg="black", highlightthickness=0,
        )
        self._canvas.pack()
        self._canvas.create_rectangle(0, 0, wa_w, wa_h, fill="black", outline="")

        self._root.bind("<Escape>", lambda _e: self.close())

        if self._on_ready:
            self._root.after(200, self._on_ready)

        self._root.mainloop()

    def close(self) -> None:
        if self._root is not None:
            self._root.destroy()
            self._root = None

    # ------------------------------------------------------------------
    # Calibration mode — solid fullscreen colour
    # ------------------------------------------------------------------

    def set_solid_color(self, r: int, g: int, b: int) -> None:
        """
        Fill the entire screen with a single colour.
        Thread-safe: schedules the redraw on the Tk main loop.
        """
        if self._root is None:
            return
        color = _rgb_to_hex(r, g, b)
        _, _, wa_w, wa_h = get_workarea()
        self._root.after(0, self._do_solid, color, wa_w, wa_h)

    def _do_solid(self, color: str, w: int, h: int) -> None:
        if self._canvas is None:
            return
        self._canvas.delete("all")
        self._canvas.create_rectangle(0, 0, w, h, fill=color, outline="")
        self._canvas.update_idletasks()

    # ------------------------------------------------------------------
    # Comparison mode — right-half patch grid
    # ------------------------------------------------------------------

    def show_comparison(self, matrix: np.ndarray | None = None) -> None:
        """
        Switch to the patch-grid view on the right half of the screen.
        Call this after calibration completes.
        """
        self._matrix = matrix
        self._in_comparison = True
        if self._root is None:
            return
        _, _, wa_w, wa_h = get_workarea()
        self._root.after(0, self._do_switch_comparison, wa_w, wa_h)

    def _do_switch_comparison(self, wa_w: int, wa_h: int) -> None:
        if self._root is None or self._canvas is None:
            return
        win_w = wa_w // 2
        self._root.geometry(f"{win_w}x{wa_h}+{win_w}+0")
        self._canvas.configure(width=win_w, height=wa_h)
        self._draw_grid(win_w, wa_h)

    def update_matrix(self, matrix: np.ndarray | None) -> None:
        """Update the correction matrix and redraw the grid (comparison mode only)."""
        self._matrix = matrix
        if not self._in_comparison or self._root is None or self._canvas is None:
            return
        _, _, wa_w, wa_h = get_workarea()
        self._root.after(0, self._draw_grid, wa_w // 2, wa_h)

    # ------------------------------------------------------------------
    # Drawing
    # ------------------------------------------------------------------

    def _draw_grid(self, w: int, h: int) -> None:
        if self._canvas is None:
            return
        self._canvas.delete("all")

        padding = int(w * 0.04)
        usable_w = w - 2 * padding
        usable_h = h - 2 * padding

        grid_h = int(usable_h * 0.75)
        cell_w = usable_w // _COLS
        cell_h = grid_h // _ROWS
        margin = 4

        for idx, (name, r, g, b) in enumerate(COLOR_PATCHES_SRGB):
            row = idx // _COLS
            col = idx % _COLS
            r2, g2, b2 = _apply_matrix((r, g, b), self._matrix)
            color = _rgb_to_hex(r2, g2, b2)
            x0 = padding + col * cell_w + margin
            y0 = padding + row * cell_h + margin
            x1 = x0 + cell_w - 2 * margin
            y1 = y0 + cell_h - 2 * margin
            self._canvas.create_rectangle(x0, y0, x1, y1, fill=color, outline="")
            self._canvas.create_text(
                (x0 + x1) // 2, y1 - 8,
                text=name, fill="white", font=("Helvetica", 7),
            )

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
            self._canvas.create_rectangle(x0, y0, x1, y1, fill=color, outline="")

        self._canvas.create_text(
            w // 2, h - padding // 2,
            text="SpyderCheckr 24 — Calibration Result",
            fill="#888888", font=("Helvetica", 10),
        )
