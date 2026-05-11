"""
screen_utils.py — Helper to get the usable screen area (excluding taskbar).
"""
from __future__ import annotations
import subprocess
import tkinter as tk


def get_workarea() -> tuple[int, int, int, int]:
    """
    Return (x, y, width, height) of the usable desktop area,
    excluding taskbars/panels.

    Tries _NET_WORKAREA (X11/Wayland-XWayland) first, falls back to
    full screen dimensions.
    """
    try:
        out = subprocess.check_output(
            ["xprop", "-root", "_NET_WORKAREA"],
            stderr=subprocess.DEVNULL,
            text=True,
        )
        # Output looks like:
        # _NET_WORKAREA(CARDINAL) = 0, 0, 1920, 1040, 0, 0, 1920, 1040, ...
        nums = [int(x.strip()) for x in out.split("=")[1].split(",")]
        x, y, w, h = nums[0], nums[1], nums[2], nums[3]
        return x, y, w, h
    except Exception:
        pass

    # Fallback: ask Tkinter
    root = tk.Tk()
    root.withdraw()
    w = root.winfo_screenwidth()
    h = root.winfo_screenheight()
    root.destroy()
    return 0, 0, w, h
