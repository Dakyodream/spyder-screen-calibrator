"""
main.py — SpyderCheckr Screen Calibrator — main entry point.

Launches a small control window (left side of screen) alongside the
calibration target display (right side).  The control window shows:
  - Status / progress log
  - ΔE improvement per pass
  - Start / Stop / Apply Profile buttons
"""

from __future__ import annotations
from .screen_utils import get_workarea

import logging
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, scrolledtext, ttk

from .calibrator import CalibrationSession

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# GUI helpers
# ---------------------------------------------------------------------------

class TextHandler(logging.Handler):
    """Route log records to a Tkinter ScrolledText widget."""

    def __init__(self, widget: scrolledtext.ScrolledText) -> None:
        super().__init__()
        self.widget = widget

    def emit(self, record: logging.LogRecord) -> None:
        msg = self.format(record) + "\n"
        self.widget.after(0, self._append, msg)

    def _append(self, msg: str) -> None:
        self.widget.configure(state="normal")
        self.widget.insert(tk.END, msg)
        self.widget.see(tk.END)
        self.widget.configure(state="disabled")


# ---------------------------------------------------------------------------
# Main application window
# ---------------------------------------------------------------------------

class CalibrationApp:
    """
    Control panel window.  Shown on the LEFT half of the screen.
    The CalibrationDisplay (target) runs on the RIGHT half.
    """

    def __init__(self) -> None:
        self._session: CalibrationSession | None = None
        self._icc_path: Path | None = None
        self._showing_corrected: bool = True  # toggle state for before/after preview

        # --- Root window ---
        self.root = tk.Tk()
        self.root.title("SpyderCheckr Screen Calibrator")
        self.root.configure(bg="#1e1e1e")

        wa_x, wa_y, wa_w, wa_h = get_workarea()
        self.root.geometry(f"{wa_w // 2}x{wa_h}+{wa_x}+{wa_y}")
        self.root.resizable(False, False)

        self._build_ui()
        self._attach_log_handler()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        root = self.root
        pad = {"padx": 16, "pady": 8}

        # Title
        tk.Label(
            root, text="SpyderCheckr Screen Calibrator",
            font=("Helvetica", 18, "bold"),
            bg="#1e1e1e", fg="#e0e0e0",
        ).pack(fill=tk.X, **pad)

        tk.Label(
            root,
            text="Place the physical SpyderCheckr 24 on the LEFT half of the frame.\n"
                 "The on-screen target is shown on the RIGHT half.",
            font=("Helvetica", 10),
            bg="#1e1e1e", fg="#888888",
            justify=tk.LEFT,
        ).pack(fill=tk.X, padx=16, pady=(0, 8))

        ttk.Separator(root, orient="horizontal").pack(fill=tk.X, padx=16)

        # Passes control
        passes_frame = tk.Frame(root, bg="#1e1e1e")
        passes_frame.pack(fill=tk.X, **pad)
        tk.Label(
            passes_frame, text="Number of passes:", bg="#1e1e1e", fg="#c0c0c0",
            font=("Helvetica", 11),
        ).pack(side=tk.LEFT)
        self._passes_var = tk.IntVar(value=3)
        ttk.Spinbox(
            passes_frame, from_=1, to=10,
            textvariable=self._passes_var, width=4,
            font=("Helvetica", 11),
        ).pack(side=tk.LEFT, padx=8)

        # Progress bar
        self._progress = ttk.Progressbar(root, mode="determinate", maximum=100)
        self._progress.pack(fill=tk.X, padx=16, pady=4)

        # ΔE display
        self._de_var = tk.StringVar(value="ΔE — not measured yet")
        tk.Label(
            root, textvariable=self._de_var,
            font=("Helvetica", 12), bg="#1e1e1e", fg="#88ccff",
        ).pack(**pad)

        # Buttons
        btn_frame = tk.Frame(root, bg="#1e1e1e")
        btn_frame.pack(fill=tk.X, **pad)

        self._btn_start = tk.Button(
            btn_frame, text="▶  Start Calibration",
            font=("Helvetica", 12, "bold"),
            bg="#2d7d46", fg="white", activebackground="#3aad60",
            relief=tk.FLAT, cursor="hand2",
            command=self._start,
        )
        self._btn_start.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=4)

        self._btn_stop = tk.Button(
            btn_frame, text="■  Stop",
            font=("Helvetica", 12),
            bg="#7d2d2d", fg="white", activebackground="#ad3a3a",
            relief=tk.FLAT, cursor="hand2", state=tk.DISABLED,
            command=self._stop,
        )
        self._btn_stop.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=4)

        self._btn_apply = tk.Button(
            btn_frame, text="🎨  Apply Profile",
            font=("Helvetica", 12),
            bg="#2d5f7d", fg="white", activebackground="#3a7dad",
            relief=tk.FLAT, cursor="hand2", state=tk.DISABLED,
            command=self._apply_profile,
        )
        self._btn_apply.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=4)

        self._btn_toggle = tk.Button(
            btn_frame, text="👁  Before",
            font=("Helvetica", 12),
            bg="#555555", fg="white", activebackground="#777777",
            relief=tk.FLAT, cursor="hand2", state=tk.DISABLED,
            command=self._toggle_preview,
        )
        self._btn_toggle.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=4)

        ttk.Separator(root, orient="horizontal").pack(fill=tk.X, padx=16, pady=8)

        # Log area
        tk.Label(
            root, text="Log", bg="#1e1e1e", fg="#888888",
            font=("Helvetica", 10, "bold"),
        ).pack(anchor=tk.W, padx=16)

        self._log_widget = scrolledtext.ScrolledText(
            root, state="disabled",
            font=("Courier", 9),
            bg="#121212", fg="#c0c0c0",
            height=20, relief=tk.FLAT,
        )
        self._log_widget.pack(fill=tk.BOTH, expand=True, padx=16, pady=(0, 16))

    def _attach_log_handler(self) -> None:
        handler = TextHandler(self._log_widget)
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s  %(message)s", "%H:%M:%S"))
        logging.getLogger().addHandler(handler)

    # ------------------------------------------------------------------
    # Button handlers
    # ------------------------------------------------------------------

    def _start(self) -> None:
        from .display import CalibrationDisplay

        n_passes = self._passes_var.get()
        self._session = CalibrationSession(
            output_dir=Path("output"),
            n_passes=n_passes,
            on_progress=self._on_progress,
            on_complete=self._on_complete,
            on_error=self._on_error,
        )

        # Launch the on-screen target in its own thread
        display_ready = threading.Event()
        self._display = CalibrationDisplay(
            on_ready=lambda: display_ready.set(),
        )

        def run_display() -> None:
            self._session.display_update_fn = self._display.update_matrix
            self._session.display_ref = self._display
            self._display.show()   # blocks until closed

        display_thread = threading.Thread(target=run_display, daemon=True, name="DisplayThread")
        display_thread.start()

        # Wait up to 3s for display to be ready
        display_ready.wait(timeout=3.0)

        # Start calibration loop
        self._session.start()

        self._btn_start.configure(state=tk.DISABLED)
        self._btn_stop.configure(state=tk.NORMAL)
        self._progress["value"] = 0
        self._de_var.set("ΔE — calibration running…")

    def _stop(self) -> None:
        if self._session:
            self._session.stop()
        self._btn_start.configure(state=tk.NORMAL)
        self._btn_stop.configure(state=tk.DISABLED)

    def _toggle_preview(self) -> None:
        """Switch the on-screen target between corrected (After) and uncorrected (Before)."""
        self._showing_corrected = not self._showing_corrected
        if self._showing_corrected:
            # Show corrected colours
            matrix = self._session.current_matrix if self._session else None
            label = "👁  Before"
        else:
            # Show original uncorrected colours
            matrix = None
            label = "✅  After"
        self._btn_toggle.configure(text=label)
        if hasattr(self, "_display") and self._display is not None:
            self._display.update_matrix(matrix)

    def _apply_profile(self) -> None:
        if self._icc_path is None or not self._icc_path.exists():
            messagebox.showerror("Error", "No ICC profile found. Run calibration first.")
            return

        import subprocess
        try:
            # Try colord (modern desktop)
            result = subprocess.run(
                ["colormgr", "import-profile", str(self._icc_path)],
                capture_output=True, text=True,
            )
            if result.returncode == 0:
                messagebox.showinfo(
                    "Profile applied",
                    f"ICC profile imported via colord.\n\n{self._icc_path}\n\n"
                    "You may need to assign it in System Settings → Color.",
                )
                return
        except FileNotFoundError:
            pass

        # Fallback: xcalib
        try:
            subprocess.run(
                ["xcalib", "-d", ":0", str(self._icc_path)],
                check=True,
            )
            messagebox.showinfo(
                "Profile applied",
                f"VCGT loaded via xcalib.\n\n{self._icc_path}\n\n"
                "Note: xcalib only applies VCGT, not the full matrix. "
                "For full correction, import via colord.",
            )
        except (FileNotFoundError, subprocess.CalledProcessError) as exc:
            messagebox.showerror(
                "Could not apply profile",
                f"Please manually load the profile:\n{self._icc_path}\n\nError: {exc}",
            )

    # ------------------------------------------------------------------
    # Session callbacks (called from background thread)
    # ------------------------------------------------------------------

    def _on_progress(self, pass_num: int, total: int, de_before: float, de_after: float) -> None:
        pct = int(pass_num / total * 100)
        self.root.after(0, self._progress.configure, {"value": pct})
        self.root.after(
            0,
            self._de_var.set,
            f"Pass {pass_num}/{total}  —  ΔE before: {de_before:.2f}  →  after: {de_after:.2f}",
        )

    def _on_complete(self, icc_path: Path) -> None:
        self._icc_path = icc_path
        self.root.after(0, self._btn_start.configure, {"state": tk.NORMAL})
        self.root.after(0, self._btn_stop.configure, {"state": tk.DISABLED})
        self.root.after(0, self._btn_apply.configure, {"state": tk.NORMAL})
        self.root.after(0, self._btn_toggle.configure, {"state": tk.NORMAL})
        self.root.after(0, self._progress.configure, {"value": 100})
        self.root.after(
            0,
            messagebox.showinfo,
            "Calibration complete",
            f"ICC profile saved:\n{icc_path}\n\nClick 'Apply Profile' to load it.",
        )

    def _on_error(self, exc: Exception) -> None:
        self.root.after(0, self._btn_start.configure, {"state": tk.NORMAL})
        self.root.after(0, self._btn_stop.configure, {"state": tk.DISABLED})
        self.root.after(0, messagebox.showerror, "Calibration error", str(exc))

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    def run(self) -> None:
        self.root.protocol("WM_DELETE_WINDOW", self._quit)
        self.root.mainloop()

    def _quit(self) -> None:
        if self._session:
            self._session.stop()
        self.root.destroy()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    app = CalibrationApp()
    app.run()


if __name__ == "__main__":
    main()
