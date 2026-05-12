"""
main.py — SpyderCheckr Screen Calibrator — control panel.

Small floating window (top-left corner, always on top) so the camera's
centre-sample region is unaffected during calibration.

The fullscreen calibration display runs in a separate thread/window.
"""

from __future__ import annotations

import logging
import threading
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, scrolledtext, ttk

from .calibrator import CalibrationSession

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


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


class CalibrationApp:
    """
    Compact floating control panel.

    Positioned at the top-left corner (400×480 px) and kept always on top
    so it remains accessible while the display window covers the screen.
    The camera samples the central 40% of the image, so this panel is safely
    outside that region on any ≥1080p screen.
    """

    def __init__(self) -> None:
        self._session: CalibrationSession | None = None
        self._icc_path: Path | None = None
        self._showing_corrected: bool = True
        self._display = None

        self.root = tk.Tk()
        self.root.title("SpyderCheckr Calibrator")
        self.root.configure(bg="#1e1e1e")
        self.root.attributes("-topmost", True)
        self.root.resizable(False, False)
        self.root.geometry("400x480+0+0")

        self._build_ui()
        self._attach_log_handler()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        root = self.root
        pad = {"padx": 12, "pady": 6}

        tk.Label(
            root, text="SpyderCheckr Screen Calibrator",
            font=("Helvetica", 13, "bold"),
            bg="#1e1e1e", fg="#e0e0e0",
        ).pack(fill=tk.X, **pad)

        ttk.Separator(root, orient="horizontal").pack(fill=tk.X, padx=12)

        # Passes
        passes_frame = tk.Frame(root, bg="#1e1e1e")
        passes_frame.pack(fill=tk.X, **pad)
        tk.Label(passes_frame, text="Passes:", bg="#1e1e1e", fg="#c0c0c0",
                 font=("Helvetica", 10)).pack(side=tk.LEFT)
        self._passes_var = tk.IntVar(value=3)
        ttk.Spinbox(passes_frame, from_=1, to=10,
                    textvariable=self._passes_var, width=4,
                    font=("Helvetica", 10)).pack(side=tk.LEFT, padx=6)

        # Progress bar
        self._progress = ttk.Progressbar(root, mode="determinate", maximum=100)
        self._progress.pack(fill=tk.X, padx=12, pady=3)

        # Status
        self._status_var = tk.StringVar(value="Ready — press Start to begin")
        tk.Label(
            root, textvariable=self._status_var,
            font=("Helvetica", 10), bg="#1e1e1e", fg="#88ccff",
            wraplength=376, justify=tk.LEFT,
        ).pack(padx=12, pady=2, anchor=tk.W)

        # ΔE
        self._de_var = tk.StringVar(value="ΔE — not measured yet")
        tk.Label(
            root, textvariable=self._de_var,
            font=("Helvetica", 10), bg="#1e1e1e", fg="#aaffaa",
        ).pack(padx=12, pady=2)

        # Buttons — row 1
        btn_frame = tk.Frame(root, bg="#1e1e1e")
        btn_frame.pack(fill=tk.X, padx=12, pady=4)

        self._btn_start = tk.Button(
            btn_frame, text="▶  Start",
            font=("Helvetica", 11, "bold"),
            bg="#2d7d46", fg="white", activebackground="#3aad60",
            relief=tk.FLAT, cursor="hand2",
            command=self._start,
        )
        self._btn_start.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=2)

        self._btn_stop = tk.Button(
            btn_frame, text="■  Stop",
            font=("Helvetica", 11),
            bg="#7d2d2d", fg="white", activebackground="#ad3a3a",
            relief=tk.FLAT, cursor="hand2", state=tk.DISABLED,
            command=self._stop,
        )
        self._btn_stop.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=2)

        # Buttons — row 2
        btn_frame2 = tk.Frame(root, bg="#1e1e1e")
        btn_frame2.pack(fill=tk.X, padx=12, pady=2)

        self._btn_apply = tk.Button(
            btn_frame2, text="Apply Profile",
            font=("Helvetica", 11),
            bg="#2d5f7d", fg="white", activebackground="#3a7dad",
            relief=tk.FLAT, cursor="hand2", state=tk.DISABLED,
            command=self._apply_profile,
        )
        self._btn_apply.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=2)

        self._btn_toggle = tk.Button(
            btn_frame2, text="👁  Before",
            font=("Helvetica", 11),
            bg="#555555", fg="white", activebackground="#777777",
            relief=tk.FLAT, cursor="hand2", state=tk.DISABLED,
            command=self._toggle_preview,
        )
        self._btn_toggle.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=2)

        ttk.Separator(root, orient="horizontal").pack(fill=tk.X, padx=12, pady=4)

        # Log
        tk.Label(root, text="Log", bg="#1e1e1e", fg="#888888",
                 font=("Helvetica", 9, "bold")).pack(anchor=tk.W, padx=12)
        self._log_widget = scrolledtext.ScrolledText(
            root, state="disabled", font=("Courier", 8),
            bg="#121212", fg="#c0c0c0", height=10, relief=tk.FLAT,
        )
        self._log_widget.pack(fill=tk.BOTH, expand=True, padx=12, pady=(0, 12))

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
            on_patch_progress=self._on_patch_progress,
            on_pass_complete=self._on_pass_complete,
            on_complete=self._on_complete,
            on_error=self._on_error,
        )

        display_ready = threading.Event()
        self._display = CalibrationDisplay(on_ready=lambda: display_ready.set())

        def run_display() -> None:
            self._session.display_ref = self._display
            self._display.show()

        display_thread = threading.Thread(target=run_display, daemon=True, name="DisplayThread")
        display_thread.start()
        display_ready.wait(timeout=3.0)

        self._session.start()

        self._btn_start.configure(state=tk.DISABLED)
        self._btn_stop.configure(state=tk.NORMAL)
        self._progress["value"] = 0
        self._status_var.set("Calibration starting…")
        self._de_var.set("ΔE — calibration running…")

    def _stop(self) -> None:
        if self._session:
            self._session.stop()
        self._btn_start.configure(state=tk.NORMAL)
        self._btn_stop.configure(state=tk.DISABLED)
        self._status_var.set("Stopped")

    def _toggle_preview(self) -> None:
        """Switch the comparison grid between corrected (After) and uncorrected (Before)."""
        self._showing_corrected = not self._showing_corrected
        if self._showing_corrected:
            matrix = self._session.current_matrix if self._session else None
            label = "👁  Before"
        else:
            matrix = None
            label = "✅  After"
        self._btn_toggle.configure(text=label)
        if self._display is not None:
            self._display.update_matrix(matrix)

    def _apply_profile(self) -> None:
        if self._icc_path is None or not self._icc_path.exists():
            messagebox.showerror("Error", "No ICC profile found. Run calibration first.")
            return
        import subprocess
        try:
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
        try:
            subprocess.run(["xcalib", "-d", ":0", str(self._icc_path)], check=True)
            messagebox.showinfo("Profile applied",
                                f"VCGT loaded via xcalib.\n\n{self._icc_path}")
        except (FileNotFoundError, subprocess.CalledProcessError) as exc:
            messagebox.showerror("Could not apply profile",
                                 f"Please manually load:\n{self._icc_path}\n\nError: {exc}")

    # ------------------------------------------------------------------
    # Session callbacks (called from background thread)
    # ------------------------------------------------------------------

    def _on_patch_progress(
        self, pass_num: int, patch_idx: int, n_patches: int, pct: int, name: str
    ) -> None:
        self.root.after(0, self._progress.configure, {"value": pct})
        self.root.after(
            0, self._status_var.set,
            f"Pass {pass_num}/{self._passes_var.get()} — "
            f"Patch {patch_idx}/{n_patches}: {name}",
        )

    def _on_pass_complete(
        self, pass_num: int, total: int, de_before: float, de_after: float
    ) -> None:
        self.root.after(
            0, self._de_var.set,
            f"Pass {pass_num}/{total}  ΔE {de_before:.2f} → {de_after:.2f}",
        )

    def _on_complete(self, icc_path: Path) -> None:
        self._icc_path = icc_path
        self.root.after(0, self._btn_start.configure, {"state": tk.NORMAL})
        self.root.after(0, self._btn_stop.configure, {"state": tk.DISABLED})
        self.root.after(0, self._btn_apply.configure, {"state": tk.NORMAL})
        self.root.after(0, self._btn_toggle.configure, {"state": tk.NORMAL})
        self.root.after(0, self._progress.configure, {"value": 100})
        self.root.after(0, self._status_var.set, "Calibration complete ✓")
        self.root.after(
            0, messagebox.showinfo,
            "Calibration complete",
            f"ICC profile saved:\n{icc_path}\n\nClick 'Apply Profile' to load it.",
        )

    def _on_error(self, exc: Exception) -> None:
        self.root.after(0, self._btn_start.configure, {"state": tk.NORMAL})
        self.root.after(0, self._btn_stop.configure, {"state": tk.DISABLED})
        self.root.after(0, self._status_var.set, f"Error: {exc}")
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


def main() -> None:
    app = CalibrationApp()
    app.run()


if __name__ == "__main__":
    main()
