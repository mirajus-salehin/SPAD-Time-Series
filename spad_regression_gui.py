"""
SPAD Regression — Tkinter GUI
==============================
A graphical front-end for spad_regression.py.

Tabs
----
  Train   : pick CSV, set test-size / random-state / model save path,
            choose which feature columns to include, then run training.
  Predict : browse per-band TIFFs (GREEN / NIR / RED / RedEdge) and an
            optional RGB TIFF, choose model + output paths, set nodata
            value, optionally display the SPAD map after inference.

All console output is mirrored into the embedded log panel.

Dependencies (same as spad_regression.py):
    pip install numpy pandas scipy scikit-learn rasterio joblib matplotlib
"""

import sys
import io
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext

# ── re-use the original module ────────────────────────────────────────────────
# Place spad_regression.py in the same folder (or on PYTHONPATH).
try:
    import spad_regression as sr
except ModuleNotFoundError:
    messagebox.showerror(
        "Import error",
        "spad_regression.py not found.\n"
        "Place it in the same directory as this GUI script.",
    )
    sys.exit(1)


# ─────────────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────────────

ALL_FEATURES = [
    "GREEN", "NIR", "RED", "RedEdge",
    "Red_RGB", "Green_RGB", "Blue_RGB",
    "NDVI", "CRE", "GNDVI", "RENDVI", "SR", "SAVI", "DVI",
]

BAND_KEYS = ["GREEN", "NIR", "RED", "RedEdge"]


class StdoutRedirector(io.TextIOBase):
    """Redirect stdout/stderr writes into a tk.Text widget."""

    def __init__(self, text_widget: tk.Text):
        self._widget = text_widget

    def write(self, s: str) -> int:
        self._widget.after(0, self._append, s)
        return len(s)

    def _append(self, s: str):
        self._widget.configure(state="normal")
        self._widget.insert(tk.END, s)
        self._widget.see(tk.END)
        self._widget.configure(state="disabled")

    def flush(self):
        pass


def browse_file(var: tk.StringVar, filetypes=(("All files", "*.*"),)):
    path = filedialog.askopenfilename(filetypes=filetypes)
    if path:
        var.set(path)


def browse_save(var: tk.StringVar, filetypes=(("All files", "*.*"),)):
    path = filedialog.asksaveasfilename(filetypes=filetypes)
    if path:
        var.set(path)


def _row(parent, label: str, row: int, colspan: int = 1) -> ttk.Label:
    lbl = ttk.Label(parent, text=label)
    lbl.grid(row=row, column=0, sticky="w", padx=8, pady=4)
    return lbl


def _entry_browse(parent, var: tk.StringVar, row: int,
                  filetypes, save: bool = False):
    """One-line Entry + Browse button pair."""
    entry = ttk.Entry(parent, textvariable=var, width=52)
    entry.grid(row=row, column=1, sticky="ew", padx=(0, 4))
    cmd = (lambda v=var, ft=filetypes: browse_save(v, ft)) if save \
        else (lambda v=var, ft=filetypes: browse_file(v, ft))
    ttk.Button(parent, text="Browse…", command=cmd, width=9).grid(
        row=row, column=2, sticky="w")


# ─────────────────────────────────────────────────────────────────────────────
#  Train tab
# ─────────────────────────────────────────────────────────────────────────────

class TrainTab(ttk.Frame):
    def __init__(self, parent, log: tk.Text):
        super().__init__(parent, padding=12)
        self._log = log
        self._build()

    def _build(self):
        self.columnconfigure(1, weight=1)

        # ── CSV input ────────────────────────────────────────────────────────
        _row(self, "Training CSV:", 0)
        self.csv_var = tk.StringVar()
        _entry_browse(self, self.csv_var, 0,
                      [("CSV files", "*.csv"), ("All files", "*.*")])

        # ── Model save path ───────────────────────────────────────────────────
        _row(self, "Save model to:", 1)
        self.model_var = tk.StringVar(value="spad_model.pkl")
        _entry_browse(self, self.model_var, 1,
                      [("Pickle files", "*.pkl"), ("All files", "*.*")],
                      save=True)

        # ── Numeric parameters ────────────────────────────────────────────────
        _row(self, "Test size (0–1):", 2)
        self.test_size_var = tk.DoubleVar(value=0.20)
        ttk.Spinbox(self, from_=0.05, to=0.50, increment=0.05,
                    textvariable=self.test_size_var,
                    width=8, format="%.2f").grid(
            row=2, column=1, sticky="w", padx=(0, 4))

        _row(self, "Random state:", 3)
        self.rng_var = tk.IntVar(value=42)
        ttk.Spinbox(self, from_=0, to=9999, increment=1,
                    textvariable=self.rng_var,
                    width=8).grid(row=3, column=1, sticky="w", padx=(0, 4))

        # ── Feature selector ──────────────────────────────────────────────────
        ttk.Separator(self, orient="horizontal").grid(
            row=4, column=0, columnspan=3, sticky="ew", pady=8)
        ttk.Label(self, text="Feature columns to include:",
                  font=("", 9, "bold")).grid(
            row=5, column=0, columnspan=3, sticky="w", padx=8)

        feat_frame = ttk.Frame(self)
        feat_frame.grid(row=6, column=0, columnspan=3, sticky="ew", padx=8)

        self.feat_vars: dict[str, tk.BooleanVar] = {}
        for i, feat in enumerate(ALL_FEATURES):
            var = tk.BooleanVar(value=True)
            self.feat_vars[feat] = var
            cb = ttk.Checkbutton(feat_frame, text=feat, variable=var)
            cb.grid(row=i // 5, column=i % 5, sticky="w", padx=6, pady=2)

        btn_row = ttk.Frame(feat_frame)
        btn_row.grid(row=(len(ALL_FEATURES) // 5) + 1,
                     column=0, columnspan=5, sticky="w", pady=(6, 0))
        ttk.Button(btn_row, text="Select all",
                   command=lambda: [v.set(True) for v in self.feat_vars.values()]
                   ).pack(side="left", padx=4)
        ttk.Button(btn_row, text="Clear all",
                   command=lambda: [v.set(False) for v in self.feat_vars.values()]
                   ).pack(side="left", padx=4)

        # ── Run button ────────────────────────────────────────────────────────
        ttk.Separator(self, orient="horizontal").grid(
            row=7, column=0, columnspan=3, sticky="ew", pady=8)
        self.run_btn = ttk.Button(self, text="▶  Train model",
                                  command=self._run, width=20)
        self.run_btn.grid(row=8, column=0, columnspan=3)

    # ── validation & execution ────────────────────────────────────────────────

    def _run(self):
        csv = self.csv_var.get().strip()
        model_path = self.model_var.get().strip()
        test_size = self.test_size_var.get()
        rng = self.rng_var.get()
        selected = [f for f, v in self.feat_vars.items() if v.get()]

        if not csv:
            messagebox.showwarning("Missing input", "Please select a training CSV file.")
            return
        if not selected:
            messagebox.showwarning("No features", "Select at least one feature column.")
            return
        if not (0 < test_size < 1):
            messagebox.showwarning("Bad value", "Test size must be between 0 and 1.")
            return

        self.run_btn.configure(state="disabled", text="Running…")
        threading.Thread(
            target=self._train_thread,
            args=(csv, model_path, test_size, rng, selected),
            daemon=True,
        ).start()

    def _train_thread(self, csv, model_path, test_size, rng, selected):
        try:
            import pandas as pd
            import numpy as np
            from sklearn.model_selection import train_test_split
            from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

            df = pd.read_csv(csv)
            df = sr.compute_indices(df)

            available = [f for f in selected if f in df.columns]
            missing_cols = [f for f in selected if f not in df.columns]
            if missing_cols:
                print(f"[WARN] Columns not in CSV (skipped): {missing_cols}")
            if not available:
                print("[ERROR] None of the selected features are present in the CSV.")
                return

            model = sr.SPADRegressor(feature_cols=available)
            X = df[available].values.astype(float)
            y = df["SPAD"].values.astype(float)

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=rng
            )
            model.fit(X_train, y_train)
            model.summary()

            y_pred = model.predict(X_test)
            print("── Test-set evaluation ──────────────────────────")
            print(f"  R²   : {r2_score(y_test, y_pred):.4f}")
            print(f"  RMSE : {np.sqrt(mean_squared_error(y_test, y_pred)):.4f}")
            print(f"  MAE  : {mean_absolute_error(y_test, y_pred):.4f}")
            print("─────────────────────────────────────────────────\n")

            model.save(model_path)
            self._log.after(0, lambda: messagebox.showinfo(
                "Done", f"Model trained and saved to:\n{model_path}"))
        except Exception as exc:
            print(f"[ERROR] {exc}")
            self._log.after(0, lambda e=exc: messagebox.showerror("Error", str(e)))
        finally:
            self.run_btn.after(0, lambda: self.run_btn.configure(
                state="normal", text="▶  Train model"))


# ─────────────────────────────────────────────────────────────────────────────
#  Predict tab
# ─────────────────────────────────────────────────────────────────────────────

class PredictTab(ttk.Frame):
    def __init__(self, parent, log: tk.Text):
        super().__init__(parent, padding=12)
        self._log = log
        self._build()

    def _build(self):
        self.columnconfigure(1, weight=1)
        tif_types = [("TIFF files", "*.tif *.tiff"), ("All files", "*.*")]
        pkl_types = [("Pickle files", "*.pkl"), ("All files", "*.*")]
        out_types = [("TIFF files", "*.tif"), ("All files", "*.*")]

        self.band_vars: dict[str, tk.StringVar] = {}

        rows = {
            "GREEN":   "GREEN band TIFF:",
            "NIR":     "NIR band TIFF:",
            "RED":     "RED band TIFF:",
            "RedEdge": "RedEdge band TIFF:",
        }
        for i, (key, label) in enumerate(rows.items()):
            _row(self, label, i)
            var = tk.StringVar()
            self.band_vars[key] = var
            _entry_browse(self, var, i, tif_types)

        # ── RGB (optional) ────────────────────────────────────────────────────
        sep_row = len(rows)
        ttk.Separator(self, orient="horizontal").grid(
            row=sep_row, column=0, columnspan=3, sticky="ew", pady=6)

        _row(self, "RGB TIFF (optional):", sep_row + 1)
        self.rgb_var = tk.StringVar()
        _entry_browse(self, self.rgb_var, sep_row + 1, tif_types)

        # ── Model path ────────────────────────────────────────────────────────
        _row(self, "Model (.pkl):", sep_row + 2)
        self.model_var = tk.StringVar(value="spad_model.pkl")
        _entry_browse(self, self.model_var, sep_row + 2, pkl_types)

        # ── Output TIFF ───────────────────────────────────────────────────────
        _row(self, "Output TIFF (optional):", sep_row + 3)
        self.output_var = tk.StringVar()
        _entry_browse(self, self.output_var, sep_row + 3, out_types, save=True)

        # ── Nodata value ──────────────────────────────────────────────────────
        _row(self, "NoData value:", sep_row + 4)
        self.nodata_var = tk.DoubleVar(value=-9999.0)
        ttk.Entry(self, textvariable=self.nodata_var, width=12).grid(
            row=sep_row + 4, column=1, sticky="w", padx=(0, 4))

        # ── Plot checkbox ─────────────────────────────────────────────────────
        self.plot_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(self, text="Show SPAD map after prediction",
                        variable=self.plot_var).grid(
            row=sep_row + 5, column=0, columnspan=3, sticky="w", padx=8, pady=4)

        # ── Run button ────────────────────────────────────────────────────────
        ttk.Separator(self, orient="horizontal").grid(
            row=sep_row + 6, column=0, columnspan=3, sticky="ew", pady=8)
        self.run_btn = ttk.Button(self, text="▶  Run prediction",
                                  command=self._run, width=22)
        self.run_btn.grid(row=sep_row + 7, column=0, columnspan=3)

    # ── validation & execution ────────────────────────────────────────────────

    def _run(self):
        single_band = {k: v.get().strip() for k, v in self.band_vars.items()}
        missing_bands = [k for k, v in single_band.items() if not v]
        if missing_bands:
            messagebox.showwarning(
                "Missing bands",
                f"Please set paths for: {', '.join(missing_bands)}"
            )
            return

        model_path = self.model_var.get().strip()
        if not model_path:
            messagebox.showwarning("Missing model", "Please select a model .pkl file.")
            return

        rgb = self.rgb_var.get().strip() or None
        output = self.output_var.get().strip() or None
        nodata = self.nodata_var.get()
        show_plot = self.plot_var.get()

        self.run_btn.configure(state="disabled", text="Running…")
        threading.Thread(
            target=self._predict_thread,
            args=(single_band, rgb, model_path, output, nodata, show_plot),
            daemon=True,
        ).start()

    def _predict_thread(self, single_band, rgb, model_path, output, nodata, show_plot):
        try:
            spad_map = sr.predict_from_band_tiffs(
                single_band_tiffs=single_band,
                rgb_tiff=rgb,
                model_path=model_path,
                output_tiff=output,
                nodata_val=nodata,
            )
            if show_plot:
                sr.plot_spad_map(spad_map, nodata_val=nodata)
            self._log.after(0, lambda: messagebox.showinfo(
                "Done", "Prediction finished. Check the log for statistics."))
        except Exception as exc:
            print(f"[ERROR] {exc}")
            self._log.after(0, lambda e=exc: messagebox.showerror("Error", str(e)))
        finally:
            self.run_btn.after(0, lambda: self.run_btn.configure(
                state="normal", text="▶  Run prediction"))


# ─────────────────────────────────────────────────────────────────────────────
#  Main application window
# ─────────────────────────────────────────────────────────────────────────────

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("SPAD Regression GUI")
        self.resizable(True, True)
        self.minsize(680, 560)

        # ── Style ─────────────────────────────────────────────────────────────
        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure("TNotebook.Tab", padding=[10, 4], font=("", 10, "bold"))

        # ── Log panel (shared between tabs) ──────────────────────────────────
        pane = ttk.PanedWindow(self, orient="vertical")
        pane.pack(fill="both", expand=True, padx=8, pady=8)

        # ── Notebook ─────────────────────────────────────────────────────────
        nb = ttk.Notebook(pane)
        pane.add(nb, weight=3)

        log_frame = ttk.LabelFrame(pane, text="Output log")
        pane.add(log_frame, weight=1)

        log_text = scrolledtext.ScrolledText(
            log_frame, state="disabled", wrap="word",
            font=("Courier", 9), height=10,
            background="#1e1e1e", foreground="#d4d4d4",
            insertbackground="white",
        )
        log_text.pack(fill="both", expand=True, padx=4, pady=4)

        # ── Clear log button ─────────────────────────────────────────────────
        def _clear_log():
            log_text.configure(state="normal")
            log_text.delete("1.0", tk.END)
            log_text.configure(state="disabled")

        ttk.Button(log_frame, text="Clear log", command=_clear_log).pack(
            side="right", padx=6, pady=(0, 4))

        # ── Redirect stdout / stderr ──────────────────────────────────────────
        redirector = StdoutRedirector(log_text)
        sys.stdout = redirector
        sys.stderr = redirector

        # ── Tabs ──────────────────────────────────────────────────────────────
        nb.add(TrainTab(nb, log_text), text="  Train  ")
        nb.add(PredictTab(nb, log_text), text="  Predict  ")

        # ── Status bar ────────────────────────────────────────────────────────
        status = ttk.Label(self, text="Ready", anchor="w",
                           relief="sunken", padding=(4, 2))
        status.pack(side="bottom", fill="x")

    def destroy(self):
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__
        super().destroy()


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app = App()
    app.mainloop()
