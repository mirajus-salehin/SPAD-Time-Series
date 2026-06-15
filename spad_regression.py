"""
SPAD Value Prediction using Multiple Linear Regression (scipy)
--------------------------------------------------------------
Train on CSV data, save model, and run inference on UAV multispectral images.

Input layout expected at inference:
  - Single-band TIFFs : GREEN, NIR, RED, RedEdge  (one file each)
  - RGB TIFF          : one 3-band file → split into Red_RGB, Green_RGB, Blue_RGB

Dependencies:
    pip install numpy pandas scipy scikit-learn rasterio joblib matplotlib
"""

import numpy as np
import pandas as pd
import joblib
import os
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


# ─────────────────────────────────────────────
#  Feature engineering helpers
# ─────────────────────────────────────────────

def compute_indices(df: pd.DataFrame) -> pd.DataFrame:
    """Compute common vegetation indices from raw band values."""
    eps = 1e-8
    df = df.copy()

    df["NDVI"]   = (df["NIR"] - df["RED"])     / (df["NIR"] + df["RED"]     + eps)
    df["CRE"]    = (df["NIR"] / (df["RedEdge"] + eps)) - 1
    df["GNDVI"]  = (df["NIR"] - df["GREEN"])   / (df["NIR"] + df["GREEN"]   + eps)
    df["RENDVI"] = (df["NIR"] - df["RedEdge"]) / (df["NIR"] + df["RedEdge"] + eps)
    df["SR"]     = df["NIR"] / (df["RED"] + eps)
    L = 0.5
    df["SAVI"]   = ((df["NIR"] - df["RED"]) / (df["NIR"] + df["RED"] + L + eps)) * (1 + L)
    df["DVI"]    = df["NIR"] - df["RED"]

    return df


# ─────────────────────────────────────────────
#  Model class
# ─────────────────────────────────────────────

class SPADRegressor:
    """
    Multiple Linear Regression for SPAD prediction.
    Uses np.linalg.lstsq (OLS) with full scipy-based statistics.
    """

    FEATURE_COLS = [
        "GREEN", "NIR", "RED", "RedEdge",
        "Red_RGB", "Green_RGB", "Blue_RGB",
        "NDVI", "CRE", "GNDVI", "RENDVI", "SR", "SAVI", "DVI",
    ]

    def __init__(self, feature_cols: list | None = None):
        self.feature_cols: list = feature_cols or self.FEATURE_COLS
        self.scaler = StandardScaler()
        self.coef_: np.ndarray | None = None
        self.stats_: dict = {}

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SPADRegressor":
        n, p = X.shape
        X_scaled = self.scaler.fit_transform(X)
        X_aug = np.column_stack([np.ones(n), X_scaled])

        self.coef_, _, _, _ = np.linalg.lstsq(X_aug, y, rcond=None)

        y_hat = X_aug @ self.coef_
        e     = y - y_hat
        SSE   = float(np.dot(e, e))
        SST   = float(np.sum((y - y.mean()) ** 2))
        SSR   = SST - SSE

        df_reg, df_res = p, n - p - 1
        MSE = SSE / max(df_res, 1)
        MSR = SSR / max(df_reg, 1)

        R2     = 1 - SSE / SST if SST > 0 else 0.0
        R2_adj = 1 - (1 - R2) * (n - 1) / max(df_res, 1)
        F_stat = MSR / MSE if MSE > 0 else np.nan
        F_pval = float(1 - stats.f.cdf(F_stat, df_reg, df_res)) if not np.isnan(F_stat) else np.nan

        XtX_inv = np.linalg.pinv(X_aug.T @ X_aug)
        se      = np.sqrt(np.maximum(MSE * np.diag(XtX_inv), 0))
        t_stats = self.coef_ / (se + 1e-15)
        p_vals  = 2 * (1 - stats.t.cdf(np.abs(t_stats), df=df_res))

        self.stats_ = {
            "R2": R2, "R2_adj": R2_adj,
            "F_stat": F_stat, "F_pval": F_pval,
            "MSE": MSE, "RMSE": np.sqrt(MSE),
            "n": n, "p": p,
            "coefficients": self.coef_,
            "std_errors": se, "t_stats": t_stats, "p_values": p_vals,
        }
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        assert self.coef_ is not None, "Model not fitted yet."
        X_scaled = self.scaler.transform(X)
        X_aug    = np.column_stack([np.ones(len(X_scaled)), X_scaled])
        return X_aug @ self.coef_

    def save(self, path: str = "spad_model.pkl") -> None:
        joblib.dump({"coef": self.coef_, "scaler": self.scaler,
                     "feature_cols": self.feature_cols, "stats": self.stats_}, path)
        print(f"Model saved → {path}")

    @classmethod
    def load(cls, path: str = "spad_model.pkl") -> "SPADRegressor":
        data  = joblib.load(path)
        model = cls(feature_cols=data["feature_cols"])
        model.coef_   = data["coef"]
        model.scaler  = data["scaler"]
        model.stats_  = data["stats"]
        print(f"Model loaded ← {path}")
        return model

    def summary(self) -> None:
        s     = self.stats_
        names = ["intercept"] + self.feature_cols
        print("\n" + "=" * 62)
        print("  SPAD Multiple Linear Regression — OLS Summary")
        print("=" * 62)
        print(f"  Observations : {s['n']}")
        print(f"  Predictors   : {s['p']}")
        print(f"  R²           : {s['R2']:.4f}")
        print(f"  Adj. R²      : {s['R2_adj']:.4f}")
        print(f"  F-statistic  : {s['F_stat']:.4f}  (p = {s['F_pval']:.4e})")
        print(f"  RMSE         : {s['RMSE']:.4f}")
        print("-" * 62)
        print(f"  {'Feature':<18} {'Coef':>10} {'SE':>10} {'t':>10} {'p-val':>12}")
        print("-" * 62)
        for name, c, se, t, p in zip(
            names, s["coefficients"], s["std_errors"], s["t_stats"], s["p_values"]
        ):
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
            print(f"  {name:<18} {c:>10.4f} {se:>10.4f} {t:>10.4f} {p:>12.4e} {sig}")
        print("=" * 62 + "\n")


# ─────────────────────────────────────────────
#  Training pipeline
# ─────────────────────────────────────────────

def train_from_csv(
    csv_path: str,
    model_save_path: str = "spad_model.pkl",
    test_size: float = 0.2,
    random_state: int = 42,
) -> SPADRegressor:
    """
    Load CSV, engineer features, train model, evaluate, and save.

    CSV must contain columns:
        Point_ID, GREEN, NIR, RED, RedEdge,
        Red_RGB, Green_RGB, Blue_RGB, SPAD
    """
    df = pd.read_csv(csv_path)
    df = compute_indices(df)

    model             = SPADRegressor()
    available_features = [c for c in model.feature_cols if c in df.columns]
    model.feature_cols = available_features

    X = df[available_features].values.astype(float)
    y = df["SPAD"].values.astype(float)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    model.fit(X_train, y_train)
    model.summary()

    y_pred = model.predict(X_test)
    print("── Test-set evaluation ──────────────────────────")
    print(f"  R²   : {r2_score(y_test, y_pred):.4f}")
    print(f"  RMSE : {np.sqrt(mean_squared_error(y_test, y_pred)):.4f}")
    print(f"  MAE  : {mean_absolute_error(y_test, y_pred):.4f}")
    print("─────────────────────────────────────────────────\n")

    model.save(model_save_path)
    return model


# ─────────────────────────────────────────────
#  TIFF loading  (single-band + RGB split)
# ─────────────────────────────────────────────

def load_band_tiffs(
    single_band_tiffs: dict,
    rgb_tiff: str | None = None,
    rgb_band_names: tuple = ("Red_RGB", "Green_RGB", "Blue_RGB"),
) -> tuple:
    """
    Load spectral bands from single-band TIFFs and optionally one 3-band RGB TIFF.

    Parameters
    ----------
    single_band_tiffs : {band_name: file_path}
        Each file must contain exactly 1 band.
        Example: {"GREEN": "green.tif", "NIR": "nir.tif",
                  "RED": "red.tif", "RedEdge": "rededge.tif"}

    rgb_tiff : Path to the 3-band RGB GeoTIFF (band1=R, band2=G, band3=B).
               The three channels are split and stored under rgb_band_names.

    rgb_band_names : Names assigned to the R, G, B channels split from rgb_tiff.
                     Default: ("Red_RGB", "Green_RGB", "Blue_RGB")

    Returns
    -------
    arrays   : {band_name: 2-D float ndarray (H x W)}
    ref_meta : rasterio profile from the first file opened (used for output GeoTIFF)
    """
    try:
        import rasterio
    except ImportError:
        raise ImportError("Install rasterio:  pip install rasterio")

    arrays: dict = {}
    ref_meta  = None
    ref_shape = None

    def _store(name: str, data: np.ndarray, src_desc: str) -> None:
        nonlocal ref_shape
        if ref_shape is None:
            ref_shape = data.shape
        elif data.shape != ref_shape:
            raise ValueError(
                f"Shape mismatch for '{name}' ({src_desc}): "
                f"got {data.shape}, expected {ref_shape}. "
                "All images must have the same spatial dimensions."
            )
        arrays[name] = data
        print(f"  {name:<16} ← {src_desc}  {data.shape}")

    # ── Single-band files ─────────────────────────────────────────────────
    for band_name, path in single_band_tiffs.items():
        if not os.path.isfile(path):
            raise FileNotFoundError(f"File not found: '{path}'  (band '{band_name}')")
        with rasterio.open(path) as src:
            if src.count != 1:
                raise ValueError(
                    f"'{path}' has {src.count} bands — expected a single-band file "
                    f"for '{band_name}'. Pass multi-band RGB via rgb_tiff instead."
                )
            data = src.read(1).astype(float)
            if ref_meta is None:
                ref_meta = src.meta.copy()
        _store(band_name, data, path)

    # ── RGB TIFF → split into R / G / B ──────────────────────────────────
    if rgb_tiff is not None:
        if not os.path.isfile(rgb_tiff):
            raise FileNotFoundError(f"RGB TIFF not found: '{rgb_tiff}'")
        with rasterio.open(rgb_tiff) as src:
            if src.count < 3:
                raise ValueError(
                    f"RGB TIFF '{rgb_tiff}' has {src.count} band(s); expected 3 (R, G, B)."
                )
            r_data = src.read(1).astype(float)   # band 1 → Red
            g_data = src.read(2).astype(float)   # band 2 → Green
            b_data = src.read(3).astype(float)   # band 3 → Blue
            if ref_meta is None:
                ref_meta = src.meta.copy()

        r_name, g_name, b_name = rgb_band_names
        _store(r_name, r_data, f"{rgb_tiff}  [band 1 → R]")
        _store(g_name, g_data, f"{rgb_tiff}  [band 2 → G]")
        _store(b_name, b_data, f"{rgb_tiff}  [band 3 → B]")

    if not arrays:
        raise ValueError("No band data was loaded. Provide at least single_band_tiffs.")

    return arrays, ref_meta


# ─────────────────────────────────────────────
#  Inference pipeline
# ─────────────────────────────────────────────

def predict_from_band_tiffs(
    single_band_tiffs: dict,
    rgb_tiff: str | None = None,
    model_path: str = "spad_model.pkl",
    output_tiff: str | None = None,
    nodata_val: float = -9999.0,
) -> np.ndarray:
    """
    Predict SPAD values across a UAV flight area from per-band images.

    Parameters
    ----------
    single_band_tiffs : dict  {band_name: file_path}
        One single-band GeoTIFF per multispectral channel.
        Required keys: "GREEN", "NIR", "RED", "RedEdge"

    rgb_tiff : str or None
        Path to the 3-band RGB GeoTIFF (band order: R=1, G=2, B=3).
        The three channels are automatically split into
        "Red_RGB", "Green_RGB", "Blue_RGB".
        Pass None if the model was trained without RGB bands.

    model_path  : Path to .pkl model saved by train_from_csv().
    output_tiff : If given, write the SPAD map as a georeferenced GeoTIFF.
    nodata_val  : Fill value for pixels outside the valid image area.

    Returns
    -------
    spad_map : 2-D numpy array (H x W) of predicted SPAD values.

    Example
    -------
    >>> spad_map = predict_from_band_tiffs(
    ...     single_band_tiffs={
    ...         "GREEN"   : "green.tif",
    ...         "NIR"     : "nir.tif",
    ...         "RED"     : "red.tif",
    ...         "RedEdge" : "rededge.tif",
    ...     },
    ...     rgb_tiff   = "rgb.tif",        # 3-band: R, G, B
    ...     model_path = "spad_model.pkl",
    ...     output_tiff= "spad_map.tif",
    ... )
    """
    try:
        import rasterio
    except ImportError:
        raise ImportError("Install rasterio:  pip install rasterio")

    print("\nLoading images …")
    arrays, ref_meta = load_band_tiffs(single_band_tiffs, rgb_tiff=rgb_tiff)

    H, W = next(iter(arrays.values())).shape
    print(f"\nImage size : {H} × {W} = {H * W:,} pixels")

    # Build pixel-level DataFrame
    pixel_df = pd.DataFrame({name: arr.ravel() for name, arr in arrays.items()})
    pixel_df = compute_indices(pixel_df)

    # Load saved model
    model = SPADRegressor.load(model_path)

    # Zero-fill any feature the model expects but wasn't supplied
    missing = set(model.feature_cols) - set(pixel_df.columns)
    if missing:
        print(f"[WARN] Features not available from supplied images (zero-filled): {missing}")
        for col in missing:
            pixel_df[col] = 0.0

    X_pixels = pixel_df[model.feature_cols].values

    # Validity mask: all values finite AND at least one raw band non-zero
    valid_mask  = np.all(np.isfinite(X_pixels), axis=1)
    band_stack  = np.stack([arr.ravel() for arr in arrays.values()], axis=1)
    valid_mask &= np.any(band_stack != 0, axis=1)

    spad_flat = np.full(H * W, nodata_val, dtype=float)
    if valid_mask.any():
        spad_flat[valid_mask] = model.predict(X_pixels[valid_mask])

    spad_map = spad_flat.reshape(H, W)

    # Write georeferenced output GeoTIFF
    if output_tiff:
        out_meta = ref_meta.copy()
        out_meta.update({"count": 1, "dtype": "float32", "nodata": nodata_val})
        with rasterio.open(output_tiff, "w", **out_meta) as dst:
            dst.write(spad_map.astype("float32"), 1)
        print(f"SPAD map saved → {output_tiff}")

    valid_vals = spad_map[spad_map != nodata_val]
    if valid_vals.size:
        print(f"SPAD stats  — min: {valid_vals.min():.2f}  "
              f"max: {valid_vals.max():.2f}  "
              f"mean: {valid_vals.mean():.2f}  "
              f"std: {valid_vals.std():.2f}")

    return spad_map


# ─────────────────────────────────────────────
#  Visualisation
# ─────────────────────────────────────────────

def plot_spad_map(
    spad_map: np.ndarray,
    title: str = "Predicted SPAD Map",
    nodata_val: float = -9999.0,
) -> None:
    """Display the SPAD prediction map with a colour ramp."""
    import matplotlib.pyplot as plt

    display = spad_map.copy().astype(float)
    display[display == nodata_val] = np.nan

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(display, cmap="RdYlGn", interpolation="nearest")
    plt.colorbar(im, ax=ax, label="Predicted SPAD")
    ax.set_title(title)
    ax.axis("off")
    plt.tight_layout()
    plt.show()


# ─────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="SPAD MLR — train & predict")
    sub    = parser.add_subparsers(dest="cmd")

    # ── train ──────────────────────────────────────────────────────────────
    tr = sub.add_parser("train", help="Train model from a CSV file")
    tr.add_argument("csv",          help="Path to training CSV")
    tr.add_argument("--model",      default="spad_model.pkl", help="Output model path")
    tr.add_argument("--test-size",  type=float, default=0.2)

    # ── predict ────────────────────────────────────────────────────────────
    pr = sub.add_parser(
        "predict",
        help="Predict SPAD from per-band images",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=(
            "Single-band TIFFs (one file per multispectral channel)\n"
            "+ one 3-band RGB TIFF that is split automatically.\n\n"
            "Example:\n"
            "  python spad_regression.py predict \\\n"
            "      --GREEN green.tif --NIR nir.tif \\\n"
            "      --RED red.tif --RedEdge rededge.tif \\\n"
            "      --rgb rgb.tif \\\n"
            "      --model spad_model.pkl --output spad_map.tif --plot"
        ),
    )
    # Required multispectral single-band inputs
    pr.add_argument("--GREEN",    required=True, metavar="FILE", help="Green band TIFF (single-band)")
    pr.add_argument("--NIR",      required=True, metavar="FILE", help="NIR band TIFF (single-band)")
    pr.add_argument("--RED",      required=True, metavar="FILE", help="Red band TIFF (single-band)")
    pr.add_argument("--RedEdge",  required=True, metavar="FILE", help="RedEdge band TIFF (single-band)")
    # Optional 3-band RGB image
    pr.add_argument("--rgb",      default=None,  metavar="FILE",
                    help="3-band RGB GeoTIFF (band1=R, band2=G, band3=B). "
                         "Channels are split into Red_RGB / Green_RGB / Blue_RGB automatically.")
    pr.add_argument("--model",    default="spad_model.pkl", help="Saved model .pkl path")
    pr.add_argument("--output",   default=None,  metavar="FILE", help="Output GeoTIFF path")
    pr.add_argument("--plot",     action="store_true",            help="Display SPAD map with matplotlib")

    args = parser.parse_args()

    if args.cmd == "train":
        train_from_csv(args.csv, model_save_path=args.model, test_size=args.test_size)

    elif args.cmd == "predict":
        spad_map = predict_from_band_tiffs(
            single_band_tiffs={
                "GREEN":   args.GREEN,
                "NIR":     args.NIR,
                "RED":     args.RED,
                "RedEdge": args.RedEdge,
            },
            rgb_tiff    = args.rgb,
            model_path  = args.model,
            output_tiff = args.output,
        )
        if args.plot:
            plot_spad_map(spad_map)

    else:
        parser.print_help()